"""Lazy, structured Qwen3-VL verification for difficult crops.

Qwen is deliberately not imported at application startup.  The verifier can
also be supplied with a small ``generator`` callback, which keeps tests and
offline deployments deterministic without downloading a large model.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from PIL import Image

from yolo_world_annotator.inference.crop_utils import crop_image
from yolo_world_annotator.utils.config import as_bool

LOGGER = logging.getLogger(__name__)
DEFAULT_VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
MATCH = "MATCH"
NOT_MATCH = "NOT_MATCH"
UNCERTAIN = "UNCERTAIN"
VALID_RESULTS = {MATCH, NOT_MATCH, UNCERTAIN}
VALID_FEATURE_VALUES = {"TRUE", "FALSE", UNCERTAIN}


@dataclass(slots=True)
class VLMResult:
    target_class: str | None = None
    features: dict[str, str] = field(default_factory=dict)
    final_result: str = UNCERTAIN
    self_reported_confidence: float | None = None
    raw_response: str | None = None
    parsed: bool = False
    parse_error: str | None = None
    model: str | None = None

    @property
    def confidence(self) -> float | None:
        return self.self_reported_confidence

    @property
    def result(self) -> str:
        return self.final_result

    @property
    def is_match(self) -> bool:
        return self.parsed and self.final_result == MATCH

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_class": self.target_class,
            "features": dict(self.features),
            "final_result": self.final_result,
            "self_reported_confidence": self.self_reported_confidence,
            "raw_response": self.raw_response,
            "parsed": self.parsed,
            "parse_error": self.parse_error,
            "model": self.model,
        }


def _profile_values(profile: Any) -> tuple[str, str, list[dict[str, Any]], list[str]]:
    if profile is None:
        return "", "", [], []
    if isinstance(profile, Mapping):
        name = str(profile.get("class_name", profile.get("name", "")))
        description = str(profile.get("vlm_description", name))
        raw_features = profile.get("features", [])
        required = profile.get("required_features", [])
    else:
        name = str(getattr(profile, "class_name", ""))
        description = str(getattr(profile, "vlm_description", name))
        raw_features = getattr(profile, "features", [])
        required = getattr(profile, "required_features", [])
    if isinstance(raw_features, Mapping):
        raw_features = [
            {"name": key, "required": value} for key, value in raw_features.items()
        ]
    features: list[dict[str, Any]] = []
    for value in raw_features if isinstance(raw_features, list) else []:
        if isinstance(value, str):
            features.append({"name": value, "required": value in required})
        elif isinstance(value, Mapping) and str(value.get("name", value.get("feature", ""))).strip():
            features.append(
                {
                    "name": str(value.get("name", value.get("feature"))).strip(),
                    "required": as_bool(value.get("required", False)),
                }
            )
    required_names = [str(value).strip() for value in required if str(value).strip()]
    required_names.extend(item["name"] for item in features if item.get("required"))
    return name, description, features, list(dict.fromkeys(required_names))


def build_vlm_prompt(
    target_class: str,
    *,
    profile: Any = None,
    candidate_classes: Sequence[str] | None = None,
) -> str:
    """Build a constrained JSON-only prompt; no open-ended visual Q&A."""

    name, description, features, required = _profile_values(profile)
    target = str(target_class or name).strip()
    candidates = [str(value).strip() for value in (candidate_classes or [target]) if str(value).strip()]
    feature_lines = [
        f'- "{item["name"]}": TRUE, FALSE, or UNCERTAIN'
        + (" (required)" if item.get("required") or item["name"] in required else "")
        for item in features
    ]
    features_block = "\n".join(feature_lines) or "- No extra features; return an empty features object."
    candidates_block = ", ".join(json.dumps(value, ensure_ascii=False) for value in candidates)
    return (
        "You are a constrained visual verifier. Inspect only the supplied crop.\n"
        f"Target class: {json.dumps(target, ensure_ascii=False)}\n"
        f"Allowed candidate classes: [{candidates_block}]\n"
        f"Class description: {description}\n"
        "For each listed feature output exactly TRUE, FALSE, or UNCERTAIN:\n"
        f"{features_block}\n"
        "Return ONLY valid JSON with keys target_class, features, final_result, "
        "self_reported_confidence. final_result must be MATCH, NOT_MATCH, or UNCERTAIN."
    )


def _extract_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    text = str(raw or "").strip()
    if not text:
        raise ValueError("VLM 返回为空")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("VLM JSON 顶层必须是对象")
    return value


def parse_vlm_response(
    raw: Any,
    *,
    target_class: str | None = None,
    profile: Any = None,
    candidate_classes: Sequence[str] | None = None,
    model: str | None = None,
) -> VLMResult:
    """Parse and validate a constrained VLM response.

    Missing/invalid JSON never becomes an automatic match.  Required feature
    failures downgrade an otherwise optimistic answer to ``UNCERTAIN`` so the
    decision engine can route it to human review.
    """

    raw_text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False) if isinstance(raw, Mapping) else str(raw or "")
    try:
        data = _extract_json(raw)
    except Exception as exc:
        return VLMResult(
            target_class=target_class,
            final_result=UNCERTAIN,
            raw_response=raw_text,
            parsed=False,
            parse_error=str(exc),
            model=model,
        )
    result = str(data.get("final_result", "")).upper().strip()
    if result not in VALID_RESULTS:
        return VLMResult(
            target_class=str(data.get("target_class", target_class or "")) or target_class,
            raw_response=raw_text,
            parsed=False,
            parse_error="final_result 必须是 MATCH / NOT_MATCH / UNCERTAIN",
            model=model,
        )
    returned_target = str(data.get("target_class", target_class or "")) or target_class
    allowed_targets = {str(value).strip() for value in (candidate_classes or []) if str(value).strip()}
    if allowed_targets and returned_target not in allowed_targets:
        return VLMResult(
            target_class=returned_target,
            raw_response=raw_text,
            parsed=False,
            parse_error="target_class 不在限定候选类别中",
            model=model,
        )
    feature_values: dict[str, str] = {}
    invalid_features: list[str] = []
    raw_features = data.get("features", {})
    if isinstance(raw_features, Mapping):
        for key, value in raw_features.items():
            normalized = str(value).upper().strip()
            if normalized not in VALID_FEATURE_VALUES:
                invalid_features.append(str(key))
            feature_values[str(key)] = normalized if normalized in VALID_FEATURE_VALUES else UNCERTAIN
    confidence = data.get("self_reported_confidence")
    try:
        confidence_value = None if confidence is None else min(1.0, max(0.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_value = None
    _, _, _, required = _profile_values(profile)
    failed_required = [name for name in required if feature_values.get(name) == "FALSE"]
    missing_required = [name for name in required if name not in feature_values]
    parse_error = f"invalid feature values: {', '.join(invalid_features)}" if invalid_features else None
    if invalid_features and result == MATCH:
        result = UNCERTAIN
    if failed_required:
        result = NOT_MATCH if result == MATCH else result
        parse_error = f"required features false: {', '.join(failed_required)}"
    elif missing_required and result == MATCH:
        result = UNCERTAIN
        parse_error = f"required features missing: {', '.join(missing_required)}"
    return VLMResult(
        target_class=returned_target,
        features=feature_values,
        final_result=result,
        self_reported_confidence=confidence_value,
        raw_response=raw_text,
        parsed=True,
        parse_error=parse_error,
        model=model,
    )


@dataclass(slots=True)
class VLMTriggerPolicy:
    yolo_low_threshold: float = 0.45
    siglip_low_threshold: float = 0.55
    margin_threshold: float = 0.10
    review_statuses: tuple[str, ...] = ("REVIEW", "REJECT")
    force_special_classes: bool = True

    def should_trigger(
        self,
        *,
        yolo_score: float | None = None,
        siglip_score: float | None = None,
        siglip_margin: float | None = None,
        agreement: bool | None = None,
        decision_status: str | None = None,
        always_vlm_verify: bool = False,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if self.force_special_classes and always_vlm_verify:
            reasons.append("SPECIAL_CLASS")
        if agreement is False:
            reasons.append("MODEL_CONFLICT")
        if yolo_score is not None and float(yolo_score) < self.yolo_low_threshold:
            reasons.append("LOW_YOLO_CONFIDENCE")
        if siglip_score is not None and float(siglip_score) < self.siglip_low_threshold:
            reasons.append("LOW_SIGLIP_SCORE")
        if siglip_margin is not None and float(siglip_margin) < self.margin_threshold:
            reasons.append("LOW_SIGLIP_MARGIN")
        if str(decision_status or "").upper() in self.review_statuses:
            reasons.append("REVIEW_STATUS")
        return bool(reasons), list(dict.fromkeys(reasons))

    def triggered(self, **kwargs) -> bool:
        return self.should_trigger(**kwargs)[0]


def should_trigger_vlm(**kwargs) -> tuple[bool, list[str]]:
    policy = kwargs.pop("policy", None) or VLMTriggerPolicy()
    return policy.should_trigger(**kwargs)


parse_vlm_json = parse_vlm_response


class VLMVerifier:
    """Lazy Qwen3-VL wrapper with graceful loading and parser fallbacks."""

    def __init__(
        self,
        model_name: str = DEFAULT_VLM_MODEL,
        *,
        lazy_load: bool = True,
        max_new_tokens: int = 128,
        low_memory: bool = True,
        generator: Callable[..., Any] | None = None,
        device: str = "auto",
    ) -> None:
        self.model_name = str(model_name)
        self.lazy_load = bool(lazy_load)
        self.max_new_tokens = max(16, int(max_new_tokens))
        self.low_memory = bool(low_memory)
        self.generator = generator
        self.device = device
        self.processor: Any | None = None
        self.model: Any | None = None
        self.load_error: str | None = None
        self.device_description = "lazy / not loaded"

    @property
    def loaded(self) -> bool:
        return self.generator is not None or (self.model is not None and self.processor is not None)

    def load_model(self) -> None:
        if self.loaded:
            return
        try:
            from transformers import AutoProcessor
            # Transformers versions expose different Qwen generation classes.
            # Prefer the native Qwen3 implementation and retain Qwen2/Auto
            # fallbacks so a missing optional class degrades gracefully.
            try:
                from transformers import Qwen3VLForConditionalGeneration as VLMModel
            except ImportError:
                try:
                    from transformers import Qwen2VLForConditionalGeneration as VLMModel
                except ImportError:
                    from transformers import AutoModelForVision2Seq as VLMModel

            kwargs: dict[str, Any] = {}
            if self.low_memory and torch.cuda.is_available():
                kwargs["device_map"] = "auto"
                kwargs["torch_dtype"] = torch.float16
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = VLMModel.from_pretrained(self.model_name, **kwargs).eval()
            if not kwargs.get("device_map"):
                target = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                self.model = self.model.to(target)
            self.device_description = "CUDA lazy VLM" if torch.cuda.is_available() else "CPU VLM"
            self.load_error = None
        except ModuleNotFoundError as exc:
            self.load_error = "启用 Qwen3-VL 需要安装支持 Qwen 的 transformers 版本"
            raise RuntimeError(self.load_error) from exc
        except Exception as exc:
            LOGGER.exception("Qwen3-VL 模型加载失败：%s", self.model_name)
            self.load_error = f"Qwen3-VL 模型加载失败：{exc}"
            self.model = None
            self.processor = None
            raise RuntimeError(self.load_error) from exc

    def try_load(self) -> bool:
        try:
            self.load_model()
            return True
        except Exception:
            return False

    def release(self) -> None:
        self.model = None
        self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _generate(self, image: Image.Image, prompt: str) -> Any:
        if self.generator is not None:
            try:
                return self.generator(image=image, prompt=prompt)
            except TypeError:
                return self.generator(image, prompt)
        self.load_model()
        assert self.model is not None and self.processor is not None
        # Qwen's processor accepts a text prompt plus an image.  Keep this
        # path intentionally conservative across Transformers minor versions.
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        target = next(self.model.parameters()).device
        inputs = {key: value.to(target) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        if hasattr(self.processor, "batch_decode"):
            return self.processor.batch_decode(output, skip_special_tokens=True)[0]
        return str(output)

    def verify(
        self,
        image: Image.Image | str | Path,
        *,
        target_class: str,
        profile: Any = None,
        candidate_classes: Sequence[str] | None = None,
        padding_ratio: float = 0.10,
        box: object | None = None,
    ) -> VLMResult:
        if isinstance(image, (str, Path)):
            with Image.open(image) as loaded:
                pil_image = loaded.convert("RGB")
        else:
            pil_image = image.convert("RGB")
        if box is not None:
            pil_image = crop_image(pil_image, box, padding_ratio)
        prompt = build_vlm_prompt(target_class, profile=profile, candidate_classes=candidate_classes)
        try:
            raw = self._generate(pil_image, prompt)
        except Exception as exc:
            return VLMResult(
                target_class=target_class,
                final_result=UNCERTAIN,
                parsed=False,
                parse_error=str(exc),
                model=self.model_name,
            )
        return parse_vlm_response(
            raw,
            target_class=target_class,
            profile=profile,
            candidate_classes=candidate_classes,
            model=self.model_name,
        )

    def verify_box(self, image, box, *, target_class: str, profile: Any = None, **kwargs) -> VLMResult:
        return self.verify(image, target_class=target_class, profile=profile, box=box, **kwargs)

    def verify_batch(self, image, boxes, classes, *, profiles: Mapping[str, Any] | None = None, **kwargs) -> list[VLMResult]:
        results: list[VLMResult] = []
        for box in boxes:
            name = str(getattr(box, "class_name", classes[0] if classes else ""))
            profile = profiles.get(name) if profiles else None
            results.append(self.verify(image, target_class=name, profile=profile, box=box, **kwargs))
        return results

    verify_boxes = verify_batch


Qwen3VLVerifier = VLMVerifier
VLMVerificationResult = VLMResult


__all__ = [
    "DEFAULT_VLM_MODEL",
    "MATCH",
    "NOT_MATCH",
    "UNCERTAIN",
    "VLMResult",
    "VLMVerificationResult",
    "VLMTriggerPolicy",
    "VLMVerifier",
    "Qwen3VLVerifier",
    "build_vlm_prompt",
    "parse_vlm_response",
    "parse_vlm_json",
    "should_trigger_vlm",
]
