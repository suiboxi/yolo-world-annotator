"""Crop-level SigLIP2 verification for YOLO-World detections.

SigLIP2 is intentionally kept independent from the YOLO detector.  The
worker owns one verifier instance for the lifetime of a task, text embeddings
are cached by class/prompt configuration, and verification accepts a list of
cropped boxes so that GPU work is batched without ever running detection on a
full image.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from core.annotation import BoundingBox
from core.verification import SigLIPPrediction


LOGGER = logging.getLogger(__name__)


DEFAULT_SIGLIP_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_PROMPT_TEMPLATE = "a photo of a {}"
DEFAULT_PROMPT_ENSEMBLE = (
    "a photo of a {}",
    "an image of a {}",
    "a close-up photo of a {}",
)


class SigLIPVerifier:
    """Persistent crop-level SigLIP2 image/text verifier."""

    def __init__(
        self,
        model_name: str = DEFAULT_SIGLIP_MODEL,
        *,
        batch_size: int = 4,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        prompt_ensemble: bool = False,
        precision: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.prompt_ensemble = bool(prompt_ensemble)
        self.precision = str(precision or "auto").casefold()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_description = (
            f"{torch.cuda.get_device_name(0)} / cuda:0"
            if torch.cuda.is_available()
            else "CPU"
        )
        self.dtype = self._select_dtype(self.precision)
        self.processor: Any | None = None
        self.model: Any | None = None
        self._text_cache_key: tuple[Any, ...] | None = None
        self._text_embeddings: torch.Tensor | None = None
        self.classes: list[str] = []

    def _select_dtype(self, precision: str) -> torch.dtype:
        if self.device.type != "cuda":
            return torch.float32
        if precision == "bf16":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            LOGGER.warning("当前 GPU 不支持 BF16，SigLIP2 回退 FP16")
        return torch.float16

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def load_model(self) -> None:
        """Load the processor/model once, with a clear optional dependency error."""

        if self.loaded:
            return
        try:
            from transformers import AutoModel, AutoProcessor

            # Auto* is used deliberately: it supports Siglip2Model on recent
            # Transformers releases and still gives a helpful error on older
            # releases instead of importing transformers at application start.
            processor = AutoProcessor.from_pretrained(self.model_name)
            model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=self.dtype if self.device.type == "cuda" else torch.float32,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "启用 SigLIP2 需要先安装 transformers：pip install -r requirements.txt"
            ) from exc
        except Exception as exc:
            LOGGER.exception("SigLIP2 模型加载失败：%s", self.model_name)
            raise RuntimeError(f"SigLIP2 模型加载失败：{self.model_name}\n{exc}") from exc

        try:
            self.model = model.to(self.device).eval()
            self.processor = processor
        except Exception as exc:
            self.model = None
            self.processor = None
            raise RuntimeError(f"SigLIP2 设备初始化失败：{exc}") from exc

    def release(self) -> None:
        """Release model and cached text embeddings, including CUDA memory."""

        self.model = None
        self.processor = None
        self._text_embeddings = None
        self._text_cache_key = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_options(
        self,
        *,
        batch_size: int | None = None,
        prompt_template: str | None = None,
        prompt_ensemble: bool | None = None,
        precision: str | None = None,
    ) -> None:
        changed = False
        if batch_size is not None and max(1, int(batch_size)) != self.batch_size:
            self.batch_size = max(1, int(batch_size))
        if prompt_template is not None and prompt_template != self.prompt_template:
            self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
            changed = True
        if prompt_ensemble is not None and bool(prompt_ensemble) != self.prompt_ensemble:
            self.prompt_ensemble = bool(prompt_ensemble)
            changed = True
        if changed:
            self._text_cache_key = None
            self._text_embeddings = None
        if precision is not None:
            normalized = str(precision or "auto").casefold()
            if normalized != self.precision:
                self.precision = normalized
                new_dtype = self._select_dtype(normalized)
                if new_dtype != self.dtype:
                    if self.loaded:
                        self.release()
                    self.dtype = new_dtype
                    self._text_cache_key = None
                    self._text_embeddings = None

    def _templates(self) -> tuple[str, ...]:
        if not self.prompt_ensemble:
            return (self.prompt_template or DEFAULT_PROMPT_TEMPLATE,)
        # The configurable template remains the first member of an ensemble;
        # the other two are stable, useful defaults for advanced users.
        values = [self.prompt_template or DEFAULT_PROMPT_TEMPLATE]
        for template in DEFAULT_PROMPT_ENSEMBLE:
            if template not in values:
                values.append(template)
        return tuple(values)

    def _render_prompt(self, template: str, class_name: str) -> str:
        try:
            return template.format(class_name)
        except (IndexError, KeyError, ValueError):
            # A user may enter a literal prompt without ``{}``; treating it as
            # a prefix is friendlier than failing an entire batch.
            return f"{template.strip()} {class_name}".strip()

    @staticmethod
    def _move_inputs(inputs: Any, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
        values = dict(inputs)
        moved: dict[str, Any] = {}
        for key, value in values.items():
            if not isinstance(value, torch.Tensor):
                moved[key] = value
            elif value.is_floating_point() and key in {"pixel_values", "input_features"}:
                # Token-related tensors are integer inputs.  Only image
                # feature tensors should be cast to the model's CUDA dtype;
                # keeping any auxiliary floating tensors in their processor
                # dtype avoids subtle dtype mismatches across Transformers
                # versions.
                moved[key] = value.to(device=device, dtype=dtype)
            elif value.is_floating_point():
                moved[key] = value.to(device=device)
            else:
                moved[key] = value.to(device=device)
        return moved

    @staticmethod
    def _as_embedding(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        for name in ("pooler_output", "text_embeds", "image_embeds", "last_hidden_state"):
            value = getattr(output, name, None)
            if isinstance(value, torch.Tensor):
                if value.ndim == 3:
                    value = value[:, 0]
                return value
        if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        raise RuntimeError("SigLIP2 未返回可用的 embedding")

    def _text_features(self, inputs: dict[str, Any]) -> torch.Tensor:
        assert self.model is not None
        if hasattr(self.model, "get_text_features"):
            output = self.model.get_text_features(**inputs)
        else:
            output = self.model(**inputs)
        return F.normalize(self._as_embedding(output), dim=-1)

    def _image_features(self, inputs: dict[str, Any]) -> torch.Tensor:
        assert self.model is not None
        if hasattr(self.model, "get_image_features"):
            output = self.model.get_image_features(**inputs)
        else:
            output = self.model(**inputs)
        return F.normalize(self._as_embedding(output), dim=-1)

    def encode_classes(
        self,
        classes: Sequence[str],
        prompt_overrides: Mapping[str, str] | None = None,
    ) -> torch.Tensor:
        """Encode class prompts once and cache them until classes/template change."""

        self.load_model()
        clean = tuple(str(item).strip() for item in classes if str(item).strip())
        if not clean:
            raise ValueError("SigLIP2 至少需要一个检测类别")
        overrides = tuple(
            (name, str(prompt_overrides.get(name, "")))
            for name in clean
            if prompt_overrides and str(prompt_overrides.get(name, "")).strip()
        )
        key = (clean, self._templates(), overrides, self.device.type, str(self.dtype))
        if key == self._text_cache_key and self._text_embeddings is not None:
            return self._text_embeddings
        assert self.processor is not None
        prompts_by_template: list[torch.Tensor] = []
        with torch.inference_mode():
            for template in self._templates():
                prompts = [
                    self._render_prompt(dict(overrides).get(name, template), name)
                    for name in clean
                ]
                inputs = self.processor(text=prompts, padding="max_length", return_tensors="pt")
                text_inputs = self._move_inputs(inputs, self.device, self.dtype)
                prompts_by_template.append(self._text_features(text_inputs))
        embeddings = F.normalize(torch.stack(prompts_by_template).mean(dim=0), dim=-1)
        self.classes = list(clean)
        self._text_cache_key = key
        self._text_embeddings = embeddings
        return embeddings

    @staticmethod
    def crop_image(image: Image.Image, box: BoundingBox, padding_ratio: float = 0.10) -> Image.Image:
        """Crop one detector box with configurable, bounded context padding."""

        width, height = image.size
        padding = min(0.50, max(0.0, float(padding_ratio)))
        box_width = max(1.0, float(box.x2) - float(box.x1))
        box_height = max(1.0, float(box.y2) - float(box.y1))
        pad_x = box_width * padding
        pad_y = box_height * padding
        left = max(0, int(np.floor(box.x1 - pad_x)))
        top = max(0, int(np.floor(box.y1 - pad_y)))
        right = min(width, int(np.ceil(box.x2 + pad_x)))
        bottom = min(height, int(np.ceil(box.y2 + pad_y)))
        if right <= left:
            if left >= width:
                left, right = max(0, width - 1), width
            else:
                right = min(width, left + 1)
        if bottom <= top:
            if top >= height:
                top, bottom = max(0, height - 1), height
            else:
                bottom = min(height, top + 1)
        return image.crop((left, top, right, bottom))

    def _verify_chunk(
        self,
        crops: Sequence[Image.Image],
        text_embeddings: torch.Tensor,
        candidate_ids: Sequence[Sequence[int] | None] | None,
    ) -> list[SigLIPPrediction]:
        assert self.processor is not None
        inputs = self.processor(images=list(crops), return_tensors="pt")
        image_inputs = self._move_inputs(inputs, self.device, self.dtype)
        image_embeddings = self._image_features(image_inputs)
        similarities = image_embeddings @ text_embeddings.T
        # Convert ranking similarities to a stable 0..1 score over the class
        # candidates.  This is still a similarity score, not a probability.
        predictions: list[SigLIPPrediction] = []
        for row_index, row in enumerate(similarities):
            allowed = candidate_ids[row_index] if candidate_ids is not None else None
            if allowed:
                valid = [idx for idx in allowed if 0 <= int(idx) < len(self.classes)]
                if valid:
                    index_tensor = torch.tensor(valid, device=row.device, dtype=torch.long)
                    selected = row.index_select(0, index_tensor)
                    probs = torch.softmax(selected, dim=0)
                    best_offset = int(torch.argmax(probs).item())
                    best_id = valid[best_offset]
                    score = float(probs[best_offset].item())
                    scores = {idx: float(value.item()) for idx, value in zip(valid, probs)}
                else:
                    probs = torch.softmax(row, dim=0)
                    best_id = int(torch.argmax(probs).item())
                    score = float(probs[best_id].item())
                    scores = {idx: float(value.item()) for idx, value in enumerate(probs)}
            else:
                probs = torch.softmax(row, dim=0)
                best_id = int(torch.argmax(probs).item())
                score = float(probs[best_id].item())
                scores = {idx: float(value.item()) for idx, value in enumerate(probs)}
            ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            if len(ranked) > 1:
                top2_id, top2_score = ranked[1]
                top2_name = self.classes[int(top2_id)]
                margin = float(score) - float(top2_score)
            else:
                top2_id, top2_name, top2_score, margin = None, None, None, 0.0
            predictions.append(
                SigLIPPrediction(
                    best_id,
                    self.classes[best_id],
                    score,
                    scores,
                    top2_class_id=(None if top2_id is None else int(top2_id)),
                    top2_class_name=top2_name,
                    top2_score=top2_score,
                    margin=margin,
                )
            )
        return predictions

    def verify_batch(
        self,
        image: Image.Image | str | Path,
        boxes: Sequence[BoundingBox],
        classes: Sequence[str],
        *,
        padding_ratio: float = 0.10,
        candidate_ids: Sequence[Sequence[int] | None] | None = None,
    ) -> list[SigLIPPrediction]:
        """Verify all YOLO boxes in *image*, retrying smaller batches on OOM."""

        if not boxes:
            return []
        self.load_model()
        text_embeddings = self.encode_classes(classes)
        if isinstance(image, (str, Path)):
            with Image.open(image) as loaded:
                pil_image = loaded.convert("RGB")
        else:
            pil_image = image.convert("RGB")
        crops = [self.crop_image(pil_image, box, padding_ratio) for box in boxes]
        if candidate_ids is not None and len(candidate_ids) != len(crops):
            raise ValueError("candidate_ids 数量必须与检测框一致")

        predictions: list[SigLIPPrediction] = []
        index = 0
        batch_size = max(1, self.batch_size)
        while index < len(crops):
            current = min(batch_size, len(crops) - index)
            try:
                chunk_ids = None if candidate_ids is None else candidate_ids[index : index + current]
                predictions.extend(
                    self._verify_chunk(crops[index : index + current], text_embeddings, chunk_ids)
                )
                index += current
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
                if not is_oom:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if current > 1:
                    batch_size = max(1, current // 2)
                    LOGGER.warning("SigLIP2 显存不足，自动将 batch size 降到 %d", batch_size)
                    continue
                raise RuntimeError(
                    "SigLIP2 显存不足，即使 batch size=1 仍无法完成。请降低 Image Size、关闭其他 GPU 程序或关闭 SigLIP2。"
                ) from exc
        return predictions
