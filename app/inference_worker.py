from __future__ import annotations

import logging
import threading
import time
from copy import deepcopy
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from core.verification import fuse_box
from core.decision_engine import DecisionEngine
from inference.sahi_runner import SAHIConfig, SAHIInferenceRunner, generate_tiles
from models.model_manager import ModelManager
from models.yolo_world import YOLOWorldDetector
from models.siglip_verifier import SigLIPVerifier
from models.vlm_verifier import VLMVerifier, VLMTriggerPolicy
from utils.config import as_bool


LOGGER = logging.getLogger(__name__)


class InferenceWorker(QObject):
    model_loaded = Signal(str)
    pipeline_ready = Signal(str, object)
    qwen_log = Signal(str)
    prediction_ready = Signal(str, object)
    batch_item_ready = Signal(str, object, int, int, float)
    batch_item_failed = Signal(str, str, int, int)
    batch_finished = Signal(bool, int, int)
    batch_paused = Signal(bool)
    failed = Signal(str, str)

    def __init__(self) -> None:
        super().__init__()
        self.detector: YOLOWorldDetector | None = None
        self.siglip_verifier: SigLIPVerifier | None = None
        self._siglip_model_name: str | None = None
        self._siglip_error: str | None = None
        self.vlm_verifier: VLMVerifier | None = None
        self._vlm_model_name: str | None = None
        self.sahi_runner: SAHIInferenceRunner | None = None
        self.model_manager = ModelManager()
        self.last_pipeline_metrics: dict = {}
        self._run_event = threading.Event()
        self._run_event.set()
        self._cancel_event = threading.Event()

    def _emit_qwen_log(self, message: str) -> None:
        """Send a human-readable progress line to the GUI log panel."""

        timestamp = time.strftime("%H:%M:%S")
        self.qwen_log.emit(f"[{timestamp}] {message}")

    @Slot(object)
    def load_model(self, payload: dict) -> None:
        try:
            model_path = Path(payload["model_path"])
            self._emit_qwen_log(f"模型准备：{model_path.name}，类别 {len(payload.get('classes', []))} 个")
            if self.detector is None or self.detector.model_path != model_path.resolve():
                self.detector = YOLOWorldDetector(model_path)
            classes = list(payload["classes"])
            self.detector.set_classes(classes)
            self._configure_siglip(payload, classes)
            description = self.detector.device_description
            if self.siglip_verifier is not None:
                description += f"；SigLIP2：{self.siglip_verifier.device_description}"
            elif self._siglip_error:
                description += f"；SigLIP2 回退到关闭（{self._siglip_error}）"
            self.model_loaded.emit(description)
            self._emit_qwen_log(f"YOLO 模型已加载：{description}")
        except Exception as exc:
            LOGGER.exception("模型加载失败")
            self._emit_qwen_log(f"模型加载失败：{exc}")
            self.failed.emit("load", str(exc))

    def _configure_siglip(self, payload: dict, classes: list[str]) -> None:
        enabled = as_bool(payload.get("siglip_enabled", False))
        if not enabled:
            if self.siglip_verifier is not None:
                self.siglip_verifier.release()
            self.siglip_verifier = None
            self._siglip_model_name = None
            self._siglip_error = None
            return
        try:
            model_name = str(payload.get("siglip_model", "google/siglip2-base-patch16-224"))
            if self.siglip_verifier is None or self._siglip_model_name != model_name:
                if self.siglip_verifier is not None:
                    self.siglip_verifier.release()
                self.siglip_verifier = SigLIPVerifier(
                    model_name,
                    batch_size=int(payload.get("siglip_batch_size", 4)),
                    prompt_template=str(payload.get("siglip_prompt_template", "a photo of a {}")),
                    prompt_ensemble=as_bool(payload.get("siglip_prompt_ensemble", False)),
                    precision=str(payload.get("siglip_precision", "auto")),
                )
                self._siglip_model_name = model_name
            self.siglip_verifier.set_options(
                batch_size=int(payload.get("siglip_batch_size", 4)),
                prompt_template=str(payload.get("siglip_prompt_template", "a photo of a {}")),
                prompt_ensemble=as_bool(payload.get("siglip_prompt_ensemble", False)),
                precision=str(payload.get("siglip_precision", "auto")),
            )
            # Load once at task start and precompute text embeddings.  Subsequent
            # images reuse both the model and the class/prompt cache.
            self.siglip_verifier.load_model()
            profiles = payload.get("class_profiles", {})
            prompt_overrides = {
                str(name): str(profile.get("siglip_prompt", ""))
                for name, profile in profiles.items()
                if isinstance(profile, dict) and str(profile.get("siglip_prompt", "")).strip()
            } if isinstance(profiles, dict) else {}
            try:
                self.siglip_verifier.encode_classes(classes, prompt_overrides=prompt_overrides)
            except TypeError:
                # Keep compatibility with lightweight test/dummy verifiers
                # implementing the original one-argument API.
                self.siglip_verifier.encode_classes(classes)
            self._siglip_error = None
        except Exception as exc:
            # SigLIP is an optional verifier.  Keep the detector usable and
            # make the fallback visible instead of failing the whole task.
            LOGGER.exception("SigLIP2 加载失败，回退到 YOLO/SAHI：%s", exc)
            self._siglip_error = str(exc)
            if self.siglip_verifier is not None:
                self.siglip_verifier.release()
            self.siglip_verifier = None
            self._siglip_model_name = None

    def _configure_vlm(self, payload: dict) -> None:
        enabled = as_bool(payload.get("vlm_enabled", False))
        if not enabled:
            if self.vlm_verifier is not None:
                self.vlm_verifier.release()
            self.vlm_verifier = None
            self._vlm_model_name = None
            return
        model_name = str(payload.get("vlm_model", "Qwen/Qwen3-VL-8B-Instruct"))
        if self.vlm_verifier is None or self._vlm_model_name != model_name:
            self.vlm_verifier = VLMVerifier(
                model_name,
                lazy_load=as_bool(payload.get("vlm_lazy_load", True), True),
                max_new_tokens=int(payload.get("vlm_max_new_tokens", 128)),
                low_memory=as_bool(payload.get("vlm_low_memory", True), True),
            )
            self._vlm_model_name = model_name
            self._emit_qwen_log(
                f"Qwen 已配置：{model_name}（{'懒加载' if as_bool(payload.get('vlm_lazy_load', True), True) else '启动时加载'}）"
            )

    @staticmethod
    def _sahi_enabled(payload: dict) -> bool:
        mode = str(payload.get("inference_mode", "NORMAL")).upper()
        return as_bool(payload.get("sahi_enabled", False)) or "SAHI" in mode or "TILED" in mode

    def _detect(self, image_path: Path, payload: dict) -> list:
        confidence = float(payload["confidence"])
        iou = float(payload["iou"])
        imgsz = int(payload["imgsz"])
        if not self._sahi_enabled(payload):
            started = time.perf_counter()
            boxes = self.detector.predict(image_path, confidence=confidence, iou=iou, imgsz=imgsz)
            elapsed = time.perf_counter() - started
            self.last_pipeline_metrics = {
                "inference_mode": "NORMAL",
                "sahi_enabled": False,
                "tile_count": 0,
                "raw_box_count": len(boxes),
                "merged_box_count": len(boxes),
                "yolo_normal_time": elapsed,
                "total_time": elapsed,
                "tile_rects": [],
            }
            self._emit_qwen_log(
                f"{image_path.name}：YOLO 完成（普通整图推理），检测框 {len(boxes)} 个"
            )
            for box in boxes:
                box.inference_mode = "NORMAL"
                box.sahi_enabled = False
                box.sahi_tile_count = 0
            return boxes
        config = SAHIConfig.from_mapping(
            {
                "slice_width": payload.get("sahi_slice_width", 1024),
                "slice_height": payload.get("sahi_slice_height", 1024),
                "overlap_width_ratio": payload.get("sahi_overlap_width_ratio", 0.20),
                "overlap_height_ratio": payload.get("sahi_overlap_height_ratio", 0.20),
                "postprocess_type": payload.get("sahi_postprocess_type", "NMS"),
                "postprocess_match_threshold": payload.get("sahi_postprocess_match_threshold", payload.get("sahi_merge_threshold", 0.50)),
                "postprocess_match_metric": payload.get("sahi_postprocess_match_metric", payload.get("sahi_merge_metric", "IOU")),
                "max_tiles": payload.get("sahi_max_tiles", 0),
            }
        )
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                tile_rects = [
                    {
                        "index": tile.index,
                        "x1": tile.x1,
                        "y1": tile.y1,
                        "x2": tile.x2,
                        "y2": tile.y2,
                    }
                    for tile in generate_tiles(image.width, image.height, config)
                ]
        except Exception:
            LOGGER.exception("生成 SAHI 预览切片失败：%s", image_path)
            tile_rects = []
        self._emit_qwen_log(
            f"{image_path.name}：开始 SAHI 图像分割/切片，预览切片 {len(tile_rects)} 个"
        )
        try:
            if self.sahi_runner is None or self.sahi_runner.detector is not self.detector:
                self.sahi_runner = SAHIInferenceRunner(self.detector)
            result = self.sahi_runner.run(
                image_path,
                config=config,
                confidence=confidence,
                iou=iou,
                imgsz=imgsz,
            )
            self.last_pipeline_metrics = result.to_dict()
            # Keep the worker/UI key name consistent with normal inference.
            self.last_pipeline_metrics["sahi_enabled"] = not result.fallback
            self.last_pipeline_metrics["inference_mode"] = "SAHI" if not result.fallback else "NORMAL_FALLBACK"
            self.last_pipeline_metrics["tile_rects"] = tile_rects
            self._emit_qwen_log(
                f"{image_path.name}：SAHI 完成，raw={result.raw_box_count}，merged={result.merged_box_count}"
            )
            return list(result)
        except Exception as exc:
            # A failed tile should not silently drop an image.  Fall back to a
            # normal detector pass and expose the reason in the metrics/log.
            LOGGER.exception("SAHI 推理失败，回退普通推理：%s", image_path)
            boxes = self.detector.predict(image_path, confidence=confidence, iou=iou, imgsz=imgsz)
            self.last_pipeline_metrics = {
                "inference_mode": "NORMAL_FALLBACK",
                "sahi_enabled": False,
                "tile_count": 0,
                "raw_box_count": len(boxes),
                "merged_box_count": len(boxes),
                "fallback": True,
                "error": str(exc),
                "yolo_normal_time": 0.0,
                "tile_rects": tile_rects,
            }
            self._emit_qwen_log(f"{image_path.name}：SAHI 失败，已回退普通整图推理：{exc}")
            for box in boxes:
                box.inference_mode = "NORMAL_FALLBACK"
                box.sahi_enabled = False
            return boxes

    def _predict_with_verification(self, image_path: Path, boxes: list, payload: dict) -> list:
        verification_started = time.perf_counter()
        predictions = []
        force_each_image = as_bool(payload.get("vlm_check_each_image", False))
        siglip_started = time.perf_counter()
        if as_bool(payload.get("siglip_enabled", False)):
            if self.siglip_verifier is None:
                LOGGER.warning("SigLIP2 不可用，使用 YOLO/SAHI 结果：%s", self._siglip_error or "unknown")
                payload = dict(payload)
                payload["siglip_enabled"] = False
                payload["siglip_fallback_reason"] = self._siglip_error or "SigLIP2 验证器尚未加载"
            else:
                try:
                    candidate_top_k = int(payload.get("candidate_top_k", 0))
                    candidate_ids = None
                    if candidate_top_k > 0:
                        candidate_ids = [getattr(box, "candidate_class_ids", None) for box in boxes]
                    predictions = self.siglip_verifier.verify_batch(
                        image_path,
                        boxes,
                        list(payload["classes"]),
                        padding_ratio=float(payload.get("siglip_padding", 0.10)),
                        candidate_ids=candidate_ids,
                    )
                    if len(predictions) != len(boxes):
                        raise RuntimeError("SigLIP2 返回结果数量与 YOLO 检测框不一致")
                    boxes = [
                        fuse_box(
                            box,
                            prediction,
                            classes=list(payload["classes"]),
                            yolo_weight=float(payload.get("yolo_weight", 0.65)),
                            siglip_weight=float(payload.get("siglip_weight", 0.35)),
                            auto_accept_threshold=float(payload.get("auto_accept_threshold", 0.75)),
                            review_threshold=float(payload.get("review_threshold", 0.50)),
                            per_class_thresholds=(
                                payload.get("per_class_thresholds", {})
                                if isinstance(payload.get("per_class_thresholds", {}), dict)
                                else {}
                            ),
                        )
                        for box, prediction in zip(boxes, predictions, strict=True)
                    ]
                except Exception as exc:
                    LOGGER.exception("SigLIP2 单图验证失败，回退到 YOLO/SAHI：%s", exc)
                    self._siglip_error = str(exc)
                    payload = dict(payload)
                    payload["siglip_enabled"] = False
                    payload["siglip_fallback_reason"] = str(exc)
                    predictions = []
        siglip_elapsed = time.perf_counter() - siglip_started
        # The decision engine records state/reason even when VLM is disabled.
        engine = DecisionEngine(
            yolo_weight=float(payload.get("yolo_weight", 0.65)),
            siglip_weight=float(payload.get("siglip_weight", 0.35)),
            auto_accept_threshold=float(payload.get("auto_accept_threshold", 0.75)),
            review_threshold=float(payload.get("review_threshold", 0.50)),
            trigger_policy=VLMTriggerPolicy(
                yolo_low_threshold=float(payload.get("vlm_yolo_low_threshold", 0.45)),
                siglip_low_threshold=float(payload.get("vlm_siglip_low_threshold", 0.55)),
                margin_threshold=float(payload.get("vlm_margin_threshold", 0.10)),
                force_special_classes=(
                    as_bool(payload.get("vlm_force_special_classes", True), True)
                    or force_each_image
                ),
            ),
        )
        self._configure_vlm(payload)
        profiles = payload.get("class_profiles", {})
        profile_map = profiles if isinstance(profiles, dict) else {}
        enriched: list = []
        vlm_elapsed = 0.0
        vlm_calls = 0
        vlm_enabled = as_bool(payload.get("vlm_enabled", False))
        force_each_image = vlm_enabled and force_each_image
        if vlm_enabled and force_each_image:
            self._emit_qwen_log(
                f"{image_path.name}：逐图 Qwen 检查已启用，将检查 {len(boxes)} 个 YOLO 框"
            )
        elif vlm_enabled:
            self._emit_qwen_log(f"{image_path.name}：Qwen 困难样本策略检查，共 {len(boxes)} 个 YOLO 框")
        if vlm_enabled and self.vlm_verifier is None:
            self._emit_qwen_log(f"{image_path.name}：Qwen 验证器不可用，结果将保留为不确定")
        for index, box in enumerate(boxes):
            prediction = predictions[index] if index < len(predictions) else None
            profile = profile_map.get(str(box.class_name), profile_map.get(box.class_name, {}))
            always_vlm = bool(profile.get("always_vlm_verify", False)) if isinstance(profile, dict) else bool(getattr(profile, "always_vlm_verify", False))
            preliminary = engine.decide(
                box,
                prediction,
                classes=list(payload["classes"]),
                enable_vlm=False,
                always_vlm_verify=always_vlm,
                sahi_enabled=bool(getattr(box, "sahi_enabled", False)),
            )
            vlm_result = None
            if vlm_enabled:
                trigger, trigger_reasons = engine.should_trigger_vlm(
                    yolo_score=box.yolo_confidence if box.yolo_confidence is not None else box.confidence,
                    siglip_score=box.siglip_score,
                    siglip_margin=box.siglip_margin,
                    agreement=box.agreement,
                    decision_status=preliminary.status,
                    always_vlm_verify=always_vlm,
                )
                if force_each_image:
                    trigger = True
                    trigger_reasons = ["EVERY_IMAGE"] + trigger_reasons
                if trigger and self.vlm_verifier is not None:
                    self._emit_qwen_log(
                        f"{image_path.name} / 框 {index + 1}/{len(boxes)}：开始 Qwen 检查，"
                        f"原因={','.join(dict.fromkeys(trigger_reasons)) or '策略触发'}"
                    )
                    vlm_started = time.perf_counter()
                    vlm_result = self.vlm_verifier.verify(
                        image_path,
                        target_class=str(box.class_name),
                        profile=profile,
                        candidate_classes=list(payload["classes"]),
                        box=box,
                        padding_ratio=float(payload.get("siglip_padding", 0.10)),
                    )
                    vlm_elapsed += time.perf_counter() - vlm_started
                    vlm_calls += 1
                    raw = str(getattr(vlm_result, "raw_response", "") or "").strip()
                    if len(raw) > 4000:
                        raw = raw[:4000] + "…"
                    self._emit_qwen_log(
                        f"{image_path.name} / 框 {index + 1}：结果={vlm_result.final_result}，"
                        f"解析={'成功' if vlm_result.parsed else '失败'}，"
                        f"置信度={vlm_result.confidence if vlm_result.confidence is not None else '—'}"
                    )
                    if raw:
                        self._emit_qwen_log(f"原始 JSON：{raw}")
                    if vlm_result.parse_error:
                        self._emit_qwen_log(f"解析提示：{vlm_result.parse_error}")
                elif trigger:
                    self._emit_qwen_log(
                        f"{image_path.name} / 框 {index + 1}：需要 Qwen，但验证器尚未可用"
                    )
                else:
                    self._emit_qwen_log(
                        f"{image_path.name} / 框 {index + 1}：按困难样本策略跳过 Qwen"
                    )
            final = engine.decide(
                box,
                prediction,
                classes=list(payload["classes"]),
                enable_vlm=vlm_enabled,
                vlm_result=vlm_result,
                profile=profile,
                always_vlm_verify=(always_vlm or force_each_image),
                sahi_enabled=bool(getattr(box, "sahi_enabled", False)),
            )
            box = engine.apply(box, final, vlm_result=vlm_result)
            box.vlm_enabled = vlm_enabled
            if payload.get("siglip_fallback_reason"):
                box.decision_reason = (
                    f"SigLIP2 不可用，已回退：{payload['siglip_fallback_reason']}"
                )
            enriched.append(box)
        if vlm_enabled and not boxes:
            self._emit_qwen_log(f"{image_path.name}：YOLO 没有检测框，无法进行框级 Qwen 检查")
        self.last_pipeline_metrics.update(
            {
                "siglip_time": siglip_elapsed if as_bool(payload.get("siglip_enabled", False)) else 0.0,
                "vlm_time": vlm_elapsed,
                "vlm_calls": vlm_calls,
                "verification_time": time.perf_counter() - verification_started,
            }
        )
        return enriched

    @Slot(object)
    def predict_single(self, payload: dict) -> None:
        image_path = str(payload.get("image_path", ""))
        try:
            if self.detector is None:
                raise RuntimeError("请先加载模型")
            classes = list(payload["classes"])
            self.detector.set_classes(classes)
            self._configure_siglip(payload, classes)
            boxes = self._detect(Path(image_path), payload)
            boxes = self._predict_with_verification(Path(image_path), boxes, payload)
            self.pipeline_ready.emit(image_path, deepcopy(self.last_pipeline_metrics))
            self.prediction_ready.emit(image_path, boxes)
        except Exception as exc:
            LOGGER.exception("单图推理失败：%s", image_path)
            self.failed.emit("predict", str(exc))

    @Slot(object)
    def start_batch(self, payload: dict) -> None:
        images = [Path(item) for item in payload.get("images", [])]
        total = len(images)
        completed = 0
        started = time.perf_counter()
        self._cancel_event.clear()
        self._run_event.set()
        try:
            if self.detector is None:
                raise RuntimeError("请先加载模型")
            classes = list(payload["classes"])
            self.detector.set_classes(classes)
            self._configure_siglip(payload, classes)
            for image_path in images:
                while not self._run_event.wait(0.1):
                    if self._cancel_event.is_set():
                        break
                if self._cancel_event.is_set():
                    break
                try:
                    boxes = self._detect(image_path, payload)
                    boxes = self._predict_with_verification(image_path, boxes, payload)
                    self.pipeline_ready.emit(str(image_path), deepcopy(self.last_pipeline_metrics))
                    completed += 1
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    self.batch_item_ready.emit(
                        str(image_path), boxes, completed, total, completed / elapsed
                    )
                except Exception as exc:
                    LOGGER.exception("批量推理跳过失败图片：%s", image_path)
                    completed += 1
                    self.batch_item_failed.emit(str(image_path), str(exc), completed, total)
            self.batch_finished.emit(self._cancel_event.is_set(), completed, total)
        except Exception as exc:
            LOGGER.exception("批量推理启动失败")
            self.failed.emit("batch", str(exc))
            self.batch_finished.emit(True, completed, total)

    def request_pause(self) -> None:
        self._run_event.clear()
        self.batch_paused.emit(True)

    def request_resume(self) -> None:
        self._run_event.set()
        self.batch_paused.emit(False)

    def request_cancel(self) -> None:
        self._cancel_event.set()
        self._run_event.set()

    @Slot()
    def release_models(self) -> None:
        """Release optional verifier memory when the application closes."""

        if self.siglip_verifier is not None:
            self.siglip_verifier.release()
        self.siglip_verifier = None
        self._siglip_model_name = None
        self._siglip_error = None
        if self.vlm_verifier is not None:
            self.vlm_verifier.release()
        self.vlm_verifier = None
        self._vlm_model_name = None
        self.sahi_runner = None
        self.model_manager.release()
        self.detector = None
