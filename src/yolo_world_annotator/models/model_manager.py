"""Centralized model lifecycle and graceful degradation helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from yolo_world_annotator.models.siglip_verifier import DEFAULT_SIGLIP_MODEL, SigLIPVerifier
from yolo_world_annotator.models.vlm_verifier import DEFAULT_VLM_MODEL, VLMVerifier
from yolo_world_annotator.models.yolo_world import YOLOWorldDetector

LOGGER = logging.getLogger(__name__)


class ModelManager:
    """Keep YOLO resident, cache SigLIP and lazy-load VLM on demand."""

    def __init__(self, *, app_root: Path | None = None) -> None:
        self.app_root = (app_root or Path(__file__).resolve().parents[1]).resolve()
        self.detector: YOLOWorldDetector | None = None
        self.siglip: SigLIPVerifier | None = None
        self.vlm: VLMVerifier | None = None
        self.last_errors: dict[str, str] = {}

    @property
    def yolo(self) -> YOLOWorldDetector | None:
        return self.detector

    @property
    def siglip_verifier(self) -> SigLIPVerifier | None:
        return self.siglip

    @property
    def vlm_verifier(self) -> VLMVerifier | None:
        return self.vlm

    def load_yolo(self, model_path: Path | str, classes: list[str]) -> YOLOWorldDetector:
        path = Path(model_path).resolve()
        try:
            if self.detector is None or self.detector.model_path != path:
                self.detector = YOLOWorldDetector(path)
            self.detector.set_classes(list(classes))
            self.last_errors.pop("yolo", None)
            return self.detector
        except Exception as exc:
            self.last_errors["yolo"] = str(exc)
            self.detector = None
            raise

    def load_siglip(
        self,
        model_name: str = DEFAULT_SIGLIP_MODEL,
        *,
        classes: list[str] | None = None,
        batch_size: int = 4,
        prompt_template: str = "a photo of a {}",
        prompt_ensemble: bool = False,
        precision: str = "auto",
    ) -> SigLIPVerifier:
        try:
            if self.siglip is None or self.siglip.model_name != model_name:
                self.unload_siglip()
                self.siglip = SigLIPVerifier(
                    model_name,
                    batch_size=batch_size,
                    prompt_template=prompt_template,
                    prompt_ensemble=prompt_ensemble,
                    precision=precision,
                )
            self.siglip.set_options(
                batch_size=batch_size,
                prompt_template=prompt_template,
                prompt_ensemble=prompt_ensemble,
                precision=precision,
            )
            self.siglip.load_model()
            if classes:
                self.siglip.encode_classes(classes)
            self.last_errors.pop("siglip", None)
            return self.siglip
        except Exception as exc:
            self.last_errors["siglip"] = str(exc)
            self.unload_siglip()
            raise

    def load_vlm(
        self,
        model_name: str = DEFAULT_VLM_MODEL,
        *,
        lazy_load: bool = True,
        max_new_tokens: int = 128,
        low_memory: bool = True,
        generator=None,
        load_now: bool = False,
    ) -> VLMVerifier:
        try:
            if self.vlm is None or self.vlm.model_name != model_name:
                self.unload_vlm()
                self.vlm = VLMVerifier(
                    model_name,
                    lazy_load=lazy_load,
                    max_new_tokens=max_new_tokens,
                    low_memory=low_memory,
                    generator=generator,
                )
            if load_now and generator is None:
                self.vlm.load_model()
            self.last_errors.pop("vlm", None)
            return self.vlm
        except Exception as exc:
            self.last_errors["vlm"] = str(exc)
            self.unload_vlm()
            raise

    def unload_siglip(self) -> None:
        if self.siglip is not None:
            try:
                self.siglip.release()
            except Exception:
                LOGGER.exception("释放 SigLIP2 失败")
        self.siglip = None
        self.clear_cuda_cache()

    def unload_vlm(self) -> None:
        if self.vlm is not None:
            try:
                self.vlm.release()
            except Exception:
                LOGGER.exception("释放 VLM 失败")
        self.vlm = None
        self.clear_cuda_cache()

    def clear_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release(self) -> None:
        self.unload_vlm()
        self.unload_siglip()
        self.detector = None
        self.clear_cuda_cache()

    close = release


ModelLifecycleManager = ModelManager


__all__ = ["ModelManager", "ModelLifecycleManager"]
