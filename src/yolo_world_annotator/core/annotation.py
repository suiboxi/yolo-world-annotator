from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


def _as_bool(value: Any, default: bool = False) -> bool:
    """Parse JSON booleans while remaining tolerant of legacy string values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    return default


class AnnotationStatus(StrEnum):
    UNLABELED = "UNLABELED"
    AUTO_LABELED = "AUTO_LABELED"
    VERIFIED = "VERIFIED"


@dataclass(slots=True)
class BoundingBox:
    class_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float | None = None
    source: str = "MANUAL"
    # The fields below are optional metadata introduced by the YOLO-World +
    # SigLIP2 verifier.  They are appended after the legacy fields so old
    # positional constructors continue to work unchanged.
    yolo_class_id: int | None = None
    yolo_class_name: str | None = None
    yolo_confidence: float | None = None
    siglip_enabled: bool = False
    siglip_class_id: int | None = None
    siglip_class_name: str | None = None
    siglip_score: float | None = None
    agreement: bool | None = None
    combined_confidence: float | None = None
    fusion_status: str | None = None
    review_required: bool = False
    review_confirmed: bool = False
    human_modified: bool = False
    candidate_class_ids: list[int] | None = None
    # Pipeline provenance.  These fields are appended so legacy positional
    # BoundingBox(...) constructors remain source-compatible.
    inference_mode: str = "NORMAL"
    sahi_enabled: bool = False
    sahi_tile_count: int = 0
    sahi_tile_index: int | None = None
    siglip_top2_class_id: int | None = None
    siglip_top2_class_name: str | None = None
    siglip_top2_score: float | None = None
    siglip_margin: float | None = None
    vlm_enabled: bool = False
    vlm_triggered: bool = False
    vlm_model: str | None = None
    vlm_target_class: str | None = None
    vlm_features: dict[str, str] = field(default_factory=dict)
    vlm_final_result: str | None = None
    vlm_confidence: float | None = None
    vlm_parse_error: str | None = None
    decision_state: str | None = None
    decision_reason: str | None = None

    def __post_init__(self) -> None:
        # Existing YOLO-only annotations only had ``confidence`` and
        # ``source``.  Populate the explicit YOLO fields lazily so the new
        # metadata can be added without changing old project files.
        if self.yolo_confidence is None and self.source in {
            "YOLO-World",
            "YOLO-World+SigLIP2",
        }:
            self.yolo_confidence = self.confidence
        if self.yolo_class_id is None and self.source in {
            "YOLO-World",
            "YOLO-World+SigLIP2",
        }:
            self.yolo_class_id = int(self.class_id)
        if self.yolo_class_name is None and self.source in {
            "YOLO-World",
            "YOLO-World+SigLIP2",
        }:
            self.yolo_class_name = str(self.class_name)

    def normalized(self, image_width: int, image_height: int) -> "BoundingBox":
        """Return an ordered box clamped to the original image bounds."""
        left, right = sorted((float(self.x1), float(self.x2)))
        top, bottom = sorted((float(self.y1), float(self.y2)))
        left = min(max(left, 0.0), float(image_width))
        right = min(max(right, 0.0), float(image_width))
        top = min(max(top, 0.0), float(image_height))
        bottom = min(max(bottom, 0.0), float(image_height))
        if right <= left or bottom <= top:
            raise ValueError("标注框宽高必须大于 0")
        return BoundingBox(
            class_id=int(self.class_id),
            class_name=str(self.class_name),
            x1=left,
            y1=top,
            x2=right,
            y2=bottom,
            confidence=None if self.confidence is None else float(self.confidence),
            source=str(self.source),
            yolo_class_id=self.yolo_class_id,
            yolo_class_name=self.yolo_class_name,
            yolo_confidence=(
                None if self.yolo_confidence is None else float(self.yolo_confidence)
            ),
            siglip_enabled=bool(self.siglip_enabled),
            siglip_class_id=self.siglip_class_id,
            siglip_class_name=self.siglip_class_name,
            siglip_score=(None if self.siglip_score is None else float(self.siglip_score)),
            agreement=self.agreement,
            combined_confidence=(
                None
                if self.combined_confidence is None
                else float(self.combined_confidence)
            ),
            fusion_status=self.fusion_status,
            review_required=bool(self.review_required),
            review_confirmed=bool(self.review_confirmed),
            human_modified=bool(self.human_modified),
            candidate_class_ids=(
                None if self.candidate_class_ids is None else list(self.candidate_class_ids)
            ),
            inference_mode=str(self.inference_mode or "NORMAL"),
            sahi_enabled=bool(self.sahi_enabled),
            sahi_tile_count=max(0, int(self.sahi_tile_count or 0)),
            sahi_tile_index=(
                None if self.sahi_tile_index is None else int(self.sahi_tile_index)
            ),
            siglip_top2_class_id=self.siglip_top2_class_id,
            siglip_top2_class_name=self.siglip_top2_class_name,
            siglip_top2_score=(
                None if self.siglip_top2_score is None else float(self.siglip_top2_score)
            ),
            siglip_margin=(None if self.siglip_margin is None else float(self.siglip_margin)),
            vlm_enabled=bool(self.vlm_enabled),
            vlm_triggered=bool(self.vlm_triggered),
            vlm_model=self.vlm_model,
            vlm_target_class=self.vlm_target_class,
            vlm_features=dict(self.vlm_features or {}),
            vlm_final_result=self.vlm_final_result,
            vlm_confidence=(None if self.vlm_confidence is None else float(self.vlm_confidence)),
            vlm_parse_error=self.vlm_parse_error,
            decision_state=self.decision_state,
            decision_reason=self.decision_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = [data.pop("x1"), data.pop("y1"), data.pop("x2"), data.pop("y2")]
        # Keep the legacy flat ``confidence``/``source`` keys and also expose
        # the structured fields requested by the verifier workflow.
        data["yolo_confidence"] = data.get("yolo_confidence")
        data["siglip"] = {
            "enabled": bool(self.siglip_enabled),
            "predicted_class_id": self.siglip_class_id,
            "predicted_class": self.siglip_class_name,
            "score": self.siglip_score,
            "agreement": self.agreement,
        }
        data["fusion"] = {
            "combined_confidence": self.combined_confidence,
            "status": self.fusion_status,
        }
        data["review"] = {
            "required": bool(self.review_required),
            "confirmed": bool(self.review_confirmed),
        }
        data["sahi"] = {
            "enabled": bool(self.sahi_enabled),
            "tile_count": int(self.sahi_tile_count or 0),
            "tile_index": self.sahi_tile_index,
            "mode": self.inference_mode,
        }
        data["siglip"].update(
            {
                "top2_class_id": self.siglip_top2_class_id,
                "top2_class": self.siglip_top2_class_name,
                "top2_score": self.siglip_top2_score,
                "margin": self.siglip_margin,
            }
        )
        data["vlm"] = {
            "enabled": bool(self.vlm_enabled),
            "triggered": bool(self.vlm_triggered),
            "model": self.vlm_model,
            "target_class": self.vlm_target_class,
            "features": dict(self.vlm_features or {}),
            "final_result": self.vlm_final_result,
            "self_reported_confidence": self.vlm_confidence,
            "parse_error": self.vlm_parse_error,
        }
        data["decision"] = {
            "state": self.decision_state or self.fusion_status,
            "reason": self.decision_reason,
        }
        # Do not duplicate the nested structures as implementation-only keys.
        for key in (
            "siglip_enabled",
            "siglip_class_id",
            "siglip_class_name",
            "siglip_score",
            "agreement",
            "combined_confidence",
            "fusion_status",
            "review_required",
            "review_confirmed",
        ):
            data.pop(key, None)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BoundingBox":
        bbox = data.get("bbox", [data.get("x1"), data.get("y1"), data.get("x2"), data.get("y2")])
        if len(bbox) != 4 or any(value is None for value in bbox):
            raise ValueError("annotations.json 中的 bbox 无效")
        siglip = data.get("siglip") if isinstance(data.get("siglip"), dict) else {}
        fusion = data.get("fusion") if isinstance(data.get("fusion"), dict) else {}
        review = data.get("review") if isinstance(data.get("review"), dict) else {}
        sahi = data.get("sahi") if isinstance(data.get("sahi"), dict) else {}
        vlm = data.get("vlm") if isinstance(data.get("vlm"), dict) else {}
        decision = data.get("decision") if isinstance(data.get("decision"), dict) else {}
        return cls(
            class_id=int(data["class_id"]),
            class_name=str(data.get("class_name", "")),
            x1=float(bbox[0]),
            y1=float(bbox[1]),
            x2=float(bbox[2]),
            y2=float(bbox[3]),
            confidence=(None if data.get("confidence") is None else float(data["confidence"])),
            source=str(data.get("source", "MANUAL")),
            yolo_class_id=(
                None
                if data.get("yolo_class_id") is None
                else int(data["yolo_class_id"])
            ),
            yolo_class_name=(
                None if data.get("yolo_class_name") is None else str(data["yolo_class_name"])
            ),
            yolo_confidence=(
                None
                if data.get("yolo_confidence") is None
                else float(data["yolo_confidence"])
            ),
            siglip_enabled=_as_bool(siglip.get("enabled", data.get("siglip_enabled", False))),
            siglip_class_id=(
                None
                if siglip.get("predicted_class_id", data.get("siglip_class_id")) is None
                else int(siglip.get("predicted_class_id", data.get("siglip_class_id")))
            ),
            siglip_class_name=(
                None
                if siglip.get("predicted_class", data.get("siglip_class_name")) is None
                else str(siglip.get("predicted_class", data.get("siglip_class_name")))
            ),
            siglip_score=(
                None
                if siglip.get("score", data.get("siglip_score")) is None
                else float(siglip.get("score", data.get("siglip_score")))
            ),
            agreement=(
                None
                if siglip.get("agreement", data.get("agreement")) is None
                else _as_bool(siglip.get("agreement", data.get("agreement")))
            ),
            combined_confidence=(
                None
                if fusion.get("combined_confidence", data.get("combined_confidence")) is None
                else float(fusion.get("combined_confidence", data.get("combined_confidence")))
            ),
            fusion_status=fusion.get("status", data.get("fusion_status")),
            review_required=_as_bool(review.get("required", data.get("review_required", False))),
            review_confirmed=_as_bool(review.get("confirmed", data.get("review_confirmed", False))),
            human_modified=_as_bool(data.get("human_modified", False)),
            candidate_class_ids=(
                None
                if data.get("candidate_class_ids") is None
                else [int(value) for value in data.get("candidate_class_ids", [])]
            ),
            inference_mode=str(
                sahi.get("mode", data.get("inference_mode", "NORMAL")) or "NORMAL"
            ),
            sahi_enabled=_as_bool(sahi.get("enabled", data.get("sahi_enabled", False))),
            sahi_tile_count=int(sahi.get("tile_count", data.get("sahi_tile_count", 0)) or 0),
            sahi_tile_index=(
                None
                if sahi.get("tile_index", data.get("sahi_tile_index")) is None
                else int(sahi.get("tile_index", data.get("sahi_tile_index")))
            ),
            siglip_top2_class_id=(
                None
                if siglip.get("top2_class_id", data.get("siglip_top2_class_id")) is None
                else int(siglip.get("top2_class_id", data.get("siglip_top2_class_id")))
            ),
            siglip_top2_class_name=(
                None
                if siglip.get("top2_class", data.get("siglip_top2_class_name")) is None
                else str(siglip.get("top2_class", data.get("siglip_top2_class_name")))
            ),
            siglip_top2_score=(
                None
                if siglip.get("top2_score", data.get("siglip_top2_score")) is None
                else float(siglip.get("top2_score", data.get("siglip_top2_score")))
            ),
            siglip_margin=(
                None
                if siglip.get("margin", data.get("siglip_margin")) is None
                else float(siglip.get("margin", data.get("siglip_margin")))
            ),
            vlm_enabled=_as_bool(vlm.get("enabled", data.get("vlm_enabled", False))),
            vlm_triggered=_as_bool(vlm.get("triggered", data.get("vlm_triggered", False))),
            vlm_model=(
                None
                if vlm.get("model", data.get("vlm_model")) is None
                else str(vlm.get("model", data.get("vlm_model")))
            ),
            vlm_target_class=(
                None
                if vlm.get("target_class", data.get("vlm_target_class")) is None
                else str(vlm.get("target_class", data.get("vlm_target_class")))
            ),
            vlm_features=(
                dict(vlm.get("features", {}))
                if isinstance(vlm.get("features", {}), dict)
                else {}
            ),
            vlm_final_result=(
                None
                if vlm.get("final_result", data.get("vlm_final_result")) is None
                else str(vlm.get("final_result", data.get("vlm_final_result"))).upper()
            ),
            vlm_confidence=(
                None
                if vlm.get("self_reported_confidence", data.get("vlm_confidence")) is None
                else float(vlm.get("self_reported_confidence", data.get("vlm_confidence")))
            ),
            vlm_parse_error=(
                None
                if vlm.get("parse_error", data.get("vlm_parse_error")) is None
                else str(vlm.get("parse_error", data.get("vlm_parse_error")))
            ),
            decision_state=decision.get("state", data.get("decision_state")),
            decision_reason=decision.get("reason", data.get("decision_reason")),
        )


@dataclass(slots=True)
class ImageAnnotation:
    status: AnnotationStatus = AnnotationStatus.UNLABELED
    objects: list[BoundingBox] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": self.status.value,
            "objects": [box.to_dict() for box in self.objects],
        }
        if self.verification:
            data["verification"] = self.verification
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageAnnotation":
        status_value = str(data.get("status", AnnotationStatus.UNLABELED.value))
        try:
            status = AnnotationStatus(status_value)
        except ValueError:
            status = AnnotationStatus.UNLABELED
        return cls(
            status=status,
            objects=[BoundingBox.from_dict(item) for item in data.get("objects", [])],
            verification=(
                dict(data.get("verification", {}))
                if isinstance(data.get("verification", {}), dict)
                else {}
            ),
        )
