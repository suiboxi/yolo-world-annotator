"""Transparent YOLO-World/SigLIP2 confidence fusion helpers.

The values produced by YOLO-World and SigLIP2 are useful ranking scores, but
they are not calibrated probabilities.  This module deliberately calls the
fused value ``combined_confidence`` and keeps the individual model scores in
the annotation metadata so that a user can inspect why a box was accepted or
sent for review.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from core.annotation import BoundingBox


AUTO_ACCEPT = "AUTO_ACCEPT"
REVIEW = "REVIEW"
REJECT = "REJECT"
HUMAN_CONFIRMED = "HUMAN_CONFIRMED"


@dataclass(slots=True)
class SigLIPPrediction:
    """One crop-level SigLIP prediction.

    ``score`` is a normalized similarity/ranking score in the range 0..1. It
    should not be interpreted as a mathematically calibrated probability.
    """

    class_id: int
    class_name: str
    score: float
    scores: dict[int, float] | None = None
    top2_class_id: int | None = None
    top2_class_name: str | None = None
    top2_score: float | None = None
    margin: float | None = None

    @property
    def top2_id(self) -> int | None:
        return self.top2_class_id

    @property
    def top2_name(self) -> str | None:
        return self.top2_class_name


def _clamp_score(value: float | None) -> float:
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, value))


def _normalise_weights(yolo_weight: float, siglip_weight: float) -> tuple[float, float]:
    yolo = max(0.0, float(yolo_weight))
    siglip = max(0.0, float(siglip_weight))
    total = yolo + siglip
    if total <= 0:
        return 0.65, 0.35
    return yolo / total, siglip / total


def fuse_box(
    box: BoundingBox,
    prediction: SigLIPPrediction,
    *,
    classes: Sequence[str],
    yolo_weight: float = 0.65,
    siglip_weight: float = 0.35,
    auto_accept_threshold: float = 0.75,
    review_threshold: float = 0.50,
    per_class_thresholds: Mapping[str, float] | None = None,
) -> BoundingBox:
    """Return a copy of *box* enriched with SigLIP and fusion metadata.

    In a conflict the YOLO class remains the provisional class and the raw
    SigLIP class is retained as metadata.  No pseudo-fusion score is produced;
    the status is always ``REVIEW`` so the decision engine can escalate to VLM
    or a human.
    """

    yolo_score = _clamp_score(box.yolo_confidence if box.yolo_confidence is not None else box.confidence)
    siglip_score = _clamp_score(prediction.score)
    yolo_id = box.yolo_class_id if box.yolo_class_id is not None else box.class_id
    yolo_name = (
        str(classes[yolo_id])
        if 0 <= int(yolo_id) < len(classes)
        else (box.yolo_class_name or box.class_name)
    )
    siglip_id = int(prediction.class_id)
    siglip_name = (
        str(classes[siglip_id])
        if 0 <= siglip_id < len(classes)
        else str(prediction.class_name)
    )
    agreement = yolo_id == siglip_id
    yolo_w, siglip_w = _normalise_weights(yolo_weight, siglip_weight)

    # The combined score is intentionally a transparent weighted score, not a
    # probability calibration.  It is only meaningful when both models agree;
    # a conflict is a routing signal for VLM/human review, never a vote.
    combined = yolo_w * yolo_score + siglip_w * siglip_score if agreement else None
    if agreement:
        final_id, final_name = yolo_id, yolo_name
    else:
        # Keep YOLO's stable class id as the provisional label.  The conflict
        # and both raw predictions remain in metadata for the next stage.
        final_id, final_name = yolo_id, yolo_name

    auto_threshold = min(1.0, max(0.0, float(auto_accept_threshold)))
    review = min(auto_threshold, max(0.0, float(review_threshold)))
    if per_class_thresholds:
        try:
            review = min(
                auto_threshold,
                max(0.0, float(per_class_thresholds.get(yolo_name, review))),
            )
        except (TypeError, ValueError):
            pass
    if not agreement:
        status = REVIEW
    elif yolo_score < review and siglip_score < review:
        status = REJECT
    elif agreement and combined >= auto_threshold and yolo_score >= review and siglip_score >= review:
        status = AUTO_ACCEPT
    elif combined >= review:
        status = REVIEW
    else:
        status = REJECT

    # The detector has already clamped the coordinates to the image.  Copy the
    # object directly so fusion never changes geometry as a side effect.
    enriched = deepcopy(box)
    enriched.class_id = int(final_id)
    enriched.class_name = str(final_name)
    enriched.confidence = combined
    enriched.source = "YOLO-World+SigLIP2"
    enriched.yolo_class_id = int(yolo_id)
    enriched.yolo_class_name = str(yolo_name)
    enriched.yolo_confidence = yolo_score
    enriched.siglip_enabled = True
    enriched.siglip_class_id = siglip_id
    enriched.siglip_class_name = siglip_name
    enriched.siglip_score = siglip_score
    enriched.siglip_top2_class_id = prediction.top2_class_id
    enriched.siglip_top2_class_name = prediction.top2_class_name
    enriched.siglip_top2_score = (
        None if prediction.top2_score is None else _clamp_score(prediction.top2_score)
    )
    enriched.siglip_margin = (
        None if prediction.margin is None else max(0.0, float(prediction.margin))
    )
    enriched.agreement = bool(agreement)
    enriched.combined_confidence = combined
    enriched.fusion_status = status
    enriched.review_required = status != AUTO_ACCEPT
    enriched.review_confirmed = False
    return enriched


def classify_yolo_only(
    box: BoundingBox,
    *,
    auto_accept_threshold: float = 0.75,
    review_threshold: float = 0.50,
) -> str:
    """Classify a YOLO-only box for filtering/statistics without changing it."""

    score = _clamp_score(box.confidence)
    auto_threshold = min(1.0, max(0.0, float(auto_accept_threshold)))
    review = min(auto_threshold, max(0.0, float(review_threshold)))
    if score < review:
        return REJECT
    if score >= auto_threshold:
        return AUTO_ACCEPT
    return REVIEW


def image_matches_filter(annotation, filter_key: str) -> bool:
    """Return whether an ImageAnnotation belongs to a UI filter bucket."""

    if filter_key in ("ALL", ""):
        return True
    if filter_key == "VERIFIED":
        return str(annotation.status) == "VERIFIED"
    objects: Iterable[BoundingBox] = annotation.objects
    if filter_key == "AUTO_ACCEPT":
        return bool(annotation.objects) and all(
            box.fusion_status == AUTO_ACCEPT
            or (
                box.fusion_status is None
                and box.confidence is not None
                and classify_yolo_only(box) == AUTO_ACCEPT
            )
            for box in objects
        )
    if filter_key == "REVIEW":
        return any(
            box.review_required
            or box.fusion_status == REVIEW
            or (
                box.fusion_status is None
                and box.confidence is not None
                and classify_yolo_only(box) == REVIEW
            )
            for box in objects
        )
    if filter_key == "LOW_CONFIDENCE":
        return any(
            box.fusion_status in (REVIEW, REJECT)
            or (
                box.fusion_status is None
                and box.confidence is not None
                and classify_yolo_only(box) in (REVIEW, REJECT)
            )
            or (box.confidence is not None and box.confidence < 0.5)
            for box in objects
        )
    if filter_key == "MODEL_CONFLICT":
        return any(box.agreement is False for box in objects)
    if filter_key == "VLM_UNCERTAIN":
        return any(
            box.vlm_final_result == "UNCERTAIN"
            or bool(box.vlm_parse_error)
            or (box.vlm_triggered and not box.vlm_final_result)
            for box in objects
        )
    if filter_key in {"HUMAN_REVIEWED", "HUMAN_CONFIRMED"}:
        return str(annotation.status) == "VERIFIED" or any(
            box.review_confirmed or box.human_modified for box in objects
        )
    return True
