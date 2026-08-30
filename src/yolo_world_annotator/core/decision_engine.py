"""Unified, explainable decision policy for the staged annotation pipeline."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Sequence

from yolo_world_annotator.core.annotation import BoundingBox
from yolo_world_annotator.core.verification import (
    AUTO_ACCEPT,
    HUMAN_CONFIRMED,
    REJECT,
    REVIEW,
    SigLIPPrediction,
    classify_yolo_only,
)
from yolo_world_annotator.models.vlm_verifier import (
    MATCH,
    NOT_MATCH,
    VLMResult,
    VLMTriggerPolicy,
)


class DecisionState:
    YOLO_ONLY_ACCEPT = "YOLO_ONLY_ACCEPT"
    YOLO_SAHI_ACCEPT = "YOLO_SAHI_ACCEPT"
    YOLO_SIGLIP_ACCEPT = "YOLO_SIGLIP_ACCEPT"
    VLM_ACCEPT = "VLM_ACCEPT"
    MODEL_CONFLICT = "MODEL_CONFLICT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    VLM_UNCERTAIN = "VLM_UNCERTAIN"
    VLM_REJECT = "VLM_REJECT"
    HUMAN_ACCEPT = "HUMAN_ACCEPT"
    HUMAN_REJECT = "HUMAN_REJECT"
    HUMAN_MODIFIED = "HUMAN_MODIFIED"


@dataclass(slots=True)
class DecisionResult:
    status: str
    state: str
    class_id: int
    class_name: str
    combined_score: float | None = None
    vlm_triggered: bool = False
    trigger_reasons: list[str] | None = None
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == AUTO_ACCEPT

    @property
    def final_status(self) -> str:
        return self.status

    @property
    def combined_confidence(self) -> float | None:
        return self.combined_score

    @property
    def review_required(self) -> bool:
        return self.status != AUTO_ACCEPT

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "state": self.state,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "combined_score": self.combined_score,
            "vlm_triggered": self.vlm_triggered,
            "trigger_reasons": list(self.trigger_reasons or []),
            "reason": self.reason,
        }


class DecisionEngine:
    """Apply YOLO -> SigLIP -> VLM -> human precedence without model voting."""

    def __init__(
        self,
        *,
        yolo_weight: float = 0.65,
        siglip_weight: float = 0.35,
        auto_accept_threshold: float = 0.75,
        review_threshold: float = 0.50,
        trigger_policy: VLMTriggerPolicy | None = None,
    ) -> None:
        self.yolo_weight = max(0.0, float(yolo_weight))
        self.siglip_weight = max(0.0, float(siglip_weight))
        total = self.yolo_weight + self.siglip_weight
        if total <= 0:
            self.yolo_weight, self.siglip_weight = 0.65, 0.35
        else:
            self.yolo_weight /= total
            self.siglip_weight /= total
        self.auto_accept_threshold = min(1.0, max(0.0, float(auto_accept_threshold)))
        self.review_threshold = min(self.auto_accept_threshold, max(0.0, float(review_threshold)))
        self.trigger_policy = trigger_policy or VLMTriggerPolicy()

    @staticmethod
    def _score(value: Any) -> float:
        try:
            return min(1.0, max(0.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    def should_trigger_vlm(
        self,
        *,
        yolo_score: float | None,
        siglip_score: float | None = None,
        siglip_margin: float | None = None,
        agreement: bool | None = None,
        decision_status: str | None = None,
        always_vlm_verify: bool = False,
    ) -> tuple[bool, list[str]]:
        return self.trigger_policy.should_trigger(
            yolo_score=yolo_score,
            siglip_score=siglip_score,
            siglip_margin=siglip_margin,
            agreement=agreement,
            decision_status=decision_status,
            always_vlm_verify=always_vlm_verify,
        )

    def decide(
        self,
        box: BoundingBox | None = None,
        siglip_prediction: SigLIPPrediction | None = None,
        *,
        classes: Sequence[str] | None = None,
        vlm_result: VLMResult | None = None,
        profile: Any = None,
        enable_vlm: bool = False,
        always_vlm_verify: bool = False,
        human_state: str | None = None,
        sahi_enabled: bool | None = None,
        yolo_score: float | None = None,
        siglip_score: float | None = None,
        siglip_margin: float | None = None,
        agreement: bool | None = None,
        yolo_class_id: int | None = None,
        yolo_class_name: str | None = None,
        siglip_class_id: int | None = None,
        siglip_class_name: str | None = None,
    ) -> DecisionResult:
        """Return a decision and an explanation; never mutates *box*."""

        if box is not None:
            yolo_score = yolo_score if yolo_score is not None else (box.yolo_confidence if box.yolo_confidence is not None else box.confidence)
            yolo_class_id = yolo_class_id if yolo_class_id is not None else (box.yolo_class_id if box.yolo_class_id is not None else box.class_id)
            yolo_class_name = yolo_class_name or box.yolo_class_name or box.class_name
            siglip_score = siglip_score if siglip_score is not None else box.siglip_score
            siglip_margin = siglip_margin if siglip_margin is not None else box.siglip_margin
            agreement = agreement if agreement is not None else box.agreement
            if sahi_enabled is None:
                sahi_enabled = bool(box.sahi_enabled)
        yolo_id = int(yolo_class_id if yolo_class_id is not None else 0)
        yolo_name = str(yolo_class_name or (classes[yolo_id] if classes and 0 <= yolo_id < len(classes) else ""))
        if siglip_prediction is not None:
            siglip_score = float(siglip_prediction.score)
            siglip_class_id = int(siglip_prediction.class_id)
            siglip_class_name = str(siglip_prediction.class_name)
            siglip_margin = siglip_prediction.margin
            agreement = yolo_id == siglip_class_id
        siglip_present = siglip_score is not None or siglip_prediction is not None or agreement is not None
        siglip_id = int(siglip_class_id if siglip_class_id is not None else yolo_id)
        siglip_name = str(siglip_class_name or (classes[siglip_id] if classes and 0 <= siglip_id < len(classes) else yolo_name))
        yolo_value = self._score(yolo_score)
        siglip_value = self._score(siglip_score)

        if human_state:
            normalized_human = str(human_state).upper()
            if normalized_human in {"HUMAN_ACCEPT", "ACCEPT", HUMAN_CONFIRMED}:
                return DecisionResult(AUTO_ACCEPT, DecisionState.HUMAN_ACCEPT, yolo_id, yolo_name, reason="人工确认接受")
            if normalized_human in {"HUMAN_REJECT", "REJECT"}:
                return DecisionResult(REJECT, DecisionState.HUMAN_REJECT, yolo_id, yolo_name, reason="人工确认拒绝")
            if normalized_human in {"HUMAN_MODIFIED", "MODIFIED"}:
                return DecisionResult(AUTO_ACCEPT, DecisionState.HUMAN_MODIFIED, yolo_id, yolo_name, reason="人工修改拥有最高优先级")

        if not siglip_present:
            status = classify_yolo_only(box or BoundingBox(yolo_id, yolo_name, 0, 0, 1, 1, yolo_value, "YOLO-World"), auto_accept_threshold=self.auto_accept_threshold, review_threshold=self.review_threshold)
            state = DecisionState.YOLO_SAHI_ACCEPT if sahi_enabled and status == AUTO_ACCEPT else DecisionState.YOLO_ONLY_ACCEPT if status == AUTO_ACCEPT else DecisionState.LOW_CONFIDENCE
            reason = "YOLO-World 结果达到自动接受阈值" if status == AUTO_ACCEPT else "YOLO-World 置信度低于自动接受阈值"
        elif agreement:
            combined = self.yolo_weight * yolo_value + self.siglip_weight * siglip_value
            if yolo_value < self.review_threshold and siglip_value < self.review_threshold:
                status = REJECT
                state = DecisionState.LOW_CONFIDENCE
                reason = "YOLO 与 SigLIP 均低于审核阈值"
            elif combined >= self.auto_accept_threshold and yolo_value >= self.review_threshold and siglip_value >= self.review_threshold:
                status = AUTO_ACCEPT
                state = DecisionState.YOLO_SIGLIP_ACCEPT
                reason = "YOLO 与 SigLIP 类别一致且加权辅助分数达标"
            elif combined >= self.review_threshold:
                status = REVIEW
                state = DecisionState.LOW_CONFIDENCE
                reason = "类别一致但加权辅助分数仅进入审核区间"
            else:
                status = REJECT
                state = DecisionState.LOW_CONFIDENCE
                reason = "类别一致但加权辅助分数过低"
        else:
            combined = None
            status = REVIEW
            state = DecisionState.MODEL_CONFLICT
            reason = "YOLO 与 SigLIP 类别冲突；不计算伪融合分数"

        trigger, reasons = self.should_trigger_vlm(
            yolo_score=yolo_value,
            siglip_score=(siglip_value if siglip_present else None),
            siglip_margin=siglip_margin,
            agreement=agreement if siglip_present else None,
            decision_status=status,
            always_vlm_verify=always_vlm_verify,
        )
        if enable_vlm and trigger:
            if vlm_result is None:
                status = REVIEW
                state = DecisionState.VLM_UNCERTAIN
                reason = "困难样本需要 VLM 验证，但尚未得到结构化结果"
            elif not vlm_result.parsed:
                status = REVIEW
                state = DecisionState.VLM_UNCERTAIN
                reason = f"VLM JSON 解析失败：{vlm_result.parse_error or 'unknown'}"
            elif vlm_result.final_result == MATCH:
                status = AUTO_ACCEPT
                state = DecisionState.VLM_ACCEPT
                reason = "VLM MATCH 且结构化特征通过"
            elif vlm_result.final_result == NOT_MATCH:
                status = REJECT
                state = DecisionState.VLM_REJECT
                reason = "VLM NOT_MATCH 或 required feature 失败"
            else:
                status = REVIEW
                state = DecisionState.VLM_UNCERTAIN
                reason = "VLM 返回 UNCERTAIN，保留人工审核"
        return DecisionResult(
            status=status,
            state=state,
            class_id=yolo_id,
            class_name=yolo_name,
            combined_score=(combined if 'combined' in locals() else None),
            vlm_triggered=bool(enable_vlm and trigger),
            trigger_reasons=reasons,
            reason=reason,
        )

    def apply(self, box: BoundingBox, result: DecisionResult, *, vlm_result: VLMResult | None = None) -> BoundingBox:
        enriched = deepcopy(box)
        enriched.class_id = int(result.class_id)
        enriched.class_name = str(result.class_name)
        enriched.confidence = result.combined_score if result.combined_score is not None else box.confidence
        enriched.combined_confidence = result.combined_score
        enriched.fusion_status = result.status
        enriched.decision_state = result.state
        enriched.decision_reason = result.reason
        enriched.review_required = result.status != AUTO_ACCEPT
        if result.vlm_triggered:
            enriched.vlm_triggered = True
        if vlm_result is not None:
            enriched.vlm_enabled = True
            enriched.vlm_triggered = True
            enriched.vlm_model = vlm_result.model
            enriched.vlm_target_class = vlm_result.target_class
            enriched.vlm_features = dict(vlm_result.features)
            enriched.vlm_final_result = vlm_result.final_result
            enriched.vlm_confidence = vlm_result.self_reported_confidence
            enriched.vlm_parse_error = vlm_result.parse_error
        return enriched

    apply_decision = apply


def decide_box(box: BoundingBox, **kwargs) -> DecisionResult:
    return DecisionEngine(**{key: kwargs.pop(key) for key in ("yolo_weight", "siglip_weight", "auto_accept_threshold", "review_threshold") if key in kwargs}).decide(box, **kwargs)


__all__ = ["DecisionState", "DecisionResult", "DecisionEngine", "decide_box"]
