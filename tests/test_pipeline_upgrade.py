from __future__ import annotations

from pathlib import Path

from PIL import Image

from yolo_world_annotator.core.annotation import BoundingBox
from yolo_world_annotator.core.class_profiles import ensure_class_profiles, load_class_profiles
from yolo_world_annotator.core.decision_engine import DecisionEngine, DecisionState
from yolo_world_annotator.inference.merge_utils import bbox_iou, merge_detections
from yolo_world_annotator.inference.sahi_runner import (
    SAHIConfig,
    SAHIInferenceRunner,
    generate_tiles,
)
from yolo_world_annotator.models.vlm_verifier import (
    MATCH,
    UNCERTAIN,
    VLMVerifier,
    parse_vlm_response,
)


def test_sahi_tiles_cover_edges_and_merge_same_class() -> None:
    tiles = generate_tiles(100, 80, SAHIConfig(slice_width=60, slice_height=50))
    assert tiles[-1].x2 == 100 and tiles[-1].y2 == 80
    boxes = [
        BoundingBox(0, "x", 0, 0, 20, 20, 0.8, "YOLO-World"),
        BoundingBox(0, "x", 1, 1, 21, 21, 0.7, "YOLO-World"),
    ]
    assert bbox_iou(boxes[0], boxes[1]) > 0.5
    assert len(merge_detections(boxes, postprocess_type="NMS", match_threshold=0.5)) == 1


def test_sahi_runner_maps_tile_coordinates() -> None:
    class Detector:
        def predict(self, image, **kwargs):
            return [BoundingBox(0, "x", 5, 5, 15, 15, 0.8, "YOLO-World")]

    result = SAHIInferenceRunner(Detector()).run(
        Image.new("RGB", (100, 80)),
        config=SAHIConfig(slice_width=60, slice_height=50, max_tiles=1),
    )
    assert result.tile_count == 1
    assert result[0].sahi_enabled is True
    assert result[0].inference_mode == "SAHI"


def test_vlm_parser_and_lazy_callback() -> None:
    profile = {
        "class_name": "chip",
        "features": [{"name": "black", "required": True}],
        "required_features": ["black"],
    }
    result = parse_vlm_response(
        '{"target_class":"chip","features":{"black":"TRUE"},"final_result":"MATCH","self_reported_confidence":0.9}',
        profile=profile,
    )
    assert result.parsed and result.final_result == MATCH
    assert parse_vlm_response("not json").final_result == UNCERTAIN
    verifier = VLMVerifier(generator=lambda image, prompt: '{"target_class":"chip","features":{"black":"TRUE"},"final_result":"MATCH","self_reported_confidence":0.9}')
    assert verifier.verify(Image.new("RGB", (10, 10)), target_class="chip", profile=profile).is_match


def test_decision_engine_conflict_never_fakes_combined_score() -> None:
    box = BoundingBox(0, "a", 0, 0, 10, 10, 0.9, "YOLO-World")
    box.yolo_class_id = 0
    box.yolo_confidence = 0.9
    box.siglip_class_id = 1
    box.siglip_class_name = "b"
    box.siglip_score = 0.95
    box.agreement = False
    result = DecisionEngine().decide(box, classes=["a", "b"])
    assert result.state == DecisionState.MODEL_CONFLICT
    assert result.combined_score is None


def test_class_profiles_create_and_reconcile_ids(tmp_path: Path) -> None:
    path = tmp_path / "class_profiles.json"
    profiles = ensure_class_profiles(path, ["a", "b"])
    profiles["a"].always_vlm_verify = True
    profiles = profiles.sync_classes(["b", "a", "c"])
    assert profiles["b"].class_id == 0
    assert profiles["a"].class_id == 1
    assert profiles["c"].class_id == 2
    loaded = load_class_profiles(path, ["a", "b"])
    assert [item.class_id for item in loaded] == [0, 1]
