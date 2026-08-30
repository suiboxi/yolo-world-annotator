from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from yolo_world_annotator.core.annotation import AnnotationStatus, BoundingBox, ImageAnnotation
from yolo_world_annotator.core.dataset import DatasetProject
from yolo_world_annotator.core.evaluation import evaluate_ab
from yolo_world_annotator.core.hard_samples import (
    append_hard_sample,
    load_hard_samples,
    record_auto_issues,
)
from yolo_world_annotator.core.verification import AUTO_ACCEPT, REVIEW, SigLIPPrediction, fuse_box
from yolo_world_annotator.models.siglip_verifier import SigLIPVerifier


def test_siglip_crop_padding_is_bounded() -> None:
    image = Image.new("RGB", (100, 80), "white")
    crop = SigLIPVerifier.crop_image(image, BoundingBox(0, "x", 0, 0, 20, 20), 0.10)
    assert crop.size == (22, 22)
    crop = SigLIPVerifier.crop_image(image, BoundingBox(0, "x", 90, 70, 100, 80), 0.50)
    assert 10 <= crop.size[0] <= 100
    assert 10 <= crop.size[1] <= 80


def test_siglip_verifier_batch_uses_cached_text_embeddings() -> None:
    class FakeProcessor:
        def __call__(self, *, text=None, images=None, **kwargs):
            count = len(text) if text is not None else len(images)
            if text is not None:
                return {
                    "input_ids": torch.ones((count, 3), dtype=torch.long),
                    "attention_mask": torch.ones((count, 3), dtype=torch.long),
                }
            return {"pixel_values": torch.ones((count, 3, 2, 2), dtype=torch.float32)}

    class FakeModel:
        def get_text_features(self, input_ids, attention_mask):
            count = input_ids.shape[0]
            return torch.eye(2)[:count]

        def get_image_features(self, pixel_values):
            return torch.tensor([[1.0, 0.0]] * pixel_values.shape[0])

    verifier = SigLIPVerifier(batch_size=2)
    verifier.processor = FakeProcessor()
    verifier.model = FakeModel()
    verifier.device = torch.device("cpu")
    verifier.dtype = torch.float32
    classes = ["person", "car"]
    first = verifier.encode_classes(classes)
    second = verifier.encode_classes(classes)
    assert first.data_ptr() == second.data_ptr()
    predictions = verifier.verify_batch(
        Image.new("RGB", (40, 40), "white"),
        [BoundingBox(0, "person", 0, 0, 20, 20)],
        classes,
    )
    assert predictions[0].class_id == 0


def test_fusion_keeps_class_ids_stable_and_marks_conflict() -> None:
    box = BoundingBox(0, "capacitor", 1, 2, 20, 30, 0.53, "YOLO-World")
    fused = fuse_box(
        box,
        SigLIPPrediction(1, "resistor", 0.82),
        classes=["capacitor", "resistor"],
    )
    assert fused.class_id == 0  # default weights keep this modest YOLO lead
    assert fused.yolo_class_id == 0
    assert fused.siglip_class_id == 1
    assert fused.agreement is False
    assert fused.fusion_status == REVIEW
    assert fused.review_required is True


def test_fusion_accepts_agreement() -> None:
    box = BoundingBox(0, "person", 1, 2, 20, 30, 0.91, "YOLO-World")
    fused = fuse_box(
        box,
        SigLIPPrediction(0, "person", 0.90),
        classes=["person"],
    )
    assert fused.fusion_status == AUTO_ACCEPT
    assert fused.review_required is False


def test_annotation_metadata_round_trip_and_hard_sample(tmp_path: Path) -> None:
    box = BoundingBox(0, "person", 1, 2, 20, 30, 0.8, "YOLO-World")
    box.siglip_enabled = True
    box.siglip_class_id = 0
    box.siglip_class_name = "person"
    box.siglip_score = 0.84
    box.agreement = True
    box.combined_confidence = 0.81
    box.fusion_status = AUTO_ACCEPT
    box.review_required = False
    annotation = ImageAnnotation(AnnotationStatus.AUTO_LABELED, [box])
    restored = ImageAnnotation.from_dict(annotation.to_dict())
    assert restored.objects[0].siglip_score == 0.84
    assert restored.objects[0].fusion_status == AUTO_ACCEPT

    path = tmp_path / "hard_samples.json"
    append_hard_sample(path, image=tmp_path / "a.jpg", box=box, error_type="LOW_CONFIDENCE")
    assert load_hard_samples(path)[0]["error_type"] == "LOW_CONFIDENCE"

    fused = fuse_box(
        BoundingBox(0, "person", 1, 2, 20, 30, 0.53, "YOLO-World"),
        SigLIPPrediction(1, "car", 0.82),
        classes=["person", "car"],
    )
    record_auto_issues(path, tmp_path / "a.jpg", [fused])
    assert load_hard_samples(path)[-1]["yolo_prediction"]["class_id"] == 0


def test_corrupt_annotation_entry_is_preserved_on_unrelated_save(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project = DatasetProject(project_root)
    project.annotations_path.write_text(
        '{"broken.jpg": {"objects": [{"class_id": "bad"}]}}\n', encoding="utf-8"
    )
    reopened = DatasetProject(project_root)
    reopened.save_metadata()
    text = reopened.annotations_path.read_text(encoding="utf-8")
    assert "broken.jpg" in text


def test_main_window_review_filter_keeps_path_index_mapping(qapp, tmp_path: Path) -> None:
    project = DatasetProject(tmp_path / "project")
    project.config["classes"] = ["capacitor", "resistor"]
    first = project.images_dir / "review.png"
    second = project.images_dir / "verified.png"
    Image.new("RGB", (40, 40), "white").save(first)
    Image.new("RGB", (40, 40), "black").save(second)
    review_box = fuse_box(
        BoundingBox(0, "capacitor", 2, 2, 20, 20, 0.53, "YOLO-World"),
        SigLIPPrediction(1, "resistor", 0.82),
        classes=project.classes,
    )
    project.save_annotation(
        first,
        ImageAnnotation(AnnotationStatus.AUTO_LABELED, [review_box]),
        (40, 40),
    )
    project.save_annotation(
        second,
        ImageAnnotation(
            AnnotationStatus.VERIFIED,
            [BoundingBox(0, "capacitor", 2, 2, 20, 20)],
        ),
        (40, 40),
    )

    from yolo_world_annotator.app.main_window import MainWindow

    window = MainWindow()
    try:
        window.open_project(project.root)
        window.filter_combo.setCurrentIndex(window.filter_combo.findData("REVIEW"))
        assert window.image_list.count() == 1
        assert window.image_list.path_at(0) == first.resolve()
        assert window.current_index == 0
    finally:
        window.close()


def test_ab_evaluation_reports_both_variants(tmp_path: Path) -> None:
    project = DatasetProject(tmp_path / "eval")
    project.config["classes"] = ["person", "car"]
    image_path = project.images_dir / "sample.png"
    Image.new("RGB", (40, 40), "white").save(image_path)
    box = fuse_box(
        BoundingBox(0, "person", 2, 2, 20, 20, 0.9, "YOLO-World"),
        SigLIPPrediction(0, "person", 0.9),
        classes=project.classes,
    )
    project.save_annotation(
        image_path,
        ImageAnnotation(AnnotationStatus.VERIFIED, [box]),
        (40, 40),
    )
    result = evaluate_ab(
        project,
        {"sample.png": [{"class_id": 0, "bbox": [2, 2, 20, 20]}]},
    )
    assert result["yolo_only"]["f1"] == 1.0
    assert result["yolo_siglip2"]["f1"] == 1.0


def test_visible_rejected_box_is_also_written_to_yolo_label(tmp_path: Path) -> None:
    project = DatasetProject(tmp_path / "reject")
    project.config["classes"] = ["person"]
    image_path = project.images_dir / "reject.png"
    Image.new("RGB", (40, 40), "white").save(image_path)
    box = BoundingBox(0, "person", 2, 2, 20, 20, 0.1, "YOLO-World")
    box.fusion_status = "REJECT"
    box.review_required = True
    project.save_annotation(
        image_path,
        ImageAnnotation(AnnotationStatus.AUTO_LABELED, [box]),
        (40, 40),
    )
    label_lines = project.label_path(image_path).read_text(encoding="utf-8").splitlines()
    assert len(label_lines) == 1
    assert label_lines[0].startswith("0 ")
    reopened = DatasetProject(project.root)
    assert reopened.get_annotation(image_path, (40, 40)).objects[0].fusion_status == "REJECT"
