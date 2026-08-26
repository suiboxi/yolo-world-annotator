from __future__ import annotations

from pathlib import Path

from app import annotator_window
from app.annotator_window import InferenceEngine
from core.annotation import BoundingBox


class _FakeDetector:
    def __init__(self, _model_path: Path) -> None:
        self.last_image_size = (200, 100)

    def set_classes(self, _classes: list[str]) -> None:
        return None

    def predict(self, _path: Path, **_kwargs) -> list[BoundingBox]:
        return [BoundingBox(0, "person", 10, 10, 80, 70, 0.9, "YOLO-World")]


def test_large_batch_writes_labels_before_chunked_ui_updates(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(annotator_window, "YOLOWorldDetector", _FakeDetector)
    paths = [tmp_path / f"sample_{index:04d}.jpg" for index in range(205)]
    engine = InferenceEngine()
    chunk_sizes: list[int] = []
    result_names: list[str] = []
    finished: list[bool] = []
    def capture(batch) -> None:
        chunk_sizes.append(len(batch))
        result_names.extend(box.class_name for result in batch for box in result[1])

    engine.results_ready.connect(capture)
    engine.finished.connect(finished.append)

    engine.run_job(
        {
            "model_path": tmp_path / "fake.pt",
            "classes": ["raspberry"],
            "prompts": ["strawberry"],
            "paths": paths,
            "confidence": 0.25,
            "iou": 0.45,
            "imgsz": 640,
        }
    )

    assert finished == [False]
    assert chunk_sizes == [20] * 10 + [5]
    assert result_names == ["raspberry"] * len(paths)
    labels = sorted(tmp_path.glob("*.txt"))
    assert len(labels) == len(paths)
    assert all(path.read_text(encoding="utf-8").startswith("0 ") for path in labels)
