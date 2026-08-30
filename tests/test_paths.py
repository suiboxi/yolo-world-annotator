from __future__ import annotations

from pathlib import Path


def test_weights_directory_can_be_overridden(monkeypatch, tmp_path: Path) -> None:
    from yolo_world_annotator.utils.paths import default_weights_dir

    target = tmp_path / "custom-weights"
    monkeypatch.setenv("YOLO_WORLD_WEIGHTS_DIR", str(target))

    assert default_weights_dir() == target.resolve()
