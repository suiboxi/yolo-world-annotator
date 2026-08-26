from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.yolo_world import YOLOWorldDetector


def test_detector_refuses_cpu_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="不会回退到 CPU"):
        YOLOWorldDetector(tmp_path / "model.pt")
