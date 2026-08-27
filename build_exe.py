from __future__ import annotations

import shutil
import sys
from pathlib import Path

import PyInstaller.__main__


PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "dist" / "YOLOWorldAnnotator"
APP_EXE = APP_DIR / "YOLOWorldAnnotator.exe"


def main() -> int:
    PyInstaller.__main__.run(
        [
            "--noconfirm",
            "--clean",
            str(PROJECT_ROOT / "YOLOWorldAnnotator.spec"),
        ]
    )

    weight_target = APP_DIR / "models" / "weights"
    weight_target.mkdir(parents=True, exist_ok=True)
    for source in (PROJECT_ROOT / "models" / "weights").iterdir():
        if source.is_file() and source.suffix.lower() in {".pt", ".pth"}:
            shutil.copy2(source, weight_target / source.name)
    shutil.copy2(PROJECT_ROOT / "README.md", APP_DIR / "README.md")

    required = [
        APP_EXE,
        APP_DIR / "_internal" / "PySide6" / "shiboken6.abi3.dll",
        APP_DIR / "_internal" / "clip" / "bpe_simple_vocab_16e6.txt.gz",
        weight_target / "yolov8s-worldv2.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Build is incomplete; missing: " + ", ".join(missing))
    for name in ("icuuc.dll", "icudt73.dll"):
        if (APP_DIR / "_internal" / name).exists():
            raise RuntimeError(f"Conflicting Qt dependency was bundled: {name}")

    print(f"Build complete: {APP_EXE}")
    print("Keep the whole YOLOWorldAnnotator folder together.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
