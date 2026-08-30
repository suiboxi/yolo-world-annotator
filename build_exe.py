"""Build and validate the Windows PyInstaller distribution."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
APP_DIR = PROJECT_ROOT / "dist" / "YOLOWorldAnnotator"
APP_EXE = APP_DIR / "YOLOWorldAnnotator.exe"


def check_sources() -> None:
    required = [
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "LICENSE",
        PROJECT_ROOT / "YOLOWorldAnnotator.spec",
        PROJECT_ROOT / "src" / "yolo_world_annotator" / "__main__.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Build sources are incomplete; missing: " + ", ".join(missing))


def build() -> None:
    if sys.platform != "win32":
        raise RuntimeError("The bundled desktop artifact currently supports Windows only.")
    try:
        import PyInstaller.__main__
    except ImportError as exc:
        raise RuntimeError("Install build dependencies with: pip install -e .[build]") from exc

    PyInstaller.__main__.run(
        ["--noconfirm", "--clean", str(PROJECT_ROOT / "YOLOWorldAnnotator.spec")]
    )
    shutil.copy2(PROJECT_ROOT / "README.md", APP_DIR / "README.md")
    shutil.copy2(PROJECT_ROOT / "LICENSE", APP_DIR / "LICENSE")

    required = [APP_EXE, APP_DIR / "README.md", APP_DIR / "LICENSE"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Build is incomplete; missing: " + ", ".join(missing))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true", help="validate build inputs without PyInstaller"
    )
    args = parser.parse_args(argv)
    check_sources()
    if args.check_only:
        print("Build inputs are complete.")
        return 0
    build()
    print(f"Build complete: {APP_EXE}")
    print("Keep the whole YOLOWorldAnnotator folder together.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
