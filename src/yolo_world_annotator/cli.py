"""Command-line entry point and delayed Qt application startup."""

from __future__ import annotations

import argparse
import os
import re
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from yolo_world_annotator import __version__

_DEVICE_PATTERN = re.compile(r"(?:auto|cpu|cuda(?::\d+)?)\Z", re.IGNORECASE)


def _device_argument(value: str) -> str:
    normalized = value.strip().lower()
    if _DEVICE_PATTERN.fullmatch(normalized) is None:
        raise argparse.ArgumentTypeError(
            "device must be auto, cpu, cuda, or cuda:N (for example cuda:1)"
        )
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="yolo-world-annotator",
        description="Local YOLO-World desktop dataset annotator.",
    )
    parser.add_argument(
        "--device",
        type=_device_argument,
        default=os.environ.get("YOLO_WORLD_DEVICE", "auto"),
        help="inference device: auto (default), cpu, cuda, or cuda:N",
    )
    parser.add_argument("--version", action="store_true", help="show version and exit")
    return parser.parse_args(argv)


def _platform_config_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))


def app_log_dir() -> Path:
    """Return the platform-appropriate writable application log directory."""

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Logs"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "yolo-world-annotator" / "logs"


def configure_runtime_environment(device: str) -> None:
    os.environ.setdefault(
        "YOLO_CONFIG_DIR", str(_platform_config_dir() / "yolo-world-annotator")
    )
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    os.environ["YOLO_WORLD_DEVICE"] = device


def run_gui(device: str) -> int:
    configure_runtime_environment(device)

    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from yolo_world_annotator.app.annotator_window import AnnotatorWindow
    from yolo_world_annotator.utils.logger import configure_logging

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(
        """
        QWidget {
            font-family: "Microsoft YaHei UI", "Noto Sans CJK SC", sans-serif;
            font-size: 13px;
        }
        QGroupBox {
            border: 1px solid #59636f;
            border-radius: 8px;
            margin-top: 14px;
            padding: 14px 10px 10px 10px;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            color: #e6edf5;
        }
        QPushButton { min-height: 32px; padding: 4px 10px; border-radius: 6px; }
        QPushButton:hover { background: #3b536b; }
        QPushButton:checked { background: #315b78; }
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
            min-height: 30px;
            padding: 2px 7px;
            border-radius: 5px;
        }
        QTextEdit { padding: 6px; }
        QLabel#settingsTitle { font-size: 18px; font-weight: 700; color: #f2f6fb; }
        QLabel#helpText { color: #a9bacb; font-size: 12px; }
        QScrollArea#settingsScroll, QWidget#settingsContent { background: #202328; }
        QScrollBar:vertical { width: 12px; margin: 2px; }
        QStatusBar { padding-left: 8px; }
        """
    )
    app.setApplicationName("YOLO-World 数据集标注器")
    app.setOrganizationName("YOLOWorldAnnotator")
    configure_logging(app_log_dir())
    window = AnnotatorWindow()
    window.show()
    return app.exec()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.version:
        print(f"yolo-world-annotator {__version__}")
        return 0
    try:
        return run_gui(args.device)
    except Exception:
        details = traceback.format_exc()
        error_path = app_log_dir() / "startup_error.log"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(details, encoding="utf-8")
        print(
            f"Application startup failed. Details were written to {error_path}\n{details}",
            file=sys.stderr,
        )
        return 1


__all__ = ["app_log_dir", "configure_runtime_environment", "main", "parse_args", "run_gui"]
