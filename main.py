from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Bind Ultralytics to the isolated YOLO26 configuration even when the environment's
# python.exe is launched directly without `conda activate`.
_appdata = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
# Ultralytics appends its own `Ultralytics` subdirectory to YOLO_CONFIG_DIR.
os.environ["YOLO_CONFIG_DIR"] = str(_appdata / "YOLO26")
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")


def app_log_dir() -> Path:
    """Return a writable log directory for source and frozen builds."""
    local_appdata = Path(
        os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")
    )
    return local_appdata / "YOLOWorldAnnotator" / "logs"


def main() -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from app.annotator_window import AnnotatorWindow
    from utils.logger import configure_logging

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(
        """
        QWidget {
            font-family: "Microsoft YaHei UI", "Microsoft YaHei", sans-serif;
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
        QPushButton {
            min-height: 32px;
            padding: 4px 10px;
            border-radius: 6px;
        }
        QPushButton:hover { background: #3b536b; }
        QPushButton:checked { background: #315b78; }
        QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit {
            min-height: 30px;
            padding: 2px 7px;
            border-radius: 5px;
        }
        QTextEdit { padding: 6px; }
        QLabel#settingsTitle {
            font-size: 18px;
            font-weight: 700;
            color: #f2f6fb;
            padding: 2px 0 0 2px;
        }
        QLabel#helpText {
            color: #a9bacb;
            font-size: 12px;
            line-height: 1.35;
        }
        QScrollArea#settingsScroll, QWidget#settingsContent {
            background: #202328;
        }
        QScrollBar:vertical { width: 12px; margin: 2px; }
        QStatusBar { padding-left: 8px; }
        """
    )
    app.setApplicationName("YOLO-World 数据集标注器")
    app.setOrganizationName("LocalYOLOTools")
    configure_logging(app_log_dir())
    window = AnnotatorWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        details = traceback.format_exc()
        error_path = app_log_dir() / "startup_error.log"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(details, encoding="utf-8")
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                0,
                f"程序启动失败，详细信息已写入：\n{error_path}\n\n{details[-1200:]}",
                "YOLO-World GPU 标注器",
                0x10,
            )
        except Exception:
            pass
        raise
