from __future__ import annotations

import logging
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QThread, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from yolo_world_annotator.app.canvas import AnnotationCanvas, SegmentationPreview
from yolo_world_annotator.app.image_list import ImageListWidget
from yolo_world_annotator.app.inference_worker import InferenceWorker
from yolo_world_annotator.app.settings_panel import SettingsPanel
from yolo_world_annotator.app.statistics_dialog import StatisticsDialog
from yolo_world_annotator.core.annotation import AnnotationStatus, BoundingBox, ImageAnnotation
from yolo_world_annotator.core.benchmark import benchmark_project
from yolo_world_annotator.core.dataset import DatasetProject
from yolo_world_annotator.core.evaluation import evaluate_ab as evaluate_ab_project
from yolo_world_annotator.core.evaluation import load_ground_truth
from yolo_world_annotator.core.exporter import export_yolo_dataset
from yolo_world_annotator.core.hard_samples import append_hard_sample, record_auto_issues
from yolo_world_annotator.core.history import HistoryManager
from yolo_world_annotator.core.statistics import collect_statistics
from yolo_world_annotator.core.verification import HUMAN_CONFIRMED, image_matches_filter
from yolo_world_annotator.utils.config import as_bool, atomic_write_text
from yolo_world_annotator.utils.image_utils import discover_images

LOGGER = logging.getLogger(__name__)
APP_ROOT = Path(__file__).resolve().parents[1]


def _model_path_for_name(model_name: str) -> Path:
    """Resolve bundled weights in source and PyInstaller layouts."""

    roots = [APP_ROOT]
    if getattr(sys, "frozen", False):
        roots.insert(0, Path(sys.executable).resolve().parent)
        bundle_root = getattr(sys, "_MEIPASS", None)
        if bundle_root:
            roots.insert(0, Path(bundle_root))
    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        candidates.append(root / "models" / "weights" / model_name)
        candidates.append(root / "_internal" / "models" / "weights" / model_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Let YOLO's normal download/error path handle a missing optional model.
    return APP_ROOT / "models" / "weights" / model_name


class MainWindow(QMainWindow):
    load_model_signal = Signal(object)
    predict_single_signal = Signal(object)
    batch_signal = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO-World 数据集自动标注器")
        self.resize(1640, 980)
        self.setMinimumSize(1220, 760)
        self.project: DatasetProject | None = None
        self.images: list[Path] = []
        self.current_index = -1
        self.current_annotation = ImageAnnotation()
        self.history_by_image: dict[str, HistoryManager] = {}
        self.label_load_failed_paths: set[Path] = set()
        self._pending_single_root: Path | None = None
        self._pending_single_path: Path | None = None
        self._active_batch_root: Path | None = None
        self._pipeline_by_image: dict[str, dict] = {}
        self.model_loaded = False
        self.filter_key = "ALL"
        self._build_ui()
        self._build_actions()
        self._build_inference_thread()
        self._update_device_label()
        self.statusBar().showMessage("请选择或创建数据集根目录")

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        left = QWidget()
        left.setMinimumWidth(235)
        left_layout = QVBoxLayout(left)
        self.open_button = QPushButton("打开 / 创建数据集")
        self.open_button.clicked.connect(self.choose_project_root)
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("全部", "ALL")
        self.filter_combo.addItem("已自动接受", "AUTO_ACCEPT")
        self.filter_combo.addItem("待审核", "REVIEW")
        self.filter_combo.addItem("已人工确认", "VERIFIED")
        self.filter_combo.addItem("低置信度", "LOW_CONFIDENCE")
        self.filter_combo.addItem("模型冲突", "MODEL_CONFLICT")
        self.filter_combo.addItem("VLM 不确定", "VLM_UNCERTAIN")
        self.filter_combo.addItem("人工已审核", "HUMAN_REVIEWED")
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        self.image_list = ImageListWidget()
        self.image_list.image_selected.connect(self.select_image)
        nav = QHBoxLayout()
        self.previous_button = QPushButton("上一张")
        self.next_button = QPushButton("下一张")
        self.previous_button.clicked.connect(self.previous_image)
        self.next_button.clicked.connect(self.next_image)
        nav.addWidget(self.previous_button)
        nav.addWidget(self.next_button)
        left_layout.addWidget(self.open_button)
        left_layout.addWidget(self.filter_combo)
        left_layout.addWidget(self.image_list, 1)
        left_layout.addLayout(nav)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)

        # Keep the editable annotation view and the pre-YOLO tile/segmentation
        # view side by side as tabs.  The first tab is always the primary
        # review surface, while the second tab makes SAHI coverage inspectable.
        preview_tabs = QTabWidget()
        self.canvas = AnnotationCanvas()
        self.canvas.load_failed.connect(self._show_load_error)
        self.canvas.boxes_edited.connect(self._on_canvas_edited)
        self.canvas.new_box_requested.connect(self._on_new_box_requested)
        self.canvas.selection_changed.connect(self._on_canvas_selection_changed)
        self.canvas.change_class_requested.connect(self._choose_box_class)
        preview_tabs.addTab(self.canvas, "标注预览（可编辑框）")

        segmentation_tab = QWidget()
        segmentation_layout = QVBoxLayout(segmentation_tab)
        segmentation_layout.setContentsMargins(4, 4, 4, 4)
        self.segmentation_status = QLabel(
            "导入图片后，这里显示 YOLO 前的 SAHI 切片覆盖范围。"
        )
        self.segmentation_status.setWordWrap(True)
        self.segmentation_preview = SegmentationPreview()
        self.segmentation_preview.load_failed.connect(self._show_load_error)
        segmentation_layout.addWidget(self.segmentation_status)
        segmentation_layout.addWidget(self.segmentation_preview, 1)
        preview_tabs.addTab(segmentation_tab, "图像分割/切片预览")

        log_group = QGroupBox("Qwen 实时理解日志（人工检查）")
        log_layout = QVBoxLayout(log_group)
        log_toolbar = QHBoxLayout()
        self.qwen_log_clear_button = QPushButton("清空日志")
        self.qwen_log_clear_button.clicked.connect(self._clear_qwen_log)
        log_toolbar.addStretch(1)
        log_toolbar.addWidget(self.qwen_log_clear_button)
        self.qwen_log = QPlainTextEdit()
        self.qwen_log.setReadOnly(True)
        self.qwen_log.setPlaceholderText(
            "启用 Qwen 后，这里会实时显示每张图/每个框的提示、原始 JSON、解析结果和异常。"
        )
        self.qwen_log.setMaximumBlockCount(5000)
        self.qwen_log.setMinimumHeight(150)
        log_layout.addLayout(log_toolbar)
        log_layout.addWidget(self.qwen_log)

        preview_splitter = QSplitter(Qt.Orientation.Vertical)
        preview_splitter.addWidget(preview_tabs)
        preview_splitter.addWidget(log_group)
        preview_splitter.setStretchFactor(0, 4)
        preview_splitter.setStretchFactor(1, 1)
        preview_splitter.setSizes([700, 210])

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        center_layout.addWidget(preview_splitter, 1)
        center_layout.addWidget(self.progress_bar)

        self.settings = SettingsPanel()
        self.settings.setMinimumWidth(430)
        self.settings.load_model_requested.connect(self.load_model)
        self.settings.annotate_current_requested.connect(self.annotate_current)
        self.settings.annotate_all_requested.connect(self.annotate_all)
        self.settings.verify_requested.connect(self.verify_current)
        self.settings.save_requested.connect(self.save_current)
        self.settings.export_requested.connect(self.export_dataset)
        self.settings.statistics_requested.connect(self.show_statistics)
        self.settings.evaluation_requested.connect(self.evaluate_ab)
        self.settings.benchmark_requested.connect(self.run_benchmark)
        self.settings.save_classes_requested.connect(self.save_classes_file)
        self.settings.load_classes_requested.connect(self.load_classes_file)
        self.settings.selected_class_changed.connect(self._change_selected_box_class)
        self.settings.create_box_requested.connect(self.canvas.start_create_mode)
        self.settings.delete_box_requested.connect(self.canvas.delete_selected)
        self.settings.model_combo.currentTextChanged.connect(self._on_model_selection_changed)
        self.settings.pause_button.clicked.connect(self.pause_batch)
        self.settings.resume_button.clicked.connect(self.resume_batch)
        self.settings.cancel_button.clicked.connect(self.cancel_batch)

        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(self.settings)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([250, 900, 470])
        self.setCentralWidget(splitter)

    def _build_actions(self) -> None:
        open_action = QAction("打开数据集", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.choose_project_root)
        self.addAction(open_action)

        previous_action = QAction("上一张", self)
        previous_action.setShortcuts([QKeySequence(Qt.Key.Key_Left), QKeySequence("A")])
        previous_action.triggered.connect(self.previous_image)
        self.addAction(previous_action)

        next_action = QAction("下一张", self)
        next_action.setShortcuts([QKeySequence(Qt.Key.Key_Right), QKeySequence("D")])
        next_action.triggered.connect(self.next_image)
        self.addAction(next_action)

        save_action = QAction("保存", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.save_current)
        self.addAction(save_action)

        verify_action = QAction("确认当前图片", self)
        verify_action.setShortcut(QKeySequence(Qt.Key.Key_Space))
        verify_action.triggered.connect(self.verify_current)
        self.addAction(verify_action)

        delete_action = QAction("删除当前框", self)
        delete_action.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        delete_action.triggered.connect(self.canvas.delete_selected)
        self.addAction(delete_action)

        undo_action = QAction("撤销", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.undo)
        self.addAction(undo_action)

        create_action = QAction("创建标注框", self)
        create_action.setShortcut(QKeySequence("B"))
        create_action.triggered.connect(self.canvas.start_create_mode)
        self.addAction(create_action)

        cancel_action = QAction("取消当前操作", self)
        cancel_action.setShortcut(QKeySequence(Qt.Key.Key_Escape))
        cancel_action.triggered.connect(self.canvas.cancel_create_mode)
        self.addAction(cancel_action)

    def _build_inference_thread(self) -> None:
        self.inference_thread = QThread(self)
        self.inference_worker = InferenceWorker()
        self.inference_worker.moveToThread(self.inference_thread)
        self.load_model_signal.connect(self.inference_worker.load_model)
        self.predict_single_signal.connect(self.inference_worker.predict_single)
        self.batch_signal.connect(self.inference_worker.start_batch)
        self.inference_worker.model_loaded.connect(self._on_model_loaded)
        self.inference_worker.pipeline_ready.connect(self._on_pipeline_ready)
        self.inference_worker.qwen_log.connect(self._append_qwen_log)
        self.inference_worker.prediction_ready.connect(self._on_prediction_ready)
        self.inference_worker.batch_item_ready.connect(self._on_batch_item_ready)
        self.inference_worker.batch_item_failed.connect(self._on_batch_item_failed)
        self.inference_worker.batch_finished.connect(self._on_batch_finished)
        self.inference_worker.batch_paused.connect(self._on_batch_paused)
        self.inference_worker.failed.connect(self._on_inference_failed)
        self.inference_thread.start()

    def _update_device_label(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                self.settings.device_label.setText(
                    f"GPU：{torch.cuda.get_device_name(0)}\nCUDA：Available\nDevice：cuda:0"
                )
            else:
                self.settings.device_label.setText(
                    "GPU：不可用\nCUDA：Unavailable\nDevice：cpu（已自动回退）"
                )
        except Exception as exc:
            self.settings.device_label.setText(f"GPU 检测失败：{exc}\nDevice：cpu")

    def _append_qwen_log(self, message: str) -> None:
        """Append a worker-side Qwen/progress event without blocking inference."""

        if not message:
            return
        self.qwen_log.appendPlainText(str(message))
        scrollbar = self.qwen_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _clear_qwen_log(self) -> None:
        self.qwen_log.clear()

    def _on_pipeline_ready(self, image_path: str, metrics: object) -> None:
        """Cache per-image metrics so queued batch signals cannot overwrite them."""

        path = Path(image_path).resolve()
        value = deepcopy(metrics) if isinstance(metrics, dict) else {}
        self._pipeline_by_image[str(path)] = value
        if self.canvas.current_path is not None and self.canvas.current_path.resolve() == path:
            self._update_segmentation_preview(Path(image_path), value)

    def _pipeline_for_path(self, image_path: Path) -> dict:
        cached = self._pipeline_by_image.get(str(image_path.resolve()))
        if isinstance(cached, dict):
            return deepcopy(cached)
        if self.project is not None:
            annotation = self.project.annotations.get(image_path.name)
            if annotation is not None and isinstance(annotation.verification, dict):
                pipeline = annotation.verification.get("pipeline", {})
                if isinstance(pipeline, dict):
                    return deepcopy(pipeline)
        return {}

    def _update_segmentation_preview(self, image_path: Path, metrics: dict | None = None) -> None:
        if not image_path.exists():
            return
        try:
            if self.segmentation_preview.current_path != image_path.resolve():
                self.segmentation_preview.load_image(image_path)
            metrics = metrics or self._pipeline_for_path(image_path)
            tiles = metrics.get("tile_rects", []) if isinstance(metrics, dict) else []
            if not tiles and self.segmentation_preview.image_size != (0, 0):
                width, height = self.segmentation_preview.image_size
                tiles = [{"index": 0, "x1": 0, "y1": 0, "x2": width, "y2": height}]
            self.segmentation_preview.set_tiles(tiles)
            if isinstance(metrics, dict) and metrics.get("sahi_enabled"):
                self.segmentation_status.setText(
                    f"SAHI 切片预览：{metrics.get('tile_count', len(tiles))} 个切片；"
                    f"原始框 {metrics.get('raw_box_count', 0)}，合并后 {metrics.get('merged_box_count', 0)}。"
                )
            else:
                self.segmentation_status.setText(
                    "未启用 SAHI：当前预览显示整张图片范围；可在右侧推理流水线开启切片。"
                )
        except Exception:
            LOGGER.exception("更新分割预览失败：%s", image_path)

    def choose_project_root(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择数据集根目录或 images 文件夹")
        if selected:
            self.open_project(Path(selected))

    def open_project(self, root: Path) -> None:
        try:
            if root.name.casefold() == "images":
                root = root.parent
            self.project = DatasetProject(root)
            self.history_by_image.clear()
            self.label_load_failed_paths.clear()
            self._pipeline_by_image.clear()
            self._pending_single_root = None
            self._pending_single_path = None
            self._clear_qwen_log()
            self.settings.load_config(self.project.config)
            self.images = self.project.list_images()
            self.image_list.set_images(self.images, self.project.status_map())
            self._refresh_image_filter()
            self.current_index = 0 if self.images else -1
            self.project.save_metadata()
            if not self.images:
                self.canvas.clear_image()
                self.segmentation_preview.clear_image()
                self.statusBar().showMessage(
                    f"数据集已创建；请把图片放入：{self.project.images_dir}"
                )
            else:
                # set_images normally selects row zero, but explicitly loading
                # it here guarantees that existing boxes are immediately
                # visible even when the list already had the same row selected.
                self.select_image(0)
                self.statusBar().showMessage(
                    f"已打开数据集，共 {len(self.images)} 张图片：{self.project.root}"
                )
            # Opening a project must remain an offline, fast operation. Model
            # loading may download large weights, so it only starts after the
            # user explicitly clicks the load action.
            self.model_loaded = False
            self.settings.busy_label.setText("项目已打开；自动标注前请加载模型")
        except Exception as exc:
            LOGGER.exception("打开数据集失败")
            QMessageBox.critical(self, "打开数据集失败", str(exc))

    def open_image_folder(self, folder: Path) -> None:
        """Phase-1 compatible image browser; full workflow uses open_project()."""
        self.project = None
        self.history_by_image.clear()
        self.label_load_failed_paths.clear()
        self._pipeline_by_image.clear()
        self._clear_qwen_log()
        self.images = discover_images(folder)
        self.filter_key = "ALL"
        self.filter_combo.blockSignals(True)
        all_index = self.filter_combo.findData("ALL")
        if all_index >= 0:
            self.filter_combo.setCurrentIndex(all_index)
        self.filter_combo.blockSignals(False)
        self.image_list.set_images(self.images)
        self._refresh_image_filter()
        self.current_index = 0 if self.images else -1
        if not self.images:
            self.canvas.clear_image()
            self.segmentation_preview.clear_image()
            self.statusBar().showMessage(f"未找到支持的图片：{folder}")

    def _on_filter_changed(self) -> None:
        self.filter_key = str(self.filter_combo.currentData() or "ALL")
        self._refresh_image_filter()

    def _path_matches_filter(self, path: Path, status: str) -> bool:
        if self.filter_key == "ALL":
            return True
        if self.project is not None and path.name in self.project.annotations:
            annotation = self.project.annotations[path.name]
        else:
            try:
                annotation = ImageAnnotation(AnnotationStatus(status))
            except ValueError:
                annotation = ImageAnnotation()
        return image_matches_filter(annotation, self.filter_key)

    def _refresh_image_filter(self) -> None:
        self.image_list.set_filter(self._path_matches_filter)
        if self.image_list.count() == 0:
            self.canvas.clear_image()
            self.segmentation_preview.clear_image()
            self.current_index = -1
            self.settings.set_verification_detail("")
            return
        current_path = self.images[self.current_index] if 0 <= self.current_index < len(self.images) else None
        row = self.image_list.row_for_path(current_path) if current_path is not None else -1
        self.image_list.setCurrentRow(0 if row < 0 else row)

    def select_image(self, index: int) -> None:
        path = self.image_list.path_at(index)
        if path is None:
            return
        try:
            self.current_index = self.images.index(path)
        except ValueError:
            return
        if not self.canvas.load_image(path):
            self.segmentation_preview.clear_image()
            return
        try:
            if self.project is not None:
                self.current_annotation = self.project.get_annotation(path, self.canvas.image_size)
                self.label_load_failed_paths.discard(path.resolve())
            else:
                self.current_annotation = ImageAnnotation()
            self.canvas.set_boxes(
                self.current_annotation.objects, self.settings.review_spin.value()
            )
            self._update_segmentation_preview(path, self._pipeline_for_path(path))
            if path.name not in self.history_by_image:
                history = HistoryManager(20)
                history.reset(self.current_annotation.objects)
                self.history_by_image[path.name] = history
        except Exception as exc:
            LOGGER.exception("加载标签失败：%s", path)
            self.current_annotation = ImageAnnotation()
            self.canvas.set_boxes([])
            self.segmentation_preview.clear_image()
            self.label_load_failed_paths.add(path.resolve())
            history = HistoryManager(20)
            history.reset([])
            self.history_by_image[path.name] = history
            QMessageBox.warning(self, "标签读取失败", f"{path.name}\n{exc}\n\n原标签未被覆盖。")
        self.statusBar().showMessage(
            f"{self.current_index + 1} / {len(self.images)}  {path.name}  "
            f"{self.canvas.image_size[0]}×{self.canvas.image_size[1]}  "
            f"{self.current_annotation.status.value}"
        )
        self._on_canvas_selection_changed(self.canvas.selected_index)

    def previous_image(self) -> None:
        if self.image_list.count():
            self.image_list.setCurrentRow(max(0, self.image_list.currentRow() - 1))

    def next_image(self) -> None:
        if self.image_list.count():
            self.image_list.setCurrentRow(
                min(self.image_list.count() - 1, self.image_list.currentRow() + 1)
            )

    def _valid_classes(self) -> list[str] | None:
        classes = self.settings.classes()
        if not classes:
            QMessageBox.warning(self, "类别为空", "请在右侧一行输入一个检测类别。")
            return None
        if len(set(classes)) != len(classes):
            QMessageBox.warning(self, "类别重复", "类别名称不能重复，否则 class id 会产生歧义。")
            return None
        if self.project is not None:
            old_classes = self.project.classes
            labels_exist = any(self.project.labels_dir.glob("*.txt"))
            append_only = classes[: len(old_classes)] == old_classes
            if labels_exist and old_classes and not append_only:
                QMessageBox.warning(
                    self,
                    "禁止改变已有 class id",
                    "项目已有 YOLO 标签，不能删除、重命名或重排现有类别。\n"
                    "可以在列表末尾追加新类别，以保证历史 class id 不变。",
                )
                return None
        self.settings.set_classes(classes)
        return classes

    def save_classes_file(self) -> None:
        classes = self._valid_classes()
        if not classes:
            return
        default = str(self.project.classes_path if self.project else APP_ROOT / "classes.txt")
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存类别配置", default, "Text files (*.txt);;All files (*)"
        )
        if not filename:
            return
        try:
            atomic_write_text(Path(filename), "\n".join(classes) + "\n")
            self._save_panel_config()
            self.statusBar().showMessage(f"类别配置已保存：{filename}")
        except Exception as exc:
            QMessageBox.critical(self, "保存类别配置失败", str(exc))

    def load_classes_file(self) -> None:
        default = str(self.project.root if self.project else APP_ROOT)
        filename, _ = QFileDialog.getOpenFileName(
            self, "读取类别配置", default, "Text files (*.txt);;All files (*)"
        )
        if not filename:
            return
        try:
            classes = [
                line.strip()
                for line in Path(filename).read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
            previous = self.settings.classes()
            self.settings.set_classes(classes)
            if self._valid_classes() is None:
                self.settings.set_classes(previous)
                return
            self._save_panel_config()
            self.statusBar().showMessage(f"类别配置已读取：{filename}")
        except Exception as exc:
            QMessageBox.critical(self, "读取类别配置失败", str(exc))

    def load_model(self) -> None:
        classes = self._valid_classes()
        if classes is None:
            return
        model_name = self.settings.model_combo.currentText()
        model_path = _model_path_for_name(model_name)
        self.settings.set_busy(True, "正在加载模型和类别编码器…")
        self.model_loaded = False
        config = self.settings.config_values()
        self.load_model_signal.emit(
            {
                "model_path": str(model_path),
                "classes": classes,
                **self._verification_payload(config),
                "class_profiles": self._class_profiles_payload(classes),
            }
        )

    def _class_profiles_payload(self, classes: list[str]) -> dict[str, dict]:
        if self.project is None:
            return {name: {"class_name": name, "class_id": index} for index, name in enumerate(classes)}
        try:
            return {profile.class_name: profile.to_dict() for profile in self.project.class_profiles.sync_classes(classes)}
        except Exception:
            LOGGER.exception("读取 class_profiles 失败")
            return {name: {"class_name": name, "class_id": index} for index, name in enumerate(classes)}

    @staticmethod
    def _verification_payload(config: dict) -> dict:
        """Extract the optional verifier settings shared by single/batch calls."""

        return {
            "inference_mode": str(config.get("inference_mode", "NORMAL")),
            "sahi_enabled": as_bool(config.get("sahi_enabled", False)),
            "sahi_slice_width": int(config.get("sahi_slice_width", 1024)),
            "sahi_slice_height": int(config.get("sahi_slice_height", 1024)),
            "sahi_overlap_width_ratio": float(config.get("sahi_overlap_width_ratio", 0.20)),
            "sahi_overlap_height_ratio": float(config.get("sahi_overlap_height_ratio", 0.20)),
            "sahi_postprocess_type": str(config.get("sahi_postprocess_type", "NMS")),
            "sahi_postprocess_match_threshold": float(config.get("sahi_postprocess_match_threshold", 0.50)),
            "sahi_postprocess_match_metric": str(config.get("sahi_postprocess_match_metric", "IOU")),
            "sahi_max_tiles": int(config.get("sahi_max_tiles", 0)),
            "siglip_enabled": as_bool(config.get("siglip_enabled", False)),
            "siglip_model": str(
                config.get("siglip_model", "google/siglip2-base-patch16-224")
            ),
            "siglip_padding": float(config.get("siglip_padding", 0.10)),
            "yolo_weight": float(config.get("yolo_weight", 0.65)),
            "siglip_weight": float(config.get("siglip_weight", 0.35)),
            "auto_accept_threshold": float(config.get("auto_accept_threshold", 0.75)),
            "siglip_batch_size": int(config.get("siglip_batch_size", 4)),
            "siglip_precision": str(config.get("siglip_precision", "auto")),
            "candidate_top_k": int(config.get("candidate_top_k", 0)),
            "siglip_prompt_template": str(
                config.get("siglip_prompt_template", "a photo of a {}")
            ),
            "siglip_prompt_ensemble": as_bool(config.get("siglip_prompt_ensemble", False)),
            "review_threshold": float(config.get("review_threshold", 0.50)),
            "per_class_thresholds": dict(config.get("per_class_thresholds", {}))
            if isinstance(config.get("per_class_thresholds", {}), dict)
            else {},
            "vlm_enabled": as_bool(config.get("vlm_enabled", False)),
            "vlm_model": str(config.get("vlm_model", "Qwen/Qwen3-VL-8B-Instruct")),
            "vlm_lazy_load": as_bool(config.get("vlm_lazy_load", True), True),
            "vlm_low_memory": as_bool(config.get("vlm_low_memory", True), True),
            "vlm_max_new_tokens": int(config.get("vlm_max_new_tokens", 128)),
            "vlm_yolo_low_threshold": float(config.get("vlm_yolo_low_threshold", 0.45)),
            "vlm_siglip_low_threshold": float(config.get("vlm_siglip_low_threshold", 0.55)),
            "vlm_margin_threshold": float(config.get("vlm_margin_threshold", 0.10)),
            "vlm_force_special_classes": as_bool(config.get("vlm_force_special_classes", True), True),
            "vlm_check_each_image": as_bool(config.get("vlm_check_each_image", False)),
        }

    def _on_model_selection_changed(self, model_name: str) -> None:
        if self.model_loaded:
            self.model_loaded = False
            self.settings.busy_label.setText(f"已选择 {model_name}，请重新点击“加载模型”")

    def _on_model_loaded(self, device_description: str) -> None:
        self.model_loaded = True
        self.settings.set_busy(False, f"模型已加载：{device_description}")
        self.statusBar().showMessage("YOLO-World V2 模型已加载，后续图片不会重复加载模型")
        self._save_panel_config()

    def annotate_current(self) -> None:
        if not self.model_loaded:
            QMessageBox.information(self, "模型未加载", "请先点击“加载模型”。")
            return
        if not 0 <= self.current_index < len(self.images):
            return
        classes = self._valid_classes()
        if classes is None:
            return
        path = self.images[self.current_index]
        if not self._confirm_replace_invalid_label(path):
            return
        has_existing_label = (
            self.project is not None and self.project.label_path(path).exists()
        )
        if (
            self.current_annotation.objects
            or has_existing_label
            or self.current_annotation.status != AnnotationStatus.UNLABELED
        ):
            answer = QMessageBox.question(
                self,
                "覆盖已有标注？",
                "当前图片已有标注。重新自动标注会替换现有框，是否覆盖？",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.settings.set_busy(True, f"正在使用 GPU 标注：{path.name}")
        self._pending_single_root = self.project.root if self.project is not None else None
        self._pending_single_path = path.resolve()
        config = self.settings.config_values()
        self.predict_single_signal.emit(
            {
                "image_path": str(path),
                "classes": classes,
                "confidence": config["confidence"],
                "iou": config["iou"],
                "imgsz": config["imgsz"],
                **self._verification_payload(config),
                "class_profiles": self._class_profiles_payload(classes),
            }
        )

    def _on_prediction_ready(self, image_path: str, boxes: list) -> None:
        path = Path(image_path)
        current_root = self.project.root if self.project is not None else None
        if (
            self._pending_single_path != path.resolve()
            or self._pending_single_root != current_root
        ):
            LOGGER.warning("丢弃来自旧项目或旧请求的单图推理结果：%s", path)
            return
        self.settings.set_busy(False, "")
        self._pending_single_path = None
        self._pending_single_root = None
        annotation = ImageAnnotation(AnnotationStatus.AUTO_LABELED, list(boxes))
        self._update_annotation_verification(annotation)
        pipeline = self._pipeline_for_path(path)
        if not pipeline:
            pipeline = deepcopy(self.inference_worker.last_pipeline_metrics)
        annotation.verification["pipeline"] = deepcopy(pipeline)
        if self.project is not None:
            try:
                self.project.config.update(self.settings.config_values())
                width, height = self.canvas.image_size if path == self.canvas.current_path else (0, 0)
                if width <= 0 or height <= 0:
                    from yolo_world_annotator.utils.image_utils import read_image

                    image = read_image(path)
                    height, width = image.shape[:2]
                self.project.save_annotation(path, annotation, (width, height))
                record_auto_issues(self.project.hard_samples_path, path, list(boxes))
            except Exception as exc:
                self._on_inference_failed("save", str(exc))
                return
        if path == self.canvas.current_path:
            self.current_annotation = annotation
            self.canvas.set_boxes(boxes, self.settings.review_spin.value())
            self._update_segmentation_preview(path, pipeline)
            history = self.history_by_image.setdefault(path.name, HistoryManager(20))
            if not history.initialized:
                history.reset(boxes)
            else:
                history.push(boxes)
            row = self.image_list.row_for_path(path)
            if row >= 0:
                self.image_list.update_status(row, AnnotationStatus.AUTO_LABELED.value)
        metrics = pipeline
        detail = ""
        if metrics.get("sahi_enabled"):
            detail = (
                f"；SAHI tiles={metrics.get('tile_count', 0)}，"
                f"raw={metrics.get('raw_box_count', len(boxes))}，"
                f"merged={metrics.get('merged_box_count', len(boxes))}"
            )
        self.statusBar().showMessage(
            f"自动标注完成：{path.name}，检测到 {len(boxes)} 个目标{detail}"
        )
        self._refresh_image_filter()

    def annotate_all(self) -> None:
        if not self.model_loaded:
            QMessageBox.information(self, "模型未加载", "请先点击“加载模型”。")
            return
        if self.project is None:
            QMessageBox.information(self, "数据集未打开", "请先打开标准数据集根目录。")
            return
        classes = self._valid_classes()
        if not classes:
            return
        mode = self.settings.existing_mode_combo.currentData()
        candidates: list[Path] = []
        for image_path in self.images:
            label_exists = self.project.label_path(image_path).exists()
            annotation = self.project.annotations.get(image_path.name)
            status = annotation.status if annotation else AnnotationStatus.UNLABELED
            if mode == "OVERWRITE":
                candidates.append(image_path)
            elif mode == "SKIP_EXISTING" and not label_exists:
                candidates.append(image_path)
            elif mode == "ONLY_UNLABELED" and not label_exists and status == AnnotationStatus.UNLABELED:
                candidates.append(image_path)
        if not candidates:
            QMessageBox.information(self, "没有待处理图片", "当前安全策略下没有需要自动标注的图片。")
            return
        if mode == "OVERWRITE":
            answer = QMessageBox.warning(
                self,
                "确认覆盖已有标签",
                f"将重新标注 {len(candidates)} 张图片并覆盖已有标签。是否继续？",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            existing_labels = list(self.project.labels_dir.glob("*.txt"))
            if existing_labels:
                backup_dir = self.project.root / f"labels_backup_{datetime.now():%Y%m%d_%H%M%S}"
                backup_dir.mkdir(parents=True, exist_ok=False)
                for label_path in existing_labels:
                    shutil.copy2(label_path, backup_dir / label_path.name)
                LOGGER.warning("批量覆盖前已备份 %d 个标签到 %s", len(existing_labels), backup_dir)
        config = self.settings.config_values()
        self._save_panel_config()
        self.progress_bar.setRange(0, len(candidates))
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"0 / {len(candidates)}")
        self.progress_bar.setVisible(True)
        self.settings.set_batch_running(True)
        self.open_button.setEnabled(False)
        self._active_batch_root = self.project.root
        self.settings.busy_label.setText(f"批量自动标注：0 / {len(candidates)}")
        self.batch_signal.emit(
            {
                "images": [str(path) for path in candidates],
                "classes": classes,
                "confidence": config["confidence"],
                "iou": config["iou"],
                "imgsz": config["imgsz"],
                **self._verification_payload(config),
                "class_profiles": self._class_profiles_payload(classes),
            }
        )

    def pause_batch(self) -> None:
        self.inference_worker.request_pause()

    def resume_batch(self) -> None:
        self.inference_worker.request_resume()

    def cancel_batch(self) -> None:
        self.inference_worker.request_cancel()
        self.settings.busy_label.setText("正在取消；当前 GPU 推理结束后停止…")

    def _on_batch_paused(self, paused: bool) -> None:
        self.settings.pause_button.setEnabled(not paused)
        self.settings.resume_button.setEnabled(paused)
        if paused:
            self.settings.busy_label.setText("批量任务已暂停")

    def _on_batch_item_ready(
        self, image_path: str, boxes: list[BoundingBox], completed: int, total: int, speed: float
    ) -> None:
        if self.project is None:
            return
        if self._active_batch_root != self.project.root:
            LOGGER.warning("丢弃来自旧项目的批量推理结果：%s", image_path)
            return
        path = Path(image_path)
        try:
            from yolo_world_annotator.utils.image_utils import read_image

            image = read_image(path)
            height, width = image.shape[:2]
            annotation = ImageAnnotation(AnnotationStatus.AUTO_LABELED, list(boxes))
            self._update_annotation_verification(annotation)
            pipeline = self._pipeline_for_path(path)
            if not pipeline:
                pipeline = deepcopy(self.inference_worker.last_pipeline_metrics)
            annotation.verification["pipeline"] = deepcopy(pipeline)
            self.project.save_annotation(path, annotation, (width, height))
            record_auto_issues(self.project.hard_samples_path, path, list(boxes))
            try:
                row = self.image_list.row_for_path(path)
                if row >= 0:
                    self.image_list.update_status(row, AnnotationStatus.AUTO_LABELED.value)
            except ValueError:
                pass
            if path == self.canvas.current_path:
                self.current_annotation = annotation
                self.canvas.set_boxes(boxes, self.settings.review_spin.value())
                self._update_segmentation_preview(path, pipeline)
                history = self.history_by_image.setdefault(path.name, HistoryManager(20))
                if not history.initialized:
                    history.reset(boxes)
                else:
                    history.push(boxes)
        except Exception as exc:
            LOGGER.exception("保存批量标注失败：%s", path)
            self._on_batch_item_failed(image_path, str(exc), completed, total)
            return
        self.progress_bar.setValue(completed)
        self.progress_bar.setFormat(f"{completed} / {total}   {speed:.2f} img/s")
        metrics = pipeline
        mode_detail = (
            f"\nSAHI tiles={metrics.get('tile_count', 0)}，merged={metrics.get('merged_box_count', len(boxes))}"
            if metrics.get("sahi_enabled")
            else ""
        )
        self.settings.busy_label.setText(
            f"正在标注：{completed} / {total}\n当前速度：{speed:.2f} img/s\n{path.name}{mode_detail}"
        )

    def _on_batch_item_failed(self, image_path: str, message: str, completed: int, total: int) -> None:
        LOGGER.error("批量任务跳过 %s：%s", image_path, message)
        self.progress_bar.setValue(completed)
        self.progress_bar.setFormat(f"{completed} / {total}（有错误，详见 logs/app.log）")
        self.settings.busy_label.setText(f"已跳过损坏/失败图片：{Path(image_path).name}")

    def _on_batch_finished(self, cancelled: bool, completed: int, total: int) -> None:
        self.settings.set_batch_running(False)
        self.open_button.setEnabled(True)
        self._active_batch_root = None
        self.settings.busy_label.setText(
            f"批量任务已取消：{completed} / {total}" if cancelled else f"批量标注完成：{completed} / {total}"
        )
        self.progress_bar.setValue(completed)
        self.statusBar().showMessage(self.settings.busy_label.text())
        self._refresh_image_filter()

    def _confirm_replace_invalid_label(self, image_path: Path) -> bool:
        resolved = image_path.resolve()
        if resolved not in self.label_load_failed_paths:
            return True
        if self.project is None:
            return False
        answer = QMessageBox.warning(
            self,
            "原标签格式错误",
            "当前图片的已有标签无法解析。为保护原文件，保存已被阻止。\n\n"
            "是否备份原标签并明确覆盖？",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.Cancel,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.statusBar().showMessage("已取消保存；格式错误的原标签未被覆盖")
            return False
        label_path = self.project.label_path(image_path)
        if label_path.exists():
            backup = label_path.with_name(
                f"{label_path.name}.invalid.{datetime.now():%Y%m%d_%H%M%S}.bak"
            )
            shutil.copy2(label_path, backup)
            LOGGER.warning("已备份格式错误标签：%s", backup)
        self.label_load_failed_paths.discard(resolved)
        return True

    @staticmethod
    def _update_annotation_verification(annotation: ImageAnnotation) -> None:
        objects = annotation.objects
        counts = {"AUTO_ACCEPT": 0, "REVIEW": 0, "REJECT": 0}
        agreements = 0
        siglip_objects = 0
        sahi_objects = 0
        vlm_triggered = 0
        vlm_uncertain = 0
        for box in objects:
            if box.fusion_status in counts:
                counts[box.fusion_status] += 1
            if box.siglip_enabled:
                siglip_objects += 1
                agreements += int(bool(box.agreement))
            if box.sahi_enabled:
                sahi_objects += 1
            if box.vlm_triggered:
                vlm_triggered += 1
            if box.vlm_final_result == "UNCERTAIN" or box.vlm_parse_error:
                vlm_uncertain += 1
        annotation.verification = {
            "enabled": siglip_objects > 0,
            "objects": len(objects),
            "auto_accept": counts["AUTO_ACCEPT"],
            "review": counts["REVIEW"],
            "reject": counts["REJECT"],
            "agreement": agreements,
            "review_required": any(box.review_required for box in objects),
            "sahi_objects": sahi_objects,
            "vlm_triggered": vlm_triggered,
            "vlm_uncertain": vlm_uncertain,
            "modes": sorted({box.inference_mode for box in objects if box.inference_mode}),
        }

    def _record_human_edit(
        self,
        image_path: Path,
        previous: list[BoundingBox],
        current: list[BoundingBox],
    ) -> None:
        if self.project is None:
            return
        try:
            paired = min(len(previous), len(current))
            for index in range(paired):
                old, new = previous[index], current[index]
                changed = (
                    old.class_id != new.class_id
                    or abs(old.x1 - new.x1) > 1e-3
                    or abs(old.y1 - new.y1) > 1e-3
                    or abs(old.x2 - new.x2) > 1e-3
                    or abs(old.y2 - new.y2) > 1e-3
                )
                if changed and old.source != "MANUAL":
                    append_hard_sample(
                        self.project.hard_samples_path,
                        image=image_path,
                        box=new,
                        original_box=old,
                        human_correction=new,
                        error_type="HUMAN_CORRECTION",
                    )
            for old in previous[paired:]:
                if old.source != "MANUAL":
                    append_hard_sample(
                        self.project.hard_samples_path,
                        image=image_path,
                        box=None,
                        original_box=old,
                        error_type="FALSE_POSITIVE",
                    )
            for new in current[paired:]:
                append_hard_sample(
                    self.project.hard_samples_path,
                    image=image_path,
                    box=new,
                    human_correction=new,
                    error_type="FALSE_NEGATIVE",
                )
        except Exception:
            LOGGER.exception("记录 hard sample 失败：%s", image_path)

    def save_current(self) -> bool:
        if self.project is None or not 0 <= self.current_index < len(self.images):
            return False
        image_path = self.images[self.current_index]
        if not self._confirm_replace_invalid_label(image_path):
            return False
        try:
            self._save_panel_config()
            self.project.save_annotation(
                image_path, self.current_annotation, self.canvas.image_size
            )
            self.statusBar().showMessage("当前标注和项目配置已保存")
            return True
        except Exception as exc:
            LOGGER.exception("保存失败")
            QMessageBox.critical(self, "保存失败", str(exc))
            return False

    def verify_current(self) -> None:
        if not 0 <= self.current_index < len(self.images):
            return
        previous_status = self.current_annotation.status
        self.current_annotation.status = AnnotationStatus.VERIFIED
        for box in self.current_annotation.objects:
            box.review_confirmed = True
            box.review_required = False
            box.fusion_status = HUMAN_CONFIRMED
            box.decision_state = "HUMAN_ACCEPT"
            box.decision_reason = "人工确认接受"
        self._update_annotation_verification(self.current_annotation)
        if not self.save_current():
            self.current_annotation.status = previous_status
            return
        row = self.image_list.row_for_path(self.images[self.current_index])
        if row >= 0:
            self.image_list.update_status(row, AnnotationStatus.VERIFIED.value)
        self._refresh_image_filter()
        self.statusBar().showMessage("当前图片已人工确认：VERIFIED")

    def _on_canvas_edited(self, boxes: list[BoundingBox], edited_index: int) -> None:
        if not 0 <= self.current_index < len(self.images):
            return
        previous_boxes = deepcopy(self.current_annotation.objects)
        previous_status = self.current_annotation.status
        self.current_annotation.objects = list(boxes)
        self.current_annotation.status = AnnotationStatus.AUTO_LABELED
        self._update_annotation_verification(self.current_annotation)
        name = self.images[self.current_index].name
        history = self.history_by_image.setdefault(name, HistoryManager(20))
        if not history.initialized:
            history.reset(boxes)
        else:
            history.push(boxes)
        row = self.image_list.row_for_path(self.images[self.current_index])
        if row >= 0:
            self.image_list.update_status(row, AnnotationStatus.AUTO_LABELED.value)
        if not self.save_current():
            previous = history.undo()
            if previous is not None:
                self.current_annotation.objects = previous
                self.current_annotation.status = previous_status
                self.canvas.set_boxes(previous, self.settings.review_spin.value())
                row = self.image_list.row_for_path(self.images[self.current_index])
                if row >= 0:
                    self.image_list.update_status(row, previous_status.value)
            return
        self._record_human_edit(self.images[self.current_index], previous_boxes, list(boxes))
        self._on_canvas_selection_changed(self.canvas.selected_index)
        self._refresh_image_filter()

    def _on_new_box_requested(self, rect: QRectF) -> None:
        classes = self._valid_classes()
        if not classes:
            return
        default_index = max(0, self.settings.selected_class_combo.currentIndex())
        selected, accepted = QInputDialog.getItem(
            self, "请选择类别", "新标注框类别：", classes, default_index, False
        )
        if not accepted:
            return
        class_id = classes.index(selected)
        self.canvas.add_box(
            BoundingBox(
                class_id,
                selected,
                rect.left(),
                rect.top(),
                rect.right(),
                rect.bottom(),
                None,
                "MANUAL",
            )
        )

    def _choose_box_class(self, index: int) -> None:
        classes = self._valid_classes()
        if not classes or not 0 <= index < len(self.current_annotation.objects):
            return
        current_id = self.current_annotation.objects[index].class_id
        selected, accepted = QInputDialog.getItem(
            self, "修改类别", "标注框类别：", classes, current_id, False
        )
        if accepted:
            class_id = classes.index(selected)
            self.canvas.set_box_class(index, class_id, selected)

    def _on_canvas_selection_changed(self, index: int) -> None:
        if not 0 <= index < len(self.current_annotation.objects):
            self.settings.set_verification_detail("")
            return
        box = self.current_annotation.objects[index]
        class_id = box.class_id
        self.settings.selected_class_combo.blockSignals(True)
        self.settings.selected_class_combo.setCurrentIndex(class_id)
        self.settings.selected_class_combo.blockSignals(False)
        final_score = box.combined_confidence
        if final_score is None:
            final_score = box.confidence
        lines = [
            f"Final：{box.class_name}",
            f"Status：{box.fusion_status or box.decision_state or 'REVIEW'}",
            f"Mode：{box.inference_mode or ('SAHI' if box.sahi_enabled else 'NORMAL')}"
            + (f"（tiles={box.sahi_tile_count}）" if box.sahi_enabled else ""),
            f"YOLO-World：{box.yolo_class_name or box.class_name} "
            f"{(box.yolo_confidence if box.yolo_confidence is not None else box.confidence or 0.0):.2f}",
        ]
        if box.siglip_enabled:
            lines.extend(
                [
                    f"SigLIP2：{box.siglip_class_name or '—'} "
                    f"{(box.siglip_score or 0.0):.2f}; top2={box.siglip_top2_class_name or '—'} "
                    f"{(box.siglip_top2_score or 0.0):.2f}; margin={box.siglip_margin if box.siglip_margin is not None else 0.0:.2f}",
                    f"Combined：{(final_score or 0.0):.2f}" if final_score is not None else "Combined：冲突，不计算",
                ]
            )
            if box.agreement is False:
                lines.extend(["⚠ MODEL CONFLICT", "REVIEW REQUIRED"])
        else:
            lines.append("SigLIP2：未启用（YOLO-only）")
        if box.vlm_triggered or box.vlm_enabled:
            lines.append(
                f"VLM：{'已触发' if box.vlm_triggered else '已启用但未触发'} "
                f"{box.vlm_final_result or '—'}"
            )
            if box.vlm_features:
                lines.append("特征：" + ", ".join(f"{key}={value}" for key, value in box.vlm_features.items()))
            if box.vlm_parse_error:
                lines.append(f"VLM 错误：{box.vlm_parse_error}")
        if box.decision_reason:
            lines.append(f"原因：{box.decision_reason}")
        if box.human_modified:
            lines.append("人工修改拥有最高优先级")
        self.settings.set_verification_detail("\n".join(lines))

    def _change_selected_box_class(self, class_id: int) -> None:
        index = self.canvas.selected_index
        classes = self.settings.classes()
        if 0 <= index < len(self.current_annotation.objects) and 0 <= class_id < len(classes):
            self.canvas.set_box_class(index, class_id, classes[class_id])

    def undo(self) -> None:
        if not 0 <= self.current_index < len(self.images):
            return
        history = self.history_by_image.get(self.images[self.current_index].name)
        boxes = history.undo() if history is not None else None
        if boxes is None:
            self.statusBar().showMessage("没有可撤销的操作")
            return
        self.current_annotation.objects = boxes
        self.current_annotation.status = AnnotationStatus.AUTO_LABELED
        self.canvas.set_boxes(boxes, self.settings.review_spin.value())
        self.save_current()
        self.statusBar().showMessage("已撤销上一步标注操作")

    def show_statistics(self) -> None:
        if self.project is None:
            return
        try:
            self.save_current()
            statistics = collect_statistics(self.project)
            for error in statistics["errors"]:
                LOGGER.error("统计时跳过：%s", error)
            StatisticsDialog(statistics, self).exec()
        except Exception as exc:
            LOGGER.exception("统计失败")
            QMessageBox.critical(self, "统计失败", str(exc))

    def evaluate_ab(self) -> None:
        """Run the offline YOLO-only vs fused evaluation from the GUI."""

        if self.project is None:
            QMessageBox.information(self, "数据集未打开", "请先打开标准数据集根目录。")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Ground Truth JSON",
            str(self.project.root),
            "JSON files (*.json);;All files (*)",
        )
        if not filename:
            return
        try:
            self.save_current()
            result = evaluate_ab_project(
                self.project,
                load_ground_truth(Path(filename)),
            )
            yolo = result["yolo_only"]
            fused = result["yolo_siglip2"]
            message = (
                "A/B 评估完成（IoU=%.2f）\n\n"
                "YOLO-only：P %.3f / R %.3f / F1 %.3f\n"
                "YOLO + SigLIP2：P %.3f / R %.3f / F1 %.3f\n\n"
                "模型一致率：%.1f%%\n\n"
                "详细 per-class 指标请运行 scripts/evaluate_ab.py 查看。"
                % (
                    result["iou_threshold"],
                    yolo["precision"],
                    yolo["recall"],
                    yolo["f1"],
                    fused["precision"],
                    fused["recall"],
                    fused["f1"],
                    result["agreement"] * 100,
                )
            )
            QMessageBox.information(self, "YOLO / SigLIP2 A/B 评估", message)
        except Exception as exc:
            LOGGER.exception("A/B 评估失败")
            QMessageBox.critical(self, "A/B 评估失败", str(exc))

    def run_benchmark(self) -> None:
        """Compare stored Mode A/B/C/D metadata against a ground truth JSON."""

        if self.project is None:
            QMessageBox.information(self, "数据集未打开", "请先打开标准数据集根目录。")
            return
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "选择 Ground Truth JSON（用于 A/B/C/D Benchmark）",
            str(self.project.root),
            "JSON files (*.json);;All files (*)",
        )
        if not filename:
            return
        try:
            self.save_current()
            result = benchmark_project(self.project, load_ground_truth(Path(filename)))
            lines = ["A/B/C/D Benchmark 完成（IoU=%.2f）" % result["iou_threshold"]]
            for mode in result["mode_order"]:
                entry = result["modes"][mode]
                metrics = entry["metrics"]
                rates = entry["rates"]
                lines.append(
                    f"{mode}: P {metrics['precision']:.3f} / R {metrics['recall']:.3f} / "
                    f"F1 {metrics['f1']:.3f} / Review {rates['review_rate']:.1%} / "
                    f"VLM {rates['vlm_trigger_rate']:.1%}"
                )
            QMessageBox.information(self, "Pipeline Benchmark", "\n".join(lines))
        except Exception as exc:
            LOGGER.exception("Pipeline Benchmark 失败")
            QMessageBox.critical(self, "Pipeline Benchmark 失败", str(exc))

    def export_dataset(self) -> None:
        if self.project is None:
            return
        try:
            self.save_current()
            destination = self.project.root / "dataset_export"
            if destination.exists() and any(destination.iterdir()):
                destination = self.project.root / f"dataset_export_{datetime.now():%Y%m%d_%H%M%S}"
            result = export_yolo_dataset(
                self.project,
                destination,
                train_ratio=self.settings.train_ratio_spin.value(),
            )
            QMessageBox.information(
                self,
                "导出完成",
                f"已导出 {result['total']} 张图片\n"
                f"Train：{result['train']}\nVal：{result['val']}\n\n"
                f"data.yaml：{destination / 'data.yaml'}",
            )
            self.statusBar().showMessage(f"标准 YOLO 数据集已导出：{destination}")
        except Exception as exc:
            LOGGER.exception("导出失败")
            QMessageBox.critical(self, "导出失败", str(exc))

    def _save_panel_config(self) -> None:
        if self.project is not None:
            self.project.update_config(self.settings.config_values())

    def _on_inference_failed(self, stage: str, message: str) -> None:
        self.settings.set_busy(False, "")
        if stage == "predict":
            self._pending_single_path = None
            self._pending_single_root = None
        title = {
            "load": "模型加载失败",
            "predict": "自动标注失败",
            "batch": "批量标注失败",
            "save": "标注保存失败",
        }.get(stage, "操作失败")
        QMessageBox.critical(self, title, message)
        self.statusBar().showMessage(f"{title}：{message}")

    def _show_load_error(self, message: str) -> None:
        LOGGER.error(message)
        QMessageBox.warning(self, "图片读取失败", message)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.project is not None:
            try:
                self._save_panel_config()
            except Exception:
                LOGGER.exception("关闭时保存项目配置失败")
        self.inference_worker.request_cancel()
        self.inference_thread.quit()
        if not self.inference_thread.wait(5000):
            event.ignore()
            QMessageBox.warning(self, "后台任务仍在运行", "请等待当前推理完成后再关闭程序。")
            return
        self.inference_worker.release_models()
        super().closeEvent(event)
