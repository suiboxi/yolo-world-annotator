from __future__ import annotations

from pathlib import Path
from time import monotonic

import torch
from PySide6.QtCore import QObject, QRectF, QSettings, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from app.canvas import AnnotationCanvas
from core.annotation import AnnotationStatus, BoundingBox, ImageAnnotation
from core.dataset import DatasetProject
from core.yolo_format import serialize_yolo
from models.yolo_world import YOLOWorldDetector
from utils.config import atomic_write_text
from utils.image_utils import read_image


MODEL_PRESETS = (
    (
        "yolov8s-worldv2.pt",
        "S 轻量版（推荐）",
        "速度最快、显存占用最低，适合批量初标和 RTX 4060。",
    ),
    (
        "yolov8m-worldv2.pt",
        "M 均衡版",
        "速度与检测能力更均衡；未下载时首次使用会自动下载。",
    ),
    (
        "yolov8l-worldv2.pt",
        "L 高精度版",
        "比 M 更大，小目标潜力更好，但速度更慢、8 GB 显存建议降低图像尺寸。",
    ),
    (
        "yolov8x-worldv2.pt",
        "X 最大版（高精度）",
        "参数量最大，更偏向精度，速度最慢且显存压力最高；RTX 4060 建议从 640 尺寸试用。",
    ),
    (
        "yolov8s-world.pt",
        "S 轻量版（经典 V1）",
        "YOLO-World 经典版的轻量权重；适合兼容旧项目，通常优先选择上方 V2。",
    ),
    (
        "yolov8m-world.pt",
        "M 均衡版（经典 V1）",
        "YOLO-World 经典版的中型权重；未下载时首次使用会自动下载。",
    ),
    (
        "yolov8l-world.pt",
        "L 高精度版（经典 V1）",
        "YOLO-World 经典版的大型权重；速度较慢，8 GB 显存建议降低图像尺寸。",
    ),
    (
        "yolov8x-world.pt",
        "X 最大版（经典 V1）",
        "YOLO-World 经典版的最大权重；显存占用最高，RTX 4060 建议从 640 尺寸试用。",
    ),
)


class NoWheelComboBox(QComboBox):
    """A combo box that never changes selection from an accidental wheel."""

    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """A spin box editable by keyboard/arrows, but not by mouse wheel."""

    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """A double spin box editable by keyboard/arrows, but not by wheel."""

    def wheelEvent(self, event) -> None:  # noqa: N802
        event.ignore()


class InferenceEngine(QObject):
    """Persistent GPU worker; the detector remains resident between jobs."""

    results_ready = Signal(object)
    status_changed = Signal(str)
    model_ready = Signal(str, str)
    failed = Signal(str)
    finished = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.detector: YOLOWorldDetector | None = None
        self.model_path: Path | None = None
        self.active_prompts: list[str] = []
        self.cancel_requested = False

    def request_cancel(self) -> None:
        self.cancel_requested = True

    @Slot(object)
    def run_job(self, job: dict) -> None:
        self.cancel_requested = False
        try:
            model_path = Path(job["model_path"])
            classes = list(job["classes"])
            prompts = list(job.get("prompts") or classes)
            if len(prompts) != len(classes):
                raise ValueError("类别数与模型提示词数不一致。")
            paths = [Path(path) for path in job["paths"]]
            if (
                self.detector is None
                or self.model_path != model_path.resolve()
                or prompts != self.active_prompts
            ):
                self.status_changed.emit("正在将 YOLO-World 加载到 GPU，首次需稍候…")
                self.detector = YOLOWorldDetector(model_path)
                self.model_path = model_path.resolve()
                self.detector.set_classes(prompts)
                self.active_prompts = prompts
            self.model_ready.emit(
                str(self.model_path),
                getattr(self.detector, "device_description", "cuda:0"),
            )
            total = len(paths)
            pending_results: list[tuple[Path, list[BoundingBox], tuple[int, int], int, int]] = []
            for index, path in enumerate(paths, start=1):
                if self.cancel_requested:
                    if pending_results:
                        self.results_ready.emit(pending_results)
                    self.finished.emit(True)
                    return
                if index == 1 or index % 10 == 0 or index == total:
                    self.status_changed.emit(
                        f"GPU 正在标注 {index}/{total}：{path.name}"
                    )
                boxes = self.detector.predict(
                    path,
                    confidence=float(job["confidence"]),
                    iou=float(job["iou"]),
                    imgsz=int(job["imgsz"]),
                )
                for box in boxes:
                    if 0 <= box.class_id < len(classes):
                        box.class_name = classes[box.class_id]
                        box.yolo_class_name = classes[box.class_id]
                image_size = self.detector.last_image_size
                if image_size is None:
                    image = read_image(path)
                    height, width = image.shape[:2]
                    image_size = (width, height)
                width, height = image_size
                serialized = serialize_yolo(boxes, width, height)
                lines = [line for line in serialized.splitlines() if line.strip()]
                if len(lines) != len(boxes):
                    raise RuntimeError(
                        f"{path.name} 保存前校验失败：{len(boxes)} 个框，"
                        f"但只生成 {len(lines)} 行标注。"
                    )
                label_path = path.with_suffix(".txt")
                atomic_write_text(label_path, serialized)
                saved_lines = [
                    line
                    for line in label_path.read_text(encoding="utf-8-sig").splitlines()
                    if line.strip()
                ]
                if len(saved_lines) != len(boxes):
                    raise RuntimeError(
                        f"{path.name} 落盘校验失败：预期 {len(boxes)} 行，"
                        f"实际 {len(saved_lines)} 行。"
                    )
                pending_results.append((path, boxes, image_size, index, total))
                if len(pending_results) >= 20 or index == total:
                    self.results_ready.emit(pending_results)
                    pending_results = []
            self.finished.emit(False)
        except Exception as exc:  # surface model/CUDA/image errors to the GUI
            if "pending_results" in locals() and pending_results:
                self.results_ready.emit(pending_results)
            self.failed.emit(str(exc))
            self.finished.emit(False)


class AnnotatorWindow(QMainWindow):
    job_requested = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO-World GPU 自动标注器")
        self.resize(1440, 880)
        self.setMinimumSize(860, 560)
        self.app_settings = QSettings("LocalYOLOTools", "YOLO-World GPU 标注器")
        self.project: DatasetProject | None = None
        self.image_paths: list[Path] = []
        self.current_path: Path | None = None
        self.current_annotation: ImageAnnotation | None = None
        self.busy = False
        self.close_when_idle = False
        self.last_batch_preview_at = 0.0
        self.batch_processed = 0
        self.batch_box_count = 0
        self.batch_zero_count = 0
        self.image_items: dict[Path, QListWidgetItem] = {}
        self.busy_actions: list[QAction] = []
        self.shortcuts: list[QShortcut] = []

        self._build_ui()
        self._build_worker()
        self._build_shortcuts()
        self._show_gpu_state()

    @property
    def default_model_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "models" / "weights" / "yolov8s-worldv2.pt"

    @property
    def weights_dir(self) -> Path:
        return self.default_model_path.parent

    def _build_ui(self) -> None:
        toolbar = QToolBar("常用操作", self)
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(toolbar)
        for text, slot in (
            ("打开图片文件夹", self.choose_folder),
            ("选择一张图片", self.choose_image),
            ("上一张", self.previous_image),
            ("下一张", self.next_image),
            ("自动标注当前图", self.annotate_current),
            ("自动标注全部", self.annotate_all),
            ("新建标注框", self.canvas_start_create),
            ("删除选中框", self.canvas_delete),
            ("适应窗口", self.fit_canvas),
        ):
            action = QAction(text, self)
            action.triggered.connect(slot)
            toolbar.addAction(action)
            self.busy_actions.append(action)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(True)
        splitter.setHandleWidth(6)
        self.setCentralWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(145)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 4, 8)
        title = QLabel("图片列表")
        title.setStyleSheet("font-size:16px;font-weight:700")
        left_layout.addWidget(title)
        self.folder_label = QLabel("尚未打开图片文件夹")
        self.folder_label.setWordWrap(True)
        self.folder_label.setToolTip("标注 txt 会直接保存在这个图片文件夹中。")
        left_layout.addWidget(self.folder_label)
        self.filter_combo = NoWheelComboBox()
        self.filter_combo.addItem("显示全部图片", "ALL")
        self.filter_combo.addItem("只显示未标注", "UNLABELED")
        self.filter_combo.addItem("只显示已标注", "LABELED")
        self.filter_combo.setToolTip("按同目录中是否存在同名 txt 筛选。")
        self.filter_combo.currentIndexChanged.connect(self.refresh_image_list)
        left_layout.addWidget(self.filter_combo)
        self.image_list = QListWidget()
        self.image_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.image_list.currentItemChanged.connect(self._list_selection_changed)
        left_layout.addWidget(self.image_list, 1)
        self.counter_label = QLabel("0 张图片")
        left_layout.addWidget(self.counter_label)

        center = QWidget()
        center.setMinimumWidth(300)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(4, 8, 4, 8)
        self.preview_title = QLabel("实时标注预览")
        self.preview_title.setStyleSheet("font-size:16px;font-weight:700")
        center_layout.addWidget(self.preview_title)
        self.canvas = AnnotationCanvas()
        self.canvas.setFrameShape(QFrame.Shape.StyledPanel)
        self.canvas.boxes_edited.connect(self._on_boxes_edited)
        self.canvas.new_box_requested.connect(self._on_new_box)
        self.canvas.change_class_requested.connect(self._apply_selected_class)
        self.canvas.load_failed.connect(self._show_error)
        center_layout.addWidget(self.canvas, 1)
        self.canvas_tip = QLabel(
            "操作：单击选框，拖动框移动，拖动四角蓝色控制点改大小；"
            "B 后在图上拖拽新建，Delete 删除。每次修改都会立即保存。"
        )
        self.canvas_tip.setWordWrap(True)
        self.canvas_tip.setStyleSheet("color:#aeb8c5;padding:4px")
        center_layout.addWidget(self.canvas_tip)

        settings = self._build_settings_panel()
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(settings)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 3)
        splitter.setStretchFactor(2, 1)
        splitter.setSizes([200, 960, 420])

        status = QStatusBar()
        self.setStatusBar(status)
        self.status_label = QLabel("就绪")
        status.addWidget(self.status_label, 1)
        self.gpu_label = QLabel()
        status.addPermanentWidget(self.gpu_label)

    def _build_settings_panel(self) -> QScrollArea:
        scroll = QScrollArea()
        self.settings_scroll = scroll
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(320)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        content = QWidget()
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)

        heading = QLabel("标注设置（全部保存在图片目录）")
        heading.setWordWrap(True)
        heading.setStyleSheet("font-size:16px;font-weight:700")
        layout.addWidget(heading)

        model_group = QGroupBox("GPU 与 YOLO-World 模型")
        model_form = QFormLayout(model_group)
        self._configure_responsive_form(model_form)
        self.model_combo = NoWheelComboBox()
        self._populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self._model_choice_changed)
        self.model_path_edit = QLineEdit(str(self.default_model_path))
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setMinimumWidth(0)
        self.model_path_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.model_path_edit.setToolTip("本地 YOLO-World .pt 权重路径。")
        self.choose_model_button = QPushButton("浏览本机 .pt 权重…")
        self.choose_model_button.clicked.connect(self.choose_model)
        self.model_description = self._help("")
        model_form.addRow(self._parameter_label("预设型号", "提供 V2 与经典版的 S/M/L/X 八种官方权重，也可选择最近使用或自定义本机权重。"), self.model_combo)
        model_form.addRow(self._parameter_label("当前权重路径", "实际加载的 .pt 文件；双击路径可选择文本复制。"), self.model_path_edit)
        model_form.addRow("", self.choose_model_button)
        model_form.addRow(self.model_description)
        self.device_value = QLabel("正在检查…")
        self.device_value.setWordWrap(True)
        model_form.addRow(self._parameter_label("GPU 设备", "程序只允许 cuda:0，不会改用 CPU。"), self.device_value)
        layout.addWidget(model_group)
        self._select_model_path(self.default_model_path)

        class_group = QGroupBox("检测类别")
        class_layout = QVBoxLayout(class_group)
        class_layout.addWidget(self._help("上方是最终写入 classes.txt 的类别；下方是给 YOLO-World 搜索目标的英文提示词。两区必须一一对应。"))
        class_layout.addWidget(QLabel("最终标签类别（每行一个）："))
        self.classes_edit = QTextEdit()
        self.classes_edit.setMinimumWidth(0)
        self.classes_edit.setPlaceholderText("person\ncar\nbicycle")
        self.classes_edit.setMinimumHeight(90)
        class_layout.addWidget(self.classes_edit)
        class_layout.addWidget(QLabel("YOLO-World 模型提示词（英文，每行对应上方一类）："))
        self.prompts_edit = QTextEdit()
        self.prompts_edit.setMinimumWidth(0)
        self.prompts_edit.setPlaceholderText("person\nautomobile\nbicycle")
        self.prompts_edit.setMinimumHeight(90)
        self.prompts_edit.setToolTip("提示词可与最终类别不同。例如最终类别 raspberry，对当前仿真果实可使用 strawberry 搜框。")
        class_layout.addWidget(self.prompts_edit)
        save_classes = QPushButton("保存类别和模型提示词")
        save_classes.clicked.connect(self.save_classes)
        class_layout.addWidget(save_classes)
        layout.addWidget(class_group)

        parameter_group = QGroupBox("自动标注参数")
        parameter_form = QFormLayout(parameter_group)
        self._configure_responsive_form(parameter_form)
        self.confidence_spin = NoWheelDoubleSpinBox()
        self.confidence_spin.setRange(0.01, 0.99)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.25)
        self.iou_spin = NoWheelDoubleSpinBox()
        self.iou_spin.setRange(0.05, 0.95)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        self.imgsz_spin = NoWheelSpinBox()
        self.imgsz_spin.setRange(320, 1920)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.review_spin = NoWheelDoubleSpinBox()
        self.review_spin.setRange(0.01, 0.99)
        self.review_spin.setSingleStep(0.05)
        self.review_spin.setValue(0.50)
        parameter_form.addRow(self._parameter_label("置信度阈值", "越低检出的框越多，但误检也可能增多；建议先用 0.25。"), self.confidence_spin)
        parameter_form.addRow(self._parameter_label("重叠框去重阈值", "控制对同一目标的重叠框合并；建议 0.45。"), self.iou_spin)
        parameter_form.addRow(self._parameter_label("推理图像尺寸", "数值越大越容易检出小目标，但更慢且更占显存；4060 建议 640–960。"), self.imgsz_spin)
        parameter_form.addRow(self._parameter_label("低置信框提示阈值", "低于该值的框用黄色虚线显示，只影响提示，不删除标注。"), self.review_spin)
        layout.addWidget(parameter_group)

        edit_group = QGroupBox("人工修改")
        edit_layout = QVBoxLayout(edit_group)
        self.selected_class_combo = NoWheelComboBox()
        self.selected_class_combo.setToolTip("新建标注框使用此类别；也可将它应用到当前选中框。")
        apply_class = QPushButton("将此类别应用到选中框")
        apply_class.clicked.connect(self._apply_selected_class)
        create = QPushButton("在图上拖拽新建框（B）")
        create.clicked.connect(self.canvas_start_create)
        delete = QPushButton("删除选中框（Delete）")
        delete.clicked.connect(self.canvas_delete)
        edit_layout.addWidget(QLabel("当前类别："))
        edit_layout.addWidget(self.selected_class_combo)
        edit_layout.addWidget(apply_class)
        edit_layout.addWidget(create)
        edit_layout.addWidget(delete)
        edit_layout.addWidget(self._help("移动、改大小、新增、改类和删除都会在操作结束时立即原子保存到同名 txt。"))
        layout.addWidget(edit_group)

        batch_group = QGroupBox("批量处理")
        batch_layout = QVBoxLayout(batch_group)
        self.batch_mode_combo = NoWheelComboBox()
        self.batch_mode_combo.addItem("跳过已有非空 txt，空 txt 重新检测（推荐）", "SKIP")
        self.batch_mode_combo.addItem("重新标注并覆盖所有 txt", "OVERWRITE")
        self.batch_mode_combo.setToolTip("选择批量自动标注时是否保留已有人工结果。")
        self.auto_current_button = QPushButton("自动标注当前图片")
        self.auto_all_button = QPushButton("自动标注全部图片")
        self.cancel_button = QPushButton("取消当前批量任务")
        self.cancel_button.setEnabled(False)
        self.auto_current_button.clicked.connect(self.annotate_current)
        self.auto_all_button.clicked.connect(self.annotate_all)
        self.cancel_button.clicked.connect(self.cancel_inference)
        batch_layout.addWidget(self.batch_mode_combo)
        batch_layout.addWidget(self.auto_current_button)
        batch_layout.addWidget(self.auto_all_button)
        batch_layout.addWidget(self.cancel_button)
        batch_layout.addWidget(self._help("批量时每处理完一张图都会立即预览并保存，中途取消不会丢失已完成结果。"))
        layout.addWidget(batch_group)
        layout.addStretch(1)
        scroll.setWidget(content)
        horizontal_bar = scroll.horizontalScrollBar()
        horizontal_bar.setValue(0)
        horizontal_bar.rangeChanged.connect(lambda _minimum, _maximum: horizontal_bar.setValue(0))
        return scroll

    @staticmethod
    def _parameter_label(title: str, explanation: str) -> QLabel:
        label = QLabel(f"<b>{title}</b><br><span style='color:#9eafc0;font-size:11px'>{explanation}</span>")
        label.setWordWrap(True)
        label.setToolTip(explanation)
        label.setMinimumWidth(0)
        return label

    @staticmethod
    def _configure_responsive_form(form: QFormLayout) -> None:
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(10)

    @staticmethod
    def _help(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color:#9eafc0;font-size:12px")
        return label

    def _build_worker(self) -> None:
        self.worker_thread = QThread(self)
        self.engine = InferenceEngine()
        self.engine.moveToThread(self.worker_thread)
        self.job_requested.connect(self.engine.run_job)
        self.engine.results_ready.connect(self._on_inference_batch)
        self.engine.status_changed.connect(self.status_label.setText)
        self.engine.model_ready.connect(self._on_model_ready)
        self.engine.failed.connect(self._show_error)
        self.engine.finished.connect(self._inference_finished)
        self.worker_thread.start()

    def _build_shortcuts(self) -> None:
        for key, slot in (
            ("B", self.canvas_start_create),
            ("A", self.previous_image),
            ("D", self.next_image),
            (QKeySequence.StandardKey.Save, self.save_current),
        ):
            shortcut = QShortcut(QKeySequence(key) if isinstance(key, str) else key, self)
            shortcut.activated.connect(slot)
            self.shortcuts.append(shortcut)

    def _show_gpu_state(self) -> None:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            text = f"{name} / cuda:0 / {memory:.1f} GB / 半精度"
            self.device_value.setText(text)
            self.device_value.setStyleSheet("color:#4bd37b;font-weight:700")
            self.gpu_label.setText(f"GPU 已就绪：{name}")
        else:
            text = "不可用：未检测到 CUDA（不允许 CPU 回退）"
            self.device_value.setText(text)
            self.device_value.setStyleSheet("color:#ff6767;font-weight:700")
            self.gpu_label.setText(text)
            self.auto_current_button.setEnabled(False)
            self.auto_all_button.setEnabled(False)

    def _recent_model_paths(self) -> list[Path]:
        raw = self.app_settings.value("recent_models", [])
        values = [raw] if isinstance(raw, str) else list(raw or [])
        paths: list[Path] = []
        for value in values:
            path = Path(str(value)).expanduser()
            if path.is_file() and path.suffix.lower() == ".pt" and path not in paths:
                paths.append(path.resolve())
        return paths[:6]

    def _populate_model_combo(self) -> None:
        current_path = (
            Path(self.model_path_edit.text()).resolve()
            if hasattr(self, "model_path_edit") and self.model_path_edit.text().strip()
            else self.default_model_path.resolve()
        )
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        preset_paths: set[Path] = set()
        for filename, title, description in MODEL_PRESETS:
            path = (self.weights_dir / filename).resolve()
            preset_paths.add(path)
            availability = "已在本机" if path.is_file() else "未下载，首次使用自动下载"
            self.model_combo.addItem(f"{title} — {availability}", str(path))
            index = self.model_combo.count() - 1
            full_description = f"{description}\n文件：{path}\n状态：{availability}"
            self.model_combo.setItemData(index, full_description, Qt.ItemDataRole.ToolTipRole)
            self.model_combo.setItemData(index, full_description, Qt.ItemDataRole.UserRole + 1)

        recent_paths = [path for path in self._recent_model_paths() if path not in preset_paths]
        if recent_paths:
            self.model_combo.insertSeparator(self.model_combo.count())
            for path in recent_paths:
                description = f"最近成功加载的本机 YOLO-World 权重。\n文件：{path}"
                self.model_combo.addItem(f"最近使用：{path.name}", str(path))
                index = self.model_combo.count() - 1
                self.model_combo.setItemData(index, description, Qt.ItemDataRole.ToolTipRole)
                self.model_combo.setItemData(index, description, Qt.ItemDataRole.UserRole + 1)

        self.model_combo.insertSeparator(self.model_combo.count())
        self.model_combo.addItem("自定义本机 .pt 权重…", "__CUSTOM__")
        index = self.model_combo.count() - 1
        description = "点击下方“浏览本机 .pt 权重”选择已有文件。普通 YOLO 权重不具备开放词汇能力，必须选择 YOLO-World 权重。"
        self.model_combo.setItemData(index, description, Qt.ItemDataRole.ToolTipRole)
        self.model_combo.setItemData(index, description, Qt.ItemDataRole.UserRole + 1)
        self.model_combo.blockSignals(False)
        if hasattr(self, "model_path_edit"):
            self._select_model_path(current_path)

    def _select_model_path(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        self.model_path_edit.setText(str(resolved))
        self.model_path_edit.setToolTip(str(resolved))
        found = -1
        for index in range(self.model_combo.count()):
            data = str(self.model_combo.itemData(index) or "")
            if data and data != "__CUSTOM__" and Path(data).resolve() == resolved:
                found = index
                break
        if found < 0:
            custom_index = self.model_combo.findData("__CUSTOM__")
            insert_at = custom_index if custom_index >= 0 else self.model_combo.count()
            description = f"当前项目指定的本机权重。\n文件：{resolved}"
            self.model_combo.insertItem(insert_at, f"当前项目：{resolved.name}", str(resolved))
            self.model_combo.setItemData(insert_at, description, Qt.ItemDataRole.ToolTipRole)
            self.model_combo.setItemData(insert_at, description, Qt.ItemDataRole.UserRole + 1)
            found = insert_at
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(found)
        self.model_combo.blockSignals(False)
        self._update_model_description(found)

    def _update_model_description(self, index: int) -> None:
        description = str(self.model_combo.itemData(index, Qt.ItemDataRole.UserRole + 1) or "")
        self.model_description.setText(description)

    def _model_choice_changed(self, index: int) -> None:
        data = str(self.model_combo.itemData(index) or "")
        self._update_model_description(index)
        if not data or data == "__CUSTOM__":
            return
        self._select_model_path(Path(data))

    @Slot(str, str)
    def _on_model_ready(self, model_path: str, device: str) -> None:
        path = Path(model_path).resolve()
        preset_names = {item[0] for item in MODEL_PRESETS}
        if path.name not in preset_names:
            recent = [item for item in self._recent_model_paths() if item != path]
            self.app_settings.setValue("recent_models", [str(path), *(str(item) for item in recent[:5])])
        # A missing official preset may just have been downloaded, so rebuild
        # the rows to change its availability text to “已在本机”.
        self._populate_model_combo()
        self._select_model_path(path)
        self.status_label.setText(f"模型已就绪：{path.name} / {device}")

    def classes(self) -> list[str]:
        return [line.strip() for line in self.classes_edit.toPlainText().splitlines() if line.strip()]

    def prompts(self) -> list[str]:
        return [line.strip() for line in self.prompts_edit.toPlainText().splitlines() if line.strip()]

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择包含图片的文件夹")
        if folder:
            self.open_folder(Path(folder))

    def choose_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片 (*.jpg *.jpeg *.png *.bmp *.webp)"
        )
        if filename:
            path = Path(filename)
            self.open_folder(path.parent, select_path=path)

    def open_folder(self, folder: Path, select_path: Path | None = None) -> None:
        try:
            self.project = DatasetProject(folder)
            self.image_paths = self.project.list_images()
            config = self.project.config
            self.classes_edit.setPlainText("\n".join(self.project.classes))
            saved_prompts = config.get("class_prompts", self.project.classes)
            if not isinstance(saved_prompts, list) or len(saved_prompts) != len(self.project.classes):
                saved_prompts = self.project.classes
            self.prompts_edit.setPlainText("\n".join(str(item) for item in saved_prompts))
            self.confidence_spin.setValue(float(config.get("confidence", 0.25)))
            self.iou_spin.setValue(float(config.get("iou", 0.45)))
            self.imgsz_spin.setValue(int(config.get("imgsz", 640)))
            self.review_spin.setValue(float(config.get("review_threshold", 0.50)))
            self._select_model_path(Path(str(config.get("model_path", self.default_model_path))))
            self._refresh_class_combo()
            self.folder_label.setText(str(folder.resolve()))
            self.folder_label.setToolTip(str(folder.resolve()))
            self.refresh_image_list()
            target = select_path or (self.image_paths[0] if self.image_paths else None)
            if target is not None:
                self.select_path(target)
            else:
                self.canvas.clear_image()
                self.status_label.setText("文件夹中没有受支持的图片")
        except Exception as exc:
            self._show_error(str(exc))

    def refresh_image_list(self) -> None:
        current = self.current_path
        self.image_list.blockSignals(True)
        self.image_list.clear()
        self.image_items.clear()
        mode = str(self.filter_combo.currentData() or "ALL")
        shown = 0
        for path in self.image_paths:
            label_path = path.with_suffix(".txt")
            labeled = label_path.exists()
            empty_label = labeled and label_path.stat().st_size == 0
            if mode == "UNLABELED" and labeled:
                continue
            if mode == "LABELED" and not labeled:
                continue
            marker = "∅" if empty_label else ("●" if labeled else "○")
            item = QListWidgetItem(f"{marker} {path.name}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(
                str(path)
                + ("\n已处理，未检测到目标（txt 为空）" if empty_label else "")
            )
            self.image_list.addItem(item)
            self.image_items[path] = item
            shown += 1
            if current is not None and path == current:
                self.image_list.setCurrentItem(item)
        self.image_list.blockSignals(False)
        self.counter_label.setText(f"当前显示 {shown} / 共 {len(self.image_paths)} 张")

    def _list_selection_changed(self, current: QListWidgetItem | None, _previous) -> None:
        if current is not None:
            self.select_path(Path(current.data(Qt.ItemDataRole.UserRole)))

    def select_path(self, path: Path) -> None:
        if self.project is None or not path.exists():
            return
        if not self.canvas.load_image(path):
            return
        self.current_path = path
        try:
            self.current_annotation = self.project.get_annotation(path, self.canvas.image_size)
        except ValueError as exc:
            self.current_annotation = ImageAnnotation()
            self._show_error(f"{path.name} 的 txt 格式有误：{exc}")
        self.canvas.set_boxes(
            self.current_annotation.objects,
            self.review_spin.value(),
        )
        self.preview_title.setText(
            f"实时标注预览 — {path.name} — {len(self.current_annotation.objects)} 个框"
        )
        self.status_label.setText(f"已加载：{path}")
        self._select_list_item(path)

    def _select_list_item(self, path: Path) -> None:
        for row in range(self.image_list.count()):
            item = self.image_list.item(row)
            if Path(item.data(Qt.ItemDataRole.UserRole)) == path:
                if self.image_list.currentItem() is not item:
                    self.image_list.blockSignals(True)
                    self.image_list.setCurrentItem(item)
                    self.image_list.blockSignals(False)
                return

    def previous_image(self) -> None:
        self._step_image(-1)

    def next_image(self) -> None:
        self._step_image(1)

    def _step_image(self, delta: int) -> None:
        if not self.image_paths:
            return
        try:
            index = self.image_paths.index(self.current_path) if self.current_path else 0
        except ValueError:
            index = 0
        self.select_path(self.image_paths[(index + delta) % len(self.image_paths)])

    def choose_model(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择 YOLO-World 权重", "", "PyTorch 权重 (*.pt)")
        if filename:
            self._select_model_path(Path(filename))

    def save_classes(self) -> None:
        if self.project is None:
            self._show_error("请先打开图片文件夹。")
            return
        classes = self.classes()
        prompts = self.prompts()
        if not classes:
            self._show_error("请至少输入一个检测类别。")
            return
        if len(prompts) != len(classes):
            self._show_error(
                f"最终类别有 {len(classes)} 行，但模型提示词有 {len(prompts)} 行；请一一对应。"
            )
            return
        self.project.update_config({"classes": classes, "class_prompts": prompts})
        self._refresh_class_combo()
        self.status_label.setText("类别顺序与模型提示词已保存")

    def _refresh_class_combo(self) -> None:
        selected = self.selected_class_combo.currentIndex()
        self.selected_class_combo.clear()
        for class_id, name in enumerate(self.classes()):
            self.selected_class_combo.addItem(f"{class_id}：{name}", class_id)
        if self.selected_class_combo.count():
            self.selected_class_combo.setCurrentIndex(max(0, min(selected, self.selected_class_combo.count() - 1)))

    def _settings(self) -> dict:
        return {
            "classes": self.classes(),
            "class_prompts": self.prompts(),
            "model_path": self.model_path_edit.text().strip(),
            "confidence": self.confidence_spin.value(),
            "iou": self.iou_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "review_threshold": self.review_spin.value(),
        }

    def save_current(self) -> None:
        if self.project is None or self.current_path is None or self.current_annotation is None:
            return
        self.project.update_config(self._settings())
        self.project.save_annotation(self.current_path, self.current_annotation, self.canvas.image_size)
        self.status_label.setText(f"已实时保存：{self.current_path.with_suffix('.txt').name}")
        self.refresh_image_list()

    def _on_boxes_edited(self, boxes: list[BoundingBox], _selected: int) -> None:
        if self.current_annotation is None:
            return
        self.current_annotation.objects = boxes
        self.current_annotation.status = AnnotationStatus.AUTO_LABELED
        self.preview_title.setText(
            f"实时标注预览 — {self.current_path.name if self.current_path else ''} — {len(boxes)} 个框"
        )
        self.save_current()

    def canvas_start_create(self) -> None:
        if not self.classes():
            self._show_error("请先输入并保存至少一个类别。")
            return
        self.canvas.start_create_mode()
        self.status_label.setText("新建框模式：请在图片上按住左键拖拽，Esc 取消。")

    def _on_new_box(self, rect: QRectF) -> None:
        classes = self.classes()
        class_id = self.selected_class_combo.currentIndex()
        if not classes or class_id < 0:
            return
        self.canvas.add_box(
            BoundingBox(
                class_id=class_id,
                class_name=classes[class_id],
                x1=rect.left(), y1=rect.top(), x2=rect.right(), y2=rect.bottom(),
                confidence=None, source="MANUAL", human_modified=True,
            )
        )

    def _apply_selected_class(self, index: int | bool = False) -> None:
        box_index = index if isinstance(index, int) and not isinstance(index, bool) else self.canvas.selected_index
        classes = self.classes()
        class_id = self.selected_class_combo.currentIndex()
        if box_index < 0:
            self.status_label.setText("请先单击选中一个标注框。")
            return
        if classes and class_id >= 0:
            self.canvas.set_box_class(box_index, class_id, classes[class_id])

    def canvas_delete(self) -> None:
        if not self.canvas.delete_selected():
            self.status_label.setText("请先单击选中要删除的标注框。")

    def fit_canvas(self) -> None:
        self.canvas.fit_image()

    def annotate_current(self) -> None:
        if self.current_path is not None:
            self._start_inference([self.current_path])
        else:
            self._show_error("请先打开并选中一张图片。")

    def annotate_all(self) -> None:
        if not self.image_paths:
            self._show_error("当前文件夹没有可标注图片。")
            return
        paths = list(self.image_paths)
        if self.batch_mode_combo.currentData() == "SKIP":
            paths = [
                path
                for path in paths
                if not (
                    path.with_suffix(".txt").exists()
                    and path.with_suffix(".txt").stat().st_size > 0
                )
            ]
        if not paths:
            self.status_label.setText("所有图片都已有非空 txt，没有需要处理的图片。")
            return
        self._start_inference(paths)

    def _start_inference(self, paths: list[Path]) -> None:
        if self.busy:
            return
        if not torch.cuda.is_available():
            self._show_error("未检测到 CUDA GPU，已拒绝 CPU 推理。")
            return
        classes = self.classes()
        prompts = self.prompts()
        model_path = Path(self.model_path_edit.text().strip())
        if not classes:
            self._show_error("请至少输入一个检测类别。")
            return
        if len(prompts) != len(classes):
            self._show_error(
                f"最终类别有 {len(classes)} 行，但模型提示词有 {len(prompts)} 行；请一一对应。"
            )
            return
        if model_path.suffix.lower() != ".pt":
            self._show_error("请选择扩展名为 .pt 的 YOLO-World 权重文件。")
            return
        preset_names = {item[0] for item in MODEL_PRESETS}
        is_downloadable_preset = (
            model_path.name in preset_names
            and model_path.parent.resolve() == self.weights_dir.resolve()
        )
        if not model_path.exists() and not is_downloadable_preset:
            self._show_error(f"自定义模型权重不存在：{model_path}")
            return
        if not model_path.exists():
            self.status_label.setText(f"首次使用 {model_path.name}，准备下载官方权重…")
        self.save_classes()
        self.busy = True
        self.last_batch_preview_at = 0.0
        self.batch_processed = 0
        self.batch_box_count = 0
        self.batch_zero_count = 0
        self._set_busy(True)
        settings = self._settings()
        settings["paths"] = paths
        settings["model_path"] = model_path
        settings["prompts"] = prompts
        self.job_requested.emit(settings)

    def cancel_inference(self) -> None:
        if self.busy:
            self.engine.request_cancel()
            self.status_label.setText("正在等待当前图片推理结束后取消…")

    @Slot(object)
    def _on_inference_batch(self, results: list[tuple]) -> None:
        if self.project is None:
            return
        if not results:
            return
        for path, boxes, _image_size, index, _total in results:
            # The worker has already committed the txt.  Do not retain every
            # rich result in RAM for huge datasets; selecting the image later
            # reloads the canonical txt.  Also remove any stale old metadata.
            self.project.discard_annotation(path)
            self.batch_processed = index
            self.batch_box_count += len(boxes)
            if not boxes:
                self.batch_zero_count += 1
            item = self.image_items.get(path)
            if item is not None:
                item.setText(("∅ " if not boxes else "● ") + path.name)
                item.setToolTip(
                    f"{path}\n" + (
                        "已处理，未检测到目标（txt 为空）"
                        if not boxes else f"已保存 {len(boxes)} 个标注框"
                    )
                )
        path, boxes, _image_size, index, total = results[-1]
        now = monotonic()
        should_preview = total == 1 or index == total or now - self.last_batch_preview_at >= 0.75
        if should_preview:
            annotation = ImageAnnotation(AnnotationStatus.AUTO_LABELED, boxes)
            self.project.record_annotation(path, annotation)
            self.last_batch_preview_at = now
            self.current_path = path
            self.current_annotation = annotation
            self.canvas.load_image(path)
            self.canvas.set_boxes(boxes, self.review_spin.value())
            self.preview_title.setText(f"实时标注预览 — {path.name} — {len(boxes)} 个框")
            self._select_list_item(path)

        if boxes:
            message = f"GPU 已完成 {index}/{total}：{path.name}，txt 已写入 {len(boxes)} 行"
        else:
            message = f"GPU 已完成 {index}/{total}：{path.name}，未检测到目标，txt 为空"
        self.status_label.setText(message)

    def _inference_finished(self, cancelled: bool) -> None:
        if self.project is not None:
            try:
                self.project.save_metadata()
            except Exception as exc:
                self._show_error(f"标注 txt 已保存，但批量元数据保存失败：\n{exc}")
        self.busy = False
        self._set_busy(False)
        self.refresh_image_list()
        summary = (
            f"已处理 {self.batch_processed} 张，共写入 {self.batch_box_count} 个框；"
            f"{self.batch_zero_count} 张未检测到目标。"
        )
        self.status_label.setText(("批量任务已取消。" if cancelled else "GPU 自动标注完成。") + summary)
        if not cancelled and self.batch_processed > 0 and self.batch_box_count == 0:
            QMessageBox.warning(
                self,
                "未检测到任何目标",
                "这次推理没有生成标注框，所以 txt 为空；这不是保存失败。\n\n"
                f"当前提示词：{', '.join(self.prompts())}\n"
                f"置信度阈值：{self.confidence_spin.value():.2f}\n"
                f"推理尺寸：{self.imgsz_spin.value()}\n\n"
                "请换用更贴近图像外观的英文模型提示词，或逐步降低置信度阈值。"
            )
        if self.close_when_idle:
            self.close_when_idle = False
            QTimer.singleShot(0, self.close)

    def _set_busy(self, busy: bool) -> None:
        self.auto_current_button.setEnabled(not busy and torch.cuda.is_available())
        self.auto_all_button.setEnabled(not busy and torch.cuda.is_available())
        self.cancel_button.setEnabled(busy)
        self.image_list.setEnabled(not busy)
        self.model_path_edit.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.choose_model_button.setEnabled(not busy)
        self.classes_edit.setEnabled(not busy)
        self.prompts_edit.setEnabled(not busy)
        self.canvas.setEnabled(not busy)
        for action in self.busy_actions:
            action.setEnabled(not busy)
        for shortcut in self.shortcuts:
            shortcut.setEnabled(not busy)

    def _show_error(self, message: str) -> None:
        self.status_label.setText(message)
        QMessageBox.critical(self, "操作失败", message)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.busy:
            answer = QMessageBox.question(self, "正在推理", "GPU 任务仍在运行，是否取消并退出？")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.engine.request_cancel()
            self.close_when_idle = True
            event.ignore()
            return
        if self.project is not None:
            self.project.update_config(self._settings())
        self.worker_thread.quit()
        self.worker_thread.wait(5000)
        event.accept()
