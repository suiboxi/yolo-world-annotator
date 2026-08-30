from __future__ import annotations

from html import escape

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from yolo_world_annotator.utils.config import as_bool


class SettingsPanel(QWidget):
    load_model_requested = Signal()
    annotate_current_requested = Signal()
    annotate_all_requested = Signal()
    verify_requested = Signal()
    save_requested = Signal()
    export_requested = Signal()
    statistics_requested = Signal()
    evaluation_requested = Signal()
    benchmark_requested = Signal()
    save_classes_requested = Signal()
    load_classes_requested = Signal()
    selected_class_changed = Signal(int)
    create_box_requested = Signal()
    delete_box_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # This form is intentionally longer than a typical window.  Keep a
        # readable width and let the user scroll through advanced options.
        self.setMinimumWidth(410)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("settingsScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        content = QWidget()
        content.setObjectName("settingsContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 10, 14, 16)
        layout.setSpacing(10)

        title = QLabel("标注设置向导")
        title.setObjectName("settingsTitle")
        subtitle = QLabel(
            "按“模型 → 类别 → 参数 → 自动标注 → 保存导出”的顺序操作。\n"
            "高级设置可以继续向下滚动，鼠标悬停在控件上可查看详细解释。"
        )
        subtitle.setObjectName("helpText")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        model_group = QGroupBox("模型 · 第 1 步")
        model_layout = QVBoxLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.addItems(
            ["yolov8s-worldv2.pt", "yolov8m-worldv2.pt", "yolov8l-worldv2.pt"]
        )
        self.device_combo = QComboBox()
        self.device_combo.addItem("Auto（优先 CUDA，回退 CPU）", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("CUDA 0", "cuda:0")
        self.load_model_button = QPushButton("加载模型")
        self.load_model_button.clicked.connect(self.load_model_requested)
        self.device_label = QLabel("推理设备：检测中…")
        self.device_label.setWordWrap(True)
        self.model_combo.setToolTip(
            "选择 YOLO-World 模型。s 速度快、显存占用低；m/l 精度更高但更慢。"
        )
        self.load_model_button.setToolTip(
            "应用当前模型、设备和类别配置。"
        )
        self.device_combo.setToolTip("Auto 在 CUDA 可用时使用显卡，否则自动使用 CPU。")
        self.device_label.setToolTip("显示当前选择解析出的实际推理设备。")
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.device_combo)
        model_layout.addWidget(self.load_model_button)
        model_layout.addWidget(self.device_label)
        model_layout.addWidget(
            self._help_label(
                "说明：Auto 会优先使用 CUDA 显卡；不可用时自动回退到 CPU。"
            )
        )

        class_group = QGroupBox("检测类别 · 第 2 步（每行一个，顺序即 class id）")
        class_layout = QVBoxLayout(class_group)
        self.classes_edit = QTextEdit()
        self.classes_edit.setPlaceholderText("football player\nfootball\nreferee\ngoalkeeper")
        self.classes_edit.setMinimumHeight(112)
        self.classes_edit.setToolTip(
            "每行输入一个检测类别。行号就是永久 class id；已有标签后不要删除、重排或重命名旧类别。"
        )
        self.selected_class_combo = QComboBox()
        self.selected_class_combo.currentIndexChanged.connect(self.selected_class_changed)
        class_layout.addWidget(self.classes_edit)
        class_buttons = QHBoxLayout()
        self.save_classes_button = QPushButton("保存类别配置")
        self.load_classes_button = QPushButton("读取类别配置")
        self.save_classes_button.clicked.connect(self.save_classes_requested)
        self.load_classes_button.clicked.connect(self.load_classes_requested)
        class_buttons.addWidget(self.save_classes_button)
        class_buttons.addWidget(self.load_classes_button)
        class_layout.addLayout(class_buttons)
        class_layout.addWidget(QLabel("新建/当前框类别"))
        class_layout.addWidget(self.selected_class_combo)
        class_layout.addWidget(
            self._help_label("说明：类别越具体越好，例如“足球运动员”；类别顺序会写入 YOLO 标签。")
        )

        parameter_group = QGroupBox("基础参数 · 第 3 步")
        form = QFormLayout(parameter_group)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.01, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.25)
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.45)
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(256, 2048)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        self.review_spin = QDoubleSpinBox()
        self.review_spin.setRange(0.01, 1.0)
        self.review_spin.setSingleStep(0.05)
        self.review_spin.setValue(0.5)
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.1, 0.95)
        self.train_ratio_spin.setSingleStep(0.05)
        self.train_ratio_spin.setValue(0.8)
        self._configure_form(form)
        form.addRow(
            self._form_label(
                "置信度 Confidence",
                "低于此分数的检测框会被过滤。数值越低，召回率越高但误检可能增加。",
            ),
            self.confidence_spin,
        )
        form.addRow(
            self._form_label(
                "重叠阈值 IoU",
                "同一目标的重复框合并阈值。数值越高，保留的相邻框通常越多。",
            ),
            self.iou_spin,
        )
        form.addRow(
            self._form_label(
                "输入尺寸 Image Size",
                "模型推理缩放尺寸。小目标可提高尺寸；尺寸越大，显存和耗时越高。",
            ),
            self.imgsz_spin,
        )
        form.addRow(
            self._form_label(
                "审核阈值 Review",
                "融合结果低于此值会进入待审核，而不是直接自动接受。",
            ),
            self.review_spin,
        )
        form.addRow(
            self._form_label(
                "训练集比例 Train Ratio",
                "导出 YOLO 数据集时分给 train 的比例，其余图片进入 val。",
            ),
            self.train_ratio_spin,
        )
        layout.addWidget(
            self._help_label(
                "说明：第一次使用建议保持默认值。发现漏检时降低置信度或提高 Image Size；发现误检时提高置信度。"
            )
        )

        pipeline_group = QGroupBox("推理流水线 · 第 4 步")
        pipeline_form = QFormLayout(pipeline_group)
        self.inference_mode_combo = QComboBox()
        self.inference_mode_combo.addItem("YOLO Only（Normal）", "YOLO_ONLY")
        self.inference_mode_combo.addItem("YOLO + SAHI", "YOLO_SAHI")
        self.inference_mode_combo.addItem("YOLO + SigLIP", "YOLO_SIGLIP")
        self.inference_mode_combo.addItem("YOLO + SAHI + SigLIP", "YOLO_SAHI_SIGLIP")
        self.inference_mode_combo.addItem("YOLO + SAHI + SigLIP + VLM", "YOLO_SAHI_SIGLIP_VLM")
        self._configure_form(pipeline_form)
        pipeline_form.addRow(
            self._form_label(
                "推理模式",
                "选择需要的处理链。YOLO Only 最快；SAHI 适合小目标；SigLIP/VLM 会增加语义复核。",
            ),
            self.inference_mode_combo,
        )

        # Keep the SigLIP2 controls out of the way for the normal YOLO-only
        # workflow.  The values are still part of config_values/load_config so
        # enabling verification later is fully persistent.
        self.siglip_toggle = QPushButton("▶ SigLIP2 二次验证（高级，可选）")
        self.siglip_toggle.setCheckable(True)
        self.siglip_toggle.setChecked(False)
        self.siglip_toggle.toggled.connect(self._toggle_siglip_advanced)
        self.siglip_advanced = QWidget()
        siglip_form = QFormLayout(self.siglip_advanced)
        self._configure_form(siglip_form)
        self.enable_siglip_check = QCheckBox("启用 SigLIP2 二次验证")
        self.enable_siglip_check.setChecked(False)
        self.siglip_model_edit = QLineEdit("google/siglip2-base-patch16-224")
        self.siglip_padding_spin = QDoubleSpinBox()
        self.siglip_padding_spin.setRange(0.0, 50.0)
        self.siglip_padding_spin.setSingleStep(5.0)
        self.siglip_padding_spin.setSuffix(" %")
        self.siglip_padding_spin.setValue(10.0)
        self.yolo_weight_spin = QDoubleSpinBox()
        self.yolo_weight_spin.setRange(0.0, 1.0)
        self.yolo_weight_spin.setSingleStep(0.05)
        self.yolo_weight_spin.setValue(0.65)
        self.siglip_weight_spin = QDoubleSpinBox()
        self.siglip_weight_spin.setRange(0.0, 1.0)
        self.siglip_weight_spin.setSingleStep(0.05)
        self.siglip_weight_spin.setValue(0.35)
        self.auto_accept_spin = QDoubleSpinBox()
        self.auto_accept_spin.setRange(0.01, 1.0)
        self.auto_accept_spin.setSingleStep(0.05)
        self.auto_accept_spin.setValue(0.75)
        self.siglip_batch_spin = QSpinBox()
        self.siglip_batch_spin.setRange(1, 32)
        self.siglip_batch_spin.setValue(4)
        self.siglip_precision_combo = QComboBox()
        self.siglip_precision_combo.addItem("Auto（CUDA FP16）", "auto")
        self.siglip_precision_combo.addItem("FP16", "fp16")
        self.siglip_precision_combo.addItem("BF16（需硬件支持）", "bf16")
        self.candidate_topk_spin = QSpinBox()
        self.candidate_topk_spin.setRange(0, 20)
        self.candidate_topk_spin.setSpecialValueText("全部类别")
        self.candidate_topk_spin.setValue(0)
        self.siglip_prompt_edit = QLineEdit("a photo of a {}")
        self.siglip_prompt_ensemble_check = QCheckBox("启用 Prompt Ensemble（较慢）")
        self.per_class_thresholds_edit = QTextEdit()
        self.per_class_thresholds_edit.setPlaceholderText("每行一个：person=0.35\ncapacitor=0.22")
        self.per_class_thresholds_edit.setMaximumHeight(70)
        siglip_form.addRow(self.enable_siglip_check)
        siglip_form.addRow(
            self._form_label("模型 Model", "SigLIP2 图文复核模型；首次启用时可能从 Hugging Face 下载。"),
            self.siglip_model_edit,
        )
        siglip_form.addRow(
            self._form_label("裁剪边缘 Crop Padding", "在检测框四周额外保留的边缘比例，帮助模型看到上下文。"),
            self.siglip_padding_spin,
        )
        siglip_form.addRow(
            self._form_label("YOLO 权重", "融合时 YOLO-World 结果的权重，越大越信任定位模型。"),
            self.yolo_weight_spin,
        )
        siglip_form.addRow(
            self._form_label("SigLIP 权重", "融合时 SigLIP2 语义结果的权重，越大越信任图文匹配。"),
            self.siglip_weight_spin,
        )
        siglip_form.addRow(
            self._form_label("自动接受阈值", "融合分数达到此值且模型一致时，框可自动标记为 AUTO_ACCEPT。"),
            self.auto_accept_spin,
        )
        siglip_form.addRow(
            self._form_label("批大小 Batch Size", "一次送入 SigLIP2 的框数量。显存不足时调小到 1–2。"),
            self.siglip_batch_spin,
        )
        siglip_form.addRow(
            self._form_label("精度 Precision", "Auto 会自动选择；FP16 节省显存；BF16 需要硬件支持。"),
            self.siglip_precision_combo,
        )
        siglip_form.addRow(
            self._form_label("候选 Top-K", "只在候选类别中复核；0 表示使用全部类别。"),
            self.candidate_topk_spin,
        )
        siglip_form.addRow(
            self._form_label("提示词模板 Prompt", "{} 会替换成类别名，例如“a photo of a football”。"),
            self.siglip_prompt_edit,
        )
        siglip_form.addRow(self.siglip_prompt_ensemble_check)
        siglip_form.addRow(
            self._form_label("类别阈值", "可按类别覆盖审核阈值，每行一个，例如 person=0.35。"),
            self.per_class_thresholds_edit,
        )
        self._set_help(self.enable_siglip_check, "启用 SigLIP2 对每个检测框进行图文语义复核。")
        self._set_help(self.siglip_prompt_ensemble_check, "使用多组提示词平均结果，通常更稳但速度更慢。")
        self.siglip_advanced.setVisible(False)

        self.sahi_group = QGroupBox("SAHI / Tiled Inference")
        self.sahi_group.setCheckable(True)
        self.sahi_group.setChecked(False)
        sahi_form = QFormLayout(self.sahi_group)
        self._configure_form(sahi_form)
        self.sahi_slice_width_spin = QSpinBox()
        self.sahi_slice_width_spin.setRange(128, 4096)
        self.sahi_slice_width_spin.setSingleStep(64)
        self.sahi_slice_width_spin.setValue(1024)
        self.sahi_slice_height_spin = QSpinBox()
        self.sahi_slice_height_spin.setRange(128, 4096)
        self.sahi_slice_height_spin.setSingleStep(64)
        self.sahi_slice_height_spin.setValue(1024)
        self.sahi_overlap_width_spin = QDoubleSpinBox()
        self.sahi_overlap_width_spin.setRange(0.0, 95.0)
        self.sahi_overlap_width_spin.setSuffix(" %")
        self.sahi_overlap_width_spin.setValue(20.0)
        self.sahi_overlap_height_spin = QDoubleSpinBox()
        self.sahi_overlap_height_spin.setRange(0.0, 95.0)
        self.sahi_overlap_height_spin.setSuffix(" %")
        self.sahi_overlap_height_spin.setValue(20.0)
        self.sahi_postprocess_combo = QComboBox()
        self.sahi_postprocess_combo.addItems(["NMS", "WBF", "None"])
        self.sahi_merge_threshold_spin = QDoubleSpinBox()
        self.sahi_merge_threshold_spin.setRange(0.0, 1.0)
        self.sahi_merge_threshold_spin.setSingleStep(0.05)
        self.sahi_merge_threshold_spin.setValue(0.50)
        self.sahi_merge_metric_combo = QComboBox()
        self.sahi_merge_metric_combo.addItems(["IOU", "IOS"])
        self.sahi_max_tiles_spin = QSpinBox()
        self.sahi_max_tiles_spin.setRange(0, 1000)
        self.sahi_max_tiles_spin.setSpecialValueText("不限制")
        sahi_form.addRow(self._form_label("切片宽度", "把大图切成多个小图的宽度，建议 768–1280。"), self.sahi_slice_width_spin)
        sahi_form.addRow(self._form_label("切片高度", "把大图切成多个小图的高度，建议与切片宽度接近。"), self.sahi_slice_height_spin)
        sahi_form.addRow(self._form_label("水平重叠", "相邻切片水平方向的重叠比例，避免目标刚好落在边缘。"), self.sahi_overlap_width_spin)
        sahi_form.addRow(self._form_label("垂直重叠", "相邻切片垂直方向的重叠比例，通常 15%–25% 即可。"), self.sahi_overlap_height_spin)
        sahi_form.addRow(self._form_label("合并方式", "将不同切片的重复框合并；NMS 更快，WBF 更平滑。"), self.sahi_postprocess_combo)
        sahi_form.addRow(self._form_label("合并阈值", "两个切片框重叠达到此值时认为是同一目标。"), self.sahi_merge_threshold_spin)
        sahi_form.addRow(self._form_label("重叠度量", "IoU 使用交并比；IoS 对被另一个框覆盖的情况更敏感。"), self.sahi_merge_metric_combo)
        sahi_form.addRow(self._form_label("最大切片数", "限制单张图片最多切多少块；0 表示不限制。"), self.sahi_max_tiles_spin)
        sahi_form.addRow(self._help_label("说明：SAHI 适合远处、小尺寸目标；普通图片建议关闭，可明显节省时间。"))

        self.vlm_group = QGroupBox("Qwen3-VL 困难样本裁判")
        self.vlm_group.setCheckable(True)
        self.vlm_group.setChecked(False)
        vlm_form = QFormLayout(self.vlm_group)
        self._configure_form(vlm_form)
        self.vlm_model_edit = QLineEdit("Qwen/Qwen3-VL-8B-Instruct")
        self.vlm_lazy_check = QCheckBox("Lazy Load（推荐）")
        self.vlm_lazy_check.setChecked(True)
        self.vlm_low_memory_check = QCheckBox("低显存 / FP16")
        self.vlm_low_memory_check.setChecked(True)
        self.vlm_tokens_spin = QSpinBox()
        self.vlm_tokens_spin.setRange(16, 1024)
        self.vlm_tokens_spin.setValue(128)
        self.vlm_yolo_threshold_spin = QDoubleSpinBox()
        self.vlm_yolo_threshold_spin.setRange(0.0, 1.0)
        self.vlm_yolo_threshold_spin.setValue(0.45)
        self.vlm_siglip_threshold_spin = QDoubleSpinBox()
        self.vlm_siglip_threshold_spin.setRange(0.0, 1.0)
        self.vlm_siglip_threshold_spin.setValue(0.55)
        self.vlm_margin_spin = QDoubleSpinBox()
        self.vlm_margin_spin.setRange(0.0, 1.0)
        self.vlm_margin_spin.setValue(0.10)
        self.vlm_special_check = QCheckBox("特殊类别强制验证")
        self.vlm_special_check.setChecked(True)
        self.vlm_each_image_check = QCheckBox("每张图 YOLO 后强制逐框 Qwen 检查")
        self.vlm_each_image_check.setChecked(False)
        vlm_form.addRow(self._form_label("模型 Model", "困难样本裁判模型，体积较大；建议先用 YOLO/SigLIP 跑通流程。"), self.vlm_model_edit)
        vlm_form.addRow(self.vlm_lazy_check)
        vlm_form.addRow(self.vlm_low_memory_check)
        vlm_form.addRow(self._form_label("最大输出 Token", "限制 VLM 每次回答长度，数值越大越慢、占用显存越多。"), self.vlm_tokens_spin)
        vlm_form.addRow(self._form_label("YOLO 低置信度", "YOLO 分数低于此值时触发困难样本复核。"), self.vlm_yolo_threshold_spin)
        vlm_form.addRow(self._form_label("SigLIP 低置信度", "SigLIP 分数低于此值时触发困难样本复核。"), self.vlm_siglip_threshold_spin)
        vlm_form.addRow(self._form_label("Margin 阈值", "Top1 与 Top2 分数差小于此值时认为类别不确定。"), self.vlm_margin_spin)
        vlm_form.addRow(self.vlm_special_check)
        vlm_form.addRow(self.vlm_each_image_check)
        self._set_help(self.vlm_lazy_check, "懒加载：只有遇到困难样本时才加载 VLM，推荐保持开启。")
        self._set_help(self.vlm_low_memory_check, "低显存模式：使用 FP16 和更节省显存的加载方式。")
        self._set_help(self.vlm_special_check, "对配置为特殊类别的 profile 强制调用 VLM。")
        self._set_help(
            self.vlm_each_image_check,
            "开启后，每张图片完成 YOLO 后，对该图检测出的每个框都调用一次 Qwen；"
            "未开启时仍按困难样本策略触发。",
        )
        vlm_form.addRow(
            self._help_label(
                "说明：VLM 是可选高级功能，会下载并加载大型模型；RTX 4060 建议保持懒加载和低显存模式。"
            )
        )
        self.inference_mode_combo.currentIndexChanged.connect(self._mode_changed)

        self.verification_detail = QLabel("选中标注框后显示 YOLO / SigLIP2 验证详情")
        self.verification_detail.setWordWrap(True)
        self.verification_detail.setStyleSheet(
            "padding: 8px; border: 1px solid #555; border-radius: 5px; color: #d7e3f4"
        )

        action_group = QGroupBox("操作")
        action_layout = QVBoxLayout(action_group)
        self.annotate_current_button = QPushButton("自动标注当前图片")
        self.annotate_all_button = QPushButton("自动标注全部图片")
        self.create_box_button = QPushButton("在预览图上拖拽新框（B）")
        self.delete_box_button = QPushButton("删除选中框（Delete）")
        self.existing_mode_combo = QComboBox()
        self.existing_mode_combo.addItem("仅处理未标注图片（默认）", "ONLY_UNLABELED")
        self.existing_mode_combo.addItem("跳过已有 labels/*.txt", "SKIP_EXISTING")
        self.existing_mode_combo.addItem("覆盖已有标签", "OVERWRITE")
        self.pause_button = QPushButton("暂停")
        self.resume_button = QPushButton("继续")
        self.cancel_button = QPushButton("取消")
        self.pause_button.setEnabled(False)
        self.resume_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.verify_button = QPushButton("确认当前图片")
        self.save_button = QPushButton("保存")
        self.export_button = QPushButton("导出 YOLO Dataset")
        self.statistics_button = QPushButton("数据集统计")
        self.evaluation_button = QPushButton("YOLO / SigLIP2 A/B 评估")
        self.benchmark_button = QPushButton("A/B/C/D Pipeline Benchmark")
        self.annotate_current_button.clicked.connect(self.annotate_current_requested)
        self.annotate_all_button.clicked.connect(self.annotate_all_requested)
        self.verify_button.clicked.connect(self.verify_requested)
        self.save_button.clicked.connect(self.save_requested)
        self.export_button.clicked.connect(self.export_requested)
        self.statistics_button.clicked.connect(self.statistics_requested)
        self.evaluation_button.clicked.connect(self.evaluation_requested)
        self.benchmark_button.clicked.connect(self.benchmark_requested)
        self.create_box_button.clicked.connect(self.create_box_requested)
        self.delete_box_button.clicked.connect(self.delete_box_requested)
        action_layout.addWidget(self.annotate_current_button)
        action_layout.addWidget(self.annotate_all_button)
        action_layout.addWidget(self.create_box_button)
        action_layout.addWidget(self.delete_box_button)
        action_layout.addWidget(self.existing_mode_combo)
        batch_controls = QHBoxLayout()
        batch_controls.addWidget(self.pause_button)
        batch_controls.addWidget(self.resume_button)
        batch_controls.addWidget(self.cancel_button)
        action_layout.addLayout(batch_controls)
        action_layout.addWidget(self.verify_button)
        buttons = QHBoxLayout()
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.export_button)
        action_layout.addLayout(buttons)
        action_layout.addWidget(self.statistics_button)
        action_layout.addWidget(self.evaluation_button)
        action_layout.addWidget(self.benchmark_button)
        action_layout.addWidget(
            self._help_label(
                "说明：先打开数据集（会自动加载模型），再检查预览框；自动标注后人工调整，最后保存或导出。"
                "批量任务可暂停、继续或取消。"
            )
        )

        self.busy_label = QLabel("")
        self.busy_label.setWordWrap(True)
        layout.addWidget(model_group)
        layout.addWidget(class_group)
        layout.addWidget(parameter_group)
        layout.addWidget(pipeline_group)
        layout.addWidget(self.sahi_group)
        layout.addWidget(self.siglip_toggle)
        layout.addWidget(self.siglip_advanced)
        layout.addWidget(self.vlm_group)
        layout.addWidget(self.verification_detail)
        layout.addWidget(action_group)
        layout.addWidget(self.busy_label)
        layout.addStretch(1)
        self.scroll_area.setWidget(content)
        outer_layout.addWidget(self.scroll_area)

    @staticmethod
    def _help_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("helpText")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    @staticmethod
    def _form_label(title: str, explanation: str) -> QLabel:
        label = QLabel(title)
        label.setText(
            f"<b>{escape(title)}</b><br>"
            f"<span style='color:#9eafc0; font-size:11px'>{escape(explanation)}</span>"
        )
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        label.setMinimumWidth(165)
        label.setToolTip(explanation)
        label.setWhatsThis(explanation)
        return label

    @staticmethod
    def _set_help(widget: QWidget, explanation: str) -> None:
        widget.setToolTip(explanation)
        widget.setWhatsThis(explanation)

    @staticmethod
    def _configure_form(form: QFormLayout) -> None:
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)

    def classes(self) -> list[str]:
        return [line.strip() for line in self.classes_edit.toPlainText().splitlines() if line.strip()]

    def set_classes(self, classes: list[str]) -> None:
        self.classes_edit.setPlainText("\n".join(classes))
        self.selected_class_combo.blockSignals(True)
        self.selected_class_combo.clear()
        for class_id, name in enumerate(classes):
            self.selected_class_combo.addItem(f"{class_id}  {name}", class_id)
        self.selected_class_combo.blockSignals(False)

    def config_values(self) -> dict:
        per_class_thresholds: dict[str, float] = {}
        for line in self.per_class_thresholds_edit.toPlainText().splitlines():
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            try:
                threshold = min(1.0, max(0.01, float(value.strip())))
            except ValueError:
                continue
            if name.strip():
                per_class_thresholds[name.strip()] = threshold
        return {
            "model": self.model_combo.currentText(),
            "device": str(self.device_combo.currentData() or "auto"),
            "classes": self.classes(),
            "confidence": self.confidence_spin.value(),
            "iou": self.iou_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "review_threshold": self.review_spin.value(),
            "train_ratio": self.train_ratio_spin.value(),
            "siglip_enabled": self.enable_siglip_check.isChecked(),
            "siglip_model": self.siglip_model_edit.text().strip()
            or "google/siglip2-base-patch16-224",
            "siglip_padding": self.siglip_padding_spin.value() / 100.0,
            "yolo_weight": self.yolo_weight_spin.value(),
            "siglip_weight": self.siglip_weight_spin.value(),
            "auto_accept_threshold": self.auto_accept_spin.value(),
            "siglip_batch_size": self.siglip_batch_spin.value(),
            "siglip_precision": str(self.siglip_precision_combo.currentData() or "auto"),
            "candidate_top_k": self.candidate_topk_spin.value(),
            "siglip_prompt_template": self.siglip_prompt_edit.text().strip()
            or "a photo of a {}",
            "siglip_prompt_ensemble": self.siglip_prompt_ensemble_check.isChecked(),
            "per_class_thresholds": per_class_thresholds,
            "inference_mode": str(self.inference_mode_combo.currentData() or "YOLO_ONLY"),
            "sahi_enabled": self.sahi_group.isChecked(),
            "sahi_slice_width": self.sahi_slice_width_spin.value(),
            "sahi_slice_height": self.sahi_slice_height_spin.value(),
            "sahi_overlap_width_ratio": self.sahi_overlap_width_spin.value() / 100.0,
            "sahi_overlap_height_ratio": self.sahi_overlap_height_spin.value() / 100.0,
            "sahi_postprocess_type": self.sahi_postprocess_combo.currentText(),
            "sahi_postprocess_match_threshold": self.sahi_merge_threshold_spin.value(),
            "sahi_postprocess_match_metric": self.sahi_merge_metric_combo.currentText(),
            "sahi_max_tiles": self.sahi_max_tiles_spin.value(),
            "vlm_enabled": self.vlm_group.isChecked(),
            "vlm_model": self.vlm_model_edit.text().strip() or "Qwen/Qwen3-VL-8B-Instruct",
            "vlm_lazy_load": self.vlm_lazy_check.isChecked(),
            "vlm_low_memory": self.vlm_low_memory_check.isChecked(),
            "vlm_max_new_tokens": self.vlm_tokens_spin.value(),
            "vlm_yolo_low_threshold": self.vlm_yolo_threshold_spin.value(),
            "vlm_siglip_low_threshold": self.vlm_siglip_threshold_spin.value(),
            "vlm_margin_threshold": self.vlm_margin_spin.value(),
            "vlm_force_special_classes": self.vlm_special_check.isChecked(),
            "vlm_check_each_image": self.vlm_each_image_check.isChecked(),
        }

    def load_config(self, config: dict) -> None:
        model = str(config.get("model", "yolov8s-worldv2.pt"))
        index = self.model_combo.findText(model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        device = str(config.get("device", "auto"))
        device_index = self.device_combo.findData(device)
        if device_index >= 0:
            self.device_combo.setCurrentIndex(device_index)
        self.set_classes(list(config.get("classes", [])))
        self.confidence_spin.setValue(float(config.get("confidence", 0.25)))
        self.iou_spin.setValue(float(config.get("iou", 0.45)))
        self.imgsz_spin.setValue(int(config.get("imgsz", 640)))
        self.review_spin.setValue(float(config.get("review_threshold", 0.5)))
        self.train_ratio_spin.setValue(float(config.get("train_ratio", 0.8)))
        mode = str(config.get("inference_mode", "YOLO_ONLY"))
        mode_index = self.inference_mode_combo.findData(mode)
        if mode_index >= 0:
            self.inference_mode_combo.setCurrentIndex(mode_index)
        self.enable_siglip_check.setChecked(as_bool(config.get("siglip_enabled", False)))
        self.siglip_model_edit.setText(
            str(config.get("siglip_model", "google/siglip2-base-patch16-224"))
        )
        self.siglip_padding_spin.setValue(float(config.get("siglip_padding", 0.10)) * 100.0)
        self.yolo_weight_spin.setValue(float(config.get("yolo_weight", 0.65)))
        self.siglip_weight_spin.setValue(float(config.get("siglip_weight", 0.35)))
        self.auto_accept_spin.setValue(float(config.get("auto_accept_threshold", 0.75)))
        self.siglip_batch_spin.setValue(int(config.get("siglip_batch_size", 4)))
        precision = str(config.get("siglip_precision", "auto"))
        precision_index = self.siglip_precision_combo.findData(precision)
        if precision_index >= 0:
            self.siglip_precision_combo.setCurrentIndex(precision_index)
        self.candidate_topk_spin.setValue(int(config.get("candidate_top_k", 0)))
        self.siglip_prompt_edit.setText(
            str(config.get("siglip_prompt_template", "a photo of a {}"))
        )
        self.siglip_prompt_ensemble_check.setChecked(
            as_bool(config.get("siglip_prompt_ensemble", False))
        )
        thresholds = config.get("per_class_thresholds", {})
        if isinstance(thresholds, dict):
            self.per_class_thresholds_edit.setPlainText(
                "\n".join(f"{name}={float(value):.3f}" for name, value in thresholds.items())
            )
        else:
            self.per_class_thresholds_edit.clear()
        self.sahi_group.setChecked(as_bool(config.get("sahi_enabled", "SAHI" in mode)))
        self.sahi_slice_width_spin.setValue(int(config.get("sahi_slice_width", 1024)))
        self.sahi_slice_height_spin.setValue(int(config.get("sahi_slice_height", 1024)))
        self.sahi_overlap_width_spin.setValue(float(config.get("sahi_overlap_width_ratio", 0.20)) * 100.0)
        self.sahi_overlap_height_spin.setValue(float(config.get("sahi_overlap_height_ratio", 0.20)) * 100.0)
        postprocess = str(config.get("sahi_postprocess_type", "NMS"))
        postprocess_index = self.sahi_postprocess_combo.findText(postprocess)
        if postprocess_index >= 0:
            self.sahi_postprocess_combo.setCurrentIndex(postprocess_index)
        self.sahi_merge_threshold_spin.setValue(float(config.get("sahi_postprocess_match_threshold", 0.50)))
        metric_index = self.sahi_merge_metric_combo.findText(str(config.get("sahi_postprocess_match_metric", "IOU")))
        if metric_index >= 0:
            self.sahi_merge_metric_combo.setCurrentIndex(metric_index)
        self.sahi_max_tiles_spin.setValue(int(config.get("sahi_max_tiles", 0)))
        self.vlm_group.setChecked(as_bool(config.get("vlm_enabled", "VLM" in mode)))
        self.vlm_model_edit.setText(str(config.get("vlm_model", "Qwen/Qwen3-VL-8B-Instruct")))
        self.vlm_lazy_check.setChecked(as_bool(config.get("vlm_lazy_load", True), True))
        self.vlm_low_memory_check.setChecked(as_bool(config.get("vlm_low_memory", True), True))
        self.vlm_tokens_spin.setValue(int(config.get("vlm_max_new_tokens", 128)))
        self.vlm_yolo_threshold_spin.setValue(float(config.get("vlm_yolo_low_threshold", 0.45)))
        self.vlm_siglip_threshold_spin.setValue(float(config.get("vlm_siglip_low_threshold", 0.55)))
        self.vlm_margin_spin.setValue(float(config.get("vlm_margin_threshold", 0.10)))
        self.vlm_special_check.setChecked(as_bool(config.get("vlm_force_special_classes", True), True))
        self.vlm_each_image_check.setChecked(as_bool(config.get("vlm_check_each_image", False)))
        should_expand = self.enable_siglip_check.isChecked()
        if self.siglip_toggle.isChecked() != should_expand:
            self.siglip_toggle.setChecked(should_expand)

    def _toggle_siglip_advanced(self, expanded: bool) -> None:
        self.siglip_advanced.setVisible(expanded)
        self.siglip_toggle.setText(
            "▼ SigLIP2 二次验证（高级，可选）" if expanded else "▶ SigLIP2 二次验证（高级，可选）"
        )

    def _mode_changed(self, _index: int) -> None:
        mode = str(self.inference_mode_combo.currentData() or "YOLO_ONLY")
        self.sahi_group.setChecked("SAHI" in mode)
        self.enable_siglip_check.setChecked("SIGLIP" in mode)
        self.vlm_group.setChecked("VLM" in mode)

    def set_verification_detail(self, text: str) -> None:
        self.verification_detail.setText(text or "选中标注框后显示 YOLO / SigLIP2 验证详情")

    def set_busy(self, busy: bool, message: str = "") -> None:
        self.model_combo.setEnabled(not busy)
        self.device_combo.setEnabled(not busy)
        self.load_model_button.setEnabled(not busy)
        self.annotate_current_button.setEnabled(not busy)
        self.annotate_all_button.setEnabled(not busy)
        self.benchmark_button.setEnabled(not busy)
        self.busy_label.setText(message)

    def set_batch_running(self, running: bool) -> None:
        self.model_combo.setEnabled(not running)
        self.device_combo.setEnabled(not running)
        self.load_model_button.setEnabled(not running)
        self.annotate_current_button.setEnabled(not running)
        self.annotate_all_button.setEnabled(not running)
        self.benchmark_button.setEnabled(not running)
        self.existing_mode_combo.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.resume_button.setEnabled(False)
        self.cancel_button.setEnabled(running)
        self.save_button.setEnabled(not running)
        self.export_button.setEnabled(not running)
        self.statistics_button.setEnabled(not running)
        self.evaluation_button.setEnabled(not running)
        self.benchmark_button.setEnabled(not running)
        self.save_classes_button.setEnabled(not running)
        self.load_classes_button.setEnabled(not running)
