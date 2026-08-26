from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)


class StatisticsDialog(QDialog):
    def __init__(self, statistics: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("数据集统计")
        self.resize(520, 520)
        layout = QVBoxLayout(self)
        summary = QGridLayout()
        entries = [
            ("图片总数", statistics["images"]),
            ("未标注", statistics["unlabeled"]),
            ("已自动标注", statistics["auto_labeled"]),
            ("已人工确认", statistics["verified"]),
            ("目标总数", statistics["objects"]),
            ("AUTO ACCEPT", statistics.get("auto_accept", 0)),
            ("REVIEW", statistics.get("review", 0)),
            ("REJECT", statistics.get("reject", 0)),
            ("模型一致率", f"{statistics.get('agreement_rate', 0.0) * 100:.1f}%"),
            ("人工修改", statistics.get("human_corrections", 0)),
            ("人工确认框", statistics.get("human_confirmed", 0)),
            ("SAHI 目标", statistics.get("sahi_objects", 0)),
            ("VLM 触发", statistics.get("vlm_triggered", 0)),
            ("VLM 不确定", statistics.get("vlm_uncertain", 0)),
        ]
        for index, (name, value) in enumerate(entries):
            label = QLabel(f"{name}\n{value}")
            label.setStyleSheet(
                "padding: 12px; border: 1px solid #777; border-radius: 6px; font-size: 15px"
            )
            summary.addWidget(label, index // 2, index % 2)
        layout.addLayout(summary)
        layout.addWidget(QLabel("类别目标数量"))
        table = QTableWidget(len(statistics["classes"]), 3)
        table.setHorizontalHeaderLabels(["类别", "目标数", "YOLO/SigLIP 一致率"])
        class_agreement = statistics.get("class_agreement", {})
        for row, (name, count) in enumerate(statistics["classes"].items()):
            table.setItem(row, 0, QTableWidgetItem(name))
            table.setItem(row, 1, QTableWidgetItem(str(count)))
            rate = class_agreement.get(name, {}).get("agreement_rate", 0.0)
            table.setItem(row, 2, QTableWidgetItem(f"{rate * 100:.1f}%"))
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(table, 1)
        if statistics["errors"]:
            error = QLabel(f"有 {len(statistics['errors'])} 张图片/标签读取失败，详见 logs/app.log")
            error.setStyleSheet("color: #ff6b6b")
            layout.addWidget(error)
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
