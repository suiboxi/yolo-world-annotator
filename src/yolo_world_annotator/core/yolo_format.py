from __future__ import annotations

from pathlib import Path

from yolo_world_annotator.core.annotation import BoundingBox


def xyxy_to_yolo(box: BoundingBox, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("图片尺寸必须大于 0")
    clean = box.normalized(image_width, image_height)
    return (
        (clean.x1 + clean.x2) / 2.0 / image_width,
        (clean.y1 + clean.y2) / 2.0 / image_height,
        (clean.x2 - clean.x1) / image_width,
        (clean.y2 - clean.y1) / image_height,
    )


def yolo_to_xyxy(
    class_id: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
    class_name: str,
) -> BoundingBox:
    values = (x_center, y_center, width, height)
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("YOLO 坐标必须位于 0~1")
    if width <= 0 or height <= 0:
        raise ValueError("YOLO 标注框宽高必须大于 0")
    pixel_width = width * image_width
    pixel_height = height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height
    return BoundingBox(
        class_id=class_id,
        class_name=class_name,
        x1=center_x - pixel_width / 2,
        y1=center_y - pixel_height / 2,
        x2=center_x + pixel_width / 2,
        y2=center_y + pixel_height / 2,
        confidence=None,
        source="MANUAL",
    ).normalized(image_width, image_height)


def serialize_yolo(boxes: list[BoundingBox], image_width: int, image_height: int) -> str:
    lines: list[str] = []
    for box in boxes:
        x_center, y_center, width, height = xyxy_to_yolo(box, image_width, image_height)
        lines.append(
            f"{box.class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def parse_yolo(
    text: str, image_width: int, image_height: int, classes: list[str]
) -> list[BoundingBox]:
    boxes: list[BoundingBox] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"第 {line_number} 行应包含 5 列")
        try:
            class_id = int(parts[0])
            coords = [float(value) for value in parts[1:]]
        except ValueError as exc:
            raise ValueError(f"第 {line_number} 行包含非数字内容") from exc
        if not 0 <= class_id < len(classes):
            raise ValueError(f"第 {line_number} 行类别 ID {class_id} 超出 classes.txt 范围")
        boxes.append(
            yolo_to_xyxy(
                class_id,
                *coords,
                image_width,
                image_height,
                classes[class_id],
            )
        )
    return boxes


def load_yolo(path: Path, image_width: int, image_height: int, classes: list[str]) -> list[BoundingBox]:
    return parse_yolo(path.read_text(encoding="utf-8-sig"), image_width, image_height, classes)

