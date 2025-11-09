from typing import Tuple


def yolo_to_xyxy_norm(
    cx: float, cy: float, w: float, h: float
) -> Tuple[float, float, float, float]:
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return x1, y1, x2, y2
