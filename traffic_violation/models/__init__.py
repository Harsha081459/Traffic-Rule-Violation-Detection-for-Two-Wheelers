"""Model backends sub-package."""

from traffic_violation.models.base import DetectorProtocol
from traffic_violation.models.yolo_detector import YOLODetector

__all__ = ["DetectorProtocol", "YOLODetector"]
