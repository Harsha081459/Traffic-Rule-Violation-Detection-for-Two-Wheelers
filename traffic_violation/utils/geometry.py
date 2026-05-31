"""
traffic_violation.utils.geometry
=================================
Pure, side-effect-free geometry primitives used throughout the pipeline.

All functions operate on plain Python floats/lists so they are trivially
unit-testable without any model or image I/O dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Box = list[float]   # [x1, y1, x2, y2] in pixel coordinates


# ---------------------------------------------------------------------------
# Detection record
# ---------------------------------------------------------------------------

@dataclass
class Det:
    """A single bounding-box detection produced by any detector model.

    Attributes:
        cls_id:   Integer class index from the model's label map.
        cls_name: Human-readable class label (e.g. ``"helmet"``).
        conf:     Detection confidence score in ``[0, 1]``.
        xyxy:     Bounding box as ``[x1, y1, x2, y2]`` in pixel coords.
    """

    cls_id:   int
    cls_name: str
    conf:     float
    xyxy:     Box

    @property
    def area(self) -> float:
        """Axis-aligned area of the bounding box in square pixels."""
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def norm_name(x: str) -> str:
    """Normalise a class label to a lowercase, underscore-separated token.

    Args:
        x: Raw class name string.

    Returns:
        Normalised name, e.g. ``"No-Helmet"`` → ``"no_helmet"``.
    """
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Box clipping / expansion / cropping
# ---------------------------------------------------------------------------

def clip_box(box: Box, w: int, h: int) -> list[int]:
    """Clip a bounding box to image boundaries and ensure non-zero size.

    Args:
        box: ``[x1, y1, x2, y2]`` float coordinates.
        w:   Image width in pixels.
        h:   Image height in pixels.

    Returns:
        Integer ``[x1, y1, x2, y2]`` clamped to ``[0, w-1] × [0, h-1]``.
    """
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(0, min(w - 1, round(x2))))
    y2 = int(max(0, min(h - 1, round(y2))))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return [x1, y1, x2, y2]


def expand_box(box: Box, w: int, h: int, xpad: float, ypad: float) -> list[int]:
    """Expand a bounding box by a fraction of its own dimensions.

    Args:
        box:  ``[x1, y1, x2, y2]`` float coordinates.
        w:    Image width in pixels.
        h:    Image height in pixels.
        xpad: Fractional horizontal padding (e.g. ``0.3`` = 30 % of box width).
        ypad: Fractional vertical padding.

    Returns:
        Expanded and clipped integer box.
    """
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return clip_box(
        [x1 - bw * xpad, y1 - bh * ypad, x2 + bw * xpad, y2 + bh * ypad],
        w, h,
    )


def crop_box(
    img: np.ndarray,
    box: Box,
    xpad: float = 0.0,
    ypad: float = 0.0,
) -> tuple[np.ndarray, list[int]]:
    """Crop a region from *img* defined by *box*, with optional padding.

    Args:
        img:  BGR image array of shape ``(H, W, 3)``.
        box:  ``[x1, y1, x2, y2]`` float coordinates.
        xpad: Fractional horizontal padding applied before cropping.
        ypad: Fractional vertical padding applied before cropping.

    Returns:
        A tuple of ``(cropped_image, expanded_box)`` where ``expanded_box``
        gives the absolute pixel offsets used (needed to map detections back
        to the original coordinate space).
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = expand_box(box, w, h, xpad, ypad)
    return img[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Intersection / IoU
# ---------------------------------------------------------------------------

def inter_area(a: Box, b: Box) -> float:
    """Compute the intersection area of two axis-aligned boxes.

    Args:
        a: First box ``[x1, y1, x2, y2]``.
        b: Second box ``[x1, y1, x2, y2]``.

    Returns:
        Intersection area in square pixels (``0.0`` if no overlap).
    """
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Box, b: Box) -> float:
    """Compute Intersection-over-Union for two axis-aligned boxes.

    Args:
        a: First box ``[x1, y1, x2, y2]``.
        b: Second box ``[x1, y1, x2, y2]``.

    Returns:
        IoU score in ``[0, 1]``.
    """
    inter = inter_area(a, b)
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1e-6, area_a + area_b - inter)


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------

def nms_same_class(dets: list[Det], iou_thr: float = 0.45) -> list[Det]:
    """Greedy NMS that only suppresses detections of the *same* class.

    Boxes are processed in descending confidence order.  A candidate is
    suppressed if it overlaps any already-kept box of the same class by
    more than *iou_thr*.

    Args:
        dets:    List of :class:`Det` objects, possibly mixed classes.
        iou_thr: IoU threshold above which a lower-confidence box is removed.

    Returns:
        Filtered list of :class:`Det` objects.
    """
    dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    kept: list[Det] = []
    for d in dets:
        if not any(
            norm_name(d.cls_name) == norm_name(k.cls_name)
            and iou(d.xyxy, k.xyxy) > iou_thr
            for k in kept
        ):
            kept.append(d)
    return kept


# ---------------------------------------------------------------------------
# Coordinate offset helper
# ---------------------------------------------------------------------------

def offset_det(
    det: Det,
    offset_box: list[int],
    new_name: Optional[str] = None,
) -> Det:
    """Translate a detection's coordinates from crop-space to image-space.

    When a detection is produced on a sub-crop of the full image, its
    coordinates must be shifted by the crop's top-left corner before they
    can be used alongside full-image coordinates.

    Args:
        det:        The detection in crop-space coordinates.
        offset_box: ``[ox, oy, ...]`` where ``ox, oy`` are the pixel offsets
                    of the crop's top-left corner in the full image.
        new_name:   If supplied, replace ``det.cls_name`` with this string.

    Returns:
        A new :class:`Det` with coordinates in full-image space.
    """
    ox, oy = offset_box[0], offset_box[1]
    x1, y1, x2, y2 = det.xyxy
    return Det(
        cls_id=det.cls_id,
        cls_name=new_name if new_name is not None else det.cls_name,
        conf=det.conf,
        xyxy=[x1 + ox, y1 + oy, x2 + ox, y2 + oy],
    )
