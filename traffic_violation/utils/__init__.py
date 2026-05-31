"""Shared utility sub-package: geometry primitives and plate text helpers."""

from traffic_violation.utils.geometry import (
    Box,
    Det,
    clip_box,
    crop_box,
    expand_box,
    inter_area,
    iou,
    nms_same_class,
    norm_name,
    offset_det,
)
from traffic_violation.utils.plate_utils import (
    INDIAN_PLATE_RE,
    clean_plate_text,
    deskew_gray,
    easyocr_regions_to_plate,
    make_plate_variants,
    plate_format_score,
    vote_plate,
)

__all__ = [
    "Box", "Det",
    "clip_box", "crop_box", "expand_box",
    "inter_area", "iou", "nms_same_class", "norm_name", "offset_det",
    "INDIAN_PLATE_RE", "clean_plate_text", "deskew_gray",
    "easyocr_regions_to_plate", "make_plate_variants",
    "plate_format_score", "vote_plate",
]
