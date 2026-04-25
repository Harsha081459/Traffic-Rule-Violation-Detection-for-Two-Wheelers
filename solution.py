"""
solution.py — Traffic Rule Violation Detection

Pipeline:
1) full_detector.pt / yolo11m.pt:
   Detects two-wheelers and riders on the full image.
2) helmet_detector.pt / yolo11s_helmet.pt:
   Detects helmet / no_helmet on expanded bike crops.
3) plate_detector.pt / yolo11n_plate.pt:
   Detects license plates for violating bikes.
4) pose_detector.pt / yolo11n-pose.pt  [OPTIONAL but STRONGLY RECOMMENDED]:
   Counts riders per bike via human keypoints/skeletons.
   Falls back to bounding-box rider counting if not present.
5) EasyOCR (models/easyocr/*.pth):
   Reads the full license plate string from plate crops.

──────────────────────────────────────────────────────────────────────────────
WHY USE A POSE MODEL FOR RIDER COUNT?
──────────────────────────────────────────────────────────────────────────────
When two riders sit close together, a standard person detector often merges
their overlapping bounding boxes into a single detection. A pose model (e.g.
yolo11n-pose, ~6 MB) identifies distinct skeletal keypoints per person, so it
can correctly count 2 riders even under heavy overlap or partial occlusion.

If you have the pose model weight, place it as:
    models/pose_detector.pt   or   models/yolo11n-pose.pt

The code auto-detects it. Without it, the system falls back to
max(rider bboxes, head detections), which is still reasonable for non-crowded
scenes.

──────────────────────────────────────────────────────────────────────────────
FULL PLATE STRING RECONSTRUCTION
──────────────────────────────────────────────────────────────────────────────
Indian license plates are often 2-line:
    Line 1: AP 07
    Line 2: AB 1234

EasyOCR returns separate text regions per line. We sort them top-to-bottom by
centroid Y and concatenate to recover the full plate string: "AP07AB1234".

──────────────────────────────────────────────────────────────────────────────
Evaluator interface:
    from solution import TrafficViolationDetector
    model = TrafficViolationDetector("./models")
    output = model.predict(image_path)

Strict output format:
{
    "violations": [
        {
            "num_riders": int,
            "helmet_violations": int,
            "license_plate": "string"   # e.g. "AP07AB1234"
        }
    ]
}
"""

from __future__ import annotations

import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import easyocr as _easyocr_lib
    EASYOCR_AVAILABLE = True
except ImportError:
    _easyocr_lib = None
    EASYOCR_AVAILABLE = False


# =============================================================================
# Geometry / detection helpers
# =============================================================================

@dataclass
class Det:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: List[float]

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _norm_name(x: str) -> str:
    return str(x).strip().lower().replace("-", "_").replace(" ", "_")


def _clip_box(box: List[float], w: int, h: int) -> List[int]:
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


def _expand_box(box: List[float], w: int, h: int, xpad: float, ypad: float) -> List[int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return _clip_box(
        [x1 - bw * xpad, y1 - bh * ypad, x2 + bw * xpad, y2 + bh * ypad],
        w, h,
    )


def _crop_box(
    img: np.ndarray,
    box: List[float],
    xpad: float = 0.0,
    ypad: float = 0.0,
) -> Tuple[np.ndarray, List[int]]:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = _expand_box(box, w, h, xpad, ypad)
    return img[y1:y2, x1:x2].copy(), [x1, y1, x2, y2]


def _inter_area(a: List[float], b: List[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a: List[float], b: List[float]) -> float:
    inter = _inter_area(a, b)
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1e-6, area_a + area_b - inter)


def _nms_same_class(dets: List[Det], iou_thr: float = 0.45) -> List[Det]:
    dets = sorted(dets, key=lambda d: d.conf, reverse=True)
    kept: List[Det] = []
    for d in dets:
        if not any(
            _norm_name(d.cls_name) == _norm_name(k.cls_name)
            and _iou(d.xyxy, k.xyxy) > iou_thr
            for k in kept
        ):
            kept.append(d)
    return kept


def _offset_det(det: Det, offset_box: List[int], new_name: Optional[str] = None) -> Det:
    ox, oy = offset_box[0], offset_box[1]
    x1, y1, x2, y2 = det.xyxy
    return Det(
        cls_id=det.cls_id,
        cls_name=new_name if new_name is not None else det.cls_name,
        conf=det.conf,
        xyxy=[x1 + ox, y1 + oy, x2 + ox, y2 + oy],
    )


# =============================================================================
# OCR helpers  (EasyOCR only — no PaddleOCR)
# =============================================================================

# Standard Indian plate regex: StateCode(2) + District(2) + Series(1-3) + Number(4)
# Examples: AP07AB1234, MH12EF5678, TS09U1234, KA03MX9901
_INDIAN_PLATE_RE = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$')


def _clean_plate_text(text: str) -> str:
    """
    Normalize raw OCR text to a plate-ready string.
    Keeps only A-Z and 0-9 (uppercase). Strips known noise tokens.
    """
    if text is None:
        return ""
    text = str(text).upper()
    text = re.sub(r'[^A-Z0-9]', '', text)
    # Noise from blue INDIA strip on Indian plates.
    for tok in ("INDIA", "IND", "BH"):
        if text.startswith(tok) and len(text) > len(tok) + 3:
            text = text[len(tok):]
    return text.strip()


def _plate_format_score(text: str) -> float:
    """
    Score a candidate plate string.
    Higher = more likely to be a real plate.
    """
    n = len(text)
    if n < 4:
        return 0.0
    score = min(n, 10) / 10.0          # length bonus (max at 10 chars)
    if _INDIAN_PLATE_RE.match(text):
        score += 1.5                    # perfect format match
    elif n >= 6:
        score += 0.5                    # reasonable length
    return score


def _deskew_gray(gray: np.ndarray) -> np.ndarray:
    """Detect and correct small rotations in a grayscale plate crop."""
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(
            edges, 1, np.pi / 180,
            threshold=max(20, gray.shape[1] // 4),
        )
        if lines is None:
            return gray
        angles = []
        for line in lines[:20]:
            theta = line[0][1]
            angle = (theta - np.pi / 2) * (180.0 / np.pi)
            if abs(angle) < 12:
                angles.append(angle)
        if not angles:
            return gray
        angle = float(np.median(angles))
        if abs(angle) < 0.5:
            return gray
        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except Exception:
        return gray


def _make_plate_variants(crop: np.ndarray) -> List[np.ndarray]:
    """
    Fast OCR preprocessing for <=5s inference.

    The old version made 7 variants, which causes EasyOCR to run many times.
    This version returns only 3 high-value variants:
      1. upscaled color crop
      2. sharpened grayscale crop
      3. Otsu binary crop
    """
    if crop is None or crop.size == 0:
        return []
    h, w = crop.shape[:2]
    if h < 3 or w < 3:
        return []

    crop = cv2.copyMakeBorder(
        crop,
        max(2, int(0.06 * h)), max(2, int(0.06 * h)),
        max(2, int(0.08 * w)), max(2, int(0.08 * w)),
        cv2.BORDER_REPLICATE,
    )

    h, w = crop.shape[:2]
    target_w = 320 if w < 160 else max(320, min(480, w * 2))
    scale = target_w / max(1, w)
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.6, blur, -0.6, 0)
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [
        resized,
        cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR),


#       2. sharpened grayscale crop
#       3. Otsu binary crop
#     """
#     if crop is None or crop.size == 0:
#         return []
#     h, w = crop.shape[:2]
#     if h < 3 or w < 3:
#         return []
# 
#     crop = cv2.copyMakeBorder(
#         crop,
#         max(2, int(0.06 * h)), max(2, int(0.06 * h)),
#         max(2, int(0.08 * w)), max(2, int(0.08 * w)),
#         cv2.BORDER_REPLICATE,
#     )
# 
#     h, w = crop.shape[:2]
#     target_w = 320 if w < 160 else max(320, min(480, w * 2))
#     scale = target_w / max(1, w)
#     resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
# 
#     gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
#     blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
#     sharp = cv2.addWeighted(clahe, 1.6, blur, -0.6, 0)
#     _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# 
#     return [
#         resized,
#         cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR),
