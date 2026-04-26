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
        cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
    ]


def _easyocr_regions_to_plate(results: list) -> Tuple[str, float]:
    """
    Convert EasyOCR readtext output to a single plate string.

    EasyOCR with paragraph=False returns separate text regions, one per line.
    For a 2-line Indian plate:
        Line 1 (top):    "AP 07"
        Line 2 (bottom): "AB 1234"

    We sort regions by their centroid Y (top → bottom) and concatenate all
    cleaned parts to get the full plate: "AP07AB1234".

    Args:
        results: list of (bbox_points, text, confidence) from easyocr.readtext
    Returns:
        (full_plate_string, mean_confidence)
    """
    if not results:
        return "", 0.0

    def _mean_y(item) -> float:
        pts = item[0]  # list of [x, y] corner points
        return sum(p[1] for p in pts) / max(1, len(pts))

    sorted_results = sorted(results, key=_mean_y)  # top → bottom

    parts = [_clean_plate_text(item[1]) for item in sorted_results]
    confs = [float(item[2]) for item in sorted_results]

    full_text = "".join(p for p in parts if p)
    avg_conf = float(np.mean(confs)) if confs else 0.0

    return full_text, avg_conf


def _vote_plate(candidates: List[Tuple[str, float]]) -> str:
    """
    Pick the best plate string from a list of (text, confidence) candidates.

    Strategy:
    1. Score each candidate by plate format score × confidence.
    2. Among same-length candidates, do character-level voting to fix
       single-character misreads across variants.
    """
    if not candidates:
        return ""

    pool: List[Tuple[str, float]] = []
    for t, c in candidates:
        t = _clean_plate_text(t)
        if len(t) < 4:
            continue
        try:
            c = float(c)
        except Exception:
            c = 0.0
        # Hard cap to avoid junk strings from OCR noise.
        pool.append((t[:12], c))

    if not pool:
        return ""

    # Weighted scoring.
    scored: Dict[str, float] = {}
    for t, c in pool:
        s = _plate_format_score(t) * max(c, 0.05)
        scored[t] = scored.get(t, 0.0) + s

    best_text = max(scored.items(), key=lambda kv: kv[1])[0]

    # Character-level voting among same-length candidates.
    same_len = [(t, c) for t, c in pool if len(t) == len(best_text)]
    if len(same_len) < 2:
        return best_text

    out = []
    for i in range(len(best_text)):
        votes: Dict[str, float] = {}
        for text, conf in same_len:
            ch = text[i]
            votes[ch] = votes.get(ch, 0.0) + max(conf, 0.01)
        out.append(max(votes.items(), key=lambda kv: kv[1])[0])

    voted = "".join(out)
    return voted if len(voted) >= len(best_text) else best_text


# =============================================================================
# Main detector class
# =============================================================================

class TrafficViolationDetector:
    """
    Two-wheeler traffic violation detector.

    Detects violations per bike:
      1. More than 2 riders.
      2. One or more riders not wearing a helmet.
      3. Combination of the above.

    For each violating bike, also reads the license plate (full string).
    """

    def __init__(self, model_dir: str = "./models"):
        """
        Load all models from model_dir.
        Internet access is NOT required — all weights must be local.
        """
        self.model_dir = Path(model_dir)

        # Avoid OpenCV over-threading overhead on small crops.
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        # ── Detection models (required) ────────────────────────────────────
        self.full_model   = YOLO(str(self._find_model(["full_detector.pt",   "yolo11m.pt"])))
        self.helmet_model = YOLO(str(self._find_model(["helmet_detector.pt", "yolo11s_helmet.pt"])))
        self.plate_model  = YOLO(str(self._find_model(["plate_detector.pt",  "yolo11n_plate.pt"])))

        # ── Optional COCO fallback model ───────────────────────────────────
        # Used only when the custom full_detector misses bike/rider.
        # Put yolo11n.pt inside models/ folder. If missing, pipeline still works normally.
        self.coco_model = self._try_load_model(["yolo11n.pt", "coco_yolo11n.pt"])

        # ── Runtime mode ──────────────────────────────────────────────────
        # Default is FAST mode because the assignment guideline is <=5 sec/image.
        # Set TV_ACCURATE_MODE=1 only for local experiments, not final submission.
        self.fast_mode = os.environ.get("TV_ACCURATE_MODE", "0") != "1"

        # Pose adds one more YOLO call per bike. Keep it disabled by default.
        self.use_pose = os.environ.get("TV_USE_POSE", "0") == "1"
        self.pose_model = self._try_load_model(["pose_detector.pt", "yolo11n-pose.pt"]) if self.use_pose else None

        # Cache device once instead of checking CUDA on every YOLO call.
        self.device = 0 if self._has_cuda() else "cpu"

        # ── Inference thresholds ──────────────────────────────────────────
        if self.fast_mode:
            # BALANCED FAST MODE: still targets <=5 sec, but keeps enough
            # resolution/recall to avoid missing smaller bikes/plates.
            # Your earlier 640/480/512 + 1 OCR variant was ~1.45s but lost
            # one violation and returned empty plate on the sample image.
            self.full_imgsz   = 768
            self.helmet_imgsz = 512
            self.plate_imgsz  = 640
            self.pose_imgsz   = 512
            self.max_bikes_to_process = 6
            self.max_plates_to_ocr = 2
            self.max_ocr_variants = 3
        else:
            self.full_imgsz   = 960
            self.helmet_imgsz = 640
            self.plate_imgsz  = 960
            self.pose_imgsz   = 640
            self.max_bikes_to_process = 8
            self.max_plates_to_ocr = 3
            self.max_ocr_variants = 3

        self.full_conf   = 0.25
        self.helmet_conf = 0.22
        self.plate_conf  = 0.12
        self.pose_conf   = 0.25
        self.iou_thr     = 0.50

        # ── OCR: EasyOCR only ─────────────────────────────────────────────
        self.ocr = self._load_easyocr()

        # ── Warm up YOLO models to avoid cold-start latency ───────────────
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        for mdl in [self.full_model, self.helmet_model, self.plate_model, self.coco_model]:
            if mdl is None:
                continue
            try:
                mdl.predict(dummy, imgsz=64, verbose=False)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Model loading helpers
    # -------------------------------------------------------------------------

    def _find_model(self, candidates: List[str]) -> Path:
        for name in candidates:
            p = self.model_dir / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"None of {candidates} found in {self.model_dir}"
        )

    def _try_load_model(self, candidates: List[str]) -> Optional[YOLO]:
        for name in candidates:
            p = self.model_dir / name
            if p.exists():
                try:
                    return YOLO(str(p))
                except Exception:
                    pass
        return None

    def _load_easyocr(self):
        """Load EasyOCR from local models/easyocr only. No internet downloads."""
        if not EASYOCR_AVAILABLE:
            return None

        ocr_dir = self.model_dir / "easyocr"
        try:
            has_pth = ocr_dir.exists() and any(ocr_dir.rglob("*.pth"))
        except Exception:
            has_pth = False

        if not has_pth:
            return None

        try:
            return _easyocr_lib.Reader(
                ["en"],
                gpu=(self.device != "cpu"),
                model_storage_directory=str(ocr_dir),
                user_network_directory=str(ocr_dir / "user_network"),
                download_enabled=False,
                verbose=False,
            )
        except TypeError:
            try:
                return _easyocr_lib.Reader(
                    ["en"],
                    gpu=(self.device != "cpu"),
                    model_storage_directory=str(ocr_dir),
                    download_enabled=False,
                    verbose=False,
                )
            except Exception:
                return None
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # YOLO inference helper
    # -------------------------------------------------------------------------

    def _predict_yolo(
        self,
        model: YOLO,
        img: np.ndarray,
        imgsz: int,
        conf: float,
    ) -> List[Det]:
        try:
            results = model.predict(
                img,
                imgsz=imgsz,
                conf=conf,
                iou=self.iou_thr,
                verbose=False,
                device=self.device,
            )
        except Exception:
            results = model.predict(
                img,
                imgsz=imgsz,
                conf=conf,
                iou=self.iou_thr,
                verbose=False,
            )

        dets: List[Det] = []
        for r in results:
            if r.boxes is None:
                continue
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                dets.append(Det(
                    cls_id=cls_id,
                    cls_name=str(names.get(cls_id, cls_id)),
                    conf=float(box.conf[0]),
                    xyxy=[float(v) for v in box.xyxy[0]],
                ))


#             results = model.predict(
#                 img,
#                 imgsz=imgsz,
#                 conf=conf,
#                 iou=self.iou_thr,
#                 verbose=False,
#                 device=self.device,
#             )
#         except Exception:
#             results = model.predict(
#                 img,
#                 imgsz=imgsz,
#                 conf=conf,
#                 iou=self.iou_thr,
#                 verbose=False,
#             )
# 
#         dets: List[Det] = []
#         for r in results:
#             if r.boxes is None:
#                 continue
#             names = r.names
#             for box in r.boxes:
#                 cls_id = int(box.cls[0])
#                 dets.append(Det(
#                     cls_id=cls_id,
#                     cls_name=str(names.get(cls_id, cls_id)),
#                     conf=float(box.conf[0]),
#                     xyxy=[float(v) for v in box.xyxy[0]],
#                 ))
