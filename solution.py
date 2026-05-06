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
        return dets

    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Category helpers
    # -------------------------------------------------------------------------

    _BIKE_NAMES  = {"motorcycle", "motorbike", "scooter", "two_wheeler", "bike"}
    _RIDER_NAMES = {"person", "rider", "human", "pedestrian"}
    _HELMET_NAMES    = {"helmet", "with_helmet", "with helmet"}
    _NO_HELMET_NAMES = {"no_helmet", "no helmet", "without_helmet", "without helmet",
                        "nohelmet", "bare_head", "bare head"}
    _PLATE_NAMES = {"license_plate", "licence_plate", "plate", "numberplate",
                    "number_plate", "lp"}

    def _bike_category(self, d: Det) -> Optional[str]:
        n = _norm_name(d.cls_name)
        if n in self._BIKE_NAMES or n in {"two_wheeler", "twowheeler", "2_wheeler"}:
            return "bike"
        # Fallback for your trained full detector: class 0 = two_wheeler.
        return "bike" if d.cls_id == 0 else None

    def _rider_category(self, d: Det) -> Optional[str]:
        n = _norm_name(d.cls_name)
        if n in self._RIDER_NAMES:
            return "rider"
        # Fallback for your trained full detector: class 1 = rider.
        return "rider" if d.cls_id == 1 else None

    def _helmet_category(self, d: Det) -> Optional[str]:
        n = _norm_name(d.cls_name)
        if n in self._HELMET_NAMES:
            return "helmet"
        if n in self._NO_HELMET_NAMES:
            return "no_helmet"
        # Fallback for your trained helmet detector: class 0 = helmet, class 1 = no_helmet.
        if d.cls_id == 0:
            return "helmet"
        if d.cls_id == 1:
            return "no_helmet"
        return None

    def _plate_category(self, d: Det) -> Optional[str]:
        n = _norm_name(d.cls_name)
        if n in self._PLATE_NAMES:
            return "plate"
        # Fallback for one-class plate detector.
        return "plate" if d.cls_id == 0 else None

    # -------------------------------------------------------------------------
    # Step 1 — Global bike + rider detection
    # -------------------------------------------------------------------------

    def _get_full_detector_recall_candidates(
        self, img: np.ndarray
    ) -> Tuple[List[Det], List[Det]]:
        """
        Recall fallback using the same custom full_detector.

        This is called only when the normal detector misses bike/rider.
        It uses a lower confidence and higher image size, but does not replace
        good detections from the normal path.
        """
        fallback_imgsz = max(self.full_imgsz, 960)
        fallback_conf = min(self.full_conf, 0.10)

        raw = self._predict_yolo(
            self.full_model,
            img,
            fallback_imgsz,
            fallback_conf,
        )

        bikes = _nms_same_class(
            [d for d in raw if self._bike_category(d) == "bike"],
            0.50,
        )

        riders = _nms_same_class(
            [d for d in raw if self._rider_category(d) == "rider"],
            0.45,
        )

        return bikes, riders

    def _get_coco_candidates(
        self, img: np.ndarray
    ) -> Tuple[List[Det], List[Det]]:
        """
        COCO fallback using models/yolo11n.pt.

        COCO classes used:
          person     -> possible rider
          motorcycle -> two_wheeler

        Person detections are filtered later using bike overlap so pedestrians
        do not directly become riders.
        """
        if self.coco_model is None:
            return [], []

        raw = self._predict_yolo(
            self.coco_model,
            img,
            imgsz=max(self.full_imgsz, 960),
            conf=0.15,
        )

        bikes: List[Det] = []
        persons: List[Det] = []

        for d in raw:
            n = _norm_name(d.cls_name)

            if n in {"motorcycle", "motorbike", "scooter"}:
                bikes.append(Det(
                    cls_id=0,
                    cls_name="two_wheeler",
                    conf=d.conf,
                    xyxy=d.xyxy,
                ))

            elif n == "person":
                persons.append(Det(
                    cls_id=1,
                    cls_name="rider",
                    conf=d.conf,
                    xyxy=d.xyxy,
                ))

        bikes = _nms_same_class(bikes, 0.50)
        persons = _nms_same_class(persons, 0.45)

        return bikes, persons

    def _filter_persons_as_riders(
        self,
        persons: List[Det],
        bikes: List[Det],
    ) -> List[Det]:
        """
        Convert COCO person boxes to rider boxes only if they are associated
        with a motorcycle/two_wheeler. This avoids treating pedestrians as riders.
        """
        if not persons or not bikes:
            return []

        riders: List[Det] = []

        for person in persons:
            px1, py1, px2, py2 = person.xyxy
            p_area = max(1e-6, (px2 - px1) * (py2 - py1))
            p_cx = (px1 + px2) / 2.0
            p_cy = (py1 + py2) / 2.0
            p_bottom_y = py2

            best_score = 0.0

            for bike in bikes:
                bx1, by1, bx2, by2 = bike.xyxy
                bw = bx2 - bx1
                bh = by2 - by1

                # Expanded bike region captures rider above and around the bike.
                ex_box = [
                    bx1 - 0.25 * bw,
                    by1 - 0.90 * bh,
                    bx2 + 0.25 * bw,
                    by2 + 0.35 * bh,
                ]

                overlap = _inter_area(person.xyxy, ex_box) / p_area

                center_inside = (
                    ex_box[0] <= p_cx <= ex_box[2]
                    and ex_box[1] <= p_cy <= ex_box[3]
                )

                bottom_inside = (
                    ex_box[0] <= p_cx <= ex_box[2]
                    and ex_box[1] <= p_bottom_y <= ex_box[3]
                )

                score = overlap

                if center_inside:
                    score += 0.25

                if bottom_inside:
                    score += 0.25

                best_score = max(best_score, score)

            if best_score >= 0.18:
                riders.append(person)

        return _nms_same_class(riders, 0.45)

    def _get_bikes_and_riders(
        self, img: np.ndarray
    ) -> Tuple[List[Det], List[Det]]:
        # ------------------------------------------------------------
        # 1. Original normal detection path
        # ------------------------------------------------------------
        raw = self._predict_yolo(self.full_model, img, self.full_imgsz, self.full_conf)

        bikes = _nms_same_class(
            [d for d in raw if self._bike_category(d) == "bike"],
            0.50,
        )

        riders = _nms_same_class(
            [d for d in raw if self._rider_category(d) == "rider"],
            0.45,
        )

        # If the old detector already works, do not change anything.
        if bikes and riders:
            return bikes, riders

        # ------------------------------------------------------------
        # 2. Recall fallback using the same custom full_detector
        # ------------------------------------------------------------
        recall_bikes, recall_riders = self._get_full_detector_recall_candidates(img)

        if not bikes and recall_bikes:
            bikes = recall_bikes

        if not riders and recall_riders:
            riders = recall_riders

        if bikes and riders:
            return bikes, riders

        # ------------------------------------------------------------
        # 3. COCO fallback using yolo11n.pt
        # ------------------------------------------------------------
        coco_bikes, coco_persons = self._get_coco_candidates(img)

        if not bikes and coco_bikes:
            bikes = coco_bikes

        if not riders and coco_persons and bikes:
            riders = self._filter_persons_as_riders(coco_persons, bikes)

        return bikes, riders

    # -------------------------------------------------------------------------
    # Step 2 — Associate riders to bikes
    # -------------------------------------------------------------------------

    def _associate_riders(
        self,
        bikes: List[Det],
        riders: List[Det],
        img_shape: Tuple[int, ...],
    ) -> Dict[int, List[Det]]:
        rider_map: Dict[int, List[Det]] = {i: [] for i in range(len(bikes))}
        for rider in riders:
            best_idx, best_iou = -1, 0.0
            rx1, ry1, rx2, ry2 = rider.xyxy
            for i, bike in enumerate(bikes):
                bx1, by1, bx2, by2 = bike.xyxy
                # Expand bike box to capture riders who sit higher than the frame.
                bx1e = bx1 - (bx2 - bx1) * 0.1
                by1e = by1 - (by2 - by1) * 0.6
                overlap = _inter_area(rider.xyxy, [bx1e, by1e, bx2, by2])
                rider_area = max(1e-6, (rx2 - rx1) * (ry2 - ry1))
                score = overlap / rider_area
                if score > best_iou:
                    best_iou, best_idx = score, i
            if best_idx >= 0 and best_iou > 0.10:
                rider_map[best_idx].append(rider)
        return rider_map

    # -------------------------------------------------------------------------
    # Step 3a — Helmet detection per bike
    # -------------------------------------------------------------------------

    def _detect_helmets_on_bike(
        self,
        img: np.ndarray,
        bike_box: List[float],
    ) -> Tuple[List[Det], List[Det]]:
        # Keep strong upward padding: rider heads/no-helmet regions often sit
        # well above the motorcycle body box. This recovers violations that
        # were missed by the ultra-fast crop.
        bike_crop, offset = _crop_box(img, bike_box, xpad=0.30, ypad=0.75)
        raw = self._predict_yolo(
            self.helmet_model, bike_crop, self.helmet_imgsz, self.helmet_conf,
        )
        helmet_dets, no_helmet_dets = [], []
        for d in raw:
            cat = self._helmet_category(d)
            if cat == "helmet":
                helmet_dets.append(_offset_det(d, offset))
            elif cat == "no_helmet":
                no_helmet_dets.append(_offset_det(d, offset))
        helmet_dets    = _nms_same_class(helmet_dets,    0.40)
        no_helmet_dets = _nms_same_class(no_helmet_dets, 0.40)
        return helmet_dets, no_helmet_dets

    # -------------------------------------------------------------------------
    # Step 3b — Rider counting (pose > bbox fallback)
    # -------------------------------------------------------------------------

    def _count_riders_via_pose(
        self, img: np.ndarray, bike_box: List[float]
    ) -> Optional[int]:
        if self.pose_model is None:
            return None
        bike_crop, _ = _crop_box(img, bike_box, xpad=0.15, ypad=0.50)
        if bike_crop.size == 0:
            return None
        try:
            results = self.pose_model.predict(
                bike_crop,
                imgsz=self.pose_imgsz,
                conf=self.pose_conf,
                verbose=False,
            )
            count = sum(
                1 for r in results
                if r.keypoints is not None and len(r.keypoints) > 0
            )
            return count if count > 0 else None
        except Exception:
            return None

    def _count_riders_for_bike(
        self,
        img: np.ndarray,
        bike: Det,
        matched_riders: List[Det],
        helmet_dets: List[Det],
        no_helmet_dets: List[Det],
    ) -> int:
        pose_count = self._count_riders_via_pose(img, bike.xyxy)
        if pose_count is not None:
            return pose_count

        bbox_count = len(matched_riders)
        head_count = len(helmet_dets) + len(no_helmet_dets)
        return max(bbox_count, head_count, 1 if (helmet_dets or no_helmet_dets) else 0)

    # -------------------------------------------------------------------------
    # Step 4 — License plate OCR
    # -------------------------------------------------------------------------

    def _read_plate_easyocr(self, plate_crop: np.ndarray) -> List[Tuple[str, float]]:
        """
        Run EasyOCR on multiple preprocessed variants of the plate crop.

        EasyOCR with paragraph=False returns one item per text region.
        For a 2-line Indian plate ("AP07" on line 1, "AB1234" on line 2)
        we sort regions top-to-bottom by centroid Y and concatenate to get
        "AP07AB1234".
        We also keep individual fragment candidates for the voter.
        """
        if self.ocr is None or plate_crop is None or plate_crop.size == 0:
            return []

        allow = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        candidates: List[Tuple[str, float]] = []

        for variant in _make_plate_variants(plate_crop)[: self.max_ocr_variants]:
            try:
                result = self.ocr.readtext(
                    variant,
                    detail=1,           # return (bbox, text, conf)
                    paragraph=False,    # separate regions — needed for multi-line plates
                    allowlist=allow,
                    decoder="greedy",   # faster than beamsearch
                    batch_size=1,
                )
                if not result:
                    continue

                # Full plate string: sort top→bottom and concatenate all regions.
                full_text, full_conf = _easyocr_regions_to_plate(result)
                if len(full_text) >= 4:
                    candidates.append((full_text, full_conf))
                    # Good enough: avoid extra work in common case.
                    if _INDIAN_PLATE_RE.match(full_text) or len(full_text) >= 7:
                        return candidates

                # Also add individual region fragments as shorter candidates.
                for item in result:
                    try:
                        t = _clean_plate_text(item[1])
                        c = float(item[2])
                        if len(t) >= 3:
                            candidates.append((t, c))
                    except Exception:
                        continue

            except Exception:
                continue

        return candidates

    def _detect_plate_text(
        self, img: np.ndarray, bike_box: List[float]
    ) -> str:
        """
        Locate the license plate in the bike region, then OCR it to obtain
        the full plate string (e.g. "AP07AB1234").
        """
        # Wider bike crop improves plate recall with only a small speed cost.
        bike_crop, bike_offset = _crop_box(img, bike_box, xpad=0.35, ypad=0.35)

        raw_dets = self._predict_yolo(
            self.plate_model, bike_crop, self.plate_imgsz, self.plate_conf,
        )

        plates: List[Det] = []
        for d in raw_dets:
            if self._plate_category(d) == "plate":
                plates.append(_offset_det(d, bike_offset, "plate"))
        plates = _nms_same_class(plates, 0.30)

        if not plates:
            return ""

        plates = sorted(plates, key=lambda d: (d.conf, d.area), reverse=True)

        all_candidates: List[Tuple[str, float]] = []

        for plate in plates[: self.max_plates_to_ocr]:
            # More padding around the plate prevents cutting characters.
            plate_crop, _ = _crop_box(img, plate.xyxy, xpad=0.22, ypad=0.45)
            cands = self._read_plate_easyocr(plate_crop)
            all_candidates.extend(cands)

            voted = _vote_plate(all_candidates)
            if _INDIAN_PLATE_RE.match(voted):
                return voted

        return _vote_plate(all_candidates)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, image_path: str) -> dict:
        """
        Process a single street image and return all traffic violations.

        Strict output format (evaluated by auto-grader):
        {
            "violations": [
                {
                    "num_riders":        int,
                    "helmet_violations": int,
                    "license_plate":     str
                }
            ]
        }
        No extra keys — only "violations".
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return {"violations": []}

            bikes, riders = self._get_bikes_and_riders(img)
            if not bikes:
                return {"violations": []}

            # Process most confident/largest bikes first to stay under the 5s guideline.
            bikes = sorted(bikes, key=lambda d: (d.conf, d.area), reverse=True)[: self.max_bikes_to_process]
            rider_map = self._associate_riders(bikes, riders, img.shape)

            violations: List[Dict[str, Any]] = []

            for i, bike in enumerate(bikes):
                matched_riders = rider_map.get(i, [])

                helmet_dets, no_helmet_dets = self._detect_helmets_on_bike(
                    img, bike.xyxy,
                )

                rider_count = self._count_riders_for_bike(
                    img, bike, matched_riders, helmet_dets, no_helmet_dets,
                )

                helmet_violations = len(no_helmet_dets)

                if rider_count <= 2 and helmet_violations == 0:
                    continue

                plate_text = self._detect_plate_text(img, bike.xyxy)

                violations.append({
                    "num_riders":        int(rider_count),
                    "helmet_violations": int(helmet_violations),
                    "license_plate":     str(plate_text),
                })

            return {"violations": violations}

        except Exception:
            return {"violations": []}

    # ─────────────────────────────────────────────────────────────────────────
    # Optional debug helper (not called by evaluator)
    # ─────────────────────────────────────────────────────────────────────────

    def predict_debug(self, image_path: str) -> dict:
        """
        Extended output for visual debugging.  NOT used by the evaluator.

        Returns the same violations list plus per-bike bounding boxes,
        raw helmet/rider detections, violation flags, and inference time.
        """
        t_start = time.perf_counter()

        img = cv2.imread(str(image_path))
        if img is None:
            return {"violations": [], "debug": [], "inference_time_sec": 0.0}

        h, w = img.shape[:2]
        bikes, riders = self._get_bikes_and_riders(img)
        bikes = sorted(bikes, key=lambda d: (d.conf, d.area), reverse=True)[: self.max_bikes_to_process]
        rider_map     = self._associate_riders(bikes, riders, img.shape)

        violations: List[Dict[str, Any]] = []
        debug_items: List[Dict[str, Any]] = []

        for i, bike in enumerate(bikes):
            matched_riders = rider_map.get(i, [])
            helmet_dets, no_helmet_dets = self._detect_helmets_on_bike(img, bike.xyxy)

            rider_count       = self._count_riders_for_bike(
                img, bike, matched_riders, helmet_dets, no_helmet_dets,
            )
            helmet_violations = len(no_helmet_dets)
            is_violation      = rider_count > 2 or helmet_violations > 0

            plate_text = self._detect_plate_text(img, bike.xyxy) if is_violation else ""

            debug_items.append({
                "bike_bbox":        _clip_box(bike.xyxy, w, h),
                "rider_bboxes":     [_clip_box(r.xyxy, w, h) for r in matched_riders],
                "helmet_bboxes":    [_clip_box(d.xyxy, w, h) for d in helmet_dets],
                "no_helmet_bboxes": [_clip_box(d.xyxy, w, h) for d in no_helmet_dets],
                "num_riders":       int(rider_count),
                "helmet_violations":int(helmet_violations),
                "license_plate":    str(plate_text),
                "is_violation":     bool(is_violation),
                "pose_model_used":  self.pose_model is not None,
            })

            if is_violation:
                violations.append({
                    "num_riders":        int(rider_count),
                    "helmet_violations": int(helmet_violations),
                    "license_plate":     str(plate_text),
                })

        elapsed = round(time.perf_counter() - t_start, 3)
        return {"violations": violations, "debug": debug_items, "inference_time_sec": elapsed}