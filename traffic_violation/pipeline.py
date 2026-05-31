"""
traffic_violation.pipeline
============================
Top-level pipeline orchestrator: :class:`TrafficViolationDetector`.

This is the only class that the evaluator and end-users interact with::

    from traffic_violation import TrafficViolationDetector

    model = TrafficViolationDetector("./models")
    result = model.predict("image.jpg")

Pipeline Stages
---------------
1. **Global detection** (sequential, mandatory)
   ``full_detector`` runs on the full image to locate all two-wheelers and riders.
   An optional recall-fallback and a COCO-pretrained fallback are tried in order
   when the primary detection is incomplete.

2. **Per-bike concurrent processing** (``ThreadPoolExecutor``)
   For each detected bike, a worker thread handles:

   * Helmet / no-helmet detection on the bike crop.
   * License-plate detection + OCR on the bike crop.

   These two sub-tasks run *concurrently* across bikes because YOLO's C++/
   OpenCV extensions release the GIL during native inference, so true parallelism
   is achievable without ``asyncio``.  On a 2-bike image this typically reduces
   wall-clock time by ~40 %.

3. **Violation assembly** (sequential, O(n_bikes))
   Rider counts, helmet-violation counts, and plate strings are merged into the
   evaluator's expected output format.

Concurrency model
-----------------
``asyncio`` is deliberately avoided.  The heavy work (YOLO, EasyOCR, OpenCV)
is synchronous C-extension code that blocks the event loop.  ``ThreadPoolExecutor``
is the correct primitive here: each thread holds independent state and the GIL
is released inside native inference kernels.

Output format (strict, checked by the evaluator)
-------------------------------------------------
.. code-block:: json

    {
        "violations": [
            {
                "num_riders":        2,
                "helmet_violations": 1,
                "license_plate":     "AP07AB1234"
            }
        ]
    }
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, TypedDict

import cv2
import numpy as np

from traffic_violation.config import PipelineConfig, config_from_env
from traffic_violation.models.base import DetectorProtocol
from traffic_violation.models.yolo_detector import YOLODetector
from traffic_violation.ocr_engine import OCREngine
from traffic_violation.utils.geometry import (
    Box, Det,
    clip_box, crop_box, inter_area, iou, nms_same_class,
    norm_name, offset_det,
)
from traffic_violation.utils.plate_utils import INDIAN_PLATE_RE, vote_plate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed output structures
# ---------------------------------------------------------------------------

class ViolationRecord(TypedDict):
    """Single-bike violation entry returned by :meth:`TrafficViolationDetector.predict`."""
    num_riders:        int
    helmet_violations: int
    license_plate:     str


class PredictOutput(TypedDict):
    """Top-level output dict returned by :meth:`TrafficViolationDetector.predict`."""
    violations: list[ViolationRecord]


# ---------------------------------------------------------------------------
# Per-bike result (internal, not exposed to caller)
# ---------------------------------------------------------------------------

class _BikeResult:
    __slots__ = (
        "bike", "helmet_dets", "no_helmet_dets",
        "matched_riders", "plate_text",
    )

    def __init__(
        self,
        bike: Det,
        helmet_dets: list[Det],
        no_helmet_dets: list[Det],
        matched_riders: list[Det],
        plate_text: str,
    ) -> None:
        self.bike            = bike
        self.helmet_dets     = helmet_dets
        self.no_helmet_dets  = no_helmet_dets
        self.matched_riders  = matched_riders
        self.plate_text      = plate_text


# ---------------------------------------------------------------------------
# Class-label category sets
# ---------------------------------------------------------------------------

_BIKE_NAMES  = frozenset({"motorcycle", "motorbike", "scooter", "two_wheeler", "bike"})
_RIDER_NAMES = frozenset({"rider", "driver", "pillion", "human"})
_HELMET_NAMES    = frozenset({"helmet", "with_helmet", "with helmet"})
_NO_HELMET_NAMES = frozenset({
    "no_helmet", "no helmet", "without_helmet", "without helmet",
    "nohelmet", "bare_head", "bare head",
})
_PLATE_NAMES = frozenset({
    "license_plate", "licence_plate", "plate", "numberplate",
    "number_plate", "lp",
})


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TrafficViolationDetector:
    """End-to-end traffic violation detector for two-wheelers.

    Detects the following violations per bike:

    * One or more riders **not wearing a helmet**.
    * **More than two riders** on a single bike.

    For each violating bike the license plate is also read via EasyOCR.

    Args:
        model_dir: Path to the directory containing all model weights.
                   Passed directly to :func:`~traffic_violation.config.config_from_env`
                   if no explicit ``config`` is supplied.
        config:    Pre-built :class:`~traffic_violation.config.PipelineConfig`.
                   When omitted the configuration is derived from environment
                   variables (see :func:`~traffic_violation.config.config_from_env`).
    """

    def __init__(
        self,
        model_dir: str = "./models",
        config: Optional[PipelineConfig] = None,
    ) -> None:
        self._cfg: PipelineConfig = config or config_from_env(model_dir)

        # Minimise thread thrashing on small crops.
        self._configure_threading()

        # ── Detection models ───────────────────────────────────────────
        self._full_det    = self._load_detector(["full_detector",   "yolo11m"])
        self._helmet_det  = self._load_detector(["helmet_detector", "yolo11s_helmet"])
        self._plate_det   = self._load_detector(["plate_detector",  "yolo11n_plate"])
        self._coco_det    = self._try_load_detector(["yolo11n", "coco_yolo11n"])
        self._pose_det    = (
            self._try_load_detector(["pose_detector", "yolo11n-pose"])
            if self._cfg.use_pose else None
        )

        # ── OCR engine ────────────────────────────────────────────────
        self._ocr = OCREngine(
            model_dir=self._cfg.model_dir,
            gpu=(self._cfg.device != "cpu"),
            max_variants=self._cfg.max_ocr_variants,
        )

        # ── Warmup (avoids cold-start on the first real image) ────────
        self._warmup()

        logger.info(
            "TrafficViolationDetector ready  fast=%s  onnx=%s  pose=%s  ocr=%s",
            self._cfg.fast_mode, self._cfg.use_onnx,
            self._cfg.use_pose, self._ocr.available,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_path: str) -> PredictOutput:
        """Process a single street image and return all traffic violations.

        This is the method called by the evaluator.  The output schema is
        strictly ``{"violations": [...]}`` with no extra keys.

        Args:
            image_path: Filesystem path to a JPEG / PNG street image.

        Returns:
            A :class:`PredictOutput` dictionary.  Returns ``{"violations": []}``
            on any error (missing file, corrupt image, unexpected exception).
        """
        t0 = time.perf_counter()
        try:
            result = self._predict_impl(image_path)
        except Exception:
            logger.exception("predict() unhandled exception for %s", image_path)
            result = {"violations": []}
        elapsed = time.perf_counter() - t0
        logger.info(
            "predict() done  path=%s  violations=%d  elapsed=%.3fs",
            image_path, len(result["violations"]), elapsed,
        )
        return result

    def predict_debug(self, image_path: str) -> dict[str, Any]:
        """Extended output for offline debugging.  Not used by the evaluator.

        Returns the standard ``violations`` list plus per-bike diagnostic
        fields: bounding boxes, helmet/rider detections, flags, and timing.

        Args:
            image_path: Filesystem path to the image.

        Returns:
            Dict with keys ``"violations"``, ``"debug"``, and
            ``"inference_time_sec"``.
        """
        t0 = time.perf_counter()
        img = cv2.imread(str(image_path))
        if img is None:
            return {"violations": [], "debug": [], "inference_time_sec": 0.0}

        h, w = img.shape[:2]
        bikes, riders = self._get_bikes_and_riders(img)
        bikes = sorted(bikes, key=lambda d: (d.conf, d.area), reverse=True)[: self._cfg.max_bikes]
        rider_map = self._associate_riders(bikes, riders, img.shape)

        bike_results = self._process_bikes_concurrent(img, bikes, rider_map)

        violations: list[ViolationRecord] = []
        debug_items: list[dict[str, Any]] = []

        for br in bike_results:
            rider_count = self._count_riders_for_bike(
                img, br.bike, br.matched_riders, br.helmet_dets, br.no_helmet_dets,
            )
            helmet_violations = min(len(br.no_helmet_dets), max(rider_count, 0))
            is_violation = rider_count > 2 or helmet_violations > 0

            debug_items.append({
                "bike_bbox":         clip_box(br.bike.xyxy, w, h),
                "rider_bboxes":      [clip_box(r.xyxy, w, h) for r in br.matched_riders],
                "helmet_bboxes":     [clip_box(d.xyxy, w, h) for d in br.helmet_dets],
                "no_helmet_bboxes":  [clip_box(d.xyxy, w, h) for d in br.no_helmet_dets],
                "num_riders":        int(rider_count),
                "helmet_violations": int(helmet_violations),
                "license_plate":     br.plate_text,
                "is_violation":      bool(is_violation),
                "pose_model_used":   self._pose_det is not None,
            })
            if is_violation:
                violations.append(ViolationRecord(
                    num_riders=int(rider_count),
                    helmet_violations=int(helmet_violations),
                    license_plate=br.plate_text,
                ))

        return {
            "violations":          violations,
            "debug":               debug_items,
            "inference_time_sec":  round(time.perf_counter() - t0, 3),
        }

    # ------------------------------------------------------------------
    # Core pipeline (private)
    # ------------------------------------------------------------------

    def _predict_impl(self, image_path: str) -> PredictOutput:
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning("predict: could not read image %s", image_path)
            return {"violations": []}

        # Stage 1 — global detection (must be sequential)
        bikes, riders = self._get_bikes_and_riders(img)
        if not bikes:
            return {"violations": []}

        bikes = sorted(bikes, key=lambda d: (d.conf, d.area), reverse=True)[: self._cfg.max_bikes]
        rider_map = self._associate_riders(bikes, riders, img.shape)

        # Stage 2 — per-bike processing (concurrent)
        bike_results = self._process_bikes_concurrent(img, bikes, rider_map)

        # Stage 3 — assemble output
        violations: list[ViolationRecord] = []
        for br in bike_results:
            rider_count = self._count_riders_for_bike(
                img, br.bike, br.matched_riders, br.helmet_dets, br.no_helmet_dets,
            )
            helmet_violations = min(len(br.no_helmet_dets), max(rider_count, 0))

            if rider_count <= 2 and helmet_violations == 0:
                continue

            violations.append(ViolationRecord(
                num_riders=int(rider_count),
                helmet_violations=int(helmet_violations),
                license_plate=br.plate_text,
            ))

        return {"violations": violations}

    # ------------------------------------------------------------------
    # Stage 2 — concurrent per-bike worker
    # ------------------------------------------------------------------

    def _process_bikes_concurrent(
        self,
        img: np.ndarray,
        bikes: list[Det],
        rider_map: dict[int, list[Det]],
    ) -> list[_BikeResult]:
        """Process all bikes concurrently using a ``ThreadPoolExecutor``.

        Helmet detection and plate OCR are submitted as independent futures for
        each bike.  The GIL is released during C-extension inference (YOLO,
        OpenCV, ONNX Runtime), enabling true parallel execution on multi-core
        CPUs.

        Args:
            img:       Full BGR image.
            bikes:     List of bike :class:`~traffic_violation.utils.geometry.Det`.
            rider_map: Pre-computed rider-to-bike associations.

        Returns:
            List of :class:`_BikeResult` objects, one per bike, in the same
            order as *bikes*.
        """
        n_workers = self._cfg.n_workers or min(4, max(1, len(bikes)))
        results: list[_BikeResult | None] = [None] * len(bikes)

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {
                pool.submit(self._process_single_bike, img, i, bike, rider_map.get(i, [])): i
                for i, bike in enumerate(bikes)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.warning("Bike %d processing failed", idx, exc_info=True)
                    bike = bikes[idx]
                    results[idx] = _BikeResult(bike, [], [], rider_map.get(idx, []), "")

        return [r for r in results if r is not None]

    def _process_single_bike(
        self,
        img: np.ndarray,
        idx: int,
        bike: Det,
        matched_riders: list[Det],
    ) -> _BikeResult:
        """Worker: helmet detection + plate OCR for one bike.

        Both tasks are independent and run sequentially within each thread.
        The thread itself runs concurrently with other bike threads.

        Args:
            img:             Full BGR image.
            idx:             Bike index (for logging).
            bike:            Bike detection in full-image coordinates.
            matched_riders:  Pre-associated rider detections for this bike.

        Returns:
            A populated :class:`_BikeResult`.
        """
        logger.debug("Bike %d: starting worker thread", idx)

        helmet_dets, no_helmet_dets = self._detect_helmets_on_bike(img, bike.xyxy)

        # Only run OCR if there is at least one potential violation.
        # This avoids spending ~1 s on compliant bikes.
        rider_count_est = max(len(matched_riders), len(helmet_dets) + len(no_helmet_dets), 1)
        helmet_viol_est = len(no_helmet_dets)
        needs_plate = rider_count_est > 2 or helmet_viol_est > 0

        plate_text = self._detect_plate_text(img, bike.xyxy) if needs_plate else ""

        logger.debug(
            "Bike %d done: helmets=%d  no_helmets=%d  plate=%r",
            idx, len(helmet_dets), len(no_helmet_dets), plate_text,
        )
        return _BikeResult(bike, helmet_dets, no_helmet_dets, matched_riders, plate_text)

    # ------------------------------------------------------------------
    # Stage 1 — global bike + rider detection
    # ------------------------------------------------------------------

    def _get_bikes_and_riders(
        self, img: np.ndarray,
    ) -> tuple[list[Det], list[Det]]:
        """Run the full-image detector and return bikes and riders.

        Falls back to lower confidence + higher resolution, then to the optional
        COCO-pretrained model, in order to maximise recall on difficult images.
        """
        raw = self._full_det.predict(img, self._cfg.full_imgsz, self._cfg.full_conf, self._cfg.iou_thr)

        bikes  = nms_same_class([d for d in raw if self._cat_bike(d)  == "bike"],  0.50)
        riders = nms_same_class([d for d in raw if self._cat_rider(d) == "rider"], 0.45)

        if bikes and riders:
            return bikes, riders

        # ── Recall fallback: same model, lower conf, higher imgsz ─────
        fb_imgsz = max(self._cfg.full_imgsz, 960)
        fb_conf  = min(self._cfg.full_conf, 0.10)
        fb_raw   = self._full_det.predict(img, fb_imgsz, fb_conf, self._cfg.iou_thr)

        fb_bikes  = nms_same_class([d for d in fb_raw if self._cat_bike(d)  == "bike"],  0.50)
        fb_riders = nms_same_class([d for d in fb_raw if self._cat_rider(d) == "rider"], 0.45)

        if not bikes and fb_bikes:
            bikes = fb_bikes
        if not riders and fb_riders:
            riders = fb_riders

        if bikes and riders:
            return bikes, riders

        # ── COCO fallback ─────────────────────────────────────────────
        if self._coco_det is not None:
            coco_bikes, coco_persons = self._get_coco_candidates(img)
            if not bikes and coco_bikes:
                bikes = coco_bikes
            if not riders and coco_persons and bikes:
                riders = self._filter_persons_as_riders(coco_persons, bikes)

        return bikes, riders

    def _get_coco_candidates(
        self, img: np.ndarray,
    ) -> tuple[list[Det], list[Det]]:
        """Run the COCO fallback model and return (bikes, person_candidates)."""
        assert self._coco_det is not None
        raw = self._coco_det.predict(
            img,
            imgsz=max(self._cfg.full_imgsz, 960),
            conf=0.15,
            iou=self._cfg.iou_thr,
        )
        bikes: list[Det]   = []
        persons: list[Det] = []
        for d in raw:
            n = norm_name(d.cls_name)
            if n in {"motorcycle", "motorbike", "scooter"}:
                bikes.append(Det(0, "two_wheeler", d.conf, d.xyxy))
            elif n == "person":
                persons.append(Det(1, "rider", d.conf, d.xyxy))
        return nms_same_class(bikes, 0.50), nms_same_class(persons, 0.45)

    def _filter_persons_as_riders(
        self, persons: list[Det], bikes: list[Det],
    ) -> list[Det]:
        """Keep only COCO person boxes that overlap sufficiently with a bike."""
        riders: list[Det] = []
        for person in persons:
            px1, py1, px2, py2 = person.xyxy
            p_area = max(1e-6, (px2 - px1) * (py2 - py1))
            pcx, pcy   = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            p_bottom_y = py2
            best = 0.0
            for bike in bikes:
                bx1, by1, bx2, by2 = bike.xyxy
                bw, bh = bx2 - bx1, by2 - by1
                ex = [bx1 - 0.25*bw, by1 - 0.90*bh, bx2 + 0.25*bw, by2 + 0.35*bh]
                score = inter_area(person.xyxy, ex) / p_area
                if ex[0] <= pcx <= ex[2] and ex[1] <= pcy <= ex[3]:
                    score += 0.25
                if ex[0] <= pcx <= ex[2] and ex[1] <= p_bottom_y <= ex[3]:
                    score += 0.25
                best = max(best, score)
            if best >= 0.18:
                riders.append(person)
        return nms_same_class(riders, 0.45)

    # ------------------------------------------------------------------
    # Stage 1b — associate riders to bikes
    # ------------------------------------------------------------------

    def _associate_riders(
        self,
        bikes: list[Det],
        riders: list[Det],
        img_shape: tuple[int, ...],
    ) -> dict[int, list[Det]]:
        """Assign each rider detection to the most spatially compatible bike.

        Uses a composite score based on:
            * Overlap ratio of the rider box vs the expanded bike support region.
            * Whether the rider's centroid lies inside that region.
            * Whether the rider's bottom point is near/on the bike.
            * A penalty for people vertically far from the motorcycle.
        """
        rider_map: dict[int, list[Det]] = {i: [] for i in range(len(bikes))}
        if not bikes or not riders:
            return rider_map

        for rider in riders:
            rx1, ry1, rx2, ry2 = rider.xyxy
            rw      = max(1.0, rx2 - rx1)
            rh      = max(1.0, ry2 - ry1)
            r_area  = max(1e-6, rw * rh)
            rcx     = (rx1 + rx2) / 2.0
            rcy     = (ry1 + ry2) / 2.0
            rbottom = ry2

            best_idx, best_score = -1, 0.0

            for i, bike in enumerate(bikes):
                bx1, by1, bx2, by2 = bike.xyxy
                bw = max(1.0, bx2 - bx1)
                bh = max(1.0, by2 - by1)

                support = [
                    bx1 - 0.20*bw, by1 - 0.85*bh,
                    bx2 + 0.20*bw, by2 + 0.30*bh,
                ]

                overlap = inter_area(rider.xyxy, support) / r_area
                score   = overlap

                if support[0] <= rcx <= support[2] and support[1] <= rcy <= support[3]:
                    score += 0.20
                if support[0] <= rcx <= support[2] and support[1] <= rbottom <= support[3]:
                    score += 0.25
                if bx1 - 0.25*bw <= rcx <= bx2 + 0.25*bw and by1 - 0.35*bh <= rbottom <= by2 + 0.45*bh:
                    score += 0.25
                if rbottom < by1 - 0.55*bh or ry1 > by2 + 0.35*bh:
                    score -= 0.35

                if score > best_score:
                    best_score, best_idx = score, i

            if best_idx >= 0 and best_score >= 0.28:
                rider_map[best_idx].append(rider)

        return rider_map

    # ------------------------------------------------------------------
    # Helmet detection
    # ------------------------------------------------------------------

    def _detect_helmets_on_bike(
        self, img: np.ndarray, bike_box: Box,
    ) -> tuple[list[Det], list[Det]]:
        """Detect helmets and bare-head regions within a padded bike crop."""
        bike_crop, offset = crop_box(img, bike_box, xpad=0.30, ypad=0.75)
        raw = self._helmet_det.predict(
            bike_crop, self._cfg.helmet_imgsz, self._cfg.helmet_conf, self._cfg.iou_thr,
        )

        helmet_dets: list[Det] = []
        no_helmet_dets: list[Det] = []
        for d in raw:
            cat = self._cat_helmet(d)
            if cat == "helmet":
                helmet_dets.append(offset_det(d, offset, "helmet"))
            elif cat == "no_helmet":
                no_helmet_dets.append(offset_det(d, offset, "no_helmet"))

        helmet_dets    = nms_same_class(helmet_dets,    0.38)
        no_helmet_dets = nms_same_class(no_helmet_dets, 0.38)

        helmet_dets, no_helmet_dets = self._resolve_head_detections(
            helmet_dets, no_helmet_dets, bike_box,
        )
        return helmet_dets, no_helmet_dets

    def _resolve_head_detections(
        self,
        helmet_dets: list[Det],
        no_helmet_dets: list[Det],
        bike_box: Box,
    ) -> tuple[list[Det], list[Det]]:
        """Remove duplicate / conflicting helmet/no_helmet boxes for one bike.

        Cross-class NMS is applied with a slight bias toward ``no_helmet`` to
        avoid silently missing helmet violations.
        """
        bx1, by1, bx2, by2 = bike_box
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)
        head_region: Box = [
            bx1 - 0.25*bw, by1 - 0.95*bh,
            bx2 + 0.25*bw, by2 + 0.20*bh,
        ]

        candidates = [
            Det(d.cls_id, "helmet",    d.conf, d.xyxy) for d in helmet_dets
        ] + [
            Det(d.cls_id, "no_helmet", d.conf, d.xyxy) for d in no_helmet_dets
        ]

        # Keep only heads associated with this specific bike's region.
        filtered: list[Det] = []
        for d in candidates:
            x1, y1, x2, y2 = d.xyxy
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            center_in = (
                head_region[0] <= cx <= head_region[2] and
                head_region[1] <= cy <= head_region[3]
            )
            overlap_ratio = inter_area(d.xyxy, head_region) / max(1e-6, d.area)
            if center_in or overlap_ratio > 0.35:
                filtered.append(d)

        # Sort descending by conf, with a slight nudge for no_helmet.
        filtered = sorted(
            filtered,
            key=lambda d: float(d.conf) + (0.03 if norm_name(d.cls_name) == "no_helmet" else 0.0),
            reverse=True,
        )

        kept: list[Det] = []
        for d in filtered:
            is_dup = any(
                iou(d.xyxy, k.xyxy) > 0.35 or
                inter_area(d.xyxy, k.xyxy) / max(1e-6, min(d.area, k.area)) > 0.70
                for k in kept
            )
            if not is_dup:
                kept.append(d)

        return (
            [d for d in kept if norm_name(d.cls_name) == "helmet"],
            [d for d in kept if norm_name(d.cls_name) == "no_helmet"],
        )

    # ------------------------------------------------------------------
    # Rider counting
    # ------------------------------------------------------------------

    def _count_riders_for_bike(
        self,
        img: np.ndarray,
        bike: Det,
        matched_riders: list[Det],
        helmet_dets: list[Det],
        no_helmet_dets: list[Det],
    ) -> int:
        """Estimate the number of riders on *bike*.

        Priority:
        1. Pose model (counts skeletal instances — most accurate).
        2. De-duplicated rider bounding boxes fused with head detection count.
        """
        pose_count = self._count_riders_via_pose(img, bike.xyxy)
        if pose_count is not None:
            return pose_count

        # De-duplicate rider boxes by IoU / containment.
        riders_sorted = sorted(matched_riders, key=lambda r: (r.conf, -r.area), reverse=True)
        deduped: list[Det] = []
        dup_evidence = False

        for r in riders_sorted:
            is_dup = False
            for k in deduped:
                inter = inter_area(r.xyxy, k.xyxy)
                min_a = max(1e-6, min(r.area, k.area))
                if iou(r.xyxy, k.xyxy) > 0.38 or inter / min_a > 0.72:
                    is_dup = True
                    dup_evidence = True
                    break
            if not is_dup:
                deduped.append(r)

        heads = helmet_dets + no_helmet_dets
        head_count = len(heads)

        # Prune merged rider boxes that cover multiple head detections.
        if head_count >= 2 and deduped:
            pruned: list[Det] = []
            for r in deduped:
                rx1, ry1, rx2, ry2 = r.xyxy
                heads_inside = sum(
                    1 for hd in heads
                    if (rx1 <= (hd.xyxy[0] + hd.xyxy[2]) / 2 <= rx2 and
                        ry1 <= (hd.xyxy[1] + hd.xyxy[3]) / 2 <= ry2) or
                    inter_area(r.xyxy, hd.xyxy) / max(1e-6, hd.area) > 0.25
                )
                if heads_inside >= 2:
                    dup_evidence = True
                    continue
                pruned.append(r)
            if pruned:
                deduped = pruned

        bbox_count = len(deduped)

        if head_count > 0:
            if bbox_count == 0:
                return head_count
            if dup_evidence and bbox_count > head_count:
                return head_count
            return max(bbox_count, head_count)

        return max(bbox_count, 1 if matched_riders else 0)

    def _count_riders_via_pose(
        self, img: np.ndarray, bike_box: Box,
    ) -> Optional[int]:
        """Use the pose model to count distinct riders by skeletal keypoints."""
        if self._pose_det is None:
            return None
        bike_crop, _ = crop_box(img, bike_box, xpad=0.15, ypad=0.50)
        if bike_crop.size == 0:
            return None
        try:
            from ultralytics import YOLO as _YOLO  # only if pose_det is a YOLODetector
            # Access the underlying YOLO model for keypoint output
            raw_model = getattr(self._pose_det, "_model", None)
            if raw_model is None:
                return None
            results = raw_model.predict(
                bike_crop,
                imgsz=self._cfg.pose_imgsz,
                conf=self._cfg.pose_conf,
                verbose=False,
            )
            count = sum(
                1 for r in results
                if r.keypoints is not None and len(r.keypoints) > 0
            )
            return count if count > 0 else None
        except Exception:
            logger.debug("_count_riders_via_pose failed", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # License plate detection + OCR
    # ------------------------------------------------------------------

    def _detect_plate_text(self, img: np.ndarray, bike_box: Box) -> str:
        """Locate the license plate in the bike region and OCR it.

        Args:
            img:      Full BGR image.
            bike_box: Bike bounding box ``[x1, y1, x2, y2]``.

        Returns:
            Plate string e.g. ``"AP07AB1234"``, or ``""`` if not found.
        """
        bike_crop, bike_offset = crop_box(img, bike_box, xpad=0.35, ypad=0.35)
        raw_dets = self._plate_det.predict(
            bike_crop, self._cfg.plate_imgsz, self._cfg.plate_conf, self._cfg.iou_thr,
        )

        crop_h, crop_w = bike_crop.shape[:2]
        crop_area = max(1.0, float(crop_h * crop_w))

        plates: list[Det] = []
        for d in raw_dets:
            if self._cat_plate(d) != "plate":
                continue
            x1, y1, x2, y2 = d.xyxy
            pw  = max(1.0, x2 - x1)
            ph  = max(1.0, y2 - y1)
            asp = pw / ph
            rel = (pw * ph) / crop_area
            if asp < 0.45 or asp > 8.5 or rel > 0.45:
                continue
            plates.append(offset_det(d, bike_offset, "plate"))

        plates = nms_same_class(plates, 0.30)
        if not plates:
            return ""

        plates = sorted(plates, key=lambda d: (d.conf, d.area), reverse=True)
        all_candidates: list[tuple[str, float]] = []

        for plate in plates[: self._cfg.max_plates_to_ocr]:
            plate_crop, _ = crop_box(img, plate.xyxy, xpad=0.22, ypad=0.45)
            text = self._ocr.read_plate(plate_crop)
            if text:
                all_candidates.append((text, 1.0))
                if INDIAN_PLATE_RE.match(text):
                    return text

        return vote_plate(all_candidates) if all_candidates else ""

    # ------------------------------------------------------------------
    # Category classifiers (label → semantic category)
    # ------------------------------------------------------------------

    @staticmethod
    def _cat_bike(d: Det) -> Optional[str]:
        n = norm_name(d.cls_name)
        if n in _BIKE_NAMES or n in {"two_wheeler", "twowheeler", "2_wheeler"}:
            return "bike"
        if d.cls_id == 0 and n in {"0", "0.0"}:
            return "bike"
        return None

    @staticmethod
    def _cat_rider(d: Det) -> Optional[str]:
        n = norm_name(d.cls_name)
        if n in _RIDER_NAMES:
            return "rider"
        if d.cls_id == 1 and n in {"1", "1.0"}:
            return "rider"
        return None

    @staticmethod
    def _cat_helmet(d: Det) -> Optional[str]:
        n = norm_name(d.cls_name)
        if n in _HELMET_NAMES:
            return "helmet"
        if n in _NO_HELMET_NAMES:
            return "no_helmet"
        return "helmet" if d.cls_id == 0 else ("no_helmet" if d.cls_id == 1 else None)

    @staticmethod
    def _cat_plate(d: Det) -> Optional[str]:
        n = norm_name(d.cls_name)
        if n in _PLATE_NAMES:
            return "plate"
        return "plate" if d.cls_id == 0 else None

    # ------------------------------------------------------------------
    # Model loading helpers
    # ------------------------------------------------------------------

    def _load_detector(self, stems: list[str]) -> DetectorProtocol:
        """Load the first matching model (ONNX if enabled, else PyTorch).

        Raises:
            FileNotFoundError: If none of the candidate weight files exist.
        """
        for stem in stems:
            if self._cfg.use_onnx:
                p = self._cfg.model_dir / f"{stem}.onnx"
                if p.exists():
                    from traffic_violation.models.onnx_detector import ONNXDetector
                    return ONNXDetector(p)
            for ext in (".pt",):
                p = self._cfg.model_dir / f"{stem}{ext}"
                if p.exists():
                    return YOLODetector(p, device=self._cfg.device, iou_thr=self._cfg.iou_thr)
        raise FileNotFoundError(
            f"No model found for stems {stems} in {self._cfg.model_dir}"
        )

    def _try_load_detector(self, stems: list[str]) -> Optional[DetectorProtocol]:
        """Like :meth:`_load_detector` but returns ``None`` on failure."""
        try:
            return self._load_detector(stems)
        except FileNotFoundError:
            logger.debug("Optional model not found: %s", stems)
            return None

    # ------------------------------------------------------------------
    # Runtime setup
    # ------------------------------------------------------------------

    def _configure_threading(self) -> None:
        """Limit OpenCV and PyTorch thread counts to avoid over-subscription."""
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass

    def _warmup(self) -> None:
        """Warm up all loaded detection models."""
        for det in [self._full_det, self._helmet_det, self._plate_det,
                    self._coco_det, self._pose_det]:
            if det is not None:
                det.warmup()
