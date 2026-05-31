"""
traffic_violation.models.yolo_detector
========================================
Ultralytics YOLO wrapper that satisfies :class:`~traffic_violation.models.base.DetectorProtocol`.

Responsibilities
----------------
* Load a single ``.pt`` weight file via Ultralytics.
* Expose a clean :meth:`predict` method that returns a ``list[Det]``.
* Handle device fallback (CUDA → CPU) transparently.
* Provide a :meth:`warmup` pass to avoid cold-start latency.
* Suppress Ultralytics' verbose console output.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from traffic_violation.utils.geometry import Det

logger = logging.getLogger(__name__)


class YOLODetector:
    """Thin wrapper around an Ultralytics YOLO model.

    Satisfies :class:`~traffic_violation.models.base.DetectorProtocol` without
    inheriting from it (structural subtyping).

    Args:
        weight_path: Path to a ``.pt`` YOLO weight file.
        device:      PyTorch device string, e.g. ``"cpu"`` or ``"cuda:0"``.
        iou_thr:     Default IoU threshold for the model's internal NMS.
                     Can be overridden per-call in :meth:`predict`.
    """

    def __init__(
        self,
        weight_path: str | Path,
        device: str = "cpu",
        iou_thr: float = 0.50,
    ) -> None:
        from ultralytics import YOLO  # local import keeps module lightweight

        self._path   = Path(weight_path)
        self._device = device
        self._iou    = iou_thr
        self._model  = YOLO(str(self._path))
        logger.info("YOLODetector loaded: %s  device=%s", self._path.name, device)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(
        self,
        img: np.ndarray,
        imgsz: int,
        conf: float,
        iou: float | None = None,
    ) -> list[Det]:
        """Run YOLO inference on *img*.

        Args:
            img:   BGR image array.
            imgsz: Longest-side resolution for letterboxed resizing.
            conf:  Minimum confidence threshold.
            iou:   IoU threshold for internal NMS; defaults to the value
                   provided at construction time.

        Returns:
            List of :class:`~traffic_violation.utils.geometry.Det` objects in
            the original image's pixel coordinates.
        """
        iou = iou if iou is not None else self._iou
        raw = self._run(img, imgsz, conf, iou)
        return self._parse(raw)

    def warmup(self) -> None:
        """Run a tiny dummy inference pass to initialise JIT / CUDA kernels."""
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            self._run(dummy, imgsz=64, conf=0.5, iou=0.5)
            logger.debug("YOLODetector warmup OK: %s", self._path.name)
        except Exception:
            logger.debug("YOLODetector warmup failed (non-fatal): %s", self._path.name, exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, img: np.ndarray, imgsz: int, conf: float, iou: float):
        """Execute model.predict with a device fallback."""
        kwargs = dict(imgsz=imgsz, conf=conf, iou=iou, verbose=False)
        try:
            return self._model.predict(img, device=self._device, **kwargs)
        except Exception:
            logger.debug(
                "YOLODetector._run: device=%s failed, retrying without device arg",
                self._device, exc_info=True,
            )
            return self._model.predict(img, **kwargs)

    @staticmethod
    def _parse(results) -> list[Det]:
        """Convert Ultralytics result objects to :class:`Det` instances."""
        dets: list[Det] = []
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
