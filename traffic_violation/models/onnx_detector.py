"""
traffic_violation.models.onnx_detector
=========================================
ONNX Runtime backend for YOLO inference.

This module provides a drop-in replacement for :class:`~traffic_violation.models.yolo_detector.YOLODetector`
that uses ``onnxruntime`` instead of PyTorch.  On a typical Intel/AMD CPU,
ONNX Runtime achieves **3–4× higher throughput** compared to PyTorch eager
mode for YOLO models, because:

* Operator graph is fused at export time.
* ``onnxruntime`` uses multi-threaded execution providers (OpenMP / TBB).
* No Python overhead from PyTorch's eager dispatch.

Prerequisites
-------------
1. Export the ``.pt`` weights to ONNX first::

       from traffic_violation.accelerate.export import export_to_onnx
       export_to_onnx("models/full_detector.pt", "models/full_detector.onnx")

2. Install ``onnxruntime``::

       pip install onnxruntime          # CPU only
       pip install onnxruntime-gpu      # GPU (CUDA)

YOLO ONNX output layout (Ultralytics export)
--------------------------------------------
The exported model outputs a single tensor of shape
``(1, num_classes + 4, num_anchors)`` in *transposed* format, or
``(1, num_anchors, num_classes + 4)`` depending on the export version.
This class handles both automatically.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from traffic_violation.utils.geometry import Det

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Minimum package version that exposes InferenceSession
_ONNXRUNTIME_MIN = "1.16.0"


class ONNXDetector:
    """YOLO detector backed by ONNX Runtime.

    Satisfies :class:`~traffic_violation.models.base.DetectorProtocol` without
    explicit inheritance.

    Args:
        onnx_path:     Path to the exported ``.onnx`` model file.
        conf_default:  Fallback confidence threshold if not supplied per-call.
        iou_default:   IoU threshold for the Python-side NMS pass.
        num_threads:   Number of CPU inference threads.  ``0`` = auto (uses
                       all available logical cores).
        class_names:   Optional mapping of ``{class_id: name}``.  When
                       omitted the class id is used as the name.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        conf_default: float = 0.25,
        iou_default:  float = 0.50,
        num_threads:  int   = 0,
        class_names:  dict[int, str] | None = None,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is not installed.  Run: pip install onnxruntime"
            ) from exc

        self._path         = Path(onnx_path)
        self._conf_default = conf_default
        self._iou_default  = iou_default
        self._names        = class_names or {}

        # Session options — limit threads to avoid over-subscription on a
        # shared inference server.
        opts = ort.SessionOptions()
        if num_threads > 0:
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = max(1, num_threads // 2)

        # Prefer CUDA if available; silently fall back to CPU.
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self._path), sess_options=opts, providers=providers)

        meta = self._session.get_inputs()[0]
        self._input_name: str = meta.name
        # Dynamic shapes are represented as strings; fall back to 640.
        h = meta.shape[2] if isinstance(meta.shape[2], int) else 640
        w = meta.shape[3] if isinstance(meta.shape[3], int) else 640
        self._model_hw: tuple[int, int] = (h, w)

        # Introspect output layout once at construction time.
        out_shape = self._session.get_outputs()[0].shape
        # Ultralytics ONNX exports: (1, 4+nc, na) [standard] or (1, na, 4+nc) [transposed].
        # _transposed=True means the output is ALREADY in (na, 4+nc) row-per-anchor layout
        # and no further .T is needed in _postprocess.
        # Standard export → (1, 6, 8400): out_shape[1]=6 < out_shape[2]=8400 → _transposed=False → will .T
        # Transposed export → (1, 8400, 6): out_shape[1]=8400 > out_shape[2]=6 → _transposed=True → skip .T
        self._transposed: bool = (
            len(out_shape) == 3 and
            isinstance(out_shape[1], int) and
            isinstance(out_shape[2], int) and
            out_shape[1] > out_shape[2]
        )
        out_layout = "(na,4+nc)" if self._transposed else "(4+nc,na)->T"
        logger.info(
            "ONNXDetector loaded: %s  hw=%s  out_layout=%s",
            self._path.name, self._model_hw, out_layout,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(
        self,
        img: np.ndarray,
        imgsz: int,
        conf: float | None = None,
        iou:  float | None = None,
    ) -> list[Det]:
        """Run ONNX inference on *img*.

        Args:
            img:   BGR image array.
            imgsz: Requested longest-side resolution.  ONNX models exported
                   with static shapes ignore this and always use the compiled
                   input size (``self._model_hw``).  This prevents
                   ``INVALID_ARGUMENT`` errors when the pipeline's recall
                   fallback requests a larger resolution than the export size.
            conf:  Confidence threshold (defaults to constructor value).
            iou:   IoU threshold for post-processing NMS.

        Returns:
            List of :class:`~traffic_violation.utils.geometry.Det` objects.
        """
        conf = conf if conf is not None else self._conf_default
        iou  = iou  if iou  is not None else self._iou_default

        # Static-shape ONNX models must always receive exactly the resolution
        # they were compiled with; ignore the caller's imgsz request.
        fixed_imgsz = self._model_hw[0]
        if imgsz != fixed_imgsz:
            logger.debug(
                "ONNXDetector: requested imgsz=%d overridden to model imgsz=%d",
                imgsz, fixed_imgsz,
            )

        orig_h, orig_w = img.shape[:2]
        blob, pad_info = self._preprocess(img, fixed_imgsz)

        raw_out = self._session.run(None, {self._input_name: blob})[0]

        return self._postprocess(raw_out, conf, iou, orig_h, orig_w, pad_info)

    def warmup(self) -> None:
        """Run a dummy inference pass to initialise ONNX Runtime internals."""
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            self.predict(dummy, imgsz=64)
            logger.debug("ONNXDetector warmup OK: %s", self._path.name)
        except Exception:
            logger.debug("ONNXDetector warmup failed (non-fatal): %s", self._path.name, exc_info=True)

    # ------------------------------------------------------------------
    # Pre / post processing
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        img: np.ndarray,
        imgsz: int,
    ) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """Letterbox-resize *img* and convert to an ONNX-ready float32 blob.

        Args:
            img:   BGR ``uint8`` image.
            imgsz: Longest-side target resolution.

        Returns:
            ``(blob, pad_info)`` where *blob* has shape ``(1, 3, H, W)``
            and *pad_info* is ``(scale, pad_top, pad_left, new_h, new_w)``.
        """
        orig_h, orig_w = img.shape[:2]
        scale = imgsz / max(orig_h, orig_w)
        new_h = int(round(orig_h * scale / 32)) * 32
        new_w = int(round(orig_w * scale / 32)) * 32

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Letterbox padding to a square if needed
        pad_h = imgsz - new_h
        pad_w = imgsz - new_w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right  = pad_w // 2, pad_w - pad_w // 2
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # BGR → RGB, HWC → CHW, [0,255] → [0,1]
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis]

        return blob, (scale, float(top), float(left), float(new_h), float(new_w))

    def _postprocess(
        self,
        raw: np.ndarray,
        conf_thr: float,
        iou_thr: float,
        orig_h: int,
        orig_w: int,
        pad_info: tuple[float, float, float, float, float],
    ) -> list[Det]:
        """Parse raw ONNX output tensor into :class:`Det` objects.

        Args:
            raw:      Raw model output tensor.
            conf_thr: Confidence threshold.
            iou_thr:  IoU threshold for NMS.
            orig_h:   Original image height before preprocessing.
            orig_w:   Original image width before preprocessing.
            pad_info: ``(scale, pad_top, pad_left, new_h, new_w)`` from
                      :meth:`_preprocess`.

        Returns:
            Filtered list of :class:`Det` objects in original coordinates.
        """
        scale, pad_top, pad_left, new_h, new_w = pad_info

        # Normalise to (num_anchors, 4 + num_classes)
        pred = raw[0]                       # remove batch dim → (4+nc, na) or (na, 4+nc)
        if not self._transposed:
            pred = pred.T                   # → (na, 4+nc)

        # Split box and class scores
        boxes_xywh = pred[:, :4]            # cx, cy, w, h  (in padded-resized space)
        scores     = pred[:, 4:]            # shape (na, nc)

        # Filter by confidence
        max_scores  = scores.max(axis=1)
        cls_ids     = scores.argmax(axis=1)
        mask        = max_scores >= conf_thr

        boxes_xywh = boxes_xywh[mask]
        max_scores  = max_scores[mask]
        cls_ids     = cls_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # Convert cx, cy, w, h → x1, y1, x2, y2 in padded-resized space
        cx, cy, bw, bh = (boxes_xywh[:, i] for i in range(4))
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Undo letterbox padding and scaling → original pixel coords
        x1 = np.clip((x1 - pad_left) / scale, 0, orig_w)
        y1 = np.clip((y1 - pad_top)  / scale, 0, orig_h)
        x2 = np.clip((x2 - pad_left) / scale, 0, orig_w)
        y2 = np.clip((y2 - pad_top)  / scale, 0, orig_h)

        # Python-side NMS (Ultralytics already does NMS in YOLO, but ONNX
        # export may omit it depending on opset/version).
        dets: list[Det] = []
        kept: list[int] = self._nms(
            np.stack([x1, y1, x2, y2], axis=1),
            max_scores,
            iou_thr,
        )

        for idx in kept:
            cid = int(cls_ids[idx])
            dets.append(Det(
                cls_id=cid,
                cls_name=self._names.get(cid, str(cid)),
                conf=float(max_scores[idx]),
                xyxy=[float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])],
            ))
        return dets

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_thr: float,
    ) -> list[int]:
        """Greedy NMS returning indices of surviving boxes.

        Args:
            boxes:   ``(N, 4)`` array of ``[x1, y1, x2, y2]``.
            scores:  ``(N,)`` confidence array.
            iou_thr: Suppression threshold.

        Returns:
            List of surviving row indices in *boxes*.
        """
        order = scores.argsort()[::-1]
        kept: list[int] = []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        while order.size > 0:
            i = int(order[0])
            kept.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            ix1 = np.maximum(x1[i], x1[rest])
            iy1 = np.maximum(y1[i], y1[rest])
            ix2 = np.minimum(x2[i], x2[rest])
            iy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
            union = areas[i] + areas[rest] - inter
            overlap = inter / np.maximum(union, 1e-6)
            order = rest[overlap <= iou_thr]

        return kept
