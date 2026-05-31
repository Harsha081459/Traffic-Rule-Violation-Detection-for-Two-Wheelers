"""
traffic_violation.config
=========================
Central configuration for the traffic-violation detection pipeline.

All tunable hyper-parameters, image sizes, confidence thresholds, and
feature flags live here as a single frozen :class:`PipelineConfig` dataclass.
This eliminates magic numbers scattered across the codebase and makes it
trivial to swap fast/accurate presets or serialise config to JSON/YAML.

Usage::

    from traffic_violation.config import PipelineConfig, fast_config, accurate_config

    cfg = fast_config(model_dir="./models")
    # or build explicitly:
    cfg = PipelineConfig(model_dir=Path("./models"), fast_mode=True, ...)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable runtime configuration for :class:`~traffic_violation.pipeline.TrafficViolationDetector`.

    Attributes:
        model_dir:          Path to the directory containing all ``.pt`` /
                            ``.onnx`` model weights.
        device:             PyTorch device string — ``"cpu"`` or ``"cuda:0"``.
        use_onnx:           When ``True``, prefer ONNX Runtime over PyTorch
                            for YOLO inference (faster CPU throughput).
        fast_mode:          When ``True``, uses smaller image sizes and fewer
                            OCR variants to stay under the 5-second budget.
        use_pose:           When ``True``, load and run the pose model for more
                            accurate rider counting at the cost of extra latency.

        full_imgsz:         Input resolution for the full-image bike/rider
                            detector (pixels, longest side).
        helmet_imgsz:       Input resolution for the per-bike helmet detector.
        plate_imgsz:        Input resolution for the license-plate detector.
        pose_imgsz:         Input resolution for the optional pose estimator.

        full_conf:          Minimum confidence for the full-image detector.
        helmet_conf:        Minimum confidence for the helmet detector.
        plate_conf:         Minimum confidence for the plate detector.
        pose_conf:          Minimum confidence for the pose estimator.
        iou_thr:            IoU threshold used in NMS across all detectors.

        max_bikes:          Maximum number of bikes to process per image
                            (sorted by confidence × area descending).
        max_plates_to_ocr:  Maximum number of plate boxes to run OCR on per bike.
        max_ocr_variants:   Maximum number of preprocessing variants sent to
                            EasyOCR per plate crop.

        n_workers:          Number of threads in the ``ThreadPoolExecutor``
                            used for concurrent per-bike processing.
                            ``0`` means auto-select (``min(4, n_bikes)``).
    """

    model_dir:         Path

    device:            str   = "cpu"
    use_onnx:          bool  = False
    fast_mode:         bool  = True
    use_pose:          bool  = False

    # --- image sizes --------------------------------------------------------
    full_imgsz:        int   = 640
    helmet_imgsz:      int   = 480
    plate_imgsz:       int   = 480
    pose_imgsz:        int   = 480

    # --- confidence thresholds ----------------------------------------------
    full_conf:         float = 0.25
    helmet_conf:       float = 0.22
    plate_conf:        float = 0.12
    pose_conf:         float = 0.25
    iou_thr:           float = 0.50

    # --- processing limits --------------------------------------------------
    max_bikes:         int   = 4
    max_plates_to_ocr: int   = 2
    max_ocr_variants:  int   = 1

    # --- concurrency --------------------------------------------------------
    n_workers:         int   = 0


# ---------------------------------------------------------------------------
# Named preset factories
# ---------------------------------------------------------------------------

def fast_config(
    model_dir: str | Path,
    *,
    use_onnx: bool = False,
    use_pose: bool = False,
) -> PipelineConfig:
    """Return a fast preset tuned for ≤5 s per image (evaluation mode).

    Args:
        model_dir: Path to the model weights directory.
        use_onnx:  Activate ONNX Runtime inference for additional CPU speedup.
        use_pose:  Enable pose-based rider counting (adds latency).

    Returns:
        A frozen :class:`PipelineConfig` instance.
    """
    device = _auto_device()
    return PipelineConfig(
        model_dir=Path(model_dir),
        device=device,
        use_onnx=use_onnx,
        fast_mode=True,
        use_pose=use_pose,
        full_imgsz=640,
        helmet_imgsz=480,
        plate_imgsz=480,
        pose_imgsz=480,
        max_bikes=4,
        max_plates_to_ocr=2,
        max_ocr_variants=1,
        n_workers=0,
    )


def accurate_config(
    model_dir: str | Path,
    *,
    use_onnx: bool = False,
    use_pose: bool = True,
) -> PipelineConfig:
    """Return a high-accuracy preset for offline / local experiments.

    Args:
        model_dir: Path to the model weights directory.
        use_onnx:  Activate ONNX Runtime inference.
        use_pose:  Enable pose-based rider counting (recommended for accuracy).

    Returns:
        A frozen :class:`PipelineConfig` instance.
    """
    device = _auto_device()
    return PipelineConfig(
        model_dir=Path(model_dir),
        device=device,
        use_onnx=use_onnx,
        fast_mode=False,
        use_pose=use_pose,
        full_imgsz=960,
        helmet_imgsz=640,
        plate_imgsz=960,
        pose_imgsz=640,
        max_bikes=8,
        max_plates_to_ocr=3,
        max_ocr_variants=3,
        n_workers=0,
    )


def config_from_env(model_dir: str | Path) -> PipelineConfig:
    """Build a :class:`PipelineConfig` driven by environment variables.

    Recognised variables (all optional):

    ==================  ==========  ==============================================
    Variable            Default     Description
    ==================  ==========  ==============================================
    ``TV_ACCURATE_MODE``  ``0``     Set to ``1`` to use the accurate preset.
    ``TV_USE_POSE``       ``0``     Set to ``1`` to enable pose estimation.
    ``TV_USE_ONNX``       ``0``     Set to ``1`` to use ONNX Runtime.
    ==================  ==========  ==============================================

    Args:
        model_dir: Path to the model weights directory.

    Returns:
        A frozen :class:`PipelineConfig` instance.
    """
    accurate = os.environ.get("TV_ACCURATE_MODE", "0") == "1"
    use_pose = os.environ.get("TV_USE_POSE", "0") == "1"
    use_onnx = os.environ.get("TV_USE_ONNX", "0") == "1"

    factory = accurate_config if accurate else fast_config
    return factory(model_dir, use_onnx=use_onnx, use_pose=use_pose)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    """Return ``"cuda:0"`` if CUDA is available, otherwise ``"cpu"``."""
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
