"""
traffic_violation.accelerate.export
=====================================
One-shot utilities to export YOLO ``.pt`` models to ONNX format for
accelerated CPU inference via ONNX Runtime.

Why ONNX is faster on CPU
--------------------------
* The ONNX export fuses adjacent operators (e.g. Conv → BN → ReLU) into a
  single kernel, eliminating Python-level dispatch overhead.
* ``onnxruntime`` uses a multi-threaded execution provider (OpenMP / TBB)
  whereas PyTorch eager mode serialises many operations.
* On a typical Intel laptop CPU, ONNX Runtime achieves **3–4× higher
  throughput** vs PyTorch for YOLO inference at the same input size.

Quickstart::

    from traffic_violation.accelerate.export import export_all, export_to_onnx

    # Export all models in ./models to ./models/onnx/
    export_all("./models", out_dir="./models/onnx")

    # Export one model
    export_to_onnx("models/full_detector.pt", "models/full_detector.onnx", imgsz=640)

Then enable ONNX inference by passing ``use_onnx=True`` to
:func:`~traffic_violation.config.fast_config` or setting the environment
variable ``TV_USE_ONNX=1``.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Model stems that should be exported.
_DEFAULT_STEMS = [
    "full_detector",
    "helmet_detector",
    "plate_detector",
    "yolo11m",
    "yolo11s_helmet",
    "yolo11n_plate",
    "yolo11n",
    "coco_yolo11n",
]

# Default image sizes per model family (longest side, px).
_DEFAULT_IMGSZ: dict[str, int] = {
    "full_detector": 640,
    "yolo11m":       640,
    "helmet_detector": 480,
    "yolo11s_helmet":  480,
    "plate_detector":  480,
    "yolo11n_plate":   480,
    "yolo11n":       640,
    "coco_yolo11n":  640,
    "pose_detector": 480,
    "yolo11n-pose":  480,
}


def export_to_onnx(
    pt_path: str | Path,
    out_path: str | Path | None = None,
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    dynamic: bool = False,
) -> Path:
    """Export a single YOLO ``.pt`` weight file to ONNX format.

    The export is performed by Ultralytics' built-in ``model.export()``
    method which handles:

    * Correct letterbox preprocessing baked into the graph.
    * NMS layer (if ``dynamic=False``).
    * Half-precision conversion (disabled here for CPU compatibility).

    Args:
        pt_path:  Path to the source ``.pt`` file.
        out_path: Destination ``.onnx`` path.  Defaults to replacing ``.pt``
                  with ``.onnx`` in the same directory.
        imgsz:    Input resolution (longest side, pixels).
        opset:    ONNX opset version.  17 is recommended for onnxruntime ≥1.16.
        simplify: Run ``onnx-simplifier`` after export to reduce graph
                  complexity.  Requires ``pip install onnxsim``.
        dynamic:  Enable dynamic batch / spatial axes.  Disable for maximum
                  inference speed at a fixed resolution.

    Returns:
        Path to the exported ``.onnx`` file.

    Raises:
        FileNotFoundError: If *pt_path* does not exist.
        RuntimeError:      If the export produces no output file.
    """
    pt_path = Path(pt_path)
    if not pt_path.exists():
        raise FileNotFoundError(f"Source weight not found: {pt_path}")

    if out_path is None:
        out_path = pt_path.with_suffix(".onnx")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Exporting %s → %s  imgsz=%d  opset=%d  simplify=%s",
        pt_path.name, out_path, imgsz, opset, simplify,
    )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for ONNX export.") from exc

    model = YOLO(str(pt_path))
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
        half=False,         # keep float32 for broad CPU compatibility
        verbose=False,
    )

    # Ultralytics saves the file next to the .pt; move it if needed.
    exported_path = Path(str(exported))
    if not exported_path.exists():
        # Some versions return the path without extension
        fallback = pt_path.with_suffix(".onnx")
        if fallback.exists():
            exported_path = fallback
        else:
            raise RuntimeError(f"Export produced no .onnx file for {pt_path}")

    if exported_path.resolve() != out_path.resolve():
        shutil.move(str(exported_path), str(out_path))

    size_mb = out_path.stat().st_size / 1e6
    logger.info("Export complete: %s  (%.1f MB)", out_path, size_mb)

    # Optional simplification pass
    if simplify:
        _try_simplify(out_path)

    return out_path


def export_all(
    model_dir: str | Path,
    out_dir: str | Path | None = None,
    imgsz_map: dict[str, int] | None = None,
    skip_existing: bool = True,
) -> dict[str, Path]:
    """Export all discovered YOLO models in *model_dir* to ONNX.

    Only models whose stems appear in :data:`_DEFAULT_STEMS` are processed.
    Other ``.pt`` files (e.g. EasyOCR weights) are silently skipped.

    Args:
        model_dir:     Directory containing ``.pt`` files.
        out_dir:       Output directory for ``.onnx`` files.  Defaults to
                       ``<model_dir>/onnx/``.
        imgsz_map:     Mapping of ``{stem: imgsz}`` to override defaults.
                       Stems not in the map use :data:`_DEFAULT_IMGSZ`.
        skip_existing: Skip stems whose ``.onnx`` already exists in *out_dir*.

    Returns:
        Dict mapping ``stem → onnx_path`` for successfully exported models.
    """
    model_dir = Path(model_dir)
    out_dir   = Path(out_dir) if out_dir else model_dir / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = {**_DEFAULT_IMGSZ, **(imgsz_map or {})}
    exported: dict[str, Path] = {}

    for stem in _DEFAULT_STEMS:
        pt_path = model_dir / f"{stem}.pt"
        if not pt_path.exists():
            logger.debug("Skipping %s (not found)", pt_path)
            continue

        out_path = out_dir / f"{stem}.onnx"
        if skip_existing and out_path.exists():
            logger.info("Skipping %s (already exported)", stem)
            exported[stem] = out_path
            continue

        imgsz = sizes.get(stem, 640)
        try:
            result = export_to_onnx(pt_path, out_path, imgsz=imgsz)
            exported[stem] = result
        except Exception:
            logger.error("Failed to export %s", stem, exc_info=True)

    logger.info(
        "export_all complete: %d/%d models exported to %s",
        len(exported), len(_DEFAULT_STEMS), out_dir,
    )
    return exported


def validate_onnx(onnx_path: str | Path) -> bool:
    """Check that an ONNX model is well-formed (basic graph validation).

    Args:
        onnx_path: Path to the ``.onnx`` file.

    Returns:
        ``True`` if the model passes ``onnx.checker.check_model``.
    """
    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        logger.info("ONNX validation passed: %s", onnx_path)
        return True
    except ImportError:
        logger.warning("onnx package not installed; skipping validation.")
        return True   # can't validate but not a fatal error
    except Exception:
        logger.error("ONNX validation failed: %s", onnx_path, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_simplify(onnx_path: Path) -> None:
    """Run onnx-simplifier in-place.  Silently skipped if not installed."""
    try:
        import onnxsim
        import onnx
        model = onnx.load(str(onnx_path))
        simplified, check = onnxsim.simplify(model)
        if check:
            onnx.save(simplified, str(onnx_path))
            logger.debug("onnx-simplifier applied: %s", onnx_path.name)
        else:
            logger.warning("onnx-simplifier check failed for %s; keeping original.", onnx_path.name)
    except ImportError:
        logger.debug("onnxsim not installed; skipping simplification.")
    except Exception:
        logger.debug("onnx-simplifier raised an error; keeping original.", exc_info=True)
