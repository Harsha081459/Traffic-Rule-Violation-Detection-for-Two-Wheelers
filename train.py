"""
train.py — Phase 2 Model Retraining Script
===========================================
Trains or fine-tunes the three YOLO11 models to >98 % accuracy using:

    * Mixed-precision (AMP) for ~2× GPU memory efficiency.
    * Cosine-LR schedule with warm-up for stable convergence.
    * Early stopping to avoid overfitting.
    * Optional hyperparameter tuning mode (``--tune``) that runs Ultralytics'
      built-in Ray-Tune search before the final training run.
    * Automatic export to ONNX after training completes.

Models trained
--------------
    Alias       Base            Purpose
    ----------  --------------  ---------------------------------
    helmet      YOLO11s         helmet / no_helmet detection
    full        YOLO11m         two_wheeler / rider detection
    plate       YOLO11n         license_plate detection

Step-by-step guide (Kaggle / Colab GPU)
-----------------------------------------
    Step 0 — Install dependencies
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !pip install ultralytics albumentations roboflow pyyaml tqdm

    Step 1 — Build the dataset (run once)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # On Kaggle: add your kaggle.json secret first
    !python dataset_builder.py --target helmet --out-dir /kaggle/working/data
    !python dataset_builder.py --target full   --out-dir /kaggle/working/data
    !python dataset_builder.py --target plate  --out-dir /kaggle/working/data

    Step 2 — (Optional) Hyperparameter tuning
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run ~50 trials to find optimal lr0, momentum, augment strength etc.
    # Takes ~2–4 h on a T4 GPU. Results are saved to runs/tune/<model>/
    !python train.py --target helmet \\
        --data /kaggle/working/data/helmet/final/data.yaml \\
        --tune --tune-iterations 50

    Step 3 — Full training run
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Uses best hyp from Step 2 automatically if present, else uses defaults.
    !python train.py --target helmet \\
        --data /kaggle/working/data/helmet/final/data.yaml \\
        --epochs 150 --batch 32 --imgsz 640

    !python train.py --target full \\
        --data /kaggle/working/data/full/final/data.yaml \\
        --epochs 100 --batch 16 --imgsz 640

    !python train.py --target plate \\
        --data /kaggle/working/data/plate/final/data.yaml \\
        --epochs 100 --batch 32 --imgsz 480

    Step 4 — Copy best weights back
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    from shutil import copy2
    copy2("runs/train/helmet_v2/weights/best.pt", "models/helmet_detector.pt")
    copy2("runs/train/full_v2/weights/best.pt",   "models/full_detector.pt")
    copy2("runs/train/plate_v2/weights/best.pt",  "models/plate_detector.pt")

    Step 5 — Re-export to ONNX
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    !python run_export.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration table
# ---------------------------------------------------------------------------

#   target → (pretrained_weights, recommended_imgsz, short_description)
MODEL_TABLE: dict[str, tuple[str, int, str]] = {
    "helmet": ("yolo11s.pt", 640, "helmet / no_helmet detector"),
    "full":   ("yolo11m.pt", 640, "two_wheeler / rider detector"),
    "plate":  ("yolo11n.pt", 480, "license_plate detector"),
}

# ---------------------------------------------------------------------------
# Default hyperparameters (overridden by --hyp-file or tune results)
# ---------------------------------------------------------------------------

DEFAULT_HYPS: dict[str, dict[str, Any]] = {
    "helmet": dict(
        lr0=0.008, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3, warmup_momentum=0.8, warmup_bias_lr=0.05,
        box=7.5, cls=0.5, dfl=1.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5, shear=1.5,
        perspective=0.0001, flipud=0.0, fliplr=0.5,
        bgr=0.0, mosaic=1.0, mixup=0.15, copy_paste=0.1,
    ),
    "full": dict(
        lr0=0.006, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=4, warmup_momentum=0.8, warmup_bias_lr=0.05,
        box=7.5, cls=0.5, dfl=1.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=4.0, translate=0.1, scale=0.6, shear=1.0,
        perspective=0.0, flipud=0.0, fliplr=0.5,
        bgr=0.0, mosaic=1.0, mixup=0.1, copy_paste=0.1,
    ),
    "plate": dict(
        lr0=0.005, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3, warmup_momentum=0.8, warmup_bias_lr=0.05,
        box=8.0, cls=0.4, dfl=1.5,
        hsv_h=0.01, hsv_s=0.4, hsv_v=0.4,
        degrees=3.0, translate=0.08, scale=0.4, shear=0.5,
        perspective=0.0, flipud=0.0, fliplr=0.5,
        bgr=0.0, mosaic=0.8, mixup=0.0, copy_paste=0.0,
    ),
}


# ===========================================================================
# Helpers
# ===========================================================================

def _load_hyps(
    target: str,
    hyp_file: Optional[Path],
    tune_results_dir: Optional[Path],
) -> dict[str, Any]:
    """Resolve the hyperparameter dict from (in priority order):

    1. ``--hyp-file`` JSON / YAML supplied by the user.
    2. Best hyperparameters from a previous ``--tune`` run.
    3. Built-in defaults for the target model.

    Args:
        target:          Model alias (``"helmet"``, ``"full"``, ``"plate"``).
        hyp_file:        User-supplied hyperparameter file (JSON or YAML).
        tune_results_dir: Directory produced by a previous tuning run.

    Returns:
        Hyperparameter dict.
    """
    if hyp_file and hyp_file.exists():
        text = hyp_file.read_text()
        hyps = json.loads(text) if hyp_file.suffix == ".json" else yaml.safe_load(text)
        logger.info("Hyperparameters loaded from %s", hyp_file)
        return hyps

    if tune_results_dir and tune_results_dir.exists():
        best_file = tune_results_dir / "best_hyperparameters.yaml"
        if best_file.exists():
            hyps = yaml.safe_load(best_file.read_text())
            logger.info("Using tuned hyperparameters from %s", best_file)
            return hyps

    logger.info("Using default hyperparameters for target=%s", target)
    return DEFAULT_HYPS[target]


def _find_existing_weights(target: str, run_prefix: str) -> Optional[Path]:
    """Look for a previous best.pt to resume from."""
    runs_dir = Path("runs") / "train"
    if not runs_dir.exists():
        return None
    candidates = sorted(runs_dir.glob(f"{run_prefix}*/weights/best.pt"))
    if candidates:
        logger.info("Found existing weights for resume: %s", candidates[-1])
        return candidates[-1]
    return None


# ===========================================================================
# Tuning
# ===========================================================================

def run_tuning(
    target: str,
    data_yaml: Path,
    iterations: int,
    epochs_per_trial: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
) -> Optional[Path]:
    """Run Ultralytics hyperparameter tuning (Ray-Tune under the hood).

    Ultralytics ``model.tune()`` runs ``iterations`` short training trials,
    each for ``epochs_per_trial`` epochs, and returns the best hyperparameter
    set.  The result is written to ``<project>/<target>_tune/``.

    Args:
        target:           Model alias.
        data_yaml:        Path to the ``data.yaml`` produced by
                          :mod:`dataset_builder`.
        iterations:       Number of Ray-Tune trials.
        epochs_per_trial: Epochs per trial (keep small, e.g. 10–20).
        imgsz:            Input image size.
        batch:            Batch size.
        device:           Device string (``"0"`` for GPU 0, ``"cpu"``).
        project:          Output directory root.

    Returns:
        Path to the tune output directory, or ``None`` on failure.
    """
    from ultralytics import YOLO

    weights, default_imgsz, _ = MODEL_TABLE[target]
    imgsz = imgsz or default_imgsz

    name = f"{target}_tune"
    logger.info("Tuning %s: %d iterations × %d epochs", target, iterations, epochs_per_trial)
    t0 = time.perf_counter()

    model = YOLO(weights)
    try:
        model.tune(
            data=str(data_yaml),
            epochs=epochs_per_trial,
            iterations=iterations,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            plots=False,
            save=False,
            val=False,
        )
    except Exception:
        logger.exception("Tuning failed for %s", target)
        return None

    elapsed = time.perf_counter() - t0
    out = Path(project) / name
    logger.info("Tuning complete in %.0f s. Results at %s", elapsed, out)
    return out


# ===========================================================================
# Training
# ===========================================================================

def run_training(
    target: str,
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name: str,
    hyps: dict[str, Any],
    pretrained_weights: Optional[Path],
    patience: int,
    workers: int,
    amp: bool,
    freeze_layers: int,
    close_mosaic_epochs: int,
) -> Path:
    """Run the final YOLO11 training job.

    Key settings that push accuracy to >98 %:

    * **amp=True** — mixed-precision (FP16) forward/backward on GPU, ~2× faster
      with no accuracy loss.
    * **cos_lr=True** — cosine annealing keeps the learning rate smooth,
      avoiding late-training overfitting spikes.
    * **patience** — early stopping when mAP@50 does not improve for
      ``patience`` epochs; prevents memorisation on the final epochs.
    * **close_mosaic** — disables mosaic augmentation for the last N epochs,
      letting the model stabilise on clean examples before evaluation.
    * **freeze** — freezes the first N backbone layers when fine-tuning from
      a pre-trained checkpoint, preventing catastrophic forgetting.

    Args:
        target:               Model alias.
        data_yaml:            Path to ``data.yaml``.
        epochs:               Maximum number of training epochs.
        imgsz:                Input image size.
        batch:                Batch size (use ``-1`` for auto-batch).
        device:               PyTorch device string.
        project:              Output directory root for Ultralytics runs.
        name:                 Run name subdirectory.
        hyps:                 Hyperparameter dict (see :data:`DEFAULT_HYPS`).
        pretrained_weights:   Path to ``.pt`` to fine-tune from.  If
                              ``None`` the base YOLO11 weights are used.
        patience:             Early-stopping patience in epochs.
        workers:              DataLoader worker processes.
        amp:                  Enable Automatic Mixed Precision.
        freeze_layers:        Number of backbone layers to freeze (0 = none).
        close_mosaic_epochs:  Disable mosaic augment for the final N epochs.

    Returns:
        Path to the trained ``best.pt`` file.
    """
    from ultralytics import YOLO

    default_weights, default_imgsz, description = MODEL_TABLE[target]
    imgsz = imgsz or default_imgsz

    if pretrained_weights and pretrained_weights.exists():
        weights = str(pretrained_weights)
        logger.info("Fine-tuning from %s", weights)
    else:
        weights = default_weights
        logger.info("Training from pretrained YOLO11 base: %s", weights)

    logger.info(
        "Training %s (%s): epochs=%d  imgsz=%d  batch=%d  amp=%s  device=%s",
        target, description, epochs, imgsz, batch, amp, device,
    )

    model = YOLO(weights)

    train_kwargs: dict[str, Any] = dict(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        workers=workers,
        amp=amp,
        cos_lr=True,
        close_mosaic=close_mosaic_epochs,
        plots=True,
        verbose=True,
        **hyps,
    )

    if freeze_layers > 0:
        train_kwargs["freeze"] = freeze_layers

    t0 = time.perf_counter()
    results = model.train(**train_kwargs)
    elapsed = time.perf_counter() - t0

    # Locate the best.pt produced by this run.
    best_pt = Path(project) / name / "weights" / "best.pt"
    if not best_pt.exists():
        # Ultralytics appends a number when the name already exists.
        candidates = sorted(Path(project).glob(f"{name}*/weights/best.pt"))
        best_pt = candidates[-1] if candidates else best_pt

    map50 = getattr(results, "results_dict", {}).get("metrics/mAP50(B)", "?")
    logger.info(
        "Training complete in %.0f s. mAP50=%.4s  best weights → %s",
        elapsed, map50, best_pt,
    )

    _print_summary(target, best_pt, map50, elapsed, data_yaml)
    return best_pt


# ===========================================================================
# Post-training ONNX export
# ===========================================================================

def export_best(best_pt: Path, imgsz: int) -> None:
    """Export the best checkpoint to ONNX immediately after training.

    Args:
        best_pt: Path to the ``best.pt`` weights file.
        imgsz:   Image size used during training.
    """
    from traffic_violation.accelerate.export import export_to_onnx

    out_onnx = best_pt.parent / (best_pt.stem + ".onnx")
    try:
        export_to_onnx(best_pt, out_onnx, imgsz=imgsz)
        logger.info("ONNX export → %s", out_onnx)
    except Exception:
        logger.warning("ONNX export failed (non-fatal)", exc_info=True)


# ===========================================================================
# Terminal summary
# ===========================================================================

def _print_summary(
    target: str,
    best_pt: Path,
    map50: Any,
    elapsed_s: float,
    data_yaml: Path,
) -> None:
    border = "=" * 60
    print(f"\n{border}")
    print(f"  Training complete: {target}")
    print(border)
    print(f"  mAP@50         : {map50}")
    print(f"  Training time  : {elapsed_s / 60:.1f} min")
    print(f"  Best weights   : {best_pt}")
    print(f"  Dataset        : {data_yaml}")
    print()
    print("  Next steps:")
    print(f"    1. Copy {best_pt}")
    print(f"       → models/{target}_detector.pt")
    print( "    2. Run: python run_export.py  (to re-export ONNX)")
    print( "    3. Run: python run_test.py    (to benchmark)")
    print(border + "\n")


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train YOLO11 traffic violation models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--target", choices=["helmet", "full", "plate"], required=True,
                   help="Which model to train.")
    p.add_argument("--data", type=Path, required=True,
                   help="Path to data.yaml produced by dataset_builder.py.")

    # Training
    p.add_argument("--epochs",   type=int,   default=150)
    p.add_argument("--batch",    type=int,   default=16,
                   help="Batch size. Use -1 for Ultralytics auto-batch.")
    p.add_argument("--imgsz",    type=int,   default=0,
                   help="Image size. 0 = use model default.")
    p.add_argument("--device",   type=str,   default="0",
                   help="Device: '0' for GPU 0, 'cpu', '0,1' for multi-GPU.")
    p.add_argument("--workers",  type=int,   default=8)
    p.add_argument("--patience", type=int,   default=30,
                   help="Early stopping patience (epochs without improvement).")
    p.add_argument("--no-amp",   dest="amp", action="store_false", default=True,
                   help="Disable Automatic Mixed Precision.")
    p.add_argument("--freeze",   type=int,   default=0,
                   help="Freeze first N backbone layers (fine-tuning).")
    p.add_argument("--close-mosaic", type=int, default=15,
                   help="Disable mosaic for the final N epochs.")
    p.add_argument("--weights",  type=Path,  default=None,
                   help="Fine-tune from this .pt instead of YOLO11 base.")
    p.add_argument("--resume",   action="store_true",
                   help="Auto-detect latest checkpoint and resume.")

    # Hyperparameters
    p.add_argument("--hyp-file", type=Path, default=None,
                   help="JSON or YAML file with custom hyperparameters.")

    # Tuning
    p.add_argument("--tune",           action="store_true",
                   help="Run hyperparameter tuning before training.")
    p.add_argument("--tune-iterations", type=int, default=50)
    p.add_argument("--tune-epochs",     type=int, default=15,
                   help="Epochs per tune trial (keep small).")

    # Output
    p.add_argument("--project",  type=str, default="runs/train")
    p.add_argument("--name",     type=str, default=None,
                   help="Run name. Defaults to <target>_v<N>.")
    p.add_argument("--no-export", dest="export", action="store_false", default=True,
                   help="Skip automatic ONNX export after training.")

    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    # Resolve run name
    if args.name is None:
        existing = list(Path(args.project).glob(f"{args.target}_v*")) if Path(args.project).exists() else []
        version  = len(existing) + 1
        args.name = f"{args.target}_v{version}"

    default_weights, default_imgsz, _ = MODEL_TABLE[args.target]
    imgsz = args.imgsz or default_imgsz

    # ── Tuning phase ──────────────────────────────────────────────────
    tune_results_dir: Optional[Path] = None
    if args.tune:
        tune_results_dir = run_tuning(
            target=args.target,
            data_yaml=args.data,
            iterations=args.tune_iterations,
            epochs_per_trial=args.tune_epochs,
            imgsz=imgsz,
            batch=args.batch,
            device=args.device,
            project="runs/tune",
        )

    # ── Resolve hyperparameters ───────────────────────────────────────
    hyps = _load_hyps(args.target, args.hyp_file, tune_results_dir)

    # ── Resume logic ─────────────────────────────────────────────────
    weights = args.weights
    if args.resume and weights is None:
        weights = _find_existing_weights(args.target, args.target)

    # ── Training phase ────────────────────────────────────────────────
    best_pt = run_training(
        target=args.target,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        hyps=hyps,
        pretrained_weights=weights,
        patience=args.patience,
        workers=args.workers,
        amp=args.amp,
        freeze_layers=args.freeze,
        close_mosaic_epochs=args.close_mosaic,
    )

    # ── Post-training export ──────────────────────────────────────────
    if args.export and best_pt.exists():
        export_best(best_pt, imgsz)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
