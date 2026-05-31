"""
download_models.py
==================
Downloads ONNX model weights from a Hugging Face Model Hub repository into
the local ./models/ directory before the FastAPI server starts.

Environment variables
---------------------
HF_MODEL_REPO   (required)  HF repo id, e.g. "your-username/traffic-sentinel-models"
HF_TOKEN        (optional)  HF access token for private repos
MODELS_DIR      (optional)  Local directory to store models (default: ./models)

Usage
-----
    python download_models.py          # standard startup
    python download_models.py --check  # only verify; exit 1 if any model missing
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REQUIRED_MODELS = [
    "full_detector.onnx",
    "helmet_detector.onnx",
    "plate_detector.onnx",
    "yolo11n.onnx",
]


def _log(msg: str) -> None:
    print(f"[download_models] {msg}", flush=True)


def download(models_dir: Path, repo_id: str, token: str | None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _log("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    models_dir.mkdir(parents=True, exist_ok=True)

    missing = [f for f in REQUIRED_MODELS if not (models_dir / f).exists()]
    if not missing:
        return

    _log(f"  Downloading {len(missing)} file(s) via snapshot_download ...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            token=token,
            local_dir=str(models_dir),
            allow_patterns=["*.onnx"],
            ignore_patterns=[".gitattributes", "*.md"],
        )
    except Exception as exc:
        _log(f"  ERROR during snapshot_download: {exc}")
        sys.exit(1)

    for filename in REQUIRED_MODELS:
        dest = models_dir / filename
        if dest.exists():
            size_mb = dest.stat().st_size / 1e6
            _log(f"  {filename} ready ({size_mb:.1f} MB)")
        else:
            _log(f"  ERROR: {filename} not found after download")
            sys.exit(1)


def check_only(models_dir: Path) -> int:
    missing = [f for f in REQUIRED_MODELS if not (models_dir / f).exists()]
    if missing:
        _log(f"Missing models: {missing}")
        return 1
    _log("All models present.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Only verify presence, do not download")
    args = parser.parse_args()

    models_dir = Path(os.environ.get("MODELS_DIR", "./models"))

    if args.check:
        sys.exit(check_only(models_dir))

    # Check if already complete
    all_present = all((models_dir / f).exists() for f in REQUIRED_MODELS)
    if all_present:
        _log("All models already present — nothing to download.")
        return

    repo_id = os.environ.get("HF_MODEL_REPO", "").strip()
    if not repo_id:
        _log(
            "ERROR: HF_MODEL_REPO environment variable is not set.\n"
            "       Set it to your Hugging Face model repo, e.g.:\n"
            "         HF_MODEL_REPO=your-username/traffic-sentinel-models"
        )
        sys.exit(1)

    token = os.environ.get("HF_TOKEN") or None
    _log(f"Downloading models from HF Hub: {repo_id}")
    download(models_dir, repo_id, token)
    _log("All models ready.")


if __name__ == "__main__":
    main()
