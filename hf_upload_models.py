"""
hf_upload_models.py
====================
One-time script: uploads the trained ONNX model weights to a Hugging Face
Model Hub repository so the HF Space can download them at startup.

Usage
-----
1. Install huggingface_hub:
       pip install huggingface_hub

2. Log in (runs a browser-based auth flow):
       huggingface-cli login

3. Run this script, passing your HF username:
       python hf_upload_models.py --username YOUR_HF_USERNAME

   This creates/updates the repo  YOUR_HF_USERNAME/traffic-sentinel-models
   and uploads all four .onnx files from ./models/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODELS = [
    "full_detector.onnx",
    "helmet_detector.onnx",
    "plate_detector.onnx",
    "yolo11n.onnx",
]
REPO_SUFFIX = "traffic-sentinel-models"


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload ONNX models to HF Hub")
    parser.add_argument("--username", required=True, help="Your Hugging Face username")
    parser.add_argument("--models-dir", default="./models", help="Local models directory")
    parser.add_argument("--private", action="store_true", help="Make the model repo private")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: Install huggingface_hub first:  pip install huggingface_hub")
        sys.exit(1)

    repo_id = f"{args.username}/{REPO_SUFFIX}"
    models_dir = Path(args.models_dir)

    api = HfApi()

    print(f"Creating/verifying repo: {repo_id}")
    create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)

    for filename in MODELS:
        src = models_dir / filename
        if not src.exists():
            print(f"  WARNING: {src} not found — skipping")
            continue
        size_mb = src.stat().st_size / 1e6
        print(f"  Uploading {filename} ({size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {filename}",
        )
        print(f"  {filename} uploaded.")

    print(f"\nDone! Model repo: https://huggingface.co/{repo_id}")
    print(f"\nAdd this as a secret in your HF Space:")
    print(f"  Name:  HF_MODEL_REPO")
    print(f"  Value: {repo_id}")


if __name__ == "__main__":
    main()
