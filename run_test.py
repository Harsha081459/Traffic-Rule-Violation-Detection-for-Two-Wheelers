"""
Quick benchmark script for the traffic violation detector.

What it does
------------
1. Loads the PyTorch backend and runs inference on a sample image.
2. Loads the ONNX backend and runs inference on the same image.
3. Prints the JSON output and a simple speed comparison in milliseconds.

If the ONNX files are missing, the script exports them on the fly into
``./models/`` so the benchmark remains self-contained.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from traffic_violation import TrafficViolationDetector
from traffic_violation.accelerate.export import export_all
from traffic_violation.config import fast_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs ONNX inference.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./models"),
        help="Directory containing the YOLO weights.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to a sample image. If omitted, the script searches locally.",
    )
    return parser.parse_args()


def _find_sample_image(explicit: Path | None) -> Path:
    if explicit is not None:
        if explicit.exists():
            return explicit
        raise FileNotFoundError(f"Image not found: {explicit}")

    candidates = [
        Path("./sample.jpg"),
        Path("./sample.jpeg"),
        Path("./sample.png"),
        Path("./static/sample.jpg"),
        Path("./static/sample.jpeg"),
        Path("./static/sample.png"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        found = sorted(Path(".").glob(pattern))
        if found:
            return found[0]

    static_dir = Path("./static")
    if static_dir.exists():
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            found = sorted(static_dir.glob(pattern))
            if found:
                return found[0]

    raise FileNotFoundError(
        "No sample image found. Pass one with --image path/to/file.jpg"
    )


def _ensure_onnx(model_dir: Path) -> None:
    required = [
        model_dir / "full_detector.onnx",
        model_dir / "helmet_detector.onnx",
        model_dir / "plate_detector.onnx",
    ]
    if all(path.exists() for path in required):
        return

    print("ONNX weights missing; exporting them now...")
    export_all(model_dir, out_dir=model_dir, skip_existing=False)


def _run_backend(label: str, detector: TrafficViolationDetector, image_path: Path) -> tuple[dict, float]:
    t0 = time.perf_counter()
    result = detector.predict(str(image_path))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"\n{label} output")
    print("--------------")
    print(json.dumps(result, indent=2))
    print(f"{label} inference time: {elapsed_ms:.2f} ms")
    return result, elapsed_ms


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    image_path = _find_sample_image(args.image)
    print(f"Using sample image: {image_path}")

    model_dir = args.model_dir
    _ensure_onnx(model_dir)

    pt_detector = TrafficViolationDetector(
        model_dir=str(model_dir),
        config=fast_config(model_dir, use_onnx=False),
    )
    onnx_detector = TrafficViolationDetector(
        model_dir=str(model_dir),
        config=fast_config(model_dir, use_onnx=True),
    )

    _, pt_ms = _run_backend("PyTorch", pt_detector, image_path)
    _, onnx_ms = _run_backend("ONNX", onnx_detector, image_path)

    if onnx_ms > 0:
        speedup = pt_ms / onnx_ms
        print("\nSpeed comparison")
        print("-----------------")
        print(f"PyTorch: {pt_ms:.2f} ms")
        print(f"ONNX:    {onnx_ms:.2f} ms")
        print(f"Speedup: {speedup:.2f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
