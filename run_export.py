"""
Standalone ONNX export helper for the traffic violation project.

This script exports the three YOLO detector weights from ``./models/``:

* ``full_detector.pt``
* ``helmet_detector.pt``
* ``plate_detector.pt``

It uses :func:`traffic_violation.accelerate.export.export_all` so the export
logic stays in one place, then prints the resulting file sizes and validates
each exported model.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from traffic_violation.accelerate.export import export_all, validate_onnx


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO models to ONNX.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./models"),
        help="Directory containing full_detector.pt, helmet_detector.pt, plate_detector.pt",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write ONNX files to (default: same as model-dir)",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    out_dir = args.out_dir if args.out_dir is not None else args.model_dir
    exported = export_all(args.model_dir, out_dir=out_dir, skip_existing=False)
    if not exported:
        print("No models were exported.")
        return 1

    print("\nExport summary")
    print("--------------")
    for stem, onnx_path in sorted(exported.items()):
        if not onnx_path.exists():
            print(f"{stem}: export missing at {onnx_path}")
            continue

        size_mb = onnx_path.stat().st_size / 1e6
        is_valid = validate_onnx(onnx_path)
        status = "valid" if is_valid else "INVALID"
        print(f"{stem}: {onnx_path} | {size_mb:.2f} MB | {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
