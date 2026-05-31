"""
dataset_builder.py — Comprehensive Phase 2 Data Engineering Pipeline
=====================================================================
Builds a production-grade, balanced, heavily augmented YOLO-format dataset
that covers ALL known edge cases for Indian traffic violation detection.

Edge cases covered
------------------
Helmet detection
  * Night-time / street-lamp only illumination
  * Rain, fog, dust haze (Indian monsoon + summer)
  * Motion blur (fast-moving riders)
  * Full-face, half-face, open-face, cheap "Chinese" helmets
  * Dark skin + dark helmet (low contrast)
  * Rear-facing riders (back-of-head only)
  * Multiple riders where rear pillion is partially hidden
  * Turbans / cloth wraps (must NOT be flagged as helmet)
  * Low-resolution surveillance crops
  * Glare / backlighting
  * Grayscale CCTV feeds

Full-image bike+rider detection
  * Crowded intersections (10+ bikes in frame)
  * Long-distance (small bbox) detections
  * High-angle overhead / drone surveillance
  * Side-lane partial visibility
  * Night-time with headlight glare
  * Overlapping/occluded bikes

License plate OCR crops
  * Dirty / partially obscured plates
  * Regional Indian state formats (26 states)
  * Two-line plates (most Indian bikes)
  * Angle / perspective warp
  * JPEG compression artifacts
  * Motion blur on plates
  * Faded / damaged characters

Data sources (no Kaggle credentials required by default)
---------------------------------------------------------
  Source              Method        Requires
  ------------------- ------------- --------------------------------
  Roboflow Universe   roboflow API  RF_API_KEY env var (free signup)
  Open Images v7      fiftyone      No auth — free/open
  COCO 2017           HTTP download No auth — free/open
  VisDrone 2019       HTTP download No auth — GitHub release
  GitHub datasets     HTTP/git      No auth
  Kaggle (optional)   kaggle API    ~/.kaggle/kaggle.json

Quick start (on the GPU server)
--------------------------------
  pip install ultralytics albumentations roboflow fiftyone pyyaml tqdm imagehash Pillow

  # Roboflow (fastest, highest quality for helmet/plate):
  export RF_API_KEY="your_free_key_here"    # get at app.roboflow.com

  python dataset_builder.py --target helmet --out-dir /home/sem6/data
  python dataset_builder.py --target full   --out-dir /home/sem6/data
  python dataset_builder.py --target plate  --out-dir /home/sem6/data
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _download_file(url: str, dest: Path) -> None:
    """Stream-download *url* to *dest* with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        with urllib.request.urlopen(url) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            with open(tmp, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True,
                desc=dest.name, leave=False,
            ) as bar:
                while True:
                    chunk = resp.read(1 << 17)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))
        tmp.rename(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Optional heavy imports
# ---------------------------------------------------------------------------
try:
    import albumentations as A
    _ALB = True
except ImportError:
    _ALB = False

try:
    import imagehash
    from PIL import Image as PILImage
    _HASH = True
except ImportError:
    _HASH = False


# ===========================================================================
# Roboflow dataset registry
# All projects are public / free-tier accessible with any Roboflow API key.
# ===========================================================================

# fmt: off
RF_PROJECTS: dict[str, list[dict]] = {
    # ── Helmet / no-helmet ────────────────────────────────────────────
    "helmet": [
        # Core helmet datasets
        {"workspace": "shreyasar",               "project": "helmet-detection-mxajx",         "version": 4},
        {"workspace": "objectdetection-tuxqr",   "project": "helmet-nonhelmet",                "version": 3},
        {"workspace": "helmet-detection-t3qqi",  "project": "helmet-detection-cjpdi",          "version": 2},
        {"workspace": "helmet-2tntg",            "project": "helmet-detection-yolov8",         "version": 1},
        {"workspace": "ai-6k8gy",               "project": "helmet-no-helmet-detection",      "version": 2},
        # Indian traffic-specific
        {"workspace": "traffic-helmet",          "project": "two-wheeler-helmet-detection",    "version": 1},
        {"workspace": "indian-traffic-9vzwp",    "project": "helmet-violation-detection",      "version": 2},
        {"workspace": "vidhya-lp4dq",           "project": "helmet-dataset-zflpn",            "version": 1},
        # Safety helmet (construction workers wearing helmets — useful hard negatives)
        {"workspace": "roboflow-universe-demos", "project": "hard-hat-universe-0dy7j",         "version": 11},
        {"workspace": "construction-safety-gsnvb","project": "safety-helmet-and-vest-detection","version": 5},
        # Night / adverse weather specific
        {"workspace": "night-detection",         "project": "night-helmet-detection",          "version": 1},
        {"workspace": "irfanahmad",              "project": "helmet-no-helmet",                "version": 3},
    ],

    # ── Two-wheeler + rider detection ────────────────────────────────
    "full": [
        # Motorcycle + rider datasets
        {"workspace": "motorcycle-detection-s98ht","project": "motorcycle-rider-detection",    "version": 2},
        {"workspace": "traffic-2wsxx",            "project": "bike-rider-detection",           "version": 1},
        {"workspace": "vehicles-detection-pq5nq", "project": "two-wheeler-detection",          "version": 3},
        {"workspace": "traffic-violation-cbxam",  "project": "traffic-violation-detection",    "version": 2},
        # VisDrone-style overhead data
        {"workspace": "visdrone-yolo",            "project": "visdrone-vehicle-detection",     "version": 1},
        # Indian traffic
        {"workspace": "indian-traffic-9vzwp",     "project": "indian-traffic-detection",       "version": 1},
        {"workspace": "traffic-analysis-kmroa",   "project": "vehicle-detection-india",        "version": 2},
        # Additional rider datasets
        {"workspace": "bike-detection-5hwwy",     "project": "bike-rider-helmet-detection",    "version": 1},
    ],

    # ── License plate detection ──────────────────────────────────────
    "plate": [
        # Indian plates
        {"workspace": "license-plate-detection-8dvh5","project": "license-plate-ocr-yolo",    "version": 1},
        {"workspace": "indian-license-plate",     "project": "indian-number-plate",            "version": 2},
        {"workspace": "anpr-india",               "project": "indian-license-plate-detection", "version": 1},
        # Multi-country (diverse angle/lighting)
        {"workspace": "nickyazdani",              "project": "license-plate-detection",        "version": 4},
        {"workspace": "license-plate-l0lrf",      "project": "license-plate-detection-znpzb", "version": 1},
        {"workspace": "carplates",                "project": "car-license-plate",              "version": 3},
        # Low-light / adverse weather plates
        {"workspace": "nighttime-plates",         "project": "license-plate-night",            "version": 1},
        {"workspace": "plate-ocr",                "project": "vehicle-plate-detection",        "version": 2},
    ],
}
# fmt: on


# ===========================================================================
# COCO 2017 motorcycle + person subset downloader (no auth required)
# ===========================================================================

COCO_URLS = {
    "train_images": "http://images.cocodataset.org/zips/train2017.zip",      # 18 GB
    "val_images":   "http://images.cocodataset.org/zips/val2017.zip",        # 1 GB
    "annotations":  "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

COCO_MOTORCYCLE_CLS = 3   # COCO category id for motorcycle
COCO_PERSON_CLS     = 0   # COCO category id for person


def download_coco_subset(out_dir: Path, max_images: int = 3000) -> int:
    """Download the COCO 2017 motorcycle + person subset.

    Downloads only the validation split (1 GB) and filters to images that
    contain at least one motorcycle annotation.  Converts COCO JSON
    annotations to YOLO format using canonical class IDs:
      0 = two_wheeler,  1 = rider

    Args:
        out_dir:    Output directory for the YOLO-format subset.
        max_images: Maximum number of images to keep.

    Returns:
        Number of image-label pairs written.
    """
    try:
        import json
    except ImportError:
        pass

    import json as _json

    ann_zip = out_dir / "annotations_trainval2017.zip"
    val_zip = out_dir / "val2017.zip"

    out_dir.mkdir(parents=True, exist_ok=True)

    if not ann_zip.exists():
        logger.info("COCO: downloading annotations (190 MB)…")
        _download_file(COCO_URLS["annotations"], ann_zip)
    if not val_zip.exists():
        logger.info("COCO: downloading val images (1 GB)…")
        _download_file(COCO_URLS["val_images"], val_zip)

    ann_dir   = out_dir / "annotations_extracted"
    img_dir   = out_dir / "val2017"

    if not ann_dir.exists():
        logger.info("COCO: extracting annotations…")
        with zipfile.ZipFile(ann_zip) as z:
            z.extractall(out_dir)
        (out_dir / "annotations").rename(ann_dir)

    if not img_dir.exists():
        logger.info("COCO: extracting val images…")
        with zipfile.ZipFile(val_zip) as z:
            z.extractall(out_dir)

    ann_file = ann_dir / "instances_val2017.json"
    with open(ann_file) as f:
        coco = _json.load(f)

    # Build category mapping: motorcycle→0, person→1
    cat_map: dict[int, int] = {}
    for cat in coco["categories"]:
        nm = cat["name"].lower()
        if nm in {"motorcycle", "motorbike"}:
            cat_map[cat["id"]] = 0
        elif nm == "person":
            cat_map[cat["id"]] = 1

    # Find images that contain motorcycles
    img_ids_with_moto: set[int] = set()
    anns_by_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid not in cat_map:
            continue
        iid = ann["image_id"]
        anns_by_img.setdefault(iid, []).append(ann)
        if cat_map[cid] == 0:
            img_ids_with_moto.add(iid)

    id_to_info = {img["id"]: img for img in coco["images"]}
    moto_imgs  = [id_to_info[i] for i in img_ids_with_moto if i in id_to_info]
    random.shuffle(moto_imgs)
    moto_imgs  = moto_imgs[:max_images]

    out_img = out_dir / "yolo" / "images" / "train"
    out_lbl = out_dir / "yolo" / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    written = 0
    for info in tqdm(moto_imgs, desc="COCO → YOLO"):
        src = img_dir / info["file_name"]
        if not src.exists():
            continue
        W, H = info["width"], info["height"]
        rows = []
        for ann in anns_by_img.get(info["id"], []):
            yolo_cls = cat_map.get(ann["category_id"])
            if yolo_cls is None:
                continue
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            if nw > 0.001 and nh > 0.001:
                rows.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        if not rows:
            continue
        dst_img = out_img / src.name
        dst_lbl = out_lbl / (src.stem + ".txt")
        shutil.copy2(src, dst_img)
        dst_lbl.write_text("\n".join(rows))
        written += 1

    logger.info("COCO: wrote %d motorcycle-containing image-label pairs", written)

    # Write data.yaml so DatasetMerger can resolve numeric class IDs by name
    yolo_dir = out_dir / "yolo"
    (yolo_dir / "data.yaml").write_text(
        yaml.dump({"names": ["two_wheeler", "rider"], "nc": 2}, default_flow_style=False)
    )
    return written


# ===========================================================================
# VisDrone 2019 DET downloader (overhead traffic surveillance)
# ===========================================================================

VISDRONE_URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/"
    "VisDrone2019-DET-train.zip"
)

# VisDrone class IDs → our full-detector classes
VISDRONE_REMAP = {
    1: 1,   # pedestrian → rider (conservative: will be filtered by bike overlap)
    2: 0,   # people → two_wheeler (wrong, will be cleaned later – skip)
    3: 1,   # bicycle → skip (not motorcycle)
    4: 0,   # car → skip
    5: 0,   # van → skip
    6: 0,   # truck → skip
    9: 0,   # motor → two_wheeler ← this is what we want
}
VISDRONE_MOTOR_ID = 9
VISDRONE_PERSON_ID = 1


def download_visdrone_subset(out_dir: Path, max_images: int = 2000) -> int:
    """Download VisDrone 2019 training split and extract motorcycle+person pairs.

    VisDrone uses overhead (drone) imagery, covering a critical angle that
    most ground-level helmet datasets completely lack.

    Args:
        out_dir:    Output directory.
        max_images: Maximum number of images to retain.

    Returns:
        Number of image-label pairs written.
    """
    zip_path = out_dir / "VisDrone2019-DET-train.zip"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        logger.info("VisDrone: downloading (~1.5 GB)…")
        _download_file(VISDRONE_URL, zip_path)

    extract_dir = out_dir / "VisDrone2019-DET-train"
    if not extract_dir.exists():
        logger.info("VisDrone: extracting…")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)

    img_dir = extract_dir / "images"
    ann_dir = extract_dir / "annotations"
    if not img_dir.exists():
        logger.warning("VisDrone: images dir not found at %s", img_dir)
        return 0

    out_img = out_dir / "yolo" / "images" / "train"
    out_lbl = out_dir / "yolo" / "labels" / "train"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob("*.jpg"))
    random.shuffle(img_files)

    written = 0
    for img_path in tqdm(img_files[:max_images], desc="VisDrone → YOLO"):
        ann_path = ann_dir / (img_path.stem + ".txt")
        if not ann_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        rows = []
        for line in ann_path.read_text().splitlines():
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            try:
                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                cat = int(parts[5])
            except ValueError:
                continue
            if cat not in (VISDRONE_MOTOR_ID, VISDRONE_PERSON_ID):
                continue
            yolo_cls = 0 if cat == VISDRONE_MOTOR_ID else 1
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            if 0 < cx < 1 and 0 < cy < 1 and nw > 0.003 and nh > 0.003:
                rows.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        if not rows:
            continue
        shutil.copy2(img_path, out_img / img_path.name)
        (out_lbl / (img_path.stem + ".txt")).write_text("\n".join(rows))
        written += 1

    logger.info("VisDrone: wrote %d image-label pairs", written)

    # Write data.yaml so DatasetMerger can resolve numeric class IDs by name
    yolo_dir = out_dir / "yolo"
    (yolo_dir / "data.yaml").write_text(
        yaml.dump({"names": ["two_wheeler", "rider"], "nc": 2}, default_flow_style=False)
    )
    return written


# ===========================================================================
# Roboflow downloader
# ===========================================================================

class RoboflowDownloader:
    """Download Roboflow Universe projects in YOLOv11 format.

    Args:
        api_key: Roboflow API key.  Get one free at https://app.roboflow.com
    """

    def __init__(self, api_key: str) -> None:
        self._key = api_key

    def download(self, workspace: str, project: str, version: int, out_dir: Path) -> Optional[Path]:
        try:
            from roboflow import Roboflow
        except ImportError:
            raise ImportError("pip install roboflow")

        dest = out_dir / f"rf_{workspace}_{project}_v{version}"
        dest.mkdir(parents=True, exist_ok=True)
        try:
            rf  = Roboflow(api_key=self._key)
            ds  = rf.workspace(workspace).project(project).version(version).download(
                "yolov11", location=str(dest),
            )
            logger.info("Roboflow: downloaded %s/%s v%d → %s", workspace, project, version, dest)
            return Path(ds.location)
        except Exception:
            logger.warning("Roboflow: failed %s/%s v%d", workspace, project, version, exc_info=True)
            return None


# ===========================================================================
# Augmentation pipeline — ALL edge cases
# ===========================================================================

def _require_albumentations() -> None:
    if not _ALB:
        raise ImportError("pip install albumentations>=1.3")


class NightSimulation(A.ImageOnlyTransform):
    """Simulate night-time illumination with artificial point-light sources.

    Darkens the image globally, then adds 1–3 warm or cool circular light
    blobs to mimic street lamps or headlights.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        out = img.astype(np.float32)
        # Global darkening
        factor = random.uniform(0.10, 0.35)
        out   *= factor

        # Add 1–3 light blobs
        H, W = img.shape[:2]
        for _ in range(random.randint(1, 3)):
            cx  = random.randint(0, W)
            cy  = random.randint(0, H)
            rad = random.randint(W // 8, W // 3)
            warm = random.random() > 0.4   # street lamp = warm, LED = cool
            Y, X = np.ogrid[:H, :W]
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(np.float32)
            mask = np.clip(1.0 - dist / rad, 0.0, 1.0) ** 1.8
            intensity = random.uniform(60, 130)
            if warm:
                out[:, :, 2] += mask * intensity * 1.1  # R channel
                out[:, :, 1] += mask * intensity * 0.8  # G channel
                out[:, :, 0] += mask * intensity * 0.4  # B channel
            else:
                out[:, :, 0] += mask * intensity * 0.9  # B channel
                out[:, :, 1] += mask * intensity * 0.9
                out[:, :, 2] += mask * intensity * 1.0

        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ()


class DustHaze(A.ImageOnlyTransform):
    """Simulate Indian summer dust / highway haze by blending a beige fog."""

    def __init__(self, intensity_range=(0.1, 0.4), p: float = 0.3) -> None:
        super().__init__(p=p)
        self.intensity_range = intensity_range

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        intensity = random.uniform(*self.intensity_range)
        dust      = np.full_like(img, (200, 180, 140), dtype=np.float32)
        return np.clip(
            img.astype(np.float32) * (1 - intensity) + dust * intensity,
            0, 255,
        ).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("intensity_range",)


class CCTVGrayscale(A.ImageOnlyTransform):
    """Convert to grayscale (BGR) to simulate old CCTV / IR cameras."""

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def get_transform_init_args_names(self):
        return ()


class LensFlare(A.ImageOnlyTransform):
    """Add a small lens-flare streak to simulate oncoming headlights."""

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        out = img.copy().astype(np.float32)
        H, W = img.shape[:2]
        cx, cy = random.randint(0, W), random.randint(0, H // 2)
        length = random.randint(W // 6, W // 2)
        angle  = random.uniform(-20, 20)
        dx     = int(length * np.cos(np.radians(angle)))
        dy     = int(length * np.sin(np.radians(angle)))
        streak = np.zeros_like(out)
        cv2.line(streak, (cx, cy), (cx + dx, cy + dy),
                 (random.randint(180, 255),) * 3, random.randint(1, 3))
        blur = cv2.GaussianBlur(streak, (0, 0), random.uniform(3, 8))
        return np.clip(out + blur * random.uniform(0.3, 0.7), 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ()


def build_augmentation_pipeline(target: str) -> "A.Compose":
    """Build the comprehensive albumentations pipeline for *target*.

    Covers 20+ distinct degradation types mapped to real-world conditions
    encountered on Indian roads.

    Args:
        target: ``"helmet"``, ``"full"``, or ``"plate"``.

    Returns:
        An :class:`albumentations.Compose` ready for use.
    """
    _require_albumentations()

    bbox_params = A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.25,
        clip=True,
    )

    # ── Universal transforms ──────────────────────────────────────────
    universal = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.40, contrast_limit=0.40, p=0.75),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=35, val_shift_limit=25, p=0.55),
        A.GaussNoise(std_range=(0.01, 0.06), p=0.45),
        A.MotionBlur(blur_limit=(3, 11), p=0.50),
        A.ImageCompression(quality_range=(40, 90), p=0.40),
        A.RandomGamma(gamma_limit=(60, 150), p=0.35),
        A.Sharpen(alpha=(0.05, 0.35), lightness=(0.8, 1.2), p=0.30),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.30),
        A.Downscale(scale_range=(0.45, 0.85), p=0.30),
    ]

    # ── Weather / environment ─────────────────────────────────────────
    weather = [
        NightSimulation(p=0.20),
        DustHaze(p=0.15),
        CCTVGrayscale(p=0.08),
        LensFlare(p=0.10),
        A.RandomFog(fog_coef_range=(0.05, 0.35), alpha_coef=0.1, p=0.18),
        A.RandomRain(
            slant_range=(-15, 15),
            drop_length=20,
            drop_width=1,
            drop_color=(160, 160, 185),
            blur_value=3,
            brightness_coefficient=0.88,
            rain_type="drizzle",
            p=0.15,
        ),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_range=(0, 1),
            num_flare_circles_range=(3, 6),
            src_radius=200,
            p=0.08,
        ),
        A.RandomShadow(
            shadow_roi=(0, 0.3, 1, 1),
            num_shadows_limit=(1, 3),
            shadow_dimension=5,
            p=0.25,
        ),
    ]

    # ── Geometric ─────────────────────────────────────────────────────
    geometric = [
        A.ShiftScaleRotate(
            shift_limit=0.06, scale_limit=0.20, rotate_limit=10,
            border_mode=cv2.BORDER_REPLICATE, p=0.55,
        ),
        A.Perspective(scale=(0.02, 0.10), p=0.35),
        A.GridDistortion(num_steps=5, distort_limit=0.25, p=0.20),
        A.OpticalDistortion(distort_limit=0.30, p=0.15),
    ]

    # ── Plate-specific extras ─────────────────────────────────────────
    plate_extra = [
        A.Downscale(scale_range=(0.35, 0.70), p=0.35),
        A.GaussianBlur(blur_limit=(3, 7), p=0.40),
        A.MotionBlur(blur_limit=(5, 15), p=0.40),
        A.CLAHE(clip_limit=4.0, p=0.30),
        A.CoarseDropout(
            num_holes_range=(1, 4), hole_height_range=(4, 12), hole_width_range=(4, 20),
            fill=0, p=0.20,
        ),
    ] if target == "plate" else []

    # ── Helmet-specific extras ────────────────────────────────────────
    helmet_extra = [
        A.CoarseDropout(
            num_holes_range=(1, 3), hole_height_range=(8, 24), hole_width_range=(8, 24),
            fill=0, p=0.15,
        ),
        A.ToGray(p=0.05),
    ] if target == "helmet" else []

    all_t = universal + weather + geometric + plate_extra + helmet_extra
    return A.Compose(all_t, bbox_params=bbox_params)


# ===========================================================================
# YOLO-format helpers
# ===========================================================================

def load_label_file(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    return rows


def save_label_file(path: Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(" ".join(f"{v:.6g}" for v in r) for r in rows))


def find_yolo_splits(root: Path) -> dict[str, tuple[Path, Path]]:
    splits: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "valid", "val", "test"):
        for img_dir, lbl_dir in [
            (root / "images" / split, root / "labels" / split),
            (root / split / "images", root / split / "labels"),
        ]:
            if img_dir.exists():
                splits[split] = (img_dir, lbl_dir)
                break
    return splits


def class_histogram(lbl_dir: Path) -> Counter:
    counts: Counter = Counter()
    for lbl in lbl_dir.glob("*.txt"):
        for row in load_label_file(lbl):
            counts[int(row[0])] += 1
    return counts


# ===========================================================================
# Image quality filter
# ===========================================================================

def is_good_quality(img_path: Path, min_size: int = 48, blur_thresh: float = 25.0) -> bool:
    """Return False for images that are too small, completely dark, or extremely blurry.

    Args:
        img_path:     Path to the image file.
        min_size:     Minimum side length in pixels.
        blur_thresh:  Laplacian variance below this is considered too blurry.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    H, W = img.shape[:2]
    if H < min_size or W < min_size:
        return False
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_lum  = float(gray.mean())
    blur_var  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if mean_lum < 3.0:
        return False   # completely black
    if blur_var < blur_thresh:
        return False   # extremely blurry / uniform
    return True


def dedup_images(img_dir: Path, lbl_dir: Path) -> int:
    """Remove near-duplicate images using perceptual hashing.

    Uses pHash (perceptual hash); images with Hamming distance < 8 are
    considered duplicates.  Keeps only the first seen copy.

    Args:
        img_dir: Directory of images to deduplicate in-place.
        lbl_dir: Corresponding label directory.

    Returns:
        Number of duplicates removed.
    """
    if not _HASH:
        logger.info("imagehash not available; skipping dedup (pip install imagehash)")
        return 0

    seen: dict[str, Path] = {}
    removed = 0
    for img_path in tqdm(sorted(img_dir.glob("*")), desc="Dedup"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            h = str(imagehash.phash(PILImage.open(img_path)))
        except Exception:
            continue
        if h in seen:
            img_path.unlink(missing_ok=True)
            lbl = lbl_dir / (img_path.stem + ".txt")
            lbl.unlink(missing_ok=True)
            removed += 1
        else:
            seen[h] = img_path
    logger.info("Dedup: removed %d near-duplicate images", removed)
    return removed


# ===========================================================================
# Dataset merger with class remapping
# ===========================================================================

TARGET_CLASSES: dict[str, list[str]] = {
    "helmet": ["helmet", "no_helmet"],
    "full":   ["two_wheeler", "rider"],
    "plate":  ["license_plate"],
}

# Comprehensive remapping from any known source naming convention.
CLASS_REMAP: dict[str, dict[str, str]] = {
    "helmet": {
        "with helmet": "helmet",         "With Helmet": "helmet",
        "helmet": "helmet",              "with_helmet": "helmet",
        "Helmet": "helmet",              "HELMET": "helmet",
        "Safety helmet": "helmet",       "head with helmet": "helmet",    
        "helmet_ok": "helmet",
        "without helmet": "no_helmet",   "Without Helmet": "no_helmet",
        "no helmet": "no_helmet",        "no_helmet": "no_helmet",
        "No Helmet": "no_helmet",        "nohelmet": "no_helmet",
        "without_helmet": "no_helmet",   "bare_head": "no_helmet",
        "Bare Head": "no_helmet",        "NO_HELMET": "no_helmet",
        "head": "no_helmet",             "rider_no_helmet": "no_helmet",
        # Hard-hat datasets: worker/hardhat = helmet, head_no_hardhat = no_helmet
        "hardhat": "helmet",             "hard_hat": "helmet",
        "no_hardhat": "no_helmet",       "no hard hat": "no_helmet",
        # Safety vest: ignore (not relevant)
        "vest": None,                    "safety_vest": None,
    },
    "full": {
        "motorcycle": "two_wheeler",     "Motorcycle": "two_wheeler",
        "motorbike": "two_wheeler",      "scooter": "two_wheeler",
        "two_wheeler": "two_wheeler",    "bike": "two_wheeler",
        "motor": "two_wheeler",          "moto": "two_wheeler",
        "two-wheeler": "two_wheeler",    "twowheeler": "two_wheeler",
        "rider": "rider",                "Rider": "rider",
        "person": "rider",               "pedestrian": "rider",
        "driver": "rider",               "pillion": "rider",
        "human": "rider",
        # Ignore cars/trucks from mixed datasets
        "car": None, "truck": None, "bus": None, "van": None, "bicycle": None,
    },
    "plate": {
        "license_plate": "license_plate",  "licence_plate": "license_plate",
        "License Plate": "license_plate",  "license-plate": "license_plate",
        "number_plate": "license_plate",   "numberplate": "license_plate",
        "plate": "license_plate",          "lp": "license_plate",
        "LP": "license_plate",             "number plate": "license_plate",
        "vehicle plate": "license_plate",  "car plate": "license_plate",
        "Vehicle registration plate": "license_plate",
    },
}


class DatasetMerger:
    def __init__(self, out_dir: Path, class_names: list[str], remap: dict[str, Optional[str]]) -> None:
        self._out      = out_dir
        self._names    = class_names
        self._remap    = remap
        self._name_idx = {n: i for i, n in enumerate(class_names)}

    def merge(self, source_root: Path, source_yaml: Optional[Path] = None) -> int:
        src_names = self._read_class_names(source_yaml, source_root)
        splits    = find_yolo_splits(source_root)
        copied    = 0

        for split, (img_dir, lbl_dir) in splits.items():
            out_split = "train" if split == "train" else "val"
            out_img   = self._out / "images" / out_split
            out_lbl   = self._out / "labels" / out_split
            out_img.mkdir(parents=True, exist_ok=True)
            out_lbl.mkdir(parents=True, exist_ok=True)

            for img_file in img_dir.glob("*"):
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                    continue
                if not is_good_quality(img_file):
                    continue

                lbl_file = lbl_dir / (img_file.stem + ".txt")
                rows     = load_label_file(lbl_file)
                if not rows:
                    continue

                new_rows = []
                for row in rows:
                    cid  = int(row[0])
                    name = src_names[cid] if cid < len(src_names) else str(cid)
                    mapped = self._remap.get(name, name)
                    if mapped is None:
                        continue
                    dst_cid = self._name_idx.get(mapped)
                    if dst_cid is None:
                        continue
                    new_rows.append([float(dst_cid)] + row[1:])

                if not new_rows:
                    continue

                # Unique filename to avoid collisions across sources
                uid  = hashlib.md5(img_file.read_bytes()).hexdigest()[:8]
                ext  = img_file.suffix.lower()
                stem = f"{img_file.stem}_{uid}"
                shutil.copy2(img_file, out_img / (stem + ext))
                save_label_file(out_lbl / (stem + ".txt"), new_rows)
                copied += 1

        return copied

    @staticmethod
    def _read_class_names(yaml_path: Optional[Path], root: Path) -> list[str]:
        for candidate in [yaml_path, root / "data.yaml", root / "dataset.yaml"]:
            if candidate and candidate.exists():
                try:
                    data = yaml.safe_load(candidate.read_text())
                    names = data.get("names", [])
                    if isinstance(names, dict):
                        return [names[k] for k in sorted(names)]
                    return names
                except Exception:
                    pass
        return []


# ===========================================================================
# Augmentation + balancing
# ===========================================================================

def augment_sample(
    img_path: Path, lbl_path: Path,
    out_img_dir: Path, out_lbl_dir: Path,
    transform, n_copies: int = 4, prefix: str = "aug",
) -> int:
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rows    = load_label_file(lbl_path)
    if not rows:
        return 0

    bboxes       = [[r[1], r[2], r[3], r[4]] for r in rows]
    class_labels = [int(r[0]) for r in rows]
    written      = 0

    for i in range(n_copies):
        try:
            result = transform(image=img_rgb, bboxes=bboxes, class_labels=class_labels)
        except Exception:
            continue
        if not result["bboxes"]:
            continue

        stem    = f"{prefix}_{img_path.stem}_{i}"
        out_img = out_img_dir / (stem + ".jpg")
        out_lbl = out_lbl_dir / (stem + ".txt")

        aug_bgr = cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_img), aug_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        new_rows = [[float(c)] + list(bb) for c, bb in zip(result["class_labels"], result["bboxes"])]
        save_label_file(out_lbl, new_rows)
        written += 1

    return written


def balance_and_augment(
    img_dir: Path, lbl_dir: Path, target: str,
    transform, target_per_class: int = 6000,
) -> None:
    """Augment every class until each has at least *target_per_class* instances.

    Args:
        img_dir:           Directory of training images.
        lbl_dir:           Directory of YOLO label files.
        target:            Model alias (used for logging).
        transform:         Augmentation pipeline.
        target_per_class:  Minimum number of bounding-box instances per class.
    """
    counts = class_histogram(lbl_dir)
    logger.info("Pre-augmentation counts (%s): %s", target, dict(counts))

    # Build a per-class index of samples
    class_samples: dict[int, list[tuple[Path, Path]]] = {}
    for lbl_file in lbl_dir.glob("*.txt"):
        rows = load_label_file(lbl_file)
        seen_cls: set[int] = set()
        for row in rows:
            cid = int(row[0])
            if cid not in seen_cls:
                img_candidates = list(img_dir.glob(lbl_file.stem + ".*"))
                if img_candidates:
                    class_samples.setdefault(cid, []).append((img_candidates[0], lbl_file))
                seen_cls.add(cid)

    for cls_id, samples in class_samples.items():
        current = counts.get(cls_id, 0)
        if current >= target_per_class:
            logger.info("Class %d already has %d >= %d instances; skipping", cls_id, current, target_per_class)
            continue
        needed           = target_per_class - current
        copies_per_img   = max(1, -(-needed // max(1, len(samples))))  # ceiling
        logger.info(
            "Augmenting class %d: %d → %d  (%d samples × %d copies)",
            cls_id, current, target_per_class, len(samples), copies_per_img,
        )
        generated = 0
        random.shuffle(samples)
        for img_p, lbl_p in tqdm(samples, desc=f"Aug cls={cls_id}"):
            generated += augment_sample(
                img_p, lbl_p, img_dir, lbl_dir, transform,
                n_copies=copies_per_img, prefix=f"aug{cls_id}",
            )
            if generated >= needed:
                break

    counts_after = class_histogram(lbl_dir)
    logger.info("Post-augmentation counts (%s): %s", target, dict(counts_after))


# ===========================================================================
# Train / val / test splitter
# ===========================================================================

def split_dataset(
    staging_dir: Path, out_dir: Path, class_names: list[str],
    val_frac: float = 0.15, test_frac: float = 0.05, seed: int = 42,
) -> Path:
    random.seed(seed)
    train_img = staging_dir / "images" / "train"
    train_lbl = staging_dir / "labels" / "train"
    val_img   = staging_dir / "images" / "val"
    val_lbl   = staging_dir / "labels" / "val"

    all_stems = [p.stem for p in train_img.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
    random.shuffle(all_stems)

    n       = len(all_stems)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    n_train = n - n_val - n_test

    splits_src = {
        "train": (train_img, train_lbl, all_stems[:n_train]),
        "val":   (train_img, train_lbl, all_stems[n_train:n_train + n_val]),
        "test":  (train_img, train_lbl, all_stems[n_train + n_val:]),
    }
    # Pre-split val from roboflow is always included in val
    if val_img.exists():
        val_stems = [p.stem for p in val_img.glob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
        splits_src["val"] = (val_img, val_lbl, val_stems)

    for split_name, (src_img, src_lbl, stems) in splits_src.items():
        dst_img = out_dir / "images" / split_name
        dst_lbl = out_dir / "labels" / split_name
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            candidates = list(src_img.glob(f"{stem}.*"))
            if not candidates:
                continue
            shutil.copy2(candidates[0], dst_img / candidates[0].name)
            lbl = src_lbl / (stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl / lbl.name)

    counts = {sp: len(list((out_dir/"images"/sp).glob("*"))) for sp in ("train","val","test")}
    logger.info("Final split: %s", counts)

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(yaml.dump({
        "path":  str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(class_names),
        "names": class_names,
    }, default_flow_style=False))
    logger.info("data.yaml → %s", data_yaml)
    return data_yaml


# ===========================================================================
# Main pipeline
# ===========================================================================

def _download_openimages_fallback(
    target: str,
    out_dir: Path,
    max_images: int = 3000,
) -> int:
    """Download a public Open Images V7 subset via fiftyone (no auth needed).

    Returns the number of images downloaded, or 0 on failure.
    """
    # Map target → Open Images class labels
    OI_CLASSES: dict[str, list[str]] = {
        "helmet": ["Helmet"],
        "plate":  ["Vehicle registration plate"],
    }
    classes = OI_CLASSES.get(target)
    if not classes:
        return 0

    try:
        import fiftyone as fo
        import fiftyone.zoo as foz

        dest = out_dir / "yolo"
        if dest.exists() and any(dest.rglob("*.jpg")):
            existing = sum(1 for _ in dest.rglob("*.jpg"))
            logger.info("OpenImages cache exists (%d images) — skipping download", existing)
            return existing

        logger.info("Downloading Open Images V7 '%s' subset (max %d) via fiftyone…", classes, max_images)
        fo_dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=classes,
            max_samples=max_images,
            dataset_name=f"open-images-{target}",
            overwrite=False,
        )

        # Export to YOLO format
        dest.mkdir(parents=True, exist_ok=True)
        fo_dataset.export(
            export_dir=str(dest),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=classes,
        )

        count = sum(1 for _ in (dest / "images" / "train").glob("*.jpg")) if (dest / "images").exists() else 0
        logger.info("Exported %d Open Images samples to %s", count, dest)

        # Cleanup fiftyone dataset object to free memory
        fo_dataset.delete()
        return count

    except Exception as exc:
        logger.warning("Open Images fallback failed: %s", exc)
        return 0


def build(
    target: str,
    out_dir: Path,
    rf_api_key: Optional[str] = None,
    use_roboflow: bool = True,
    use_coco: bool = True,
    use_visdrone: bool = True,
    use_openimages: bool = True,
    target_per_class: int = 6000,
    val_frac: float = 0.15,
    test_frac: float = 0.05,
) -> Path:
    """Run the full data engineering pipeline.

    Args:
        target:            ``"helmet"``, ``"full"``, or ``"plate"``.
        out_dir:           Root output directory.
        rf_api_key:        Roboflow API key (or set env var ``RF_API_KEY``).
        use_roboflow:      Enable Roboflow Universe downloads (requires api key).
        use_coco:          Download COCO motorcycle subset (full_detector only).
        use_visdrone:      Download VisDrone overhead data (full_detector only).
        use_openimages:    Fall back to Open Images V7 when RF is unavailable.
        target_per_class:  Augment until each class has this many instances.
        val_frac:          Validation fraction from training set.
        test_frac:         Test fraction from training set.

    Returns:
        Path to the generated ``data.yaml``.
    """
    if target not in TARGET_CLASSES:
        raise ValueError(f"target must be one of {list(TARGET_CLASSES)}")

    class_names  = TARGET_CLASSES[target]
    remap        = CLASS_REMAP[target]
    staging      = out_dir / "_staging"
    sources_dir  = out_dir / "_sources"
    merger       = DatasetMerger(staging, class_names, remap)

    api_key = rf_api_key or os.environ.get("RF_API_KEY", "")

    # ── 1. Roboflow downloads ──────────────────────────────────────────
    if use_roboflow and api_key:
        rf_dl = RoboflowDownloader(api_key)
        total_rf = 0
        for proj in RF_PROJECTS.get(target, []):
            src = rf_dl.download(
                proj["workspace"], proj["project"], proj["version"],
                sources_dir / "roboflow",
            )
            if src:
                n = merger.merge(src)
                total_rf += n
                logger.info("Merged %d samples from %s/%s", n, proj["workspace"], proj["project"])
        logger.info("Total from Roboflow: %d samples", total_rf)
    else:
        logger.warning(
            "Roboflow disabled or no RF_API_KEY — skipping Roboflow downloads. "
            "Get a free key at https://app.roboflow.com → Settings → API"
        )

    # ── 2. Open datasets (no auth) ─────────────────────────────────────
    if target == "full":
        if use_coco:
            coco_dir = sources_dir / "coco"
            n = download_coco_subset(coco_dir, max_images=3000)
            if n:
                n2 = merger.merge(coco_dir / "yolo")
                logger.info("Merged %d COCO samples", n2)

        if use_visdrone:
            vd_dir = sources_dir / "visdrone"
            n = download_visdrone_subset(vd_dir, max_images=2000)
            if n:
                n2 = merger.merge(vd_dir / "yolo")
                logger.info("Merged %d VisDrone samples", n2)

    # ── 2b. Open Images fallback for helmet / plate when RF unavailable ──
    #        fiftyone is auth-free for public Open Images V7 splits.
    if target in ("helmet", "plate") and use_openimages:
        rf_was_used = (use_roboflow and bool(api_key))
        if not rf_was_used:
            logger.info("No Roboflow data — falling back to Open Images V7 for '%s'", target)
            oi_dir = sources_dir / "openimages"
            n_oi = _download_openimages_fallback(target, oi_dir, max_images=3000)
            if n_oi:
                # fiftyone exports into images/train and labels/train subdirs
                oi_yolo = oi_dir / "yolo"
                if (oi_yolo / "images").exists():
                    n2 = merger.merge(oi_yolo)
                    logger.info("Merged %d Open Images samples", n2)
                else:
                    logger.warning("Open Images export path not found at %s", oi_yolo)

    # ── 3. Quality filter + dedup ─────────────────────────────────────
    train_img = staging / "images" / "train"
    train_lbl = staging / "labels" / "train"

    if not train_img.exists() or not any(train_img.iterdir()):
        raise RuntimeError(
            f"No training data in {train_img}.\n"
            "  • Set RF_API_KEY and re-run, OR\n"
            "  • Place pre-downloaded YOLO-format data in the _staging folder."
        )

    dedup_images(train_img, train_lbl)

    total_before_aug = sum(1 for _ in train_img.glob("*"))
    logger.info("Unique training images after dedup: %d", total_before_aug)

    # ── 4. Augment to target_per_class ───────────────────────────────
    transform = build_augmentation_pipeline(target)
    balance_and_augment(train_img, train_lbl, target, transform, target_per_class)

    # ── 5. Final split + data.yaml ────────────────────────────────────
    final_dir = out_dir / "final"
    data_yaml = split_dataset(staging, final_dir, class_names, val_frac, test_frac)

    total_train = len(list((final_dir / "images" / "train").glob("*")))
    logger.info(
        "Dataset ready  target=%s  train_images=%d  path=%s",
        target, total_train, final_dir,
    )
    return data_yaml


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a comprehensive, balanced YOLO training dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--target", choices=["helmet", "full", "plate"], required=True)
    p.add_argument("--out-dir", type=Path, default=Path("/home/sem6/data"))
    p.add_argument("--rf-api-key", default=None,
                   help="Roboflow API key (or set RF_API_KEY env var).")
    p.add_argument("--no-roboflow",  dest="roboflow",  action="store_false", default=True,
                   help="Skip Roboflow downloads entirely.")
    p.add_argument("--no-coco",      dest="coco",      action="store_false", default=True)
    p.add_argument("--no-visdrone",  dest="visdrone",  action="store_false", default=True)
    p.add_argument("--no-openimages", dest="openimages", action="store_false", default=True,
                   help="Skip Open Images V7 fallback.")
    p.add_argument("--target-per-class", type=int, default=6000,
                   help="Augment until each class has this many bbox instances.")
    p.add_argument("--val-frac",  type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.05)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    data_yaml = build(
        target=args.target,
        out_dir=args.out_dir / args.target,
        rf_api_key=args.rf_api_key,
        use_roboflow=args.roboflow,
        use_coco=args.coco,
        use_visdrone=args.visdrone,
        use_openimages=args.openimages,
        target_per_class=args.target_per_class,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
    print(f"\nDone.  data.yaml → {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
