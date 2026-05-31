"""
traffic_violation.utils.plate_utils
=====================================
Pure helper functions for Indian license-plate text processing.

No model or I/O dependencies — all functions are unit-testable in isolation.

Indian plate format (standard):
    <StateCode 2A><DistrictCode 2D><Series 1-3A><Number 4D>
    e.g.  AP07AB1234   MH12EF5678   TS09U1234
"""

from __future__ import annotations

import logging
import re

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex for a standard Indian plate
# ---------------------------------------------------------------------------

INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$")

# Noise tokens that appear on the blue "INDIA" strip of physical plates and
# bleed into OCR output.
_NOISE_PREFIXES: tuple[str, ...] = ("INDIA", "IND", "BH")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_plate_text(text: str | None) -> str:
    """Normalise raw OCR output to a plate-ready alphanumeric string.

    Steps applied in order:

    1. Convert to uppercase.
    2. Strip all characters that are not ``A-Z`` or ``0-9``.
    3. Remove known noise prefixes that originate from the blue INDIA strip.

    Args:
        text: Raw string returned by the OCR engine.  ``None`` is accepted
              and treated as an empty string.

    Returns:
        Cleaned string containing only ``[A-Z0-9]``, e.g. ``"AP07AB1234"``.
    """
    if text is None:
        return ""
    text = str(text).upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    for tok in _NOISE_PREFIXES:
        if text.startswith(tok) and len(text) > len(tok) + 3:
            text = text[len(tok):]
    return text.strip()


# ---------------------------------------------------------------------------
# Format scoring
# ---------------------------------------------------------------------------

def plate_format_score(text: str) -> float:
    """Score a candidate plate string by how well it matches the Indian format.

    Higher scores indicate a more probable real plate.  Used by
    :func:`vote_plate` to pick the best candidate from multiple OCR reads.

    Args:
        text: Cleaned (uppercase alphanumeric) plate string.

    Returns:
        A non-negative float score.  ``0.0`` means almost certainly not a plate.
    """
    n = len(text)
    if n < 4:
        return 0.0
    score = min(n, 10) / 10.0          # length bonus (maxes out at 10 chars)
    if INDIAN_PLATE_RE.match(text):
        score += 1.5                    # perfect format match
    elif n >= 6:
        score += 0.5                    # reasonable length
    return score


# ---------------------------------------------------------------------------
# Multi-candidate voting
# ---------------------------------------------------------------------------

def vote_plate(candidates: list[tuple[str, float]]) -> str:
    """Choose the best plate string from multiple ``(text, confidence)`` pairs.

    Strategy:

    1. Clean and filter obviously bad candidates (length < 4 or pure noise).
    2. Score each via ``plate_format_score × confidence``.
    3. Among same-length candidates, apply character-level voting to correct
       single-character OCR misreads that vary across preprocessing variants.

    Args:
        candidates: List of ``(raw_text, confidence)`` tuples from one or more
                    OCR reads.  Confidence should be in ``[0, 1]``.

    Returns:
        The single best plate string, or ``""`` if no valid candidate exists.
    """
    if not candidates:
        return ""

    pool: list[tuple[str, float]] = []
    for t, c in candidates:
        t = clean_plate_text(t)
        if len(t) < 4:
            continue
        try:
            c = float(c)
        except (TypeError, ValueError):
            c = 0.0
        pool.append((t[:12], c))   # hard-cap to avoid junk from OCR noise

    if not pool:
        return ""

    scored: dict[str, float] = {}
    for t, c in pool:
        s = plate_format_score(t) * max(c, 0.05)
        scored[t] = scored.get(t, 0.0) + s

    best_text = max(scored.items(), key=lambda kv: kv[1])[0]

    # Character-level voting over same-length candidates to fix 1-char errors.
    same_len = [(t, c) for t, c in pool if len(t) == len(best_text)]
    if len(same_len) < 2:
        return best_text

    out: list[str] = []
    for i in range(len(best_text)):
        votes: dict[str, float] = {}
        for text, conf in same_len:
            ch = text[i]
            votes[ch] = votes.get(ch, 0.0) + max(conf, 0.01)
        out.append(max(votes.items(), key=lambda kv: kv[1])[0])

    voted = "".join(out)
    logger.debug("vote_plate: best=%s  voted=%s  pool_size=%d", best_text, voted, len(pool))
    return voted if len(voted) >= len(best_text) else best_text


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def deskew_gray(gray: np.ndarray) -> np.ndarray:
    """Detect and correct small rotational skew in a grayscale plate crop.

    Uses Hough line detection to estimate the dominant text angle, then
    applies an affine rotation to align horizontal text.  Silently returns
    the original array if no reliable angle can be estimated.

    Args:
        gray: Single-channel (grayscale) plate crop.

    Returns:
        Deskewed grayscale array of the same shape, or the unchanged input.
    """
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(
            edges, 1, np.pi / 180,
            threshold=max(20, gray.shape[1] // 4),
        )
        if lines is None:
            return gray
        angles: list[float] = []
        for line in lines[:20]:
            theta = line[0][1]
            angle = (theta - np.pi / 2) * (180.0 / np.pi)
            if abs(angle) < 12:
                angles.append(angle)
        if not angles:
            return gray
        angle = float(np.median(angles))
        if abs(angle) < 0.5:
            return gray
        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except Exception:
        logger.debug("deskew_gray: skipped due to exception", exc_info=True)
        return gray


def make_plate_variants(crop: np.ndarray) -> list[np.ndarray]:
    """Generate a small set of high-value OCR preprocessing variants.

    Three variants are produced to cover common failure modes (low contrast,
    skew, binarisation artefacts) while keeping total OCR time bounded:

    1. Upscaled colour crop (baseline).
    2. CLAHE-enhanced + unsharp-masked grayscale (improves low-contrast plates).
    3. Otsu-binarised crop (cleans noisy backgrounds).

    Args:
        crop: BGR plate crop from the original image.

    Returns:
        A list of BGR ``np.ndarray`` images ready for EasyOCR.  Returns an
        empty list if the input crop is too small or empty.
    """
    if crop is None or crop.size == 0:
        return []
    h, w = crop.shape[:2]
    if h < 3 or w < 3:
        return []

    # Add a small border so characters at the edge are not clipped by OCR.
    crop = cv2.copyMakeBorder(
        crop,
        max(2, int(0.06 * h)), max(2, int(0.06 * h)),
        max(2, int(0.08 * w)), max(2, int(0.08 * w)),
        cv2.BORDER_REPLICATE,
    )

    h, w = crop.shape[:2]
    target_w = 320 if w < 160 else max(320, min(480, w * 2))
    scale = target_w / max(1, w)
    resized = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.6, blur, -0.6, 0)
    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return [
        resized,
        cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR),
    ]


def easyocr_regions_to_plate(results: list) -> tuple[str, float]:
    """Collapse multi-region EasyOCR output into a single plate string.

    EasyOCR with ``paragraph=False`` returns one entry per text region.
    Indian plates are often printed on two lines::

        Line 1 (top):    "AP 07"
        Line 2 (bottom): "AB 1234"

    Regions are sorted top-to-bottom by centroid Y and concatenated to
    reconstruct the full plate: ``"AP07AB1234"``.

    Args:
        results: Raw EasyOCR ``readtext`` output — a list of
                 ``(bbox_points, text, confidence)`` triples.

    Returns:
        A ``(full_plate_string, mean_confidence)`` tuple.  Both values are
        empty / zero if *results* is empty.
    """
    if not results:
        return "", 0.0

    def _mean_y(item: tuple) -> float:
        pts = item[0]
        return sum(p[1] for p in pts) / max(1, len(pts))

    sorted_results = sorted(results, key=_mean_y)

    parts = [clean_plate_text(item[1]) for item in sorted_results]
    confs = [float(item[2]) for item in sorted_results]

    full_text = "".join(p for p in parts if p)
    avg_conf = float(np.mean(confs)) if confs else 0.0

    return full_text, avg_conf
