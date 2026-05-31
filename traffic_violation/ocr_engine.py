"""
traffic_violation.ocr_engine
==============================
EasyOCR-based license-plate reading engine.

All OCR concerns are encapsulated here:
    * EasyOCR model loading (local weights only, no internet).
    * Multi-variant plate preprocessing.
    * Multi-region top-to-bottom text assembly.
    * Confidence-weighted character-level voting across variants.

The single public method :meth:`OCREngine.read_plate` accepts a BGR crop and
returns the best plate string (e.g. ``"AP07AB1234"``).

If EasyOCR is unavailable or the local model directory is missing, the engine
degrades gracefully: every call to :meth:`read_plate` returns ``""``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from traffic_violation.utils.plate_utils import (
    INDIAN_PLATE_RE,
    easyocr_regions_to_plate,
    make_plate_variants,
    vote_plate,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional EasyOCR import
# ---------------------------------------------------------------------------
try:
    import easyocr as _easyocr_lib
    _EASYOCR_AVAILABLE = True
except ImportError:
    _easyocr_lib = None       # type: ignore[assignment]
    _EASYOCR_AVAILABLE = False

_OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class OCREngine:
    """License-plate OCR engine backed by EasyOCR.

    Args:
        model_dir: Path to the root model directory.  EasyOCR weights must be
                   present in ``<model_dir>/easyocr/`` for the engine to
                   activate.  If missing, all :meth:`read_plate` calls return
                   ``""`` silently.
        gpu:       Use GPU inference if ``True``.  Automatically disabled when
                   no CUDA device is detected.
        max_variants: Maximum number of preprocessing variants to send to
                      EasyOCR per plate crop.  Fewer = faster; more = higher
                      accuracy ceiling.
    """

    def __init__(
        self,
        model_dir: str | Path,
        gpu: bool = False,
        max_variants: int = 1,
    ) -> None:
        self._max_variants = max_variants
        self._reader: Optional[object] = self._load(Path(model_dir), gpu)

        if self._reader is None:
            logger.warning(
                "OCREngine: EasyOCR not available or model weights not found "
                "in %s/easyocr/. Plate OCR disabled.", model_dir,
            )
        else:
            logger.info("OCREngine: EasyOCR ready  gpu=%s  max_variants=%d", gpu, max_variants)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """``True`` when EasyOCR is loaded and ready for inference."""
        return self._reader is not None

    def read_plate(self, plate_crop: np.ndarray) -> str:
        """Read the license plate text from a BGR plate crop.

        Generates up to :attr:`max_variants` preprocessed versions of the crop,
        runs EasyOCR on each, assembles multi-line regions top-to-bottom, then
        applies confidence-weighted voting to pick the best plate string.

        Short-circuits early when a valid Indian plate format is recognised.

        Args:
            plate_crop: BGR image crop focused on the license plate.

        Returns:
            Best-voted plate string (uppercase alphanumeric), e.g.
            ``"AP07AB1234"``.  Returns ``""`` if OCR is unavailable or no
            text was detected.
        """
        if self._reader is None or plate_crop is None or plate_crop.size == 0:
            return ""

        candidates: list[tuple[str, float]] = []

        for variant in make_plate_variants(plate_crop)[: self._max_variants]:
            try:
                result = self._reader.readtext(  # type: ignore[union-attr]
                    variant,
                    detail=1,
                    paragraph=False,
                    allowlist=_OCR_ALLOWLIST,
                    decoder="greedy",
                    batch_size=1,
                )
            except Exception:
                logger.debug("OCREngine.read_plate: readtext raised", exc_info=True)
                continue

            if not result:
                continue

            full_text, full_conf = easyocr_regions_to_plate(result)
            if len(full_text) >= 4:
                candidates.append((full_text, full_conf))
                if INDIAN_PLATE_RE.match(full_text) or len(full_text) >= 7:
                    logger.debug("OCREngine: early exit on good plate %s", full_text)
                    return full_text

            # Add per-region fragments as shorter candidates for the voter.
            for item in result:
                try:
                    from traffic_violation.utils.plate_utils import clean_plate_text
                    t = clean_plate_text(item[1])
                    c = float(item[2])
                    if len(t) >= 3:
                        candidates.append((t, c))
                except Exception:
                    continue

        result_text = vote_plate(candidates)
        logger.debug("OCREngine.read_plate → %r  candidates=%d", result_text, len(candidates))
        return result_text

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load(model_dir: Path, gpu: bool) -> Optional[object]:
        """Load EasyOCR from a local model directory.

        No internet access is attempted.  Returns ``None`` on any failure.
        """
        if not _EASYOCR_AVAILABLE:
            return None

        ocr_dir = model_dir / "easyocr"
        try:
            has_weights = ocr_dir.exists() and any(ocr_dir.rglob("*.pth"))
        except Exception:
            has_weights = False

        if not has_weights:
            return None

        common_kwargs: dict = dict(
            gpu=gpu,
            model_storage_directory=str(ocr_dir),
            download_enabled=False,
            verbose=False,
        )

        # Try with user_network_directory first (some EasyOCR versions require it).
        try:
            return _easyocr_lib.Reader(  # type: ignore[union-attr]
                ["en"],
                user_network_directory=str(ocr_dir / "user_network"),
                **common_kwargs,
            )
        except TypeError:
            pass
        except Exception:
            logger.debug("OCREngine._load: first attempt failed", exc_info=True)
            return None

        try:
            return _easyocr_lib.Reader(["en"], **common_kwargs)  # type: ignore[union-attr]
        except Exception:
            logger.debug("OCREngine._load: second attempt failed", exc_info=True)
            return None
