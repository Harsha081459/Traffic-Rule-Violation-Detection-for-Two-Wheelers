"""
app.py — Traffic Sentinel AI — Production FastAPI Server
=========================================================
Features
--------
* ONNX-accelerated inference (TV_USE_ONNX=1 set before any import).
* Per-IP rate limiting via a thread-safe in-memory token bucket.
* Strict input validation: MIME type, file extension, magic bytes, file size.
* Async file upload with executor-based inference (non-blocking event loop).
* Structured JSON errors with consistent {"detail": "..."} shape.
* /api/health and /api/info endpoints for monitoring.
* CORS configured for local + LAN development.
"""

from __future__ import annotations

import asyncio
import imghdr
import logging
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TV_USE_ONNX", "1")  # env var can override; set before import

import cv2                        # noqa: E402
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from traffic_violation import TrafficViolationDetector, __version__

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB   = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
RATE_LIMIT_RPM     = 20          # requests per minute per IP
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/webp",
    "image/bmp", "image/x-bmp",
}
# imghdr → MIME fallback map
MAGIC_MIME: dict[str, str] = {
    "jpeg": "image/jpeg",
    "png":  "image/png",
    "bmp":  "image/bmp",
    "webp": "image/webp",
    "gif":  "image/gif",
}


# ---------------------------------------------------------------------------
# In-memory token-bucket rate limiter
# ---------------------------------------------------------------------------
class _RateLimiter:
    """Thread-safe token-bucket rate limiter keyed by IP address."""

    def __init__(self, rpm: int) -> None:
        self._rate       = rpm / 60.0       # tokens per second
        self._capacity   = float(rpm)
        self._buckets:   dict[str, float] = defaultdict(lambda: float(rpm))
        self._last_refill: dict[str, float] = defaultdict(time.monotonic)

    def is_allowed(self, ip: str) -> bool:
        now     = time.monotonic()
        elapsed = now - self._last_refill[ip]
        self._buckets[ip] = min(
            self._capacity,
            self._buckets[ip] + elapsed * self._rate,
        )
        self._last_refill[ip] = now
        if self._buckets[ip] >= 1.0:
            self._buckets[ip] -= 1.0
            return True
        return False


_limiter = _RateLimiter(RATE_LIMIT_RPM)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Traffic Sentinel AI",
    description="Real-time two-wheeler violation detection powered by YOLO11 + ONNX.",
    version=__version__,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Detector (loaded once at startup, shared across all requests)
# ---------------------------------------------------------------------------
_detector: TrafficViolationDetector | None = None


@app.on_event("startup")
async def _startup() -> None:
    global _detector
    logger.info("Loading TrafficViolationDetector…")
    t0 = time.perf_counter()
    _detector = TrafficViolationDetector(model_dir="./models")
    logger.info("Detector ready in %.2f s", time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    return forwarded.split(",")[0].strip() if forwarded else (request.client.host or "unknown")


def _validate_image(content: bytes, filename: str) -> None:
    """Raise HTTPException if *content* is not an acceptable image.

    Checks (in order):
    1. File size.
    2. File extension from original filename.
    3. Content-Type MIME header (advisory).
    4. Magic bytes via :mod:`imghdr` (authoritative).
    """
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
        )
    if len(content) < 8:
        raise HTTPException(status_code=400, detail="File is empty or too small.")

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Extension '{suffix}' not allowed. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    detected = imghdr.what(None, h=content)
    if detected not in MAGIC_MIME:
        raise HTTPException(
            status_code=415,
            detail="File content does not look like a supported image (JPEG/PNG/WebP/BMP).",
        )


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Liveness probe — returns 200 when the server is up."""
    return {
        "status":  "ok",
        "version": __version__,
        "onnx":    os.environ.get("TV_USE_ONNX") == "1",
        "ts":      time.time(),
    }


@app.get("/api/info")
async def info() -> dict[str, Any]:
    """Return runtime information about the loaded detector."""
    if _detector is None:
        raise HTTPException(status_code=503, detail="Detector not ready yet.")
    cfg = _detector._cfg
    return {
        "model_dir":     str(cfg.model_dir),
        "fast_mode":     cfg.fast_mode,
        "use_onnx":      cfg.use_onnx,
        "device":        cfg.device,
        "rate_limit":    f"{RATE_LIMIT_RPM} req/min per IP",
        "max_file_mb":   MAX_FILE_SIZE_MB,
    }


@app.post("/predict")
async def predict_image(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """Analyse an uploaded image and return violation detections.

    Returns the full debug payload (bounding boxes, rider counts, plates,
    inference time) used by the frontend to render annotated results.

    Rate limit: ``RATE_LIMIT_RPM`` requests per minute per client IP.
    """
    # ── Rate limiting ───────────────────────────────────────────────
    ip = _client_ip(request)
    if not _limiter.is_allowed(ip):
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {RATE_LIMIT_RPM} requests/minute.",
        )

    # ── Detector readiness ──────────────────────────────────────────
    if _detector is None:
        raise HTTPException(status_code=503, detail="Detector is still loading. Try again in a moment.")

    # ── Read & validate ─────────────────────────────────────────────
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {exc}") from exc

    fname = file.filename or "upload.jpg"
    _validate_image(content, fname)

    # ── Write temp file ─────────────────────────────────────────────
    suffix = Path(fname).suffix.lower() or ".jpg"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # ── Run inference in a thread (keeps event loop free) ───────
        loop   = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, _detector.predict_debug, tmp_path)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference failed for %s", fname)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    logger.info(
        "predict  ip=%s  file=%s  size=%.1fKB  violations=%d  time=%.3fs",
        ip, fname, len(content) / 1024,
        len(result.get("violations", [])),
        result.get("inference_time_sec", 0),
    )
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Static files (frontend) — must be mounted LAST
# ---------------------------------------------------------------------------
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # PORT env var is injected by HF Spaces; falls back to 8000 locally.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, workers=1)
