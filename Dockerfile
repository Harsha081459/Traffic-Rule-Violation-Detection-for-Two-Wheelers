# Multi-stage Dockerfile for Traffic Sentinel AI
# Optimized for production: minimal image size, security hardening

# ──────────────────────────────────────────────────────────────────────────
# Stage 1: Builder
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies (only here, not in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install wheels into /build/wheels
COPY requirements-inference.txt requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# ──────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# Set non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy wheels from builder
COPY --from=builder /build/wheels /tmp/wheels

# Install wheels (no build tools needed)
RUN pip install --upgrade pip && \
    pip install --no-cache /tmp/wheels/* && \
    rm -rf /tmp/wheels && \
    pip cache purge

# Copy application code
COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser traffic_violation/ ./traffic_violation/
COPY --chown=appuser:appuser static/ ./static/
COPY --chown=appuser:appuser download_models.py .
RUN mkdir -p /app/models && chown appuser:appuser /app/models
ENV HF_MODEL_REPO=hv-123/traffic-sentinel-models TV_USE_ONNX=1

# Compile Python to bytecode for faster startup
RUN python -m compileall /app

# Create a non-root user and switch to it
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/ready', timeout=5)" || exit 1

EXPOSE 8000

# Run with gunicorn in production (or uvicorn in dev)
CMD ["sh", "-c", "python download_models.py && exec uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1"]
