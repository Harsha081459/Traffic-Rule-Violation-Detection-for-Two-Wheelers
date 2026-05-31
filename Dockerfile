# Multi-stage Dockerfile for Traffic Sentinel AI
# Optimized for production: minimal image size, security hardening

# ──────────────────────────────────────────────────────────────────────────
# Stage 1: Builder
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim as builder

WORKDIR /build

# Install build dependencies (only here, not in final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install wheels into /build/wheels
COPY <<EOF requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
ultralytics==8.0.208
opencv-python==4.8.1.78
easyocr==1.7.0
onnxruntime==1.16.3
onnx==1.15.0
onnxsim==0.4.33
albumentations==1.3.1
imagehash==4.3.1
pillow==10.1.0
pyyaml==6.0.1
numpy==1.24.3
opencv-contrib-python==4.8.1.78
torch==2.1.1+cpu
torchvision==0.16.1+cpu
torchaudio==2.1.1+cpu
fiftyone==0.20.1
tqdm==4.66.1
roboflow==0.2.28
requests==2.31.0
paramiko==3.3.1
scp==0.14.5
ray[tune]==2.8.1
tzdata==2023.3
EOF

RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -r requirements.txt

# ──────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime
# ──────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

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
COPY --chown=appuser:appuser models/ ./models/

# Compile Python to bytecode for faster startup
RUN python -m compileall -b /app && \
    find /app -name "*.py" -delete && \
    find /app -type d -name "__pycache__" -exec chmod 755 {} \;

# Create a non-root user and switch to it
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health')" || exit 1

EXPOSE 8000

# Run with gunicorn in production (or uvicorn in dev)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
