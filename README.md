# Traffic Sentinel AI

![CI](https://github.com/Harsha081459/Traffic-Rule-Violation-Detection-for-Two-Wheelers/actions/workflows/ci.yml/badge.svg)

> Real-time two-wheeler violation detection powered by **YOLOv11 + ONNX Runtime**.  
> Detects riders without helmets, over-loaded bikes (>2 riders), and reads license plates via EasyOCR.
**🚀 Live Public Application:** [https://hv-123-traffic-sentinel-ai.hf.space/](https://hv-123-traffic-sentinel-ai.hf.space/)

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HF%20Spaces-yellow?logo=huggingface)](https://hv-123-traffic-sentinel-ai.hf.space/)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.16-orange)](https://onnxruntime.ai)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](Dockerfile.hf)

---

## Quick Start

### Option 1 — Local detection demo (Python 3.12)

```bash
git clone https://github.com/Harsha081459/Traffic-Rule-Violation-Detection-for-Two-Wheelers
cd Traffic-Rule-Violation-Detection-for-Two-Wheelers

python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS / Linux

pip install -r requirements-inference.txt

# Model weights are not committed — download them from HF Hub:
# Windows:  set HF_MODEL_REPO=hv-123/traffic-sentinel-models
# macOS/Linux:  export HF_MODEL_REPO=hv-123/traffic-sentinel-models
python download_models.py   # populates ./models/

uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**

> The app needs the ONNX weights in `./models/` to serve `/predict`. Without them the
> server still starts and `/api/health` works, but inference returns 503. The unit
> tests (`pytest -m unit`) run without any model files.

Check `/api/ready`: HTTP 200 means detection models loaded, 503 means setup failed. `/api/health` is liveness only. `/api/info` and every prediction expose `ocr_available`. The published Hub repository contains four ONNX detectors **but no EasyOCR weights**. This minimal demo therefore detects bikes/helmets/plates but returns blank plate text (`ocr_available=false`). To enable text recognition, install EasyOCR and its PyTorch dependencies, provision its English recognition and detection weights under `models/easyocr/`, then restart and confirm `ocr_available=true`. OCR accuracy has not been benchmarked; blank text must not be interpreted as a successful read.

`requirements.txt` remains the broader training dependency set; it is not required for this detection-only demo. In PowerShell set `$env:HF_MODEL_REPO='hv-123/traffic-sentinel-models'` before running the downloader. Downloads require internet access; inference uses the downloaded files.

### Option 2 — Docker (single command)

```bash
docker compose up --build -d --wait --wait-timeout 300
```

Open **http://localhost:8000**

### Option 3 — Hugging Face Spaces (public link)

See the [Deployment to HF Spaces](#deployment-to-hf-spaces) section below.

---

## Architecture

```
traffic_violation/
├── pipeline.py          # End-to-end orchestration (TrafficViolationDetector)
├── config.py            # Frozen PipelineConfig dataclass (fast / accurate presets)
├── ocr_engine.py        # EasyOCR wrapper with multi-variant preprocessing
├── models/
│   ├── base.py          # DetectorProtocol (structural typing)
│   ├── yolo_detector.py # PyTorch backend (YOLODetector)
│   └── onnx_detector.py # ONNX Runtime backend (ONNXDetector) ← 1.5x faster
├── utils/
│   └── geometry.py      # Det, clip_box, inter_area, nms_same_class
└── accelerate/
    └── export.py        # export_to_onnx / export_all helpers
```

**Inference pipeline (3 concurrent stages per bike):**

```
Image ──► full_detector (bikes + riders)
              │
              ├──► helmet_detector  ─┐
              ├──► plate_detector   ─┼──► EasyOCR ──► plate text
              └──► COCO fallback    ─┘
                       │
                       └──► ViolationRecord { num_riders, helmet_violations, plate }
```

---

## Models

Three specialized YOLOv11 models trained sequentially on an NVIDIA RTX 4060 Ti:

| Model | Architecture | Training Images | mAP50 | ONNX Size |
|-------|-------------|-----------------|-------|-----------|
| full\_detector | YOLOv11m | ~5 000 (COCO + VisDrone) | **0.763** | 80.3 MB |
| helmet\_detector | YOLOv11s | 3 100 (Roboflow) | **0.838** | 37.8 MB |
| plate\_detector | YOLOv11n | 9 570 (Roboflow) | **0.935** | 10.5 MB |

> These are historical per-detector validation results, not an end-to-end safety or production-readiness guarantee. The complete violation/OCR pipeline has no independently reproduced accuracy benchmark.

---

## Backend API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | `POST` | Upload image → JSON with bounding boxes, rider counts, plate text |
| `/api/health` | `GET` | Liveness probe |
| `/api/info` | `GET` | Runtime config (ONNX mode, model dir, rate limit) |
| `/api/docs` | `GET` | Swagger UI |

**Input validation:** file size ≤ 15 MB, extensions `.jpg .png .webp .bmp`, magic-bytes check.  
**Rate limiting:** 20 requests/minute per IP (token-bucket).

---

## Frontend

- Dark-mode glassmorphism UI with animated bounding-box drawing
- Step-by-step processing tracker (Upload → Detect → Helmets → Plates)
- Inference-time badge (ONNX backend typically **< 2 s** on CPU)
- Per-bike violation cards with rider count, helmet status, plate text
- Toast notifications for errors (rate limit, invalid file, server error)

---

## Performance

| Backend | Inference time (640×480 image, CPU) |
|---------|-------------------------------------|
| PyTorch | ~4.5 s |
| ONNX Runtime | ~2.5 s **(1.8× faster)** |

---

## Deployment to HF Spaces

> One-time setup (~10 minutes). After this your app is live at a public URL forever.

### Step 1 — Upload model weights to HF Hub

```bash
pip install huggingface_hub
huggingface-cli login          # opens browser auth

python hf_upload_models.py --username hv-123
# Creates:  https://huggingface.co/hv-123/traffic-sentinel-models
```

### Step 2 — Create a new HF Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name it `traffic-sentinel-ai`
3. SDK: **Docker**
4. Visibility: **Public**
5. Click **Create Space**

### Step 3 — Push your code

```bash
# Add the Space as a remote
git remote add spaces https://huggingface.co/spaces/hv-123/traffic-sentinel-ai

# HF Spaces uses the file named "Dockerfile" — copy our HF version
cp Dockerfile.hf Dockerfile.spaces_deploy
# Push (HF will build the Docker image automatically)
git push spaces main
```

> Or use the HF web UI to drag-and-drop the files into the Space.

### Step 4 — Add the secret

In your Space → **Settings → Repository secrets**:

| Name | Value |
|------|-------|
| `HF_MODEL_REPO` | `hv-123/traffic-sentinel-models` |

The container will download models on first boot (~2 min), then serve on port 7860.

> `requirements_hf.txt` pins target the Space's `python:3.10-slim` image; they were
> not independently revalidated against the deployed Space. For the verified local
> detection-only path on Python 3.12 use `requirements-inference.txt`.

Live URL: **[https://hv-123-traffic-sentinel-ai.hf.space/](https://hv-123-traffic-sentinel-ai.hf.space/)**

---

## CI/CD Pipeline

`.github/workflows/ci.yml` runs on every push to `main` and on pull requests:

| Job | Steps |
|-----|-------|
| **test** | `ruff check` (E9,F63,F7,F82,F401) + `pytest -m unit` on Python 3.12 |
| **security** | `bandit -r app.py traffic_violation/ -ll` |
| **model-smoke** | Downloads published ONNX weights, uploads a committed sample through FastAPI, builds Docker Compose and checks readiness/inference |

---

## Testing

```bash
pip install -r requirements-dev.txt fastapi python-multipart numpy opencv-python-headless
pytest -m unit          # fast unit tests (no model files needed)
pytest                  # unit tests; downloaded-model smoke is opt-in
# After downloading weights, Linux/macOS:
RUN_MODEL_SMOKE=1 python -m pytest tests/test_model_smoke.py -q
# PowerShell: $env:RUN_MODEL_SMOKE='1'; python -m pytest tests/test_model_smoke.py -q
```

---

## Limitations

- **No end-to-end violation-level accuracy.** The reported numbers are per-detector
  mAP50 scores on each model's own validation split (see `TRAINING_SUMMARY.md` §3);
  the full pipeline (rider counting → helmet logic → plate read) was never scored
  end-to-end. The only end-to-end check documented is a single-image sanity test
  after the ONNX postprocessing fix (`TRAINING_SUMMARY.md` §6).
- **OCR quality is unquantified.** The plate model's 0.935 mAP50 measures plate
  *localization*; character-level read accuracy of the EasyOCR stage was not
  separately measured.
- **Latency figures are single-image CPU benchmarks.** The ~1.8× ONNX speedup was
  measured on 640×480 images on CPU (`TRAINING_SUMMARY.md` §4); real-world latency
  varies with image size and hardware.
- **Model weights are not in this repo.** A clean clone can run the tests, but
  serving inference requires downloading the `.onnx` files from
  [hv-123/traffic-sentinel-models](https://huggingface.co/hv-123/traffic-sentinel-models)
  (verified reachable) or retraining via `train.py`.

---

## Project Structure

```
.
├── app.py                    # FastAPI server (CORS, rate limiting, validation)
├── traffic_violation/        # Core detection package
│   ├── pipeline.py
│   ├── config.py
│   ├── ocr_engine.py
│   ├── models/
│   └── utils/
├── static/                   # Frontend (HTML + CSS + JS)
├── models/                   # Model weights (git-ignored)
│   ├── full_detector.onnx
│   ├── helmet_detector.onnx
│   ├── plate_detector.onnx
│   └── yolo11n.onnx
├── tests/                    # pytest suite
├── Dockerfile                # Local multi-stage Docker build (port 8000)
├── Dockerfile.hf             # HF Spaces build (port 7860, downloads models)
├── docker-compose.yml        # Single-command local deployment
├── download_models.py        # Downloads models from HF Hub at startup
├── hf_upload_models.py       # One-time: upload .onnx files to HF Hub
├── run_export.py             # Export .pt → .onnx
├── run_test.py               # PyTorch vs ONNX speed benchmark
├── train.py                  # YOLOv11 training script
├── dataset_builder.py        # Data engineering pipeline
├── requirements.txt          # Full dependencies (training + API, Python 3.12)
├── requirements_hf.txt       # Inference-only (for HF Spaces, Python 3.10)
├── requirements-dev.txt      # Test/lint tooling
├── LICENSE                   # MIT
├── README.md
└── TRAINING_SUMMARY.md
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `No two-wheelers detected` | Model is running but ONNX postprocessing was misconfigured — fixed in v2.0.1 |
| Server won't start | Check `models/` has the `.onnx` files; run `python download_models.py --check` |
| Rate limit 429 | Wait 60 s or increase `RATE_LIMIT_RPM` in `app.py` |
| Docker image too large | Use `Dockerfile.hf` (inference-only deps, ~3 GB) instead of `Dockerfile` |

---

## License

MIT — see [LICENSE](LICENSE)

## Team

- **Harsha Vardhan** (Harsha.Vardhan@iiitb.ac.in)
- **Vishal Sriram** (Vishal.Sriram@iiitb.ac.in)
- **Anish Reddy R** (Anish.R@iiitb.ac.in)
