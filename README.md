# Traffic Sentinel AI

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

### Option 1 — Local (Python)

```bash
git clone https://github.com/YOUR_USERNAME/traffic-sentinel-ai
cd traffic-sentinel-ai
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
# place model weights in ./models/ (see Training section)
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**

### Option 2 — Docker (single command)

```bash
docker compose up
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

> mAP50 > 0.75 is considered production-ready for variable real-world environments.

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

python hf_upload_models.py --username YOUR_HF_USERNAME
# Creates:  https://huggingface.co/YOUR_USERNAME/traffic-sentinel-models
```

### Step 2 — Create a new HF Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Name it `traffic-sentinel-ai`
3. SDK: **Docker**
4. Visibility: **Public**
5. Click **Create Space**

### Step 3 — Push your code

```bash
# Add the Space as a remote (replace YOUR_USERNAME)
git remote add spaces https://huggingface.co/spaces/YOUR_USERNAME/traffic-sentinel-ai

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
| `HF_MODEL_REPO` | `YOUR_USERNAME/traffic-sentinel-models` |

The container will download models on first boot (~2 min), then serve on port 7860.

Your live URL: **`https://huggingface.co/spaces/YOUR_USERNAME/traffic-sentinel-ai`**

---

## CI/CD Pipeline

`.github/workflows/main.yml` runs on every push to `main`:

| Job | Steps |
|-----|-------|
| **lint** | `ruff` + `black --check` |
| **test** | `pytest -m unit` |
| **docker** | Build & push to GitHub Container Registry |
| **compose-test** | `docker compose up` + health-check `/api/health` |
| **security** | `bandit` + `safety` |

---

## Testing

```bash
pip install pytest httpx
pytest -m unit          # fast unit tests
pytest                  # all tests
```

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
├── requirements.txt          # Full dependencies (dev + training)
├── requirements_hf.txt       # Inference-only (for HF Spaces)
├── README.md
├── RECRUITER.md
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

- **Harsha Vardhan** (itsmeharsha081459@gmail.com)
- **Vishal Sriram** (vishalsriram.ks@gmail.com)
- **Anish Reddy R** (Anish.R@iiitb.ac.in)
