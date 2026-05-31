# Traffic Sentinel AI — Recruiter Quick-Start

> This document gets you from zero to a working live demo in under 5 minutes.

---

## 30-Second Docker Demo

```bash
docker compose up        # builds image, starts server on port 8000
# open http://localhost:8000
# upload any motorcycle image — violations appear instantly
```

---

## Live Public Demo

| | |
|-|-|
| **Live URL** | https://huggingface.co/spaces/YOUR_USERNAME/traffic-sentinel-ai |
| **API Docs** | https://huggingface.co/spaces/YOUR_USERNAME/traffic-sentinel-ai/api/docs |

_(Replace `YOUR_USERNAME` with the owner's HF username)_

---

## What This Project Demonstrates

### 1. Computer Vision Engineering
- **3 custom YOLOv11 models** trained from scratch on a remote GPU (RTX 4060 Ti)
- Cascade pipeline: full-frame detection → per-bike helmet check → plate OCR
- mAP50 scores: Full=0.763, Helmet=0.838, Plate=**0.935**

### 2. Performance Optimization
- **ONNX Runtime** backend replaces PyTorch for CPU inference: ~1.8× faster
- Concurrent per-bike processing via `ThreadPoolExecutor`
- Cold-start warmup eliminates first-request latency spike

### 3. Production-Quality API
- **FastAPI** with async request handling (`run_in_executor` keeps event loop free)
- Per-IP token-bucket rate limiting (20 req/min)
- Strict input validation: file size, extension, MIME header **and** magic bytes
- Structured `{"detail": "..."}` error responses across all failure modes

### 4. Modern Frontend
- Dark-mode glassmorphism SaaS UI
- Animated bounding-box drawing (canvas, `setLineDash`)
- Step-by-step processing tracker + inference-time badge
- Toast notifications for every error type

### 5. DevOps & MLOps
- **Multi-stage Dockerfile** (build tools stripped from runtime image)
- **docker-compose** for one-command local deployment
- **GitHub Actions** CI/CD: lint → test → Docker build → security scan
- **HF Spaces** public deployment via `Dockerfile.hf` + model hub download
- pytest suite covering health, validation, rate-limiting, error formats

---

## Files Worth Reading (in order)

| File | Why it's interesting |
|------|---------------------|
| `traffic_violation/pipeline.py` | Full cascade pipeline — OOP, type hints, concurrent execution |
| `traffic_violation/models/onnx_detector.py` | Hand-rolled ONNX pre/post-processing with letterbox undo |
| `traffic_violation/config.py` | Frozen `PipelineConfig` dataclass — clean config management |
| `app.py` | Production FastAPI: rate limiting, magic-bytes validation, async inference |
| `static/main.js` | Canvas animation, AbortController, step tracker, toast system |
| `.github/workflows/main.yml` | Full CI/CD pipeline |
| `dataset_builder.py` | Data engineering: COCO + VisDrone + Roboflow + OpenImages |
| `train.py` | YOLOv11 training with AMP, early stopping, hyperparameter tuning |

---

## API Endpoints

```bash
# Health check
curl http://localhost:8000/api/health

# Run detection
curl -X POST http://localhost:8000/predict \
     -F "file=@motorcycle.jpg" | python -m json.tool
```

---

## Architecture at a Glance

```
Browser  ──POST /predict──►  FastAPI (app.py)
                                │
                                ▼
                     TrafficViolationDetector
                        ├── full_detector.onnx   (bikes + riders)
                        ├── helmet_detector.onnx (per-bike crop)
                        ├── plate_detector.onnx  (per-bike crop)
                        └── EasyOCR              (plate text)
                                │
                                ▼
                     JSON { violations, debug, inference_time_sec }
                                │
                                ▼
                     Canvas animation + result cards
```

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Inference time (ONNX, CPU) | ~2.5 s per image |
| Inference time (PyTorch, CPU) | ~4.5 s per image |
| Speedup from ONNX | **1.8×** |
| Plate detection mAP50 | **0.935** |
| Helmet detection mAP50 | **0.838** |
| Full model mAP50 | **0.763** |
| Docker image size | ~3 GB (inference only) |
| Test coverage | Health ✓ Validation ✓ Rate-limit ✓ |

---

## FAQs

**Q: Why three separate models instead of one?**  
Each sub-task has different scale requirements. The plate model needs high-resolution crops; the full-frame model needs broad context. Cascade allows each model to specialize.

**Q: Why ONNX over TensorRT or OpenVINO?**  
ONNX Runtime runs on any hardware without vendor lock-in — critical for portability across deployment targets (local CPU, Docker, HF Spaces).

**Q: How is rate limiting implemented?**  
In-memory token-bucket per IP address (no Redis needed for a demo). Thread-safe with Python's GIL since bucket updates are atomic float operations.

**Q: Why not use a single YOLOv11 pose model for everything?**  
Pose estimation is available as an optional feature (`TV_USE_POSE=1`) but disabled by default to keep inference under 3 s on CPU.
