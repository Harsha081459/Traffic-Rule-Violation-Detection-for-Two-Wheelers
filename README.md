# 🚦 Traffic Violation Detection – Two-Wheelers

> **YOLOv11 · EasyOCR · FastAPI · Docker · HuggingFace Spaces**

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python" />
  <img src="https://img.shields.io/badge/YOLOv11-Ultralytics-purple?logo=yolo" />
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/🤗_HuggingFace-Spaces-yellow" />
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" />
</p>

---

## 🎯 Problem Statement

Indian roads see **thousands of two-wheeler fatalities** every year due to helmetless riding and overcrowding. Manual enforcement by traffic police is limited in scale. This project automates the entire violation detection pipeline using computer vision:

1. **Detect** all two-wheelers and their riders in a traffic image
2. **Classify** whether each rider is wearing a helmet or not
3. **Count** riders per vehicle to flag overcrowding (>2 riders)
4. **Extract** the license plate number of violating vehicles using OCR

🔗 **[Try the Live Demo on HuggingFace Spaces →](https://huggingface.co/spaces/hv-123/2-Wheeler_Traffic-violation-ai)**

---

## 🌐 Live Web Application

We deployed a premium, dark-mode web application directly to HuggingFace Spaces. You can interact with the models in real-time by dragging and dropping images into your browser.

**[Click here to access the Live Web App](https://huggingface.co/spaces/hv-123/2-Wheeler_Traffic-violation-ai)**

---

## 🏗️ Pipeline Architecture

```
                              ┌──────────────────────────────────────┐
                              │         INPUT IMAGE                  │
                              └──────────────┬───────────────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────────────┐
                              │     Stage 1: Full Detector           │
                              │     YOLOv11m (imgsz=640)             │
                              │     → Detects: two_wheeler, rider    │
                              └──────────────┬───────────────────────┘
                                             │
                          ┌──────────────────┼──────────────────┐
                          ▼                                     ▼
               ┌─────────────────────┐              ┌─────────────────────┐
               │  Stage 2: Helmet    │              │  Stage 3: Plate     │
               │  YOLOv11s           │              │  YOLOv11n           │
               │  → helmet/no_helmet │              │  → license_plate    │
               └────────┬────────────┘              └────────┬────────────┘
                        │                                    │
                        ▼                                    ▼
               ┌─────────────────────┐              ┌─────────────────────┐
               │  Violation Logic    │              │  Stage 4: EasyOCR   │
               │  • >2 riders?       │              │  → plate text       │
               │  • no helmet?       │              │  → post-processing  │
               └────────┬────────────┘              └────────┬────────────┘
                        │                                    │
                        └──────────────┬─────────────────────┘
                                       ▼
                              ┌──────────────────────────────────────┐
                              │        JSON RESPONSE                 │
                              │  violations[], bboxes, plate_text    │
                              └──────────────────────────────────────┘
```

---

## 📊 Model Performance (Validation Set)

| Model | Task | mAP50 | mAP50-95 | Precision | Recall |
|:---|:---|:---:|:---:|:---:|:---:|
| YOLOv11m — Full Detector | Two-wheeler + Rider | 69.8% | 42.8% | 84.9% | 61.5% |
| YOLOv11s — Helmet Detector | Helmet / No-helmet | **99.3%** | **89.2%** | **99.1%** | 97.2% |
| YOLOv11n — Plate Detector | License Plate | **96.9%** | 67.5% | **98.2%** | 94.6% |

**Training details:**
- All models trained on **custom-annotated datasets** using **Kaggle T4 GPUs**
- Helmet detector was **finetuned** on a cleaned dataset with low LR (0.00025), boosting mAP50 from 94.9% → 99.3%
- Full detector uses a COCO-pretrained YOLOv11n as a **fallback** to recover missed detections

---

## ✨ Key Features

| Feature | Description |
|:---|:---|
| 🪖 **Helmet Detection** | Classifies each rider as helmet / no-helmet with 99.3% mAP50 |
| 🔢 **Rider Counting** | Flags overcrowded vehicles (>2 riders per motorcycle) |
| 🪪 **License Plate OCR** | Localizes plates (96.9% mAP50) and reads text using EasyOCR with multi-variant post-processing |
| 🎨 **Premium Web UI** | Dark-mode glassmorphism interface with drag-and-drop upload and real-time bounding-box canvas rendering |
| 🐳 **Dockerized Deployment** | One-click deploy to HuggingFace Spaces or any Docker-compatible cloud |
| ⚡ **Request Cancellation** | AbortController cancels in-flight API requests when a new image is uploaded |

---

## 🚀 Quick Start

### Run Locally

```bash
# Clone the repository
git clone https://github.com/Harsha081459/traffic-violation-detection.git
cd traffic-violation-detection

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows (cmd)
# source .venv/bin/activate     # Linux / Mac

# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py
```

Open **http://localhost:8000** in your browser → drag and drop any traffic image.

### Run with Docker

```bash
docker build -t traffic-violation-ai .
docker run -p 8000:7860 traffic-violation-ai
```

---

## 📁 Project Structure


# 
# ```bash
# # Clone the repository
# git clone https://github.com/Harsha081459/traffic-violation-detection.git
# cd traffic-violation-detection
# 
# # Create and activate virtual environment
# python -m venv .venv
# .venv\Scripts\activate          # Windows (cmd)
# # source .venv/bin/activate     # Linux / Mac
# 
# # Install dependencies
# pip install -r requirements.txt
# 
# # Start the server
# python app.py
# ```
# 
# Open **http://localhost:8000** in your browser → drag and drop any traffic image.
# 
# ### Run with Docker
# 
# ```bash
# docker build -t traffic-violation-ai .
# docker run -p 8000:7860 traffic-violation-ai
# ```
# 
# ---
# 
# ## 📁 Project Structure
