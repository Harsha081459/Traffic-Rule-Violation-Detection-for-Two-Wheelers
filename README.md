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


# 
# Indian roads see **thousands of two-wheeler fatalities** every year due to helmetless riding and overcrowding. Manual enforcement by traffic police is limited in scale. This project automates the entire violation detection pipeline using computer vision:
# 
# 1. **Detect** all two-wheelers and their riders in a traffic image
# 2. **Classify** whether each rider is wearing a helmet or not
# 3. **Count** riders per vehicle to flag overcrowding (>2 riders)
# 4. **Extract** the license plate number of violating vehicles using OCR
# 
# 🔗 **[Try the Live Demo on HuggingFace Spaces →](https://huggingface.co/spaces/hv-123/2-Wheeler_Traffic-violation-ai)**
# 
# ---
# 
# ## 🌐 Live Web Application
# 
# We deployed a premium, dark-mode web application directly to HuggingFace Spaces. You can interact with the models in real-time by dragging and dropping images into your browser.
# 
# **[Click here to access the Live Web App](https://huggingface.co/spaces/hv-123/2-Wheeler_Traffic-violation-ai)**
# 
# ---
# 
# ## 🏗️ Pipeline Architecture
# 
# ```
#                               ┌──────────────────────────────────────┐
#                               │         INPUT IMAGE                  │
#                               └──────────────┬───────────────────────┘
#                                              │
#                                              ▼
#                               ┌──────────────────────────────────────┐
#                               │     Stage 1: Full Detector           │
