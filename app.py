import os
import tempfile
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import the existing pipeline
from solution_final_strong import TrafficViolationDetector

app = FastAPI(title="Traffic Violation Detection API")

# Setup CORS just in case
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the detector
# Assuming model_dir is correctly set to "./models" relative to where uvicorn is run
detector = TrafficViolationDetector(model_dir="./models")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    # Create a temporary file to save the uploaded image
    # We do this because predict_debug expects a file path
    suffix = os.path.splitext(file.filename)[1]
    if not suffix:
        suffix = ".png"
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_img:
        content = await file.read()
        temp_img.write(content)
        temp_file_path = temp_img.name

    try:
        # Run inference using the debug method to get bounding boxes
        result = detector.predict_debug(temp_file_path)
    except Exception as e:
        os.remove(temp_file_path)


#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# 
# # Initialize the detector
# # Assuming model_dir is correctly set to "./models" relative to where uvicorn is run
# detector = TrafficViolationDetector(model_dir="./models")
# 
# @app.post("/predict")
# async def predict_image(file: UploadFile = File(...)):
#     if not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="File must be an image.")
# 
#     # Create a temporary file to save the uploaded image
#     # We do this because predict_debug expects a file path
#     suffix = os.path.splitext(file.filename)[1]
#     if not suffix:
#         suffix = ".png"
#         
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_img:
#         content = await file.read()
#         temp_img.write(content)
#         temp_file_path = temp_img.name
# 
#     try:
#         # Run inference using the debug method to get bounding boxes
#         result = detector.predict_debug(temp_file_path)
#     except Exception as e:
#         os.remove(temp_file_path)
