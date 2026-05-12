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



# import cv2
# import numpy as np
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel
# 
# # Import the existing pipeline
# from solution_final_strong import TrafficViolationDetector
# 
# app = FastAPI(title="Traffic Violation Detection API")
# 
# # Setup CORS just in case
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
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
