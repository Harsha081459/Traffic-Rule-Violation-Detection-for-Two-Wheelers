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


# import os
# import tempfile
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
