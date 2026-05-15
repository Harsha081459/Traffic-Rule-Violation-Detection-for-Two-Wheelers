FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \


# FROM python:3.10-slim
# 
# # Set working directory
# WORKDIR /app
# 
# # Install system dependencies required by OpenCV and EasyOCR
# RUN apt-get update && apt-get install -y \
