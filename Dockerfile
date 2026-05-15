FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV and EasyOCR
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (HuggingFace requirement for security)
RUN useradd -m -u 1000 user
USER user


# FROM python:3.10-slim
# 
# # Set working directory
# WORKDIR /app
# 
# # Install system dependencies required by OpenCV and EasyOCR
# RUN apt-get update && apt-get install -y \
#     libgl1 \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*
# 
# # Create a non-root user (HuggingFace requirement for security)
# RUN useradd -m -u 1000 user
# USER user
