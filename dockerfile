# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set environment variable

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 && \
    pip install --upgrade pip && \
    pip install opencv-python-headless

# Specify the default command to run main.py when the container starts
CMD ["python", "main.py"]
