# Use Python 3.9 slim as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    python3-pyaudio \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY voice/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the shared module
COPY shared /app/shared

# Copy the application code
COPY voice/app /app/app
COPY voice/scripts /app/scripts

# Create models directory
RUN mkdir -p /app/models

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "-m", "app.main"]
