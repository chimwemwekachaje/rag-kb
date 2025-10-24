# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Install system dependencies for building llama-cpp-python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimized build flags for llama-cpp-python
RUN CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY models/ ./models/
COPY data/ ./data/

# Create directory for ChromaDB (will be mounted as volume)
RUN mkdir -p /app/chroma

# Expose the port that Gradio will run on
EXPOSE 7860

# Define volume for ChromaDB persistence
VOLUME ["/app/chroma"]

# Set the default command to run the application
CMD ["python", "app.py"]
