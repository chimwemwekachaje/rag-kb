# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Install system dependencies (minimal set)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies using pre-built wheels
RUN pip install --no-cache-dir --only-binary=all -r requirements.txt

# Copy application files
COPY  --chown=user app.py .
COPY  --chown=user models/ ./models/
COPY  --chown=user data/ ./data/

# Create directory for ChromaDB
RUN mkdir -p /app/chroma

EXPOSE 7860
VOLUME ["/app/chroma"]
CMD ["python", "app.py"]