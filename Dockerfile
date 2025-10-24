FROM abetlen/llama-cpp-python:latest

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

WORKDIR /app

# Copy requirements and install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir langchain-chroma langchain-text-splitters \
    langchain-core langchain-community gradio gradio-pdf pypdf>=6.1.3 \
    pytest>=8.0.0 pytest-cov>=4.1.0 pytest-mock>=3.12.0 coverage[toml]>=7.4.0

# Copy application files
COPY app.py .
COPY models/ ./models/
COPY data/ ./data/

# Create directory for ChromaDB
RUN mkdir -p /app/chroma

EXPOSE 7860
VOLUME ["/app/chroma"]
CMD ["python", "app.py"]