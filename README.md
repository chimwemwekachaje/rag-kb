---
title: Llama RAG Knowledge Base
emoji: 🏆
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: 'Ed Donner''s LLM Udemy course knowledge base '
---

# Llama RAG Knowledge Base

[![CI/CD Pipeline](https://github.com/chimwemwekachaje/rag-kb/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/chimwemwekachaje/rag-kb/actions/workflows/ci-cd.yml)
[![HuggingFace Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/kachaje/llm-kb)

A Retrieval-Augmented Generation (RAG) knowledge base system using Llama.cpp and GGUF models. This application combines the power of local LLMs with document retrieval to provide accurate, context-aware responses.

## Features

- **Local LLM Support**: Uses TinyLlama and Nomic Embed Text models via Llama.cpp
- **Hierarchical PDF Navigation**: Nested accordion interface for browsing course content
- **Chunk-based Tracking**: Advanced chunk ID system for precise source tracking
- **Configurable Models**: Support for custom model paths via CLI args or environment variables
- **Gradio UI**: Modern web interface with PDF viewer integration

## Live Demo

🚀 **Try the live application**: [HuggingFace Spaces](https://huggingface.co/spaces/kachaje/llm-kb)

The application is automatically deployed to HuggingFace Spaces and uses sentence-transformers for embeddings, providing faster startup and better performance in the cloud environment.

## Installation

### Docker Deployment (Recommended)

The easiest way to run this application is using Docker, which includes all dependencies and models:

#### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd rag-kb

# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

The application will be available at `http://localhost:7860`

#### Using Docker directly

```bash
# Build the image
docker build -t rag-kb .

# Run the container with persistent data
docker run -d \
  --name rag-kb \
  -p 7860:7860 \
  -v rag-kb-data:/app/chroma \
  rag-kb

# View logs
docker logs -f rag-kb

# Stop and remove container
docker stop rag-kb && docker rm rag-kb
```

**Note**: The Docker image is approximately 2-3GB due to the included GGUF models.

### Manual Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required GGUF models to the `models/` directory:
   - [nomic-embed-text-v1.5.Q4_K_M.gguf](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) (embedding model)
   - [tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) (LLM model)

## Usage

### Basic Usage

```bash
python app.py
```

### Command Line Options

```bash
python app.py --help
```

- `--embedding-model PATH`: Path to embedding GGUF model
- `--llm-model PATH`: Path to LLM GGUF model  
- `--reset`: Clear the vector database

### Environment Variables

You can also set model paths using environment variables:

```bash
export EMBEDDING_MODEL_PATH="path/to/embedding/model.gguf"
export LLM_MODEL_PATH="path/to/llm/model.gguf"
python app.py
```

For HuggingFace Spaces deployment, set the `HF_TOKEN` environment variable to automatically use sentence-transformers embeddings instead of GGUF models. See `.env.example` for all available environment variables.

### Priority Order

1. Command line arguments (highest priority)
2. Environment variables
3. Default paths (lowest priority)

## Directory Structure

```
rag-kb/
├── app.py                 # Main application
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
├── .dockerignore         # Docker ignore file
├── models/               # GGUF model files
│   ├── nomic-embed-text-v1.5.Q4_K_M.gguf
│   └── tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
├── data/                 # PDF documents
│   ├── Course Summary.pdf
│   ├── Week 1/
│   │   ├── Week 1 Day 1.pdf
│   │   └── ...
│   └── ...
├── chroma/              # Vector database (auto-created)
├── requirements.txt
├── pyproject.toml
├── ctl.sh               # Control script
└── README.md
```

## Adding Documents

1. Place PDF files in the `data/` directory
2. Organize them in subdirectories as needed (e.g., by week, topic, etc.)
3. The application will automatically scan and index all PDFs on startup
4. Use the hierarchical navigation to browse and select documents

## Model Configuration

### Default Models

- **Embedding**: `nomic-embed-text-v1.5.Q4_K_M.gguf` (768 dimensions)
- **LLM**: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (1.1B parameters)

### Custom Models

You can use any GGUF-compatible models by specifying their paths:

```bash
python app.py --embedding-model "path/to/your/embedding.gguf" --llm-model "path/to/your/llm.gguf"
```

## Chunk Configuration

- **Chunk Size**: 800 characters
- **Chunk Overlap**: 80 characters
- **Chunk ID Format**: `source:page:chunk_index`

## Control Script

Use the included `ctl.sh` script for common operations:

```bash
./ctl.sh start    # Start the application
./ctl.sh reset    # Reset the database
./ctl.sh help     # Show help
```

## Docker Volume Management

### ChromaDB Persistence

The ChromaDB vector database is stored in a Docker volume for persistence across container restarts:

```bash
# List volumes
docker volume ls

# Inspect the rag-kb-data volume
docker volume inspect rag-kb-data

# Remove the volume (this will delete all indexed data)
docker volume rm rag-kb-data
```

### Reset Database in Docker

To reset the vector database in a Docker container:

```bash
# Using docker-compose
docker-compose exec rag-kb python app.py --reset

# Using docker directly
docker exec rag-kb python app.py --reset
```

## Troubleshooting

### Model Not Found

Ensure your GGUF models are in the correct location and have the right filenames. Check the console output for the exact paths being used.

### Database Issues

If you encounter database corruption or want to start fresh:

```bash
# Local installation
python app.py --reset

# Docker installation
docker-compose exec rag-kb python app.py --reset
```

### Memory Issues

For large documents or limited memory, consider:
- Using smaller chunk sizes
- Reducing the number of retrieved documents (k parameter)
- Using quantized models (Q4_K_M, Q5_K_M, etc.)

### Docker Build Issues

If you encounter issues building the Docker image:

```bash
# Clean build (no cache)
docker build --no-cache -t rag-kb .

# Check build logs
docker build -t rag-kb . 2>&1 | tee build.log
```

### Container Health Check

The Docker Compose configuration includes a health check. Monitor container health:

```bash
# Check container status
docker-compose ps

# View health check logs
docker inspect rag-kb | grep -A 10 Health
```

## License

This project is open source. Please check the individual model licenses for commercial use restrictions.
