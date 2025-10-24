<!-- b90676e4-c3e4-4d24-967f-29080a6d8abf 8184acd2-2b85-473d-a667-509755a9e39b -->
# Dockerfile for rag-kb

## Overview

Create a production-ready Dockerfile that packages the entire RAG knowledge base system with all dependencies, GGUF models, and PDF documents included in the image.

## Implementation Steps

### 1. Create Dockerfile

Create `/Users/kachaje/andela/genAi-bootcamp/projects/code/rag-kb/Dockerfile` with:

- **Base image**: Use `python:3.12-slim` for Python 3.12 support
- **System dependencies**: Install build tools for llama-cpp-python compilation:
- `build-essential`, `cmake`, `git` for building native extensions
- `libopenblas-dev` for CPU optimizations
- **Working directory**: Set to `/app`
- **Copy requirements**: Copy `requirements.txt` first for Docker layer caching
- **Install Python packages**: 
- Set `CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"` for optimized llama-cpp-python
- Install with `pip install --no-cache-dir`
- **Copy application files**: Copy `app.py`, `models/`, `data/`, and other necessary files
- **Create volume mount point**: Define volume for `/app/chroma` for database persistence
- **Expose port**: Expose port 7860 for Gradio interface
- **Set environment variables**: Configure Gradio to bind to 0.0.0.0
- **Entry point**: Run `python app.py`

### 2. Create .dockerignore

Create `/Users/kachaje/andela/genAi-bootcamp/projects/code/rag-kb/.dockerignore` to exclude:

- `__pycache__/`, `*.pyc`, `.pytest_cache/`
- `chroma/` (will be in volume)
- `.git/`, `.gitignore`
- `venv/`, `.venv/`, `uv.lock`
- `htmlcov/`, `test_chroma/`, `tests/`, `TESTING_SUMMARY.md`
- `*.sh` (control scripts not needed in container)

### 3. Create docker-compose.yml (Optional)

Create `/Users/kachaje/andela/genAi-bootcamp/projects/code/rag-kb/docker-compose.yml` for easier management:

- Service definition for rag-kb
- Named volume for ChromaDB persistence
- Port mapping 7860:7860
- Environment variables for model paths
- Auto-restart policy

### 4. Update README.md

Add Docker deployment section with:

- Build instructions: `docker build -t rag-kb .`
- Run instructions: `docker run -p 7860:7860 -v rag-kb-data:/app/chroma rag-kb`
- Docker Compose instructions: `docker-compose up -d`
- Expected image size note (~2-3GB due to models)
- Volume management for ChromaDB persistence

## Key Considerations

- **Image size**: Will be large (2-3GB) due to GGUF models being included
- **Build time**: Initial build will be slow due to compiling llama-cpp-python from source
- **CPU optimization**: Using OpenBLAS for better CPU inference performance
- **Data persistence**: ChromaDB vector store persists across container restarts via Docker volume
- **Network access**: Container binds to 0.0.0.0 to allow external access
- **Self-contained**: No external downloads required at runtime - truly standalone deployment

### To-dos

- [ ] Create Dockerfile with Python 3.12, system dependencies, and optimized llama-cpp-python build
- [ ] Create .dockerignore to exclude unnecessary files from image
- [ ] Create docker-compose.yml for simplified container orchestration
- [ ] Add Docker deployment instructions to README.md