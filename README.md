---
title: Llama RAG Knowledge Base
emoji: 🏆
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: 'Ed Donner''s LLM Udemy course knowledge base '
---

# Llama RAG Knowledge Base

A Retrieval-Augmented Generation (RAG) knowledge base system using Llama.cpp and GGUF models. This application combines the power of local LLMs with document retrieval to provide accurate, context-aware responses.

## Features

- **Local LLM Support**: Uses TinyLlama and Nomic Embed Text models via Llama.cpp
- **Hierarchical PDF Navigation**: Nested accordion interface for browsing course content
- **Chunk-based Tracking**: Advanced chunk ID system for precise source tracking
- **Configurable Models**: Support for custom model paths via CLI args or environment variables
- **Gradio UI**: Modern web interface with PDF viewer integration

## Installation

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

### Priority Order

1. Command line arguments (highest priority)
2. Environment variables
3. Default paths (lowest priority)

## Directory Structure

```
llama-rag-kb/
├── app.py                 # Main application
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

## Troubleshooting

### Model Not Found

Ensure your GGUF models are in the correct location and have the right filenames. Check the console output for the exact paths being used.

### Database Issues

If you encounter database corruption or want to start fresh:

```bash
python app.py --reset
```

### Memory Issues

For large documents or limited memory, consider:
- Using smaller chunk sizes
- Reducing the number of retrieved documents (k parameter)
- Using quantized models (Q4_K_M, Q5_K_M, etc.)

## License

This project is open source. Please check the individual model licenses for commercial use restrictions.
