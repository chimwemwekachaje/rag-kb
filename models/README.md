# Models Directory

This directory contains the GGUF model files required for the Llama RAG Knowledge Base.

## Required Models

### 1. Embedding Model
- **File**: `nomic-embed-text-v1.5.Q4_K_M.gguf`
- **Purpose**: Generate embeddings for document retrieval
- **Dimensions**: 768
- **Download**: [Hugging Face - nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

### 2. LLM Model
- **File**: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
- **Purpose**: Generate responses based on retrieved context
- **Parameters**: 1.1B
- **Download**: [Hugging Face - TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)

## Download Instructions

1. Visit the Hugging Face links above
2. Download the Q4_K_M quantized versions for optimal performance
3. Place the downloaded `.gguf` files in this directory
4. Ensure the filenames match exactly as listed above

## Alternative Models

You can use different GGUF models by specifying their paths when running the application:

```bash
python app.py --embedding-model "path/to/your/embedding.gguf" --llm-model "path/to/your/llm.gguf"
```

## Model Quantization

- **Q4_K_M**: Recommended for most use cases (good balance of size and quality)
- **Q5_K_M**: Higher quality, larger size
- **Q8_0**: Highest quality, largest size
- **Q2_K**: Smallest size, lower quality

Choose based on your hardware capabilities and quality requirements.
