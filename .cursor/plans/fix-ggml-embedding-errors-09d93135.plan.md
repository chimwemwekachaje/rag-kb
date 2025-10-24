<!-- 09d93135-f28c-4e2d-b2f5-5f8fbd7bb4cf c4aedca8-cb28-427c-ba4d-95a9d0ba4db5 -->
# Fix GGML Assertion Errors in RAG Application

## Problem

The application crashes with GGML assertion errors (exit code 139) when running on HuggingFace Spaces due to tensor dimension mismatches and batch size issues in the llama-cpp-python embedding process.

## Solution Overview

Implement a hybrid approach that:

1. Uses HuggingFace embeddings API when HF_TOKEN is available (for Spaces deployment)
2. Falls back to local GGUF models with optimized parameters
3. Reduces batch size from 512 to 128
4. Adds comprehensive error handling and validation

## Changes to `/Users/kachaje/andela/genAi-bootcamp/projects/code/rag-kb/app.py`

### 1. Add HuggingFace Dependencies (after imports at line 15)

Add import for HuggingFace embeddings:

```python
from langchain_huggingface import HuggingFaceEmbeddings
```

### 2. Update NomicEmbeddingFunction Class (lines 18-80)

Modify the class to:

- Reduce `n_batch` from 512 to 128 (line 35)
- Add model file validation in `_get_embedder()`
- Improve error logging with text length information

### 3. Create Hybrid Embedding Function (add new function after line 80)

Add a new function `get_embedding_function()` that:

- Checks for HF_TOKEN environment variable
- Returns HuggingFaceEmbeddings if token exists
- Falls back to NomicEmbeddingFunction for local GGUF models

### 4. Update RAGSystem.**init** (lines 83-105)

Modify to use the new `get_embedding_function()` instead of directly instantiating NomicEmbeddingFunction

### 5. Update requirements.txt

Add `langchain-huggingface` dependency

## Key Code Changes

**Line 35:** Change `n_batch=512` to `n_batch=128`

**Lines 27-42:** Add validation and update parameters:

```python
def _get_embedder(self) -> Llama:
    """Get a thread-local embedder instance"""
    if not os.path.exists(self.model_path):
        raise FileNotFoundError(f"Embedding model not found at {self.model_path}")
    
    return Llama(
        model_path=self.model_path,
        embedding=True,
        n_ctx=512,
        n_threads=1,
        n_batch=128,  # Reduced from 512
        n_gpu_layers=0,
        use_mmap=True,
        use_mlock=False,
        embedding_mode=True,
        verbose=False,
        logits_all=False,
    )
```

**New function after line 80:**

```python
def get_embedding_function():
    """Get embedding function with HuggingFace fallback for Spaces"""
    if os.getenv("HF_TOKEN"):
        print("Using HuggingFace embeddings (HF_TOKEN detected)")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    else:
        print("Using local GGUF embeddings")
        embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH") or "models/nomic-embed-text-v1.5.Q4_K_M.gguf"
        return NomicEmbeddingFunction(embedding_model_path)
```

**Line 96:** Change from:

```python
self.embedding_function = NomicEmbeddingFunction(embedding_model_path)
```

To:

```python
self.embedding_function = get_embedding_function()
```

This approach ensures the app works reliably on HuggingFace Spaces (using their API) while maintaining local GGUF model support for development.

### To-dos

- [ ] Add HuggingFaceEmbeddings import statement after line 15
- [ ] Change n_batch parameter from 512 to 128 in _get_embedder method (line 35)
- [ ] Add file existence check in _get_embedder method before creating Llama instance
- [ ] Enhance _embed_single error handling to log text length
- [ ] Create new get_embedding_function() that returns HuggingFace or GGUF embeddings based on HF_TOKEN
- [ ] Update RAGSystem.__init__ to use get_embedding_function() instead of direct NomicEmbeddingFunction
- [ ] Add langchain-huggingface to requirements.txt