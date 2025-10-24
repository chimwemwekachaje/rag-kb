<!-- 8115c2b5-a7d0-4c72-9522-8abb21684c0a fb03a896-99b1-4714-b2a5-ffc6996c80c5 -->
# Optimize Embedding Loading in NomicEmbeddingFunction

## Current Performance Bottlenecks

The `NomicEmbeddingFunction` (lines 18-65 in `app.py`) currently:

- Processes texts sequentially in a for-loop (no batching)
- Makes individual `create_embedding()` calls for each text
- Uses conservative parameters that don't leverage system capabilities

## Optimization Strategy

### 1. Add Batch Processing with Memory-Aware Chunking

**File:** `app.py` - `NomicEmbeddingFunction` class

**Changes:**

- Add a `batch_size` parameter to `__init__` (default: 16 for memory efficiency)
- Modify `embed_documents()` to process texts in batches instead of one-by-one
- Use `ThreadPoolExecutor` for parallel processing within batches
- Keep batch size small to avoid OOM on limited memory systems

**Key code changes:**

```python
# In __init__: add batch_size and n_batch parameters
self.batch_size = 16  # Process 16 texts concurrently
self.n_batch = 512  # llama.cpp batch parameter

# In embed_documents: use ThreadPoolExecutor for parallel processing
from concurrent.futures import ThreadPoolExecutor, as_completed

def embed_documents(self, texts: List[str]) -> List[List[float]]:
    embeddings = [None] * len(texts)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for idx, text in enumerate(texts):
            future = executor.submit(self._embed_single, text)
            futures[future] = idx
        
        for future in as_completed(futures):
            idx = futures[future]
            embeddings[idx] = future.result()
    
    return embeddings
```

### 2. Optimize llama.cpp Initialization Parameters

**File:** `app.py` - `NomicEmbeddingFunction.__init__`

**Changes:**

- Add `n_batch=512` for better throughput
- Reduce `n_threads` to 2-3 to avoid CPU contention (better for limited memory)
- Add `use_mmap=True` and `use_mlock=False` for memory-efficient model loading
- Set `n_gpu_layers=0` explicitly (since using CPU)

### 3. Extract Single Embedding Method

**File:** `app.py` - `NomicEmbeddingFunction` class

**Changes:**

- Create a private `_embed_single()` method that handles embedding of one text
- This method will be called by both `embed_query()` and the parallel workers in `embed_documents()`
- Reduces code duplication and ensures consistent error handling

### 4. Add Progress Tracking for Large Batches

**File:** `app.py` - `NomicEmbeddingFunction.embed_documents`

**Changes:**

- Add optional progress callback for monitoring large batch operations
- Helps identify bottlenecks during database population

## Memory Considerations

- Batch size of 16 keeps memory usage low (~12MB per batch for embeddings)
- Thread pool limited to 4 workers prevents memory spikes
- `use_mmap=True` allows OS to manage model memory efficiently
- No caching layer (would increase memory usage)

## Expected Performance Improvements

- **Batch operations (document ingestion):** 3-4x faster through parallel processing
- **Query operations:** Minimal overhead, maintains fast response times
- **Memory usage:** Similar or slightly lower due to mmap optimization

### To-dos

- [ ] Add concurrent.futures imports for ThreadPoolExecutor
- [ ] Update NomicEmbeddingFunction.__init__ with optimized llama.cpp parameters (n_batch, use_mmap, reduced n_threads)
- [ ] Create private _embed_single() method for embedding one text with error handling
- [ ] Refactor embed_query() to use _embed_single() method
- [ ] Rewrite embed_documents() to use ThreadPoolExecutor for parallel batch processing
- [ ] Test the optimized embedding function with sample documents to verify performance improvements