<!-- 716c33fe-648d-42d1-8b5b-204690dbc1bb a7b87be3-2e0d-4298-8108-f438c3105ddf -->
# Fix Slow Query Fetching Performance

## Problem Analysis

Current issues causing 10-20+ second query times:

1. **Model reloading on every query**: `embed_query()` creates/destroys Llama model instance for each query
2. **No vector indexing**: ChromaDB uses brute-force search across 351 embeddings
3. **Inefficient embedding architecture**: No model reuse or caching

## Implementation Plan

### 1. Implement Persistent Embedding Model Instance

**File**: `app.py` - `NomicEmbeddingFunction` class (lines 19-89)

**Changes**:

- Add thread-local storage using `threading.local()` for embedder instances
- Initialize embedder once per thread and reuse it
- Remove the wasteful create/destroy pattern in `_embed_single()`
- Add proper cleanup method for graceful shutdown

**Key code locations**:

- Line 26: Replace comment with actual `threading.local()` initialization
- Lines 48-71: Refactor `_embed_single()` to reuse embedder instance
- Lines 87-89: Update `embed_query()` to use persistent instance

### 2. Enable ChromaDB HNSW Indexing

**File**: `app.py` - `RAGSystem._setup_vectorstore()` (lines 130-145)

**Changes**:

- Configure ChromaDB with HNSW index parameters for efficient ANN search
- Add collection metadata for optimal indexing
- Set appropriate `hnsw:space` (cosine) and `hnsw:construction_ef` parameters

**Configuration to add**:

```python
collection_metadata = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,
    "hnsw:M": 16
}
```

### 3. Add Query Performance Monitoring

**File**: `app.py` - `RAGSystem.query()` (lines 266-286)

**Changes**:

- Add timing breakdowns for embedding vs search vs generation
- Log performance metrics to help diagnose future issues
- Return detailed timing information in response

### 4. Optimize Embedding Batch Processing

**File**: `app.py` - `NomicEmbeddingFunction.embed_documents()` (lines 73-85)

**Changes**:

- Use the same persistent model instance pattern
- Remove per-document model creation
- Maintain sequential processing to avoid threading issues

### 5. Update Tests for New Architecture

**Files**: `tests/unit/test_embedding_function.py`, `tests/integration/test_rag_workflow.py`

**Changes**:

- Update mocks to reflect persistent model architecture
- Add performance regression tests
- Test cleanup methods work correctly

## Expected Performance Improvements

- **Before**: 10-20+ seconds per query
- **After**: <2 seconds per query
  - Query embedding: ~0.5s (reusing model)
  - Vector search with HNSW: ~0.1s
  - LLM generation: ~1-1.5s

## Files to Modify

1. `app.py` (main changes)
2. `tests/unit/test_embedding_function.py` (test updates)
3. `tests/integration/test_rag_workflow.py` (test updates)

## Migration Notes

- Existing ChromaDB will need to be rebuilt with HNSW indexing (automatic on first query)
- Users should run with `--reset` flag once to regenerate with new index
- No breaking API changes - all improvements are internal

### To-dos

- [ ] Implement thread-local persistent embedder instances in NomicEmbeddingFunction
- [ ] Enable ChromaDB HNSW indexing in _setup_vectorstore()
- [ ] Add detailed timing breakdowns in query() method
- [ ] Update unit and integration tests for new architecture
- [ ] Test and verify query performance improvements