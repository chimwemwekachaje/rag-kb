<!-- a51facb1-4a77-4619-afd0-438cc4513d37 f344b665-95b5-4cc5-b403-86b1001d6df9 -->
# RAG-KB Comprehensive Testing Suite

## Overview

Create a complete testing infrastructure for the rag-kb project with unit tests for core/utility functions, integration tests with in-memory ChromaDB, and Gradio UI component tests. Mock all Llama model calls while using real ChromaDB operations with test fixtures.

## Test Structure

### Directory Organization

```
rag-kb/
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # Shared fixtures and configuration
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_embedding_function.py
│   │   ├── test_rag_system.py
│   │   └── test_utility_functions.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_rag_workflow.py
│   │   └── test_database_operations.py
│   └── ui/
│       ├── __init__.py
│       └── test_gradio_components.py
```

## Core Components to Test

### 1. NomicEmbeddingFunction Class (Unit Tests)

**File**: `tests/unit/test_embedding_function.py`

- `__init__`: Test initialization with mocked Llama model
- `embed_documents`: Test with normal text, empty text, errors
- `embed_query`: Test with normal text, empty text, errors
- Error handling and fallback to zero vectors

### 2. RAGSystem Class (Unit Tests)

**File**: `tests/unit/test_rag_system.py`

- `__init__`: Test initialization with mocked models
- `_setup_vectorstore`: Test new/existing vectorstore loading
- `calculate_chunk_ids`: Test ID generation logic
- `load_documents`: Test with mock PDF directory loader
- `split_documents`: Test chunking logic
- `add_documents`: Test deduplication and adding new chunks
- `populate_database`: Test database population flow
- `retrieve_documents`: Test similarity search
- `generate_response`: Test LLM response generation
- `query`: Test end-to-end query flow
- `clear_database`: Test database cleanup

### 3. Utility Functions (Unit Tests)

**File**: `tests/unit/test_utility_functions.py`

- `build_nested_accordions`: Test folder structure scanning
- `create_accordion_ui`: Test UI creation (without launching)
- `main` function argument parsing

### 4. Integration Tests

**File**: `tests/integration/test_rag_workflow.py`

- Full RAG workflow with in-memory ChromaDB
- Document loading → chunking → embedding → storage → retrieval → generation
- Test with mock PDF content using fixtures

**File**: `tests/integration/test_database_operations.py`

- ChromaDB persistence and retrieval with real operations
- Document deduplication across multiple adds
- Batch document processing

### 5. Gradio UI Tests

**File**: `tests/ui/test_gradio_components.py`

- Test accordion UI creation
- Test button click handlers
- Test PDF navigation components
- Test query input/output components
- Test sources dropdown functionality

## Test Configuration

### conftest.py Fixtures

```python
# Key fixtures to include:
- mock_llama_embedder: Mock Llama embedding model
- mock_llama_llm: Mock Llama LLM model
- mock_embedding_function: Mock NomicEmbeddingFunction
- in_memory_vectorstore: In-memory ChromaDB instance
- mock_pdf_documents: Sample Document objects with mock content
- mock_pdf_chunks: Pre-chunked documents
- temp_data_dir: Temporary directory with mock PDFs
- sample_folder_structure: Mock folder structure for UI tests
```

## Mocking Strategy

### Unit Tests (Full Mocking)

- Mock `llama_cpp.Llama` completely
- Mock `PyPDFDirectoryLoader`
- Mock file system operations
- Mock ChromaDB for isolated tests

### Integration Tests (Selective Mocking)

- Use real in-memory ChromaDB (`chromadb.EphemeralClient`)
- Mock only Llama model calls
- Use mock PDF content without actual files
- Test real chunking and embedding logic with mocked models

## Dependencies to Add

Update `requirements.txt`:

```
pytest>=8.0.0
pytest-cov>=4.1.0
pytest-mock>=3.12.0
coverage[toml]>=7.4.0
```

Update `pyproject.toml` with pytest and coverage configuration:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--cov=app",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--cov-branch"
]

[tool.coverage.run]
source = ["."]
omit = [
    "tests/*",
    "venv/*",
    ".venv/*",
    "*/site-packages/*"
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
```

## Coverage Targets

### High Priority (Core Functionality)

- `NomicEmbeddingFunction`: 90%+
- `RAGSystem.__init__`, `query`, `retrieve_documents`: 90%+
- `calculate_chunk_ids`, `split_documents`: 95%+

### Medium Priority

- `add_documents`, `populate_database`: 85%+
- `generate_response`, `_setup_vectorstore`: 85%+

### Lower Priority (UI/Utility)

- `build_nested_accordions`: 75%+
- Gradio UI functions: 70%+
- `main` function: 70%+

## Test Execution Commands

Create test execution script in project root:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html --cov-report=term-missing

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/ui/

# Run with verbose output
pytest -v -s
```

## Key Testing Considerations

1. **Model Mocking**: All Llama model initialization and calls mocked to avoid loading actual GGUF files
2. **ChromaDB**: Use ephemeral in-memory client for integration tests
3. **PDF Content**: Generate mock Document objects without requiring actual PDF files
4. **Gradio Components**: Test component creation and event handler registration without launching server
5. **Coverage Reporting**: Generate HTML reports but don't fail builds if below 80%

## Files to Create/Modify

### New Files

- `tests/conftest.py`
- `tests/unit/test_embedding_function.py`
- `tests/unit/test_rag_system.py`
- `tests/unit/test_utility_functions.py`
- `tests/integration/test_rag_workflow.py`
- `tests/integration/test_database_operations.py`
- `tests/ui/test_gradio_components.py`
- `tests/__init__.py`
- `tests/unit/__init__.py`
- `tests/integration/__init__.py`
- `tests/ui/__init__.py`
- `.coveragerc` (optional, config can be in pyproject.toml)

### Modified Files

- `requirements.txt` (add testing dependencies)
- `pyproject.toml` (add pytest and coverage configuration)

## Success Criteria

- All tests pass independently and together
- Coverage reaches ~80% overall
- Integration tests use in-memory ChromaDB successfully
- No actual model files loaded during tests
- Tests run in under 30 seconds total
- Coverage HTML report generated in `htmlcov/`

### To-dos

- [ ] Add pytest, pytest-cov, pytest-mock, and coverage to requirements.txt and configure pyproject.toml
- [ ] Create tests/conftest.py with shared fixtures for mocked models, ChromaDB, and mock documents
- [ ] Create unit tests for NomicEmbeddingFunction class with full mocking
- [ ] Create unit tests for RAGSystem class methods with mocked dependencies
- [ ] Create unit tests for utility functions (build_nested_accordions, etc.)
- [ ] Create integration tests for full RAG workflow with in-memory ChromaDB
- [ ] Create integration tests for database operations with real ChromaDB operations
- [ ] Create Gradio UI component tests without launching the application
- [ ] Run full test suite and verify coverage meets ~80% target, generate coverage reports