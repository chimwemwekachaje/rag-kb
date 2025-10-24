"""Minimal shared fixtures for rag-kb tests without heavy dependencies."""

import pytest
import os
import tempfile
import shutil
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, MagicMock, patch


@pytest.fixture
def mock_llama_embedder():
    """Mock Llama embedding model."""
    mock_embedder = Mock()
    mock_embedder.create_embedding.return_value = {
        "data": [{"embedding": [0.1] * 768}]  # 768-dimensional embedding
    }
    return mock_embedder


@pytest.fixture
def mock_llama_llm():
    """Mock Llama LLM model."""
    mock_llm = Mock()
    mock_llm.return_value = {
        "choices": [{"text": "This is a mock response from the LLM."}]
    }
    return mock_llm


@pytest.fixture
def mock_embedding_function(mock_llama_embedder):
    """Mock NomicEmbeddingFunction."""
    mock_instance = Mock()
    mock_instance.embedder = mock_llama_embedder
    mock_instance.embed_documents.return_value = [[0.1] * 768]
    mock_instance.embed_query.return_value = [0.1] * 768
    return mock_instance


@pytest.fixture
def mock_pdf_documents():
    """Sample Document objects with mock content."""
    # Create mock Document objects
    mock_doc1 = Mock()
    mock_doc1.page_content = "This is the first page of a test document. It contains information about machine learning and artificial intelligence."
    mock_doc1.metadata = {"source": "data/test_doc1.pdf", "page": 0}
    
    mock_doc2 = Mock()
    mock_doc2.page_content = "This is the second page of the test document. It discusses neural networks and deep learning algorithms."
    mock_doc2.metadata = {"source": "data/test_doc1.pdf", "page": 1}
    
    mock_doc3 = Mock()
    mock_doc3.page_content = "This is content from another document about natural language processing and transformers."
    mock_doc3.metadata = {"source": "data/test_doc2.pdf", "page": 0}
    
    mock_doc4 = Mock()
    mock_doc4.page_content = "Additional content about RAG systems and vector databases for information retrieval."
    mock_doc4.metadata = {"source": "data/test_doc2.pdf", "page": 1}
    
    return [mock_doc1, mock_doc2, mock_doc3, mock_doc4]


@pytest.fixture
def mock_pdf_chunks():
    """Pre-chunked documents for testing."""
    mock_chunk1 = Mock()
    mock_chunk1.page_content = "This is the first page of a test document. It contains information about machine learning and artificial intelligence."
    mock_chunk1.metadata = {"source": "data/test_doc1.pdf", "page": 0, "id": "data/test_doc1.pdf:0:0"}
    
    mock_chunk2 = Mock()
    mock_chunk2.page_content = "This is the second page of the test document. It discusses neural networks and deep learning algorithms."
    mock_chunk2.metadata = {"source": "data/test_doc1.pdf", "page": 1, "id": "data/test_doc1.pdf:1:0"}
    
    mock_chunk3 = Mock()
    mock_chunk3.page_content = "This is content from another document about natural language processing and transformers."
    mock_chunk3.metadata = {"source": "data/test_doc2.pdf", "page": 0, "id": "data/test_doc2.pdf:0:0"}
    
    mock_chunk4 = Mock()
    mock_chunk4.page_content = "Additional content about RAG systems and vector databases for information retrieval."
    mock_chunk4.metadata = {"source": "data/test_doc2.pdf", "page": 1, "id": "data/test_doc2.pdf:1:0"}
    
    return [mock_chunk1, mock_chunk2, mock_chunk3, mock_chunk4]


@pytest.fixture
def temp_data_dir():
    """Temporary directory with mock PDF files."""
    temp_dir = tempfile.mkdtemp()
    
    # Create subdirectories
    week1_dir = os.path.join(temp_dir, "Week 1")
    week2_dir = os.path.join(temp_dir, "Week 2")
    os.makedirs(week1_dir, exist_ok=True)
    os.makedirs(week2_dir, exist_ok=True)
    
    # Create mock PDF files (empty files with .pdf extension)
    mock_files = [
        "Course Summary.pdf",
        "Week 1/Week 1 Day 1.pdf",
        "Week 1/Week 1 Day 2.pdf",
        "Week 2/Week 2 Day 1.pdf",
        "Week 2/Week 2 Day 2.pdf"
    ]
    
    for file_path in mock_files:
        full_path = os.path.join(temp_dir, file_path)
        with open(full_path, 'w') as f:
            f.write("Mock PDF content")
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_folder_structure():
    """Mock folder structure for UI tests."""
    return {
        "Week 1": {
            "Week 1 Day 1.pdf": "data/Week 1/Week 1 Day 1.pdf",
            "Week 1 Day 2.pdf": "data/Week 1/Week 1 Day 2.pdf"
        },
        "Week 2": {
            "Week 2 Day 1.pdf": "data/Week 2/Week 2 Day 1.pdf",
            "Week 2 Day 2.pdf": "data/Week 2/Week 2 Day 2.pdf"
        },
        "Course Summary.pdf": "data/Course Summary.pdf"
    }


@pytest.fixture
def mock_pdf_loader():
    """Mock PyPDFDirectoryLoader."""
    mock_loader = Mock()
    mock_documents = [
        Mock(page_content="Mock PDF content 1", metadata={"source": "data/test1.pdf", "page": 0}),
        Mock(page_content="Mock PDF content 2", metadata={"source": "data/test2.pdf", "page": 0})
    ]
    mock_loader.load.return_value = mock_documents
    return mock_loader


@pytest.fixture
def mock_text_splitter():
    """Mock RecursiveCharacterTextSplitter."""
    mock_splitter = Mock()
    mock_chunks = [
        Mock(page_content="Chunk 1 content", metadata={"source": "data/test1.pdf", "page": 0}),
        Mock(page_content="Chunk 2 content", metadata={"source": "data/test1.pdf", "page": 0})
    ]
    mock_splitter.split_documents.return_value = mock_chunks
    return mock_splitter


@pytest.fixture
def mock_vectorstore():
    """Mock Chroma vectorstore for unit tests."""
    mock_vs = Mock()
    mock_vs.get.return_value = {"ids": []}
    mock_vs.add_documents.return_value = None
    mock_vs.persist.return_value = None
    mock_vs.similarity_search_with_score.return_value = [
        (Mock(page_content="Test content", metadata={"source": "test.pdf"}), 0.9)
    ]
    return mock_vs


@pytest.fixture
def sample_query_result():
    """Sample RAG query result."""
    return {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data.",
        "context_docs": [
            Mock(
                page_content="Machine learning algorithms can learn patterns from data.",
                metadata={"source": "data/ml_doc.pdf", "page": 0, "id": "data/ml_doc.pdf:0:0"}
            )
        ],
        "time_taken": 0.5
    }


@pytest.fixture
def mock_gradio_components():
    """Mock Gradio components for UI testing."""
    mock_gr = Mock()
    # Mock common Gradio components
    mock_gr.Blocks.return_value.__enter__ = Mock()
    mock_gr.Blocks.return_value.__exit__ = Mock()
    mock_gr.Accordion.return_value.__enter__ = Mock()
    mock_gr.Accordion.return_value.__exit__ = Mock()
    mock_gr.Column.return_value.__enter__ = Mock()
    mock_gr.Column.return_value.__exit__ = Mock()
    mock_gr.Row.return_value.__enter__ = Mock()
    mock_gr.Row.return_value.__exit__ = Mock()
    
    # Mock component classes
    mock_gr.Button.return_value.click = Mock()
    mock_gr.TextArea.return_value = Mock()
    mock_gr.Dropdown.return_value = Mock()
    mock_gr.Number.return_value = Mock()
    mock_gr.File.return_value = Mock()
    mock_gr.State.return_value = Mock()
    
    # Mock PDF component
    mock_pdf = Mock()
    mock_pdf.return_value = Mock()
    
    return mock_gr, mock_pdf


@pytest.fixture(autouse=True)
def mock_os_path_exists():
    """Mock os.path.exists to control file system behavior in tests."""
    with patch('os.path.exists') as mock_exists:
        # Default to True for most paths, can be overridden in specific tests
        mock_exists.return_value = True
        yield mock_exists


@pytest.fixture(autouse=True)
def mock_shutil_rmtree():
    """Mock shutil.rmtree to prevent actual file deletion in tests."""
    with patch('shutil.rmtree') as mock_rmtree:
        yield mock_rmtree


@pytest.fixture
def mock_time():
    """Mock time.time for consistent timing in tests."""
    with patch('time.time') as mock_time_func:
        mock_time_func.side_effect = [0.0, 0.5]  # start_time, end_time
        yield mock_time_func
