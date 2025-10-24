"""Shared fixtures and configuration for rag-kb tests."""

import os
import tempfile
import shutil
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, MagicMock, patch
import pytest
from langchain_core.documents import Document
# Try different import paths for Chroma
try:
    from langchain_chroma import Chroma
except ImportError:
    try:
        from langchain.vectorstores import Chroma
    except ImportError:
        # Fallback mock for testing
        Chroma = Mock
import chromadb
from chromadb.config import Settings


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
    with patch('app.NomicEmbeddingFunction') as mock_class:
        mock_instance = Mock()
        mock_instance.embedder = mock_llama_embedder
        mock_instance.embed_documents.return_value = [[0.1] * 768]
        mock_instance.embed_query.return_value = [0.1] * 768
        mock_class.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def in_memory_vectorstore():
    """In-memory ChromaDB instance for integration tests."""
    # Create ephemeral client for testing
    client = chromadb.EphemeralClient()
    collection = client.create_collection("test_collection")
    
    # Create a mock embedding function for the vectorstore
    mock_embedding_function = Mock()
    mock_embedding_function.embed_documents.return_value = [[0.1] * 768]
    mock_embedding_function.embed_query.return_value = [0.1] * 768
    
    vectorstore = Chroma(
        client=client,
        collection_name="test_collection",
        embedding_function=mock_embedding_function
    )
    
    yield vectorstore
    
    # Cleanup
    client.delete_collection("test_collection")


@pytest.fixture
def mock_pdf_documents():
    """Sample Document objects with mock content."""
    return [
        Document(
            page_content="This is the first page of a test document. It contains information about machine learning and artificial intelligence.",
            metadata={"source": "data/test_doc1.pdf", "page": 0}
        ),
        Document(
            page_content="This is the second page of the test document. It discusses neural networks and deep learning algorithms.",
            metadata={"source": "data/test_doc1.pdf", "page": 1}
        ),
        Document(
            page_content="This is content from another document about natural language processing and transformers.",
            metadata={"source": "data/test_doc2.pdf", "page": 0}
        ),
        Document(
            page_content="Additional content about RAG systems and vector databases for information retrieval.",
            metadata={"source": "data/test_doc2.pdf", "page": 1}
        )
    ]


@pytest.fixture
def mock_pdf_chunks():
    """Pre-chunked documents for testing."""
    return [
        Document(
            page_content="This is the first page of a test document. It contains information about machine learning and artificial intelligence.",
            metadata={"source": "data/test_doc1.pdf", "page": 0, "id": "data/test_doc1.pdf:0:0"}
        ),
        Document(
            page_content="This is the second page of the test document. It discusses neural networks and deep learning algorithms.",
            metadata={"source": "data/test_doc1.pdf", "page": 1, "id": "data/test_doc1.pdf:1:0"}
        ),
        Document(
            page_content="This is content from another document about natural language processing and transformers.",
            metadata={"source": "data/test_doc2.pdf", "page": 0, "id": "data/test_doc2.pdf:0:0"}
        ),
        Document(
            page_content="Additional content about RAG systems and vector databases for information retrieval.",
            metadata={"source": "data/test_doc2.pdf", "page": 1, "id": "data/test_doc2.pdf:1:0"}
        )
    ]


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
    with patch('app.PyPDFDirectoryLoader') as mock_loader_class:
        mock_loader = Mock()
        mock_documents = [
            Document(
                page_content="Mock PDF content 1",
                metadata={"source": "data/test1.pdf", "page": 0}
            ),
            Document(
                page_content="Mock PDF content 2", 
                metadata={"source": "data/test2.pdf", "page": 0}
            )
        ]
        mock_loader.load.return_value = mock_documents
        mock_loader_class.return_value = mock_loader
        yield mock_loader


@pytest.fixture
def mock_text_splitter():
    """Mock RecursiveCharacterTextSplitter."""
    with patch('app.RecursiveCharacterTextSplitter') as mock_splitter_class:
        mock_splitter = Mock()
        mock_chunks = [
            Document(
                page_content="Chunk 1 content",
                metadata={"source": "data/test1.pdf", "page": 0}
            ),
            Document(
                page_content="Chunk 2 content",
                metadata={"source": "data/test1.pdf", "page": 0}
            )
        ]
        mock_splitter.split_documents.return_value = mock_chunks
        mock_splitter_class.return_value = mock_splitter
        yield mock_splitter


@pytest.fixture
def mock_vectorstore():
    """Mock Chroma vectorstore for unit tests."""
    mock_vs = Mock()
    mock_vs.get.return_value = {"ids": []}
    mock_vs.add_documents.return_value = None
    mock_vs.persist.return_value = None
    mock_vs.similarity_search_with_score.return_value = [
        (Document(page_content="Test content", metadata={"source": "test.pdf"}), 0.9)
    ]
    return mock_vs


@pytest.fixture
def sample_query_result():
    """Sample RAG query result."""
    return {
        "question": "What is machine learning?",
        "answer": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data.",
        "context_docs": [
            Document(
                page_content="Machine learning algorithms can learn patterns from data.",
                metadata={"source": "data/ml_doc.pdf", "page": 0, "id": "data/ml_doc.pdf:0:0"}
            )
        ],
        "time_taken": 0.5
    }


@pytest.fixture
def mock_gradio_components():
    """Mock Gradio components for UI testing."""
    with patch('app.gr') as mock_gr:
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
        with patch('app.PDF') as mock_pdf:
            mock_pdf.return_value = Mock()
            yield mock_gr, mock_pdf


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
