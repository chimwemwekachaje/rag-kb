"""Unit tests for RAGSystem class."""

import pytest
import os
import shutil
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document
from app import RAGSystem


class TestRAGSystem:
    """Test cases for RAGSystem class."""

    def test_init_success(self, mock_embedding_function, mock_llama_llm, mock_vectorstore):
        """Test successful initialization of RAGSystem."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_get_embedding.return_value = mock_embedding_function
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf",
                chroma_persist_dir="./test_chroma",
                data_path="./test_data"
            )
            
            # Verify initialization
            assert rag.embedding_model_path == "test_embedding.gguf"
            assert rag.llm_model_path == "test_llm.gguf"
            assert rag.chroma_persist_dir == "./test_chroma"
            assert rag.data_path == "./test_data"
            assert rag.embedding_function == mock_embedding_function
            assert rag.llm == mock_llama_llm
            assert rag.vectorstore == mock_vectorstore

    def test_setup_vectorstore_existing_directory(self, mock_embedding_function, mock_llama_llm):
        """Test _setup_vectorstore with existing directory."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding, \
             patch('os.path.exists') as mock_exists:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore = Mock()
            mock_get_embedding.return_value = mock_embedding_function
            mock_exists.return_value = True
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf",
                chroma_persist_dir="./test_chroma"
            )
            
            # Verify Chroma was called with persist_directory
            mock_chroma_class.assert_called_once_with(
                persist_directory="./test_chroma",
                embedding_function=mock_embedding_function
            )

    def test_setup_vectorstore_new_directory(self, mock_embedding_function, mock_llama_llm):
        """Test _setup_vectorstore with new directory."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding, \
             patch('os.path.exists') as mock_exists:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore = Mock()
            mock_get_embedding.return_value = mock_embedding_function
            mock_exists.return_value = False
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf",
                chroma_persist_dir="./test_chroma"
            )
            
            # Verify Chroma was called with persist_directory
            mock_chroma_class.assert_called_once_with(
                persist_directory="./test_chroma",
                embedding_function=mock_embedding_function
            )

    def test_calculate_chunk_ids(self, mock_embedding_function, mock_llama_llm, mock_vectorstore):
        """Test calculate_chunk_ids method."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_get_embedding.return_value = mock_embedding_function
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            
            # Test chunks with same page
            chunks = [
                Document(
                    page_content="First chunk",
                    metadata={"source": "data/test.pdf", "page": 0}
                ),
                Document(
                    page_content="Second chunk",
                    metadata={"source": "data/test.pdf", "page": 0}
                ),
                Document(
                    page_content="Third chunk",
                    metadata={"source": "data/test.pdf", "page": 1}
                )
            ]
            
            result = rag.calculate_chunk_ids(chunks)
            
            # Verify chunk IDs
            assert result[0].metadata["id"] == "data/test.pdf:0:0"
            assert result[1].metadata["id"] == "data/test.pdf:0:1"
            assert result[2].metadata["id"] == "data/test.pdf:1:0"

    def test_load_documents(self, mock_embedding_function, mock_llama_llm, mock_vectorstore, mock_pdf_loader):
        """Test load_documents method."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding, \
             patch('app.PyPDFDirectoryLoader') as mock_pdf_loader_class:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_get_embedding.return_value = mock_embedding_function
            mock_pdf_loader_class.return_value = mock_pdf_loader
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf",
                data_path="./test_data"
            )
            
            result = rag.load_documents()
            
            # Verify PyPDFDirectoryLoader was called
            mock_pdf_loader_class.assert_called_once_with("./test_data")
            mock_pdf_loader.load.assert_called_once()
            assert result is not None
            assert len(result) == 2  # From mock_pdf_loader fixture

    def test_split_documents(self, mock_embedding_function, mock_llama_llm, mock_vectorstore, mock_text_splitter):
        """Test split_documents method."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding, \
             patch('app.RecursiveCharacterTextSplitter') as mock_text_splitter_class:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_get_embedding.return_value = mock_embedding_function
            mock_text_splitter_class.return_value = mock_text_splitter
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            
            documents = [Document(page_content="Test content", metadata={"source": "test.pdf"})]
            result = rag.split_documents(documents)
            
            # Verify text splitter was called
            mock_text_splitter_class.assert_called_once()
            mock_text_splitter.split_documents.assert_called_once_with(documents)
            assert result is not None
            assert len(result) == 2  # From mock_text_splitter fixture

    def test_add_documents_new_chunks(self, mock_embedding_function, mock_llama_llm, mock_vectorstore, mock_pdf_chunks):
        """Test add_documents with new chunks."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_get_embedding.return_value = mock_embedding_function
            
            # Mock vectorstore.get to return empty existing items
            mock_vectorstore.get.return_value = {"ids": []}
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            
            rag.add_documents(mock_pdf_chunks)
            
            # Verify add_documents was called
            mock_vectorstore.add_documents.assert_called_once()
            mock_vectorstore.persist.assert_called_once()

    def test_add_documents_no_new_chunks(self, mock_embedding_function, mock_llama_llm, mock_vectorstore, mock_pdf_chunks):
        """Test add_documents with no new chunks."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_get_embedding.return_value = mock_embedding_function
            
            # Mock vectorstore.get to return existing IDs
            existing_ids = [chunk.metadata["id"] for chunk in mock_pdf_chunks]
            mock_vectorstore.get.return_value = {"ids": existing_ids}
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            
            rag.add_documents(mock_pdf_chunks)
            
            # Verify add_documents was not called
            mock_vectorstore.add_documents.assert_not_called()
            mock_vectorstore.persist.assert_not_called()

    def test_populate_database_success(self, mock_embedding_function, mock_llama_llm, mock_vectorstore, mock_pdf_loader):
        """Test populate_database with existing data directory."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding, \
             patch('app.PyPDFDirectoryLoader') as mock_pdf_loader_class, \
             patch('os.path.exists') as mock_exists:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_get_embedding.return_value = mock_embedding_function
            mock_pdf_loader_class.return_value = mock_pdf_loader
            mock_exists.return_value = True
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf",
                data_path="./test_data"
            )
            
            rag.populate_database()
            
            # Verify load_documents was called (through mock_pdf_loader)
            mock_pdf_loader_class.assert_called_once_with("./test_data")
            mock_pdf_loader.load.assert_called_once()

    def test_populate_database_no_data_directory(self, mock_embedding_function, mock_llama_llm, mock_vectorstore):
        """Test populate_database with non-existent data directory."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding, \
             patch('os.path.exists') as mock_exists:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_get_embedding.return_value = mock_embedding_function
            mock_exists.return_value = False
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf",
                data_path="./nonexistent_data"
            )
            
            rag.populate_database()
            
            # Should not call any document loading methods

    def test_retrieve_documents(self, mock_embedding_function, mock_llama_llm, mock_vectorstore):
        """Test retrieve_documents method."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_get_embedding.return_value = mock_embedding_function
            
            # Mock similarity search results
            mock_docs = [
                (Document(page_content="Test content 1", metadata={"source": "test1.pdf"}), 0.9),
                (Document(page_content="Test content 2", metadata={"source": "test2.pdf"}), 0.8)
            ]
            mock_vectorstore.similarity_search_with_score.return_value = mock_docs
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            
            result = rag.retrieve_documents("test query", k=2)
            
            # Verify similarity search was called
            mock_vectorstore.similarity_search_with_score.assert_called_once_with("test query", k=2)
            
            # Verify result format
            assert len(result) == 2
            assert all(isinstance(doc, Document) for doc in result)

    def test_generate_response(self, mock_embedding_function, mock_llama_llm, mock_vectorstore):
        """Test generate_response method."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_get_embedding.return_value = mock_embedding_function
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            
            query = "What is machine learning?"
            context = "Machine learning is a subset of AI."
            
            result = rag.generate_response(query, context)
            
            # Verify LLM was called with correct prompt
            mock_llama_llm.assert_called_once()
            call_args = mock_llama_llm.call_args
            
            # Check that the prompt contains the query and context
            prompt = call_args[0][0]
            assert query in prompt
            assert context in prompt
            assert "<|system|>" in prompt
            assert "<|user|>" in prompt
            assert "<|assistant|>" in prompt
            
            # Verify response format
            assert result == "This is a mock response from the LLM."

    def test_query_complete_flow(self, mock_embedding_function, mock_llama_llm, mock_vectorstore, mock_time):
        """Test complete query flow."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_get_embedding.return_value = mock_embedding_function
            
            # Mock similarity search results
            mock_docs = [
                (Document(page_content="Test content", metadata={"source": "test.pdf"}), 0.9)
            ]
            mock_vectorstore.similarity_search_with_score.return_value = mock_docs
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            
            result = rag.query("What is AI?", k=3)
            
            # Verify result structure
            assert "question" in result
            assert "answer" in result
            assert "context_docs" in result
            assert "time_taken" in result
            assert "timing_breakdown" in result
            
            assert result["question"] == "What is AI?"
            assert result["answer"] == "This is a mock response from the LLM."
            assert len(result["context_docs"]) == 1
            assert result["time_taken"] == 0.5
            
            # Verify timing breakdown structure
            timing = result["timing_breakdown"]
            assert "retrieval_time" in timing
            assert "context_time" in timing
            assert "generation_time" in timing
            assert "total_time" in timing

    def test_clear_database(self, mock_embedding_function, mock_llama_llm, mock_vectorstore):
        """Test clear_database method."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding, \
             patch('os.path.exists') as mock_exists, \
             patch('shutil.rmtree') as mock_rmtree:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_get_embedding.return_value = mock_embedding_function
            mock_exists.return_value = True
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf",
                chroma_persist_dir="./test_chroma"
            )
            
            rag.clear_database()
            
            # Verify rmtree was called
            mock_rmtree.assert_called_once_with("./test_chroma")

    def test_clear_database_nonexistent(self, mock_embedding_function, mock_llama_llm, mock_vectorstore):
        """Test clear_database with non-existent directory."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.get_embedding_function') as mock_get_embedding, \
             patch('os.path.exists') as mock_exists, \
             patch('shutil.rmtree') as mock_rmtree:
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_get_embedding.return_value = mock_embedding_function
            mock_exists.return_value = False
            
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf",
                chroma_persist_dir="./nonexistent_chroma"
            )
            
            rag.clear_database()
            
            # Verify rmtree was not called
            mock_rmtree.assert_not_called()
