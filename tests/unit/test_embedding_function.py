"""Unit tests for NomicEmbeddingFunction class."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app import NomicEmbeddingFunction


class TestNomicEmbeddingFunction:
    """Test cases for NomicEmbeddingFunction class."""

    def test_init_success(self, mock_llama_embedder):
        """Test successful initialization of NomicEmbeddingFunction."""
        with patch('app.Llama') as mock_llama_class:
            mock_llama_class.return_value = mock_llama_embedder
            
            embedding_function = NomicEmbeddingFunction("test_model.gguf")
            
            # Verify Llama was called with correct parameters
            mock_llama_class.assert_called_once_with(
                model_path="test_model.gguf",
                embedding=True,
                n_ctx=512,
                n_threads=4,
                embedding_mode=True,
                verbose=False,
                logits_all=False
            )
            assert embedding_function.embedder == mock_llama_embedder

    def test_embed_documents_success(self, mock_llama_embedder):
        """Test successful embedding of multiple documents."""
        with patch('app.Llama') as mock_llama_class:
            mock_llama_class.return_value = mock_llama_embedder
            
            embedding_function = NomicEmbeddingFunction("test_model.gguf")
            
            texts = ["First document", "Second document"]
            result = embedding_function.embed_documents(texts)
            
            # Verify create_embedding was called for each text
            assert mock_llama_embedder.create_embedding.call_count == 2
            mock_llama_embedder.create_embedding.assert_any_call("First document")
            mock_llama_embedder.create_embedding.assert_any_call("Second document")
            
            # Verify result format
            assert len(result) == 2
            assert all(len(embedding) == 768 for embedding in result)
            assert all(isinstance(embedding, list) for embedding in result)

    def test_embed_documents_empty_text(self, mock_llama_embedder):
        """Test embedding with empty text."""
        with patch('app.Llama') as mock_llama_class:
            mock_llama_class.return_value = mock_llama_embedder
            
            embedding_function = NomicEmbeddingFunction("test_model.gguf")
            
            texts = ["", "   ", "Valid text"]
            result = embedding_function.embed_documents(texts)
            
            # Empty texts should not call create_embedding
            assert mock_llama_embedder.create_embedding.call_count == 1
            mock_llama_embedder.create_embedding.assert_called_once_with("Valid text")
            
            # Verify result format - empty texts get zero vectors
            assert len(result) == 3
            assert result[0] == [0.0] * 768  # Empty text
            assert result[1] == [0.0] * 768  # Whitespace only
            assert len(result[2]) == 768     # Valid text

    def test_embed_documents_error_handling(self, mock_llama_embedder):
        """Test error handling in embed_documents."""
        with patch('app.Llama') as mock_llama_class:
            mock_llama_class.return_value = mock_llama_embedder
            
            # Make create_embedding raise an exception
            mock_llama_embedder.create_embedding.side_effect = Exception("Embedding error")
            
            embedding_function = NomicEmbeddingFunction("test_model.gguf")
            
            texts = ["Test document"]
            result = embedding_function.embed_documents(texts)
            
            # Should return zero vector on error
            assert len(result) == 1
            assert result[0] == [0.0] * 768

    def test_embed_query_success(self, mock_llama_embedder):
        """Test successful embedding of a single query."""
        with patch('app.Llama') as mock_llama_class:
            mock_llama_class.return_value = mock_llama_embedder
            
            embedding_function = NomicEmbeddingFunction("test_model.gguf")
            
            query = "What is machine learning?"
            result = embedding_function.embed_query(query)
            
            # Verify create_embedding was called
            mock_llama_embedder.create_embedding.assert_called_once_with(query)
            
            # Verify result format
            assert len(result) == 768
            assert isinstance(result, list)

    def test_embed_query_empty_text(self, mock_llama_embedder):
        """Test embedding with empty query text."""
        with patch('app.Llama') as mock_llama_class:
            mock_llama_class.return_value = mock_llama_embedder
            
            embedding_function = NomicEmbeddingFunction("test_model.gguf")
            
            result = embedding_function.embed_query("")
            
            # Empty text should not call create_embedding
            mock_llama_embedder.create_embedding.assert_not_called()
            
            # Should return zero vector
            assert result == [0.0] * 768

    def test_embed_query_whitespace_only(self, mock_llama_embedder):
        """Test embedding with whitespace-only query."""
        with patch('app.Llama') as mock_llama_class:
            mock_llama_class.return_value = mock_llama_embedder
            
            embedding_function = NomicEmbeddingFunction("test_model.gguf")
            
            result = embedding_function.embed_query("   \n\t   ")
            
            # Whitespace-only text should not call create_embedding
            mock_llama_embedder.create_embedding.assert_not_called()
            
            # Should return zero vector
            assert result == [0.0] * 768

    def test_embed_query_error_handling(self, mock_llama_embedder):
        """Test error handling in embed_query."""
        with patch('app.Llama') as mock_llama_class:
            mock_llama_class.return_value = mock_llama_embedder
            
            # Make create_embedding raise an exception
            mock_llama_embedder.create_embedding.side_effect = Exception("Query embedding error")
            
            embedding_function = NomicEmbeddingFunction("test_model.gguf")
            
            result = embedding_function.embed_query("Test query")
            
            # Should return zero vector on error
            assert result == [0.0] * 768

    def test_embed_documents_mixed_content(self, mock_llama_embedder):
        """Test embedding with mixed valid and invalid content."""
        with patch('app.Llama') as mock_llama_class:
            mock_llama_class.return_value = mock_llama_embedder
            
            embedding_function = NomicEmbeddingFunction("test_model.gguf")
            
            texts = ["Valid text", "", "Another valid text", "   "]
            result = embedding_function.embed_documents(texts)
            
            # Only valid texts should call create_embedding
            assert mock_llama_embedder.create_embedding.call_count == 2
            mock_llama_embedder.create_embedding.assert_any_call("Valid text")
            mock_llama_embedder.create_embedding.assert_any_call("Another valid text")
            
            # Verify result format
            assert len(result) == 4
            assert len(result[0]) == 768  # Valid text
            assert result[1] == [0.0] * 768  # Empty text
            assert len(result[2]) == 768  # Valid text
            assert result[3] == [0.0] * 768  # Whitespace only

    def test_embed_query_strips_whitespace(self, mock_llama_embedder):
        """Test that embed_query strips whitespace from input."""
        with patch('app.Llama') as mock_llama_class:
            mock_llama_class.return_value = mock_llama_embedder
            
            embedding_function = NomicEmbeddingFunction("test_model.gguf")
            
            query = "  What is AI?  "
            embedding_function.embed_query(query)
            
            # Should call create_embedding with stripped text
            mock_llama_embedder.create_embedding.assert_called_once_with("What is AI?")

    def test_embed_documents_strips_whitespace(self, mock_llama_embedder):
        """Test that embed_documents strips whitespace from input."""
        with patch('app.Llama') as mock_llama_class:
            mock_llama_class.return_value = mock_llama_embedder
            
            embedding_function = NomicEmbeddingFunction("test_model.gguf")
            
            texts = ["  First text  ", "Second text"]
            embedding_function.embed_documents(texts)
            
            # Should call create_embedding with stripped text
            mock_llama_embedder.create_embedding.assert_any_call("First text")
            mock_llama_embedder.create_embedding.assert_any_call("Second text")
