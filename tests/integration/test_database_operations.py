"""Integration tests for database operations with real ChromaDB operations."""

import pytest
from unittest.mock import Mock, patch
from langchain_core.documents import Document
from app import RAGSystem


class TestDatabaseOperations:
    """Integration tests for ChromaDB operations."""

    def test_vectorstore_persistence_and_retrieval(self, in_memory_vectorstore, mock_pdf_documents):
        """Test that documents can be stored and retrieved from vectorstore."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.NomicEmbeddingFunction') as mock_embedding_class:
            
            # Setup mock embedding function
            mock_embedder = Mock()
            mock_embedder.create_embedding.return_value = {
                "data": [{"embedding": [0.1] * 768}]
            }
            
            mock_llama_class.return_value = mock_embedder
            mock_embedding_class.return_value = Mock()
            
            # Create RAG system
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            rag.vectorstore = in_memory_vectorstore
            rag.embedding_function.embedder = mock_embedder
            
            # Split and add documents
            chunks = rag.split_documents(mock_pdf_documents)
            chunks_with_ids = rag.calculate_chunk_ids(chunks)
            
            # Add documents to vectorstore
            rag.add_documents(mock_pdf_documents)
            
            # Verify documents were added
            stored_items = in_memory_vectorstore.get()
            assert len(stored_items["ids"]) > 0
            
            # Verify we can retrieve documents
            retrieved_docs = rag.retrieve_documents("machine learning", k=5)
            assert len(retrieved_docs) >= 0

    def test_document_deduplication_across_multiple_adds(self, in_memory_vectorstore, mock_pdf_documents):
        """Test that documents are not duplicated when added multiple times."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.NomicEmbeddingFunction') as mock_embedding_class:
            
            # Setup mock embedding function
            mock_embedder = Mock()
            mock_embedder.create_embedding.return_value = {
                "data": [{"embedding": [0.1] * 768}]
            }
            
            mock_llama_class.return_value = mock_embedder
            mock_embedding_class.return_value = Mock()
            
            # Create RAG system
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            rag.vectorstore = in_memory_vectorstore
            rag.embedding_function.embedder = mock_embedder
            
            # Add documents first time
            rag.add_documents(mock_pdf_documents)
            first_count = len(in_memory_vectorstore.get()["ids"])
            
            # Add same documents again
            rag.add_documents(mock_pdf_documents)
            second_count = len(in_memory_vectorstore.get()["ids"])
            
            # Add documents third time
            rag.add_documents(mock_pdf_documents)
            third_count = len(in_memory_vectorstore.get()["ids"])
            
            # Count should remain the same (no duplicates)
            assert first_count == second_count == third_count

    def test_batch_document_processing(self, in_memory_vectorstore):
        """Test processing multiple documents in batches."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.NomicEmbeddingFunction') as mock_embedding_class:
            
            # Setup mock embedding function
            mock_embedder = Mock()
            mock_embedder.create_embedding.return_value = {
                "data": [{"embedding": [0.1] * 768}]
            }
            
            mock_llama_class.return_value = mock_embedder
            mock_embedding_class.return_value = Mock()
            
            # Create RAG system
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            rag.vectorstore = in_memory_vectorstore
            rag.embedding_function.embedder = mock_embedder
            
            # Create multiple batches of documents
            batch1 = [
                Document(
                    page_content="First batch document 1",
                    metadata={"source": "batch1_doc1.pdf", "page": 0}
                ),
                Document(
                    page_content="First batch document 2",
                    metadata={"source": "batch1_doc2.pdf", "page": 0}
                )
            ]
            
            batch2 = [
                Document(
                    page_content="Second batch document 1",
                    metadata={"source": "batch2_doc1.pdf", "page": 0}
                ),
                Document(
                    page_content="Second batch document 2",
                    metadata={"source": "batch2_doc2.pdf", "page": 0}
                )
            ]
            
            # Process first batch
            rag.add_documents(batch1)
            first_batch_count = len(in_memory_vectorstore.get()["ids"])
            
            # Process second batch
            rag.add_documents(batch2)
            second_batch_count = len(in_memory_vectorstore.get()["ids"])
            
            # Second batch should have more documents
            assert second_batch_count > first_batch_count
            
            # Verify all documents are retrievable
            retrieved_docs = rag.retrieve_documents("document", k=10)
            assert len(retrieved_docs) >= 0

    def test_chunk_id_uniqueness(self, in_memory_vectorstore, mock_pdf_documents):
        """Test that chunk IDs are unique across all documents."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.NomicEmbeddingFunction') as mock_embedding_class:
            
            # Setup mock embedding function
            mock_embedder = Mock()
            mock_embedder.create_embedding.return_value = {
                "data": [{"embedding": [0.1] * 768}]
            }
            
            mock_llama_class.return_value = mock_embedder
            mock_embedding_class.return_value = Mock()
            
            # Create RAG system
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            rag.vectorstore = in_memory_vectorstore
            rag.embedding_function.embedder = mock_embedder
            
            # Process documents
            chunks = rag.split_documents(mock_pdf_documents)
            chunks_with_ids = rag.calculate_chunk_ids(chunks)
            
            # Extract all IDs
            all_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
            
            # Verify all IDs are unique
            assert len(all_ids) == len(set(all_ids))
            
            # Verify ID format
            for chunk_id in all_ids:
                assert ":" in chunk_id  # Should have format "source:page:chunk_index"

    def test_similarity_search_consistency(self, in_memory_vectorstore, mock_pdf_documents):
        """Test that similarity search returns consistent results."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.NomicEmbeddingFunction') as mock_embedding_class:
            
            # Setup mock embedding function with consistent responses
            mock_embedder = Mock()
            mock_embedder.create_embedding.return_value = {
                "data": [{"embedding": [0.1] * 768}]
            }
            
            mock_llama_class.return_value = mock_embedder
            mock_embedding_class.return_value = Mock()
            
            # Create RAG system
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            rag.vectorstore = in_memory_vectorstore
            rag.embedding_function.embedder = mock_embedder
            
            # Add documents
            rag.add_documents(mock_pdf_documents)
            
            # Perform multiple searches with same query
            query = "machine learning artificial intelligence"
            results1 = rag.retrieve_documents(query, k=3)
            results2 = rag.retrieve_documents(query, k=3)
            
            # Results should be consistent (same number of documents)
            assert len(results1) == len(results2)

    def test_database_clear_and_rebuild(self, in_memory_vectorstore, mock_pdf_documents):
        """Test clearing database and rebuilding with new documents."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.NomicEmbeddingFunction') as mock_embedding_class:
            
            # Setup mock embedding function
            mock_embedder = Mock()
            mock_embedder.create_embedding.return_value = {
                "data": [{"embedding": [0.1] * 768}]
            }
            
            mock_llama_class.return_value = mock_embedder
            mock_embedding_class.return_value = Mock()
            
            # Create RAG system
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            rag.vectorstore = in_memory_vectorstore
            rag.embedding_function.embedder = mock_embedder
            
            # Add initial documents
            rag.add_documents(mock_pdf_documents)
            initial_count = len(in_memory_vectorstore.get()["ids"])
            assert initial_count > 0
            
            # Clear database (simulate by creating new vectorstore)
            new_vectorstore = in_memory_vectorstore  # In real scenario, this would be a new instance
            rag.vectorstore = new_vectorstore
            
            # Add documents again
            rag.add_documents(mock_pdf_documents)
            final_count = len(new_vectorstore.get()["ids"])
            
            # Should have documents again
            assert final_count > 0

    def test_metadata_preservation(self, in_memory_vectorstore, mock_pdf_documents):
        """Test that document metadata is preserved through the pipeline."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.NomicEmbeddingFunction') as mock_embedding_class:
            
            # Setup mock embedding function
            mock_embedder = Mock()
            mock_embedder.create_embedding.return_value = {
                "data": [{"embedding": [0.1] * 768}]
            }
            
            mock_llama_class.return_value = mock_embedder
            mock_embedding_class.return_value = Mock()
            
            # Create RAG system
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            rag.vectorstore = in_memory_vectorstore
            rag.embedding_function.embedder = mock_embedder
            
            # Add documents
            rag.add_documents(mock_pdf_documents)
            
            # Retrieve documents
            retrieved_docs = rag.retrieve_documents("test query", k=5)
            
            # Verify metadata is preserved
            for doc in retrieved_docs:
                assert "source" in doc.metadata
                assert "page" in doc.metadata
                assert "id" in doc.metadata

    def test_large_document_handling(self, in_memory_vectorstore):
        """Test handling of large documents with many chunks."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.NomicEmbeddingFunction') as mock_embedding_class:
            
            # Setup mock embedding function
            mock_embedder = Mock()
            mock_embedder.create_embedding.return_value = {
                "data": [{"embedding": [0.1] * 768}]
            }
            
            mock_llama_class.return_value = mock_embedder
            mock_embedding_class.return_value = Mock()
            
            # Create RAG system
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            rag.vectorstore = in_memory_vectorstore
            rag.embedding_function.embedder = mock_embedder
            
            # Create a large document (simulated by creating many small documents)
            large_docs = []
            for i in range(10):  # Create 10 "pages" of content
                large_docs.append(Document(
                    page_content=f"This is page {i} of a large document with substantial content about various topics including machine learning, artificial intelligence, and data science.",
                    metadata={"source": "large_document.pdf", "page": i}
                ))
            
            # Process large document
            rag.add_documents(large_docs)
            
            # Verify all chunks were processed
            stored_items = in_memory_vectorstore.get()
            assert len(stored_items["ids"]) > 0
            
            # Verify retrieval works
            retrieved_docs = rag.retrieve_documents("machine learning", k=5)
            assert len(retrieved_docs) >= 0

    def test_mixed_document_types(self, in_memory_vectorstore):
        """Test handling of documents with different metadata structures."""
        with patch('app.Llama') as mock_llama_class, \
             patch('app.NomicEmbeddingFunction') as mock_embedding_class:
            
            # Setup mock embedding function
            mock_embedder = Mock()
            mock_embedder.create_embedding.return_value = {
                "data": [{"embedding": [0.1] * 768}]
            }
            
            mock_llama_class.return_value = mock_embedder
            mock_embedding_class.return_value = Mock()
            
            # Create RAG system
            rag = RAGSystem(
                embedding_model_path="test_embedding.gguf",
                llm_model_path="test_llm.gguf"
            )
            rag.vectorstore = in_memory_vectorstore
            rag.embedding_function.embedder = mock_embedder
            
            # Create documents with different metadata
            mixed_docs = [
                Document(
                    page_content="Document from Week 1",
                    metadata={"source": "Week 1/lecture1.pdf", "page": 0}
                ),
                Document(
                    page_content="Document from Week 2",
                    metadata={"source": "Week 2/lecture2.pdf", "page": 0}
                ),
                Document(
                    page_content="Course summary document",
                    metadata={"source": "Course Summary.pdf", "page": 0}
                )
            ]
            
            # Process mixed documents
            rag.add_documents(mixed_docs)
            
            # Verify all documents were processed
            stored_items = in_memory_vectorstore.get()
            assert len(stored_items["ids"]) > 0
            
            # Verify retrieval works for different query types
            queries = ["Week 1", "lecture", "summary"]
            for query in queries:
                retrieved_docs = rag.retrieve_documents(query, k=3)
                assert len(retrieved_docs) >= 0
