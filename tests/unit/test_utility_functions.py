"""Unit tests for utility functions."""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from app import build_nested_accordions, create_accordion_ui, main


class TestBuildNestedAccordions:
    """Test cases for build_nested_accordions function."""

    def test_build_nested_accordions_empty_directory(self, temp_data_dir):
        """Test with empty directory."""
        empty_dir = tempfile.mkdtemp()
        try:
            folder_structure, pdf_files = build_nested_accordions(empty_dir)
            assert folder_structure == {}
            assert pdf_files == []
        finally:
            shutil.rmtree(empty_dir)

    def test_build_nested_accordions_nonexistent_directory(self):
        """Test with non-existent directory."""
        folder_structure, pdf_files = build_nested_accordions("/nonexistent/path")
        assert folder_structure == {}
        assert pdf_files == []

    def test_build_nested_accordions_with_pdfs(self, temp_data_dir):
        """Test with PDF files in directory structure."""
        folder_structure, pdf_files = build_nested_accordions(temp_data_dir)
        
        # Verify folder structure
        assert "Week 1" in folder_structure
        assert "Week 2" in folder_structure
        assert "Course Summary.pdf" in folder_structure
        
        # Verify nested structure
        assert "Week 1 Day 1.pdf" in folder_structure["Week 1"]
        assert "Week 1 Day 2.pdf" in folder_structure["Week 1"]
        assert "Week 2 Day 1.pdf" in folder_structure["Week 2"]
        assert "Week 2 Day 2.pdf" in folder_structure["Week 2"]
        
        # Verify PDF files list
        assert len(pdf_files) == 5  # 5 PDF files total
        pdf_names = [name for name, _ in pdf_files]
        assert "Course Summary.pdf" in pdf_names
        assert "Week 1 Day 1.pdf" in pdf_names
        assert "Week 1 Day 2.pdf" in pdf_names
        assert "Week 2 Day 1.pdf" in pdf_names
        assert "Week 2 Day 2.pdf" in pdf_names

    def test_build_nested_accordions_ignores_dot_files(self, temp_data_dir):
        """Test that dot files are ignored."""
        # Create a dot file
        dot_file = os.path.join(temp_data_dir, ".hidden_file.pdf")
        with open(dot_file, 'w') as f:
            f.write("hidden content")
        
        folder_structure, pdf_files = build_nested_accordions(temp_data_dir)
        
        # Verify dot file is not included
        pdf_names = [name for name, _ in pdf_files]
        assert ".hidden_file.pdf" not in pdf_names

    def test_build_nested_accordions_ignores_non_pdf_files(self, temp_data_dir):
        """Test that non-PDF files are ignored."""
        # Create a non-PDF file
        txt_file = os.path.join(temp_data_dir, "readme.txt")
        with open(txt_file, 'w') as f:
            f.write("text content")
        
        folder_structure, pdf_files = build_nested_accordions(temp_data_dir)
        
        # Verify non-PDF file is not included
        assert "readme.txt" not in folder_structure
        pdf_names = [name for name, _ in pdf_files]
        assert "readme.txt" not in pdf_names

    def test_build_nested_accordions_empty_subdirectories(self, temp_data_dir):
        """Test with empty subdirectories."""
        # Create empty subdirectory
        empty_subdir = os.path.join(temp_data_dir, "Empty Week")
        os.makedirs(empty_subdir, exist_ok=True)
        
        folder_structure, pdf_files = build_nested_accordions(temp_data_dir)
        
        # Empty subdirectory should not appear in structure
        assert "Empty Week" not in folder_structure

    def test_build_nested_accordions_sorted_output(self, temp_data_dir):
        """Test that output is sorted."""
        folder_structure, pdf_files = build_nested_accordions(temp_data_dir)
        
        # Verify folder structure keys are sorted
        folder_keys = list(folder_structure.keys())
        assert folder_keys == sorted(folder_keys)
        
        # Verify PDF files are sorted
        pdf_names = [name for name, _ in pdf_files]
        assert pdf_names == sorted(pdf_names)


class TestCreateAccordionUI:
    """Test cases for create_accordion_ui function."""

    def test_create_accordion_ui_basic_structure(self, sample_folder_structure, mock_gradio_components):
        """Test basic accordion UI creation."""
        mock_gr, mock_pdf = mock_gradio_components
        pdf_files = [("Course Summary.pdf", "data/Course Summary.pdf")]
        
        # Patch the gr module in the app module
        with patch('app.gr', mock_gr):
            result = create_accordion_ui(sample_folder_structure, pdf_files)
        
        # Verify gr.Blocks was called
        mock_gr.Blocks.assert_called_once()
        
        # Verify gr.Accordion was called
        mock_gr.Accordion.assert_called()

    def test_create_accordion_ui_with_nested_structure(self, sample_folder_structure, mock_gradio_components):
        """Test accordion UI with nested folder structure."""
        mock_gr, mock_pdf = mock_gradio_components
        pdf_files = [
            ("Week 1 Day 1.pdf", "data/Week 1/Week 1 Day 1.pdf"),
            ("Week 2 Day 1.pdf", "data/Week 2/Week 2 Day 1.pdf")
        ]
        
        # Patch the gr module in the app module
        with patch('app.gr', mock_gr):
            result = create_accordion_ui(sample_folder_structure, pdf_files)
        
        # Verify multiple accordions were created (nested structure)
        assert mock_gr.Accordion.call_count >= 2  # At least main accordion and sub-accordions

    def test_create_accordion_ui_empty_structure(self, mock_gradio_components):
        """Test accordion UI with empty structure."""
        mock_gr, mock_pdf = mock_gradio_components
        empty_structure = {}
        empty_pdf_files = []
        
        # Patch the gr module in the app module
        with patch('app.gr', mock_gr):
            result = create_accordion_ui(empty_structure, empty_pdf_files)
        
        # Should still create the main structure
        mock_gr.Blocks.assert_called_once()
        mock_gr.Accordion.assert_called()


class TestMainFunction:
    """Test cases for main function."""

    def test_main_with_cli_arguments(self, mock_embedding_function, mock_llama_llm, mock_vectorstore, mock_gradio_components):
        """Test main function with CLI arguments."""
        mock_gr, mock_pdf = mock_gradio_components
        
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.NomicEmbeddingFunction') as mock_embedding_class, \
             patch('app.RAGSystem') as mock_rag_class, \
             patch('app.build_nested_accordions') as mock_build_accordions, \
             patch('app.gr', mock_gr), \
             patch('os.path.exists') as mock_exists, \
             patch('sys.argv', ['app.py', '--embedding-model', 'test_embedding.gguf', '--llm-model', 'test_llm.gguf']):
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_embedding_class.return_value = mock_embedding_function
            mock_rag_instance = Mock()
            mock_rag_class.return_value = mock_rag_instance
            mock_build_accordions.return_value = ({}, [])
            mock_exists.return_value = True
            
            # Mock the launch method to prevent actual server startup
            mock_gr.Blocks.return_value.launch = Mock()
            
            main()
            
            # Verify RAGSystem was initialized with CLI arguments
            mock_rag_class.assert_called_once()
            call_args = mock_rag_class.call_args
            assert call_args[1]['embedding_model_path'] == 'test_embedding.gguf'
            assert call_args[1]['llm_model_path'] == 'test_llm.gguf'

    def test_main_with_environment_variables(self, mock_embedding_function, mock_llama_llm, mock_vectorstore, mock_gradio_components):
        """Test main function with environment variables."""
        mock_gr, mock_pdf = mock_gradio_components
        
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.NomicEmbeddingFunction') as mock_embedding_class, \
             patch('app.RAGSystem') as mock_rag_class, \
             patch('app.build_nested_accordions') as mock_build_accordions, \
             patch('app.gr', mock_gr), \
             patch('os.path.exists') as mock_exists, \
             patch('os.getenv') as mock_getenv, \
             patch('sys.argv', ['app.py']):
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_embedding_class.return_value = mock_embedding_function
            mock_rag_instance = Mock()
            mock_rag_class.return_value = mock_rag_instance
            mock_build_accordions.return_value = ({}, [])
            mock_exists.return_value = True
            mock_getenv.side_effect = lambda key: {
                'EMBEDDING_MODEL_PATH': 'env_embedding.gguf',
                'LLM_MODEL_PATH': 'env_llm.gguf'
            }.get(key)
            
            # Mock the launch method to prevent actual server startup
            mock_gr.Blocks.return_value.launch = Mock()
            
            main()
            
            # Verify RAGSystem was initialized with environment variables
            mock_rag_class.assert_called_once()
            call_args = mock_rag_class.call_args
            assert call_args[1]['embedding_model_path'] == 'env_embedding.gguf'
            assert call_args[1]['llm_model_path'] == 'env_llm.gguf'

    def test_main_with_default_paths(self, mock_embedding_function, mock_llama_llm, mock_vectorstore, mock_gradio_components):
        """Test main function with default model paths."""
        mock_gr, mock_pdf = mock_gradio_components
        
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.NomicEmbeddingFunction') as mock_embedding_class, \
             patch('app.RAGSystem') as mock_rag_class, \
             patch('app.build_nested_accordions') as mock_build_accordions, \
             patch('app.gr', mock_gr), \
             patch('os.path.exists') as mock_exists, \
             patch('os.getenv') as mock_getenv, \
             patch('sys.argv', ['app.py']):
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_embedding_class.return_value = mock_embedding_function
            mock_rag_instance = Mock()
            mock_rag_class.return_value = mock_rag_instance
            mock_build_accordions.return_value = ({}, [])
            mock_exists.return_value = True
            mock_getenv.return_value = None  # No environment variables
            
            # Mock the launch method to prevent actual server startup
            mock_gr.Blocks.return_value.launch = Mock()
            
            main()
            
            # Verify RAGSystem was initialized with default paths
            mock_rag_class.assert_called_once()
            call_args = mock_rag_class.call_args
            assert call_args[1]['embedding_model_path'] == 'models/nomic-embed-text-v1.5.Q4_K_M.gguf'
            assert call_args[1]['llm_model_path'] == 'models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'

    def test_main_missing_embedding_model(self):
        """Test main function with missing embedding model."""
        with patch('os.path.exists') as mock_exists, \
             patch('sys.argv', ['app.py']):
            
            # Mock embedding model doesn't exist
            def mock_exists_side_effect(path):
                if 'embed' in path:
                    return False
                return True
            
            mock_exists.side_effect = mock_exists_side_effect
            
            # Should return early without initializing RAG system
            main()

    def test_main_missing_llm_model(self):
        """Test main function with missing LLM model."""
        with patch('os.path.exists') as mock_exists, \
             patch('sys.argv', ['app.py']):
            
            # Mock LLM model doesn't exist
            def mock_exists_side_effect(path):
                if 'llm' in path or 'tinyllama' in path:
                    return False
                return True
            
            mock_exists.side_effect = mock_exists_side_effect
            
            # Should return early without initializing RAG system
            main()

    def test_main_with_reset_flag(self, mock_embedding_function, mock_llama_llm, mock_vectorstore, mock_gradio_components):
        """Test main function with reset flag."""
        mock_gr, mock_pdf = mock_gradio_components
        
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.NomicEmbeddingFunction') as mock_embedding_class, \
             patch('app.RAGSystem') as mock_rag_class, \
             patch('app.gr', mock_gr), \
             patch('os.path.exists') as mock_exists, \
             patch('sys.argv', ['app.py', '--reset']):
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_embedding_class.return_value = mock_embedding_function
            mock_rag_instance = Mock()
            mock_rag_class.return_value = mock_rag_instance
            mock_exists.return_value = True
            
            # Mock the launch method to prevent actual server startup
            mock_gr.Blocks.return_value.launch = Mock()
            
            main()
            
            # Verify RAGSystem was initialized and clear_database was called
            mock_rag_class.assert_called_once()
            mock_rag_instance.clear_database.assert_called_once()

    def test_main_populate_database_called(self, mock_embedding_function, mock_llama_llm, mock_vectorstore, mock_gradio_components):
        """Test that populate_database is called in main."""
        mock_gr, mock_pdf = mock_gradio_components
        
        with patch('app.Llama') as mock_llama_class, \
             patch('app.Chroma') as mock_chroma_class, \
             patch('app.NomicEmbeddingFunction') as mock_embedding_class, \
             patch('app.RAGSystem') as mock_rag_class, \
             patch('app.build_nested_accordions') as mock_build_accordions, \
             patch('app.gr', mock_gr), \
             patch('os.path.exists') as mock_exists, \
             patch('sys.argv', ['app.py']):
            
            mock_llama_class.return_value = mock_llama_llm
            mock_chroma_class.return_value = mock_vectorstore
            mock_embedding_class.return_value = mock_embedding_function
            mock_rag_instance = Mock()
            mock_rag_class.return_value = mock_rag_instance
            mock_build_accordions.return_value = ({}, [])
            mock_exists.return_value = True
            
            # Mock the launch method to prevent actual server startup
            mock_gr.Blocks.return_value.launch = Mock()
            
            main()
            
            # Verify populate_database was called
            mock_rag_instance.populate_database.assert_called_once()
