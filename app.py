import argparse
import os
import shutil
import signal
import sys
import threading
import time
from typing import List, Dict, Any, Tuple
from llama_cpp import Llama
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
import gradio as gr
from gradio_pdf import PDF
from langchain_huggingface import HuggingFaceEmbeddings


class NomicEmbeddingFunction:
    """Custom embedding function using nomic-embed-text.gguf"""

    def __init__(self, model_path: str, batch_size: int = 16):
        self.model_path = model_path
        self.batch_size = batch_size
        self._lock = threading.Lock()
        # Don't initialize the embedder here - create it per-thread

    def _get_embedder(self) -> Llama:
        """Get a thread-local embedder instance"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Embedding model not found at {self.model_path}")
        
        # Create a new embedder instance for each thread to avoid threading issues
        return Llama(
            model_path=self.model_path,
            embedding=True,
            n_ctx=512,  # Smaller context for embeddings
            n_threads=1,  # Single thread per instance
            n_batch=128,  # Reduced batch size to prevent GGML errors
            n_gpu_layers=0,  # Explicitly use CPU
            use_mmap=True,  # Memory-efficient model loading
            use_mlock=False,  # Don't lock memory pages
            embedding_mode=True,
            verbose=False,
            logits_all=False,  # Reduce warnings for embedding models
        )

    def _embed_single(self, text: str) -> List[float]:
        """Embed a single text with error handling"""
        try:
            if not text.strip():
                return [0.0] * 768  # Assuming 768-dimensional embeddings

            # Create a new embedder instance for this thread
            embedder = self._get_embedder()
            
            # Add validation for the model
            if not hasattr(embedder, 'create_embedding'):
                raise AttributeError("Model does not support embedding")
                
            response = embedder.create_embedding(text.strip())
            
            # Clean up the embedder instance
            if hasattr(embedder, 'close'):
                embedder.close()
            
            return response["data"][0]["embedding"]
        except Exception as e:
            print(f"Error embedding text: {e}")
            print(f"Text length: {len(text)}")
            return [0.0] * 768

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents sequentially to avoid threading issues"""
        if not texts:
            return []
        
        embeddings = []
        
        # Process sequentially to avoid threading issues with the model
        for text in texts:
            embedding = self._embed_single(text)
            embeddings.append(embedding)
        
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        return self._embed_single(text)


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


class RAGSystem:
    def __init__(
        self,
        embedding_model_path: str,
        llm_model_path: str,
        chroma_persist_dir: str = "./chroma",
        data_path: str = "./data",
    ):
        self.embedding_model_path = embedding_model_path
        self.llm_model_path = llm_model_path
        self.chroma_persist_dir = chroma_persist_dir
        self.data_path = data_path

        # Initialize embedding function
        self.embedding_function = get_embedding_function()

        # Initialize LLM
        self.llm = Llama(
            model_path=llm_model_path, n_ctx=2048, n_threads=4, verbose=False
        )

        # Initialize ChromaDB
        self.vectorstore = None
        self._setup_vectorstore()

    def _setup_vectorstore(self):
        """Setup or load existing vectorstore"""
        if os.path.exists(self.chroma_persist_dir):
            # Load existing vectorstore
            self.vectorstore = Chroma(
                persist_directory=self.chroma_persist_dir,
                embedding_function=self.embedding_function,
            )
            print(f"Loaded existing vectorstore from {self.chroma_persist_dir}")
        else:
            # Create new vectorstore
            self.vectorstore = Chroma(
                persist_directory=self.chroma_persist_dir,
                embedding_function=self.embedding_function,
            )
            print(f"Created new vectorstore at {self.chroma_persist_dir}")

    def calculate_chunk_ids(self, chunks):
        """
        This will create IDs like "data/monopoly.pdf:6:2"
        Page Source : Page Number : Chunk Index
        """
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id

        return chunks

    def load_documents(self):
        """Load documents from data directory"""
        document_loader = PyPDFDirectoryLoader(self.data_path)
        return document_loader.load()

    def split_documents(self, documents: List[Document]):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=80,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def add_documents(self, documents: List[Document]):
        """Add documents to the vectorstore"""
        chunks = self.split_documents(documents)
        chunks_with_ids = self.calculate_chunk_ids(chunks)
        
        # Only add documents that don't exist in the DB
        existing_items = self.vectorstore.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            
            # Process in smaller batches to avoid memory issues
            batch_size = 50  # Process 50 chunks at a time
            for i in range(0, len(new_chunks), batch_size):
                batch = new_chunks[i:i + batch_size]
                batch_ids = [chunk.metadata["id"] for chunk in batch]
                
                try:
                    print(f"Processing batch {i//batch_size + 1}/{(len(new_chunks) + batch_size - 1)//batch_size} ({len(batch)} chunks)")
                    self.vectorstore.add_documents(batch, ids=batch_ids)
                except Exception as e:
                    print(f"Error processing batch {i//batch_size + 1}: {e}")
                    # Continue with next batch
                    continue
            
            # Only persist if the vectorstore supports it (e.g., not in-memory)
            if hasattr(self.vectorstore, 'persist'):
                self.vectorstore.persist()
        else:
            print("✅ No new documents to add")

    def populate_database(self):
        """Populate database with documents from data directory"""
        if not os.path.exists(self.data_path):
            print(f"Data directory {self.data_path} does not exist")
            return
            
        documents = self.load_documents()
        if not documents:
            print("No documents found in data directory")
            return
            
        self.add_documents(documents)

    def retrieve_documents(self, query: str, k: int = 5) -> List[Document]:
        """Retrieve relevant documents for a query"""
        docs = self.vectorstore.similarity_search_with_score(query, k=k)
        return [doc for doc, _score in docs]

    def generate_response(self, query: str, context: str) -> str:
        """Generate response using the LLM with context"""
        prompt = f"""<|system|>
You are a helpful assistant. Answer the question based on the provided context. If the context doesn't contain enough information to answer the question, say so.

Context:
{context}

<|user|>
{query}

<|assistant|>
"""

        response = self.llm(
            prompt, max_tokens=512, temperature=0.7, stop=["<|user|>", "<|system|>"]
        )

        return response["choices"][0]["text"].strip()

    def query(self, question: str, k: int = 5) -> Dict[str, Any]:
        """Main RAG query function"""
        start_time = time.time()

        # Retrieve relevant documents
        docs = self.retrieve_documents(question, k=k)

        # Format context
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # Generate response
        response = self.generate_response(question, context)

        end_time = time.time()

        return {
            "question": question,
            "answer": response,
            "context_docs": docs,
            "time_taken": end_time - start_time,
        }

    def clear_database(self):
        """Clear the vectorstore database"""
        if os.path.exists(self.chroma_persist_dir):
            shutil.rmtree(self.chroma_persist_dir)
            print(f"Cleared database at {self.chroma_persist_dir}")


class RAGSystemContainer:
    """Thread-safe container for RAG system reference"""
    def __init__(self):
        self._rag = None
        self._lock = threading.Lock()
    
    def set(self, rag_system: RAGSystem):
        """Set the RAG system instance"""
        with self._lock:
            self._rag = rag_system
    
    def get(self) -> RAGSystem:
        """Get the RAG system instance"""
        with self._lock:
            return self._rag
    
    def is_ready(self) -> bool:
        """Check if RAG system is initialized"""
        with self._lock:
            return self._rag is not None


def initialize_rag_system(embedding_model_path: str, llm_model_path: str, shutdown_event: threading.Event, rag_ready_event: threading.Event, rag_container: RAGSystemContainer) -> RAGSystem:
    """Initialize RAG system in a separate thread"""
    try:
        print("Initializing RAG system...")
        rag = RAGSystem(
            embedding_model_path=embedding_model_path,
            llm_model_path=llm_model_path
        )
        
        if not shutdown_event.is_set():
            print("Populating database...")
            rag.populate_database()
            print("RAG system initialization completed")
            rag_container.set(rag)  # Store RAG system in container
            rag_ready_event.set()  # Signal that RAG is ready
        else:
            print("RAG system initialization cancelled due to shutdown signal")
            
        return rag
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return None


def build_nested_accordions(data_path: str) -> Tuple[Dict, List]:
    """Recursively scan data/ and build nested accordion structure"""
    pdf_files = []
    folder_structure = {}
    
    def scan_directory(path: str, relative_path: str = "") -> Dict:
        items = {}
        if not os.path.exists(path):
            return items
            
        try:
            for item in sorted(os.listdir(path)):
                if item.startswith('.'):
                    continue
                    
                item_path = os.path.join(path, item)
                item_relative = os.path.join(relative_path, item) if relative_path else item
                
                if os.path.isfile(item_path) and item.endswith('.pdf'):
                    pdf_files.append((item, item_path))
                    items[item] = item_path
                elif os.path.isdir(item_path):
                    sub_items = scan_directory(item_path, item_relative)
                    if sub_items:
                        items[item] = sub_items
        except (FileNotFoundError, PermissionError):
            # Handle cases where directory access fails
            pass
                    
        return items
    
    folder_structure = scan_directory(data_path)
    return folder_structure, pdf_files


def create_accordion_ui(folder_structure: Dict, pdf_files: List[Tuple[str, str]]) -> gr.Blocks:
    """Create nested accordion UI for PDF navigation"""
    
    def on_pdf_select(pdf_path):
        """Handle PDF selection"""
        if not pdf_path:
            return None, 1
        return pdf_path, 1
    
    def on_page_change(pdf_path, page_num):
        """Handle page change"""
        if not pdf_path:
            return None
        return PDF(starting_page=page_num)
    
    with gr.Blocks() as accordion_ui:
        with gr.Accordion("Course Contents", open=False):
            def create_nested_accordions(structure, level=0):
                """Recursively create nested accordions"""
                for name, content in structure.items():
                    if isinstance(content, dict):
                        # It's a folder
                        with gr.Accordion(f"📁 {name}", open=False):
                            create_nested_accordions(content, level + 1)
                    else:
                        # It's a PDF file
                        gr.Button(
                            f"📄 {name}",
                            elem_id=f"pdf_btn_{name}",
                            size="sm"
                        ).click(
                            fn=lambda path=content: on_pdf_select(path),
                            outputs=[gr.State(), gr.Number()]
                        )
            
            create_nested_accordions(folder_structure)
    
    return accordion_ui


def launch_gradio_ui(rag_container: RAGSystemContainer, folder_structure: Dict, shutdown_event: threading.Event, rag_ready_event: threading.Event) -> gr.Blocks:
    """Launch Gradio UI in a separate thread"""
    try:
        # Global variables for sources
        sources = []
        
        def rag_query(query_text: str):
            """Handle RAG query"""
            nonlocal sources
            
            # Check if RAG system is ready
            if not rag_ready_event.is_set():
                return "RAG system is still initializing, please wait..."
            
            # Check if RAG system is available
            if not rag_container.is_ready():
                return "RAG system initialization failed. Please check the logs and restart the application."
            
            rag = rag_container.get()
            result = rag.query(query_text)
            sources = [
                (
                    os.path.basename(doc.metadata.get("source", "Unknown")),
                    doc.metadata.get("id", "Unknown")
                )
                for doc in result["context_docs"]
            ]
            
            return result["answer"]
        
        def update_sources_dropdown():
            """Update sources dropdown after query"""
            nonlocal sources
            return gr.update(
                choices=sources, 
                value=(sources[0][1] if sources and len(sources) > 0 else None)
            )
        
        def load_pdf(pdf_file_name):
            """Load PDF from chunk ID"""
            parts = pdf_file_name.split(":")
            filename = parts[0]
            page = parts[1] if len(parts) > 1 else "1"
            return filename, int(page)
        
        def load_pdf_and_page(pdf_file, page_num):
            """Load PDF file and page"""
            if pdf_file is None:
                return None, None
            return pdf_file, int(page_num)
        
        def change_start_page(page_num):
            """Change PDF starting page"""
            return PDF(starting_page=page_num)
        
        def clear_inputs():
            """Clear query input and output fields"""
            return "", ""
        
        # Create Gradio interface
        with gr.Blocks(fill_width=True, title="Llama RAG Knowledge Base") as view:
            pdf_input = gr.File(label="Upload PDF", visible=False)
            page_number = gr.Number(
                value=1, label="Page Number", precision=0, visible=False
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Nested Accordions for hierarchical navigation
                    with gr.Accordion("Course Contents", open=False):
                        def create_nested_accordions(structure, level=0):
                            """Recursively create nested accordions"""
                            for name, content in structure.items():
                                if isinstance(content, dict):
                                    # It's a folder
                                    with gr.Accordion(f"📁 {name}", open=False):
                                        create_nested_accordions(content, level + 1)
                                else:
                                    # It's a PDF file
                                    gr.Button(
                                        f"📄 {name}",
                                        size="sm"
                                    ).click(
                                        fn=lambda path=content: load_pdf_and_page(path, 1),
                                        outputs=[pdf_input, page_number]
                                    )
                        
                        create_nested_accordions(folder_structure)
                    
                    query_input = gr.TextArea(
                        label="Question",
                        value="What is the key subject of these documents?",
                        lines=3
                    )
                    with gr.Row():
                        query_button = gr.Button("Search", variant="primary")
                        clear_button = gr.Button("Clear", variant="secondary")
                    query_output = gr.TextArea(label="Response", lines=8)
                    sources_input = gr.Dropdown(label="Sources", choices=[])
                
                with gr.Column(scale=2):
                    pdf_display = PDF(label="PDF Viewer", scale=1, interactive=False)
            
            # Event handlers
            pdf_display.allow_file_upload = False
            query_button.click(rag_query, inputs=query_input, outputs=query_output)
            query_input.submit(rag_query, inputs=query_input, outputs=query_output)
            clear_button.click(clear_inputs, outputs=[query_input, query_output])
            query_output.change(fn=update_sources_dropdown, outputs=sources_input)
            sources_input.change(
                load_pdf, inputs=sources_input, outputs=[pdf_input, page_number]
            )
            
            page_number.submit(
                fn=load_pdf_and_page,
                inputs=[pdf_input, page_number],
                outputs=[pdf_display],
            )
            
            pdf_input.change(
                fn=load_pdf_and_page,
                inputs=[pdf_input, page_number],
                outputs=[pdf_display, page_number],
            )
            
            page_number.change(
                fn=change_start_page, inputs=page_number, outputs=pdf_display
            )
        
        view.title = "Llama RAG Knowledge Base"
        view.description = "Tool for querying content from loaded documents using Llama.cpp and GGUF models"
        view.flagging_mode = "never"
        
        if not shutdown_event.is_set():
            print("Starting Gradio server...")
            view.launch(prevent_thread_lock=True)
        else:
            print("Gradio server launch cancelled due to shutdown signal")
            
        return view
    except Exception as e:
        print(f"Error launching Gradio UI: {e}")
        return None


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Llama RAG Knowledge Base")
    parser.add_argument("--embedding-model", help="Path to embedding GGUF model")
    parser.add_argument("--llm-model", help="Path to LLM GGUF model")
    parser.add_argument("--reset", action="store_true", help="Clear database")
    args = parser.parse_args()
    
    # Get model paths with priority: CLI args > env vars > defaults
    embedding_model_path = (
        args.embedding_model or 
        os.getenv("EMBEDDING_MODEL_PATH") or 
        "models/nomic-embed-text-v1.5.Q4_K_M.gguf"
    )
    
    llm_model_path = (
        args.llm_model or 
        os.getenv("LLM_MODEL_PATH") or 
        "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    )
    
    # Check if models exist
    if not os.path.exists(embedding_model_path):
        print(f"Error: Embedding model not found at {embedding_model_path}")
        return
    
    if not os.path.exists(llm_model_path):
        print(f"Error: LLM model not found at {llm_model_path}")
        return
    
    # Handle reset flag early
    if args.reset:
        print("✨ Clearing Database")
        rag = RAGSystem(
            embedding_model_path=embedding_model_path,
            llm_model_path=llm_model_path
        )
        rag.clear_database()
        return
    
    # Create shutdown event for coordination
    shutdown_event = threading.Event()
    rag_ready_event = threading.Event()
    
    # Global variables for threads and resources
    rag_container = RAGSystemContainer()
    gradio_view = None
    rag_thread = None
    gradio_thread = None
    
    def signal_handler(signum, frame):
        """Handle SIGTERM and SIGINT signals"""
        print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
        shutdown_event.set()
        
        # Close Gradio server if it exists
        if gradio_view is not None:
            try:
                print("Closing Gradio server...")
                gradio_view.close()
            except Exception as e:
                print(f"Error closing Gradio server: {e}")
        
        # Wait for threads to finish (with timeout)
        if rag_thread and rag_thread.is_alive():
            print("Waiting for RAG initialization to complete...")
            rag_thread.join(timeout=5)
        
        if gradio_thread and gradio_thread.is_alive():
            print("Waiting for Gradio server to shutdown...")
            gradio_thread.join(timeout=5)
        
        # Cleanup RAG system resources if needed
        if rag_container.is_ready():
            try:
                rag_system = rag_container.get()
                # Close LLM if it has cleanup methods
                if hasattr(rag_system.llm, 'close'):
                    rag_system.llm.close()
                # Note: NomicEmbeddingFunction creates embedder instances on-demand
                # and cleans them up immediately after use, so no cleanup needed here
            except Exception as e:
                print(f"Error during cleanup: {e}")
        
        print("Graceful shutdown completed")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Build folder structure for navigation
    folder_structure, pdf_files = build_nested_accordions("data")
    
    # Start RAG initialization in a separate thread
    def rag_init_wrapper():
        initialize_rag_system(embedding_model_path, llm_model_path, shutdown_event, rag_ready_event, rag_container)
    
    rag_thread = threading.Thread(target=rag_init_wrapper, daemon=True)
    rag_thread.start()
    
    # Start Gradio server in a separate thread immediately (don't wait for RAG init)
    def gradio_wrapper():
        nonlocal gradio_view
        gradio_view = launch_gradio_ui(rag_container, folder_structure, shutdown_event, rag_ready_event)
    
    gradio_thread = threading.Thread(target=gradio_wrapper, daemon=True)
    gradio_thread.start()
    
    print("Both RAG initialization and Gradio server are starting...")
    
    # Keep main thread alive and wait for shutdown signal
    try:
        while not shutdown_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        # This should be handled by the signal handler, but just in case
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
