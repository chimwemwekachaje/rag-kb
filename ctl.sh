#!/bin/bash

# Llama RAG Knowledge Base Control Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default model paths
DEFAULT_EMBEDDING_MODEL="models/nomic-embed-text-v1.5.Q4_K_M.gguf"
DEFAULT_LLM_MODEL="models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

add_dependency() {
    local dependency="$1"
    if [ -z "$dependency" ]; then
        print_error "Dependency not provided"
        return 1
    fi

    echo "Adding dependency: $dependency"

    uv add $dependency
    return 0
}

setup_venv() {
    uv venv
    uv init
    return 0
}

install_dependencies() {
    uv add -r requirements.txt
    return 0
}

# Function to check if models exist
check_models() {
    local embedding_model="$1"
    local llm_model="$2"
    
    if [ ! -f "$embedding_model" ]; then
        print_error "Embedding model not found: $embedding_model"
        print_status "Please download the model to the models/ directory"
        return 1
    fi
    
    if [ ! -f "$llm_model" ]; then
        print_error "LLM model not found: $llm_model"
        print_status "Please download the model to the models/ directory"
        return 1
    fi
    
    return 0
}

# Function to start the application
start_app() {
    local embedding_model="${1:-$DEFAULT_EMBEDDING_MODEL}"
    local llm_model="${2:-$DEFAULT_LLM_MODEL}"
    
    print_status "Starting Llama RAG Knowledge Base..."
    print_status "Embedding model: $embedding_model"
    print_status "LLM model: $llm_model"
    
    if check_models "$embedding_model" "$llm_model"; then
        print_status "Models found, starting application..."
        python app.py --embedding-model "$embedding_model" --llm-model "$llm_model"
    else
        print_error "Failed to start application due to missing models"
        exit 1
    fi
}

# Function to reset the database
reset_db() {
    print_warning "This will clear the vector database. Are you sure? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_status "Clearing database..."
        python app.py --reset
        print_success "Database cleared"
    else
        print_status "Database reset cancelled"
    fi
}

# Function to show help
show_help() {
    echo "Llama RAG Knowledge Base Control Script"
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  setup                                Setup the virtual environment"
    echo "  install                              Install dependencies"
    echo "  add [dependency]                     Add a dependency"
    echo "  start [embedding_model] [llm_model]  Start the application"
    echo "  reset                                Reset the vector database"
    echo "  help                                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 start models/my-embedding.gguf models/my-llm.gguf"
    echo "  $0 reset"
    echo "  $0 install"
    echo "  $0 add [dependency]"
    echo "  $0 setup"
    echo ""
    echo "Environment Variables:"
    echo "  EMBEDDING_MODEL_PATH                 Path to embedding model"
    echo "  LLM_MODEL_PATH                       Path to LLM model"
}

# Main script logic
case "${1:-start}" in
    start)
        embedding_model="${EMBEDDING_MODEL_PATH:-$DEFAULT_EMBEDDING_MODEL}"
        llm_model="${LLM_MODEL_PATH:-$DEFAULT_LLM_MODEL}"
        
        # Override with command line arguments if provided
        if [ -n "$2" ]; then
            embedding_model="$2"
        fi
        if [ -n "$3" ]; then
            llm_model="$3"
        fi
        
        start_app "$embedding_model" "$llm_model"
        ;;
    reset)
        reset_db
        ;;
    install)
        install_dependencies
        ;;
    add)
        add_dependency "$2"
        ;;
    setup)
        setup_venv
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
