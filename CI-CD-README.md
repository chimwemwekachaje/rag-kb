# CI/CD Pipeline Documentation

This repository includes a comprehensive GitHub Actions workflow for continuous integration and deployment.

## Workflow Overview

The CI/CD pipeline (`ci-cd.yml`) performs the following actions:

### 1. Testing Phase
- **Triggers**: Push to `main`/`develop` branches, Pull Requests to `main`
- **Matrix Strategy**: Tests on Python 3.12
- **Steps**:
  - Checkout code
  - Set up Python environment
  - Cache pip dependencies for faster builds
  - Install system dependencies (build tools, OpenBLAS)
  - Install Python dependencies
  - Run tests with coverage reporting
  - Upload coverage reports to Codecov

### 2. Build and Push Phase
- **Dependencies**: Only runs after tests pass
- **Steps**:
  - Checkout code
  - Set up Docker Buildx for multi-platform builds
  - Login to GitHub Container Registry (ghcr.io)
  - Extract metadata and generate tags
  - Build and push Docker image for `linux/amd64` and `linux/arm64`
  - Generate artifact attestation for security

## Docker Configuration

### Dockerfile Features
- **Base Image**: Python 3.12 slim
- **Optimizations**: 
  - Multi-stage build with dependency caching
  - OpenBLAS integration for llama-cpp-python performance
  - Non-root user for security
- **Port**: Exposes port 7860 for Gradio interface
- **Volumes**: ChromaDB data persistence

### .dockerignore
Excludes unnecessary files from Docker build context:
- Git files and history
- Python cache and virtual environments
- Test files and documentation
- IDE and OS-specific files
- Build artifacts

## Container Registry

Images are pushed to GitHub Container Registry with the following naming:
- **Registry**: `ghcr.io`
- **Image Name**: `{owner}/{repository}`
- **Tags**:
  - `latest` (for main branch)
  - Branch name (for feature branches)
  - Semantic version tags (when using releases)
  - PR numbers (for pull requests)

## Usage

### Automatic Triggers
The workflow automatically runs on:
- Push to `main` or `develop` branches
- Pull requests targeting `main` branch

### Manual Validation
Use the provided validation script:
```bash
# Basic validation
./validate-workflow.sh

# Include Docker build test
./validate-workflow.sh --test-docker
```

### Local Testing with Act
Install `act` to test workflows locally:
```bash
# macOS
brew install act

# Test the workflow
act --dry-run
```

## Security Features

1. **Artifact Attestation**: Build provenance is generated and attached to images
2. **Minimal Permissions**: Workflow only requests necessary permissions
3. **Dependency Caching**: Reduces build time and external dependencies
4. **Multi-platform Builds**: Ensures compatibility across architectures

## HuggingFace Spaces Deployment

### Automatic Deployment

The repository includes an automated deployment workflow (`sync-to-hf-spaces.yml`) that syncs the main branch to HuggingFace Spaces on every push.

**Target Space**: [kachaje/llm-kb](https://huggingface.co/spaces/kachaje/llm-kb)

### Setup Requirements

1. **HuggingFace Token**: Add `HF_TOKEN` as a repository secret in GitHub Settings
   - Go to Repository Settings → Secrets and variables → Actions
   - Add new repository secret named `HF_TOKEN`
   - Value should be your HuggingFace access token with write permissions

2. **Space Configuration**: The Space is configured to use Docker SDK with the following metadata in README.md:
   ```yaml
   sdk: docker
   app_port: 7860
   ```

### Deployment Process

1. **Trigger**: Automatically runs on push to `main` branch
2. **Sync**: Uses `git push --force` to sync repository to HF Space
3. **Build**: HF Space automatically rebuilds using Docker
4. **Runtime**: Application detects `HF_TOKEN` and uses HuggingFace sentence-transformers embeddings

### Model Handling

- **Local Development**: Uses GGUF models (nomic-embed-text, tinyllama) for offline operation
- **HuggingFace Spaces**: Automatically switches to sentence-transformers/all-MiniLM-L6-v2 when `HF_TOKEN` is detected
- **Benefits**: Faster startup, smaller image size, no large model downloads

### Manual Deployment

To manually trigger deployment:
```bash
# Push to main branch
git push origin main

# Or use GitHub CLI to trigger workflow
gh workflow run sync-to-hf-spaces.yml
```

## Monitoring

- **GitHub Actions**: Check the Actions tab in your repository
- **Coverage Reports**: Available in Codecov integration
- **Container Registry**: View images in the Packages section of your repository
- **HuggingFace Spaces**: Monitor deployment at [kachaje/llm-kb](https://huggingface.co/spaces/kachaje/llm-kb)

## Troubleshooting

### Common Issues

1. **Build Failures**: Check system dependencies in the test phase
2. **Docker Build Issues**: Verify Dockerfile and .dockerignore configuration
3. **Permission Errors**: Ensure GitHub token has package write permissions
4. **Test Failures**: Review test output and coverage reports

### Debug Commands
```bash
# Test Docker build locally
docker build -t rag-kb-test .

# Run tests locally
pytest --cov=. --cov-report=html

# Check workflow syntax
act --dry-run
```

## Customization

### Adding More Python Versions
Update the matrix strategy in the workflow:
```yaml
strategy:
  matrix:
    python-version: [3.11, 3.12]
```

### Adding More Test Steps
Add additional test commands in the test job:
```yaml
- name: Run linting
  run: |
    pip install flake8 black
    flake8 .
    black --check .
```

### Custom Tags
Modify the metadata extraction step to add custom tags:
```yaml
tags: |
  type=ref,event=branch
  type=raw,value=custom-tag
```
