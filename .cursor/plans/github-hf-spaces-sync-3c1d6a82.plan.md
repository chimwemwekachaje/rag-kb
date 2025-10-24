<!-- 3c1d6a82-c729-4889-80f3-80e12302f3d6 acf51fca-767c-4399-9c25-d94e00917363 -->
# GitHub to HuggingFace Spaces CI/CD Integration

## Overview

Add automated deployment workflow that syncs the main branch from GitHub to HuggingFace Spaces using GitHub Actions. The workflow will leverage the existing HuggingFace embeddings fallback (sentence-transformers) instead of large GGUF models.

## Implementation Steps

### 1. Create HuggingFace Spaces Sync Workflow

Create `.github/workflows/sync-to-hf-spaces.yml` with:

- Trigger on push to `main` branch only
- Use `git push --force` to sync repository to HF Space
- Authentication via HF_TOKEN secret stored in GitHub repository secrets
- Target Space: `kachaje/llm-kb` at https://huggingface.co/spaces/kachaje/llm-kb

**Key workflow components:**

- Checkout code with full git history (`fetch-depth: 0`)
- Configure git with GitHub Actions bot identity
- Add HF Space as remote: `https://huggingface.co/spaces/kachaje/llm-kb`
- Force push to HF Space repository
- Use GitHub secret `HF_TOKEN` for authentication

### 2. Update README.md HF Space Metadata

Ensure the HuggingFace Space metadata at the top of README.md is correctly configured:

- Confirm `sdk: docker` (already correct)
- Confirm `app_port: 7860` (already correct)
- Update space URL references to match `kachaje/llm-kb`

### 3. Create .env.example File

Document required environment variables for HuggingFace Spaces:

- `HF_TOKEN`: Triggers use of HuggingFace sentence-transformers embeddings
- Model paths (optional, not needed for HF Spaces deployment)

### 4. Update CI-CD-README.md

Add new section documenting:

- HuggingFace Spaces deployment workflow
- How to set up `HF_TOKEN` secret in GitHub repository settings
- Automatic vs manual deployment options
- Model handling differences (GGUF locally vs sentence-transformers on HF Spaces)
- Link to deployed Space

### 5. Update Dockerfile (Optional Enhancement)

Add conditional logic to handle HF Spaces environment:

- Skip copying large GGUF model files when HF_TOKEN is present
- Optimize for HuggingFace Spaces deployment

## Key Files to Modify

- `.github/workflows/sync-to-hf-spaces.yml` (new)
- `CI-CD-README.md` (update)
- `.env.example` (new)
- `README.md` (minor updates if needed)
- `Dockerfile` (optional optimization)

## Testing & Validation

After implementation, the workflow will:

1. Automatically trigger on push to main branch
2. Sync code to HuggingFace Space
3. HF Space will rebuild using Docker
4. Application will use sentence-transformers embeddings automatically (via HF_TOKEN detection)

## References

- Existing HF token detection: `app.py` lines 106-114
- Current Space: https://huggingface.co/spaces/kachaje/llm-kb
- Current CI/CD: `.github/workflows/ci-cd.yml`

### To-dos

- [ ] Create .github/workflows/sync-to-hf-spaces.yml with automatic sync to kachaje/llm-kb on push to main
- [ ] Create .env.example documenting HF_TOKEN and other environment variables
- [ ] Update CI-CD-README.md with HuggingFace Spaces deployment documentation
- [ ] Update README.md with HuggingFace Spaces deployment badge and instructions
- [ ] Optimize Dockerfile to conditionally handle GGUF models vs HF transformers