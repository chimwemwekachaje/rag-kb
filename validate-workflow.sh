#!/bin/bash

# Script to validate GitHub Actions workflow locally
# This script helps test the workflow components before pushing

set -e

echo "🔍 Validating GitHub Actions workflow..."

# Check if required files exist
echo "📁 Checking required files..."
if [ ! -f ".github/workflows/ci-cd.yml" ]; then
    echo "❌ Workflow file not found!"
    exit 1
fi

if [ ! -f "Dockerfile" ]; then
    echo "❌ Dockerfile not found!"
    exit 1
fi

if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found!"
    exit 1
fi

echo "✅ All required files found"

# Validate workflow syntax using act (if available)
if command -v act &> /dev/null; then
    echo "🧪 Testing workflow syntax with act..."
    if act --list > /dev/null 2>&1; then
        echo "✅ Workflow syntax is valid"
        echo "📋 Available jobs:"
        act --list
    else
        echo "❌ Workflow syntax validation failed"
        exit 1
    fi
else
    echo "⚠️  'act' not installed. Install it to test workflow locally:"
    echo "   brew install act  # macOS"
    echo "   or visit: https://github.com/nektos/act"
fi

# Test Docker build locally (optional)
if [ "$1" = "--test-docker" ]; then
    echo "🐳 Testing Docker build locally..."
    docker build -t rag-kb-test .
    echo "✅ Docker build successful"
    docker rmi rag-kb-test
fi

echo "🎉 Validation complete!"
echo ""
echo "Next steps:"
echo "1. Push your changes to trigger the workflow"
echo "2. Check the Actions tab in your GitHub repository"
echo "3. Monitor the workflow execution"
