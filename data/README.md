# Data Directory

This directory contains the PDF documents that will be indexed and made searchable by the RAG system.

## Directory Structure

Organize your PDF files in a hierarchical structure for better navigation:

```
data/
├── Course Summary.pdf
├── Week 1/
│   ├── Week 1 Day 1.pdf
│   ├── Week 1 Day 2.pdf
│   ├── updated slides from week 1 refresh/
│   │   ├── Beautiful.ai - Week 1 Day 2.pdf
│   │   └── Beautiful.ai - Week 1 Day 3.pdf
│   └── ...
├── Week 2/
│   ├── Week 2 Day 1.pdf
│   └── ...
└── ...
```

## Supported Formats

- **PDF files** (`.pdf`) - Primary format
- The system will automatically scan all subdirectories recursively

## Adding Documents

1. Place your PDF files in this directory
2. Organize them in subdirectories as needed
3. The application will automatically detect and index new documents on startup
4. Use the hierarchical navigation in the UI to browse your documents

## Document Processing

- Documents are automatically split into chunks (800 characters with 80 character overlap)
- Each chunk gets a unique ID for precise source tracking
- The system only processes new documents (avoids duplicates)

## Tips

- Use descriptive filenames for better organization
- Group related documents in subdirectories
- The UI will show a nested accordion structure matching your directory layout
- Large documents are automatically chunked for better retrieval performance
