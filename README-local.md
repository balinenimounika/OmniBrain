# OmniBrain - Text & Image Embeddings + Qdrant Setup

This project implements the foundational ingestion and retrieval modules (Text & Image Embeddings, and Qdrant database management) for the OmniBrain Agentic Multi-Modal RAG Orchestrator (Week 1, Task 3).

## Folder Structure

```
omnibrain/
│
├── app/
│   ├── embeddings/
│   │   ├── __init__.py          # Embeddings submodule initialization
│   │   ├── text_embeddings.py   # Text embedding generation using SentenceTransformers
│   │   └── image_embeddings.py  # Image embedding generation using CLIP
│   │
│   └── qdrant/
│       ├── __init__.py          # Qdrant submodule initialization
│       ├── client.py            # Local/Remote Qdrant client connection setup
│       ├── collections.py       # Functions to create collections
│       ├── insert.py            # Functions to insert vectors into Qdrant
│       └── search.py            # Similarity search functions
│
├── data/
│   ├── documents/               # Folder for sample PDFs / text files
│   └── images/                  # Folder for sample images
│
├── tests/
│   ├── test_text_embeddings.py  # Test script for text embeddings
│   ├── test_image_embeddings.py # Test script for image embeddings
│   └── test_qdrant.py           # Verification script for Qdrant setup, insertion, and search
│
├── .env                         # Local environment configuration
├── .gitignore                   # Excluded files (venvs, secret envs, caches)
├── requirements.txt             # Python packages lists
└── README.md                    # This description file
```

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Environment Variables**:
   Configure `.env` if using a remote Qdrant server. Otherwise, leaving it empty will run a local disk-based instance.
