# NikosKonstantinou-pfizer-ai-engineer-assignment

A **Retrieval-Augmented Generation (RAG)** system for querying pharmaceutical regulatory documents using natural language. Built with FastAPI, ChromaDB, and Ollama for local, GPU-accelerated document Q&A.

## Features

- **Hybrid Retrieval**: BM25 keyword search (40%) + semantic embeddings (60%)
- **Section-Aware Parsing**: Detects hierarchical document sections (Roman numerals, letters, decimal notation)
- **GPU Acceleration**: Fast embeddings and LLM inference
- **Auto-Processing**: PDFs in `data/uploads/` are indexed automatically on startup
- **Source Attribution**: Answers include citations with relevance scores and confidence

## Tech Stack

- **API**: FastAPI 0.109.0
- **LLM**: Ollama + Qwen 2.5 7B (local inference)
- **Embeddings**: mixedbread-ai/mxbai-embed-large-v1 (1024-dim)
- **Vector DB**: ChromaDB 0.4.22 (cosine similarity)
- **Keyword Search**: BM25Okapi
- **PDF Processing**: PyMuPDF + pdfplumber
- **Tokenization**: tiktoken (512-token chunks, 50-token overlap)

## Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with Docker GPU support
- 8GB+ RAM
- 10GB+ disk space

**Tested on:** RTX 2070 (8GB VRAM) - uses ~6GB during operation

## Quick Start

1. **Clone and prepare data**
   ```bash
   git clone <repository-url>
   cd NikosKonstantinou-pfizer-ai-engineer-assignment
   
   # Place your PDFs for auto-indexing
   mkdir -p data/uploads
   cp your-pdfs/*.pdf data/uploads/
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   ```

3. **Build and Start the system**
   ```bash
   docker compose up -d --build
   ```

   First startup downloads:
   - Qwen 2.5 7B model (~4.7GB, takes 5-10 min)
   - mxbai-embed-large-v1 (~1.2GB)
   - Then auto-processes PDFs in `data/uploads/`

4. **Check health**
   ```bash
   curl http://localhost:8000/api/health
   ```

## API Usage

### Upload PDF (manual)
```curl -X POST http://localhost:8000/api/ingest \
  -F "file=@document.pdf" \
  -v
```

Response:
```json
{
  "message": "Document ingested successfully",
  "document_name": "document.pdf",
  "chunks_created": 87,
  "sections_processed": 23
}
```

### Query Documents
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a placebo-controlled trial?"
  }'
```

Response:
```json
{
  "answer": "According to Section III.A.2...",
  "sources": [
    {
      "document": "ich_guideline.pdf",
      "chunk": "A placebo is a pharmaceutical...",
      "section": "III.A.2",
      "sections_all": "III,A,2",
      "relevance": 0.89
    }
  ],
  "confidence": 0.85
}
```


## Environment Variables

```bash
# LLM Configuration
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=qwen2.5:7b

# Embedding Model
EMBEDDING_MODEL=mixedbread-ai/mxbai-embed-large-v1

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Retrieval
TOP_K=3
SIMILARITY_THRESHOLD=0.45

# Hybrid Search (BM25 + Semantic)
USE_HYBRID_SEARCH=true
BM25_WEIGHT=0.4
SEMANTIC_WEIGHT=0.6
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DOCKER SERVICES                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────┐              ┌───────────────────────────────┐    │
│  │   Ollama Service     │              │      RAG API Service          │    │
│  │   (GPU-Accelerated)  │◄─────────────┤      (FastAPI)                │    │
│  │                      │              │                               │    │
│  │  • Qwen 2.5 7B LLM   │              │  POST /api/ingest             │    │
│  │  • Port 11434        │              │  POST /api/query              │    │
│  └──────────────────────┘              │  GET  /api/health             │    │
│                                        └───────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          CORE FUNCTIONALITIES                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐    │
│  │   INGESTION     │      │    RETRIEVAL     │      │   GENERATION     │    │
│  ├─────────────────┤      ├──────────────────┤      ├──────────────────┤    │
│  │                 │      │                  │      │                  │    │
│  │ • PDF Parsing   │      │  Hybrid Search:  │      │ • Context Build  │    │
│  │ • Section       │──────►                  │──────► • LLM Prompting  │    │
│  │   Detection     │      │  ┌────────────┐  │      │ • Answer Gen.    │    │
│  │ • Chunking      │      │  │ BM25 (40%) │  │      │ • Confidence     │    │
│  │   (512 tokens)  │      │  └─────┬──────┘  │      │   Scoring        │    │
│  │ • Embedding     │      │        │         │      │                  │    │
│  │   (1024-dim)    │      │  ┌─────▼──────┐  │      │                  │    │
│  │                 │      │  │  Fusion    │  │      │                  │    │ 
│  │ → ChromaDB      │      │  └─────┬──────┘  │      │                  │    │
│  │ → BM25 Index    │      │        │         │      │                  │    │
│  │                 │      │  ┌─────▼──────┐  │      │                  │    │
│  └─────────────────┘      │  │Semantic(60%)│ │      └──────────────────┘    │
│                           │  └────────────┘  │                              │
│                           │                  │                              │
│                           │  → Ranked Chunks │                              │
│                           └──────────────────┘                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
NikosKonstantinou-pfizer-ai-engineer-assignment/
├── docker-compose.yml          # Multi-service orchestration
├── Dockerfile                  # API container
├── requirements.txt            # Python dependencies
├── .env                        # Environment config
│
├── app/
│   ├── main.py                # FastAPI entry point
│   ├── config.py              # Settings
│   ├── models.py              # Pydantic models
│   ├── logging_config.py      # Logging setup
│   │
│   ├── ingestion/
│   │   ├── pdf_parser.py     # Section detection
│   │   └── chunker.py        # Token-based chunking
│   │
│   ├── retrieval/
│   │   ├── vector_store.py   # ChromaDB + BM25
│   │   └── embeddings.py     # Sentence transformers
│   │
│   ├── generation/
│   │   └── llm.py            # Ollama integration
│   │
│   └── services/
│       └── document_service.py
│
└── data/
    ├── uploads/               # PDFs (auto-processed on startup)
    ├── vectordb/              # ChromaDB + BM25 persistence
    └── logs/
```

## Docker Services

The system runs 3 containers:

1. **ollama** - LLM engine (port 11434, uses GPU)
2. **ollama-pull-model** - Auto-downloads Qwen 2.5 7B on first run
3. **rag-api** - FastAPI application (port 8000)

## Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f rag-api
docker compose logs -f ollama
```

## Troubleshooting

**Ollama not starting:**
```bash
# Check GPU
nvidia-smi

# Verify Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

**Model not found:**
```bash
docker compose exec ollama ollama pull qwen2.5:7b
```

**Reset database:**
```bash
docker compose down
rm -rf data/vectordb/*
rm data/uploads/.processed_files.json
docker compose up -d
```

## Performance

**RTX 2070 (8GB VRAM):**
- GPU usage: ~6GB during operation
