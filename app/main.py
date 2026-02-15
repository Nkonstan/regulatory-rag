"""
Pharmaceutical RAG System API.
"""

import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil

from app.models import QueryRequest, QueryResponse, IngestResponse, HealthResponse, Source
from app.config import settings
from app.logging_config import setup_logging, get_logger
from app.services import document_service
from app.retrieval.vector_store import vector_store
from app.generation.llm import llm_service

# Configure logging
setup_logging(
    log_level="INFO",
    log_file=settings.upload_dir.parent / "logs" / "rag_system.log"
)
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pharmaceutical RAG System",
    description="RAG system for pharmaceutical regulatory documents",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Auto-process PDFs on startup."""
    logger.info("=" * 70)
    logger.info("🚀 PHARMACEUTICAL RAG SYSTEM - STARTUP")
    logger.info("=" * 70)
    
    summary = document_service.process_startup_pdfs()
    
    logger.info("=" * 70)
    logger.info("📊 STARTUP SUMMARY:")
    logger.info(f"   📁 Total PDFs: {summary.get('total_files', 0)}")
    logger.info(f"   ✅ Processed: {summary.get('processed', 0)}")
    logger.info(f"   💾 Total chunks: {summary.get('database_chunks', 0)}")
    logger.info("=" * 70)


@app.post(
    "/api/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and Ingest a PDF Document",
    description="""
    Upload a new PDF document to the system.
    
    **How it works:**
    1. Upload a PDF file (regulatory document)
    2. System extracts text and detects sections
    3. Creates 512-token chunks with 50-token overlap
    4. Generates semantic embeddings using mxbai-embed-large-v1
    5. Stores chunks in ChromaDB vector database
    
    **Note:** PDFs placed in `data/uploads/` are auto-processed on startup.
    This endpoint allows you to manually upload additional documents.
    
    **Returns:** Number of chunks created and sections processed.
    """
)
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a PDF document."""
    logger.info(f"Ingest request: {file.filename}")
    
    # Validate PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )
    
    # Save file
    file_path = settings.upload_dir / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process
    result = document_service.process_pdf(file_path)
    
    if result['status'] == 'failed':
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {result.get('error')}"
        )
    
    logger.info(f"Ingested {file.filename}: {result['chunks']} chunks")
    
    return IngestResponse(
        message="Document ingested successfully",
        document_name=file.filename,
        chunks_created=result['chunks'],
        sections_processed=result['sections']
    )


@app.post(
    "/api/query",
    response_model=QueryResponse,
    summary="Query Documents with Natural Language",
    description="""
    Ask questions about the ingested regulatory documents.
    
    **How it works:**
    1. Your question is converted to an embedding (semantic vector)
    2. System searches the vector database for most relevant chunks (cosine similarity)
    3. Top-K chunks are retrieved (default: 5)
    4. LLM (Qwen 2.5 7B) generates an answer using only the retrieved context
    5. Returns answer with source citations and confidence score
    
    **Features:**
    - Semantic search (understands meaning, not just keywords)
    - GPU-accelerated inference 
    - Source attribution with relevance scores (0.0-1.0)
    - Confidence scoring based on retrieval quality
    
    **Example:** "What is a placebo-controlled trial?"
    """
)
async def query_documents(request: QueryRequest):
    """Query the document database."""
    logger.info(f"Query: '{request.question}'")
    
    # Retrieve chunks
    chunks = vector_store.query(request.question, top_k=request.top_k)
    
    if not chunks:
        return QueryResponse(
            answer="I couldn't find relevant information to answer this question.",
            sources=[],
            confidence=0.0
        )
    
    # Generate answer
    result = llm_service.generate_answer(request.question, chunks)
    logger.info(f"Answer generated (confidence: {result['confidence']:.2f})")
    
    # Format sources
    sources = [
        Source(
            document=chunk['document'],
            chunk=chunk['chunk'][:300] + "..." if len(chunk['chunk']) > 300 else chunk['chunk'],
            section=f"{chunk['section']} - {chunk['section_title']}",
            sections_all=chunk.get('metadata', {}).get('sections_all'),  # ← Add this
            relevance=chunk['relevance']
        )
        for chunk in chunks
    ]
    
    return QueryResponse(
        answer=result['answer'],
        sources=sources,
        confidence=result['confidence']
    )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check."""
    ollama_ok = llm_service.check_health()
    embedding_ok = document_service.chunker is not None
    vector_ok = vector_store.collection is not None
    
    all_ok = ollama_ok and embedding_ok and vector_ok
    
    return HealthResponse(
        status="healthy" if all_ok else "degraded",
        ollama_available=ollama_ok,
        embedding_model_loaded=embedding_ok,
        vector_store_initialized=vector_ok
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
