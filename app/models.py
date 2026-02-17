from pydantic import BaseModel, Field
from typing import List, Optional


class Source(BaseModel):
    """Source citation with metadata."""
    document: str = Field(..., description="Name of the source document")
    chunk: str = Field(..., description="Relevant text chunk")
    section: str = Field(..., description="Primary section identifier")
    sections_all: Optional[str] = Field(None, description="All sections in chunk (comma-separated)")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., min_length=1, description="User's question")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of sources to retrieve")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(..., description="Source citations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""
    message: str
    document_name: str
    chunks_created: int
    sections_processed: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    ollama_available: bool
    embedding_model_loaded: bool
    vector_store_initialized: bool
