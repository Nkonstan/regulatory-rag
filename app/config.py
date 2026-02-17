from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """Application configuration."""
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    
    # Embedding Model
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1"
    
    # Chunking Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Retrieval Configuration
    top_k: int = 3
    similarity_threshold: float = 0.45
    
    # Hybrid Retrieval Configuration (NEW)
    use_hybrid_search: bool = True  # Enable BM25 + semantic hybrid
    bm25_weight: float = 0.40  # Weight for BM25 scores (0.0-1.0)
    semantic_weight: float = 0.60  # Weight for semantic scores (0.0-1.0)
    # Note: bm25_weight + semantic_weight should = 1.0
    
    # Application Paths
    upload_dir: Path = Path("./data/uploads")
    vector_db_dir: Path = Path("./data/vectordb")
    bm25_index_path: Path = Path("./data/vectordb/bm25_index.pkl")  # BM25 persistence
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()

