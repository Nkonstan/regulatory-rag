from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import logging
import time

from app.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Generate embeddings using mixedbread-ai model."""
    
    def __init__(self):
        logger.info("Initializing EmbeddingService...")
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            logger.info("=" * 70)
            logger.info(f"⏳ Loading embedding model: {settings.embedding_model}")
            logger.info("   This may take 15-20 seconds on first run...")
            logger.info("=" * 70)
            
            start_time = time.time()
            self.model = SentenceTransformer(settings.embedding_model)
            elapsed = time.time() - start_time
            
            logger.info("=" * 70)
            logger.info(f"✅ Embedding model loaded successfully ({elapsed:.1f}s)")
            logger.info(f"   Model: {settings.embedding_model}")
            logger.info(f"   Dimension: {self.model.get_sentence_embedding_dimension()}")
            logger.info("=" * 70)
            
        except Exception as e:
            logger.error(f"✗ Failed to load embedding model: {e}")
            raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
        
        Returns:
            List of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        logger.info(f"⏳ Generating embeddings for {len(texts)} text(s)...")
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            show_progress_bar=len(texts) > 10,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Query string
        
        Returns:
            Embedding vector
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        logger.debug(f"🔍 Generating query embedding: {query[:50]}...")
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.tolist()
    
    def get_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if not self.model:
            return 1024  # Default for mxbai-embed-large-v1
        return self.model.get_sentence_embedding_dimension()


# Global instance
embedding_service = EmbeddingService()
