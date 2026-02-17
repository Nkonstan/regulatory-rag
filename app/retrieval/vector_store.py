import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np
import logging

from app.config import settings
from app.retrieval.embeddings import embedding_service

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store using ChromaDB + BM25 for hybrid retrieval."""
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.bm25_index = None
        self.bm25_corpus = []  # Store tokenized documents for BM25
        self.bm25_metadata = []  # Store metadata parallel to corpus
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and collection."""
        try:
            # Create ChromaDB client
            self.client = chromadb.PersistentClient(
                path=str(settings.vector_db_dir),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("✅ ChromaDB client created")
            # Get or create collection with COSINE similarity
            self.collection = self.client.get_or_create_collection(
                name="regulatory_documents",
                metadata={
                    "description": "Pharmaceutical regulatory documents",
                    "hnsw:space": "cosine"  # Use cosine similarity!
                }
            )
            logger.info("✅ Collection initialized")
            # Load BM25 index if exists
            self._load_bm25_index()
            doc_count = self.collection.count()
            logger.info(f"✅ Vector store ready ({doc_count} chunks in database)")
            if self.bm25_index:
                logger.info(f"✅ BM25 hybrid search enabled ({len(self.bm25_corpus)} documents)")
        except Exception as e:
            logger.error(f"✗ Failed to initialize vector store: {e}")
            raise
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on whitespace/punctuation
        tokens = text.lower().split()
        return tokens
    
    def _save_bm25_index(self):
        """Save BM25 index to disk."""
        try:
            bm25_data = {
                'corpus': self.bm25_corpus,
                'metadata': self.bm25_metadata,
                'index': self.bm25_index
            }
            with open(settings.bm25_index_path, 'wb') as f:
                pickle.dump(bm25_data, f)
            print(f"✓ BM25 index saved ({len(self.bm25_corpus)} documents)")
        except Exception as e:
            print(f"⚠️ Failed to save BM25 index: {e}")
    
    def _load_bm25_index(self):
        """Load BM25 index from disk."""
        if not settings.bm25_index_path.exists():
            logger.info("ℹ️  No existing BM25 index found (will use semantic search only)")
            return
        
        try:
            logger.info("⏳ Loading BM25 index...")
            with open(settings.bm25_index_path, 'rb') as f:
                bm25_data = pickle.load(f)
            
            self.bm25_corpus = bm25_data['corpus']
            self.bm25_metadata = bm25_data['metadata']
            self.bm25_index = bm25_data['index']
            logger.info(f"✅ BM25 index loaded ({len(self.bm25_corpus)} documents)")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load BM25 index: {e}")
            self.bm25_corpus = []
            self.bm25_metadata = []
            self.bm25_index = None
    
    def add_chunks(
        self, 
        chunks: List[Dict], 
        document_name: str,
        embeddings: List[List[float]] = None
    ):
        """
        Add document chunks to vector store AND BM25 index.
        """
        if not chunks:
            return
        
        # Generate embeddings if not provided
        if embeddings is None:
            texts = [chunk['text'] for chunk in chunks]
            embeddings = embedding_service.embed_documents(texts)
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_name}_sec{chunk['section']}_chunk{chunk['chunk_index']}_{i}"
            
            ids.append(chunk_id)
            documents.append(chunk['text'])
            
            # Build metadata with section lists
            metadata = {
                'document': document_name,
                'section': chunk['section'],  # Primary section
                'section_title': chunk['section_title'],  # Primary title
                'page_start': chunk.get('page_start', 0),
                'page_end': chunk.get('page_end', 0),
                'chunk_index': chunk['chunk_index'],
                'total_chunks': chunk['total_chunks']
            }
            
            # Add section lists if they exist
            if 'sections_list' in chunk:
                # Join list into comma-separated string (ChromaDB metadata must be primitives)
                metadata['sections_all'] = ','.join(chunk['sections_list'])
                metadata['section_titles_all'] = ','.join(chunk['section_titles_list'])
                metadata['section_count'] = len(chunk['sections_list'])
            else:
                metadata['sections_all'] = chunk['section']
                metadata['section_titles_all'] = chunk['section_title']
                metadata['section_count'] = 1
            
            metadatas.append(metadata)
            
            # Add to BM25 corpus
            tokenized = self._tokenize(chunk['text'])
            self.bm25_corpus.append(tokenized)
            self.bm25_metadata.append({
                'id': chunk_id,
                'document': document_name,
                'text': chunk['text'],
                'section': chunk['section'],
                'section_title': chunk['section_title'],
                'page_start': chunk.get('page_start', 0),
                'page_end': chunk.get('page_end', 0),
                'chunk_index': chunk['chunk_index'],
                'total_chunks': chunk['total_chunks'],
                'sections_all': metadata['sections_all'],
                'section_titles_all': metadata['section_titles_all'],
                'section_count': metadata['section_count']
            })
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        # Rebuild BM25 index
        if self.bm25_corpus:
            self.bm25_index = BM25Okapi(self.bm25_corpus)
            self._save_bm25_index()
        
        print(f"✓ Added {len(chunks)} chunks from {document_name}")
    
    def _bm25_search(self, query_text: str, top_k: int) -> List[Dict]:
        """
        Perform BM25 keyword search.
        
        Returns:
            List of dicts with 'id', 'score', and 'metadata'
        """
        if not self.bm25_index or not self.bm25_corpus:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query_text)
        
        # Get BM25 scores
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k * 2]  # Get 2x for merging
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append({
                    'id': self.bm25_metadata[idx]['id'],
                    'score': float(scores[idx]),
                    'metadata': self.bm25_metadata[idx]
                })
        
        return results
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def query(
        self, 
        query_text: str, 
        top_k: int = None
    ) -> List[Dict]:
        """
        Query using hybrid BM25 + semantic search.
        
        Args:
            query_text: Query string
            top_k: Number of results to return
        
        Returns:
            List of results with metadata
        """
        top_k = top_k or settings.top_k
        
        # DEBUG: Print query info
        print(f"\n🔍 DEBUG - Query: '{query_text}'")
        print(f"🔍 DEBUG - Requested top_k: {top_k}")
        print(f"🔍 DEBUG - Similarity threshold: {settings.similarity_threshold}")
        print(f"🔍 DEBUG - Hybrid search: {settings.use_hybrid_search}")
        
        if not settings.use_hybrid_search or not self.bm25_index:
            # Fallback to pure semantic search
            print("🔍 DEBUG - Using pure semantic search")
            return self._semantic_search_only(query_text, top_k)
        
        # === HYBRID SEARCH ===
        
        # 1. BM25 Search
        print("🔍 DEBUG - Running BM25 search...")
        bm25_results = self._bm25_search(query_text, top_k)
        print(f"🔍 DEBUG - BM25 found {len(bm25_results)} results")
        
        # 2. Semantic Search
        print("🔍 DEBUG - Running semantic search...")
        query_embedding = embedding_service.embed_query(query_text)
        
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # Get more for merging
            include=['documents', 'metadatas', 'distances']
        )
        
        print(f"🔍 DEBUG - Semantic found {len(semantic_results['ids'][0]) if semantic_results['ids'] else 0} results")
        
        # 3. Merge and Re-rank
        combined_results = self._merge_results(
            bm25_results, 
            semantic_results,
            query_text
        )
        
        # 4. Filter by similarity threshold and limit to top_k
        filtered_results = [
            r for r in combined_results 
            if r['relevance'] >= settings.similarity_threshold
        ][:top_k]
        
        print(f"🔍 DEBUG - Final results after filtering: {len(filtered_results)}\n")
        
        return filtered_results
    
    def _merge_results(
        self, 
        bm25_results: List[Dict],
        semantic_results: Dict,
        query_text: str
    ) -> List[Dict]:
        """
        Merge BM25 and semantic results with weighted scoring.
        
        Returns:
            Sorted list of results by combined score
        """
        # Build score maps
        bm25_scores = {r['id']: r['score'] for r in bm25_results}
        
        semantic_scores = {}
        semantic_data = {}
        
        if semantic_results['ids'] and semantic_results['ids'][0]:
            for i in range(len(semantic_results['ids'][0])):
                doc_id = semantic_results['ids'][0][i]
                
                # Convert cosine distance to similarity
                distance = semantic_results['distances'][0][i]
                similarity = 1 - (distance / 2)
                
                semantic_scores[doc_id] = similarity
                semantic_data[doc_id] = {
                    'document': semantic_results['documents'][0][i],
                    'metadata': semantic_results['metadatas'][0][i]
                }
        
        # Normalize scores separately
        bm25_score_values = list(bm25_scores.values())
        semantic_score_values = list(semantic_scores.values())
        
        bm25_normalized = dict(zip(
            bm25_scores.keys(),
            self._normalize_scores(bm25_score_values)
        )) if bm25_score_values else {}
        
        semantic_normalized = dict(zip(
            semantic_scores.keys(),
            self._normalize_scores(semantic_score_values)
        )) if semantic_score_values else {}
        
        # Combine all unique document IDs
        all_ids = set(bm25_scores.keys()) | set(semantic_scores.keys())
        
        # Calculate weighted combined scores
        combined = []
        for doc_id in all_ids:
            bm25_score = bm25_normalized.get(doc_id, 0.0)
            semantic_score = semantic_normalized.get(doc_id, 0.0)
            
            # Weighted combination
            combined_score = (
                settings.bm25_weight * bm25_score +
                settings.semantic_weight * semantic_score
            )
            
            # Get document data (prefer semantic as it has ChromaDB metadata)
            if doc_id in semantic_data:
                doc_text = semantic_data[doc_id]['document']
                metadata = semantic_data[doc_id]['metadata']
            else:
                # Fallback to BM25 metadata
                bm25_item = next((r for r in bm25_results if r['id'] == doc_id), None)
                if bm25_item:
                    doc_text = bm25_item['metadata']['text']
                    metadata = bm25_item['metadata']
                else:
                    continue
            
            # Debug output
            print(f"🔍 DEBUG - {doc_id[:30]}... BM25={bm25_score:.3f}, Semantic={semantic_score:.3f}, Combined={combined_score:.3f}")
            
            combined.append({
                'document': metadata.get('document', 'unknown'),
                'chunk': doc_text,
                'section': metadata.get('section', ''),
                'section_title': metadata.get('section_title', ''),
                'relevance': round(combined_score, 3),
                'metadata': metadata  # Include full metadata
            })
        
        # Sort by combined score
        combined.sort(key=lambda x: x['relevance'], reverse=True)
        
        return combined
    
    def _semantic_search_only(self, query_text: str, top_k: int) -> List[Dict]:
        """
        Fallback to pure semantic search (original behavior).
        """
        # Generate query embedding
        query_embedding = embedding_service.embed_query(query_text)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        
        if results['ids'] and results['ids'][0]:
            print(f"🔍 DEBUG - Raw results found: {len(results['ids'][0])}")
            
            for i in range(len(results['ids'][0])):
                # Convert cosine distance to similarity
                distance = results['distances'][0][i]
                similarity = 1 - (distance / 2)
                
                print(f"🔍 DEBUG - Result {i+1}: distance={distance:.4f}, similarity={similarity:.4f}")
                
                # Skip results below threshold
                if similarity < settings.similarity_threshold:
                    print(f"   ⚠️ FILTERED OUT (below threshold {settings.similarity_threshold})")
                    continue
                
                formatted_results.append({
                    'document': results['metadatas'][0][i]['document'],
                    'chunk': results['documents'][0][i],
                    'section': results['metadatas'][0][i]['section'],
                    'section_title': results['metadatas'][0][i]['section_title'],
                    'relevance': round(similarity, 3),
                    'metadata': results['metadatas'][0][i]  # Include full metadata
                })
        else:
            print(f"🔍 DEBUG - No results returned from ChromaDB!")
        
        return formatted_results
    
    def delete_document(self, document_name: str):
        """Delete all chunks from a specific document."""
        # Query for all chunks from this document
        results = self.collection.get(
            where={"document": document_name},
            include=['metadatas']
        )
        
        if results['ids']:
            # Delete from ChromaDB
            self.collection.delete(ids=results['ids'])
            
            # Delete from BM25 index
            self.bm25_corpus = [
                corpus for corpus, meta in zip(self.bm25_corpus, self.bm25_metadata)
                if meta['document'] != document_name
            ]
            self.bm25_metadata = [
                meta for meta in self.bm25_metadata
                if meta['document'] != document_name
            ]
            
            # Rebuild BM25 index
            if self.bm25_corpus:
                self.bm25_index = BM25Okapi(self.bm25_corpus)
                self._save_bm25_index()
            else:
                self.bm25_index = None
            
            print(f"✓ Deleted {len(results['ids'])} chunks from {document_name}")
    
    def count(self) -> int:
        """Get total number of chunks in store."""
        return self.collection.count()
    
    def reset(self):
        """Clear all data from vector store."""
        self.client.delete_collection("regulatory_documents")
        self.collection = self.client.create_collection(
            name="regulatory_documents",
            metadata={
                "description": "Pharmaceutical regulatory documents",
                "hnsw:space": "cosine"
            }
        )
        
        # Reset BM25
        self.bm25_corpus = []
        self.bm25_metadata = []
        self.bm25_index = None
        
        # Delete BM25 index file
        if settings.bm25_index_path.exists():
            settings.bm25_index_path.unlink()
        
        print("✓ Vector store reset")


# Global instance
vector_store = VectorStore()