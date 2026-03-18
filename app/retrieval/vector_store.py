import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np
import logging
import re

from app.config import settings
from app.retrieval.embeddings import embedding_service

logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store using ChromaDB + BM25 for hybrid retrieval."""
    STRUCTURAL_KEYWORDS = {'table', 'figure', 'flowchart', 'chart', 'diagram', 'decision tree'}
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
        # Strip punctuation, lowercase, remove stopwords
        text = text.lower()
        tokens = re.findall(r'\b[a-z][a-z0-9]*\b', text)
        stopwords = {
            'the','a','an','is','are','was','were','be','been',
            'being','have','has','had','do','does','did','will',
            'would','could','should','may','might','of','in',
            'to','for','on','at','by','with','from','that','this',
            'or','and','not','it','its','as','such','if','but'
        }
        return [t for t in tokens if t not in stopwords]
    
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
                'total_chunks': chunk['total_chunks'],
                'chunk_type': chunk.get('chunk_type', 'text')
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
                'section_count': metadata['section_count'],
                'chunk_type': chunk.get('chunk_type', 'text')
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
        top_indices = np.argsort(scores)[::-1][:top_k]
        
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

    
    def query(self, query_text: str, top_k: int = None) -> List[Dict]:
        """
        Query using hybrid BM25 + semantic search with Reciprocal Rank Fusion (RRF).

        Retrieves candidates from both BM25 and semantic search using an equal-sized
        pool, merges them via RRF, and returns the top_k results.

        For queries targeting structural artifacts (tables, figures, etc.), the
        retrieval pool is expanded by 3 extra candidates to improve recall of
        sparse table/figure chunks, which tend to rank lower in both retrievers.

        Args:
            query_text: The natural language query string.
            top_k: Number of final results to return. Defaults to settings.top_k.

        Returns:
            List of result dicts sorted by RRF relevance score (descending),
            each containing 'document', 'chunk', 'section', 'relevance', 'metadata'.
        """
        top_k = top_k or settings.top_k

        # Expand pool for structural queries (tables, figures, etc.)
        is_structural = any(kw in query_text.lower() for kw in self.STRUCTURAL_KEYWORDS)
        effective_k = top_k + 3 if is_structural else top_k

        print(f"\n🔍 DEBUG - Query: '{query_text}'")
        print(f"🔍 DEBUG - Structural query: {is_structural}, effective_k: {effective_k}")

        # Fall back to pure semantic search if hybrid is disabled or BM25 not loaded
        if not settings.use_hybrid_search or not self.bm25_index:
            print("🔍 DEBUG - Using pure semantic search")
            return self._semantic_search_only(query_text, effective_k)[:top_k]

        # === HYBRID SEARCH ===

        # Use a fixed equal pool size for both retrievers to ensure
        # RRF rank positions are comparable across both lists
        pool_size = max(effective_k * 3, 20)

        print("🔍 DEBUG - Running BM25 search...")
        bm25_results = self._bm25_search(query_text, pool_size)
        print(f"🔍 DEBUG - BM25 found {len(bm25_results)} results")

        print("🔍 DEBUG - Running semantic search...")
        query_embedding = embedding_service.embed_query(query_text)
        semantic_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=pool_size,  # same pool_size as BM25 for symmetric RRF ranking
            include=['documents', 'metadatas', 'distances']
        )
        print(f"🔍 DEBUG - Semantic found {len(semantic_results['ids'][0]) if semantic_results['ids'] else 0} results")

        combined_results = self._merge_results(
            bm25_results,
            semantic_results,
            query_text,
            is_structural=is_structural
        )

        # No threshold applied — RRF already ranks by quality.
        # Cutting at top_k is sufficient; a cosine-based threshold
        # is incompatible with the scaled RRF relevance score.
        final_results = combined_results[:top_k]

        print(f"🔍 DEBUG - Final results: {len(final_results)}\n")
        return final_results

    
    def _merge_results(
        self,
        bm25_results: List[Dict],
        semantic_results: Dict,
        query_text: str,
        is_structural: bool = False
    ) -> List[Dict]:
        """
        Merge BM25 and semantic results using Reciprocal Rank Fusion (RRF).
        
        Instead of normalizing raw scores (which is sensitive to score distribution
        and pool size), RRF works purely on rank positions. Each document gets a
        score of 1 / (K + rank) from each retriever, where K=60 is a constant that
        dampens the impact of very high ranks. These per-retriever RRF scores are
        then combined with the configured BM25/semantic weights.
        
        Example for a document ranked 1st in semantic, 5th in BM25 (K=60):
            semantic_rrf = 1 / (60 + 1) = 0.01639
            bm25_rrf     = 1 / (60 + 5) = 0.01538
            combined     = 0.4 * 0.01538 + 0.6 * 0.01639 = 0.01598

        A document missing from one retriever gets 0.0 for that retriever's
        contribution, cleanly distinguishing "absent" from "ranked last".

        Args:
            bm25_results: Ranked list of dicts from BM25 search, each with 'id', 'score', 'metadata'
            semantic_results: Raw ChromaDB query response with 'ids', 'distances', 'documents', 'metadatas'
            query_text: Original query string (unused in scoring, kept for potential future use)
        
        Returns:
            List of result dicts sorted by combined RRF score, each with
            'document', 'chunk', 'section', 'section_title', 'relevance', 'metadata'.
            Relevance is scaled to [0, 1] where 1.0 = ranked #1 in both retrievers.
        """
        # --- Step 1: Build raw score maps ---
        # bm25_scores preserves insertion order (Python 3.7+), which reflects BM25 rank
        bm25_scores = {r['id']: r['score'] for r in bm25_results}
        
        # semantic_scores similarly preserves ChromaDB's rank order (closest distance first)
        semantic_scores = {}
        semantic_data = {}  # also store the actual text + metadata for later retrieval
        
        if semantic_results['ids'] and semantic_results['ids'][0]:
            for i in range(len(semantic_results['ids'][0])):
                doc_id = semantic_results['ids'][0][i]
                
                # ChromaDB returns cosine *distance* in [0, 2], convert to similarity in [0, 1]
                # distance=0 means identical vectors, distance=2 means opposite
                distance = semantic_results['distances'][0][i]
                similarity = 1 - (distance / 2)
                
                semantic_scores[doc_id] = similarity
                semantic_data[doc_id] = {
                    'document': semantic_results['documents'][0][i],
                    'metadata': semantic_results['metadatas'][0][i]
                }
        
        # --- Step 2: Convert score-ordered dicts into rank maps ---
        # enumerate() gives 0-based positions, +1 makes them 1-indexed ranks
        # Rank 1 = best result, rank 2 = second best, etc.
        bm25_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(bm25_scores.keys())}
        semantic_ranks = {doc_id: rank + 1 for rank, doc_id in enumerate(semantic_scores.keys())}

        # --- Step 3: Compute RRF scores ---
        # K=60 is the standard constant from the original RRF paper (Cormack et al., 2009).
        # It prevents the highest-ranked document from dominating too strongly.
        K = 60

        # Theoretical maximum RRF score: a document ranked #1 in both retrievers.
        # = bm25_weight * 1/(K+1) + semantic_weight * 1/(K+1)
        # = 1/(K+1) since weights sum to 1.0
        # Used to scale final relevance scores to a human-readable [0, 1] range.
        theoretical_max = 1.0 / (K + 1)  # ≈ 0.01639

        # Union of all doc IDs seen by either retriever
        all_ids = set(bm25_ranks.keys()) | set(semantic_ranks.keys())

        combined = []
        for doc_id in all_ids:
            # If a doc was not found by a retriever, its contribution is 0.0
            # (not penalized, just absent — unlike min-max where "absent" == "worst found")
            bm25_score = 1.0 / (K + bm25_ranks[doc_id]) if doc_id in bm25_ranks else 0.0
            semantic_score = 1.0 / (K + semantic_ranks[doc_id]) if doc_id in semantic_ranks else 0.0

            # Weighted combination using configured BM25/semantic weights (default 0.4 / 0.6)
            combined_score = (
                settings.bm25_weight * bm25_score +
                settings.semantic_weight * semantic_score
            )

            # When the query explicitly targets a structural artifact, boost
            # table/figure chunks by 40% so they outrank prose chunks that
            # only mention the table by name.
            chunk_meta = semantic_data.get(doc_id, {}).get('metadata', {})
            if not chunk_meta:
                bm25_item = next((r for r in bm25_results if r['id'] == doc_id), None)
                if bm25_item:
                    chunk_meta = bm25_item['metadata']
            if is_structural and chunk_meta.get('chunk_type') in ('table', 'figure'):
                combined_score *= 1.4

            scaled_relevance = round(min(combined_score / theoretical_max, 1.0), 3)

            
            # --- Step 4: Resolve document text and metadata ---
            # Prefer semantic_data since it comes directly from ChromaDB with full metadata.
            # Fall back to BM25 metadata for docs that only appeared in keyword search.
            if doc_id in semantic_data:
                doc_text = semantic_data[doc_id]['document']
                metadata = semantic_data[doc_id]['metadata']
            else:
                bm25_item = next((r for r in bm25_results if r['id'] == doc_id), None)
                if bm25_item:
                    doc_text = bm25_item['metadata']['text']
                    metadata = bm25_item['metadata']
                else:
                    continue
            
            print(f"🔍 DEBUG - {doc_id[:30]}... BM25_rank={bm25_ranks.get(doc_id, 'N/A')}, Semantic_rank={semantic_ranks.get(doc_id, 'N/A')}, Relevance={scaled_relevance:.3f}")
            
            combined.append({
                'document': metadata.get('document', 'unknown'),
                'chunk': doc_text,
                'section': metadata.get('section', ''),
                'section_title': metadata.get('section_title', ''),
                'relevance': scaled_relevance,  # [0, 1] scaled
                'metadata': metadata
            })
        
        # --- Step 5: Sort by combined RRF score descending ---
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