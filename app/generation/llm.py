import httpx
from typing import List, Dict
from app.config import settings


class LLMService:
    """Interface to Ollama for generation."""
    
    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
        self.client = httpx.AsyncClient(
            timeout=60.0,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    async def generate_answer(
        self, 
        question: str, 
        context_chunks: List[Dict]
    ) -> Dict[str, any]:
        """
        Generate answer using retrieved context.
        
        Args:
            question: User's question
            context_chunks: Retrieved chunks from vector store
        
        Returns:
            Dict with 'answer' and 'confidence'
        """
        # Build context from retrieved chunks
        context = self._build_context(context_chunks)
        
        # Create prompt
        prompt = self._create_prompt(question, context)
        
        # Call Ollama
        try:
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,  # Low temperature for factual responses
                        "top_p": 1,
                        "num_predict": 1024
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result.get('response', '').strip()
            
            # Calculate confidence based on context relevance
            confidence = self._calculate_confidence(context_chunks)
            
            return {
                'answer': answer,
                'confidence': confidence
            }
            
        except httpx.HTTPError as e:
            print(f"✗ Ollama request failed: {e}")
            # Fallback to extractive answer
            return self._extractive_answer(question, context_chunks)
    
    def _build_context(self, chunks: List[Dict]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            section_info = f"[Section {chunk['section']}: {chunk.get('section_title', 'N/A')}]"
            context_parts.append(f"{section_info}\n{chunk['chunk']}\n")
        
        return "\n".join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for LLM."""
        prompt = f"""You are a pharmaceutical regulatory expert assistant. Answer based ONLY on the provided context from FDA/ICH regulatory documents.

    Context (ranked by relevance):
    {context}

    Question: {question}

    Instructions:
    1. Chunks are ordered by relevance - prioritize information from TOP chunks
    2. Read the question carefully to understand what is specifically being asked, then verify which chunk directly answers that specific question
    3. Answer directly using ONLY information from the context
    4. Cite the specific section (e.g., "According to Section III.A.1...")
    5. If context is insufficient, say "Based on the provided documents, I cannot fully answer this question"

    Answer:"""
        return prompt

    
    def _calculate_confidence(self, chunks: List[Dict]) -> float:
        """
        Calculate confidence score based on retrieval quality.
        
        Args:
            chunks: Retrieved chunks with relevance scores
        
        Returns:
            Confidence score between 0 and 1
        """
        if not chunks:
            return 0.0
        
        # Average relevance of top chunks
        relevances = [chunk['relevance'] for chunk in chunks]
        avg_relevance = sum(relevances) / len(relevances)
        
        # Boost if top result is very relevant
        top_relevance = relevances[0] if relevances else 0
        
        # Confidence formula
        confidence = (avg_relevance * 0.6) + (top_relevance * 0.4)
        
        return round(min(confidence, 1.0), 2)
    
    def _extractive_answer(self, question: str, chunks: List[Dict]) -> Dict[str, any]:
        """
        Fallback extractive answer if LLM fails.
        
        Returns most relevant chunk as answer.
        """
        if not chunks:
            return {
                'answer': "I couldn't find relevant information to answer this question.",
                'confidence': 0.0
            }
        
        # Use the most relevant chunk
        best_chunk = chunks[0]
        
        answer = f"Based on {best_chunk['section_title']} (Section {best_chunk['section']}): {best_chunk['chunk'][:300]}..."
        
        return {
            'answer': answer,
            'confidence': best_chunk['relevance'] * 0.5  # Lower confidence for extractive
        }
    
    async def check_health(self) -> bool:
        """Check if Ollama is available."""
        try:
            response =  await self.client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False
            
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

# Global instance
llm_service = LLMService()
