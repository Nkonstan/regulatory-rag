import requests
from typing import List, Dict
from app.config import settings


class LLMService:
    """Interface to Ollama for generation."""
    
    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model
    
    def generate_answer(
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
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # Low temperature for factual responses
                        "top_p": 0.9,
                        "num_predict": 512
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
            
        except requests.exceptions.RequestException as e:
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
        prompt = f"""You are a pharmaceutical regulatory expert assistant. Answer the question based ONLY on the provided context from FDA/ICH regulatory documents.

Context:
{context}

Question: {question}

Instructions:
1. Answer directly and concisely
2. Use only information from the context
3. If the context doesn't contain enough information, say "Based on the provided documents, I cannot fully answer this question."
4. Cite specific sections when relevant (e.g., "According to Section 2.1.3...")
5. Be precise about regulatory requirements

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
    
    def check_health(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


# Global instance
llm_service = LLMService()
