import tiktoken
from typing import List, Dict,Tuple
from app.config import settings


class SectionChunker:
    """Split document sections into chunks of ~512 tokens."""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")


    def chunk_sections(self, sections: List[Dict]) -> List[Dict]:
        """
        Chunk sections into smaller pieces with proper section tracking.
        
        Args:
            sections: List of sections from PDFParser
        
        Returns:
            List of chunks with accurate section metadata
        """
        chunks = []
        
        for section in sections:
            section_text = section['text'].strip()
            
            # Skip truly empty sections (except Document Start)
            if not section_text and section['section'] != '0':
                continue
            
            # Token count for this section
            tokens = self.encoding.encode(section_text)

            # Table chunks must never be split across token boundaries.
            # Splitting mid-row destroys the column-cell relationship entirely.
            if section.get('chunk_type') == 'table':
                chunks.append({
                    'text': section_text,
                    'section': section['section'],
                    'section_title': section['section_title'],
                    'page_start': section.get('page_start'),
                    'page_end': section.get('page_end'),
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'sections_list': [section['section']],
                    'section_titles_list': [section['section_title']],
                    'section_count': 1,
                    'chunk_type': 'table'
                })
                continue
            
            if len(tokens) <= self.chunk_size:
                chunk = {
                    'text': section_text,
                    'section': section['section'],
                    'section_title': section['section_title'],
                    'page_start': section.get('page_start'),
                    'page_end': section.get('page_end'),
                    'chunk_index': 0,
                    'total_chunks': 1,
                    'sections_list': [section['section']],
                    'section_titles_list': [section['section_title']],
                    'section_count': 1,
                    'chunk_type': 'text'
                }

                chunks.append(chunk)

                
            else:
                # Split section into multiple chunks
                section_chunks = self._split_with_overlap(
                    section_text,
                    tokens,
                    section
                )
                
                # Keep all chunks assigned to parser's section
                for chunk in section_chunks:
                    # Add section list metadata (all chunks belong to same section)
                    chunk['sections_list'] = [section['section']]
                    chunk['section_titles_list'] = [section['section_title']]
                    chunk['section_count'] = 1
                    chunk['chunk_type'] = 'text'
                chunks.extend(section_chunks)
        
        return chunks
        
    def _split_with_overlap(
        self, 
        text: str, 
        tokens: List[int], 
        section: Dict
    ) -> List[Dict]:
        """Split a large section into overlapping chunks."""
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(tokens):
            # Get chunk slice
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Clean up chunk boundaries (try to end at sentence)
            if end < len(tokens):
                chunk_text = self._clean_chunk_boundary(chunk_text)
            
            chunks.append({
                'text': chunk_text,
                'section': section['section'],
                'section_title': section['section_title'],
                'page_start': section.get('page_start'),
                'page_end': section.get('page_end'),
                'chunk_index': chunk_idx,
                'total_chunks': -1  # Will update after loop
            })
            
            # Move to next chunk with overlap
            start = end - self.overlap
            chunk_idx += 1
        
        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total
        
        return chunks
    
    def _clean_chunk_boundary(self, text: str) -> str:
        """Try to end chunk at a sentence boundary."""
        # Look for last period, question mark, or exclamation
        for delimiter in ['. ', '.\n', '? ', '! ']:
            last_idx = text.rfind(delimiter)
            if last_idx > len(text) * 0.7:  # Only if it's near the end
                return text[:last_idx + 1].strip()
        
        # Fallback: return as is
        return text.strip()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
