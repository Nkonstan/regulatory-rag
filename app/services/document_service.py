"""
Document processing service.

Handles all business logic for PDF ingestion and processing.
"""

import json
from pathlib import Path
from typing import List, Dict, Set
import logging

from app.config import settings
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import SectionChunker
from app.retrieval.vector_store import vector_store
from app.retrieval.embeddings import embedding_service

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for managing document ingestion and processing."""
    
    def __init__(self):
        """Initialize document service."""
        logger.info("⏳ Initializing document service...")
        self.pdf_parser = PDFParser()
        logger.info("✅ PDF parser ready")
        
        self.chunker = SectionChunker()
        logger.info("✅ Text chunker ready")

        self.tracking_file = settings.upload_dir / ".processed_files.json"
        logger.info("✅ Document service ready")
    
    def load_processed_files(self) -> Set[str]:
        """
        Load set of already processed filenames.
        
        Returns:
            Set of processed filenames
        """
        if not self.tracking_file.exists():
            return set()
        
        try:
            with open(self.tracking_file, 'r') as f:
                files = json.load(f)
                logger.debug(f"Loaded {len(files)} processed files from tracking")
                return set(files)
        except Exception as e:
            logger.warning(f"Failed to load tracking file: {e}")
            return set()
    
    def save_processed_file(self, filename: str):
        """
        Mark a file as processed.
        
        Args:
            filename: Name of processed file
        """
        try:
            processed = self.load_processed_files()
            processed.add(filename)
            
            with open(self.tracking_file, 'w') as f:
                json.dump(list(processed), f, indent=2)
            
            logger.debug(f"Marked {filename} as processed")
        except Exception as e:
            logger.error(f"Failed to save tracking file: {e}")
    
    def process_pdf(self, file_path: Path) -> Dict:
        """
        Process a single PDF file.
        
        Args:
            file_path: Path to PDF file
        
        Returns:
            Dict with processing results
        """
        logger.info(f"Processing PDF: {file_path.name}")
        
        try:
            # Step 1: Extract sections
            logger.debug(f"Extracting sections from {file_path.name}")
            sections = self.pdf_parser.extract_text_with_sections(file_path)
            logger.info(f"Extracted {len(sections)} sections from {file_path.name}")
            
            # Step 2: Create chunks
            logger.debug(f"Chunking {len(sections)} sections")
            chunks = self.chunker.chunk_sections(sections)
            logger.info(f"Created {len(chunks)} chunks from {file_path.name}")
            
            # Step 3: Generate embeddings
            logger.debug(f"Generating embeddings for {len(chunks)} chunks")
            texts = [chunk['text'] for chunk in chunks]
            embeddings = embedding_service.embed_documents(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Step 4: Store in vector database
            logger.debug(f"Storing chunks in vector database")
            vector_store.add_chunks(chunks, file_path.name, embeddings)
            logger.info(f"Successfully stored {len(chunks)} chunks for {file_path.name}")
            
            # Mark as processed
            self.save_processed_file(file_path.name)
            
            return {
                'filename': file_path.name,
                'sections': len(sections),
                'chunks': len(chunks),
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}", exc_info=True)
            return {
                'filename': file_path.name,
                'status': 'failed',
                'error': str(e)
            }
    
    def process_startup_pdfs(self) -> Dict:
        """
        Process all PDFs in uploads directory on startup.
        
        Returns:
            Dict with processing summary
        """
        logger.info("Starting PDF auto-processing on startup")
        
        # Find PDFs
        pdf_files = list(settings.upload_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDFs found in {settings.upload_dir}")
            return {
                'total_files': 0,
                'processed': 0,
                'skipped': 0,
                'failed': 0
            }
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Check which are already processed
        processed_files = self.load_processed_files()
        new_files = [f for f in pdf_files if f.name not in processed_files]
        
        if not new_files:
            logger.info(f"All {len(pdf_files)} PDFs already processed")
            return {
                'total_files': len(pdf_files),
                'processed': 0,
                'skipped': len(pdf_files),
                'failed': 0,
                'total_chunks': vector_store.count()
            }
        
        logger.info(f"Processing {len(new_files)} new PDFs")
        
        # Process each new file
        results = []
        for pdf_file in new_files:
            result = self.process_pdf(pdf_file)
            results.append(result)
        
        # Calculate summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = len(results) - successful
        total_chunks = sum(r.get('chunks', 0) for r in results if r['status'] == 'success')
        
        summary = {
            'total_files': len(pdf_files),
            'processed': successful,
            'skipped': len(processed_files),
            'failed': failed,
            'total_chunks': total_chunks,
            'database_chunks': vector_store.count()
        }
        
        logger.info(f"Startup processing complete: {summary}")
        return summary
    
    def get_pdf_files(self) -> List[Path]:
        """
        Get list of all PDF files in uploads directory.
        
        Returns:
            List of PDF file paths
        """
        pdf_files = list(settings.upload_dir.glob("*.pdf"))
        logger.debug(f"Found {len(pdf_files)} PDF files")
        return pdf_files


# Singleton instance
document_service = DocumentService()
