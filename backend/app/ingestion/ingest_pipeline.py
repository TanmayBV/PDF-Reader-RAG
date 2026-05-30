import os
import uuid
import logging
import time
from typing import Dict, List, Any
from backend.app.ingestion.extract_text import extract_page_text_native
from backend.app.ingestion.ocr import extract_text_via_ocr
from backend.app.ingestion.cleaner import clean_text
from backend.app.ingestion.chunking import chunk_document
from typing import Optional

logger = logging.getLogger(__name__)

def ingest_pdf(
    pdf_path: str,
    pdf_id: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    progress_callback: Any = None
) -> List[Dict[str, Any]]:
    """
    Orchestrates the entire ingestion pipeline for a single PDF.
    
    1. Extracts native text.
    2. Identifies scanned pages and runs OCR.
    3. Normalizes and cleans text.
    4. Generates overlapping chunks with metadata.
    
    Args:
        pdf_path: Path to the local PDF file.
        pdf_id: Optional unique identifier. If not provided, it is generated.
        chunk_size: Token size limit for each chunk.
        chunk_overlap: Overlap in tokens between consecutive chunks.
        progress_callback: A function that accepts (page_index, total_pages, phase_name) to report progress.
        
    Returns:
        A list of chunk dictionaries with texts and metadata.
    """
    start_time = time.time()
    filename = os.path.basename(pdf_path)
    
    if not pdf_id:
        pdf_id = str(uuid.uuid4())[:8]
        
    logger.info(f"Starting ingestion pipeline for {filename} (ID: {pdf_id})")
    
    # Phase 1: Native text extraction
    if progress_callback:
        progress_callback(0, 100, "Extracting native text")
        
    native_pages = extract_page_text_native(pdf_path)
    total_pages = len(native_pages)
    
    if total_pages == 0:
        logger.warning(f"Document {filename} has 0 pages or failed to open.")
        return []
        
    processed_pages = []
    
    # Phase 2: Iterate and OCR if necessary, then clean
    for idx, page_info in enumerate(native_pages):
        page_num = page_info["page_number"]
        page_text = page_info["text"]
        is_scanned = page_info["is_scanned"]
        
        if progress_callback:
            # Scale progress from 10% to 80% during page processing
            pct = int(10 + (idx / total_pages) * 70)
            phase = "Performing OCR" if is_scanned else "Reading native page"
            progress_callback(pct, 100, f"{phase} ({page_num}/{total_pages})")
            
        final_text = page_text
        if is_scanned:
            logger.debug(f"Page {page_num} in {filename} appears scanned. Invoking OCR...")
            ocr_text = extract_text_via_ocr(pdf_path, page_num)
            if ocr_text:
                final_text = ocr_text
                
        # Clean text
        cleaned = clean_text(final_text)
        processed_pages.append({
            "page_number": page_num,
            "text": cleaned
        })
        
    # Phase 3: Chunking
    if progress_callback:
        progress_callback(85, 100, "Segmenting text into chunks")
        
    chunks = chunk_document(
        pages=processed_pages,
        filename=filename,
        pdf_id=pdf_id,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    if progress_callback:
        progress_callback(100, 100, "Ingestion complete")
        
    duration = time.time() - start_time
    logger.info(f"Ingestion for {filename} completed in {duration:.2f}s. Generated {len(chunks)} chunks.")
    
    return chunks
