import fitz  # PyMuPDF
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def extract_page_text_native(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Opens a PDF and extracts selectable text page-by-page.
    Detects if pages are scanned (i.e. contain very little or no selectable text).
    
    Returns a list of dicts:
        [
            {
                "page_number": 1,  # 1-indexed
                "text": "Extracted text contents...",
                "is_scanned": False
            },
            ...
        ]
    """
    results = []
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF at {pdf_path}: {e}", exc_info=True)
        raise RuntimeError(f"Could not open PDF file: {e}")
        
    try:
        for i in range(len(doc)):
            page = doc[i]
            page_num = i + 1
            text = page.get_text().strip()
            image_count = len(page.get_images(full=True))
            draw_count = len(page.get_drawings()) if hasattr(page, "get_drawings") else 0

            # Scanned/image-heavy page: little selectable text but visual content exists
            is_scanned = len(text) < 150 and (image_count > 0 or draw_count > 0 or len(text) == 0)
            # Also OCR pages with almost no text even if image metadata is missing
            if len(text) < 50:
                is_scanned = True
            
            results.append({
                "page_number": page_num,
                "text": text,
                "is_scanned": is_scanned
            })
            logger.debug(f"Page {page_num}/{len(doc)} processed. Length: {len(text)}. Scanned fallback: {is_scanned}")
    except Exception as e:
        logger.error(f"Error during native text extraction from {pdf_path}: {e}", exc_info=True)
    finally:
        doc.close()
        
    return results
