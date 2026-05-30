import os
import fitz
import pytesseract
from PIL import Image
import io
import logging
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment configurations from both workspace root and backend folder
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Try to configure tesseract command path from environment variable (strip any surrounding quotes)
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip("\"'")

def configure_tesseract():
    """
    Attempts to locate and configure the pytesseract executable path.
    Looks at environment variable and then common installation locations.
    """
    global TESSERACT_CMD
    
    # If already set and valid, use it
    if TESSERACT_CMD and os.path.exists(TESSERACT_CMD):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        return True
        
    # Search common paths on Windows
    common_windows_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%APPDATA%\Tesseract-OCR\tesseract.exe")
    ]
    
    for path in common_windows_paths:
        if os.path.exists(path):
            TESSERACT_CMD = path
            pytesseract.pytesseract.tesseract_cmd = path
            logger.info(f"Automatically configured Tesseract executable at: {path}")
            return True
            
    # Check if tesseract is in PATH by running a quick check (will raise exception if not found)
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        pass
        
    logger.warning(
        "Tesseract OCR executable could not be automatically located. "
        "Scanned pages requiring OCR will be skipped. "
        "Please configure TESSERACT_CMD in your .env file."
    )
    return False

# Initialize configuration
_tesseract_available = configure_tesseract()

def extract_text_via_ocr(pdf_path: str, page_number: int, dpi: int = 150) -> str:
    """
    Renders a specific page of a PDF as an image and extracts text using Tesseract OCR.
    
    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-indexed page number to OCR.
        dpi: Dots Per Inch for rendering quality. Default 150 is a good trade-off between speed and accuracy.
        
    Returns:
        Extracted text as a string, or empty string if Tesseract is not available or extraction fails.
    """
    global _tesseract_available
    if not _tesseract_available:
        # Re-check configuration in case env was set later
        _tesseract_available = configure_tesseract()
        if not _tesseract_available:
            logger.warning(f"Skipping OCR for page {page_number} because Tesseract is not available.")
            return ""
            
    try:
        doc = fitz.open(pdf_path)
        # Convert 1-indexed page_number to 0-indexed for PyMuPDF
        page = doc[page_number - 1]
        
        # Render page to image bytes
        pix = page.get_pixmap(dpi=dpi)
        image_bytes = pix.tobytes("png")
        doc.close()
        
        # Open image with PIL and run OCR
        image = Image.open(io.BytesIO(image_bytes))
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        logger.error(f"Failed to extract text via OCR on {pdf_path} (Page {page_number}): {e}", exc_info=True)
        return ""
