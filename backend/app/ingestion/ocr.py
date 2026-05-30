import io
import logging
import os
import shutil
from typing import Optional

import fitz
import numpy as np
import pytesseract
from dotenv import load_dotenv
from PIL import Image, ImageFilter, ImageOps

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

def _normalize_tesseract_path(raw: str) -> str:
    """Fix Windows .env paths (backslashes / unquoted Program Files paths)."""
    if not raw:
        return ""
    path = raw.strip().strip("\"'")
    path = path.replace("/", os.sep)
    if path and not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


TESSERACT_CMD = _normalize_tesseract_path(os.getenv("TESSERACT_CMD", ""))
TESSERACT_LANG = os.getenv("TESSERACT_LANG", "eng").strip()
# Page segmentation: 6 = uniform block of text (good for PDF pages)
TESSERACT_PSM = os.getenv("TESSERACT_PSM", "6").strip()

_tesseract_available = False
_tesseract_version: Optional[str] = None

# Tesseract CLI flags tuned for document pages
TESSERACT_CONFIG = f"--oem 3 --psm {TESSERACT_PSM} -l {TESSERACT_LANG}"


def configure_tesseract() -> bool:
    global TESSERACT_CMD, _tesseract_available, _tesseract_version

    candidates: list[str] = []
    if TESSERACT_CMD:
        candidates.append(TESSERACT_CMD)

    which = shutil.which("tesseract")
    if which:
        candidates.append(which)

    candidates.extend(
        [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
            os.path.expandvars(r"%ProgramFiles%\Tesseract-OCR\tesseract.exe"),
        ]
    )

    for path in candidates:
        if path and os.path.isfile(path):
            TESSERACT_CMD = path
            pytesseract.pytesseract.tesseract_cmd = path
            try:
                # Must be str — pytesseract returns a Version object that breaks FastAPI JSON
                _tesseract_version = str(pytesseract.get_tesseract_version())
                _tesseract_available = True
                logger.info(
                    "Tesseract configured at %s (version %s)",
                    path,
                    _tesseract_version,
                )
                return True
            except Exception as e:
                logger.debug("Tesseract at %s failed version check: %s", path, e)

    logger.warning(
        "Tesseract OCR was not found. Install Tesseract and/or set TESSERACT_CMD in .env. "
        "Windows: https://github.com/UB-Mannheim/tesseract/wiki"
    )
    _tesseract_available = False
    return False


def is_tesseract_available() -> bool:
    return _tesseract_available


def get_tesseract_status() -> dict:
    return {
        "available": _tesseract_available,
        "version": str(_tesseract_version) if _tesseract_version is not None else None,
        "path": TESSERACT_CMD or None,
        "lang": TESSERACT_LANG,
        "config": TESSERACT_CONFIG,
    }


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Improve OCR accuracy on scanned PDF pages: grayscale, contrast, denoise, binarize.
    """
    try:
        import cv2
    except ImportError:
        gray = image.convert("L")
        gray = ImageOps.autocontrast(gray)
        gray = gray.filter(ImageFilter.SHARPEN)
        return gray.convert("RGB")

    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Mild denoise without blurring text edges
    gray = cv2.fastNlMeansDenoising(gray, h=8, templateWindowSize=7, searchWindowSize=21)

    # Adaptive threshold handles uneven lighting on scans
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    # Small morphological close to reconnect broken characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(binary).convert("RGB")


def extract_text_via_ocr(pdf_path: str, page_number: int, dpi: int = 200) -> str:
    """
    Render a PDF page and run Tesseract OCR with preprocessing.
    """
    global _tesseract_available

    if not _tesseract_available:
        _tesseract_available = configure_tesseract()
    if not _tesseract_available:
        logger.warning("Skipping OCR for page %s — Tesseract not available.", page_number)
        return ""

    doc = None
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]

        # Higher DPI improves character recognition on scans (trade-off: speed)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        processed = preprocess_image_for_ocr(image)

        text = pytesseract.image_to_string(processed, config=TESSERACT_CONFIG)
        cleaned = " ".join(text.split())

        if len(cleaned) < 20:
            # Retry with full-page auto segmentation if block mode returned little text
            alt_config = f"--oem 3 --psm 3 -l {TESSERACT_LANG}"
            text_alt = pytesseract.image_to_string(processed, config=alt_config)
            cleaned_alt = " ".join(text_alt.split())
            if len(cleaned_alt) > len(cleaned):
                cleaned = cleaned_alt

        if cleaned:
            logger.info(
                "OCR page %s of %s: extracted %d characters (dpi=%d)",
                page_number,
                os.path.basename(pdf_path),
                len(cleaned),
                dpi,
            )
        else:
            logger.warning("OCR page %s of %s returned no text.", page_number, os.path.basename(pdf_path))

        return cleaned
    except pytesseract.TesseractNotFoundError:
        _tesseract_available = False
        logger.error(
            "Tesseract executable not found at runtime. Set TESSERACT_CMD in .env to your tesseract.exe path."
        )
        return ""
    except Exception as e:
        logger.error(
            "OCR failed on %s page %s: %s",
            pdf_path,
            page_number,
            e,
            exc_info=True,
        )
        return ""
    finally:
        if doc is not None:
            doc.close()


# Configure on module import
_tesseract_available = configure_tesseract()
