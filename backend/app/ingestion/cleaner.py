import re
import logging

logger = logging.getLogger(__name__)

# Compile common page number and header/footer regex patterns
PAGE_NUMBER_PATTERNS = [
    re.compile(r'^\s*page\s+\d+\s*(?:of\s+\d+)?\s*$', re.IGNORECASE),
    re.compile(r'^\s*\d+\s*(?:of\s+\d+)?\s*$', re.IGNORECASE),
    re.compile(r'^\s*-\s*\d+\s*-\s*$', re.IGNORECASE),
    re.compile(r'^\s*\[\s*\d+\s*\]\s*$', re.IGNORECASE),
]

def clean_text(text: str) -> str:
    """
    Cleans and normalizes raw text extracted from a PDF.
    
    1. Removes obvious page numbers and running footers/headers.
    2. Standardizes unicode quotes and dashes.
    3. Normalizes multiple spaces and newlines.
    """
    if not text:
        return ""
        
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Skip empty lines
        if not line_stripped:
            continue
            
        # Check if line matches a page number pattern
        is_page_num = False
        for pattern in PAGE_NUMBER_PATTERNS:
            if pattern.match(line_stripped):
                is_page_num = True
                break
                
        if is_page_num:
            continue
            
        # Standardize whitespace inside lines
        line_cleaned = re.sub(r'\s+', ' ', line_stripped)
        cleaned_lines.append(line_cleaned)
        
    # Reconstruct text with standard single newlines
    cleaned_text = "\n".join(cleaned_lines)
    
    # Normalize multiple newlines and spaces across the whole document
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    cleaned_text = re.sub(r' +', ' ', cleaned_text)
    
    # Standardize curly quotes and dashes
    cleaned_text = (
        cleaned_text.replace('\u201c', '"')
        .replace('\u201d', '"')
        .replace('\u2018', "'")
        .replace('\u2019', "'")
        .replace('\u2014', '—')
        .replace('\u2013', '-')
    )
    
    return cleaned_text.strip()
