import logging
from typing import Dict, List, Any, Tuple
import re

logger = logging.getLogger(__name__)

# Try loading the tokenizer for BAAI/bge-small-en-v1.5
# If offline or errors, we fall back to a character/word approximation.
_tokenizer = None
try:
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    logger.info("Successfully loaded BAAI/bge-small-en-v1.5 tokenizer.")
except Exception as e:
    logger.warning(
        f"Could not download/load tokenizer from HuggingFace ({e}). "
        "Falling back to word-based token estimation (1 token ≈ 0.75 words)."
    )

def count_tokens(text: str) -> int:
    """
    Counts tokens in a string using the BGE tokenizer (fallback to word estimation).
    """
    if _tokenizer is not None:
        try:
            return len(_tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
            
    # Fallback estimation: split on whitespaces and scale
    words = len(text.split())
    return int(words * 1.3)  # Standard English token-to-word ratio is ~1.3

def chunk_document(
    pages: List[Dict[str, Any]],
    filename: str,
    pdf_id: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150
) -> List[Dict[str, Any]]:
    """
    Chunks extracted document pages into overlapping passages while keeping track of page numbers.
    
    Args:
        pages: List of dictionaries with "page_number" and "text" (already cleaned).
        filename: Name of the PDF file.
        pdf_id: Unique identifier for the PDF.
        chunk_size: Target size in tokens.
        chunk_overlap: Target overlap in tokens.
        
    Returns:
        List of chunks with payloads and metadata:
        [
            {
                "chunk_id": "pdf_id_0",
                "text": "Chunk text content...",
                "metadata": {
                    "pdf_id": "...",
                    "filename": "...",
                    "page_number": 2,
                    "chunk_id": "..."
                }
            },
            ...
        ]
    """
    chunks = []
    
    # 1. Break pages into sentences/sentences-with-page-metadata
    # This prevents breaking chunks in the middle of words and keeps clean context.
    sentence_records: List[Tuple[str, int]] = []
    
    for page in pages:
        page_num = page["page_number"]
        page_text = page["text"]
        
        if not page_text:
            continue
            
        # Split on sentences using regex. Split by sentence terminators followed by space.
        sentences = re.split(r'(?<=[.!?])\s+', page_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence_records.append((sentence, page_num))
                
    if not sentence_records:
        return []
        
    # 2. Build chunks using a sliding window over sentences
    current_chunk_sentences = []
    current_chunk_tokens = 0
    
    i = 0
    chunk_idx = 0
    
    while i < len(sentence_records):
        sentence, page_num = sentence_records[i]
        sentence_tokens = count_tokens(sentence)
        
        # If a single sentence is extremely long, split it into smaller character-based parts
        if sentence_tokens > chunk_size:
            # Add what we currently have if not empty
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                # Primary page is the page of the first sentence in the chunk
                primary_page = sentence_records[i - len(current_chunk_sentences)][1]
                chunks.append({
                    "chunk_id": f"{pdf_id}_{chunk_idx}",
                    "text": chunk_text,
                    "metadata": {
                        "pdf_id": pdf_id,
                        "filename": filename,
                        "page_number": primary_page,
                        "chunk_id": f"{pdf_id}_{chunk_idx}"
                    }
                })
                chunk_idx += 1
                current_chunk_sentences = []
                current_chunk_tokens = 0
                
            # Break down the massive sentence by words
            words = sentence.split()
            sub_chunk_words = []
            sub_chunk_tokens = 0
            
            for word in words:
                word_tokens = count_tokens(word)
                if sub_chunk_tokens + word_tokens > chunk_size:
                    chunks.append({
                        "chunk_id": f"{pdf_id}_{chunk_idx}",
                        "text": " ".join(sub_chunk_words),
                        "metadata": {
                            "pdf_id": pdf_id,
                            "filename": filename,
                            "page_number": page_num,
                            "chunk_id": f"{pdf_id}_{chunk_idx}"
                        }
                    })
                    chunk_idx += 1
                    # Overlap handling for huge sentences: keep last 20% of words
                    overlap_w_count = max(1, int(len(sub_chunk_words) * 0.2))
                    sub_chunk_words = sub_chunk_words[-overlap_w_count:]
                    sub_chunk_tokens = count_tokens(" ".join(sub_chunk_words))
                    
                sub_chunk_words.append(word)
                sub_chunk_tokens += word_tokens
                
            if sub_chunk_words:
                chunks.append({
                    "chunk_id": f"{pdf_id}_{chunk_idx}",
                    "text": " ".join(sub_chunk_words),
                    "metadata": {
                        "pdf_id": pdf_id,
                        "filename": filename,
                        "page_number": page_num,
                        "chunk_id": f"{pdf_id}_{chunk_idx}"
                    }
                })
                chunk_idx += 1
                
            i += 1
            continue
            
        # Standard sentence adding
        if current_chunk_tokens + sentence_tokens > chunk_size:
            # We reached the limit, save current chunk
            chunk_text = " ".join(current_chunk_sentences)
            primary_page = sentence_records[i - len(current_chunk_sentences)][1]
            chunks.append({
                "chunk_id": f"{pdf_id}_{chunk_idx}",
                "text": chunk_text,
                "metadata": {
                    "pdf_id": pdf_id,
                    "filename": filename,
                    "page_number": primary_page,
                    "chunk_id": f"{pdf_id}_{chunk_idx}"
                }
            })
            chunk_idx += 1
            
            # Backtrack window to achieve target overlap
            overlap_tokens = 0
            overlap_sentences_count = 0
            
            # Iterate backwards to find how many sentences to keep for overlap
            for back_idx in range(len(current_chunk_sentences) - 1, -1, -1):
                sent_back = current_chunk_sentences[back_idx]
                tokens_back = count_tokens(sent_back)
                if overlap_tokens + tokens_back > chunk_overlap:
                    break
                overlap_tokens += tokens_back
                overlap_sentences_count += 1
                
            # If overlap sentences count is 0, but we need overlap, keep at least 1
            if overlap_sentences_count == 0 and current_chunk_sentences:
                overlap_sentences_count = 1
                
            current_chunk_sentences = current_chunk_sentences[-overlap_sentences_count:] if overlap_sentences_count > 0 else []
            current_chunk_tokens = count_tokens(" ".join(current_chunk_sentences)) if current_chunk_sentences else 0
            
        current_chunk_sentences.append(sentence)
        current_chunk_tokens += sentence_tokens
        i += 1
        
    # Flush remaining sentences
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        primary_page = sentence_records[len(sentence_records) - len(current_chunk_sentences)][1]
        chunks.append({
            "chunk_id": f"{pdf_id}_{chunk_idx}",
            "text": chunk_text,
            "metadata": {
                "pdf_id": pdf_id,
                "filename": filename,
                "page_number": primary_page,
                "chunk_id": f"{pdf_id}_{chunk_idx}"
            }
        })
        
    logger.info(f"Chunked document {filename} (ID: {pdf_id}) into {len(chunks)} chunks.")
    return chunks
