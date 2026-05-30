import logging
import torch
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Reranker model configuration
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

_reranker = None

def get_reranker_model() -> CrossEncoder:
    """
    Initializes and returns the CrossEncoder reranker model (singleton).
    Automatically maps to GPU (CUDA) if available.
    """
    global _reranker
    if _reranker is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading reranker model '{RERANKER_MODEL_NAME}' on device: {device}")
        try:
            # Load CrossEncoder model
            _reranker = CrossEncoder(RERANKER_MODEL_NAME, device=device)
            logger.info("Reranker model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}", exc_info=True)
            raise RuntimeError(f"Could not load CrossEncoder reranker: {e}")
    return _reranker

def rerank_chunks(query: str, chunks: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Reranks a list of retrieved chunks against a query using a cross-encoder.
    Returns the top_n chunks sorted by their relevance score.
    
    Args:
        query: The user query string.
        chunks: List of candidate chunks from retriever.
        top_n: Number of chunks to return. Default 5.
        
    Returns:
        Sorted list of top_n chunks containing an added "rerank_score" field.
    """
    if not chunks:
        return []
        
    try:
        reranker = get_reranker_model()
        
        # Prepare pairs for cross-encoder scoring: [[query, text1], [query, text2], ...]
        pairs = [[query, chunk["text"]] for chunk in chunks]
        
        logger.info(f"Rerank scoring {len(chunks)} pairs for query: '{query}'")
        
        # Compute scores (higher is better)
        scores = reranker.predict(pairs)
        
        # Add score to each chunk dict
        for idx, score in enumerate(scores):
            chunks[idx]["rerank_score"] = float(score)
            
        # Sort chunks in descending order of rerank score
        sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        # Limit to top_n
        top_chunks = sorted_chunks[:top_n]
        logger.info(f"Reranking complete. Best score: {top_chunks[0]['rerank_score']:.4f} (if any).")
        return top_chunks
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}", exc_info=True)
        # Fallback: if reranking fails, return the original top_n chunks from FAISS directly
        # and assign a fallback score
        for chunk in chunks:
            chunk["rerank_score"] = chunk.get("score", 0.0)
        return chunks[:top_n]
