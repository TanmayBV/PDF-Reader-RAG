import logging
from typing import List, Dict, Any, Optional
from backend.app.retrieval.embedder import embed_query
from backend.app.retrieval.faiss_db import search_similar_chunks

logger = logging.getLogger(__name__)

def retrieve_candidates(query: str, top_k: int = 20, filter_filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves the top-K candidate chunks for a query by generating
    an embedding and performing a vector search in FAISS.
    
    Args:
        query: The user's search query.
        top_k: Number of chunks to retrieve from the database. Default 20.
        filter_filename: Optional filename to narrow retrieval scope.
        
    Returns:
        List of retrieved chunks with payloads, metadata, and cosine similarity scores.
    """
    logger.info(f"Retrieving top {top_k} candidate chunks for query: '{query}'")
    
    try:
        # 1. Embed query
        query_vector = embed_query(query)
        
        # 2. Search FAISS
        hits = search_similar_chunks(
            query_vector=query_vector,
            top_k=top_k,
            filter_filename=filter_filename
        )
        
        logger.info(f"Retrieved {len(hits)} raw candidates from FAISS.")
        return hits
    except Exception as e:
        logger.error(f"Failed to retrieve candidates: {e}", exc_info=True)
        return []
