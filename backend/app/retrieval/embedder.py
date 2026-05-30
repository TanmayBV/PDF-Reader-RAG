import logging
import torch
from sentence_transformers import SentenceTransformer
from typing import List, Union

logger = logging.getLogger(__name__)

# Model name configuration
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models require a prefix instruction for queries to work optimally in retrieval.
# Chunks (documents) do not need any prefix.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None

def get_embedding_model() -> SentenceTransformer:
    """
    Initializes and returns the SentenceTransformer model instance (singleton).
    Configures device automatically (CUDA if available, otherwise CPU).
    """
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model '{MODEL_NAME}' on device: {device}")
        try:
            _model = SentenceTransformer(MODEL_NAME, device=device)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Could not load SentenceTransformer: {e}")
    return _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of document chunks (passages).
    No query prefix is added.
    """
    model = get_embedding_model()
    try:
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True  # BGE recommends normalized embeddings for cosine similarity
        )
        return embeddings.tolist()
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}", exc_info=True)
        raise

def embed_query(query: str) -> List[float]:
    """
    Generates embedding for a user query.
    Prepends the recommended BGE retrieval instruction.
    """
    model = get_embedding_model()
    # Prepend retrieval instruction
    instruction_query = f"{QUERY_INSTRUCTION}{query}"
    try:
        embedding = model.encode(
            instruction_query,
            normalize_embeddings=True
        )
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}", exc_info=True)
        raise
