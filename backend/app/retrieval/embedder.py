import logging
from typing import List

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# E5-small: strong retrieval quality at 384 dims with fast CPU/GPU inference
MODEL_NAME = "intfloat/e5-small-v2"
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "

_model = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading embedding model '%s' on device: %s", MODEL_NAME, device)
        try:
            _model = SentenceTransformer(MODEL_NAME, device=device)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load embedding model: %s", e, exc_info=True)
            raise RuntimeError(f"Could not load SentenceTransformer: {e}") from e
    return _model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embeddings for document chunks (E5 passage prefix)."""
    model = get_embedding_model()
    prefixed = [f"{PASSAGE_PREFIX}{t}" for t in texts]
    try:
        embeddings = model.encode(
            prefixed,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()
    except Exception as e:
        logger.error("Failed to generate embeddings: %s", e, exc_info=True)
        raise


def embed_query(query: str) -> List[float]:
    """Embedding for a user query (E5 query prefix)."""
    model = get_embedding_model()
    instruction_query = f"{QUERY_PREFIX}{query}"
    try:
        embedding = model.encode(instruction_query, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        logger.error("Failed to generate query embedding: %s", e, exc_info=True)
        raise
