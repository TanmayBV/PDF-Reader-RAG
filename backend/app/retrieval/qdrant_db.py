import os
import logging
from typing import Dict, List, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

# Vector DB configuration
COLLECTION_NAME = "pdf_documents"
VECTOR_SIZE = 384  # Size of BAAI/bge-small-en-v1.5 embeddings
DISTANCE_METRIC = models.Distance.COSINE

_client = None

def get_qdrant_client() -> QdrantClient:
    """
    Initializes and returns a Qdrant client singleton.
    Supports network connection (host/port) or local filesystem fallback.
    """
    global _client
    if _client is None:
        host = os.getenv("QDRANT_HOST")
        port = os.getenv("QDRANT_PORT")
        url = os.getenv("QDRANT_URL")
        
        if url:
            logger.info(f"Connecting to Qdrant via URL: {url}")
            _client = QdrantClient(url=url)
        elif host:
            port_val = int(port) if port else 6333
            logger.info(f"Connecting to Qdrant service at {host}:{port_val}")
            _client = QdrantClient(host=host, port=port_val)
        else:
            # Local developer disk storage fallback (no docker needed!)
            local_path = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_db")
            logger.info(f"Initializing local disk-backed Qdrant client at: {local_path}")
            _client = QdrantClient(path=local_path)
            
    return _client

def init_collection() -> None:
    """
    Initializes the Qdrant collection if it does not already exist.
    Configures HNSW parameters for fast Approximate Nearest Neighbor (ANN) search.
    """
    client = get_qdrant_client()
    try:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not exists:
            logger.info(f"Creating Qdrant collection '{COLLECTION_NAME}' with HNSW indexing...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=DISTANCE_METRIC
                ),
                # Setup ANN HNSW index parameters
                hnsw_config=models.HnswConfigDiff(
                    m=16,          # Number of connections per node
                    ef_construct=100, # Build-time exploration speed/accuracy tradeoff
                    full_scan_threshold=10000,
                    on_disk=False  # Keep in RAM for sub-millisecond retrieval latency
                )
            )
            logger.info("Collection created successfully.")
        else:
            logger.debug(f"Qdrant collection '{COLLECTION_NAME}' already exists.")
    except Exception as e:
        logger.error(f"Failed to initialize Qdrant collection: {e}", exc_info=True)
        raise RuntimeError(f"Could not setup Qdrant vector database: {e}")

def upsert_chunks(chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
    """
    Upserts document chunks and their embeddings into the Qdrant collection.
    
    Args:
        chunks: List of chunk dictionaries containing text and metadata.
        embeddings: Corresponding embeddings for the chunks.
    """
    if not chunks:
        return
        
    client = get_qdrant_client()
    init_collection()
    
    points = []
    for idx, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        text = chunk["text"]
        metadata = chunk["metadata"]
        
        # Merge text into metadata payload for easy extraction during retrieval
        payload = {
            "text": text,
            **metadata
        }
        
        # Qdrant requires UUIDs or integer IDs. Since pdf_id_idx might not be standard UUID,
        # we generate deterministic/reproducible integer hashes or just generate UUIDv4 strings.
        # But chunk_id has format e.g. "pdf_id_index". Let's use it directly as a string ID
        # (Qdrant supports string UUIDs, so we'll hash the chunk_id to UUID if it's not a standard UUID,
        # or just generate a deterministic UUID based on chunk_id).
        import uuid
        deterministic_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        
        points.append(
            models.PointStruct(
                id=deterministic_uuid,
                vector=embeddings[idx],
                payload=payload
            )
        )
        
    # Bulk upsert points in batches
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
            logger.debug(f"Upserted batch of {len(batch)} points.")
        except Exception as e:
            logger.error(f"Failed to upsert points in Qdrant: {e}", exc_info=True)
            raise RuntimeError(f"Qdrant upsert failure: {e}")
            
    logger.info(f"Successfully indexed {len(chunks)} chunks in Qdrant.")

def search_similar_chunks(
    query_vector: List[float],
    top_k: int = 20,
    filter_filename: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Searches the vector DB for the nearest neighbors of a query vector.
    
    Args:
        query_vector: The embedded query.
        top_k: Number of candidates to retrieve.
        filter_filename: Optional filename filter for metadata scoping.
        
    Returns:
        A list of matching records with similarity scores.
    """
    client = get_qdrant_client()
    init_collection()
    
    query_filter = None
    if filter_filename:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="filename",
                    match=models.MatchValue(value=filter_filename)
                )
            ]
        )
        
    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        )
        
        hits = []
        for hit in response.points:
            hits.append({
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "metadata": {
                    "pdf_id": hit.payload.get("pdf_id", ""),
                    "filename": hit.payload.get("filename", ""),
                    "page_number": hit.payload.get("page_number", 0),
                    "chunk_id": hit.payload.get("chunk_id", "")
                }
            })
        return hits
    except Exception as e:
        logger.error(f"Qdrant query_points search failed: {e}", exc_info=True)
        raise RuntimeError(f"Database search failed: {e}")
