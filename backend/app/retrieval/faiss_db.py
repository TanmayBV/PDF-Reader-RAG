import json
import logging
import os
import threading
import uuid
from typing import Any, Dict, List, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)

VECTOR_SIZE = 384  # intfloat/e5-small-v2
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"
CHUNK_MAP_FILENAME = "chunk_id_map.json"

_lock = threading.Lock()
_index: Optional[faiss.Index] = None
_id_to_payload: Dict[int, Dict[str, Any]] = {}
_chunk_id_to_faiss_id: Dict[str, int] = {}


def _store_dir() -> str:
    path = os.getenv("FAISS_STORE_PATH", "./faiss_store")
    os.makedirs(path, exist_ok=True)
    return path


def _paths() -> tuple[str, str, str]:
    base = _store_dir()
    return (
        os.path.join(base, INDEX_FILENAME),
        os.path.join(base, METADATA_FILENAME),
        os.path.join(base, CHUNK_MAP_FILENAME),
    )


def chunk_id_to_faiss_id(chunk_id: str) -> int:
    """Deterministic positive int64 ID for FAISS IndexIDMap2."""
    value = uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id).int
    return value & 0x7FFFFFFFFFFFFFFF


def _new_index() -> faiss.Index:
    """HNSW + inner product on L2-normalized vectors = cosine similarity."""
    base = faiss.IndexHNSWFlat(VECTOR_SIZE, 32, faiss.METRIC_INNER_PRODUCT)
    base.hnsw.efConstruction = 128
    base.hnsw.efSearch = 64
    return faiss.IndexIDMap2(base)


def _save_locked() -> None:
    index_path, meta_path, map_path = _paths()
    faiss.write_index(_index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in _id_to_payload.items()}, f, ensure_ascii=False)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(_chunk_id_to_faiss_id, f, ensure_ascii=False)
    logger.debug("FAISS index and metadata persisted to %s", _store_dir())


def _load_locked() -> None:
    global _index, _id_to_payload, _chunk_id_to_faiss_id
    index_path, meta_path, map_path = _paths()

    if os.path.exists(index_path):
        _index = faiss.read_index(index_path)
        logger.info("Loaded FAISS index from %s (%d vectors)", index_path, _index.ntotal)
    else:
        _index = _new_index()
        logger.info("Created new FAISS HNSW index (dim=%d)", VECTOR_SIZE)

    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            raw = json.load(f)
        _id_to_payload = {int(k): v for k, v in raw.items()}
    else:
        _id_to_payload = {}

    if os.path.exists(map_path):
        with open(map_path, encoding="utf-8") as f:
            _chunk_id_to_faiss_id = json.load(f)
    else:
        _chunk_id_to_faiss_id = {}


def _ensure_loaded() -> faiss.Index:
    global _index
    with _lock:
        if _index is None:
            _load_locked()
        assert _index is not None
        return _index


def get_vector_store() -> faiss.Index:
    return _ensure_loaded()


def init_collection() -> None:
    """Ensures the FAISS index and metadata store exist on disk."""
    _ensure_loaded()


def is_healthy() -> bool:
    try:
        init_collection()
        return _index is not None
    except Exception as e:
        logger.error("FAISS health check failed: %s", e)
        return False


def upsert_chunks(chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
    if not chunks:
        return

    with _lock:
        if _index is None:
            _load_locked()
        assert _index is not None

        vectors = np.asarray(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)

        ids_to_add: List[int] = []
        vectors_to_add: List[np.ndarray] = []

        for idx, chunk in enumerate(chunks):
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]
            metadata = chunk["metadata"]
            faiss_id = chunk_id_to_faiss_id(chunk_id)
            payload = {"text": text, **metadata}

            if chunk_id in _chunk_id_to_faiss_id and _index.ntotal > 0:
                old_id = _chunk_id_to_faiss_id[chunk_id]
                try:
                    _index.remove_ids(np.array([old_id], dtype=np.int64))
                except Exception:
                    pass
                _id_to_payload.pop(old_id, None)

            ids_to_add.append(faiss_id)
            vectors_to_add.append(vectors[idx])
            _id_to_payload[faiss_id] = payload
            _chunk_id_to_faiss_id[chunk_id] = faiss_id

        add_matrix = np.vstack(vectors_to_add).astype(np.float32)
        add_ids = np.array(ids_to_add, dtype=np.int64)
        _index.add_with_ids(add_matrix, add_ids)

        _save_locked()

    logger.info("Indexed %d chunks in FAISS (%d total vectors)", len(chunks), get_vector_store().ntotal)


def search_similar_chunks(
    query_vector: List[float],
    top_k: int = 20,
    filter_filename: Optional[str] = None,
) -> List[Dict[str, Any]]:
    with _lock:
        if _index is None:
            _load_locked()
        assert _index is not None

        if _index.ntotal == 0:
            return []

        q = np.asarray([query_vector], dtype=np.float32)
        faiss.normalize_L2(q)

        fetch_k = min(top_k * 4 if filter_filename else top_k, _index.ntotal)
        scores, ids = _index.search(q, fetch_k)

        hits: List[Dict[str, Any]] = []
        for score, faiss_id in zip(scores[0], ids[0]):
            if faiss_id == -1:
                continue
            payload = _id_to_payload.get(int(faiss_id))
            if not payload:
                continue
            if filter_filename and payload.get("filename") != filter_filename:
                continue
            hits.append(
                {
                    "score": float(score),
                    "text": payload.get("text", ""),
                    "metadata": {
                        "pdf_id": payload.get("pdf_id", ""),
                        "filename": payload.get("filename", ""),
                        "page_number": payload.get("page_number", 0),
                        "chunk_id": payload.get("chunk_id", ""),
                    },
                }
            )
            if len(hits) >= top_k:
                break

        return hits
