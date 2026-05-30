import os
import uuid
import logging
import asyncio
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, File, UploadFile, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel
from backend.app.utils.helpers import measure_time, ensure_directory
from backend.app.ingestion.ingest_pipeline import ingest_pdf
from backend.app.retrieval.embedder import embed_texts
from backend.app.retrieval.faiss_db import upsert_chunks, is_healthy as faiss_is_healthy
from backend.app.ingestion.ocr import get_tesseract_status, configure_tesseract
from backend.app.retrieval.retriever import retrieve_candidates
from backend.app.retrieval.reranker import rerank_chunks
from backend.app.generation.llm import generate_answer

logger = logging.getLogger(__name__)

router = APIRouter()

# Directories configuration (portable paths relative to backend package)
_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(_BACKEND_ROOT, "data", "raw_pdfs")
PROCESSED_DIR = os.path.join(_BACKEND_ROOT, "data", "processed")

ensure_directory(RAW_DIR)
ensure_directory(PROCESSED_DIR)

# Global in-memory ingestion tracker
ingestion_jobs: Dict[str, Dict[str, Any]] = {}

# Global query cache for instant duplicates retrieval
query_response_cache: Dict[str, Dict[str, Any]] = {}

class QueryRequest(BaseModel):
    query: str
    filter_filename: Optional[str] = None

class IngestRequest(BaseModel):
    filename: str

# Background worker function for ingestion
def run_ingestion_background(job_id: str, filepath: str, filename: str):
    try:
        ingestion_jobs[job_id]["status"] = "processing"
        
        # 1. Pipeline callback to capture page-by-page progress
        def update_progress(pct: int, total: int, phase: str):
            # Scale parsing to 0-70%
            current_pct = int(pct * 0.7)
            ingestion_jobs[job_id]["progress"] = current_pct
            ingestion_jobs[job_id]["message"] = f"{phase}..."
            
        chunks = ingest_pdf(
            pdf_path=filepath,
            pdf_id=job_id,
            progress_callback=update_progress
        )
        
        if not chunks:
            ingestion_jobs[job_id]["status"] = "failed"
            ingestion_jobs[job_id]["message"] = "No text could be extracted from PDF."
            return
            
        # 2. Embedding generation (70% - 90%)
        ingestion_jobs[job_id]["progress"] = 70
        ingestion_jobs[job_id]["message"] = "Generating BGE embeddings in batches..."
        
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_texts(chunk_texts)
        
        # 3. Save to FAISS vector store (90% - 98%)
        ingestion_jobs[job_id]["progress"] = 90
        ingestion_jobs[job_id]["message"] = "Indexing embeddings into FAISS..."
        
        upsert_chunks(chunks, embeddings)
        
        # Clear cache so new documents are immediately discoverable
        query_response_cache.clear()
        logger.info("Cleared RAG query response cache due to new document ingestion.")
        
        # 4. Finished (100%)
        ingestion_jobs[job_id]["progress"] = 100
        ingestion_jobs[job_id]["status"] = "completed"
        ingestion_jobs[job_id]["message"] = "Document successfully ingested and indexed!"
        ingestion_jobs[job_id]["chunks_count"] = len(chunks)
        
    except Exception as e:
        logger.error(f"Background ingestion failed for job {job_id}: {e}", exc_info=True)
        ingestion_jobs[job_id]["status"] = "failed"
        ingestion_jobs[job_id]["message"] = f"Ingestion error: {str(e)}"

@router.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Saves an uploaded PDF to the raw_pdfs folder.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    ensure_directory(RAW_DIR)
    filepath = os.path.join(RAW_DIR, file.filename)
    
    try:
        # Write file in chunks
        with open(filepath, "wb") as f:
            while content := await file.read(1024 * 1024):  # 1MB chunks
                f.write(content)
        logger.info(f"File uploaded successfully: {file.filename}")
        return {"filename": file.filename, "filepath": filepath}
    except Exception as e:
        logger.error(f"Failed to write uploaded file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File saving failed: {e}")

@router.post("/api/ingest")
async def start_ingestion(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Triggers the asynchronous ingestion pipeline for a saved PDF.
    """
    filepath = os.path.join(RAW_DIR, request.filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Uploaded file not found. Upload it first.")
        
    job_id = str(uuid.uuid4())[:8]
    ingestion_jobs[job_id] = {
        "job_id": job_id,
        "filename": request.filename,
        "status": "queued",
        "progress": 0,
        "message": "Queued in background task...",
        "chunks_count": 0
    }
    
    # Delegate to FastAPI BackgroundTasks to prevent event-loop freezing
    background_tasks.add_task(
        run_ingestion_background,
        job_id=job_id,
        filepath=filepath,
        filename=request.filename
    )
    
    return {"ingestion_id": job_id, "status": "queued"}

@router.get("/api/ingest/status/{ingestion_id}")
async def get_ingestion_status(ingestion_id: str):
    """
    Polls the status and progress percentage of a background ingestion task.
    """
    job = ingestion_jobs.get(ingestion_id)
    if not job:
        raise HTTPException(status_code=404, detail="Ingestion job not found.")
    return job

@router.post("/api/query")
async def query_chatbot(request: QueryRequest):
    """
    Runs the full RAG pipeline:
    1. Retrieval: retrieves top 10 candidate chunks.
    2. Reranking: filters down to top 5 chunks.
    3. Generation: Calls Groq API for the final answer.
    Tracks precise execution latency for all steps.
    """
    # 0. Check exact duplicate cache for sub-millisecond response times
    cache_key = f"{request.query.strip().lower()}_filter:{request.filter_filename or ''}"
    if cache_key in query_response_cache:
        logger.info(f"Query Cache Hit! Serving cached RAG response instantly.")
        cached_res = query_response_cache[cache_key].copy()
        # Report 0ms latency for cache hit
        cached_res["latencies"] = {
            "retrieval_ms": 0.0,
            "rerank_ms": 0.0,
            "generation_ms": 0.0,
            "total_ms": 0.0
        }
        return cached_res

    latencies = {}
    
    # 1. Retrieval (Fetch 10 candidates instead of 20 to halve cross-encoder CPU time)
    with measure_time() as retrieve_timer:
        candidates = retrieve_candidates(
            query=request.query,
            top_k=10,
            filter_filename=request.filter_filename
        )
    latencies["retrieval_ms"] = round(retrieve_timer["elapsed"] * 1000, 2)
    
    # 2. Reranking
    with measure_time() as rerank_timer:
        top_chunks = rerank_chunks(
            query=request.query,
            chunks=candidates,
            top_n=5
        )
    latencies["rerank_ms"] = round(rerank_timer["elapsed"] * 1000, 2)
    
    # 3. Generation
    with measure_time() as generate_timer:
        answer, citations = await generate_answer(
            query=request.query,
            retrieved_chunks=top_chunks
        )
    latencies["generation_ms"] = round(generate_timer["elapsed"] * 1000, 2)
    
    # Compute total latency
    latencies["total_ms"] = round(
        latencies["retrieval_ms"] + latencies["rerank_ms"] + latencies["generation_ms"], 2
    )
    
    logger.info(f"Query processed. Total latency: {latencies['total_ms']}ms")
    
    response_data = {
        "answer": answer,
        "citations": citations,
        "chunks": top_chunks,
        "latencies": latencies
    }
    
    # Cache the fresh result
    query_response_cache[cache_key] = response_data
    
    return response_data

@router.post("/api/chunks")
async def get_retrieved_chunks(request: QueryRequest):
    """
    Returns the raw retrieved and reranked chunks without generating an LLM response.
    Useful for pipeline debugging and search evaluation.
    """
    candidates = retrieve_candidates(
        query=request.query,
        top_k=10,
        filter_filename=request.filter_filename
    )
    top_chunks = rerank_chunks(
        query=request.query,
        chunks=candidates,
        top_n=5
    )
    return {
        "raw_candidates_count": len(candidates),
        "reranked_chunks": top_chunks
    }

@router.get("/api/health")
async def health_check():
    """
    Verifies API, FAISS vector store, and OCR (Tesseract) availability.
    """
    if faiss_is_healthy():
        db_status = "connected"
    else:
        db_status = "disconnected"

    ocr_status = get_tesseract_status()
    if not ocr_status["available"]:
        configure_tesseract()
        ocr_status = get_tesseract_status()

    return {
        "status": "healthy",
        "vector_db": db_status,
        "ocr": ocr_status,
    }
