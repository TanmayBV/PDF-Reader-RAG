import os
import asyncio
import time
import logging
from dotenv import load_dotenv

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("test_rag")

# Load environment
load_dotenv()

# We need to append the project directory to sys.path if running directly
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

async def run_test():
    logger.info("Starting RAG Chatbot Integration Test...")
    
    # Set local qdrant database path for testing
    os.environ["QDRANT_LOCAL_PATH"] = "./qdrant_test_db"
    
    # 1. Warm up embedding & reranker models
    logger.info("Step 1: Pre-loading BGE embedding and reranking models...")
    start = time.perf_counter()
    
    from backend.app.retrieval.embedder import get_embedding_model, embed_texts
    from backend.app.retrieval.reranker import get_reranker_model, rerank_chunks
    from backend.app.retrieval.qdrant_db import init_collection, upsert_chunks
    from backend.app.retrieval.retriever import retrieve_candidates
    from backend.app.generation.llm import generate_answer
    
    get_embedding_model()
    get_reranker_model()
    init_collection()
    
    model_load_time = time.perf_counter() - start
    logger.info(f"Models loaded and Vector DB collection initialized in {model_load_time:.2f}s.")
    
    # 2. Simulate PDF chunking & ingestion
    logger.info("Step 2: Simulating chunking and database indexing...")
    mock_pdf_id = "test_doc_123"
    mock_filename = "aerospace_manual.pdf"
    
    # Formulate mock pages
    mock_pages = [
        {
            "page_number": 1,
            "text": (
                "The Saturn V was a human-rated expendable rocket used by NASA between 1967 and 1973. "
                "It was developed to support the Apollo program for human exploration of the Moon. "
                "The rocket was a three-stage liquid-fueled launch vehicle. It stands 111 meters tall and "
                "weighed nearly 3 million kilograms when fully loaded with fuel."
            )
        },
        {
            "page_number": 2,
            "text": (
                "Saturn V's first stage, the S-IC, was powered by five F-1 rocket engines. "
                "These engines burned kerosene (RP-1) and liquid oxygen (LOX). The five engines produced a total "
                "thrust of 34.5 million newtons, which is equivalent to 160 million horsepower. "
                "The S-IC stage burned for approximately 150 seconds, lifting the rocket to an altitude of 68 kilometers."
            )
        },
        {
            "page_number": 3,
            "text": (
                "The second stage (S-II) was powered by five J-2 rocket engines burning liquid hydrogen and liquid oxygen. "
                "The third stage (S-IVB) was powered by a single J-2 engine and was responsible for inserting the spacecraft "
                "into orbit and then sending it towards the Moon (Trans-Lunar Injection)."
            )
        }
    ]
    
    from backend.app.ingestion.chunking import chunk_document
    chunks = chunk_document(
        pages=mock_pages,
        filename=mock_filename,
        pdf_id=mock_pdf_id,
        chunk_size=100,  # small chunk size for testing
        chunk_overlap=20
    )
    
    logger.info(f"Generated {len(chunks)} chunks from mock pages.")
    
    # Embed & Index
    logger.info("Step 3: Generating BGE embeddings and inserting into local Qdrant DB...")
    start_index = time.perf_counter()
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    upsert_chunks(chunks, embeddings)
    indexing_time = time.perf_counter() - start_index
    logger.info(f"Successfully indexed chunks in Qdrant in {indexing_time:.2f}s.")
    
    # 3. Test Query & Retrieval
    query = "How much thrust did the Saturn V first stage produce and what engines did it use?"
    logger.info(f"Step 4: Running retrieval pipeline for query: '{query}'")
    
    start_retrieval = time.perf_counter()
    candidates = retrieve_candidates(query, top_k=5)
    retrieval_time = time.perf_counter() - start_retrieval
    logger.info(f"Retrieved {len(candidates)} candidates from Qdrant in {retrieval_time*1000:.2f}ms.")
    
    # 4. Test Reranking
    logger.info("Step 5: Reranking candidates using BGE-reranker...")
    start_rerank = time.perf_counter()
    top_chunks = rerank_chunks(query, candidates, top_n=2)
    rerank_time = time.perf_counter() - start_rerank
    logger.info(f"Reranking completed in {rerank_time*1000:.2f}ms.")
    
    for idx, c in enumerate(top_chunks):
        logger.info(f"Top {idx+1} Match (Score: {c['rerank_score']:.4f}): Page {c['metadata']['page_number']} - {c['text'][:120]}...")
        
    # 5. Test LLM Generation with citations
    logger.info("Step 6: Generating final answer with citations from Groq API...")
    start_llm = time.perf_counter()
    
    try:
        answer, citations = await generate_answer(query, top_chunks)
        llm_time = time.perf_counter() - start_llm
        
        logger.info(f"LLM generated answer in {llm_time:.2f}s.")
        logger.info(f"Answer: {answer}")
        logger.info(f"Citations: {citations}")
        
        # Summary Latency Checklist
        total_time = retrieval_time + rerank_time + llm_time
        logger.info("\n" + "="*40 + "\nLATENCY METRICS SUMMARY:\n" + "="*40)
        logger.info(f"Vector Retrieval:   {retrieval_time*1000:.2f}ms")
        logger.info(f"Reranker:           {rerank_time*1000:.2f}ms")
        logger.info(f"LLM Generation:     {llm_time:.2f}s")
        logger.info(f"End-to-End Latency: {total_time:.2f}s")
        logger.info("="*40)
        
        if 2.0 <= total_time <= 5.0:
            logger.info("SUCCESS: End-to-end response latency is within the target 2-5 seconds range!")
        elif total_time < 2.0:
            logger.info("SUCCESS: End-to-end response latency is ultra-fast, under 2 seconds!")
        else:
            logger.warning("WARNING: End-to-end latency exceeded 5 seconds. Consider hardware limits or model parameter tuning.")
            
    except Exception as e:
        logger.error(f"Failed to generate answer (check if GROQ_API_KEY is configured in .env): {e}")

if __name__ == "__main__":
    asyncio.run(run_test())
