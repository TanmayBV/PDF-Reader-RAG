import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Initialize logging configuration before importing models
from backend.app.utils.helpers import logger
from backend.app.api.routes import router

# Load .env file from root/backend folders
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown lifecycle events.
    Pre-loads heavy models to prevent cold-start latency on first query.
    """
    logger.info("RAG Backend: Initializing startup routines...")
    
    try:
        # Import models and collections inside startup to prevent circular references
        from backend.app.retrieval.qdrant_db import init_collection
        from backend.app.retrieval.embedder import get_embedding_model
        from backend.app.retrieval.reranker import get_reranker_model
        
        # 1. Initialize Qdrant database & collections
        logger.info("RAG Backend: Initializing Qdrant collection...")
        init_collection()
        
        # 2. Warm up BGE Embedding model
        logger.info("RAG Backend: Pre-loading embedding model BAAI/bge-small-en-v1.5...")
        get_embedding_model()
        
        # 3. Warm up BGE Reranker model
        logger.info("RAG Backend: Pre-loading reranking model BAAI/bge-reranker-base...")
        get_reranker_model()
        
        logger.info("RAG Backend: Startup finished. App is ready to receive queries.")
    except Exception as e:
        logger.critical(f"RAG Backend: Startup initialization failed: {e}", exc_info=True)
        
    yield
    
    logger.info("RAG Backend: Shutting down application services...")

# Initialize FastAPI application
app = FastAPI(
    title="RAG PDF Chatbot API",
    description="High-performance, async API backend for PDF ingestion, vector search, and Groq-powered generation with citations.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration to allow local/docker frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production security if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Production-Grade RAG PDF Chatbot API. Head to /docs for Swagger documentation."}

if __name__ == "__main__":
    import uvicorn
    # Read host and port from environment or use defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("backend.app.main:app", host=host, port=port, reload=True)
