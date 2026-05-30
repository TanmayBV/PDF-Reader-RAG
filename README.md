# 🧠 AetherRAG - Production-Grade RAG Chatbot for Large PDFs

AetherRAG is a high-performance, asynchronous Retrieval-Augmented Generation (RAG) chatbot system built using Python, FastAPI, and Streamlit. It enables users to upload, ingest, and query a large private corpus of PDF documents (including scanned documents using OCR) in real-time, providing factual answers backed by source citations within a **2–5 second end-to-end response window**.

---

## ⚡ Key Architectural Features

* **Hybrid Ingestion Pipeline:** Extracts native selectable text using PyMuPDF and automatically triggers system OCR using Tesseract for scanned/image-heavy pages.
* **Token-Accurate Chunking:** Implements sliding-window chunking (700-1000 tokens with 100-200 token overlap) aligned with the BGE tokenizer.
* **Open-Source Embeddings & Fast Retrieval:** Uses the highly accurate `BAAI/bge-small-en-v1.5` embeddings (384-dimensions) stored in a `Qdrant` vector database utilizing HNSW indexes for sub-millisecond similarity lookups.
* **Deep Reranking Stage:** Employs a cross-encoder model (`BAAI/bge-reranker-base`) to score the top 20 candidate passages relative to the user query, narrowing context down to the top 5 chunks.
* **Async & Hosted LLM:** Integrates with the high-speed Groq API (`llama-3.1-8b-instant`) via async HTTP clients, generating answers under strict context-only instructions and outputting precise citations (`filename.pdf (Page X)`).
* **Dual-Pane Telemetry UI:** Streamlit interface features a split layout—displaying chat dialogue on the left and a live "Cognitive Telemetry" dashboard on the right showcasing retrieval logs, cosine similarity scores, BGE reranker relevance, and execution latency.
* **Docker Support:** Ready to run anywhere via `Docker Compose` with services for the vector database, backend API, and Streamlit application.

---

## 📂 Project Structure

```text
rag-chatbot/
│
├── backend/
│   ├── app/
│   │   ├── ingestion/
│   │   │   ├── extract_text.py      # PyMuPDF text reader
│   │   │   ├── ocr.py               # Pytesseract OCR pipeline
│   │   │   ├── cleaner.py           # Header/footer/whitespace cleanup
│   │   │   ├── chunking.py          # BGE-token-based chunking
│   │   │   └── ingest_pipeline.py   # Ingestion orchestrator
│   │   │
│   │   ├── retrieval/
│   │   │   ├── embedder.py          # BGE embedding generator
│   │   │   ├── qdrant_db.py         # Qdrant client & index settings
│   │   │   ├── retriever.py         # Candidate searcher
│   │   │   └── reranker.py          # BGE Cross-Encoder reranker
│   │   │
│   │   ├── generation/
│   │   │   └── llm.py               # Groq API integration
│   │   │
│   │   ├── api/
│   │   │   └── routes.py            # FastAPI async handlers
│   │   │
│   │   ├── utils/
│   │   │   └── helpers.py           # Latency metrics logger
│   │   │
│   │   └── main.py                  # API entry point & lifespan model warming
│   │
│   ├── data/
│   │   ├── raw_pdfs/                # Uploaded PDFs directory
│   │   └── processed/               # Local cache or processed outputs
│   │
│   ├── requirements.txt
│   ├── .env                         # Backend variables
│   └── Dockerfile                   # Backend build container
│
├── frontend/
│   ├── app.py                       # Streamlit UI
│   ├── requirements.txt             # Streamlit dependencies
│   └── Dockerfile                   # Frontend build container
│
├── test_rag.py                      # Local integration test pipeline
└── docker-compose.yml               # Multi-service coordinator
```

---

## 🚀 Setup & Installation

### Option 1: Running with Docker Compose (Recommended)

1. Make sure **Docker** and **Docker Compose** are installed on your machine.
2. Configure the `.env` file in the root directory (copy the template below).
3. Run the following command in the root folder:
   ```bash
   docker-compose up --build
   ```
4. Access the applications:
   * **Streamlit Frontend:** [http://localhost:8501](http://localhost:8501)
   * **FastAPI Backend Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
   * **Qdrant DB Dashboard:** [http://localhost:6333/dashboard](http://localhost:6333/dashboard)

### Option 2: Running Locally

#### 1. System Dependencies (OCR Support)
To support scanned PDFs/images, you must install Tesseract OCR on your system:
* **Windows:** Download the installer from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and install. Add the install path (e.g. `C:\Program Files\Tesseract-OCR`) to your system environment variables, or specify it in your `.env` file as shown below.
* **macOS:** `brew install tesseract`
* **Linux:** `sudo apt-get install tesseract-ocr`

#### 2. Virtual Environment Setup
Ensure Python 3.12 is installed, then set up the virtual environment:
```bash
# Create environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

#### 3. Environment Variables
Create a file named `.env` in the root workspace folder:
```env
# Groq LLM API Key (Supports both spellings)
GROQ_API_KEY="gsk_your_groq_api_key_here"

# System path to Tesseract (Only required on Windows if not in global PATH)
TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"

# FastAPI configuration (optional defaults)
API_HOST="0.0.0.0"
API_PORT=8000
```

#### 4. Launch Backend API
```bash
# From workspace root:
python -m backend.app.main
```
The backend initializes the Qdrant database (by default in `./qdrant_db`) and pre-loads the embedding/reranking models to warm up memory.

#### 5. Launch Frontend Dashboard
Open a new terminal, activate the virtual environment, and run:
```bash
streamlit run frontend/app.py
```
This opens the web browser at [http://localhost:8501](http://localhost:8501).

---

## 🛰️ REST API Specification

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/upload` | `POST` | Uploads a `.pdf` file. Saves it to `data/raw_pdfs`. |
| `/api/ingest` | `POST` | Triggers background PDF extraction, OCR, token-chunking, and Qdrant indexing. |
| `/api/ingest/status/{id}` | `GET` | Returns parsing progress (0-100%) and current pipeline log message. |
| `/api/query` | `POST` | Processes query: embeds question, runs similarity search, reranks top 5, queries Groq API, and logs latencies. |
| `/api/chunks` | `POST` | Retrieves candidate chunks (Top 20 cosine matches, Top 5 reranked matches) with scoring metrics. |
| `/api/health` | `GET` | Reports status of database connection and web server. |

---

## 🔍 Ingestion Pipeline Details

The ingestion pipeline handles documents asynchronously to handle large files (200+ pages) without blocking the FastAPI event loop:
1. **Extraction:** PyMuPDF (`fitz`) parses the page.
2. **Scanned Page Evaluation:** If the page yields fewer than 150 characters, it is flagged as scanned/image-heavy.
3. **OCR Integration:** The page is rendered as a 150 DPI image in memory and processed using `pytesseract`. If Tesseract is not configured or missing, it logs a warning and proceeds with native contents.
4. **Text Cleaning:** `cleaner.py` strips repeating page numbers (e.g. `page 3 of 40`, `- 10 -`, etc.) and running header lines, normalizes whitespaces, and standardizes unicode symbols.
5. **Overlapping Chunking:** `chunking.py` splits the cleaned text using a sliding window of 800 tokens and 150 tokens overlap. Chunking preserves sentence boundaries to maintain semantic cohesiveness.

---

## 🎯 Verification & Evaluation

We provide a comprehensive verification script `test_rag.py` to evaluate the system performance under a mock context indexing pipeline:
```bash
python test_rag.py
```
The script runs the complete pipeline locally and outputs:
1. Loading speeds of models.
2. Insertion latency into Qdrant.
3. Top retrieved matching chunks with Cosine search vs. BGE reranker relevance.
4. Generated answer and citation validation from Groq.
5. Latency metrics breakdowns for the retrieval, reranking, and generation stages.
