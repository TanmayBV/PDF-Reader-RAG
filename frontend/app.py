import os
import time
import requests
import streamlit as st
from typing import Dict, List, Any

# Configure Streamlit page layout and styling
st.set_page_config(
    page_title="AetherRAG - Large PDF Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Custom CSS for modern premium styling (dark-ish elements, nice cards)
st.markdown("""
<style>
    /* Main container tweaks */
    .reportview-container {
        background: #0f1116;
    }
    
    /* Premium Metric Card */
    .metric-card {
        background-color: #1a1e28;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-title {
        color: #8a92a6;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .metric-value {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 700;
    }
    .metric-unit {
        font-size: 0.9rem;
        font-weight: normal;
        color: #4ade80;
        margin-left: 2px;
    }

    /* Retrieved Chunk Card */
    .chunk-card {
        background-color: #161a23;
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 12px;
        margin-bottom: 12px;
        font-size: 0.9rem;
    }
    .chunk-meta {
        font-weight: bold;
        color: #60a5fa;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
    }
    .chunk-scores {
        color: #a78bfa;
        font-size: 0.8rem;
    }
    .chunk-text {
        color: #e2e8f0;
        font-style: italic;
        line-height: 1.4;
    }
    
    /* Citations list styling */
    .citation-tag {
        display: inline-block;
        background-color: #1e293b;
        color: #38bdf8;
        border: 1px solid #0284c7;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to perform backend API health check
def check_backend_health() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/api/health", timeout=3)
        if r.status_code == 200:
            return r.json().get("vector_db") == "connected"
    except Exception:
        pass
    return False

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_query_telemetry" not in st.session_state:
    # Keeps track of the currently selected message's retrieval metadata
    st.session_state.selected_query_telemetry = None
if "ingesting_jobs" not in st.session_state:
    st.session_state.ingesting_jobs = []

# App Header
st.title("🧠 AetherRAG Cognitive PDF Search Engine")
st.caption("Production-grade Retrieval-Augmented Generation using BGE-small embeddings, Qdrant HNSW ANN indices, and Groq LLM.")

# Side Bar for Configuration & Document Ingestion
with st.sidebar:
    st.header("⚙️ Document Control")
    
    # 1. Health Status
    is_healthy = check_backend_health()
    if is_healthy:
        st.success("🟢 Connected to Qdrant & Backend")
    else:
        st.error("🔴 Connection Error: Check API/Database")
        
    st.markdown("---")
    
    # 2. PDF Upload Section
    st.subheader("📁 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Select PDF files to ingest",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload native text or scanned image PDFs. Large documents (>200 pages) supported."
    )
    
    if uploaded_files:
        if st.button("🚀 Upload and Index Documents", disabled=not is_healthy):
            for uploaded_file in uploaded_files:
                with st.spinner(f"Uploading {uploaded_file.name}..."):
                    # 1. POST file to upload endpoint
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    try:
                        up_res = requests.post(f"{BACKEND_URL}/api/upload", files=files)
                        if up_res.status_code == 200:
                            # 2. Trigger async background ingestion
                            filename = up_res.json()["filename"]
                            ing_res = requests.post(
                                f"{BACKEND_URL}/api/ingest", 
                                json={"filename": filename}
                            )
                            if ing_res.status_code == 200:
                                ing_id = ing_res.json()["ingestion_id"]
                                st.session_state.ingesting_jobs.append({
                                    "id": ing_id,
                                    "filename": filename,
                                    "status": "queued",
                                    "progress": 0,
                                    "message": "Initializing..."
                                })
                                st.info(f"Queued ingestion for {filename}")
                            else:
                                st.error(f"Failed to queue ingestion: {ing_res.text}")
                        else:
                            st.error(f"Failed to upload: {up_res.text}")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
                        
    # 3. Dynamic Progress Bars for Active Ingestions
    if st.session_state.ingesting_jobs:
        st.markdown("---")
        st.subheader("⚡ Processing Pipelines")
        
        still_running = []
        for job in st.session_state.ingesting_jobs:
            job_id = job["id"]
            filename = job["filename"]
            
            try:
                # Poll status
                res = requests.get(f"{BACKEND_URL}/api/ingest/status/{job_id}")
                if res.status_code == 200:
                    status_data = res.json()
                    status = status_data.get("status", "queued")
                    progress = status_data.get("progress", 0)
                    msg = status_data.get("message", "Processing...")
                    
                    st.text(f"📄 {filename}")
                    st.progress(progress / 100.0)
                    st.caption(f"Status: **{status}** | *{msg}*")
                    
                    # Update job dict in state
                    job["status"] = status
                    job["progress"] = progress
                    job["message"] = msg
                    
                    if status in ["queued", "processing"]:
                        still_running.append(job)
                    elif status == "completed":
                        st.toast(f"✅ Ingested {filename}! {status_data.get('chunks_count', 0)} chunks created.")
                    elif status == "failed":
                        st.error(f"❌ {filename} failed: {msg}")
                else:
                    st.caption(f"⚠️ {filename}: Fetch error")
            except Exception as e:
                st.caption(f"⚠️ {filename}: {e}")
                still_running.append(job)
                
        # Keep only active jobs in session state for next rerun
        st.session_state.ingesting_jobs = still_running
        
        # If any jobs are still running, rerun the Streamlit script in 2 seconds to refresh progress
        if still_running:
            time.sleep(2)
            st.rerun()

# Split Main UI Layout: 60% Chat, 40% Telemetry
col_chat, col_telemetry = st.columns([0.6, 0.4])

# ==================== COLUMN 1: CHAT INTERFACE ====================
with col_chat:
    st.subheader("💬 Cognitive Dialogue")
    
    # Display Chat History
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                st.markdown("**Cited Sources:**")
                citation_html = "".join([f'<span class="citation-tag">{c}</span>' for c in msg["citations"]])
                st.markdown(citation_html, unsafe_allow_html=True)
                
                # Button to select this message's retrieval metadata for the Telemetry panel
                if st.button("🔍 View Search Details", key=f"telemetry_btn_{idx}"):
                    st.session_state.selected_query_telemetry = {
                        "query": msg.get("query", ""),
                        "latencies": msg.get("latencies", {}),
                        "chunks": msg.get("chunks", [])
                    }
                    st.rerun()

    # User Query Input
    if user_query := st.chat_input("Ask a question about your ingested documents...", disabled=not is_healthy):
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # 2. Make Request to Backend API
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            spinner_placeholder = st.spinner("Retrieving, Reranking, and Generating...")
            
            with spinner_placeholder:
                try:
                    payload = {"query": user_query}
                    r = requests.post(f"{BACKEND_URL}/api/query", json=payload, timeout=60)
                    if r.status_code == 200:
                        data = r.json()
                        answer = data["answer"]
                        citations = data["citations"]
                        chunks = data["chunks"]
                        latencies = data["latencies"]
                        
                        # Display answer
                        response_placeholder.markdown(answer)
                        
                        # Display citations tags
                        if citations:
                            st.markdown("**Cited Sources:**")
                            citation_html = "".join([f'<span class="citation-tag">{c}</span>' for c in citations])
                            st.markdown(citation_html, unsafe_allow_html=True)
                            
                        # Save assistant response to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "query": user_query,
                            "citations": citations,
                            "chunks": chunks,
                            "latencies": latencies
                        })
                        
                        # Automatically select this query for side-by-side Telemetry view
                        st.session_state.selected_query_telemetry = {
                            "query": user_query,
                            "latencies": latencies,
                            "chunks": chunks
                        }
                        
                    else:
                        response_placeholder.error(f"Failed to query backend (HTTP {r.status_code}): {r.text}")
                except Exception as e:
                    response_placeholder.error(f"Error connecting to backend: {e}")
                    
        # Trigger page rerun to show updated conversation and telemetry
        st.rerun()

# ==================== COLUMN 2: COGNITIVE TELEMETRY ====================
with col_telemetry:
    st.subheader("📊 Cognitive Telemetry & Provenance")
    
    tel = st.session_state.selected_query_telemetry
    
    if tel is None:
        st.info("Ask a question in the dialogue pane or select 'View Search Details' to visualize RAG pipeline telemetry and source chunks.")
    else:
        st.markdown(f"**Inspecting query:** *\"{tel['query']}\"*")
        
        # 1. Latency Metrics Dashboard
        lat = tel["latencies"]
        st.markdown("#### ⚡ Latency Breakdown")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Vector Retr</div>
                <div class="metric-value">{lat.get('retrieval_ms', 0)}<span class="metric-unit">ms</span></div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Cross Rerank</div>
                <div class="metric-value">{lat.get('rerank_ms', 0)}<span class="metric-unit">ms</span></div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">LLM Gen</div>
                <div class="metric-value">{lat.get('generation_ms', 0)}<span class="metric-unit">ms</span></div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">End-to-End</div>
                <div class="metric-value">{lat.get('total_ms', 0)}<span class="metric-unit">ms</span></div>
            </div>
            """, unsafe_allow_html=True)
            
        # Target latency checker
        total_s = lat.get('total_ms', 0) / 1000.0
        if 2.0 <= total_s <= 5.0:
            st.success(f"🎯 Latency Target Achieved: {total_s:.2f}s (Target: 2–5s)")
        elif total_s < 2.0:
            st.success(f"⚡ Superfast Retrieval: {total_s:.2f}s (Below 2s target!)")
        else:
            st.warning(f"⚠️ Latency Target Missed: {total_s:.2f}s (Above 5s target)")
            
        st.markdown("---")
        
        # 2. Retrieved Chunks Visualization
        st.markdown("#### 📄 Top Retrieved Chunks (Reranked Context)")
        
        chunks = tel["chunks"]
        if not chunks:
            st.warning("No context chunks retrieved.")
        else:
            for rank, chunk in enumerate(chunks):
                meta = chunk.get("metadata", {})
                filename = meta.get("filename", "Unknown")
                page = meta.get("page_number", "Unknown")
                score = chunk.get("score", 0.0)
                rerank_score = chunk.get("rerank_score", 0.0)
                text = chunk.get("text", "")
                
                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-meta">
                        <span>Rank {rank + 1} | {filename} (Page {page})</span>
                    </div>
                    <div class="chunk-scores">
                        🔍 Vector Cosine Similarity: <b>{score:.4f}</b><br/>
                        🧬 BGE Rerank Score: <b>{rerank_score:.4f}</b>
                    </div>
                    <hr style="margin: 6px 0; border: 0; border-top: 1px solid #2d3139;"/>
                    <div class="chunk-text">"{text}"</div>
                </div>
                """, unsafe_allow_html=True)
