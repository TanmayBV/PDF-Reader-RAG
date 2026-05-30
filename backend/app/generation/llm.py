import os
import logging
import httpx
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"

def get_groq_api_key() -> str:
    """
    Retrieves the Groq API key from environment variables.
    Handles potential typo GORQ_API_KEY.
    """
    key = os.getenv("GROQ_API_KEY") or os.getenv("GORQ_API_KEY")
    if not key:
        logger.error("Groq API Key is not set in environment variables.")
        raise ValueError("GROQ_API_KEY or GORQ_API_KEY is not configured in .env file.")
    return key.strip()

async def generate_answer(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL
) -> Tuple[str, List[str]]:
    """
    Synthesizes an answer using Groq API conditioned strictly on the retrieved chunks.
    
    Args:
        query: User query string.
        retrieved_chunks: Top reranked chunks containing text and metadata.
        model_name: Groq model to use. Default llama-3.1-8b-instant.
        
    Returns:
        A tuple of (generated_answer_string, list_of_citations_used).
    """
    api_key = get_groq_api_key()
    
    if not retrieved_chunks:
        return "Information not found in documents.", []
        
    # 1. Format the retrieved context for the prompt
    context_str_list = []
    citations_mapping = {}  # Map a simple identifier to the full citation
    
    for idx, chunk in enumerate(retrieved_chunks):
        meta = chunk.get("metadata", {})
        filename = meta.get("filename", "Unknown")
        page = meta.get("page_number", "Unknown")
        text = chunk.get("text", "")
        
        citation_label = f"{filename} (Page {page})"
        context_str_list.append(f"--- CONTEXT CHUNK {idx + 1} (Source: {citation_label}) ---\n{text}\n")
        
    context_data = "\n".join(context_str_list)
    
    # 2. Build the system prompt enforcing requirements
    system_prompt = (
        "You are an advanced RAG Chatbot assistant. Your goal is to answer the user's question "
        "using ONLY the provided Context Chunks. Do NOT use any external or background knowledge.\n\n"
        "Rules:\n"
        "1. Answer the question based STRICTLY and ONLY on the facts present in the Context Chunks. "
        "If the answer cannot be found in the Context Chunks, you must reply exactly and only: "
        "\"Information not found in documents.\"\n"
        "2. Avoid any speculation, assumption, or hallucination. If a detail is missing, do not invent it.\n"
        "3. For every sentence or fact you write, you MUST cite the source document and page number "
        "at the end of that sentence or phrase. Use the format: filename.pdf (Page X). "
        "Never cite sources that are not in the Context Chunks.\n"
        "4. Be concise, direct, and factual."
    )
    
    user_prompt = f"Context Chunks:\n{context_data}\n\nQuestion: {query}\n\nAnswer:"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.4,  # Minimize creativity to prevent hallucination
        "max_tokens": 1024
    }
    
    logger.info(f"Sending async chat completion request to Groq ({model_name})")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(GROQ_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            
            answer = response_json["choices"][0]["message"]["content"].strip()
            
            # 3. Extract unique citations that are referenced in the answer
            citations = []
            for chunk in retrieved_chunks:
                meta = chunk.get("metadata", {})
                filename = meta.get("filename", "")
                page = meta.get("page_number", "")
                if filename:
                    cit_str = f"{filename} (Page {page})"
                    # Check if the generated answer references this file or page
                    if cit_str in answer or filename in answer:
                        if cit_str not in citations:
                            citations.append(cit_str)
                            
            logger.info(f"Groq API call succeeded. Answer length: {len(answer)}. Citations extracted: {len(citations)}")
            return answer, citations
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Groq HTTP status error: {e.response.status_code} - {e.response.text}", exc_info=True)
            return f"Error: Failed to contact the language model service (HTTP {e.response.status_code}).", []
        except Exception as e:
            logger.error(f"Failed to generate answer from Groq: {e}", exc_info=True)
            return f"Error: An unexpected failure occurred in the generation pipeline: {e}", []
