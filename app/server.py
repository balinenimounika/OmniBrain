import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

# Add project root to path so 'app' module can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval.langgraph_integration import retrieval_node
from app.qdrant.client import get_qdrant_client
from app.guardrails.guardrails import should_answer_question, get_guardrail_response, guardrail_status_summary

app = FastAPI(title="OmniBrain API Server")

# Serve the static data/images directory on /images
images_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "images")
os.makedirs(images_dir, exist_ok=True)
app.mount("/images", StaticFiles(directory=images_dir), name="images")

# ==================== REQUEST MODELS ====================

class RetrievalRequest(BaseModel):
    query: str
    route: Optional[str] = "text"
    top_k: Optional[int] = 3

class GuardrailCheckRequest(BaseModel):
    """Request for guardrail validation on retrieval results."""
    query: str
    retrieved_context: str
    similarity_scores: Optional[List[float]] = None
    retrieved_results_count: Optional[int] = 0

class AnswerRequest(BaseModel):
    """Request for combined retrieval + guardrail check + response."""
    query: str
    route: Optional[str] = "text"
    top_k: Optional[int] = 3

@app.post("/retrieve")
def retrieve(req: RetrievalRequest):
    try:
        initial_state = {
            "user_query": req.query,
            "retrieval_mode": req.route,
            "top_k": req.top_k,
            "conversation_state": {},
            "retrieved_text": [],
            "retrieved_images": [],
            "similarity_scores": [],
            "document_name": [],
            "page_number": [],
            "chunk_id": [],
            "image_id": [],
            "source_path": [],
            "retrieval_status": "empty",
            "error_message": ""
        }
        
        updated_state = retrieval_node(initial_state, client=get_qdrant_client())
        
        # Translate source_path to FastAPI HTTP URL for frontend rendering
        # e.g., "data/images/image_025_01.png" -> "http://localhost:8000/images/image_025_01.png"
        host_url = "http://localhost:8000"
        
        def convert_path_to_url(path_val: str) -> str:
            if not path_val:
                return ""
            # If it is already an HTTP URL, return as is
            if path_val.startswith("http://") or path_val.startswith("https://"):
                return path_val
            filename = os.path.basename(path_val)
            return f"{host_url}/images/{filename}"
            
        # Update state results with HTTP URLs
        if "retrieved_images" in updated_state:
            for item in updated_state["retrieved_images"]:
                item["image_path"] = convert_path_to_url(item.get("source_path", ""))
                item["source_path"] = item["image_path"]
                
        if "retrieval_results" in updated_state:
            for item in updated_state["retrieval_results"]:
                if item.get("type") == "image":
                    item["source_path"] = convert_path_to_url(item.get("source_path", ""))
                    item["image_path"] = item["source_path"]
                elif item.get("image_id") and item.get("image_path"):
                    item["image_path"] = convert_path_to_url(item["image_path"])
                    
        # Update list source paths
        if "source_path" in updated_state:
            updated_state["source_path"] = [
                convert_path_to_url(p) if ("data/images" in p or "OIP" in p or p.endswith(".png") or p.endswith(".jpeg") or p.endswith(".jpg")) else p
                for p in updated_state["source_path"]
            ]
            
        return updated_state
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WEEK 3: GUARDRAILS ENDPOINTS ====================

@app.post("/check-guardrail")
def check_guardrail(req: GuardrailCheckRequest):
    """
    Week 3: Validates if a query should be answered based on retrieved context.
    
    NeMo Guardrails check ensures OmniBrain only answers questions within document scope.
    
    Args:
        req: GuardrailCheckRequest with query and retrieved context
        
    Returns:
        Dict with guardrail decision:
        - is_allowed: bool - whether to answer the question
        - reason: str - why decision was made
        - score: float - confidence score (0.0-1.0)
        - context_quality: str - "empty", "low", "medium", "high"
        - message: str or null - block message if not allowed
    """
    try:
        # Create a minimal state for guardrail check
        state = {
            "user_query": req.query,
            "query": req.query,
            "retrieved_context": req.retrieved_context,
            "similarity_scores": req.similarity_scores or [],
            "retrieval_results": [{"score": s} for s in (req.similarity_scores or [])] if req.similarity_scores else [],
            "retrieval_status": "success" if req.retrieved_context else "empty"
        }
        
        # Apply guardrail check
        checked_state = should_answer_question(state)
        
        # Extract guardrail results
        return {
            "is_allowed": checked_state.get("guardrail_allowed", False),
            "reason": checked_state.get("guardrail_reason", "Unknown"),
            "score": checked_state.get("guardrail_score", 0.0),
            "context_quality": checked_state.get("guardrail_context_quality", "empty"),
            "message": checked_state.get("guardrail_message"),
            "status_summary": guardrail_status_summary(checked_state)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/answer")
def answer(req: AnswerRequest):
    """
    Week 3: Combined retrieval + guardrails + response endpoint.
    
    This endpoint:
    1. Retrieves documents from Qdrant
    2. Applies NeMo Guardrails to check if question is within scope
    3. Returns either the answer decision or a block message
    
    If guardrails allow, includes retrieved context for LLM.
    If guardrails block, returns the standard block message.
    
    Args:
        req: AnswerRequest with query, route, and top_k
        
    Returns:
        Dict with retrieval results and guardrail decision:
        - retrieval_results: List of retrieved items
        - guardrail_allowed: bool - whether answer is allowed
        - guardrail_message: str or null - block message if not allowed
        - retrieved_context: str - context for LLM (if allowed)
    """
    try:
        # Step 1: Perform retrieval
        initial_state = {
            "user_query": req.query,
            "retrieval_mode": req.route,
            "top_k": req.top_k,
            "conversation_state": {},
            "retrieved_text": [],
            "retrieved_images": [],
            "similarity_scores": [],
            "document_name": [],
            "page_number": [],
            "chunk_id": [],
            "image_id": [],
            "source_path": [],
            "retrieval_status": "empty",
            "error_message": ""
        }
        
        retrieved_state = retrieval_node(initial_state, client=get_qdrant_client())
        
        # Step 2: Apply guardrails
        checked_state = should_answer_question(retrieved_state)
        
        # Step 3: Convert image paths to URLs
        host_url = "http://localhost:8000"
        
        def convert_path_to_url(path_val: str) -> str:
            if not path_val:
                return ""
            if path_val.startswith("http://") or path_val.startswith("https://"):
                return path_val
            filename = os.path.basename(path_val)
            return f"{host_url}/images/{filename}"
        
        if "retrieved_images" in checked_state:
            for item in checked_state["retrieved_images"]:
                item["image_path"] = convert_path_to_url(item.get("source_path", ""))
                item["source_path"] = item["image_path"]
        
        if "source_path" in checked_state:
            checked_state["source_path"] = [
                convert_path_to_url(p) if ("data/images" in p or "OIP" in p or p.endswith((".png", ".jpeg", ".jpg"))) else p
                for p in checked_state["source_path"]
            ]
        
        # Step 4: Return combined result
        guardrail_message = get_guardrail_response(checked_state)
        
        return {
            "query": req.query,
            "retrieval_results": checked_state.get("retrieval_results", []),
            "retrieved_context": checked_state.get("retrieved_context", ""),
            "retrieved_text": checked_state.get("retrieved_text", []),
            "retrieved_images": checked_state.get("retrieved_images", []),
            "similarity_scores": checked_state.get("similarity_scores", []),
            "retrieval_status": checked_state.get("retrieval_status", "empty"),
            # Week 3 Guardrails
            "guardrail_allowed": checked_state.get("guardrail_allowed", False),
            "guardrail_message": guardrail_message,
            "guardrail_reason": checked_state.get("guardrail_reason", "Unknown"),
            "guardrail_score": checked_state.get("guardrail_score", 0.0),
            "guardrail_context_quality": checked_state.get("guardrail_context_quality", "empty"),
            "status_summary": guardrail_status_summary(checked_state)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
