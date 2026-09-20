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
from app.retrieval.router import route_query
from app.qdrant.client import get_qdrant_client
from app.guardrails.guardrails import (
    should_answer_question,
    get_guardrail_response,
    guardrail_status_summary,
    validate_query_scope,
    check_query_relevance,
    OUT_OF_SCOPE_MESSAGE,
)

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
    route: Optional[str] = None
    top_k: Optional[int] = 3


def build_grounded_response(state: Dict[str, Any]) -> str:
    """Build a response from retrieved document evidence without adding facts."""
    text_results = state.get("retrieved_text") or [
        item for item in (state.get("retrieval_results") or [])
        if item.get("type") == "text" or item.get("modality") == "text"
    ]
    if text_results:
        evidence = []
        for item in text_results:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            evidence.append(
                f"{content} (Document: {item.get('document_name', 'Unknown')}, "
                f"Page: {item.get('page_number', -1)})"
            )
        if evidence:
            return "Based on the provided document:\n\n" + "\n\n".join(evidence)

    image_results = state.get("retrieved_images") or []
    if image_results:
        references = [
            f"{item.get('image_id') or 'Image'} (Document: {item.get('document_name', 'Unknown')}, "
            f"Page: {item.get('page_number', -1)})"
            for item in image_results
        ]
        return "Matching document images:\n\n" + "\n".join(references)

    return ""

@app.post("/retrieve")
def retrieve(req: RetrievalRequest):
    try:
        route = route_query(req.query, req.route)
        client = get_qdrant_client()
        query_allowed, query_message, guardrail_score = check_query_relevance(req.query, client=client, route=route)

        if not query_allowed:
            return {
                "query": req.query,
                "user_query": req.query,
                "route": route,
                "selected_route": route,
                "retrieval_results": [],
                "retrieved_context": "",
                "retrieved_text": [],
                "retrieved_images": [],
                "similarity_scores": [],
                "retrieval_status": "skipped",
                "retrieval_skipped": True,
                "error_message": query_message,
                "guardrail_allowed": False,
                "guardrail_message": query_message,
                "guardrail_status": "BLOCKED",
                "guardrail_score": round(float(guardrail_score), 4),
                "retrieved_text_results": [],
                "retrieved_image_results": [],
            }

        initial_state = {
            "user_query": req.query,
            "query": req.query,
            "route": route,
            "selected_route": route,
            "retrieval_mode": route,
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
            "retrieval_status": "started",
            "retrieval_skipped": False,
            "guardrail_allowed": True,
            "guardrail_status": "ALLOWED",
            "guardrail_score": round(float(guardrail_score), 4),
            "guardrail_message": None,
            "error_message": ""
        }
        
        updated_state = retrieval_node(initial_state, client=client)
        updated_state["selected_route"] = route
        updated_state["guardrail_allowed"] = True
        updated_state["guardrail_status"] = "ALLOWED"
        updated_state["guardrail_score"] = round(float(guardrail_score), 4)
        updated_state["guardrail_message"] = None
        updated_state["retrieval_skipped"] = False
        updated_state["retrieved_text_results"] = updated_state.get("retrieved_text", [])
        updated_state["retrieved_image_results"] = updated_state.get("retrieved_images", [])

        
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
        route = route_query(req.query, req.route)
        client = get_qdrant_client()
        query_allowed, query_message, guardrail_score = check_query_relevance(req.query, client=client, route=route)
        if not query_allowed:
            checked_state = {
                "user_query": req.query,
                "selected_route": route,
                "guardrail_allowed": False,
                "guardrail_message": query_message,
                "guardrail_reason": "Empty query" if not req.query.strip() else "Out of scope query",
                "guardrail_score": round(float(guardrail_score), 4),
                "guardrail_context_quality": "empty",
            }
            return {
                "query": req.query,
                "user_query": req.query,
                "route": route,
                "selected_route": route,
                "retrieval_results": [],
                "retrieved_context": "",
                "retrieved_text": [],
                "retrieved_images": [],
                "similarity_scores": [],
                "retrieval_status": "skipped",
                "retrieval_skipped": True,
                "guardrail_allowed": False,
                "guardrail_message": query_message,
                "guardrail_reason": checked_state["guardrail_reason"],
                "guardrail_score": round(float(guardrail_score), 4),
                "guardrail_context_quality": "empty",
                "guardrail_status": "BLOCKED",
                "retrieved_text_results": [],
                "retrieved_image_results": [],
                "answer": query_message,
                "status_summary": guardrail_status_summary(checked_state),
            }

        # Step 1: Supervisor route, then LangGraph retrieval node.
        initial_state = {
            "user_query": req.query,
            "query": req.query,
            "route": route,
            "selected_route": route,
            "retrieval_mode": route,
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
            "retrieval_status": "started",
            "retrieval_skipped": False,
            "guardrail_allowed": True,
            "guardrail_status": "ALLOWED",
            "guardrail_score": round(float(guardrail_score), 4),
            "guardrail_message": None,
            "error_message": ""
        }
        
        retrieved_state = retrieval_node(initial_state, client=client)
        
        # Step 2: Apply guardrails
        checked_state = should_answer_question(retrieved_state, client=client)
        
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
        final_allowed = checked_state.get("guardrail_allowed", False)
        
        return {
            "query": req.query,
            "user_query": req.query,
            "route": checked_state.get("retrieval_mode", route),
            "selected_route": route,
            "retrieval_results": checked_state.get("retrieval_results", []),
            "retrieved_context": checked_state.get("retrieved_context", ""),
            "retrieved_text": checked_state.get("retrieved_text", []),
            "retrieved_images": checked_state.get("retrieved_images", []),
            "retrieved_text_results": checked_state.get("retrieved_text", []),
            "retrieved_image_results": checked_state.get("retrieved_images", []),
            "similarity_scores": checked_state.get("similarity_scores", []),
            "retrieval_status": checked_state.get("retrieval_status", "empty"),
            "retrieval_skipped": False,
            # Guardrails info
            "guardrail_allowed": final_allowed,
            "guardrail_message": guardrail_message,
            "guardrail_reason": checked_state.get("guardrail_reason", "Unknown"),
            "guardrail_score": checked_state.get("guardrail_score", round(float(guardrail_score), 4)),
            "guardrail_context_quality": checked_state.get("guardrail_context_quality", "empty"),
            "guardrail_status": "ALLOWED" if final_allowed else "BLOCKED",
            "answer": build_grounded_response(checked_state) if final_allowed else (guardrail_message or OUT_OF_SCOPE_MESSAGE),
            "status_summary": guardrail_status_summary(checked_state)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
