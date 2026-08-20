import logging
from typing import TypedDict, List, Dict, Any, Optional

from app.retrieval.services import retrieve_text, retrieve_images, retrieve_multimodal
from app.retrieval.models import RetrievalResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalState(TypedDict, total=False):
    """
    LangGraph State Schema matching the Supervisor flow expectations.
    Uses TypedDict for serializability and compatibility with LangGraph.
    """
    # Inputs
    query: str
    user_query: str
    route: str  # "text", "image", or "multimodal"
    retrieval_mode: str  # "text", "image", or "multimodal"
    limit: Optional[int]
    top_k: Optional[int]
    
    # Outputs
    retrieved_text: Optional[List[Dict[str, Any]]]
    retrieved_images: Optional[List[Dict[str, Any]]]
    retrieval_results: Optional[List[Dict[str, Any]]]
    retrieved_context: Optional[str]
    similarity_scores: Optional[List[float]]
    document_name: Optional[List[str]]
    page_number: Optional[List[int]]
    chunk_id: Optional[List[str]]
    image_id: Optional[List[str]]
    source_path: Optional[List[str]]
    retrieval_status: Optional[str]  # "success", "empty", or "error"
    error_message: Optional[str]
    
    # Context
    conversation_state: Optional[Dict[str, Any]]

def serialize_result(res: RetrievalResult) -> Dict[str, Any]:
    """
    Converts Pydantic model to plain JSON-serializable dictionary.
    Supports both Pydantic v1 and v2.
    """
    if hasattr(res, "model_dump"):
        return res.model_dump()
    return res.dict()

def retrieval_node(state: RetrievalState, client=None) -> Dict[str, Any]:
    """
    LangGraph node that performs vector database retrieval.
    Routes queries to the appropriate search service (text, image, multimodal)
    based on the state's routing information, then returns state updates.
    
    Ensures absolute safety by returning status and error messages instead of crashing.
    
    Args:
        state (RetrievalState): Current LangGraph state.
        
    Returns:
        Dict[str, Any]: State update containing retrieved results, metadata lists, and status.
    """
    # Extract query with user_query fallback
    query = state.get("user_query") or state.get("query", "")
    query = query.strip()
    
    # Extract route with retrieval_mode fallback
    route = state.get("retrieval_mode") or state.get("route", "text")
    route = route.strip().lower()
    
    # Extract limit with top_k fallback
    limit = state.get("top_k") or state.get("limit", 3)
    if limit is None:
        limit = 3
        
    logger.info(f"LangGraph Retrieval Node. Route: '{route}', Query: '{query}', Limit: {limit}")
    
    # Clean initial return structure in case of validation issues
    empty_result_state = {
        "query": query,
        "user_query": query,
        "route": route,
        "retrieval_mode": route,
        "limit": limit,
        "top_k": limit,
        "retrieved_text": [],
        "retrieved_images": [],
        "retrieval_results": [],
        "retrieved_context": "",
        "similarity_scores": [],
        "document_name": [],
        "page_number": [],
        "chunk_id": [],
        "image_id": [],
        "source_path": [],
        "retrieval_status": "empty",
        "error_message": ""
    }
    
    # Handle empty/invalid queries gracefully without crashing the pipeline
    if not query:
        logger.warning("Empty query received in LangGraph retrieval node.")
        empty_result_state["error_message"] = "Empty or invalid query provided."
        return empty_result_state
        
    try:
        # Perform retrieval based on route/mode
        if route == "image":
            results = retrieve_images(query, top_k=limit, client=client)
        elif route == "multimodal":
            results = retrieve_multimodal(query, top_k=limit, client=client)
        elif route == "text":
            results = retrieve_text(query, top_k=limit, client=client)
        else:
            logger.warning(f"Unknown route '{route}' received. Defaulting to text retrieval.")
            results = retrieve_text(query, top_k=limit, client=client)
            
        # Serialize results for state storage
        serialized = [serialize_result(r) for r in results]
        
        # Segment into retrieved_text and retrieved_images for specific state fields
        retrieved_text = [r for r in serialized if r.get("type") == "text"]
        retrieved_images = [r for r in serialized if r.get("type") == "image"]
        
        # Gather list attributes for state metadata arrays
        scores = [r.get("score", 0.0) for r in serialized]
        doc_names = [r.get("document_name", "Unknown") for r in serialized]
        page_nums = [r.get("page_number", -1) for r in serialized]
        chunk_ids = [r.get("chunk_id") for r in serialized if r.get("chunk_id")]
        image_ids = [r.get("image_id") for r in serialized if r.get("image_id")]
        source_paths = [r.get("source_path", "Unknown") for r in serialized]
        
        # Build text-based context string for LLM ingestion
        context_blocks = []
        for idx, res in enumerate(results, 1):
            if res.modality == "text":
                context_blocks.append(
                    f"[{idx}] Text Chunk (Document: {res.document_name}, Page: {res.page_number}):\n{res.content}"
                )
            elif res.modality == "image":
                context_blocks.append(
                    f"[{idx}] Image Reference (Document: {res.document_name}, Page: {res.page_number}, Source: {res.source_path})"
                )
        retrieved_context = "\n\n".join(context_blocks)
        
        status = "success" if results else "empty"
        logger.info(f"Retrieval Node completed. Status: {status}, Hits: {len(results)}")
        
        return {
            "query": query,
            "user_query": query,
            "route": route,
            "retrieval_mode": route,
            "limit": limit,
            "top_k": limit,
            "retrieved_text": retrieved_text,
            "retrieved_images": retrieved_images,
            "retrieval_results": serialized,
            "retrieved_context": retrieved_context,
            "similarity_scores": scores,
            "document_name": doc_names,
            "page_number": page_nums,
            "chunk_id": chunk_ids,
            "image_id": image_ids,
            "source_path": source_paths,
            "retrieval_status": status,
            "error_message": ""
        }
        
    except Exception as e:
        logger.error(f"Error during retrieval node execution: {e}")
        # Crash safety: return status "error" and log error_message in state instead of raising
        err_state = empty_result_state.copy()
        err_state["retrieval_status"] = "error"
        err_state["error_message"] = str(e)
        return err_state


