import os
import logging
from pathlib import Path
from typing import List, Optional, Union
from qdrant_client import QdrantClient

from app.qdrant.client import get_qdrant_client
from app.qdrant.collections import TEXT_COLLECTION, IMAGE_COLLECTION
from app.qdrant.search import (
    _deduplicate_image_results,
    search_text_similarity,
    search_image_similarity,
)
from app.embeddings.image_embeddings import get_image_model
from app.retrieval.models import RetrievalResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def resolve_stored_image_path(stored_path: Optional[str]) -> Optional[str]:
    if not isinstance(stored_path, str) or not stored_path.strip():
        return None
    normalized_path = stored_path.replace("\\", "/")
    path = Path(normalized_path)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / normalized_path]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return stored_path
    return None

def find_image_metadata(client: QdrantClient, image_id: Optional[str]):
    if not image_id:
        return None, None
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        points, _ = client.scroll(
            collection_name=IMAGE_COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="image_id", match=MatchValue(value=image_id))
            ]),
            limit=1
        )
        if not points:
            return None, None
        payload = points[0].payload or {}
        if payload.get("image_id") != image_id:
            return None, None
        image_path = resolve_stored_image_path(
            payload.get("image_path") or payload.get("source_path")
        )
        return image_id if image_path else None, image_path
    except Exception:
        return None, None

def map_text_hit(
    hit_id: Union[str, int],
    score: float,
    payload: Optional[dict],
    image_id: Optional[str] = None,
    image_path: Optional[str] = None
) -> RetrievalResult:
    """
    Safely maps raw Qdrant text hit payload to structured RetrievalResult.
    Handles missing metadata elements gracefully.
    """
    payload = payload or {}
    return RetrievalResult(
        id=hit_id,
        modality="text",
        score=score,
        document_name=payload.get("document_name", "Unknown"),
        page_number=payload.get("page_number", -1),
        chunk_id=payload.get("chunk_id"),
        image_id=image_id if image_id is not None else payload.get("image_id"),
        source_path=payload.get("source_path", "Unknown"),
        content=payload.get("text", ""),
        image_path=image_path
    )

def map_image_hit(hit_id: Union[str, int], score: float, payload: Optional[dict]) -> RetrievalResult:
    """
    Safely maps raw Qdrant image hit payload to structured RetrievalResult.
    Handles missing metadata elements gracefully.
    """
    payload = payload or {}
    return RetrievalResult(
        id=hit_id,
        modality="image",
        score=score,
        document_name=payload.get("document_name", "Unknown"),
        page_number=payload.get("page_number", -1),
        image_id=payload.get("image_id") or None,
        source_path=payload.get("source_path", "Unknown"),
        image_path=resolve_stored_image_path(
            payload.get("image_path") or payload.get("source_path")
        )
    )

def retrieve_text(query: str, top_k: int = 3, client: Optional[QdrantClient] = None) -> List[RetrievalResult]:
    """
    Retrieves relevant text chunks from the vector database using query text.
    
    Args:
        query (str): The search query.
        top_k (int): Number of top results to return.
        client (QdrantClient, optional): The Qdrant client instance.
        
    Returns:
        List[RetrievalResult]: List of structured text results.
    """
    if not isinstance(query, str) or not query.strip():
        logger.error("Empty or invalid query passed to retrieve_text.")
        raise ValueError("Query must be a non-empty string.")
        
    if client is None:
        client = get_qdrant_client()
        
    try:
        logger.info(f"Retrieving text results for query: '{query}' (top_k={top_k})")
        raw_results = search_text_similarity(client, query, top_k=top_k)
        
        results = []
        for item in raw_results:
            payload = item["payload"].copy() if item.get("payload") else {}
            image_id, image_path = find_image_metadata(client, payload.get("image_id"))
            results.append(map_text_hit(
                item["id"], item["score"], payload,
                image_id=image_id,
                image_path=image_path
            ))
        return results
    except Exception as e:
        logger.error(f"Error in retrieve_text: {e}")
        raise RuntimeError(f"Text retrieval failed: {e}")

def retrieve_images(query: str, top_k: int = 3, client: Optional[QdrantClient] = None) -> List[RetrievalResult]:
    """
    Retrieves relevant images from the vector database.
    Supports:
      - Image-to-Image search: if query matches an existing image file path.
      - Text-to-Image search: if query is a text string (using CLIP text embedding).
      
    Args:
        query (str): The search query (text description or image path).
        top_k (int): Number of top results to return.
        client (QdrantClient, optional): The Qdrant client instance.
        
    Returns:
        List[RetrievalResult]: List of structured image results.
    """
    if not isinstance(query, str) or not query.strip():
        logger.error("Empty or invalid query passed to retrieve_images.")
        raise ValueError("Query must be a non-empty string.")
        
    if client is None:
        client = get_qdrant_client()
        
    try:
        # Check if query is a local file path that exists (image query)
        is_image_file = os.path.exists(query) and os.path.isfile(query)
        
        if is_image_file:
            logger.info(f"Retrieving images via image-to-image similarity search for: '{query}' (top_k={top_k})")
            raw_results = search_image_similarity(client, query, top_k=top_k)
            results = []
            for item in raw_results:
                results.append(map_image_hit(item["id"], item["score"], item["payload"]))
            return results
        else:
            logger.info(f"Retrieving images via text-to-image similarity search for: '{query}' (top_k={top_k})")
            # Text-to-image search using CLIP text encoding
            model = get_image_model()
            query_vector = model.encode(query)
            if hasattr(query_vector, "tolist"):
                query_vector = query_vector.tolist()
            elif not isinstance(query_vector, list):
                query_vector = list(query_vector)
            
            # Query the Qdrant image collection directly using query_points API
            try:
                collection_count = int(client.count(collection_name=IMAGE_COLLECTION).count)
            except (AttributeError, TypeError, ValueError):
                collection_count = top_k
            response = client.query_points(
                collection_name=IMAGE_COLLECTION,
                query=query_vector,
                limit=max(top_k, collection_count)
            )
            raw_results = []
            for hit in response.points:
                raw_results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                })
            unique_results = _deduplicate_image_results(raw_results, top_k)
            return [
                map_image_hit(item["id"], item["score"], item["payload"])
                for item in unique_results
            ]
    except Exception as e:
        logger.error(f"Error in retrieve_images: {e}")
        raise RuntimeError(f"Image retrieval failed: {e}")

def retrieve_multimodal(query: str, top_k: int = 3, client: Optional[QdrantClient] = None) -> List[RetrievalResult]:
    """
    Retrieves both relevant text chunks and images and returns them sorted by similarity score.
    
    Args:
        query (str): The search query.
        top_k (int): Number of top results to return.
        client (QdrantClient, optional): The Qdrant client instance.
        
    Returns:
        List[RetrievalResult]: Unified list of structured retrieval results.
    """
    if not isinstance(query, str) or not query.strip():
        logger.error("Empty or invalid query passed to retrieve_multimodal.")
        raise ValueError("Query must be a non-empty string.")
        
    if client is None:
        client = get_qdrant_client()
        
    try:
        logger.info(f"Retrieving multimodal results for query: '{query}' (top_k={top_k})")
        # Fetch results from both spaces
        text_results = retrieve_text(query, top_k=top_k, client=client)
        image_results = retrieve_images(query, top_k=top_k, client=client)
        
        # Merge and sort by score descending
        combined = text_results + image_results
        combined.sort(key=lambda x: x.score, reverse=True)
        
        return combined[:top_k]
    except Exception as e:
        logger.error(f"Error in retrieve_multimodal: {e}")
        raise RuntimeError(f"Multimodal retrieval failed: {e}")
