import logging
import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from app.embeddings.text_embeddings import generate_text_embedding
from app.embeddings.image_embeddings import generate_image_embedding
from app.qdrant.collections import TEXT_COLLECTION, IMAGE_COLLECTION

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _normalized_source_path(source_path):
    if not isinstance(source_path, str) or not source_path.strip():
        return None
    return os.path.normcase(os.path.normpath(source_path.replace("\\", "/")))

def _image_result_key(result: Dict[str, Any]):
    payload = result.get("payload") or {}
    image_id = payload.get("image_id")
    if image_id:
        return ("image_id", image_id)
    source_path = _normalized_source_path(payload.get("source_path"))
    if source_path:
        return ("source_path", source_path)
    return ("point_id", result.get("id"))

def _deduplicate_image_results(results: List[Dict[str, Any]], top_k: int):
    unique_results = {}
    for result in results:
        key = _image_result_key(result)
        current = unique_results.get(key)
        if current is None or result.get("score", 0.0) > current.get("score", 0.0):
            unique_results[key] = result
    return sorted(
        unique_results.values(),
        key=lambda result: result.get("score", 0.0),
        reverse=True
    )[:top_k]

def _deduplicate_text_results(results: List[Dict[str, Any]], top_k: int):
    unique_results = {}
    for result in results:
        payload = result.get("payload") or {}
        key = (
            payload.get("document_name"),
            payload.get("page_number"),
            payload.get("chunk_id"),
            payload.get("text", payload.get("content"))
        )
        current = unique_results.get(key)
        if current is None or result.get("score", 0.0) > current.get("score", 0.0):
            unique_results[key] = result
    return sorted(
        unique_results.values(),
        key=lambda result: result.get("score", 0.0),
        reverse=True
    )[:top_k]

def search_text_similarity(
    client: QdrantClient,
    query_text: str,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Generates embedding for query_text, performs similarity search in 'omnibrain_text',
    and returns matches with score and metadata.
    
    Args:
        client (QdrantClient): Reusable Qdrant Client.
        query_text (str): Input text query.
        top_k (int): Number of top results to return.
        
    Returns:
        List[Dict[str, Any]]: List of dictionary results containing 'id', 'score', and 'payload'.
    """
    if not isinstance(query_text, str) or not query_text.strip():
        logger.error("Empty or invalid query_text passed to text search.")
        raise ValueError("Query text must be a non-empty string.")
        
    try:
        # 1. Generate text embedding on-the-fly
        logger.info(f"Generating embedding for text query: '{query_text}'")
        query_vector = generate_text_embedding(query_text)
        
        # 2. Query Qdrant text collection using modern query_points API
        logger.info(f"Querying Qdrant '{TEXT_COLLECTION}' collection for top {top_k} matches...")
        try:
            collection_count = int(client.count(collection_name=TEXT_COLLECTION).count)
        except (AttributeError, TypeError, ValueError):
            collection_count = top_k
        response = client.query_points(
            collection_name=TEXT_COLLECTION,
            query=query_vector,
            limit=max(top_k, collection_count)
        )
        
        # 3. Format hits
        results = []
        for hit in response.points:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
            
        results = _deduplicate_text_results(results, top_k)
        logger.info(f"Text search complete. Retrieved {len(results)} unique hits.")
        return results
        
    except Exception as e:
        logger.error(f"Error executing text similarity search: {e}")
        raise RuntimeError(f"Text similarity search failure: {e}")


def search_image_similarity(
    client: QdrantClient,
    query_image_path: str,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Generates embedding for query_image_path, performs similarity search in 'omnibrain_images',
    and returns matches with score and metadata.
    
    Args:
        client (QdrantClient): Reusable Qdrant Client.
        query_image_path (str): Filepath of the query image.
        top_k (int): Number of top results to return.
        
    Returns:
        List[Dict[str, Any]]: List of dictionary results containing 'id', 'score', and 'payload'.
    """
    if not isinstance(query_image_path, str) or not query_image_path.strip():
        logger.error("Empty or invalid query_image_path passed to image search.")
        raise ValueError("Query image path must be a non-empty string.")
        
    try:
        # 1. Generate image embedding on-the-fly
        logger.info(f"Generating embedding for image query: '{query_image_path}'")
        query_vector = generate_image_embedding(query_image_path)
        
        # 2. Query Qdrant image collection using modern query_points API
        logger.info(f"Querying Qdrant '{IMAGE_COLLECTION}' collection for top {top_k} matches...")
        try:
            collection_count = int(client.count(collection_name=IMAGE_COLLECTION).count)
        except (AttributeError, TypeError, ValueError):
            collection_count = top_k
        response = client.query_points(
            collection_name=IMAGE_COLLECTION,
            query=query_vector,
            limit=max(top_k, collection_count)
        )
        
        # 3. Format hits
        results = []
        for hit in response.points:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
            
        results = _deduplicate_image_results(results, top_k)
        logger.info(f"Image search complete. Retrieved {len(results)} unique hits.")
        return results
        
    except Exception as e:
        logger.error(f"Error executing image similarity search: {e}")
        raise RuntimeError(f"Image similarity search failure: {e}")
