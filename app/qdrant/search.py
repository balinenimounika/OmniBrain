import logging
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from app.embeddings.text_embeddings import generate_text_embedding
from app.embeddings.image_embeddings import generate_image_embedding
from app.qdrant.collections import TEXT_COLLECTION, IMAGE_COLLECTION

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        response = client.query_points(
            collection_name=TEXT_COLLECTION,
            query=query_vector,
            limit=top_k
        )
        
        # 3. Format hits
        results = []
        for hit in response.points:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
            
        logger.info(f"Text search complete. Retrieved {len(results)} hits.")
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
        response = client.query_points(
            collection_name=IMAGE_COLLECTION,
            query=query_vector,
            limit=top_k
        )
        
        # 3. Format hits
        results = []
        for hit in response.points:
            results.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
            
        logger.info(f"Image search complete. Retrieved {len(results)} hits.")
        return results
        
    except Exception as e:
        logger.error(f"Error executing image similarity search: {e}")
        raise RuntimeError(f"Image similarity search failure: {e}")
