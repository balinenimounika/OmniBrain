import logging
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from app.qdrant.collections import TEXT_COLLECTION, IMAGE_COLLECTION

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def insert_text_vector(
    client: QdrantClient,
    point_id: str,
    vector: List[float],
    document_name: str,
    page_number: int,
    chunk_id: str,
    source_path: str,
    text: str
):
    """
    Inserts a text embedding vector along with its PDF-chunk metadata into Qdrant.
    
    Args:
        client (QdrantClient): The Qdrant client instance.
        point_id (str): A unique ID (string UUID or integer) for the vector point.
        vector (List[float]): The 384-dimensional text embedding vector.
        document_name (str): Original document name.
        page_number (int): Page number in the document.
        chunk_id (str): Unique chunk identifier.
        source_path (str): File system/remote source path.
        text (str): The original text block.
    """
    # Safe validation of vector data and IDs
    if not point_id:
        raise ValueError("point_id cannot be empty.")
    if not isinstance(vector, list) or len(vector) == 0:
        raise ValueError("vector must be a non-empty list of floats.")
        
    payload = {
        "document_name": document_name,
        "page_number": page_number,
        "content_type": "text",
        "chunk_id": chunk_id,
        "source_path": source_path,
        "text": text
    }
    
    try:
        logger.info(f"Inserting text vector to collection '{TEXT_COLLECTION}' [ID: {point_id}]")
        client.upsert(
            collection_name=TEXT_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        logger.info("Text vector successfully inserted.")
    except Exception as e:
        logger.error(f"Failed to insert text vector into Qdrant: {e}")
        raise RuntimeError(f"Qdrant text upsert failure: {e}")


def insert_image_vector(
    client: QdrantClient,
    point_id: str,
    vector: List[float],
    document_name: str,
    page_number: int,
    image_id: str,
    source_path: str
):
    """
    Inserts an image embedding vector along with its document-image metadata into Qdrant.
    
    Args:
        client (QdrantClient): The Qdrant client instance.
        point_id (str): A unique ID (string UUID or integer) for the vector point.
        vector (List[float]): The 512-dimensional CLIP image embedding vector.
        document_name (str): Original document name from which the image was extracted.
        page_number (int): Page number in the document.
        image_id (str): Unique image identifier.
        source_path (str): Path to the image file in the filesystem.
    """
    # Safe validation of vector data and IDs
    if not point_id:
        raise ValueError("point_id cannot be empty.")
    if not isinstance(vector, list) or len(vector) == 0:
        raise ValueError("vector must be a non-empty list of floats.")
        
    payload = {
        "document_name": document_name,
        "page_number": page_number,
        "content_type": "image",
        "image_id": image_id,
        "source_path": source_path
    }
    
    try:
        logger.info(f"Inserting image vector to collection '{IMAGE_COLLECTION}' [ID: {point_id}]")
        client.upsert(
            collection_name=IMAGE_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        logger.info("Image vector successfully inserted.")
    except Exception as e:
        logger.error(f"Failed to insert image vector into Qdrant: {e}")
        raise RuntimeError(f"Qdrant image upsert failure: {e}")
