import logging
import os
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct
from app.qdrant.collections import TEXT_COLLECTION, IMAGE_COLLECTION

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _normalized_source_path(source_path):
    if not isinstance(source_path, str) or not source_path.strip():
        return None
    return os.path.normcase(os.path.normpath(source_path.replace("\\", "/")))

def _find_existing_point(client: QdrantClient, collection_name: str, conditions):
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(must=conditions),
        limit=1
    )
    return points[0] if points else None

def insert_text_vector(
    client: QdrantClient,
    point_id: str,
    vector: List[float],
    document_name: str,
    page_number: int,
    chunk_id: str,
    source_path: str,
    text: str,
    image_id: str = None
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
        image_id (str, optional): ID of the image on the same page.
    """
    # Safe validation of vector data and IDs
    if not point_id:
        raise ValueError("point_id cannot be empty.")
    if not isinstance(vector, list) or len(vector) == 0:
        raise ValueError("vector must be a non-empty list of floats.")

    existing_point = _find_existing_point(
        client,
        TEXT_COLLECTION,
        [
            FieldCondition(key="document_name", match=MatchValue(value=document_name)),
            FieldCondition(key="page_number", match=MatchValue(value=page_number)),
            FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))
        ]
    )
    if existing_point:
        point_id = existing_point.id
        
    image_path = None
    if image_id:
        try:
            scroll_res = client.scroll(
                collection_name=IMAGE_COLLECTION,
                scroll_filter=Filter(
                    must=[FieldCondition(key="image_id", match=MatchValue(value=image_id))]
                ),
                limit=1
            )
            points = scroll_res[0]
            if points and points[0].payload:
                image_path = points[0].payload.get("image_path") or points[0].payload.get("source_path")
        except Exception:
            pass
        if not image_path:
            image_id = None
        
    payload = {
        "document_name": document_name,
        "page_number": page_number,
        "content_type": "text",
        "chunk_id": chunk_id,
        "source_path": source_path,
        "text": text,
        "image_id": image_id,
        "image_path": image_path
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

    existing_point = None
    if image_id:
        existing_point = _find_existing_point(
            client,
            IMAGE_COLLECTION,
            [FieldCondition(key="image_id", match=MatchValue(value=image_id))]
        )
    if not existing_point:
        source_path_values = [source_path]
        normalized_source_path = _normalized_source_path(source_path)
        if normalized_source_path not in source_path_values:
            source_path_values.append(normalized_source_path)
        for source_path_value in source_path_values:
            if not source_path_value:
                continue
            existing_point = _find_existing_point(
                client,
                IMAGE_COLLECTION,
                [FieldCondition(key="source_path", match=MatchValue(value=source_path_value))]
            )
            if existing_point:
                break
    if existing_point:
        point_id = existing_point.id
        
    payload = {
        "document_name": document_name,
        "page_number": page_number,
        "content_type": "image",
        "image_id": image_id,
        "source_path": source_path,
        "image_path": source_path
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
        
        # Cross-update existing text chunks for the same page/document with this new image_id
        try:
            scroll_res = client.scroll(
                collection_name=TEXT_COLLECTION,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="document_name", match=MatchValue(value=document_name)),
                        FieldCondition(key="page_number", match=MatchValue(value=page_number))
                    ]
                )
            )
            points = scroll_res[0]
            for p in points:
                new_payload = p.payload.copy() if p.payload else {}
                if new_payload.get("image_id") != image_id:
                    new_payload["image_id"] = image_id
                    client.set_payload(
                        collection_name=TEXT_COLLECTION,
                        payload=new_payload,
                        points=[p.id]
                    )
                    logger.info(f"Updated text chunk [ID: {p.id}] payload with image_id: {image_id}")
        except Exception as update_err:
            logger.warning(f"Could not cross-update text chunks with image_id: {update_err}")
            
    except Exception as e:
        logger.error(f"Failed to insert image vector into Qdrant: {e}")
        raise RuntimeError(f"Qdrant image upsert failure: {e}")

