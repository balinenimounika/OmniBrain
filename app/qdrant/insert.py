import logging
import os
import re
import uuid
from pathlib import Path
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct
from app.qdrant.collections import TEXT_COLLECTION, IMAGE_COLLECTION

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _normalized_source_path(source_path):
    if not isinstance(source_path, str) or not source_path.strip():
        return None
    return os.path.normcase(os.path.normpath(source_path.replace("\\", "/")))


def _stored_image_path(source_path: str) -> str:
    """Store image paths relative to the project so every caller resolves them consistently."""
    original = str(source_path or "")
    normalized = original.replace("\\", "/")
    path = Path(normalized)
    if not path.is_absolute():
        return original
    if path.is_absolute():
        try:
            normalized = str(path.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            normalized = path.as_posix()
    return normalized


def _qdrant_point_id(point_id, *identity_parts: str):
    if point_id is None:
        return _stable_point_id(*identity_parts)
    if isinstance(point_id, int):
        return point_id
    try:
        uuid.UUID(str(point_id))
        return str(point_id)
    except (ValueError, AttributeError, TypeError):
        return _stable_point_id(*identity_parts)

def _find_existing_point(client: QdrantClient, collection_name: str, conditions):
    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(must=conditions),
        limit=1
    )
    return points[0] if points else None


def _find_existing_text_point(client: QdrantClient, document_name: str, page_number: int, chunk_id: str):
    if not chunk_id:
        return None
    try:
        scroll_res = client.scroll(
            collection_name=TEXT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="document_name", match=MatchValue(value=document_name)),
                    FieldCondition(key="page_number", match=MatchValue(value=page_number)),
                    FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id)),
                ]
            ),
            limit=10,
        )
        points = scroll_res[0]
        for point in points:
            payload = point.payload or {}
            if (
                payload.get("document_name") == document_name
                and payload.get("page_number") == page_number
                and payload.get("chunk_id") == chunk_id
            ):
                return point
    except Exception:
        pass
    return None


def _find_existing_text_content(client: QdrantClient, document_name: str, page_number: int, text: str):
    try:
        points, _ = client.scroll(collection_name=TEXT_COLLECTION, limit=10000)
        for point in points:
            payload = point.payload or {}
            if (
                payload.get("document_name") == document_name
                and payload.get("page_number") == page_number
                and payload.get("content", payload.get("text")) == text
            ):
                return point
    except Exception:
        pass
    return None


def _stable_point_id(*parts: str) -> str:
    seed = "|".join(str(part) for part in parts if part is not None)
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def _existing_payloads(client: QdrantClient, collection_name: str):
    points, _ = client.scroll(collection_name=collection_name, limit=10000)
    return [point.payload or {} for point in points]


def next_chunk_id(client: QdrantClient, document_name: str, page_number: int) -> str:
    """Return the next document/page-scoped chunk reference."""
    prefix = f"chunk_{page_number:03d}_"
    numbers = []
    for payload in _existing_payloads(client, TEXT_COLLECTION):
        if (
            payload.get("document_name") == document_name
            and payload.get("page_number") == page_number
        ):
            match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", str(payload.get("chunk_id", "")))
            if match:
                numbers.append(int(match.group(1)))
    return f"{prefix}{max(numbers, default=0) + 1:02d}"


def next_image_id(client: QdrantClient, document_name: str, page_number: int) -> str:
    """Return the next document/page-scoped image reference."""
    prefix = f"image_{page_number:03d}_"
    numbers = set()
    for payload in _existing_payloads(client, IMAGE_COLLECTION):
        if (
            payload.get("document_name") == document_name
            and payload.get("page_number") == page_number
        ):
            match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", str(payload.get("image_id", "")))
            if match:
                numbers.add(int(match.group(1)))
    return f"{prefix}{max(numbers, default=0) + 1:02d}"


def _find_existing_image_point(client: QdrantClient, image_id: str):
    if not image_id:
        return None
    try:
        scroll_res = client.scroll(
            collection_name=IMAGE_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="image_id", match=MatchValue(value=image_id))]
            ),
            limit=10,
        )
        points = scroll_res[0]
        for point in points:
            if point.payload and point.payload.get("image_id") == image_id:
                return point
    except Exception:
        pass
    return None


def _find_existing_image_source(client: QdrantClient, document_name: str, page_number: int, source_path: str):
    normalized_source = _normalized_source_path(source_path)
    if not normalized_source:
        return None
    try:
        points, _ = client.scroll(collection_name=IMAGE_COLLECTION, limit=10000)
        for point in points:
            payload = point.payload or {}
            if (
                payload.get("document_name") == document_name
                and payload.get("page_number") == page_number
                and _normalized_source_path(payload.get("source_path")) == normalized_source
            ):
                return point
    except Exception:
        pass
    return None

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
    if not isinstance(vector, list) or len(vector) == 0:
        raise ValueError("vector must be a non-empty list of floats.")

    existing_point = _find_existing_text_point(client, document_name, page_number, chunk_id)
    if existing_point is None:
        existing_point = _find_existing_text_content(client, document_name, page_number, text)
    if existing_point is not None:
        point_id = existing_point.id
        chunk_id = (existing_point.payload or {}).get("chunk_id") or chunk_id
    elif existing_point is None:
        point_id = _qdrant_point_id(point_id, document_name, str(page_number), chunk_id)

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
        "content": text,
        "text": text,
        "image_id": image_id,
        "image_path": image_path
    }
    
    try:
        if existing_point:
            logger.info(f"Replacing existing text vector for chunk '{chunk_id}' [Qdrant ID: {point_id}]")
        else:
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
        logger.info("Text vector successfully inserted/updated.")
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
    if not isinstance(vector, list) or len(vector) == 0:
        raise ValueError("vector must be a non-empty list of floats.")

    existing_point = _find_existing_image_point(client, image_id) if image_id else None
    if existing_point is None:
        existing_point = _find_existing_image_source(client, document_name, page_number, source_path)
    if existing_point is not None:
        point_id = existing_point.id
        image_id = (existing_point.payload or {}).get("image_id") or image_id
    elif existing_point is None:
        point_id = _qdrant_point_id(point_id, document_name, str(page_number), image_id)

    stored_source_path = _stored_image_path(source_path)
    payload = {
        "document_name": document_name,
        "page_number": page_number,
        "content_type": "image",
        "image_id": image_id,
        "source_path": stored_source_path,
        "image_path": stored_source_path
    }

    try:
        if existing_point:
            logger.info(f"Replacing existing image vector for image_id '{image_id}' [Qdrant ID: {point_id}]")
        else:
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
        logger.info("Image vector successfully inserted/updated.")

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

