from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.qdrant.collections import TEXT_COLLECTION


def remove_text_content(
    client: QdrantClient,
    content: str,
    document_name: Optional[str] = None,
) -> int:
    """Remove only text points whose stored content exactly matches the target."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a non-empty string")

    conditions = [FieldCondition(key="content", match=MatchValue(value=content))]
    if document_name:
        conditions.append(
            FieldCondition(key="document_name", match=MatchValue(value=document_name))
        )

    points, _ = client.scroll(
        collection_name=TEXT_COLLECTION,
        scroll_filter=Filter(must=conditions),
        limit=10000,
    )
    ids = [point.id for point in points]
    if ids:
        client.delete(collection_name=TEXT_COLLECTION, points_selector=ids)
    return len(ids)
