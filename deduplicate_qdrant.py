import os
import sys
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.qdrant.client import get_qdrant_client
from app.qdrant.collections import IMAGE_COLLECTION, TEXT_COLLECTION


def normalized_source_path(source_path):
    if not isinstance(source_path, str) or not source_path.strip():
        return None
    return os.path.normcase(os.path.normpath(source_path.replace("\\", "/")))


def scroll_all(client, collection_name):
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset
        )
        points.extend(batch)
        if offset is None:
            return points


def remove_duplicate_groups(client, collection_name, key_builder):
    points = scroll_all(client, collection_name)
    groups = defaultdict(list)
    for point in points:
        groups[key_builder(point)].append(point)

    duplicate_ids = []
    duplicate_groups = 0
    for group in groups.values():
        if len(group) > 1:
            duplicate_groups += 1
            duplicate_ids.extend(point.id for point in group[1:])

    if duplicate_ids:
        client.delete(collection_name=collection_name, points_selector=duplicate_ids)

    return len(points), duplicate_groups, len(duplicate_ids), len(groups)


def text_key(point):
    payload = point.payload or {}
    return (
        payload.get("chunk_id"),
        payload.get("document_name"),
        payload.get("page_number"),
        payload.get("text", payload.get("content"))
    )


def image_key(point):
    payload = point.payload or {}
    if payload.get("image_id"):
        return ("image_id", payload["image_id"])
    source_path = normalized_source_path(payload.get("source_path"))
    if source_path:
        return ("source_path", source_path)
    return ("point_id", point.id)


def main():
    client = get_qdrant_client()
    text_total, text_groups, text_removed, text_unique = remove_duplicate_groups(
        client, TEXT_COLLECTION, text_key
    )
    image_total, image_groups, image_removed, image_unique = remove_duplicate_groups(
        client, IMAGE_COLLECTION, image_key
    )

    print(f"Text records before cleanup: {text_total}")
    print(f"Text duplicate groups found: {text_groups}")
    print(f"Duplicate text records removed: {text_removed}")
    print(f"Unique text records remaining: {text_unique}")
    print(f"Image records before cleanup: {image_total}")
    print(f"Image duplicate groups found: {image_groups}")
    print(f"Duplicate image records removed: {image_removed}")
    print(f"Unique image records remaining: {image_unique}")
    client.close()


if __name__ == "__main__":
    main()
