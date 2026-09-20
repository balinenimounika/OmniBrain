import json
import re
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient

from app.embeddings.image_embeddings import generate_image_embedding
from app.embeddings.text_embeddings import generate_text_embedding
from app.qdrant.collections import IMAGE_COLLECTION, TEXT_COLLECTION
from app.qdrant.insert import insert_image_vector, insert_text_vector

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = PROJECT_ROOT / "OmniBrain PDF Text Extraction" / "data" / "output" / "chunks" / "chunks.json"
IMAGES_FILE = PROJECT_ROOT / "data" / "images" / "images.json"
SOURCE_DOCUMENT = "sample.pdf"
SOURCE_PATH = "OmniBrain PDF Text Extraction/data/input/sample.pdf"


def looks_like_user_query(text: str) -> bool:
    """Identify question-shaped input before the manual document-ingestion UI stores it."""
    normalized = " ".join(str(text or "").split())
    return bool(
        normalized.endswith("?")
        and re.match(r"^(what|why|how|who|where|when|which|is|are|can|does|do|tell|write|create|show)\b", normalized, re.IGNORECASE)
    )


def _load_json(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Ingestion artifact not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in ingestion artifact: {path}")
    return data


def ingest_extracted_artifacts(client: QdrantClient) -> Dict[str, int]:
    """Embed extracted source artifacts into Qdrant using idempotent identities."""
    chunks = _load_json(CHUNKS_FILE)
    images = _load_json(IMAGES_FILE)
    inserted_text = 0
    inserted_images = 0

    for chunk in chunks:
        text = str(chunk.get("text") or "").strip()
        page_number = int(chunk.get("page", 0))
        source_chunk_id = str(chunk.get("chunk_id") or "")
        if not text or page_number < 1 or not source_chunk_id:
            continue
        chunk_id = f"chunk_{page_number:03d}_{source_chunk_id.zfill(2)}"
        insert_text_vector(
            client=client,
            point_id=None,
            vector=generate_text_embedding(text),
            document_name=SOURCE_DOCUMENT,
            page_number=page_number,
            chunk_id=chunk_id,
            source_path=SOURCE_PATH,
            text=text,
            image_id=None,
        )
        inserted_text += 1

    for image in images:
        image_id = str(image.get("image_id") or "")
        page_number = int(image.get("page", 0))
        image_path = Path(str(image.get("path") or ""))
        if not image_id or page_number < 1 or not image_path.is_file():
            continue
        insert_image_vector(
            client=client,
            point_id=None,
            vector=generate_image_embedding(str(image_path)),
            document_name=SOURCE_DOCUMENT,
            page_number=page_number,
            image_id=image_id,
            source_path=str(image_path),
        )
        inserted_images += 1

    return {"text": inserted_text, "images": inserted_images}
