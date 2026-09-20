import os
from pathlib import Path

import app.qdrant.client as client_module
from app.qdrant.cleanup import remove_text_content
from app.qdrant.collections import TEXT_COLLECTION, create_omnibrain_collections
from app.qdrant.ingest import looks_like_user_query
from app.qdrant.insert import insert_image_vector, insert_text_vector
from app.qdrant.insert import _stored_image_path
from qdrant_client import QdrantClient


def test_process_client_is_created_once(monkeypatch):
    fake_client = object()
    calls = []

    monkeypatch.setattr(client_module, "_CLIENT_INSTANCE", None)
    monkeypatch.setattr(
        client_module,
        "_create_qdrant_client",
        lambda: calls.append(True) or fake_client,
    )

    first = client_module._get_process_client()
    second = client_module._get_process_client()

    assert first is fake_client
    assert second is first
    assert len(calls) == 1


def test_absolute_image_path_is_stored_relative_to_project():
    absolute_path = client_module.PROJECT_ROOT / "data" / "images" / "sample.png"

    assert _stored_image_path(str(absolute_path)) == os.path.join("data", "images", "sample.png")


def test_external_image_path_remains_absolute():
    external_path = Path("C:/external/image.png")

    assert _stored_image_path(str(external_path)) == external_path.as_posix()


def test_targeted_cleanup_removes_only_exact_bad_query(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    create_omnibrain_collections(client)
    vector = [0.1] * 384
    insert_text_vector(client, None, vector, "annual_report.pdf", 25, "chunk_025_01", "doc.pdf", "Valid source text")
    insert_text_vector(client, None, vector, "annual_report.pdf", 25, "chunk_025_02", "doc.pdf", "What factors contributed to revenue growth in 2025?")

    removed = remove_text_content(
        client,
        "What factors contributed to revenue growth in 2025?",
        document_name="annual_report.pdf",
    )

    points, _ = client.scroll(collection_name=TEXT_COLLECTION, limit=10)
    contents = [point.payload.get("content") for point in points]
    assert removed == 1
    assert contents == ["Valid source text"]
    client.close()


def test_same_text_reingestion_with_new_chunk_id_updates_one_record(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    create_omnibrain_collections(client)
    vector = [0.1] * 384
    insert_text_vector(client, None, vector, "annual_report.pdf", 25, "chunk_025_01", "doc.pdf", "Same source text")
    insert_text_vector(client, None, vector, "annual_report.pdf", 25, "chunk_025_02", "doc.pdf", "Same source text")

    points, _ = client.scroll(collection_name=TEXT_COLLECTION, limit=10)
    matches = [point for point in points if (point.payload or {}).get("content") == "Same source text"]
    assert len(matches) == 1
    assert matches[0].payload["chunk_id"] == "chunk_025_01"
    client.close()


def test_same_image_source_reingestion_with_new_image_id_updates_one_record(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    create_omnibrain_collections(client)
    vector = [0.1] * 512
    insert_image_vector(client, None, vector, "annual_report.pdf", 25, "image_025_01", "data/images/same.png")
    insert_image_vector(client, None, vector, "annual_report.pdf", 25, "image_025_02", "data/images/same.png")

    points, _ = client.scroll(collection_name="omnibrain_images", limit=10)
    matches = [point for point in points if (point.payload or {}).get("source_path") == "data/images/same.png"]
    assert len(matches) == 1
    assert matches[0].payload["image_id"] == "image_025_01"
    client.close()


def test_extracted_artifacts_ingest_as_source_metadata(tmp_path, monkeypatch):
    import app.qdrant.ingest as ingest

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    create_omnibrain_collections(client)
    monkeypatch.setattr(ingest, "generate_text_embedding", lambda text: [0.1] * 384)
    monkeypatch.setattr(ingest, "generate_image_embedding", lambda path: [0.1] * 512)

    counts = ingest.ingest_extracted_artifacts(client)
    text_points, _ = client.scroll(collection_name=TEXT_COLLECTION, limit=100)

    assert counts["text"] == 18
    assert counts["images"] == 6
    assert len(text_points) == 18
    assert all(point.payload["document_name"] == "sample.pdf" for point in text_points)
    assert all(point.payload["content_type"] == "text" for point in text_points)
    assert not any(
        point.payload["content"] == "What factors contributed to revenue growth in 2025?"
        for point in text_points
    )
    client.close()


def test_question_shaped_input_is_not_document_content():
    assert looks_like_user_query("What factors contributed to revenue growth in 2025?")
    assert looks_like_user_query("Tell me a joke?")
    assert not looks_like_user_query("Revenue growth was driven by improved sales performance.")
