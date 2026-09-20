import os
import sys
import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval.router import route_query
from app.server import app
from app.qdrant.collections import TEXT_COLLECTION, create_omnibrain_collections
from app.qdrant.insert import insert_text_vector, next_chunk_id


def test_supervisor_routes_text_and_image_queries():
    assert route_query("What was revenue growth in 2025?") == "text"
    assert route_query("Show the chart image") == "image"
    assert route_query("Compare both the text and image") == "multimodal"
    assert route_query("anything", "text") == "text"


def test_answer_returns_grounded_text_with_metadata():
    state = {
        "retrieval_mode": "text",
        "retrieval_results": [{
            "type": "text",
            "content": "Revenue increased by 25 percent in 2025.",
            "document_name": "annual_report.pdf",
            "page_number": 25,
        }],
        "retrieved_context": "Revenue increased by 25 percent in 2025.",
        "retrieved_text": [],
        "retrieved_images": [],
        "similarity_scores": [0.9],
        "source_path": [],
        "retrieval_status": "success",
        "user_query": "What was the company's revenue growth in 2025?",
    }
    with patch("app.server.get_qdrant_client"), patch("app.server.retrieval_node", return_value=state):
        response = TestClient(app).post("/answer", json={"query": state["user_query"]})

    body = response.json()
    assert response.status_code == 200
    assert body["guardrail_allowed"] is True
    assert "Revenue increased by 25 percent in 2025" in body["answer"]
    assert "annual_report.pdf" in body["answer"]
    assert "Page: 25" in body["answer"]


def test_answer_uses_exact_unsupported_document_message():
    state = {
        "retrieval_mode": "text",
        "retrieval_results": [],
        "retrieved_context": "",
        "retrieved_text": [],
        "retrieved_images": [],
        "similarity_scores": [],
        "source_path": [],
        "retrieval_status": "empty",
        "user_query": "What was the company's profit in 2018?",
    }
    with patch("app.server.get_qdrant_client"), patch("app.server.retrieval_node", return_value=state):
        response = TestClient(app).post("/answer", json={"query": state["user_query"]})

    assert response.status_code == 200
    assert response.json()["answer"] == "I could not find enough information in the provided document to answer that question."


def test_answer_preserves_image_results_and_metadata():
    state = {
        "retrieval_mode": "image",
        "retrieval_results": [{
            "type": "image",
            "image_id": "image_025_01",
            "document_name": "annual_report.pdf",
            "page_number": 25,
            "source_path": "data/images/sample.png",
        }],
        "retrieved_context": "[1] Image Reference (Document: annual_report.pdf, Page: 25)",
        "retrieved_text": [],
        "retrieved_images": [{
            "image_id": "image_025_01",
            "document_name": "annual_report.pdf",
            "page_number": 25,
        }],
        "similarity_scores": [0.95],
        "source_path": ["data/images/sample.png"],
        "retrieval_status": "success",
        "user_query": "Show the image",
    }
    with patch("app.server.get_qdrant_client"), patch("app.server.retrieval_node", return_value=state):
        response = TestClient(app).post("/answer", json={"query": state["user_query"], "route": "image"})

    body = response.json()
    assert response.status_code == 200
    assert body["route"] == "image"
    assert body["retrieval_results"][0]["image_id"] == "image_025_01"
    assert "annual_report.pdf" in body["answer"]


def test_retrieve_skips_qdrant_for_out_of_scope_query():
    client = TestClient(app)
    with patch("app.server.retrieval_node") as retrieval:
        response = client.post("/retrieve", json={"query": "What is the capital of Japan?"})

    assert response.status_code == 200
    assert response.json()["retrieval_status"] == "skipped"
    assert response.json()["guardrail_allowed"] is False
    retrieval.assert_not_called()


@pytest.mark.parametrize("query", [
    "What is the capital of India?",
    "Tell me a joke.",
    "What is the current population of Mars?",
    "How do I bake a chocolate cake?",
])
def test_answer_blocks_unrelated_queries_before_retrieval(query):
    client = TestClient(app)
    with patch("app.server.retrieval_node") as retrieval:
        response = client.post("/answer", json={"query": query})

    body = response.json()
    assert response.status_code == 200
    assert body["guardrail_allowed"] is False
    assert body["guardrail_status"] == "BLOCKED"
    assert body["retrieval_status"] == "skipped"
    assert body["retrieval_skipped"] is True
    assert "outside the scope of the available documents" in body["answer"]
    retrieval.assert_not_called()


def test_three_text_chunks_get_unique_ids_and_incremented_chunk_ids(tmp_path):
    from qdrant_client import QdrantClient

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    create_omnibrain_collections(client)
    vector = [0.1] * 384

    for text, page_number in (
        ("Revenue increased in 2025.", 25),
        ("Operating costs declined in 2025.", 25),
        ("The company expanded into new markets.", 26),
    ):
        insert_text_vector(
            client=client,
            point_id="ignored-by-insertion",
            vector=vector,
            document_name="annual_report.pdf",
            page_number=page_number,
            chunk_id=next_chunk_id(client, "annual_report.pdf", page_number),
            source_path="data/documents/annual_report.pdf",
            text=text,
        )

    points, _ = client.scroll(collection_name=TEXT_COLLECTION, limit=10)
    point_ids = [str(point.id) for point in points]
    payloads = [point.payload for point in points]

    assert len(point_ids) == 3
    assert len(set(point_ids)) == 3
    assert all(len(point_id) == 36 and point_id.count("-") == 4 for point_id in point_ids)
    assert {payload["chunk_id"] for payload in payloads} == {
        "chunk_025_01", "chunk_025_02", "chunk_026_01"
    }
    assert all(payload["content"] == payload["text"] for payload in payloads)
    assert all(payload["source_path"] == "data/documents/annual_report.pdf" for payload in payloads)
    client.close()