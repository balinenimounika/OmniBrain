import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.guardrails.guardrails import (
    EMPTY_QUERY_MESSAGE,
    INSUFFICIENT_CONTEXT_MESSAGE,
    OUT_OF_SCOPE_MESSAGE,
    NEMO_AVAILABLE,
    rails,
    validate_query_scope,
    should_answer_question,
)
from app.server import app


def test_in_scope_query_is_allowed_with_context():
    state = should_answer_question({
        "user_query": "What was the company's revenue growth in 2025?",
        "retrieved_context": "Revenue increased by 25 percent in 2025.",
    })
    assert state["guardrail_allowed"] is True


def test_out_of_scope_queries_are_rejected():
    for query in (
        "What is the weather today?",
        "What is the capital of France?",
        "Write a Python program to reverse a string.",
        "Who won the FIFA World Cup in 2022?",
    ):
        state = should_answer_question({"user_query": query, "retrieved_context": "document content"})
        assert state["guardrail_allowed"] is False
        assert state["guardrail_message"] == OUT_OF_SCOPE_MESSAGE


def test_unrelated_sports_query_is_rejected_even_when_context_is_financial():
    query = "Who won the FIFA World Cup in 2022?"
    state = should_answer_question({
        "user_query": query,
        "retrieved_context": "The company's revenue increased by 25 percent in 2025 due to strong market growth.",
    })
    assert validate_query_scope(query)[0] is False
    assert state["guardrail_allowed"] is False
    assert state["guardrail_message"] == OUT_OF_SCOPE_MESSAGE


def test_empty_and_whitespace_queries_are_controlled():
    for query in ("", "   "):
        state = should_answer_question({"user_query": query, "retrieved_context": ""})
        assert state["guardrail_allowed"] is False
        assert state["guardrail_message"] == EMPTY_QUERY_MESSAGE


def test_missing_context_is_controlled():
    state = should_answer_question({"user_query": "What does the annual report say?"})
    assert state["guardrail_allowed"] is False
    assert state["guardrail_message"] == INSUFFICIENT_CONTEXT_MESSAGE


def test_unknown_document_fact_is_rejected():
    state = should_answer_question({
        "user_query": "What was the company's profit in 2018?",
        "retrieved_context": "The company's revenue increased by 25 percent in 2025.",
    })
    assert state["guardrail_allowed"] is False
    assert state["guardrail_message"] == INSUFFICIENT_CONTEXT_MESSAGE


def test_low_relevance_context_is_rejected():
    state = should_answer_question({
        "user_query": "What was the company's revenue growth in 2025?",
        "retrieved_context": "Revenue increased by 25 percent in 2025.",
        "similarity_scores": [0.20],
    })
    assert state["guardrail_allowed"] is False
    assert state["guardrail_reason"] == "Low retrieval relevance"
    assert state["guardrail_message"] == INSUFFICIENT_CONTEXT_MESSAGE


def test_broad_document_question_is_allowed():
    state = should_answer_question({
        "user_query": "What financial performance information is mentioned in the annual report?",
        "retrieved_context": "The company's revenue increased by 25 percent in 2025.",
    })
    assert state["guardrail_allowed"] is True


def test_nemo_configuration_initializes():
    assert NEMO_AVAILABLE is True
    assert rails is not None


def test_answer_allows_document_query_and_runs_retrieval():
    retrieved_state = {
        "retrieval_results": [{"type": "text", "content": "Revenue increased in 2025."}],
        "retrieved_context": "Revenue increased in 2025.",
        "retrieved_text": [{"content": "Revenue increased in 2025."}],
        "retrieved_images": [],
        "similarity_scores": [0.9],
        "source_path": [],
        "retrieval_status": "success",
        "user_query": "What was the company's revenue growth in 2025?",
    }
    client = TestClient(app)
    with patch("app.server.get_qdrant_client"), patch(
        "app.server.retrieval_node", return_value=retrieved_state
    ) as retrieval:
        response = client.post("/answer", json={"query": "What was the company's revenue growth in 2025?"})
    assert response.status_code == 200
    assert response.json()["guardrail_allowed"] is True
    retrieval.assert_called_once()


def test_answer_skips_retrieval_for_out_of_scope_query():
    client = TestClient(app)
    with patch("app.server.retrieval_node") as retrieval:
        response = client.post("/answer", json={"query": "What is the weather today?"})
    assert response.status_code == 200
    assert response.json()["guardrail_message"] == OUT_OF_SCOPE_MESSAGE
    assert response.json()["retrieval_status"] == "skipped"
    retrieval.assert_not_called()
