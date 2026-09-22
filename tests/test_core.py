from agents_week3.relevance_checker import relevance_checker
from agents_week3.search_agent import search_agent
from ingestion.text_chunker import chunk_text, create_chunks


def test_chunk_text_preserves_overlap():
    assert chunk_text("abcdefghij", chunk_size=6, overlap=2) == ["abcdef", "efghij"]


def test_create_chunks_keeps_page_and_ids():
    chunks = create_chunks([{"page": 3, "text": "abcdefgh"}, {"page": 4, "text": "ijklmnop"}], chunk_size=6, overlap=2)
    assert [item["chunk_id"] for item in chunks] == [1, 2, 3, 4]
    assert [item["page"] for item in chunks] == [3, 3, 4, 4]


def test_search_returns_ranked_week_two_context():
    result = search_agent({"query": "What is the Week 2 development plan?", "attempt": 1})
    scores = [item["score"] for item in result["retrieved_docs"]]
    assert scores and scores == sorted(scores, reverse=True)


def test_relevance_rejects_unrelated_revenue_context():
    result = relevance_checker({"query": "What is the revenue in 2024?", "retrieved_docs": [{"chunk_id": 1, "page": 1, "text": "OmniBrain uses LangGraph.", "score": 1}]})
    assert result == {"retrieved_docs": [], "relevant": False}

