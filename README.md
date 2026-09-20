# OmniBrain
OmniBrain Agentic Multi-Modal RAG Orchestrator

## Development Status

- Week 01: Completed - text/image embeddings and Qdrant storage/search
- Week 02: Completed - Supervisor routing, LangGraph retrieval, and state integration
- Week 03: Completed - NeMo Guardrails for document-scope enforcement
- Week 04: Completed - end-to-end pipeline validation and grounded API responses

The validated flow is:

```text
User Query -> NeMo Guardrails -> Supervisor/Router -> LangGraph Retrieval
-> Qdrant -> Text/Image Results -> Document-Grounded Response
```

Run the non-destructive validation suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_qdrant_lifecycle.py tests/test_guardrails_standalone.py tests/test_retrieval.py tests/test_full_pipeline.py -q
```

This validation preserves `data/qdrant_db`; `tests/test_qdrant.py` contains a
destructive integration harness and should only be run against an isolated test
database. The current non-destructive suite covers embeddings, Qdrant lifecycle,
retrieval, routing, compiled LangGraph state, guardrails, API responses, image
metadata, duplicate ingestion, and low-relevance rejection.

When collections are empty, the dashboard and reset utility ingest the extracted
source artifacts from `chunks.json` and `images.json`; they do not seed user
queries or synthetic document content.

Qdrant collections use cosine similarity. Text evidence uses a default minimum
score of `0.35`; CLIP image evidence uses `0.25`, calibrated from the valid
text-to-image sample query. Set `OMNIBRAIN_MIN_TEXT_COSINE_SCORE` or
`OMNIBRAIN_MIN_IMAGE_COSINE_SCORE` to tune these floors for a larger corpus.
