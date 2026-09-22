# Architecture

## Request flow

1. The React client sends a question to `POST /query`.
2. FastAPI invokes the Week 2 LangGraph supervisor.
3. The supervisor selects `text`, `sql`, or `vision`.
4. Text questions use the Week 3 Self-RAG graph; SQL questions use local SQLite data; vision questions inspect extracted PDF images.
5. FastAPI returns the response plus sources, Self-RAG status, or an image URL.

## Self-RAG flow

```text
question -> search -> relevance check -> answer
                               |
                               `-> rewrite -> search -> relevance check -> answer
```

There are at most two retrieval attempts. If context remains irrelevant, the answer node returns a safe document-not-found response.

## Data flow

`ingestion/pipeline.py` converts `data/input/sample.pdf` into page text, chunks, and image metadata under `data/output/`. The search and vision agents consume those outputs. `create_database.py` creates `data/database/omni.db` for SQL examples.
