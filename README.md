# OmniBrain

OmniBrain is an agentic, multimodal Retrieval-Augmented Generation (RAG) application for PDF documents. It ingests document text and images, routes questions to text, SQL, or vision agents, and uses a self-correcting retrieval flow for document questions.

## Features

- PDF text extraction, chunking, and embedded-image extraction
- LangGraph supervisor for text, structured-data, and image questions
- Self-RAG: search, relevance check, query rewrite, one retry, and grounded answers
- FastAPI endpoints plus a React + Vite user interface

## Architecture

```text
React UI -> FastAPI -> LangGraph supervisor
                          |-- Text agent -> Self-RAG -> sources + grounded answer
                          |-- SQL agent  -> SQLite sample data
                          `-- Vision agent -> extracted PDF image
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full data flow.

## Repository layout

```text
agents_week2/    Supervisor, text, SQL, and vision agents
agents_week3/    Self-RAG nodes (search, relevance, rewrite, answer)
graph_week2/     Agent-routing graph
graph_week3/     Self-RAG graph
ingestion/       PDF parsing, chunking, and image extraction
data/            Sample PDF and generated outputs
frontend/        React + Vite client
tests/           Automated API-key-free regression tests
scripts/         Manual smoke-test and demo scripts
docs/            Architecture, testing guide, and results
```

## Setup

Requirements: Python 3.11+, Node.js 18+, and a Groq API key for LLM-backed routing and answers.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `GROQ_API_KEY=your_key_here` in `.env`, then create the sample SQLite database:

```powershell
python create_database.py
```

## Run

Start the API from the repository root:

```powershell
uvicorn main:app --reload
```

The API is available at `http://127.0.0.1:8000`, and Swagger UI is at `/docs`.

In a second terminal, start the frontend:

```powershell
cd frontend
npm install
npm run dev
```

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /` | API confirmation |
| `GET /health` | Health check |
| `POST /query` | Route and answer a question |
| `GET /images/{filename}` | Serve an extracted PDF image |

## Test

```powershell
python -m pytest tests -q
cd frontend
npm run build
```

The complete test plan, manual scenarios, and current results are in [docs/TESTING.md](docs/TESTING.md) and [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md).

Manual demo scripts are kept separately from automated tests. For example, after configuring the database, run `python -m scripts.run_sql` from the repository root.

## Team contributions

The project was delivered as a four-week team effort:

| Team member | Primary contribution |
| --- | --- |
| Mounika | Self-RAG completion: relevance checking and query rewrite/retry behavior |
| Sudheer | NeMo Guardrails work and out-of-scope query handling |
| Kundan | **Week 4 project handoff:** documentation, README, test results, automated regression checks, and repository organization |
| Mehaboob | Final Streamlit/FastAPI integration, end-to-end testing, and demo preparation |

Kundan's Week 4 handoff is recorded in Git commit [`11cef18`](https://github.com/balinenimounika/OmniBrain/commit/11cef18232b962227d8fabf22bedc56d5bc9a3ed), authored by **Kundan Pandey**. An earlier end-to-end testing contribution is also preserved in the repository history.

## Security and limitations

Never commit `.env`, API keys, virtual environments, build output, or local databases. The SQL agent uses example SQLite records, and the vision agent currently returns metadata for an extracted image; both are designed as extension points.
