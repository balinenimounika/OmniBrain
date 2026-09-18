import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from graph_week2.workflow import graph

app = FastAPI(
    title="OmniBrain",
    description="Agentic Multi-Modal RAG Orchestrator",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "output", "images"))


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"message": "OmniBrain API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/images/{filename}")
def get_image(filename: str):
    # Prevent path traversal attacks
    safe_path = os.path.abspath(os.path.join(IMAGES_DIR, filename))
    if not safe_path.startswith(IMAGES_DIR) or not os.path.isfile(safe_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(safe_path)


@app.post("/query")
def query_document(request: QueryRequest):
    result = graph.invoke({"query": request.query})
    next_agent = result.get("next_agent")
    response_text = result.get("response", "No response generated.")

    response_data = {
        "query": result.get("query", request.query),
        "next_agent": next_agent,
        "response": response_text
    }

    if next_agent == "text":
        attempt = result.get("attempt", 1)
        relevant = result.get("relevant", False)
        query_rewritten = result.get("query_rewritten", attempt > 1)
        retrieved_docs = result.get("retrieved_docs", [])

        response_data["self_rag"] = {
            "enabled": True,
            "attempts": attempt,
            "relevance_found": relevant,
            "query_rewritten": query_rewritten,
            "status": "completed"
        }

        sources = []
        for doc in retrieved_docs:
            sources.append({
                "chunk_id": doc.get("chunk_id"),
                "page": doc.get("page"),
                "text": doc.get("text", "")
            })
        response_data["sources"] = sources

    elif next_agent == "vision":
        image_file = result.get("image_file")
        if image_file:
            response_data["image_file"] = image_file
            response_data["image_url"] = f"/images/{image_file}"

    return response_data