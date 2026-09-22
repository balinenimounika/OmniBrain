from fastapi import FastAPI
from pydantic import BaseModel

from graph.workflow import graph

app = FastAPI(title="OmniBrain API")


@app.get("/")
def home():
    return {"message": "OmniBrain API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query(request: QueryRequest):
    result = graph.invoke({
        "query": request.query
    })

    return {
        "query": request.query,
        "answer": result.get("response", ""),
        "next_agent": result.get("next_agent", ""),
        "sources": [],
        "images": []
    }