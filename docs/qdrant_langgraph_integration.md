# Qdrant & LangGraph Integration Contract (Week 2)

This document describes the interface and usage of the retrieval layer developed in Week 2. It is designed to allow the LangGraph Supervisor developer to integrate the retrieval node seamlessly without reading the implementation details.

---

## 1. Supported Modalities & APIs

The retrieval layer exposes three main capabilities from `app.retrieval`:

- **Text Retrieval**: Semantic similarity search over PDF text chunks using `all-MiniLM-L6-v2`.
- **Image Retrieval**: Similarity search over images. Supports both **image-to-image** (by providing an existing image file path) and **text-to-image** semantic queries (using `clip-ViT-B-32` embeddings).
- **Multimodal Retrieval**: Retrieves both text chunks and images relevant to the query and returns them ranked together by similarity score.

---

## 2. Functions & Parameters

### `retrieve_text`
```python
def retrieve_text(query: str, top_k: int = 3, client: Optional[QdrantClient] = None) -> List[RetrievalResult]:
```
- **`query`** (str): The search text query. Must be non-empty.
- **`top_k`** (int): Maximum number of text results to return. Default is `3`.
- **`client`** (QdrantClient, optional): Reusable client connection. If `None`, will initialize using `get_qdrant_client()`.

### `retrieve_images`
```python
def retrieve_images(query: str, top_k: int = 3, client: Optional[QdrantClient] = None) -> List[RetrievalResult]:
```
- **`query`** (str): An image file path (for image-to-image search) OR a text query description (for text-to-image search).
- **`top_k`** (int): Maximum number of image results to return. Default is `3`.
- **`client`** (QdrantClient, optional): Reusable client connection.

### `retrieve_multimodal`
```python
def retrieve_multimodal(query: str, top_k: int = 3, client: Optional[QdrantClient] = None) -> List[RetrievalResult]:
```
- **`query`** (str): The text search query.
- **`top_k`** (int): Maximum combined results to return (ranked together by similarity score).
- **`client`** (QdrantClient, optional): Reusable client connection.

---

## 3. Return Structure (`RetrievalResult`)

Each retrieval function returns a list of Pydantic models with the following schema:

```python
class RetrievalResult(BaseModel):
    id: Union[str, int]               # Qdrant Point ID (UUID string or integer)
    modality: str                     # Either "text" or "image"
    score: float                      # Cosine similarity score
    document_name: str                # Name of document source (e.g. "annual_report.pdf")
    page_number: int                  # Page number (1-indexed, or -1 if unknown)
    chunk_id: Optional[str] = None    # Unique chunk ID (only for text modality)
    image_id: Optional[str] = None    # Unique image ID (only for image modality)
    source_path: str                  # Filesystem/remote source path
    content: Optional[str] = None     # Raw text content (only for text modality)
    image_path: Optional[str] = None   # Filepath to image (only for image modality)
```

---

## 4. LangGraph State Integration

### Expected State Schema (`RetrievalState`)
The retrieval node expects and processes the following TypedDict fields in the state:

```python
from typing import TypedDict, List, Dict, Any, Optional

class RetrievalState(TypedDict, total=False):
    query: str                                # User's query input string
    route: str                                # Modality route: "text", "image", or "multimodal"
    retrieval_results: Optional[List[Dict]]   # Output storage for serialized results
    retrieved_context: Optional[str]          # Structured text summary for LLM ingestion
    conversation_state: Optional[Dict]        # Memory/memory state (preserved)
```

### Retrieval Node Function
```python
from app.retrieval import retrieval_node
```
The node performs retrieval using the `route` and `query` parameters in the state, then returns state updates without mutating other fields:
- `retrieval_results`: Serialized list of search matches (dicts of `RetrievalResult` fields).
- `retrieved_context`: Markdown-formatted context block containing all search hits sequentially (ready for ingestion by reasoning agents).

---

## 5. Error Behavior
- **Empty Queries**: Raises `ValueError` for empty/whitespace-only input strings.
- **Connection Failures**: Raises `RuntimeError` describing the connection failure when Qdrant is unavailable.
- **Missing Metadata**: If payloads in database contain missing fields, the parser gracefully fills in defaults (e.g. `"Unknown"`, `-1`) instead of throwing a KeyError, keeping the pipeline stable.

---

## 6. Example Usage in LangGraph

```python
from langgraph.graph import StateGraph
from app.retrieval import RetrievalState, retrieval_node

# 1. Define flow graph
workflow = StateGraph(RetrievalState)

# 2. Add retrieval node
workflow.add_node("retrieval_node", retrieval_node)

# 3. Add other nodes (Supervisor/LLM Generator etc.)
# workflow.add_node("supervisor", supervisor_node)
# workflow.add_node("generator", generator_node)

# 4. Compile and Run
app = workflow.compile()
state = app.invoke({
    "query": "revenue growth in 2025",
    "route": "text"
})

print(state["retrieved_context"])
```

---

## 7. Verification and Testing

To run the full test suite (including unit test mocks and real local database integration tests):
```powershell
python -m unittest tests/test_retrieval.py
```
