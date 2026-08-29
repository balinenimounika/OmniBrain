import json
import os
import re


# --------------------------------------------------
# Path to Week 1 chunks
# --------------------------------------------------

CHUNKS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "output",
    "chunks",
    "chunks.json"
)


# --------------------------------------------------
# Load chunks
# --------------------------------------------------

def load_chunks():
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"\nChunks file not found:\n{CHUNKS_PATH}"
        )

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# Extract useful words
# --------------------------------------------------

def clean_words(text):
    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )


# --------------------------------------------------
# Search Agent
# --------------------------------------------------

def search_agent(state):

    query = state["query"]
    attempt = state.get("attempt", 1)

    print("\n[SEARCH AGENT]")
    print(f"Search attempt: {attempt}")
    print(f"Searching for: {query}")

    chunks = load_chunks()

    query_words = clean_words(query)

    results = []

    for chunk in chunks:

        chunk_text = chunk.get("text", "")

        if not chunk_text:
            continue

        chunk_words = clean_words(chunk_text)

        # Basic keyword matching
        matches = query_words.intersection(chunk_words)

        score = len(matches)

        # ------------------------------------------
        # Important phrase matching
        # ------------------------------------------

        query_lower = query.lower()
        text_lower = chunk_text.lower()

        # Week 2
        if "week 2" in query_lower and "week 2" in text_lower:
            score += 5

        # Revenue
        if "revenue" in query_lower and "revenue" in text_lower:
            score += 5

        # Supervisor
        if "supervisor" in query_lower and "supervisor" in text_lower:
            score += 3

        # LangGraph
        if "langgraph" in query_lower and "langgraph" in text_lower:
            score += 3

        # Development plan
        if (
            "development plan" in query_lower
            and "development plan" in text_lower
        ):
            score += 3

        # ------------------------------------------
        # Store result if there is a match
        # ------------------------------------------

        if score > 0:

            results.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "page": chunk.get("page"),
                    "text": chunk_text,
                    "score": score
                }
            )

    # Sort highest score first
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # Return top 3
    results = results[:3]

    print(f"Retrieved {len(results)} chunks.")

    for result in results:

        print(
            f"  Chunk {result['chunk_id']} "
            f"| Page {result['page']} "
            f"| Score {result['score']}"
        )

    return {
        "retrieved_docs": results
    }