from graph_week3.self_rag_workflow import graph as self_rag_graph


def text_agent(state):

    query = state["query"]

    print("\n[TEXT AGENT]")
    print("Handling document/text query...")

    print("\n[SELF-RAG AGENT]")
    print("Sending query to Week 3 Self-RAG...")

    result = self_rag_graph.invoke({
        "query": query
    })

    attempt = result.get("attempt", 1)
    relevant = result.get("relevant", False)
    retrieved_docs = result.get("retrieved_docs", [])

    return {
        "response": result.get(
            "response",
            "No response generated."
        ),
        "retrieved_docs": retrieved_docs,
        "relevant": relevant,
        "attempt": attempt,
        "query_rewritten": attempt > 1
    }