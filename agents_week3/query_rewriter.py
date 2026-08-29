def query_rewriter(state):
    """
    Rewrites the query when the retrieved information
    is considered irrelevant.
    """

    original_query = state["query"]
    attempt = state.get("attempt", 1)

    print("\n[QUERY REWRITER]")
    print("Original query:", original_query)

    # Remove unnecessary wording and create a cleaner search query
    rewritten_query = original_query.strip()

    # Query-specific improvements
    if "revenue" in original_query.lower():
        rewritten_query = (
            "company revenue 2024 financial results"
        )

    elif "week 2" in original_query.lower():
        rewritten_query = (
            "Week 2 development plan Agentic Architecture LangGraph Supervisor"
        )

    elif "omnibrain" in original_query.lower():
        rewritten_query = (
            "OmniBrain project development architecture objectives"
        )

    else:
        rewritten_query = (
            f"information about {original_query}"
        )

    print("Rewritten query:", rewritten_query)

    return {
        "query": rewritten_query,
        "attempt": attempt + 1
    }