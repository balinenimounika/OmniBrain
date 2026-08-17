def vision_agent(state):
    """
    Handles image, chart, graph and visual queries.
    """

    query = state["query"]

    print("\n[VISION AGENT]")
    print("Handling image/chart query...")

    return {
        "response": f"Vision Agent selected for: {query}"
    }