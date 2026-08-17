def sql_agent(state):

    query = state["query"]

    print("\n[SQL AGENT]")
    print("Handling structured/database query...")

    return {
        "response": f"SQL Agent selected for: {query}"
    }