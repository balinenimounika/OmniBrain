def text_agent(state):

    query = state["query"]

    print("\n[TEXT AGENT]")
    print("Handling document/text query...")

    return {
        "response": f"Text Agent selected for: {query}"
    }