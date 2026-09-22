def answer_node(state):
    query = state.get("query", "")
    documents = state.get("retrieved_docs", [])

    print("\n[ANSWER NODE]")

    if not documents:
        response = "I could not find relevant information in the document."
        print(response)
        return {"response": response}

    answers = []

    for doc in documents[:3]:
        page = doc.get("page", "Unknown")
        text = doc.get("text", "").strip()

        if text:
            answers.append(f"Page {page}: {text}")

    if not answers:
        response = "I could not find relevant information in the document."
    else:
        response = "Based on the retrieved document context:\n\n" + "\n\n".join(answers)

    print("Document-based answer:")
    print(response)

    return {"response": response}
