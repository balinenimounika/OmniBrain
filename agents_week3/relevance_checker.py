import re


def get_words(text):
    return set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))


def relevance_checker(state):
    query = state["query"]
    documents = state.get("retrieved_docs", [])

    print("\n[RELEVANCE CHECKER]")

    query_lower = query.lower()
    query_words = get_words(query)

    stop_words = {
        "what", "is", "are", "was", "were", "the", "a", "an",
        "and", "or", "of", "to", "in", "on", "for", "from",
        "about", "does", "do", "how", "why", "when", "where",
        "which", "who", "can", "could", "would", "should",
        "this", "that", "document", "say", "tell", "me"
    }

    meaningful_words = query_words - stop_words

    relevant_docs = []

    technology_terms = {
        "python", "pytorch", "tensorflow", "react", "three",
        "langgraph", "langchain", "fastapi", "mongodb",
        "mysql", "postgresql", "opencv", "mediapipe",
        "yolo", "numpy", "pandas", "javascript", "typescript"
    }

    for doc in documents:
        text = doc.get("text", "")
        text_lower = text.lower()
        text_words = get_words(text)

        score = 0

        if "revenue" in query_lower and "revenue" in text_lower:
            score += 10

        if "stock price" in query_lower:
            if "stock" in text_lower:
                score += 5
            if "price" in text_lower:
                score += 5

        if "week 2" in query_lower and "week 2" in text_lower:
            score += 5

        if "supervisor" in query_lower and "supervisor" in text_lower:
            score += 5

        if "langgraph" in query_lower and "langgraph" in text_lower:
            score += 5

        technology_question = any(
            x in query_lower
            for x in [
                "technology",
                "technologies",
                "tech stack",
                "tools",
                "frameworks",
                "libraries"
            ]
        )

        if technology_question:
            score += len(
                technology_terms.intersection(text_words)
            ) * 2

        useful_matches = (
            meaningful_words.intersection(text_words)
            - {
                "project",
                "development",
                "architecture",
                "information",
                "system",
                "data",
                "plan",
                "company"
            }
        )

        score += len(useful_matches)

        if score >= 1:
            relevant_docs.append(doc)

    if relevant_docs:
        print(f"Relevant chunks: {len(relevant_docs)}")
        return {
            "retrieved_docs": relevant_docs,
            "relevant": True
        }

    print("No relevant chunks detected.")

    return {
        "retrieved_docs": [],
        "relevant": False
    }