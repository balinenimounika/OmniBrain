import re


def get_words(text):
    return set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )


def relevance_checker(state):

    query = state["query"]
    documents = state.get("retrieved_docs", [])

    print("\n[RELEVANCE CHECKER]")

    query_lower = query.lower()
    query_words = get_words(query)

    stop_words = {
        "what", "is", "are", "was", "were",
        "the", "a", "an", "and", "or",
        "of", "to", "in", "on", "for",
        "from", "about", "does", "do",
        "how", "why", "when", "where",
        "which", "who", "can", "could",
        "would", "should", "this", "that",
        "document", "say", "tell",
        "used", "use"
    }

    meaningful_words = query_words - stop_words

    relevant_docs = []

    for doc in documents:

        text = doc.get("text", "")
        text_lower = text.lower()
        text_words = get_words(text)

        score = 0

        # -----------------------------------------
        # 1. Strong concept matching
        # -----------------------------------------

        if "revenue" in query_lower:
            if "revenue" in text_lower:
                score += 10

        if "stock price" in query_lower:
            if "stock" in text_lower:
                score += 5
            if "price" in text_lower:
                score += 5

        if "week 2" in query_lower:
            if "week 2" in text_lower:
                score += 5

        if "supervisor" in query_lower:
            if "supervisor" in text_lower:
                score += 5

        if "langgraph" in query_lower:
            if "langgraph" in text_lower:
                score += 5

        # -----------------------------------------
        # 2. Technology-related questions
        # -----------------------------------------

        technology_terms = {
            "python",
            "pytorch",
            "tensorflow",
            "react",
            "three.js",
            "three",
            "langgraph",
            "langchain",
            "pettingzoo",
            "openai",
            "gym",
            "rllib",
            "ray",
            "javascript",
            "typescript",
            "fastapi",
            "mongodb",
            "mysql",
            "postgresql",
            "opencv",
            "mediapipe",
            "yolo",
            "numpy",
            "pandas"
        }

        technology_question = (
            "technology" in query_lower
            or "technologies" in query_lower
            or "tech stack" in query_lower
            or "tools" in query_lower
            or "frameworks" in query_lower
            or "libraries" in query_lower
        )

        if technology_question:

            technology_matches = (
                technology_terms.intersection(text_words)
            )

            score += len(technology_matches) * 2

        # -----------------------------------------
        # 3. General meaningful keyword overlap
        # -----------------------------------------

        matched_words = meaningful_words.intersection(text_words)

        generic_words = {
            "project",
            "development",
            "architecture",
            "information",
            "system",
            "data",
            "plan",
            "company"
        }

        useful_matches = matched_words - generic_words

        score += len(useful_matches)

    

        # -----------------------------------------
# 5. Prevent unrelated chunks for specific
#    questions
# -----------------------------------------

    if "revenue" in query_lower and "revenue" not in text_lower:
            score = 0

# -----------------------------------------
# 6. Strict relevance threshold
# -----------------------------------------

    if score >= 2:
        relevant_docs.append(doc)

    # -----------------------------------------
    # 7. Final result
    # -----------------------------------------

    if relevant_docs:

        print(
            f"Relevant chunks: {len(relevant_docs)}"
        )

        return {
            "retrieved_docs": relevant_docs,
            "relevant": True
        }

    print("No relevant chunks detected.")

    return {
        "retrieved_docs": [],
        "relevant": False
    }