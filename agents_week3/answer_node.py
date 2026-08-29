from agents_week3.llm import llm


def answer_node(state):

    query = state["query"]
    documents = state.get("retrieved_docs", [])

    print("\n[ANSWER NODE]")

    if not documents:
        response = "I could not find relevant information in the document."

        print(response)

        return {
            "response": response
        }

    context_parts = []

    for doc in documents:
        page = doc.get("page", "Unknown")
        text = doc.get("text", "")

        context_parts.append(
            f"Page {page}:\n{text}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are the Answer Agent of OmniBrain.

Answer the user's question using ONLY the retrieved document context.

Rules:
- Do not invent information.
- Do not use outside knowledge.
- If the answer is not present in the context, say:
  "I could not find this information in the document."
- Give a concise and direct answer.
- Mention the page number when useful.
- Do not copy large portions of the document.

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

ANSWER:
"""

    try:
        result = llm.invoke(prompt)

        answer = result.content

        print("LLM-generated answer:")
        print(answer)

        return {
            "response": answer
        }

    except Exception as e:

        print("❌ LLM error:", e)

        return {
            "response": "Unable to generate an answer."
        }