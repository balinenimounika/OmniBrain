from graph_week3.self_rag_workflow import graph as self_rag_graph
from app.guardrails.guardrails import should_answer_question, get_guardrail_response

def text_agent(state):
    query = state["query"]

    print("\n[TEXT AGENT]")
    print("Running Self-RAG...")

    result = self_rag_graph.invoke({
        "query": query,
        "attempt": 1
    })

    retrieved_docs = result.get("retrieved_docs", [])
    response = result.get("response", "")

    guardrail_state = {
        "query": query,
        "user_query": query,
        "retrieved_context": retrieved_docs
    }

    guardrail_result = should_answer_question(guardrail_state)

    if not guardrail_result.get("guardrail_allowed", True):
        response = get_guardrail_response(guardrail_result) or response

    return {
        "response": response,
        "next_agent": "text"
    }
