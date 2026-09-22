from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from agents_week3.search_agent import search_agent
from agents_week3.relevance_checker import relevance_checker
from agents_week3.query_rewriter import query_rewriter
from agents_week3.answer_node import answer_node


class SelfRAGState(TypedDict, total=False):
    query: str
    retrieved_docs: list
    relevant: bool
    attempt: int
    response: str


def route_after_relevance(state):
    if state.get("relevant") and state.get("retrieved_docs"):
        return "answer"

    if state.get("attempt", 1) < 2:
        return "rewrite"

    return "answer"


builder = StateGraph(SelfRAGState)

builder.add_node("search", search_agent)
builder.add_node("check_relevance", relevance_checker)
builder.add_node("rewrite", query_rewriter)
builder.add_node("answer", answer_node)

builder.add_edge(START, "search")
builder.add_edge("search", "check_relevance")

builder.add_conditional_edges(
    "check_relevance",
    route_after_relevance,
    {
        "answer": "answer",
        "rewrite": "rewrite"
    }
)

builder.add_edge("rewrite", "search")
builder.add_edge("answer", END)

graph = builder.compile()
