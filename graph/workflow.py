from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from agents.supervisor import supervisor, route_query
from agents.text_agent import text_agent
from agents.sql_agent import sql_agent
from agents.vision_agent import vision_agent


class GraphState(TypedDict, total=False):
    query: str
    next_agent: str
    response: str


builder = StateGraph(GraphState)

builder.add_node("supervisor", supervisor)
builder.add_node("text", text_agent)
builder.add_node("sql", sql_agent)
builder.add_node("vision", vision_agent)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_query,
    {
        "text": "text",
        "sql": "sql",
        "vision": "vision"
    }
)

builder.add_edge("text", END)
builder.add_edge("sql", END)
builder.add_edge("vision", END)

graph = builder.compile()