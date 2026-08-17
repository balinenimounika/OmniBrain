from typing import Literal


def supervisor(state):
    """
    Supervisor decides which agent should handle the query.
    """

    query = state["query"].lower()

    sql_keywords = [
        "stock price",
        "revenue by year",
        "sales by year",
        "database",
        "sql",
        "average",
        "total",
        "sum",
        "count"
    ]

    vision_keywords = [
        "chart",
        "graph",
        "image",
        "figure",
        "diagram",
        "bar chart",
        "pie chart",
        "visual"
    ]

    for keyword in sql_keywords:
        if keyword in query:
            return {"next_agent": "sql"}

    for keyword in vision_keywords:
        if keyword in query:
            return {"next_agent": "vision"}

    return {"next_agent": "text"}


def route_query(state) -> Literal["text", "sql", "vision"]:
    return state["next_agent"]