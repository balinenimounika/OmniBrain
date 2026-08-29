from typing import Literal
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


def supervisor(state):
    """
    Supervisor uses Groq LLM to decide which agent
    should handle the user query.
    """

    query = state["query"]

    prompt = f"""
You are the supervisor of an AI system called OmniBrain.

Choose exactly ONE agent for the given query:

- text → questions about document text, explanations, summaries, facts
- sql → questions requiring numerical data, calculations, tables, database information
- vision → questions about images, charts, graphs, figures, diagrams

User query:
{query}

Respond with ONLY one word:
text
sql
or
vision
"""

    response = llm.invoke(prompt)

    decision = response.content.strip().lower()

    # Safety check
    if decision not in ["text", "sql", "vision"]:
        decision = "text"

    print(f"🤖 Supervisor selected: {decision}")

    return {
        "next_agent": decision
    }


def route_query(state) -> Literal["text", "sql", "vision"]:
    return state["next_agent"]