from typing import Literal
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


def supervisor(state):
    """
    Supervisor decides which agent should handle the user query.
    """

    query = state["query"]

    prompt = f"""
You are the supervisor of an AI system called OmniBrain.

Choose exactly ONE agent for the given query.

IMPORTANT ROUTING RULES:

1. Choose TEXT for:
- Questions about information contained in uploaded documents
- Document facts, explanations, summaries, plans, descriptions
- Questions asking "according to the document"
- Questions asking what the document says
- Questions about information that should be searched in document chunks
- Questions about document content even if the answer contains numbers

2. Choose SQL for:
- Questions explicitly asking for database information
- Structured database queries
- SQL operations
- Calculations using database records
- Tables stored in the database
- Stock/database records when the query clearly refers to the database

3. Choose VISION for:
- Images
- Charts
- Graphs
- Figures
- Diagrams
- Visual information

IMPORTANT:
A question containing numbers, years, revenue, price, quantity, etc.
does NOT automatically mean SQL.

If the user is asking about information from the document,
choose TEXT.

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

    if decision not in ["text", "sql", "vision"]:
        decision = "text"

    print(f"\n[SUPERVISOR] Selected agent: {decision}")

    return {
        "next_agent": decision
    }


def route_query(state) -> Literal["text", "sql", "vision"]:
    return state["next_agent"]