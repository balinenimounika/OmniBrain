README — OmniBrain Week 3
🧠 OmniBrain – Week 3: Self-RAG & Self-Correction
📌 Overview

In Week 3, OmniBrain was enhanced with a Self-RAG (Self-Retrieval-Augmented Generation) workflow.

The system can now:

Search the document knowledge base
Check whether retrieved chunks are relevant
Detect irrelevant retrieval results
Rewrite the user's query
Perform a second search
Generate the final answer using the Groq LLM
Avoid generating answers when the required information is not present
🎯 Objectives

The main objectives of Week 3 were:

Implement a Search Agent
Implement a Relevance Checker
Implement a Query Rewriter
Implement an LLM-powered Answer Node
Build the complete workflow using LangGraph
Add self-correction through query rewriting
Prevent hallucinated answers from unrelated document content
🏗️ Self-RAG Architecture
                    USER QUERY
                        │
                        ▼
                ┌───────────────┐
                │  SEARCH AGENT │
                └───────┬───────┘
                        │
                        ▼
              Retrieved Documents
                        │
                        ▼
             ┌────────────────────┐
             │ RELEVANCE CHECKER  │
             └─────────┬──────────┘
                       │
              ┌────────┴────────┐
              │                 │
          Relevant           Not Relevant
              │                 │
              ▼                 ▼
        ANSWER NODE       QUERY REWRITER
              │                 │
              │                 ▼
              │            SEARCH AGAIN
              │                 │
              │                 ▼
              │        RELEVANCE CHECK
              │                 │
              └────────┬────────┘
                       ▼
                  FINAL ANSWER
                       │
                       ▼
                      END
📂 Week 3 Project Structure
agents_week3/
│
├── __init__.py
├── answer_node.py
├── llm.py
├── query_rewriter.py
├── relevance_checker.py
├── search_agent.py
├── test_llm.py
├── requirements.txt
└── README.md

graph_week3/
│
├── __init__.py
├── self_rag_workflow.py
└── workflow.py

data/
└── output/
    └── chunks/
        └── chunks.json

test_self_rag.py
🔹 Week 3 Components
1. Search Agent

File:

agents_week3/search_agent.py

The Search Agent searches the document chunks stored in:

data/output/chunks/chunks.json

It performs keyword-based matching and assigns a retrieval score to each chunk.

The top matching chunks are returned to the next stage.

Example:

[SEARCH AGENT]
Search attempt: 1
Searching for: What technologies are used in OmniBrain?
Retrieved 3 chunks.
2. Relevance Checker

File:

agents_week3/relevance_checker.py

The Relevance Checker determines whether the retrieved chunks actually answer the user's question.

It:

Extracts meaningful words from the query
Compares them with document content
Checks important concepts
Calculates a relevance score
Removes unrelated chunks
Decides whether the retrieved information is sufficient

Example:

[RELEVANCE CHECKER]
Relevant chunks: 1

If nothing is relevant:

No relevant chunks detected.
3. Query Rewriter

File:

agents_week3/query_rewriter.py

If the retrieved documents are not relevant, the Query Rewriter generates a better search query.

Example:

Original query:
What is the company's revenue in 2024?

Rewritten query:
company revenue 2024 financial results

The rewritten query is then sent back to the Search Agent.

4. Answer Node

File:

agents_week3/answer_node.py

The Answer Node generates the final response using the retrieved document context.

It sends the prompt to the shared LLM.

result = llm.invoke(prompt)

The Answer Node follows these rules:

Use only retrieved document information
Do not invent information
Do not use outside knowledge
Give concise answers
Mention page numbers when useful
Return a safe response when information is unavailable

If the information cannot be found:

I could not find this information in the document.
🤖 LLM Integration

File:

agents_week3/llm.py

The project uses Groq ChatGroq as the LLM provider.

The LLM is loaded using the API key stored in the .env file.

GROQ_API_KEY=your_api_key

The LLM is shared with the Answer Node and Query Rewriter.

🔄 LangGraph Workflow

File:

graph_week3/self_rag_workflow.py

The workflow is implemented using StateGraph.

State

The Self-RAG state contains:

query
retrieved_docs
relevant
attempt
response
Workflow
START
  ↓
SEARCH
  ↓
RELEVANCE CHECK
  ↓
Is Relevant?
 ├── YES → ANSWER → END
 │
 └── NO
       ↓
   QUERY REWRITE
       ↓
   SEARCH AGAIN
       ↓
 RELEVANCE CHECK
       ↓
 ANSWER
       ↓
 END

The workflow allows a maximum of two search attempts before generating the final response.

🧪 Testing
1. Test LLM

Run:

python -m agents_week3.test_llm

Expected output:

LLM RESPONSE:
...

This verifies that the Groq LLM connection is working.

2. Test Self-RAG

Run:

python test_self_rag.py

This tests the complete Self-RAG pipeline.

Expected flow:

SEARCH AGENT
      ↓
RELEVANCE CHECKER
      ↓
QUERY REWRITER (if required)
      ↓
SEARCH AGAIN
      ↓
ANSWER NODE
      ↓
FINAL RESULT
🧪 Example 1 — Relevant Query

Command:

python -c "from graph_week3.self_rag_workflow import graph; print(graph.invoke({'query':'What technologies are used in OmniBrain?','attempt':1}))"

Expected behavior:

[SEARCH AGENT]
Searching for: What technologies are used in OmniBrain?

[RELEVANCE CHECKER]
Relevant chunks: 1

[ANSWER NODE]
LLM-generated answer:
OmniBrain uses a range of technologies...

The LLM generates the final answer from the retrieved document context.

🧪 Example 2 — Irrelevant Query

Command:

python test_self_rag.py

For a query such as:

What is the company's revenue in 2024?

the document does not contain the required revenue information.

The system performs:

Search Attempt 1
       ↓
No Relevant Chunks
       ↓
Query Rewrite
       ↓
Search Attempt 2
       ↓
No Relevant Chunks
       ↓
Safe Final Response

Final response:

I could not find relevant information in the document.

This is an intentional negative test to verify that the system does not hallucinate an answer from unrelated content.

🛡️ Hallucination Prevention

One of the main improvements in Week 3 is preventing the LLM from answering questions using unrelated information.

For example:

Question:
What is the company's revenue in 2024?

If the retrieved document only contains information about:

Reinforcement Learning
PyTorch
PettingZoo
React
Three.js

the system should not invent a revenue value.

Instead:

I could not find this information in the document.
🛠️ Technologies Used
Python
LangGraph
LangChain
Groq
ChatGroq
LLM
Retrieval-Augmented Generation (RAG)
Self-RAG
Regular Expressions
JSON
TypedDict
Python Virtual Environment
📊 Week 3 Workflow Output

A successful run demonstrates:

✅ Document Search
        ↓
✅ Relevance Detection
        ↓
✅ Query Rewriting
        ↓
✅ Retry Mechanism
        ↓
✅ LLM Answer Generation
        ↓
✅ Hallucination Prevention
✅ Week 3 Achievements

By the end of Week 3, OmniBrain successfully implemented:

✅ Search Agent
✅ Relevance Checker
✅ Query Rewriter
✅ LLM-powered Answer Node
✅ Groq LLM integration
✅ LangGraph Self-RAG workflow
✅ Automatic search retry
✅ Irrelevant-context detection
✅ Safe handling of unavailable information
✅ End-to-end Self-RAG testing
🚀 Week 3 Final Outcome

The OmniBrain system evolved from a basic retrieval pipeline into a self-correcting RAG system.

User Query
    ↓
Retrieve
    ↓
Evaluate
    ↓
Correct if Needed
    ↓
Retrieve Again
    ↓
Generate Answer

This makes the system more reliable by ensuring th