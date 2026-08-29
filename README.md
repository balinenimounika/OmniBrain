# 🧠 OmniBrain – Week 3

## Self-RAG & Self-Correction

---

## 📌 Overview

In Week 3, OmniBrain was enhanced with a Self-RAG (Self-Retrieval-Augmented Generation) workflow.

The system can:

- Search the document knowledge base
- Retrieve relevant document chunks
- Check whether retrieved chunks are relevant
- Detect irrelevant retrieval results
- Rewrite the user's query
- Perform a second search
- Generate the final answer using the Groq LLM
- Prevent unsupported or hallucinated answers

---

## 🎯 Objectives

The main objectives of Week 3 are:

- Implement a Search Agent
- Implement a Relevance Checker
- Implement a Query Rewriter
- Implement an LLM-powered Answer Node
- Build the complete workflow using LangGraph
- Add self-correction through query rewriting
- Improve retrieval quality
- Reduce hallucinations
- Handle questions when the required information is not available

---

## 🏗️ Self-RAG Architecture

```text
                    USER QUERY
                        |
                        v
                +---------------+
                |  SEARCH AGENT |
                +-------+-------+
                        |
                        v
                RETRIEVED DOCUMENTS
                        |
                        v
              +--------------------+
              | RELEVANCE CHECKER  |
              +---------+----------+
                        |
              +---------+---------+
              |                   |
          RELEVANT           NOT RELEVANT
              |                   |
              v                   v
        +-----------+       +---------------+
        | ANSWER    |       | QUERY         |
        | NODE      |       | REWRITER      |
        +-----+-----+       +-------+-------+
              |                     |
              |                     v
              |               SEARCH AGAIN
              |                     |
              |                     v
              |              RELEVANCE CHECK
              |                     |
              +----------+----------+
                         |
                         v
                   FINAL ANSWER
                         |
                         v
                        END
📂 Project Structure
OmniBrain/
│
├── agents_week3/
│   ├── __init__.py
│   ├── answer_node.py
│   ├── llm.py
│   ├── query_rewriter.py
│   ├── relevance_checker.py
│   ├── search_agent.py
│   ├── test_llm.py
│   ├── requirements.txt
│   └── README.md
│
├── graph_week3/
│   ├── __init__.py
│   ├── self_rag_workflow.py
│   └── workflow.py
│
├── data/
│   └── output/
│       └── chunks/
│           └── chunks.json
│
└── test_self_rag.py
🔹 Week 3 Components
1. Search Agent
File
agents_week3/search_agent.py

The Search Agent searches the document knowledge base stored in:

data/output/chunks/chunks.json

It performs keyword-based matching between the user query and document chunks.

The agent:

Loads document chunks
Extracts words from the query
Compares query words with document words
Calculates a retrieval score
Sorts the results
Returns the top 3 matching chunks
Example
[SEARCH AGENT]
Search attempt: 1
Searching for: What technologies are used in OmniBrain?
Retrieved 3 chunks.
2. Relevance Checker
File
agents_week3/relevance_checker.py

The Relevance Checker determines whether the retrieved chunks are actually relevant to the user's question.

It:

Extracts meaningful words
Removes common stop words
Compares query terms with document content
Checks important concepts
Calculates a relevance score
Removes unrelated chunks
Determines whether relevant information exists
Example
[RELEVANCE CHECKER]
Relevant chunks: 1

If no relevant information is found:

[RELEVANCE CHECKER]
No relevant chunks detected.
3. Query Rewriter
File
agents_week3/query_rewriter.py

If the retrieved documents are not relevant, the Query Rewriter creates a new search query.

Example
Original query:
What is the company's revenue in 2024?

Rewritten query:
company revenue 2024 financial results

The rewritten query is then sent back to the Search Agent.

This creates the self-correction loop.

4. Answer Node
File
agents_week3/answer_node.py

The Answer Node generates the final response using the retrieved document context.

It uses the Groq LLM through:

result = llm.invoke(prompt)

The Answer Node follows these rules:

Use only retrieved document context
Do not invent information
Do not use outside knowledge
Give concise answers
Mention page numbers when useful
Return a safe response if information is unavailable
Example
[ANSWER NODE]

LLM-generated answer:

OmniBrain uses a range of technologies including
LangGraph, Ray, React, and other AI frameworks.
5. LLM Integration
File
agents_week3/llm.py

The LLM configuration is centralized in this file.

The project uses Groq ChatGroq for LLM-based response generation.

The API key is loaded from the .env file.

GROQ_API_KEY=your_api_key

The shared LLM is used by the Answer Node and other LLM-based components.

🔄 LangGraph Workflow
File
graph_week3/self_rag_workflow.py

The Self-RAG workflow is implemented using LangGraph StateGraph.

The workflow maintains the following state:

query
retrieved_docs
relevant
attempt
response
Workflow
START
  |
  v
SEARCH
  |
  v
RELEVANCE CHECK
  |
  +-----------------------+
  |                       |
  v                       v
RELEVANT              NOT RELEVANT
  |                       |
  v                       v
ANSWER               QUERY REWRITE
  |                       |
  |                       v
  |                  SEARCH AGAIN
  |                       |
  |                       v
  |                 RELEVANCE CHECK
  |                       |
  +-----------+-----------+
              |
              v
           ANSWER
              |
              v
             END

The workflow supports up to two search attempts.

🧠 Self-Correction Process

The main feature introduced in Week 3 is self-correction.

The system does not immediately trust the retrieved documents.

Instead:

User Query
    |
    v
Search
    |
    v
Check Relevance
    |
    +---- Relevant ----> Answer
    |
    +---- Not Relevant
             |
             v
       Rewrite Query
             |
             v
        Search Again
             |
             v
       Check Relevance
             |
             v
           Answer

This helps improve the reliability of the retrieval process.

🛡️ Hallucination Prevention

The Answer Node is instructed to answer using only the retrieved document context.

For example:

Question:
What is the company's revenue in 2024?

If the retrieved documents contain only information about:

Reinforcement Learning
PyTorch
PettingZoo
React
Three.js

the system should not invent a revenue value.

Instead, it returns:

I could not find this information in the document.

This prevents the LLM from generating unsupported information.

🧪 Testing
Test 1 – LLM Connection

Run:

python -m agents_week3.test_llm

Expected output:

LLM RESPONSE:
...

This verifies that the Groq LLM connection is working correctly.

Test 2 – Complete Self-RAG Workflow

Run:

python test_self_rag.py

This tests the complete Self-RAG pipeline.

Expected flow:

SEARCH AGENT
      |
      v
RELEVANCE CHECKER
      |
      +---- Relevant ----> ANSWER NODE
      |
      +---- Not Relevant
                |
                v
          QUERY REWRITER
                |
                v
           SEARCH AGAIN
                |
                v
        RELEVANCE CHECKER
                |
                v
           ANSWER NODE
                |
                v
          FINAL RESULT
🧪 Test Case 1 – Relevant Query

Run:

python -c "from graph_week3.self_rag_workflow import graph; print(graph.invoke({'query':'What technologies are used in OmniBrain?','attempt':1}))"

Expected behavior:

[SEARCH AGENT]
Search attempt: 1
Searching for: What technologies are used in OmniBrain?
Retrieved 3 chunks.

[RELEVANCE CHECKER]
Relevant chunks: 1

[ANSWER NODE]
LLM-generated answer:
...

The final answer is generated by the Groq LLM using the retrieved document context.

🧪 Test Case 2 – Irrelevant Query

Example query:

What is the company's revenue in 2024?

If the document does not contain revenue information, the system should not invent an answer.

The workflow becomes:

Search Attempt 1
       |
       v
No Relevant Chunks
       |
       v
Query Rewrite
       |
       v
Search Attempt 2
       |
       v
No Relevant Chunks
       |
       v
Safe Final Response

Expected response:

I could not find relevant information in the document.

This is an intentional negative test for hallucination prevention.

🧪 Test Case 3 – Technology Query

Example:

What technologies are used in OmniBrain?

Expected behavior:

[SEARCH AGENT]
Searching for: What technologies are used in OmniBrain?

[RELEVANCE CHECKER]
Relevant chunks: 1

[ANSWER NODE]
LLM-generated answer:
OmniBrain uses a range of technologies including:
- Ray
- LangGraph
- React
- WebGL
- Vision-Language Models
- Other AI frameworks

The answer is generated from the retrieved document context.

🛠️ Technologies Used
Python
LangGraph
LangChain
Groq
ChatGroq
Large Language Models (LLMs)
Retrieval-Augmented Generation (RAG)
Self-RAG
Regular Expressions
JSON
TypedDict
Python Virtual Environment
📊 Week 3 Execution Flow
                  USER QUERY
                      |
                      v
                SEARCH AGENT
                      |
                      v
              DOCUMENT CHUNKS
                      |
                      v
             RELEVANCE CHECKER
                      |
             +--------+--------+
             |                 |
          RELEVANT         NOT RELEVANT
             |                 |
             v                 v
        ANSWER NODE      QUERY REWRITER
             |                 |
             |                 v
             |            SEARCH AGAIN
             |                 |
             |                 v
             |          RELEVANCE CHECK
             |                 |
             +--------+--------+
                      |
                      v
                 GROQ LLM
                      |
                      v
                FINAL ANSWER
📈 Example Successful Output
============================================================
SELF-RAG TEST
============================================================

[SEARCH AGENT]
Search attempt: 1
Searching for: What technologies are used in OmniBrain?
Retrieved 3 chunks.

[RELEVANCE CHECKER]
Relevant chunks: 1

[ANSWER NODE]
LLM-generated answer:
OmniBrain uses a range of technologies including
Ray, LangGraph, React, WebGL and advanced AI models.

============================================================
FINAL RESULT
============================================================
❌ Example Negative Test Output
============================================================
SELF-RAG TEST
============================================================

[SEARCH AGENT]
Search attempt: 1
Searching for: What is the company's revenue in 2024?
Retrieved 3 chunks.

[RELEVANCE CHECKER]
No relevant chunks detected.

[QUERY REWRITER]
Original query:
What is the company's revenue in 2024?

Rewritten query:
company revenue 2024 financial results

[SEARCH AGENT]
Search attempt: 2
Searching for: company revenue 2024 financial results
Retrieved 3 chunks.

[RELEVANCE CHECKER]
No relevant chunks detected.

[ANSWER NODE]
I could not find relevant information in the document.

============================================================
FINAL RESULT
============================================================
🔐 Important Design Principle

OmniBrain follows a simple principle:

RETRIEVE
   ↓
VERIFY
   ↓
CORRECT
   ↓
ANSWER

The system does not blindly trust retrieved information.

It first checks relevance and attempts query correction when necessary.

✅ Week 3 Achievements

By the end of Week 3, OmniBrain successfully implemented:

✅ Search Agent
✅ Relevance Checker
✅ Query Rewriter
✅ LLM-powered Answer Node
✅ Groq LLM Integration
✅ LangGraph Self-RAG Workflow
✅ Automatic Search Retry
✅ Query Rewriting
✅ Irrelevant Context Detection
✅ Relevance Threshold
✅ Hallucination Prevention
✅ Safe Handling of Missing Information
✅ End-to-End Self-RAG Testing
🚀 Final Outcome

Week 3 transforms OmniBrain from a basic RAG pipeline into a self-correcting RAG system.

The complete process is:

USER QUERY
    ↓
SEARCH
    ↓
RETRIEVE DOCUMENTS
    ↓
CHECK RELEVANCE
    ↓
RELEVANT?
   /   \
 YES    NO
  |      |
  |      v
  |   REWRITE QUERY
  |      |
  |      v
  |   SEARCH AGAIN
  |      |
  |      v
  |   CHECK AGAIN
  |      |
  +------+
    ↓
ANSWER NODE
    ↓
GROQ LLM
    ↓
FINAL ANSWER
🎯 Week 3 Conclusion

The Week 3 implementation adds self-correction, relevance evaluation, query rewriting, and LLM-based answer generation to OmniBrain.
