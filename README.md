# 🧠 OmniBrain – Week 3

## Self-RAG & Self-Correction

### 📌 Overview

In Week 3, OmniBrain was enhanced with a **Self-RAG (Self-Retrieval-Augmented Generation)** workflow.

The system can evaluate retrieved information, detect irrelevant results, rewrite the query, perform another search, and generate a final answer using the Groq LLM.

---

## 🎯 Objectives

The main objectives of Week 3 are:

- Implement a Search Agent
- Implement a Relevance Checker
- Implement a Query Rewriter
- Implement an LLM-powered Answer Node
- Build the workflow using LangGraph
- Add self-correction through query rewriting
- Reduce hallucinations by using only retrieved document context
- Handle queries when the required information is not available

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
                +-------+-------+
                |               |
            Relevant        Not Relevant
                |               |
                v               v
          +-----------+   +---------------+
          | ANSWER    |   | QUERY         |
          | NODE      |   | REWRITER      |
          +-----+-----+   +-------+-------+
                |                 |
                |                 v
                |           SEARCH AGAIN
                |                 |
                |                 v
                |          RELEVANCE CHECK
                |                 |
                +--------+--------+
                         |
                         v
                   FINAL ANSWER
                         |
                         v
                        END