# 🧠 OmniBrain – Agentic Multi-Modal RAG Orchestrator

## 📌 Overview

OmniBrain is an Agentic Multi-Modal RAG Orchestrator designed to process documents, understand user queries, route them to specialized AI agents, retrieve relevant information, verify retrieved content, rewrite queries when necessary, and generate grounded responses.

The project was developed progressively over four weeks:

```text
Week 1 → Multi-Modal Document Ingestion
        ↓
Week 2 → Multi-Agent Architecture & Query Routing
        ↓
Week 3 → Self-RAG & Self-Correction
        ↓
Week 4 → Final Integration, Testing & Explainable UI

The final system combines:

PDF document processing
Text extraction
Text chunking
Image extraction
Multi-agent architecture
LangGraph Supervisor
Text Agent
SQL Agent
Vision Agent
Self-RAG
Relevance checking
Query rewriting
Retrieval retry
Grounded answer generation
Source citations
Vision image display
FastAPI backend
React frontend
🎯 Project Objectives

The main objectives of OmniBrain are:

Build a multi-modal document ingestion pipeline
Extract text and images from PDF documents
Divide documents into searchable text chunks
Build a Supervisor-based multi-agent architecture
Implement intelligent query routing
Create specialized Text, SQL, and Vision agents
Implement Self-RAG
Check retrieved information for relevance
Rewrite queries when relevant information is not retrieved
Retry the search with the rewritten query
Prevent unsupported or hallucinated answers
Provide source and page-level citations
Display extracted images for Vision queries
Build a complete FastAPI backend
Build an interactive React frontend
Test the complete end-to-end workflow
📅 WEEK 1 — Multi-Modal Document Ingestion
📌 Overview

In Week 1, OmniBrain was developed with a basic document ingestion pipeline.

The main goal was to process PDF documents and extract both textual and visual information.

🎯 Objectives
Parse PDF documents
Extract text page by page
Divide text into smaller chunks
Extract images from PDF documents
Store extracted text
Store generated chunks
Store extracted images
Create the initial ingestion pipeline
🏗️ Architecture
                    PDF Document
                         │
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
      Text Extraction         Image Extraction
             │                       │
             ▼                       ▼
       Page-wise Text          Extracted Images
             │
             ▼
        Text Chunking
             │
             ▼
        chunks.json
📂 Week 1 Files
ingestion/

├── pdf_parser.py
├── text_chunker.py
├── image_extractor.py
├── pipeline.py
└── README.md
🔹 Components
1. PDF Parser

pdf_parser.py

Responsible for extracting text from PDF documents page by page.

2. Text Chunker

text_chunker.py

Divides extracted document text into smaller chunks that can later be used for retrieval.

3. Image Extractor

image_extractor.py

Extracts images embedded in the PDF document.

4. Ingestion Pipeline

pipeline.py

Coordinates the complete document ingestion process.

📊 Example Output
Starting OmniBrain PDF ingestion...
[1/3] Extracting text...
Extracted text from 6 pages.

[2/3] Creating text chunks...
Created 18 chunks.

[3/3] Extracting images...
Extracted 6 images.

OmniBrain ingestion completed!
📂 Output Structure
data/

├── input/
│   └── sample.pdf
│
└── output/
    │
    ├── text/
    │   ├── page_001.txt
    │   ├── page_002.txt
    │   ├── page_003.txt
    │   ├── page_004.txt
    │   ├── page_005.txt
    │   └── page_006.txt
    │
    ├── chunks/
    │   └── chunks.json
    │
    └── images/
        ├── images.json
        ├── page_001_img_001.jpeg
        ├── page_001_img_001.png
        ├── page_002_img_001.jpeg
        ├── page_003_img_001.jpeg
        ├── page_004_img_001.jpeg
        ├── page_005_img_001.jpeg
        └── page_006_img_001.jpeg
🛠️ Technologies Used
Python
PyMuPDF
PDF Processing
Text Chunking
Image Extraction
JSON
✅ Week 1 Outcome

By the end of Week 1, OmniBrain successfully implemented:

✅ PDF parsing
✅ Page-wise text extraction
✅ Text chunking
✅ Image extraction
✅ Structured output generation
✅ Initial document ingestion pipeline
📅 WEEK 2 — Multi-Agent Architecture & Query Routing
📌 Overview

In Week 2, OmniBrain was extended from the basic ingestion pipeline into a multi-agent AI system using LangGraph.

The main goal was to create a Supervisor Agent that analyzes the user's query and routes it to the most appropriate specialized agent.

🎯 Objectives
Build a Supervisor Agent
Implement intelligent query routing
Create specialized agents
Build a LangGraph state-based workflow
Integrate Text, SQL, and Vision agents
Test the complete routing workflow
🏗️ Architecture
                    User Query
                        │
                        ▼
                ┌───────────────┐
                │   Supervisor  │
                │     Agent     │
                └───────┬───────┘
                        │
                 Query Classification
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     Text Agent     SQL Agent    Vision Agent
          │             │             │
          └─────────────┼─────────────┘
                        │
                        ▼
                    Response
📂 Week 2 Files
agents_week2/

├── supervisor.py
├── text_agent.py
├── sql_agent.py
├── vision_agent.py
└── __init__.py


graph_week2/

├── workflow.py
└── __init__.py


test_supervisor.py
🔹 Components
1. Supervisor Agent

supervisor.py

The Supervisor Agent acts as the central decision-maker.

It analyzes the user's query and determines which agent should handle it.

Example:

"What is the project architecture?"
             ↓
        Text Agent
"Show me the employee records"
             ↓
         SQL Agent
"Analyze this image"
             ↓
       Vision Agent
2. Text Agent

text_agent.py

Handles queries related to:

Documents
Text information
Project descriptions
Document-based questions
3. SQL Agent

sql_agent.py

Handles structured-data queries that require interaction with a database.

Examples:

"Show all users"

"What is the total number of records?"
4. Vision Agent

vision_agent.py

Handles image-related queries.

Examples:

"Describe this image"

"What objects are present?"
🔄 LangGraph Workflow

The workflow is implemented using LangGraph StateGraph.

START
  ↓
Supervisor
  ↓
Route Query
  ↓
┌───────────────┬───────────────┐
│               │               │
▼               ▼               ▼
Text            SQL           Vision
Agent           Agent          Agent
│               │               │
└───────────────┼───────────────┘
                │
                ▼
             Response
                │
                ▼
               END

The workflow maintains state containing information such as:

query
next_agent
response
🛠️ Technologies Used
Python
LangGraph
LangChain
Groq LLM
SQL / MySQL
Vision Processing
TypedDict
Virtual Environment
🧪 Testing

The Supervisor workflow can be tested using:

python test_supervisor.py
📊 Example
User Query:
What is the project architecture?

Supervisor Decision:
text_agent

Another example:

User Query:
Show the database records

Supervisor Decision:
sql_agent
✅ Week 2 Outcome

By the end of Week 2, OmniBrain successfully implemented:

✅ Supervisor-based query routing
✅ Multi-agent architecture
✅ Text Agent
✅ SQL Agent
✅ Vision Agent
✅ LangGraph workflow
✅ Agent routing tests
📅 WEEK 3 — Self-RAG & Self-Correction
📌 Overview

In Week 3, OmniBrain was extended with a Self-RAG workflow.

The main goal was to make the system evaluate retrieved information before generating an answer.

Instead of directly answering from retrieved chunks, OmniBrain checks whether the retrieved information is relevant to the user's query.

If the retrieved information is not relevant, the system automatically rewrites the query and performs another search.

🎯 Objectives
Implement Self-RAG
Implement document search
Implement relevance checking
Implement query rewriting
Implement retrieval retry
Generate grounded answers
Reduce hallucination
Handle missing information safely
🏗️ Self-RAG Architecture
                    User Query
                        │
                        ▼
                      SEARCH
                        │
                        ▼
                RELEVANCE CHECK
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
          Relevant          Not Relevant
              │                   │
              ▼                   ▼
            ANSWER          QUERY REWRITE
                                  │
                                  ▼
                            SEARCH AGAIN
                                  │
                                  ▼
                           RELEVANCE CHECK
                                  │
                         ┌────────┴────────┐
                         │                 │
                         ▼                 ▼
                     Relevant         Not Relevant
                         │                 │
                         ▼                 ▼
                       ANSWER       INFORMATION NOT
                                    FOUND IN DOCUMENT

The system supports a maximum of two retrieval attempts.

📂 Week 3 Files
agents_week3/

├── answer_node.py
├── llm.py
├── query_rewriter.py
├── relevance_checker.py
├── search_agent.py
├── test_llm.py
├── TESTING.md
└── __init__.py


graph_week3/

├── self_rag_workflow.py
├── workflow.py
└── __init__.py


test_self_rag.py
🔹 Components
1. Search Agent

search_agent.py

Responsible for retrieving candidate document chunks based on the user's query.

2. Relevance Checker

relevance_checker.py

Evaluates whether the retrieved document chunks contain information relevant to the user's query.

3. Query Rewriter

query_rewriter.py

If the retrieved information is irrelevant, this component reformulates the original query to improve the next retrieval attempt.

4. Answer Node

answer_node.py

Generates the final response using relevant retrieved context.

5. LLM

llm.py

Provides the language model configuration used by the Self-RAG components.

6. Self-RAG Workflow

self_rag_workflow.py

Defines the complete Self-RAG state-based workflow.

🔄 Query Rewrite & Retry

When the first search does not provide relevant information, OmniBrain automatically rewrites the query.

Example:

Original Query:

What does the document say about revenue?

↓

Rewritten Query:

company revenue 2024 financial results

↓

Second Search

↓

Second Relevance Check

If relevant information is still unavailable, the system responds:

I could not find relevant information in the document.
🛡️ Anti-Hallucination Behavior

One of the important features of OmniBrain is its ability to avoid generating unsupported information.

Example:

User:
What does the document say about revenue?

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
Information Not Found

Final response:

I could not find relevant information in the document.

This ensures that OmniBrain does not invent information that is not supported by the document.

🧪 Week 3 Testing

Example relevant query:

What is the Week 2 development plan?

Expected flow:

Search
  ↓
Relevance Check
  ↓
Relevant Chunks
  ↓
Grounded Answer

Example missing-information query:

What does the document say about revenue?

Expected flow:

Search Attempt 1
  ↓
Irrelevant
  ↓
Query Rewrite
  ↓
Search Attempt 2
  ↓
Irrelevant
  ↓
Information Not Found
🛠️ Technologies Used
Python
LangGraph
LangChain
Groq
ChatGroq
Self-RAG
Query Rewriting
Relevance Checking
Retrieval
Grounded Generation
✅ Week 3 Outcome

By the end of Week 3, OmniBrain successfully implemented:

✅ Self-RAG workflow
✅ Search Agent
✅ Relevance Checker
✅ Query Rewriter
✅ Retrieval retry
✅ Grounded answer generation
✅ Missing-information handling
✅ Anti-hallucination behavior
📅 WEEK 4 — Final Integration, Testing & Explainable UI
📌 Overview

In Week 4, the complete OmniBrain system was finalized and integrated with a React frontend and FastAPI backend.

The main goal was to finalize the Self-RAG flow, test the complete system, and provide an explainable interface showing the retrieval process, sources, citations, and extracted images.

🎯 Objectives
Finalize Self-RAG
Test relevance checking
Test query rewriting
Test retrieval retry
Integrate FastAPI backend
Build React frontend
Display Self-RAG process
Display source citations
Display page and chunk information
Display actual extracted images
Handle loading and error states
Test complete end-to-end system
🏗️ Final Architecture
                       USER QUERY
                           │
                           ▼
                    React Frontend
                           │
                           ▼
                    FastAPI Backend
                           │
                           ▼
                  LangGraph Supervisor
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
         Text Agent    SQL Agent   Vision Agent
              │            │            │
              ▼            ▼            ▼
          Self-RAG     Structured    Visual
          Workflow        Data      Processing
              │                         │
              ▼                         ▼
           Search                Extracted Images
              │
              ▼
       Relevance Checker
              │
        ┌─────┴─────┐
        │           │
        ▼           ▼
    Relevant    Not Relevant
        │           │
        ▼           ▼
      Answer    Query Rewrite
                    │
                    ▼
                 Search
                    │
                    ▼
             Relevance Check
                    │
                    ▼
                  Answer
                    │
                    ▼
             React Frontend
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Response   Sources   Images
📂 Week 4 Frontend

The frontend was built using React and Vite.

frontend/

├── .npmrc
├── index.html
├── package.json
├── package-lock.json
├── vite.config.js
│
├── public/
│   └── robots.txt
│
└── src/
    ├── App.jsx
    ├── index.css
    └── main.jsx
🔹 Frontend Features

The frontend provides:

OmniBrain branding
Query input
Agent response display
Text Agent
SQL Agent
Vision Agent
Self-RAG status
Search attempt information
Relevance status
Query rewrite status
Source citations
Page numbers
Chunk IDs
Expandable source excerpts
Actual extracted images
Image links
Loading states
Error states
🔍 Self-RAG Status UI

For a relevant query, the frontend displays:

Self-RAG Process Status
Completed

✓ Initial Search (Attempt 1)
Retrieved candidate document chunks from index

✓ Relevance Assessment
Relevant chunks identified

✓ Grounded Context Confirmed
Generated answer using retrieved context

For an irrelevant query:

Self-RAG Process Status
Completed

✓ Initial Search (Attempt 1)
Retrieved candidate document chunks from index

✓ Relevance Assessment
Retrieved chunks evaluated as irrelevant or insufficient

⚠ Query Rewritten
Autonomous reformulator expanded query for precision

✓ Search (Attempt 2)
Executed secondary retrieval with reformulated query

✓ Second Relevance Check
Evaluated results against reformulated query

✕ Relevant Information Not Found in Document
Correctly prevented hallucination by acknowledging missing data
📚 Source Citations

The frontend displays source information for retrieved document content.

Example:

Retrieved Sources & Citations

sample.pdf
Page 2
Chunk 4

The user can expand the source to view the retrieved context excerpt.

Example:

Source:
sample.pdf • Page 2 • Chunk 4

Extracted Context:
Week-wise Development Plan:
...

This makes the generated response traceable to the retrieved document content.

🖼️ Vision Image Display

The Vision Agent can display the actual extracted image from the PDF.

Example:

Vision Agent

Extracted Image:
page_001_img_001.jpeg

Image Size:
1249 × 1249

Open Full Image ↗

The image is served through the FastAPI backend.

The frontend does not use a hardcoded placeholder image.

🔌 FastAPI Backend

The final system uses FastAPI as the backend API.

Main API
GET /

Response:

{
  "message": "OmniBrain API is running"
}
Health API
GET /health

Response:

{
  "status": "healthy"
}
Query API
POST /query

Example:

{
  "query": "What is the Week 2 development plan?"
}

The response can contain:

query
next_agent
response
self_rag
sources
image_file
image_url

depending on the selected agent and query.

Image API
GET /images/{filename}

This endpoint serves extracted images from:

data/output/images/
🧪 Week 4 Testing
Test Case 1 — Relevant Document Query

Query:

What is the Week 2 development plan?

Expected routing:

User Query
    ↓
Supervisor
    ↓
Text Agent
    ↓
Self-RAG
    ↓
Search
    ↓
Relevance Check
    ↓
Relevant Context
    ↓
Grounded Answer

The UI displays the answer together with source information.

🧪 Test Case 2 — Missing Information

Query:

What does the document say about revenue?

Expected flow:

Search Attempt 1
       ↓
No Relevant Information
       ↓
Query Rewrite
       ↓
Search Attempt 2
       ↓
No Relevant Information
       ↓
Information Not Found

Expected response:

I could not find relevant information in the document.

The system does not generate an unsupported revenue value.

🧪 Test Case 3 — SQL Query

Query:

What was the stock price in 2024?

Expected routing:

User Query
    ↓
Supervisor
    ↓
SQL Agent
    ↓
Structured Data Search

If matching structured data is unavailable:

I could not find matching structured data in the database.

The Self-RAG process display is not used for SQL queries.

🧪 Test Case 4 — Vision Query

Query:

What does this bar chart show?

Expected routing:

User Query
    ↓
Supervisor
    ↓
Vision Agent
    ↓
Extracted Image
    ↓
Visual Processing
    ↓
Vision Response

The frontend displays the actual extracted image.

Example:

page_001_img_001.jpeg
🛠️ Complete Technology Stack
Programming Languages
Python
JavaScript
Backend
Python
FastAPI
Pydantic
Uvicorn
Agentic AI
LangGraph
LangChain
Supervisor-based routing
Specialized AI agents
Self-RAG
Relevance checking
Query rewriting
Retrieval retry
Grounded answer generation
LLM
Groq
ChatGroq
openai/gpt-oss-20b
Document Processing
PyMuPDF
PDF parsing
Text extraction
Text chunking
Image extraction
Frontend
React
Vite
JavaScript
CSS
Database / Structured Data
SQL
SQLite / structured database workflow
Development Tools
Git
GitHub
npm
PowerShell
Virtual Environment
VS Code / Antigravity
📂 Complete Project Structure
OmniBrain/
│
├── README.md
├── main.py
├── requirements.txt
├── create_database.py
├── list_models.py
│
├── test_chunk_search.py
├── test_full_system.py
├── test_integrated.py
├── test_self_rag.py
├── test_sql.py
└── test_supervisor.py
│
├── agents_week2/
│   ├── __init__.py
│   ├── README.md
│   ├── requirements.txt
│   ├── supervisor.py
│   ├── text_agent.py
│   ├── sql_agent.py
│   └── vision_agent.py
│
├── agents_week3/
│   ├── __init__.py
│   ├── README.md
│   ├── TESTING.md
│   ├── requirements.txt
│   ├── answer_node.py
│   ├── llm.py
│   ├── query_rewriter.py
│   ├── relevance_checker.py
│   ├── search_agent.py
│   └── test_llm.py
│
├── graph_week2/
│   ├── __init__.py
│   └── workflow.py
│
├── graph_week3/
│   ├── __init__.py
│   ├── self_rag_workflow.py
│   └── workflow.py
│
├── ingestion/
│   ├── README.md
│   ├── requirements.txt
│   ├── pdf_parser.py
│   ├── text_chunker.py
│   ├── image_extractor.py
│   └── pipeline.py
│
├── data/
│   ├── input/
│   │   └── sample.pdf
│   │
│   └── output/
│       ├── text/
│       ├── chunks/
│       │   └── chunks.json
│       └── images/
│           ├── images.json
│           ├── page_001_img_001.jpeg
│           ├── page_001_img_001.png
│           ├── page_002_img_001.jpeg
│           ├── page_003_img_001.jpeg
│           ├── page_004_img_001.jpeg
│           ├── page_005_img_001.jpeg
│           └── page_006_img_001.jpeg
│
├── demo-samples/
│   ├── sample_electrical_appliance.pdf
│   ├── sample_electrical_cable.pdf
│   ├── sample_plug_socket.pdf
│   └── sample_tender_standardsense.pdf
│
├── frontend/
│   ├── .npmrc
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── vite.config.js
│   │
│   ├── public/
│   │   └── robots.txt
│   │
│   └── src/
│       ├── App.jsx
│       ├── index.css
│       └── main.jsx
│
└── OmniBrain PDF Text Extraction/
    ├── README.md
    ├── requirements.txt
    │
    ├── ingestion/
    │   ├── image_extractor.py
    │   ├── pdf_parser.py
    │   ├── pipeline.py
    │   └── text_chunker.py
    │
    └── data/
        ├── input/
        └── output/
▶️ How to Run OmniBrain
1. Open the Project
cd C:\Users\Lenovo\Documents\OmniBrain
2. Activate Virtual Environment
venv\Scripts\activate
3. Start FastAPI Backend
uvicorn main:app --reload

Backend:

http://127.0.0.1:8000
📖 Swagger API Documentation

Open:

http://127.0.0.1:8000/docs

Swagger can be used to test the FastAPI endpoints.

▶️ Run React Frontend

Open another terminal.

cd C:\Users\Lenovo\Documents\OmniBrain\frontend

Install dependencies if required:

npm install

Start the frontend:

npm run dev

Vite will provide a local URL such as:

http://localhost:5174/

The port may change if the default Vite port is already in use.

🔐 Environment Variables

Create a .env file in the project root.

GROQ_API_KEY=your_api_key

Sensitive information should never be committed to GitHub.

The .gitignore file excludes:

.env
venv/
.venv/
__pycache__/
*.pyc
frontend/node_modules/
frontend/dist/
*.db

Never expose:

API keys
Authentication tokens
Passwords
Private credentials
🧪 Testing Files

OmniBrain contains multiple testing files:

test_chunk_search.py
test_full_system.py
test_integrated.py
test_self_rag.py
test_sql.py
test_supervisor.py

Additional Week 3 testing:

agents_week3/test_llm.py
agents_week3/TESTING.md
🔄 Complete OmniBrain Workflow
                         USER
                          │
                          ▼
                    User Query
                          │
                          ▼
                 React Frontend
                          │
                          ▼
                  FastAPI Backend
                          │
                          ▼
                LangGraph Supervisor
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
       Text Agent      SQL Agent    Vision Agent
            │             │             │
            ▼             ▼             ▼
         Self-RAG      Structured    Image Data
         Workflow         Data
            │
            ▼
          Search
            │
            ▼
     Relevance Checker
            │
       ┌────┴────┐
       │         │
       ▼         ▼
   Relevant   Not Relevant
       │         │
       ▼         ▼
     Answer   Query Rewrite
                 │
                 ▼
              Search
                 │
                 ▼
          Relevance Check
                 │
                 ▼
               Answer
                 │
                 ▼
          React Frontend
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
   Response    Sources    Images
📊 Four-Week Development Summary
Week	Development Area	Main Deliverables
Week 1	Multi-Modal Ingestion	PDF parsing, text extraction, chunking, image extraction
Week 2	Multi-Agent Architecture	Supervisor, Text Agent, SQL Agent, Vision Agent, LangGraph
Week 3	Self-RAG & Self-Correction	Search, relevance checking, query rewriting, retry, grounded answers
Week 4	Final Integration	React UI, FastAPI integration, Self-RAG status, citations, images, testing
🏆 Final Project Outcome

By the end of four weeks, OmniBrain successfully implemented:

✅ Multi-Modal PDF Ingestion
        ↓
✅ Text Extraction
        ↓
✅ Text Chunking
        ↓
✅ Image Extraction
        ↓
✅ LangGraph Supervisor
        ↓
✅ Multi-Agent Architecture
        ↓
✅ Text Agent
        ↓
✅ SQL Agent
        ↓
✅ Vision Agent
        ↓
✅ Self-RAG
        ↓
✅ Relevance Checking
        ↓
✅ Query Rewriting
        ↓
✅ Retrieval Retry
        ↓
✅ Grounded Answer Generation
        ↓
✅ Anti-Hallucination Handling
        ↓
✅ Source Citations
        ↓
✅ Page & Chunk References
        ↓
✅ Actual Vision Image Display
        ↓
✅ FastAPI Backend
        ↓
✅ React + Vite Frontend
        ↓
✅ End-to-End Testing
📌 Week-by-Week Final Status
Week 1
Multi-Modal Document Ingestion
✅ COMPLETED


Week 2
Multi-Agent Architecture & Query Routing
✅ COMPLETED


Week 3
Self-RAG & Self-Correction
✅ COMPLETED


Week 4
Final Integration, Testing & Explainable UI
✅ COMPLETED
🎓 Conclusion

OmniBrain demonstrates the development of an end-to-end Agentic Multi-Modal RAG system.

The project progresses from basic PDF ingestion to an intelligent multi-agent system capable of:

Understanding user queries
Routing queries to specialized agents
Retrieving document information
Checking retrieval relevance
Rewriting unsuccessful queries
Retrying retrieval
Generating grounded responses
Avoiding unsupported information
Providing source citations
Displaying extracted visual content
Providing an interactive web interface
Week 1
Ingestion
   ↓
Week 2
Agentic Routing
   ↓
Week 3
Self-RAG
   ↓
Week 4
Full-Stack Integration
   ↓
Complete OmniBrain System