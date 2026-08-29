README — OmniBrain Week 2
🧠 OmniBrain – Week 2: Multi-Agent Architecture & Query Routing
📌 Overview

In Week 2, OmniBrain was extended from the basic ingestion pipeline into a multi-agent AI system using LangGraph.

The main goal was to create a Supervisor Agent that analyzes the user's query and routes it to the most appropriate specialized agent.

🎯 Objectives
Build a Supervisor Agent
Implement intelligent query routing
Create specialized agents for different types of queries
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
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Text Agent       SQL Agent       Vision Agent
        │               │               │
        └───────────────┼───────────────┘
                        ▼
                    Response
📂 Week 2 Files
agents_week2/
│
├── supervisor.py
├── text_agent.py
├── sql_agent.py
├── vision_agent.py
└── __init__.py

graph_week2/
│
├── workflow.py
└── __init__.py

test_supervisor.py
🔹 Components
1. Supervisor Agent

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

Handles queries related to:

Documents
Text information
Project descriptions
General knowledge from the available context
3. SQL Agent

Handles structured-data queries that require interaction with a database.

Example:

"Show all users"
"What is the total number of records?"
4. Vision Agent

Handles image-related queries.

Example:

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
┌───────────────┐
│               │
▼               ▼
Text           SQL
Agent          Agent
│               │
└───────┬───────┘
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
Vision processing
TypedDict
Virtual Environment (venv)
🧪 Testing

The Supervisor workflow can be tested using:

python test_supervisor.py

Or, if running the module:

python -m agents_week2.test_supervisor
Expected Output

The system should display the selected agent based on the query.

Example:

User Query: What is the project architecture?

Supervisor Decision: text_agent

Another example:

User Query: Show the database records

Supervisor Decision: sql_agent
✅ Week 2 Outcome

By the end of Week 2, OmniBrain successfully implemented:

✅ Supervisor-based query routing
✅ Multi-agent architecture
✅ Text Agent
✅ SQL Agent
✅ Vision Agent
✅ LangGraph workflow
✅ Agent routing tests