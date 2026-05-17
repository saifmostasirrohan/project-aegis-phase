# Project Aegis

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Agents-1C3C3C)
![LangGraph](https://img.shields.io/badge/LangGraph-State%20Machines-1C3C3C)
![MCP](https://img.shields.io/badge/MCP-Tool%20Server-000000)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Memory-5B3FD3)
![LangSmith](https://img.shields.io/badge/LangSmith-Evaluation-1C3C3C)
![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-E6522C?logo=prometheus&logoColor=white)

Project Aegis is a staged AI assistant system that evolves from a structured agronomy assistant into an autonomous research assistant. The current capstone combines MCP tools, LangGraph orchestration, FastAPI background jobs, Streamlit human approval, Chroma memory, LangSmith evaluation, Prometheus metrics, and retry/circuit-breaker resiliency.

## Table Of Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Environment Variables](#environment-variables)
- [Run Phase 4 Capstone](#run-phase-4-capstone)
- [Run Phase 1 Assistant](#run-phase-1-assistant)
- [MCP Server](#mcp-server)
- [Evaluation And Observability](#evaluation-and-observability)
- [Security And Git Hygiene](#security-and-git-hygiene)
- [Limitations](#limitations)

## What It Does

### Phase 1: Agronomy Assistant

- Provides crop pathology support through a FastAPI backend and Streamlit frontend.
- Uses structured request and response contracts with Pydantic.
- Stores conversation state in SQLite.

### Phase 4: Aegis Research Assistant

- Accepts a research topic from a Streamlit dashboard.
- Starts long-running research jobs through FastAPI background tasks.
- Uses a LangGraph state machine to search, read, compare, pause, and write.
- Calls a local MCP server for ArXiv search, paper fetching, and local paper search.
- Compares external findings against the local Chroma publication database.
- Pauses before drafting so the user can approve found papers.
- Produces a final Markdown literature review with a comparison table.

## Architecture

```text
Streamlit UI
    |
    | POST /research, GET /status, POST /approve
    v
FastAPI Capstone Backend
    |
    | background task + thread_id
    v
LangGraph Research State Machine
    |
    | LangChain MCP adapter tools
    v
CP-06 MCP Research Server
    |
    +-- arxiv_search
    +-- fetch_paper
    +-- search_my_papers
    +-- papers://list
```

The graph persists state with SQLite checkpointing and pauses before `write_node` for human approval.

## Tech Stack

| Layer | Tools |
| --- | --- |
| Frontend | Streamlit |
| Backend | FastAPI, Uvicorn |
| Agent orchestration | LangGraph, LangChain |
| Tool protocol | MCP, langchain-mcp-adapters |
| LLM provider | Groq, Llama 3.3 |
| Memory | SQLite, ChromaDB |
| Evaluation | LangSmith, LLM-as-judge |
| Metrics | Prometheus client |
| Resiliency | Tenacity retries, circuit breaker demo |
| Multi-agent experiments | CrewAI, AutoGen |

## Repository Structure

```text
Project Aegis/
|-- backend/                         # Phase 1 FastAPI backend
|-- frontend/                        # Phase 1 Streamlit frontend
|-- phase4_research/
|   |-- cp01_manual_agent.py
|   |-- cp02_langchain_agent.py
|   |-- cp03_langgraph_agent.py
|   |-- cp04_crewai.py
|   |-- cp04_autogen.py
|   |-- cp05_memory_agent.py
|   |-- cp06_mcp_server.py
|   |-- cp07_evaluator.py
|   |-- cp07_metrics.py
|   |-- cp07_resiliency.py
|   |-- requirements.txt
|   `-- capstone/
|       |-- mcp_client.py            # LangChain MCP adapter tools
|       |-- agent.py                 # LangGraph research workflow
|       |-- api.py                   # FastAPI backend
|       `-- app.py                   # Streamlit dashboard
|-- .env.example
|-- .gitignore
|-- requirements.txt
`-- start-all.ps1
```

## Environment Variables

Copy `.env.example` into your local `.env` file and fill in real keys.

```env
GROQ_API_KEY=your_groq_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=Phase4_Research_Agent
AEGIS_API_BASE_URL=http://127.0.0.1:8001
```

Never commit `.env` files.

## Run Phase 4 Capstone

Use the primary Phase 4 environment.

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Terminal 1, start the FastAPI backend:

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
.\venv\Scripts\Activate.ps1
uvicorn capstone.api:app --reload --port 8001
```

Terminal 2, start the Streamlit UI:

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
.\venv\Scripts\Activate.ps1
streamlit run .\capstone\app.py
```

Open:

- Capstone API docs: `http://127.0.0.1:8001/docs`
- Capstone UI: `http://localhost:8501`

Expected capstone routes:

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/research` | Start a research job and return a `thread_id`. |
| `GET` | `/status/{thread_id}` | Read graph progress, approval state, and final review. |
| `POST` | `/approve/{thread_id}` | Resume the graph after human approval. |

## Run Phase 1 Assistant

```powershell
cd "E:\AI Development\Project Aegis"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
.\start-all.ps1
```

Default services:

- Phase 1 backend: `http://127.0.0.1:8001`
- Phase 1 frontend: `http://localhost:8501`

## MCP Server

Run the CP-06 MCP server directly:

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
.\venv\Scripts\Activate.ps1
python .\cp06_mcp_server.py
```

Test it with MCP Inspector:

```powershell
npx @modelcontextprotocol/inspector python cp06_mcp_server.py
```

Available MCP capabilities:

| Type | Name |
| --- | --- |
| Tool | `arxiv_search` |
| Tool | `fetch_paper` |
| Tool | `search_my_papers` |
| Resource | `papers://list` |

## Evaluation And Observability

Run the LangSmith evaluator:

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
.\venv\Scripts\Activate.ps1
python .\cp07_evaluator.py
```

Run the Prometheus metrics demo:

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
.\venv\Scripts\Activate.ps1
uvicorn cp07_metrics:app --reload
```

Open:

- `http://127.0.0.1:8000/research?query=MicroHybridNet`
- `http://127.0.0.1:8000/metrics`

Run the resiliency demo:

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
.\venv\Scripts\Activate.ps1
python .\cp07_resiliency.py
```

## Security And Git Hygiene

Ignored by Git:

- `.env` and `*.env`
- virtual environments
- local SQLite databases
- Chroma vector stores
- user profile memory
- generated cache folders
- model weights and PDFs

Tracked safely:

- `.env.example` with placeholder values only
- source code
- dependency manifests
- documentation

## Limitations

- The capstone currently fetches paper metadata and includes best-effort PDF extraction hooks. Full production PDF parsing can be expanded later.
- Research outputs are AI-generated drafts and require human review.
- This project is a research prototype, not a clinical decision system.

## Status

Phase 4 capstone is complete as a working local prototype:

- MCP server tested with Inspector
- LangGraph flow tested with SQLite checkpointing
- FastAPI backend tested through Swagger docs
- Streamlit UI tested end-to-end
- LangSmith evaluation pipeline tested
- Prometheus metrics and resiliency demos implemented
