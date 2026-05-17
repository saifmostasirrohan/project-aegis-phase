# Project Aegis

Project Aegis is a staged AI assistant project that grows from a structured agronomy assistant into an autonomous research assistant. The repository now includes FastAPI services, Streamlit interfaces, LangGraph workflows, MCP tools, persistent memory, evaluation, observability, and human-in-the-loop research drafting.

## Current Highlights

- **Phase 1 Agronomy Assistant:** FastAPI + Streamlit crop pathology assistant with SQLite-backed conversation memory.
- **Phase 4 Research Capstone:** Autonomous literature review assistant using MCP tools, LangGraph orchestration, FastAPI background jobs, and Streamlit approval UI.
- **Persistent Memory:** User profile memory plus Chroma-backed episodic memory experiments.
- **Evaluation & Observability:** LangSmith LLM-as-judge evaluation, Prometheus metrics, retries, and circuit breaker demos.
- **Secret Safety:** `.env`, local databases, virtual environments, Chroma stores, and model weights are ignored.

## Repository Map

```text
Project Aegis/
├── backend/                    # Phase 1 FastAPI backend
├── frontend/                   # Phase 1 Streamlit frontend
├── phase4_research/
│   ├── cp01_manual_agent.py
│   ├── cp02_langchain_agent.py
│   ├── cp03_langgraph_agent.py
│   ├── cp04_crewai.py
│   ├── cp04_autogen.py
│   ├── cp05_memory_agent.py
│   ├── cp06_mcp_server.py
│   ├── cp07_evaluator.py
│   ├── cp07_metrics.py
│   ├── cp07_resiliency.py
│   └── capstone/
│       ├── mcp_client.py       # LangChain MCP adapter tools
│       ├── agent.py            # LangGraph research state machine
│       ├── api.py              # FastAPI backend for long-running research
│       └── app.py              # Streamlit dashboard
├── .env.example
├── requirements.txt
└── start-all.ps1
```

## Environment Variables

Create local `.env` files from `.env.example`. Never commit real keys.

```env
GROQ_API_KEY=your_groq_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=Phase4_Research_Agent
AEGIS_API_BASE_URL=http://127.0.0.1:8001
```

## Phase 1: Agronomy Assistant

### Setup

```powershell
cd "E:\AI Development\Project Aegis"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run

```powershell
.\start-all.ps1
```

Default services:

- Backend API: `http://127.0.0.1:8001`
- Frontend UI: `http://localhost:8501`

## Phase 4: Research Capstone

The capstone is an autonomous literature review assistant. It searches ArXiv through a local MCP server, fetches paper metadata, compares findings against the local Chroma publication database, pauses for human approval, and writes a Markdown literature review.

### Install

Use the primary Phase 4 environment:

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run The Capstone

Terminal 1, FastAPI backend:

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
.\venv\Scripts\Activate.ps1
uvicorn capstone.api:app --reload --port 8001
```

Terminal 2, Streamlit frontend:

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
.\venv\Scripts\Activate.ps1
streamlit run .\capstone\app.py
```

Open:

- API docs: `http://127.0.0.1:8001/docs`
- UI: `http://localhost:8501`

Expected API routes:

- `POST /research` - Start a background research job and return a `thread_id`.
- `GET /status/{thread_id}` - Check current node, paper count, approval state, and final review.
- `POST /approve/{thread_id}` - Resume the graph after the human review pause.

### MCP Server

Run the CP-06 MCP server directly:

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
.\venv\Scripts\Activate.ps1
python .\cp06_mcp_server.py
```

Test with MCP Inspector:

```powershell
npx @modelcontextprotocol/inspector python cp06_mcp_server.py
```

The server exposes:

- `arxiv_search`
- `fetch_paper`
- `search_my_papers`
- `papers://list`

## Evaluation And Observability

LangSmith evaluation:

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
.\venv\Scripts\Activate.ps1
python .\cp07_evaluator.py
```

Prometheus metrics demo:

```powershell
uvicorn cp07_metrics:app --reload
```

Open:

- `http://127.0.0.1:8000/research?query=MicroHybridNet`
- `http://127.0.0.1:8000/metrics`

Resiliency demo:

```powershell
python .\cp07_resiliency.py
```

## Development Notes

- Use `phase4_research\venv` for LangChain, LangGraph, MCP, memory, metrics, and capstone work.
- Use `phase4_research\multiagent_venv` for CrewAI and AutoGen experiments.
- Runtime state such as `.env`, `chroma_db/`, `user_profile.json`, `research.db`, and virtual environments should stay local.
- The capstone currently uses placeholder PDF extraction unless a PDF URL is available; `pypdf` support is wired for future full-text extraction.

## Safety

This repository contains research prototypes. Outputs should be reviewed before being used for medical, clinical, agronomic, financial, or operational decisions.
