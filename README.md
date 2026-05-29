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

Project Aegis is an Autonomous Medical Research Assistant designed to accelerate literature reviews, query scientific publications, and compare external research findings against local databases. The system combines MCP tools, LangGraph state-machine orchestration, FastAPI background tasks, Streamlit human-in-the-loop approval, Chroma vector memory, LangSmith evaluation, Prometheus metrics, and robust retry/circuit-breaker resiliency.

## Table Of Contents

- [What It Does](#what-it-does)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Environment Variables](#environment-variables)
- [Run Aegis Research Assistant](#run-aegis-research-assistant)
- [MCP Server](#mcp-server)
- [Evaluation And Observability](#evaluation-and-observability)
- [Security And Git Hygiene](#security-and-git-hygiene)
- [Hugging Face Spaces Production Deployment](#hugging-face-spaces-production-deployment)
- [Limitations](#limitations)

## What It Does

The Aegis Autonomous Research Assistant accelerates medical research by automating literature acquisition and synthesis:

- **Interactive Dashboard**: Accepts research queries and displays real-time progress via a Streamlit interface.
- **Asynchronous Execution**: Launches long-running research tasks in the background using FastAPI background tasks.
- **LangGraph State Machine**: Orchestrates state-driven steps to search, read, compare, pause, and write.
- **Model Context Protocol (MCP)**: Communicates with a localized tool server to perform ArXiv paper search, metadata fetching, and vector search.
- **Publication Merging & Comparison**: Queries local Chroma vector database to find Saif's published papers, comparing new external findings against local expertise.
- **Human-in-the-Loop Approval**: Pauses execution right before writing the final review, enabling the researcher to review, curate, and approve the discovered bibliography.
- **Real-Time Streaming**: Emits a streamed, structured Markdown literature review containing a comprehensive comparison table (Theme, New Papers, Local Papers, Gaps/Opportunities) and concrete next-step research directions.

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
|-- phase4_research/
|   |-- cp01_manual_agent.py          # Checkpoint 1: Hand-built ReAct loop
|   |-- cp02_langchain_agent.py       # Checkpoint 2: LangChain creation
|   |-- cp03_langgraph_agent.py       # Checkpoint 3: LangGraph state-machine
|   |-- cp04_crewai.py                # Checkpoint 4: CrewAI Multi-agent
|   |-- cp04_autogen.py               # Checkpoint 4: AutoGen Multi-agent
|   |-- cp05_memory_agent.py          # Checkpoint 5: Semantic & Episodic memory
|   |-- cp06_mcp_server.py            # Checkpoint 6: FastMCP Server implementation
|   |-- cp07_evaluator.py             # Checkpoint 7: LangSmith evaluations
|   |-- cp07_metrics.py               # Checkpoint 7: Prometheus observability
|   |-- cp07_resiliency.py            # Checkpoint 7: Circuit-breaker & tenacity retries
|   |-- requirements.txt              # Shared requirements
|   |-- requirements.backend.txt      # Production backend requirements
|   |-- entrypoint.py                 # Production Docker multi-process coordinator
|   `-- capstone/
|       |-- mcp_client.py             # LangChain MCP adapter tool adapter (Singleton)
|       |-- agent.py                  # LangGraph research state graph (Mock Key safe)
|       |-- api.py                    # FastAPI service (SSE streams, optional auth bypass)
|       |-- app.py                    # Streamlit research dashboard client
|       `-- free_fallback.py          # Secondary LLM backup failover router
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

## Run Aegis Research Assistant

Navigate to the core research workspace:

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.backend.txt
```

### Running Locally (Multi-process entrypoint)

To boot both the FastAPI backend and Streamlit dashboard together locally in parallel (just like in production):

```powershell
python entrypoint.py
```

Open:
- Streamlit Dashboard: `http://localhost:7860`
- FastAPI Backend API: `http://localhost:8000`
- FastAPI Interactive Docs: `http://localhost:8000/docs`

### Running Manually

Terminal 1, start the FastAPI backend:

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
.\venv\Scripts\Activate.ps1
uvicorn capstone.api:app --reload --port 8000
```

Terminal 2, start the Streamlit UI:

```powershell
cd "E:\AI Development\Project Aegis\phase4_research"
.\venv\Scripts\Activate.ps1
streamlit run .\capstone\app.py --server.port 7860
```

### Core Research API Routes

| Method | Route | Auth | Purpose |
| --- | --- | --- | --- |
| `POST` | `/research` | Optional Bearer | Start a background research task and return a `thread_id`. |
| `GET` | `/status/{thread_id}` | Open | Read graph progress, approval state, and final Markdown review. |
| `POST` | `/approve/{thread_id}` | Optional Bearer | Approve discovered papers and resume the state graph. |
| `GET` | `/research/stream/{thread_id}` | Optional Bearer | SSE stream of the real-time Markdown review draft. |

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

## Hugging Face Spaces Production Deployment

The Project Aegis Capstone is fully deployed in a highly resilient, unified **Docker-based Hugging Face Space**:

- **Live URL**: [Project Aegis on Hugging Face Spaces](https://huggingface.co/spaces/saifrohan44/project-aegis)

### Unified Multi-Process Docker Architecture

To host both the FastAPI backend and Streamlit UI in a single standard Hugging Face container, we implemented a custom **Python Entrypoint Coordinator** (`phase4_research/entrypoint.py`):
1. **Routing Port**: The container exposes port `7860` for public Streamlit web traffic, as required by Hugging Face Spaces.
2. **Private API**: The FastAPI backend runs silently in the background on localhost (`127.0.0.1:8000`), keeping the API completely private and secure.
3. **Orchestration**: The coordinator launches both services in parallel, monitors their exit states, and terminates the container gracefully if either service crashes.

### Production Resiliency Upgrades

1. **LLM Startup-Time Crash Protection**: Solved import-time and instantiation-time validation crashes caused by missing host API keys (e.g. `GROQ_API_KEY` or `OPENAI_API_KEY`). The service coordinator and the LangGraph engine dynamically inject fallback mock keys into the environment at startup if keys are not already configured, allowing the application server to start successfully before secrets are set.
2. **Graceful Bearer Authentication Bypass**: Relaxed `capstone/api.py` to use `HTTPBearer(auto_error=False)` and dynamically bypass key verification if `AEGIS_API_KEY` is unconfigured on the host. This prevents framework-level `401 Unauthorized` errors when no authorization header is sent by the Streamlit frontend, while fully preserving key verification when a token is configured.
3. **Robust Relevance-based Search**: Modified the ArXiv search tool in the MCP server (`cp06_mcp_server.py`) to query by **Relevance** instead of raw *Submission Date*. This completely resolves a Lucene indexing bug where search queries on diverse/older topics returned `0 papers found`. Any research topic searched now returns matching scientific literature perfectly.
4. **Singleton Connection Pooling**: Refactored `load_server_tools()` in the MCP client (`capstone/mcp_client.py`) to pool and reuse the MCP stdio connection as a thread-safe and async-safe global singleton. This completely prevents Python subprocess leakage, drops CPU/memory usage inside the container, and eliminates concurrent ChromaDB database locks.

## Limitations

- The capstone currently fetches paper metadata and includes best-effort PDF extraction hooks. Full production PDF parsing can be expanded later.
- Research outputs are AI-generated drafts and require human review.
- This project is a research prototype, not a clinical decision system.

## Status

The Aegis Autonomous Medical Research Assistant is fully complete, highly optimized, and **deployed in production**:

- **Production deployment**: Unified multi-process Docker container live on Hugging Face Spaces.
- **Singleton pooling**: Clean, leak-free FastMCP stdio server connection.
- **Relevance searches**: Rebuilt tool search using Relevance sorting for robust medical queries.
- **Startup protections**: Integrated mock API-key injection and grace bypass variables.
- **LangGraph orchestration**: Pause and resume states tested with SQLite checkpointers.
- **FastAPI backend**: Rate limiting, SSE streams, optional Bearer authentication.
- **LangSmith pipeline**: Complete QA LLM-as-judge evaluation loop fully validated in CI.
