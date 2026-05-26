"""LangGraph brain for the Capstone Autonomous Research Assistant."""

import os
import re
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import structlog
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pypdf import PdfReader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capstone.free_fallback import invoke_free_backup_fallback
from capstone.mcp_client import arxiv_search, fetch_paper, search_my_papers


load_dotenv()

CAPSTONE_DIR = Path(__file__).resolve().parent
RESEARCH_DB = Path(os.getenv("AEGIS_RESEARCH_DB", CAPSTONE_DIR / "research.db"))
RESEARCH_DB.parent.mkdir(parents=True, exist_ok=True)


class ResearchState(TypedDict):
    topic: str
    found_papers: list[dict]
    extracted_texts: list[str]
    comparison_data: str
    final_review: str
    messages: Annotated[list[BaseMessage], add_messages]

logger = structlog.get_logger()

# Enforce Configuration-Driven Architecture
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "GROQ").upper()

if LLM_PROVIDER == "VLLM":
    # vLLM exposes an entrypoint that mirrors the OpenAI API schema exactly.
    # When deployed to a cloud GPU server, this connects to your internal network service container.
    VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8001/v1")
    llm = ChatOpenAI(
        model="meta-llama/Llama-3.1-8B-Instruct",
        api_key="mock_token_for_cloud_vllm",
        base_url=VLLM_URL,
        streaming=True,
    )
    logger.info("llm_client_initialized", provider="vllm", target_url=VLLM_URL)
else:
    # Default sustainable fallback for your local 16GB CPU machine testing loop
    groq_model = os.getenv("AEGIS_GROQ_MODEL", "llama-3.3-70b-versatile")
    llm = ChatGroq(
        model=groq_model,
        streaming=True,
    )
    logger.info("llm_client_initialized", provider="groq", model=groq_model)


def _parse_arxiv_results(raw_results: str) -> list[dict]:
    papers = []
    chunks = [chunk.strip() for chunk in raw_results.split("---") if chunk.strip()]

    if len(chunks) <= 1:
        chunks = re.split(r"\n(?=\d+\.\s|\s*Title:)", raw_results)

    for chunk in chunks:
        title = _match_field(chunk, "Title") or "Untitled"
        authors = _match_field(chunk, "Authors") or "Unknown authors"
        arxiv_id = _match_field(chunk, "ArXiv ID") or _match_arxiv_id(chunk)
        abstract = _match_field(chunk, "Abstract") or ""

        if not arxiv_id:
            continue

        papers.append(
            {
                "title": title.strip(),
                "authors": authors.strip(),
                "arxiv_id": arxiv_id.strip(),
                "abstract": abstract.strip(),
            }
        )

    return papers


def _match_field(text: str, field_name: str) -> str | None:
    pattern = rf"{re.escape(field_name)}:\s*(.*?)(?=\n[A-Z][A-Za-z ]+:\s|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return None
    return " ".join(match.group(1).split())


def _match_arxiv_id(text: str) -> str | None:
    match = re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", text)
    return match.group(0) if match else None


def _extract_pdf_text_from_url(pdf_url: str, max_pages: int = 3) -> str:
    """Best-effort PDF extraction for tools that return a PDF URL."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        urllib.request.urlretrieve(pdf_url, temp_path)
        reader = PdfReader(str(temp_path))
        pages = []
        for page in reader.pages[:max_pages]:
            pages.append(page.extract_text() or "")

        temp_path.unlink(missing_ok=True)
        extracted = "\n".join(page.strip() for page in pages if page.strip())
        return extracted or "PDF downloaded, but no extractable text was found."
    except Exception as exc:
        return f"PDF extraction failed gracefully: {exc}"


def _maybe_extract_pdf_text(fetch_result: str) -> str:
    match = re.search(r"https?://\S+\.pdf", fetch_result)
    if not match:
        return fetch_result

    pdf_text = _extract_pdf_text_from_url(match.group(0))
    return f"{fetch_result}\n\nExtracted PDF text excerpt:\n{pdf_text[:4000]}"


def search_node(state: ResearchState) -> dict:
    topic = state["topic"]
    try:
        raw_results = arxiv_search.invoke({"query": topic, "max_results": 10})
        found_papers = _parse_arxiv_results(raw_results)
        message = f"Found {len(found_papers)} ArXiv papers for topic: {topic}"
    except Exception as exc:
        raw_results = f"ArXiv search failed gracefully: {exc}"
        found_papers = []
        message = raw_results

    return {
        "found_papers": found_papers,
        "messages": [AIMessage(content=f"{message}\n\n{raw_results[:2000]}")],
    }


def read_node(state: ResearchState) -> dict:
    extracted_texts = []

    for paper in state.get("found_papers", []):
        arxiv_id = paper.get("arxiv_id")
        title = paper.get("title", "Untitled")

        if not arxiv_id:
            continue

        try:
            fetched = fetch_paper.invoke({"arxiv_id": arxiv_id})
            text = _maybe_extract_pdf_text(fetched)
            extracted_texts.append(f"Paper: {title}\nArXiv ID: {arxiv_id}\n{text}")
        except Exception as exc:
            extracted_texts.append(
                f"Paper: {title}\nArXiv ID: {arxiv_id}\nFetch failed gracefully: {exc}"
            )

    if not extracted_texts:
        extracted_texts.append("No paper text could be extracted. Continue with search metadata only.")

    return {
        "extracted_texts": extracted_texts,
        "messages": [AIMessage(content=f"Read/extracted {len(extracted_texts)} paper records.")],
    }


def compare_node(state: ResearchState) -> dict:
    topic = state["topic"]
    try:
        local_context = search_my_papers.invoke({"query": topic})
    except Exception as exc:
        local_context = f"Local paper search failed gracefully: {exc}"

    new_findings = "\n\n".join(state.get("extracted_texts", []))[:6000]
    comparison_data = (
        f"Topic: {topic}\n\n"
        "New external findings from ArXiv/fetched papers:\n"
        f"{new_findings}\n\n"
        "Relevant context from Saif's published paper database:\n"
        f"{local_context}"
    )

    return {
        "comparison_data": comparison_data,
        "messages": [AIMessage(content="Prepared comparison data against local publications.")],
    }


def write_node(state: ResearchState) -> dict:
    system_prompt = SystemMessage(
        content=(
            "You are Aegis, a careful autonomous medical research assistant. "
            "Write rigorous but readable literature reviews. Be precise about uncertainty."
        )
    )
    user_prompt = HumanMessage(
        content=(
            f"Research topic: {state['topic']}\n\n"
            f"Comparison data:\n{state.get('comparison_data', '')[:12000]}\n\n"
            "Write the final literature review in Markdown. Requirements:\n"
            "1. Start with a concise executive summary.\n"
            "2. Discuss the new ArXiv findings.\n"
            "3. Compare them against Saif's published work.\n"
            "4. Include a Markdown comparison table with columns: Theme, New Papers, Saif's Papers, Gap/Opportunity.\n"
            "5. End with 3 concrete next-step research directions."
        )
    )

    try:
        response = llm.invoke([system_prompt, user_prompt])
        final_review = response.content
    except Exception as exc:
        logger.warning(
            "primary_llm_failed_triggering_free_failover",
            provider=LLM_PROVIDER,
            error=str(exc),
        )
        final_review = invoke_free_backup_fallback(
            f"{system_prompt.content}\n\n{user_prompt.content}"
        )

    return {
        "final_review": final_review,
        "messages": [AIMessage(content=final_review)],
    }


graph_builder = StateGraph(ResearchState)
graph_builder.add_node("search_node", search_node)
graph_builder.add_node("read_node", read_node)
graph_builder.add_node("compare_node", compare_node)
graph_builder.add_node("write_node", write_node)

graph_builder.add_edge(START, "search_node")
graph_builder.add_edge("search_node", "read_node")
graph_builder.add_edge("read_node", "compare_node")
graph_builder.add_edge("compare_node", "write_node")
graph_builder.add_edge("write_node", END)

_checkpointer_context = SqliteSaver.from_conn_string(str(RESEARCH_DB))
checkpointer = _checkpointer_context.__enter__()

graph = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["write_node"],
)


def run_research(topic: str, thread_id: str = "capstone-demo") -> ResearchState:
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "topic": topic,
        "found_papers": [],
        "extracted_texts": [],
        "comparison_data": "",
        "final_review": "",
        "messages": [HumanMessage(content=f"Research topic: {topic}")],
    }
    return graph.invoke(initial_state, config=config)


def resume_research(thread_id: str = "capstone-demo") -> ResearchState:
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke(None, config=config)


if __name__ == "__main__":
    demo_topic = "Transformers in medical imaging"
    demo_thread = "capstone-demo"

    print(f"Starting research: {demo_topic}")
    paused_state = run_research(demo_topic, demo_thread)
    print("\n--- Paused Before write_node ---")
    print(f"Papers found: {len(paused_state.get('found_papers', []))}")
    print(f"Extracted records: {len(paused_state.get('extracted_texts', []))}")
    print("Review has not been written yet because the graph is waiting for approval.")

    input("\nPress Enter to approve write_node and resume...")

    final_state = resume_research(demo_thread)
    print("\n--- Final Literature Review ---")
    print(final_state.get("final_review", "No final review generated."))
