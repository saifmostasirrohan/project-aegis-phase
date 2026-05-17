"""FastAPI backend for the Capstone Autonomous Research Assistant."""

import threading
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from capstone.agent import graph, run_research


app = FastAPI(title="Aegis Capstone API")

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, description="Research topic to investigate.")


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _set_job(thread_id: str, **updates) -> None:
    with jobs_lock:
        current = jobs.setdefault(thread_id, {})
        current.update(updates)


def _get_job(thread_id: str) -> dict | None:
    with jobs_lock:
        job = jobs.get(thread_id)
        return dict(job) if job else None


def _run_graph_job(thread_id: str, topic: str) -> None:
    _set_job(thread_id, status="running", current_node="search_node")

    try:
        state = run_research(topic, thread_id=thread_id)
        waiting_for_approval = not state.get("final_review")
        _set_job(
            thread_id,
            status="paused" if waiting_for_approval else "completed",
            current_node="write_node" if waiting_for_approval else "END",
            waiting_for_approval=waiting_for_approval,
            error=None,
        )
    except Exception as exc:
        _set_job(
            thread_id,
            status="failed",
            current_node="error",
            waiting_for_approval=False,
            error=str(exc),
        )


def _resume_graph_job(thread_id: str) -> None:
    _set_job(thread_id, status="running", current_node="write_node", waiting_for_approval=False)

    try:
        graph.invoke(Command(resume=True), config=_config(thread_id))
        _set_job(
            thread_id,
            status="completed",
            current_node="END",
            waiting_for_approval=False,
            error=None,
        )
    except Exception as exc:
        _set_job(
            thread_id,
            status="failed",
            current_node="error",
            waiting_for_approval=False,
            error=str(exc),
        )


def _snapshot_summary(thread_id: str) -> dict:
    try:
        snapshot = graph.get_state(_config(thread_id))
    except Exception:
        return {}

    values = snapshot.values or {}
    next_nodes = list(snapshot.next or [])
    waiting_for_approval = "write_node" in next_nodes and not values.get("final_review")

    if next_nodes:
        current_node = next_nodes[0]
    elif values.get("final_review"):
        current_node = "END"
    else:
        current_node = "unknown"

    return {
        "current_node": current_node,
        "next_nodes": next_nodes,
        "waiting_for_approval": waiting_for_approval,
        "topic": values.get("topic", ""),
        "found_papers": values.get("found_papers", []),
        "papers_found": len(values.get("found_papers", [])),
        "extracted_texts": len(values.get("extracted_texts", [])),
        "has_comparison_data": bool(values.get("comparison_data")),
        "has_final_review": bool(values.get("final_review")),
        "final_review": values.get("final_review", ""),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research")
async def start_research(request: ResearchRequest, background_tasks: BackgroundTasks) -> dict:
    thread_id = str(uuid4())
    _set_job(
        thread_id,
        status="queued",
        topic=request.topic,
        current_node="START",
        waiting_for_approval=False,
        error=None,
    )
    background_tasks.add_task(_run_graph_job, thread_id, request.topic)
    return {"thread_id": thread_id, "status": "queued"}


@app.get("/status/{thread_id}")
async def research_status(thread_id: str) -> dict:
    job = _get_job(thread_id)
    snapshot = _snapshot_summary(thread_id)

    if not job and not snapshot:
        raise HTTPException(status_code=404, detail="Unknown thread_id")

    status = job.get("status", "unknown") if job else "unknown"
    if snapshot.get("waiting_for_approval"):
        status = "paused"
    elif snapshot.get("has_final_review"):
        status = "completed"

    return {
        "thread_id": thread_id,
        "status": status,
        "current_node": snapshot.get("current_node") or job.get("current_node", "unknown"),
        "waiting_for_approval": snapshot.get(
            "waiting_for_approval", job.get("waiting_for_approval", False)
        ),
        "topic": snapshot.get("topic") or job.get("topic", ""),
        "found_papers": snapshot.get("found_papers", []),
        "papers_found": snapshot.get("papers_found", 0),
        "extracted_texts": snapshot.get("extracted_texts", 0),
        "has_comparison_data": snapshot.get("has_comparison_data", False),
        "has_final_review": snapshot.get("has_final_review", False),
        "final_review": snapshot.get("final_review", ""),
        "error": job.get("error") if job else None,
    }


@app.post("/approve/{thread_id}")
async def approve_research(thread_id: str, background_tasks: BackgroundTasks) -> dict:
    job = _get_job(thread_id)
    snapshot = _snapshot_summary(thread_id)

    if not job and not snapshot:
        raise HTTPException(status_code=404, detail="Unknown thread_id")

    if snapshot.get("has_final_review"):
        return {"thread_id": thread_id, "status": "already_completed"}

    if not snapshot.get("waiting_for_approval"):
        raise HTTPException(status_code=409, detail="Research is not paused for approval.")

    background_tasks.add_task(_resume_graph_job, thread_id)
    return {"thread_id": thread_id, "status": "resume_queued"}
