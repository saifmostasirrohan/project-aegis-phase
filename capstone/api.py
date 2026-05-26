"""FastAPI backend for the Capstone Autonomous Research Assistant."""

import os
import secrets
import threading
import time
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langgraph.types import Command
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse
import structlog

from capstone.agent import graph, run_research


structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()

app = FastAPI(title="Aegis Capstone API")
security_scheme = HTTPBearer()
EXPECTED_KEY = os.getenv("AEGIS_API_KEY")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

HTTP_REQUESTS_TOTAL = Counter(
    "aegis_http_requests_total",
    "Total HTTP Requests",
    ["endpoint", "status_code"],
)
HTTP_REQUEST_DURATION = Histogram(
    "aegis_http_request_duration_seconds",
    "HTTP Request Latency",
    ["endpoint"],
)
LLM_TOKENS_TOTAL = Counter(
    "aegis_llm_tokens_total",
    "Total tokens burned by LLM",
    ["model", "token_type"],
)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
token_metrics_recorded: set[str] = set()
token_metrics_lock = threading.Lock()


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, description="Research topic to investigate.")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", f"req_{int(time.time())}")
    endpoint = request.url.path

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        path=endpoint,
        method=request.method,
    )

    logger.info("request_started")
    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - start_time
        duration = (time.perf_counter() - start_time) * 1000
        HTTP_REQUESTS_TOTAL.labels(endpoint=endpoint, status_code="500").inc()
        HTTP_REQUEST_DURATION.labels(endpoint=endpoint).observe(elapsed)
        logger.exception("request_failed", duration_ms=round(duration, 2))
        raise

    elapsed = time.perf_counter() - start_time
    duration = elapsed * 1000
    response.headers["X-Request-ID"] = request_id
    HTTP_REQUESTS_TOTAL.labels(endpoint=endpoint, status_code=str(response.status_code)).inc()
    HTTP_REQUEST_DURATION.labels(endpoint=endpoint).observe(elapsed)
    logger.info("request_completed", status_code=response.status_code, duration_ms=round(duration, 2))
    return response


def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security_scheme)) -> str:
    if not EXPECTED_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API authentication layer misconfigured on host server.",
        )

    if not secrets.compare_digest(credentials.credentials, EXPECTED_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access Denied: Invalid or missing Aegis Authorization Token.",
        )

    return credentials.credentials


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


def _first_int(*values) -> int:
    for value in values:
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    return 0


def _record_llm_tokens(thread_id: str, state: dict) -> None:
    with token_metrics_lock:
        if thread_id in token_metrics_recorded:
            return

    model_fallback = os.getenv("AEGIS_GROQ_MODEL", "llama-3.3-70b-versatile")
    totals = {"input": 0, "output": 0, "total": 0}
    model_name = model_fallback

    for message in state.get("messages", []):
        usage_metadata = getattr(message, "usage_metadata", None) or {}
        response_metadata = getattr(message, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
        model_name = response_metadata.get("model_name") or response_metadata.get("model") or model_name

        totals["input"] += _first_int(
            usage_metadata.get("input_tokens"),
            token_usage.get("prompt_tokens"),
            token_usage.get("input_tokens"),
        )
        totals["output"] += _first_int(
            usage_metadata.get("output_tokens"),
            token_usage.get("completion_tokens"),
            token_usage.get("output_tokens"),
        )
        totals["total"] += _first_int(
            usage_metadata.get("total_tokens"),
            token_usage.get("total_tokens"),
        )

    if not totals["total"] and (totals["input"] or totals["output"]):
        totals["total"] = totals["input"] + totals["output"]

    if not any(totals.values()):
        logger.info("llm_token_metrics_unavailable", thread_id=thread_id)
        return

    for token_type, count in totals.items():
        if count:
            LLM_TOKENS_TOTAL.labels(model=model_name, token_type=token_type).inc(count)

    with token_metrics_lock:
        token_metrics_recorded.add(thread_id)
    logger.info("llm_tokens_recorded", thread_id=thread_id, model=model_name, **totals)


def _run_graph_job(thread_id: str, topic: str) -> None:
    _set_job(thread_id, status="running", current_node="search_node")
    log = logger.bind(thread_id=thread_id, topic=topic)
    log.info("research_job_started")

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
        log.info(
            "research_job_paused" if waiting_for_approval else "research_job_completed",
            waiting_for_approval=waiting_for_approval,
        )
        if not waiting_for_approval:
            _record_llm_tokens(thread_id, state)
    except Exception as exc:
        _set_job(
            thread_id,
            status="failed",
            current_node="error",
            waiting_for_approval=False,
            error=str(exc),
        )
        log.exception("research_job_failed", error=str(exc))


def _resume_graph_job(thread_id: str) -> None:
    _set_job(thread_id, status="running", current_node="write_node", waiting_for_approval=False)
    log = logger.bind(thread_id=thread_id)
    log.info("research_resume_started")

    try:
        state = graph.invoke(Command(resume=True), config=_config(thread_id))
        _set_job(
            thread_id,
            status="completed",
            current_node="END",
            waiting_for_approval=False,
            error=None,
        )
        _record_llm_tokens(thread_id, state)
        log.info("research_resume_completed")
    except Exception as exc:
        _set_job(
            thread_id,
            status="failed",
            current_node="error",
            waiting_for_approval=False,
            error=str(exc),
        )
        log.exception("research_resume_failed", error=str(exc))


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


def _stream_chunk_text(event) -> str:
    message = event[0] if isinstance(event, tuple) else event
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text_parts.append(str(item.get("text", "")))
        return "".join(text_parts)

    return ""


def _stream_final_review(thread_id: str):
    _set_job(thread_id, status="running", current_node="write_node", waiting_for_approval=False)
    emitted_tokens = False
    log = logger.bind(thread_id=thread_id)
    log.info("research_stream_started")

    try:
        for event in graph.stream(
            Command(resume=True),
            config=_config(thread_id),
            stream_mode="messages",
        ):
            chunk = _stream_chunk_text(event)
            if not chunk:
                continue

            emitted_tokens = True
            yield {"event": "token", "data": chunk}

        snapshot = _snapshot_summary(thread_id)
        final_review = snapshot.get("final_review", "")

        if final_review and not emitted_tokens:
            yield {"event": "token", "data": final_review}

        _set_job(
            thread_id,
            status="completed",
            current_node="END",
            waiting_for_approval=False,
            error=None,
        )
        final_state = graph.get_state(_config(thread_id)).values or {}
        _record_llm_tokens(thread_id, final_state)
        log.info("research_stream_completed", emitted_tokens=emitted_tokens)
        yield {"event": "done", "data": "[DONE]"}
    except Exception as exc:
        _set_job(
            thread_id,
            status="failed",
            current_node="error",
            waiting_for_approval=False,
            error=str(exc),
        )
        log.exception("research_stream_failed", error=str(exc))
        yield {"event": "error", "data": str(exc)}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def start_research(
    request: Request,
    payload: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    thread_id = str(uuid4())
    _set_job(
        thread_id,
        status="queued",
        topic=payload.topic,
        current_node="START",
        waiting_for_approval=False,
        error=None,
    )
    background_tasks.add_task(_run_graph_job, thread_id, payload.topic)
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


@app.get("/research/stream/{thread_id}", dependencies=[Depends(verify_api_key)])
async def stream_research(thread_id: str) -> EventSourceResponse:
    job = _get_job(thread_id)
    snapshot = _snapshot_summary(thread_id)

    if not job and not snapshot:
        raise HTTPException(status_code=404, detail="Unknown thread_id")

    if snapshot.get("has_final_review"):
        return EventSourceResponse(
            iter(
                [
                    {"event": "token", "data": snapshot.get("final_review", "")},
                    {"event": "done", "data": "[DONE]"},
                ]
            )
        )

    if not snapshot.get("waiting_for_approval"):
        raise HTTPException(status_code=409, detail="Research is not paused for streaming.")

    return EventSourceResponse(_stream_final_review(thread_id))


@app.post("/approve/{thread_id}", dependencies=[Depends(verify_api_key)])
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
