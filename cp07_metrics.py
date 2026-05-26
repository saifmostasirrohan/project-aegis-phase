import random
import time

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.responses import Response


app = FastAPI(title="Aegis Agent Metrics")

token_usage_total = Counter(
    "token_usage_total",
    "Total number of tokens used by the mock research agent.",
)

agent_latency_seconds = Histogram(
    "agent_latency_seconds",
    "Latency of mock research agent requests in seconds.",
)


@app.get("/research")
def research(query: str = "Transformers in medical imaging") -> dict:
    """Simulate an agent research request and record custom metrics."""
    start_time = time.perf_counter()

    simulated_latency = random.uniform(0.2, 1.2)
    time.sleep(simulated_latency)

    tokens_used = random.randint(250, 1500)
    token_usage_total.inc(tokens_used)

    elapsed = time.perf_counter() - start_time
    agent_latency_seconds.observe(elapsed)

    return {
        "query": query,
        "answer": "Mock research complete. Replace this with the real Aegis agent later.",
        "tokens_used": tokens_used,
        "latency_seconds": round(elapsed, 3),
    }


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
