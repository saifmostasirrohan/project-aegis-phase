"""Streamlit frontend for the Capstone Autonomous Research Assistant."""

import os
import time

import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv("AEGIS_API_BASE_URL", "http://127.0.0.1:8000")
AEGIS_API_KEY = os.getenv("AEGIS_API_KEY", "")
POLL_SECONDS = 2


st.set_page_config(page_title="Aegis Research Assistant", page_icon="A", layout="wide")

st.title("Aegis Research Assistant")


def init_state() -> None:
    st.session_state.setdefault("thread_id", "")
    st.session_state.setdefault("last_status", {})
    st.session_state.setdefault("auto_poll", False)
    st.session_state.setdefault("api_base_url", DEFAULT_API_BASE_URL)


def api_post(path: str, payload: dict | None = None) -> dict:
    response = requests.post(
        f"{st.session_state.api_base_url}{path}",
        json=payload,
        headers=auth_headers(),
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {AEGIS_API_KEY}"} if AEGIS_API_KEY else {}


def api_get(path: str) -> dict:
    response = requests.get(f"{st.session_state.api_base_url}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def stream_review(thread_id: str):
    with requests.get(
        f"{st.session_state.api_base_url}/research/stream/{thread_id}",
        headers=auth_headers(),
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()

        event_type = "message"
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                event_type = "message"
                continue

            if line.startswith("event:"):
                event_type = line.removeprefix("event:").strip()
                continue

            if not line.startswith("data:"):
                continue

            data = line.removeprefix("data:")
            if data.startswith(" "):
                data = data[1:]
            if event_type == "done":
                break
            if event_type == "error":
                raise RuntimeError(data)
            if data:
                yield data


def progress_label(status: dict) -> str:
    if status.get("waiting_for_approval"):
        return "Waiting for approval before writing the draft."

    current_node = status.get("current_node", "")
    if current_node == "search_node":
        return "Searching ArXiv..."
    if current_node == "read_node":
        return "Reading papers and extracting text..."
    if current_node == "compare_node":
        return "Comparing against Saif's publication memory..."
    if current_node == "write_node":
        return "Writing the literature review..."
    if current_node == "END" or status.get("status") == "completed":
        return "Review complete."
    if status.get("status") == "queued":
        return "Queued..."
    if status.get("status") == "failed":
        return "Research failed."
    return "Working..."


def refresh_status() -> dict:
    thread_id = st.session_state.get("thread_id", "")
    if not thread_id:
        return {}

    status = api_get(f"/status/{thread_id}")
    st.session_state.last_status = status
    return status


def render_papers(papers: list[dict]) -> None:
    if not papers:
        st.info("No papers found yet.")
        return

    for index, paper in enumerate(papers, start=1):
        title = paper.get("title", "Untitled")
        arxiv_id = paper.get("arxiv_id", "unknown")
        authors = paper.get("authors", "Unknown authors")
        abstract = paper.get("abstract", "")

        with st.expander(f"{index}. {title}", expanded=index <= 3):
            st.caption(f"ArXiv ID: {arxiv_id}")
            st.write(f"Authors: {authors}")
            if abstract:
                st.write(abstract)


init_state()

with st.sidebar:
    st.subheader("Backend")
    st.session_state.api_base_url = st.text_input("API URL", value=st.session_state.api_base_url)
    if st.button("Refresh Status", use_container_width=True):
        try:
            refresh_status()
        except requests.RequestException as exc:
            st.error(f"Could not reach API: {exc}")

topic = st.text_input(
    "Research topic",
    value="Transformers in medical imaging",
    placeholder="Enter a research topic...",
)

start_col, id_col = st.columns([1, 3])
with start_col:
    if st.button("Start Research", type="primary", use_container_width=True):
        if not topic.strip():
            st.warning("Enter a research topic first.")
        else:
            try:
                result = api_post("/research", {"topic": topic.strip()})
                st.session_state.thread_id = result["thread_id"]
                st.session_state.last_status = result
                st.session_state.auto_poll = True
                st.success(f"Started research thread: {result['thread_id']}")
            except requests.RequestException as exc:
                st.error(f"Could not start research. Is FastAPI running? {exc}")

with id_col:
    st.text_input("Thread ID", value=st.session_state.thread_id, disabled=True)

status = st.session_state.get("last_status", {})

if st.session_state.thread_id:
    try:
        status = refresh_status()
    except requests.RequestException as exc:
        st.error(f"Could not fetch status: {exc}")
        status = st.session_state.get("last_status", {})

if status:
    st.subheader("Progress")
    st.status(progress_label(status), state="complete" if status.get("status") == "completed" else "running")

    metrics = st.columns(4)
    metrics[0].metric("Status", status.get("status", "unknown"))
    metrics[1].metric("Current Node", status.get("current_node", "unknown"))
    metrics[2].metric("Papers Found", status.get("papers_found", 0))
    metrics[3].metric("Extracted Records", status.get("extracted_texts", 0))

    if status.get("error"):
        st.error(status["error"])

    if status.get("waiting_for_approval"):
        st.subheader("Approval")
        st.write("Review the found papers before drafting the literature review.")
        render_papers(status.get("found_papers", []))

        if st.button("Approve & Stream Draft", type="primary"):
            try:
                st.session_state.auto_poll = False
                st.write_stream(stream_review(st.session_state.thread_id))
                refresh_status()
                st.success("Draft complete.")
            except (requests.RequestException, RuntimeError) as exc:
                st.error(f"Could not approve research: {exc}")

    if status.get("final_review"):
        st.subheader("Final Literature Review")
        st.markdown(status["final_review"])
        st.session_state.auto_poll = False

    should_poll = st.session_state.auto_poll and status.get("status") not in {
        "completed",
        "failed",
        "paused",
    }
    if should_poll:
        time.sleep(POLL_SECONDS)
        st.rerun()
