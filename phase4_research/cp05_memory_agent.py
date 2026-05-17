import os
import json
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

import chromadb
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

langchain_api_key = os.getenv("LANGCHAIN_API_KEY", "")
if os.getenv("CP05_ENABLE_TRACING", "").lower() != "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ==========================================
# 1. Semantic Memory (User Profile)
# ==========================================
import threading

PROFILE_FILE = Path("user_profile.json")
# Create a thread lock to prevent WinError 32
profile_lock = threading.Lock()


def _load_profile() -> dict:
    with profile_lock:
        if PROFILE_FILE.exists():
            return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
        return {}


def _save_profile(data: dict):
    with profile_lock:
        # Write safely, allowing other processes to queue rather than crash
        PROFILE_FILE.write_text(json.dumps(data, indent=4), encoding="utf-8")


@tool("update_user_profile")
def update_user_profile(key: str, value: str) -> str:
    """Update a specific fact about the user in their permanent profile (e.g., 'name': 'Saif')."""
    profile = _load_profile()
    profile[key] = value
    _save_profile(profile)
    return f"Successfully updated profile: {key} = {value}"


@tool("get_user_profile")
def get_user_profile() -> str:
    """Retrieve all known facts and preferences about the user."""
    profile = _load_profile()
    if not profile:
        return "No user profile data found."
    return json.dumps(profile, indent=2)


# ==========================================
# 2. Episodic Memory (Chroma Vector DB)
# ==========================================
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Sentence Transformers is the default embedding function in Chroma
memory_collection = chroma_client.get_or_create_collection(name="agent_episodic_memory")


@tool("save_memory")
def save_memory(memory_text: str) -> str:
    """Save a summary of a completed task, a long-term fact, or research note to permanent vector memory."""
    memory_id = str(uuid.uuid4())
    memory_collection.add(
        documents=[memory_text],
        ids=[memory_id],
        metadatas=[{"type": "episodic_memory"}]
    )
    return "Memory successfully saved to vector database."


def _search_memory_text(query: str) -> str:
    results = memory_collection.query(
        query_texts=[query],
        n_results=3
    )

    documents = results.get("documents", [[]])[0]
    if not documents:
        return "No relevant memories found."

    return "Retrieved Memories:\n" + "\n---\n".join(documents)


@tool("search_memory")
def search_memory(query: str) -> str:
    """Search past memories, past tasks, and past research notes for relevant context."""
    return _search_memory_text(query)


def _build_memory_context(query: str) -> str:
    profile = _load_profile()
    profile_text = (
        json.dumps(profile, indent=2)
        if profile
        else "No user profile data found."
    )

    try:
        memory_text = _search_memory_text(query)
    except Exception as exc:
        memory_text = f"Memory search unavailable: {exc}"

    return (
        "Known user profile:\n"
        f"{profile_text}\n\n"
        "Relevant episodic memories:\n"
        f"{memory_text}"
    )


# ==========================================
# 3. LangGraph Agent Setup
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Add our new memory tools to the agent's arsenal
tools = [update_user_profile, get_user_profile, save_memory, search_memory]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)
llm_with_tools = llm.bind_tools(tools)


def call_model(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools, name="tools")

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")

# We still use MemorySaver for the short-term conversation thread
graph = graph_builder.compile(checkpointer=MemorySaver())


# ==========================================
# 4. Memory Testing Loop
# ==========================================
def run_agent(query: str, thread_id: str):
    print(f"\n[User]: {query}")
    config = {"configurable": {"thread_id": thread_id}}
    memory_context = _build_memory_context(query)

    # Inject the system prompt and the user query
    messages = [
        SystemMessage(content=(
            "You are Aegis, a persistent research assistant. Use the memory context below "
            "to answer personal or historical questions. Use memory tools when you need to "
            "save new facts or search for additional memories.\n\n"
            f"{memory_context}"
        )),
        {"role": "user", "content": query}
    ]

    result_state = graph.invoke({"messages": messages}, config=config)
    last_message = result_state["messages"][-1].content
    print(f"[Aegis]: {last_message}")


if __name__ == "__main__":
    # Test 1: Setting Semantic Memory
    run_agent("My name is Saif. I prefer my answers formatted as concise bullet points. Please save this to my profile.", "session_1")

    # Test 2: Setting Episodic Memory
    run_agent("I am currently researching MicroHybridNet for medical imaging. Please save a memory noting that my focus is on dual-branch frameworks.", "session_1")

    print("\n" + "=" * 50 + "\n--- CLEARING SHORT TERM CONVERSATION THREAD ---\n" + "=" * 50)

    # Test 3: Retrieving Memory (New Session)
    # Notice the thread_id is different! The agent has NO short-term memory of the previous lines.
    run_agent("What is my name, how do I like my answers formatted, and what specific framework am I researching?", "session_2")
