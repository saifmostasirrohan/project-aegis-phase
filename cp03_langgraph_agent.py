import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

import pytz
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

try:
    from ddgs import DDGS  # Preferred package name.
except ImportError:
    from duckduckgo_search import DDGS  # Fallback for older installs.

load_dotenv()


# ==========================================
# 1. State
# ==========================================


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================
# 2. Tools
# ==========================================


class CalculatorInput(BaseModel):
    expression: str = Field(description="Math expression to evaluate, e.g., '13800000 * 14'")


@tool("calculator", args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result."""
    try:
        import math

        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as exc:
        return f"Calculator error: {exc}"


class WebSearchInput(BaseModel):
    query: str = Field(description="The search query string")


@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo for current information."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No web search results found."

        formatted_results = []
        for index, result in enumerate(results, start=1):
            title = result.get("title", "Untitled")
            body = result.get("body", "No snippet available.")
            formatted_results.append(f"{index}. {title}\nSnippet: {body}")
        return "\n\n".join(formatted_results)
    except Exception as exc:
        return f"Web search error: {exc}"


class TimeInput(BaseModel):
    timezone: str = Field(description="IANA timezone like 'Asia/Dhaka' or 'UTC'")


@tool("get_current_time", args_schema=TimeInput)
def get_current_time(timezone: str) -> str:
    """Return the current date and time in a requested IANA timezone."""
    try:
        tz = pytz.timezone(timezone)
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    except pytz.UnknownTimeZoneError:
        return f"Unknown timezone: {timezone}."


class ReadFileInput(BaseModel):
    filepath: str = Field(description="Path to the local text file")


@tool("read_file", args_schema=ReadFileInput)
def read_file(filepath: str) -> str:
    """Read a local UTF-8 text file and return its contents."""
    try:
        return Path(filepath).expanduser().read_text(encoding="utf-8")
    except Exception as exc:
        return f"Read file error: {exc}"


class SaveNoteInput(BaseModel):
    title: str = Field(description="Title of the note")
    content: str = Field(description="The body content of the note")


@tool("save_note", args_schema=SaveNoteInput)
def save_note(title: str, content: str) -> str:
    """Save a note as a local Markdown file."""
    import re

    safe_title = re.sub(r"[^a-zA-Z0-9._ -]+", "", title).strip().replace(" ", "_")
    filepath = Path(f"{safe_title}.md")
    filepath.write_text(content, encoding="utf-8")
    return f"Saved note to {filepath.absolute()}"


tools = [calculator, web_search, get_current_time, read_file, save_note]


# ==========================================
# 3. Nodes
# ==========================================


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
)
llm_with_tools = llm.bind_tools(tools)


def call_model(state: AgentState) -> dict[str, list[AIMessage]]:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools, name="tools")


# ==========================================
# 4. Graph
# ==========================================


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", call_model)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")

graph = graph_builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],
)


# ==========================================
# 5. Demo Run
# ==========================================


def print_last_message(state: dict) -> None:
    last_message = state["messages"][-1]
    print(last_message.pretty_repr())


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "phase4-cp03-demo"}}
    test_query = "Search the web for the capital of France."

    print(f"Executing: {test_query}\n")

    paused_state = graph.invoke(
        {"messages": [{"role": "user", "content": test_query}]},
        config=config,
    )
    print("--- Paused Before Tools ---")
    print_last_message(paused_state)

    input("\nPress Enter to approve the tool call and resume...")

    final_state = graph.invoke(None, config=config)
    print("\n--- Final Output ---")
    print_last_message(final_state)
