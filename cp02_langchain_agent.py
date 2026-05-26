import os
from datetime import datetime
from pathlib import Path

import pytz
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

try:
    from ddgs import DDGS  # Preferred package name.
except ImportError:
    from duckduckgo_search import DDGS  # Fallback for older installs.

load_dotenv()

# ==========================================
# 1. The Pydantic Schemas & LangChain Tools
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


# ==========================================
# 2. The LCEL Agent Executor
# ==========================================

# Group our beautifully decorated tools
tools = [calculator, web_search, get_current_time, read_file, save_note]

# Initialize the Groq LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Build the agent using the current LangChain API
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are Aegis, a helpful research agent. Use your tools to answer questions. If you don't know something, search for it.",
    debug=True,
)

# ==========================================
# 3. Run the Test
# ==========================================
if __name__ == "__main__":
    test_query = "What time is it currently in Tokyo? Once you have the time, save a note titled 'Tokyo Time' with the time in the content."

    print(f"Executing: {test_query}\n")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": test_query}]}
    )
    final_message = response["messages"][-1]
    final_output = getattr(final_message, "content", final_message)

    print("\n--- Final Output ---")
    print(final_output)
