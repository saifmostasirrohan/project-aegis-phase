import os
import json
import ast
import re
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import pytz
from duckduckgo_search import DDGS
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv

# ==========================================
# 1. The 5 Python Tools
# ==========================================
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result."""
    try:
        # A simple safe eval for basic math
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")} if 'math' in globals() else {}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as exc:
        return f"Calculator error: {exc}"

def web_search(query: str) -> str:
    """Search the web with DuckDuckGo and return concise top text results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
    except Exception as exc:
        return f"Web search error: {exc}"

    if not results:
        return "No web search results found."

    formatted_results = []
    for index, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        body = result.get("body", "No snippet available.")
        formatted_results.append(f"{index}. {title}\nSnippet: {body}")
    return "\n\n".join(formatted_results)

def get_current_time(timezone: str) -> str:
    """Return the current date and time in a requested IANA timezone."""
    try:
        tz = pytz.timezone(timezone)
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    except pytz.UnknownTimeZoneError:
        return f"Unknown timezone: {timezone}."

def read_file(filepath: str) -> str:
    """Read a local UTF-8 text file and return its contents."""
    try:
        return Path(filepath).expanduser().read_text(encoding="utf-8")
    except Exception as exc:
        return f"Read file error: {exc}"

def save_note(title: str, content: str) -> str:
    """Save a note as a local Markdown file."""
    safe_title = re.sub(r"[^a-zA-Z0-9._ -]+", "", title).strip().replace(" ", "_")
    filepath = Path(f"{safe_title}.md")
    filepath.write_text(content, encoding="utf-8")
    return f"Saved note to {filepath.absolute()}"


# ==========================================
# 2. The JSON Tool Schemas for Groq
# ==========================================
def get_tool_schemas():
    """Tells the LLM exactly what tools exist and what arguments they take."""
    return [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a mathematical expression and return the numeric result.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string", "description": "Math expression, e.g., '13800000 * 14'"}},
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The search query."}},
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current time in a specific timezone.",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string", "description": "IANA timezone like 'Asia/Dhaka'"}},
                    "required": ["timezone"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a local text file.",
                "parameters": {
                    "type": "object",
                    "properties": {"filepath": {"type": "string"}},
                    "required": ["filepath"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "save_note",
                "description": "Save text content to a local markdown file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["title", "content"]
                }
            }
        }
    ]


# ==========================================
# 3. API Caller & ReAct Loop
# ==========================================
@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def call_groq(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "gsk_your_actual_api_key_here":
        raise RuntimeError("GROQ_API_KEY is missing or still set to the placeholder value in .env.")

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
        },
        timeout=30,
    )
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        raise requests.exceptions.HTTPError(
            f"{exc} | Groq response: {response.text}",
            response=response,
        ) from exc
    return response.json()


def agent_loop(user_query: str):
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True, encoding="utf-8-sig")
    
    available_tools = {
        "calculator": calculator,
        "web_search": web_search,
        "get_current_time": get_current_time,
        "read_file": read_file,
        "save_note": save_note,
    }

    tools = get_tool_schemas()
    messages = [
        {"role": "system", "content": "You are Aegis, a helpful AI. Use tools to find information before answering."},
        {"role": "user", "content": user_query},
    ]

    print(f"\n[User Query]: {user_query}\n" + "-"*50)

    for step in range(10): # Max 10 iterations to prevent infinite loops
        response_json = call_groq(messages, tools)
        assistant_message = response_json["choices"][0]["message"]
        tool_calls = assistant_message.get("tool_calls") or []

        # If no tools were called, the agent has its final answer!
        if not tool_calls:
            final_answer = assistant_message.get("content", "")
            print(f"\n[Final Answer]: {final_answer}\n")
            return

        # Add the agent's intent to call a tool into the history
        messages.append(assistant_message)

        # Execute the requested tools
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            
            print(f"[Tool Call] -> {tool_name}({arguments})")
            
            # Run the python function
            tool_func = available_tools.get(tool_name)
            tool_result = tool_func(**arguments)
            
            print(f"[Tool Result] <- {str(tool_result)[:200]}...\n")

            # Feed the result back to the LLM
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": tool_name,
                "content": str(tool_result),
            })

    print("Agent timed out after 10 steps.")


# ==========================================
# 4. Run the Test
# ==========================================
if __name__ == "__main__":
    test_query = "Find the current population of Tokyo, then multiply that number by 14. Show the source population number and final product."
    agent_loop(test_query)
