"""LangChain MCP tool wrappers for the CP-06 Aegis Research Server."""

import asyncio
import sys
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


CAPSTONE_DIR = Path(__file__).resolve().parent
PHASE4_DIR = CAPSTONE_DIR.parent
CP06_SERVER_PATH = PHASE4_DIR / "cp06_mcp_server.py"
SERVER_NAME = "aegis_research"


class ArxivSearchInput(BaseModel):
    query: str = Field(description="Research query to search on ArXiv.")
    max_results: int = Field(default=3, ge=1, le=10, description="Number of papers to return.")


class FetchPaperInput(BaseModel):
    arxiv_id: str = Field(description="ArXiv ID, such as '1706.03762'.")


class SearchMyPapersInput(BaseModel):
    query: str = Field(description="Search query for Saif's local paper database.")


def make_mcp_client() -> MultiServerMCPClient:
    """Create a LangChain MCP client connected to the local CP-06 stdio server."""
    return MultiServerMCPClient(
        {
            SERVER_NAME: {
                "command": sys.executable,
                "args": [str(CP06_SERVER_PATH)],
                "transport": "stdio",
                "cwd": str(PHASE4_DIR),
            }
        }
    )


async def load_server_tools() -> dict[str, Any]:
    """Load raw LangChain tools exposed by the CP-06 MCP server."""
    client = make_mcp_client()
    tools = await client.get_tools(server_name=SERVER_NAME)
    return {tool.name: tool for tool in tools}


async def _call_mcp_tool(tool_name: str, tool_args: dict[str, Any]) -> str:
    tools = await load_server_tools()
    if tool_name not in tools:
        available = ", ".join(sorted(tools))
        raise RuntimeError(f"MCP tool '{tool_name}' not found. Available tools: {available}")

    result = await tools[tool_name].ainvoke(tool_args)
    return _content_to_text(result)


def _content_to_text(result: Any) -> str:
    """Normalize MCP content blocks into a plain string for the agent."""
    if isinstance(result, str):
        return result

    if isinstance(result, list):
        text_parts = []
        for item in result:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            else:
                text_parts.append(str(item))
        return "\n".join(part for part in text_parts if part)

    return str(result)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _retry_arxiv_search(query: str, max_results: int = 3) -> str:
    return await _call_mcp_tool(
        "arxiv_search",
        {"query": query, "max_results": max_results},
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _retry_fetch_paper(arxiv_id: str) -> str:
    return await _call_mcp_tool("fetch_paper", {"arxiv_id": arxiv_id})


async def _arxiv_search_async(query: str, max_results: int = 3) -> str:
    """Search ArXiv through the CP-06 MCP server with retry protection."""
    return await _retry_arxiv_search(query, max_results)


def _arxiv_search_sync(query: str, max_results: int = 3) -> str:
    return asyncio.run(_arxiv_search_async(query, max_results))


async def _fetch_paper_async(arxiv_id: str) -> str:
    """Fetch paper metadata through the CP-06 MCP server with retry protection."""
    return await _retry_fetch_paper(arxiv_id)


def _fetch_paper_sync(arxiv_id: str) -> str:
    return asyncio.run(_fetch_paper_async(arxiv_id))


async def _search_my_papers_async(query: str) -> str:
    """Search Saif's local paper database through the CP-06 MCP server."""
    return await _call_mcp_tool("search_my_papers", {"query": query})


def _search_my_papers_sync(query: str) -> str:
    return asyncio.run(_search_my_papers_async(query))


arxiv_search = StructuredTool.from_function(
    func=_arxiv_search_sync,
    coroutine=_arxiv_search_async,
    name="arxiv_search",
    description="Search ArXiv for research papers using the CP-06 MCP server.",
    args_schema=ArxivSearchInput,
)

fetch_paper = StructuredTool.from_function(
    func=_fetch_paper_sync,
    coroutine=_fetch_paper_async,
    name="fetch_paper",
    description="Fetch metadata for an ArXiv paper using the CP-06 MCP server.",
    args_schema=FetchPaperInput,
)

search_my_papers = StructuredTool.from_function(
    func=_search_my_papers_sync,
    coroutine=_search_my_papers_async,
    name="search_my_papers",
    description="Search Saif's local Phase 2 publication database using the CP-06 MCP server.",
    args_schema=SearchMyPapersInput,
)

tools = [arxiv_search, fetch_paper, search_my_papers]


async def list_mcp_tool_names() -> list[str]:
    """Return raw tool names currently exposed by the MCP server."""
    raw_tools = await load_server_tools()
    return sorted(raw_tools)


if __name__ == "__main__":
    names = asyncio.run(list_mcp_tool_names())
    print("Connected to CP-06 MCP server.")
    print("Available MCP tools:", ", ".join(names))
    print("LangChain wrapped tools:", ", ".join(tool.name for tool in tools))
