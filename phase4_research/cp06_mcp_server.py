from pathlib import Path

import arxiv
import chromadb
from mcp.server.fastmcp import FastMCP


# Initialize the FastMCP server
mcp = FastMCP("Aegis Research Server")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PAPERS_DB_PATH = PROJECT_ROOT / "chroma_db"
PAPERS_COLLECTION_NAME = "aegis_papers"


@mcp.tool()
def arxiv_search(query: str, max_results: int = 3) -> str:
    """Search ArXiv for research papers. Returns Title, Authors, ID, and Abstract."""
    try:
        max_results = max(1, min(int(max_results), 10))
        client = arxiv.Client(page_size=max_results, num_retries=1)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for paper in client.results(search):
            authors = ", ".join([author.name for author in paper.authors])
            results.append(
                f"Title: {paper.title}\n"
                f"Authors: {authors}\n"
                f"ArXiv ID: {paper.get_short_id()}\n"
                f"Abstract: {paper.summary}\n"
            )

        if not results:
            return "No papers found."
        return "\n---\n".join(results)
    except Exception as exc:
        return f"ArXiv search failed: {exc}"


@mcp.tool()
def fetch_paper(arxiv_id: str) -> str:
    """Download and extract text from an ArXiv paper by its ID."""
    # Placeholder for the actual PDF extraction you will build in the Capstone.
    return (
        f"Successfully established connection to paper {arxiv_id}. "
        "Full PDF extraction ready for implementation."
    )


@mcp.tool()
def search_my_papers(query: str) -> str:
    """Search through my published papers in the local Chroma database."""
    try:
        chroma_client = chromadb.PersistentClient(path=str(PAPERS_DB_PATH))
        collection = chroma_client.get_collection(name=PAPERS_COLLECTION_NAME)

        results = collection.query(query_texts=[query], n_results=3)
        documents = results.get("documents", [[]])[0]

        if not documents:
            return "No relevant papers found in personal database."
        return "My Relevant Papers:\n" + "\n---\n".join(documents)
    except Exception as exc:
        return f"Database search failed: {exc}"


@mcp.resource("papers://list")
def list_my_papers() -> str:
    """Return a static list of my published work."""
    return (
        "1. MicroHybridNet: Dual-branch framework for cellular morphology\n"
        "2. OsteoNet: Hybrid Transformer-Ensemble architecture for bone density\n"
        "3. ReedMap: Accepted at HCII 2026\n"
        "4. AirTune: Accepted at HCII 2026\n"
        "[Note: Remaining 9 papers omitted for brevity]\n\n"
        f"Local Chroma collection: {PAPERS_DB_PATH}\\{PAPERS_COLLECTION_NAME}"
    )


if __name__ == "__main__":
    # Run the server using Standard Input/Output, required for Claude Desktop and Inspector.
    mcp.run(transport="stdio")
