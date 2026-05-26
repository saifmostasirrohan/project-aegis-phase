import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("CREWAI_STORAGE_DIR", str(BASE_DIR / "crewai_storage"))

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from ddgs import DDGS

load_dotenv(BASE_DIR / ".env")


@tool("duckduckgo_search")
def duckduckgo_search(query: str) -> str:
    """Search DuckDuckGo for current facts and return concise top results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as exc:
        return f"DuckDuckGo search error: {exc}"

    if not results:
        return "No DuckDuckGo results found."

    formatted_results = []
    for index, result in enumerate(results, start=1):
        title = result.get("title", "Untitled")
        href = result.get("href", "No URL")
        body = result.get("body", "No snippet available.")
        formatted_results.append(f"{index}. {title}\nURL: {href}\nSnippet: {body}")

    return "\n\n".join(formatted_results)


llm = "groq/llama-3.3-70b-versatile"
topic = "Transformers in Medical Imaging"

researcher = Agent(
    role="Medical AI Researcher",
    goal=f"Find accurate, current facts about {topic}.",
    backstory=(
        "You are a careful research analyst who specializes in medical AI. "
        "You prefer concrete facts, recent context, and clear source-backed findings."
    ),
    tools=[duckduckgo_search],
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

writer = Agent(
    role="Medical AI Writer",
    goal="Write clear, concise summaries based on research findings.",
    backstory=(
        "You turn technical research notes into readable summaries for AI builders "
        "who want the useful signal without losing scientific nuance."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

research_task = Task(
    description=(
        f"Research the topic: {topic}. Use DuckDuckGo search to identify key facts, "
        "applications, benefits, limitations, and recent trends. Focus on medical imaging."
    ),
    expected_output=(
        "A structured research brief with 5-7 bullet points, including concrete facts "
        "about how transformers are used in medical imaging."
    ),
    agent=researcher,
)

summary_task = Task(
    description=(
        "Using the researcher's findings, write a clean 2-paragraph summary. "
        "The first paragraph should explain what transformers contribute to medical imaging. "
        "The second paragraph should cover practical benefits, limitations, and why the topic matters."
    ),
    expected_output="A polished 2-paragraph summary with no bullet points.",
    agent=writer,
    context=[research_task],
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, summary_task],
    process=Process.sequential,
    verbose=True,
)


if __name__ == "__main__":
    result = crew.kickoff()
    print("\n--- CrewAI Final Output ---")
    print(result)
