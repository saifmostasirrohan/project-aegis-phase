import asyncio
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient


def build_groq_client() -> OpenAIChatCompletionClient:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from phase4_research/.env")

    return OpenAIChatCompletionClient(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "family": "llama",
            "multiple_system_messages": False,
        },
    )


async def main() -> None:
    model_client = build_groq_client()

    # AutoGen 0.7 separates user input from code execution. This proxy represents
    # the user/terminal role; the CodeExecutorAgent below owns local execution.
    user_proxy = UserProxyAgent(
        name="user_proxy",
        description="An automated user proxy. Human input is disabled for this demo.",
        input_func=lambda _: "TERMINATE",
    )

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message=(
            "You are a coding assistant. When asked to execute code, your first response must contain "
            "only one executable Python code block and no explanation. Do not say TERMINATE until after "
            "the user_proxy_terminal returns the execution output. After execution output is returned, "
            "briefly explain the result and then reply with TERMINATE."
        ),
    )

    with tempfile.TemporaryDirectory(prefix="autogen_cp04_", dir=BASE_DIR) as work_dir:
        code_executor = LocalCommandLineCodeExecutor(
            timeout=30,
            work_dir=work_dir,
        )
        terminal = CodeExecutorAgent(
            name="user_proxy_terminal",
            code_executor=code_executor,
            description="Executes code locally in a temporary working directory.",
        )

        team = RoundRobinGroupChat(
            participants=[assistant, terminal],
            termination_condition=TextMentionTermination("TERMINATE"),
            max_turns=6,
        )

        task = (
            "Write a Python script to calculate the first 15 Fibonacci numbers and print them. "
            "Then execute it."
        )

        print(f"Temporary code execution directory: {work_dir}")
        print(f"User proxy: {user_proxy.name}")
        print(f"Task: {task}\n")

        try:
            await Console(team.run_stream(task=task))
        finally:
            await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
