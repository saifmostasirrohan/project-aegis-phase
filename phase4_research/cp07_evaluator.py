import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langsmith import Client, evaluate


load_dotenv()

DATASET_NAME = "Medical Agent Eval"

EVAL_EXAMPLES = [
    {
        "inputs": {"question": "What is MicroHybridNet?"},
        "outputs": {
            "answer": (
                "MicroHybridNet is a dual-branch framework for medical imaging that combines "
                "complementary feature extraction paths to improve morphology-focused analysis."
            )
        },
    },
    {
        "inputs": {"question": "How should Aegis answer Saif's questions?"},
        "outputs": {
            "answer": "Aegis should answer Saif in concise bullet points when possible."
        },
    },
    {
        "inputs": {"question": "What is the purpose of semantic memory in CP-05?"},
        "outputs": {
            "answer": (
                "Semantic memory stores durable user facts and preferences, such as the user's "
                "name or preferred response format, in a user profile."
            )
        },
    },
    {
        "inputs": {"question": "What is episodic memory used for in the memory agent?"},
        "outputs": {
            "answer": (
                "Episodic memory stores past research notes or task summaries in a vector database "
                "so the agent can retrieve relevant context later through semantic search."
            )
        },
    },
    {
        "inputs": {"question": "What does the CP-06 MCP server provide?"},
        "outputs": {
            "answer": (
                "The CP-06 MCP server exposes research tools such as ArXiv search, paper fetching, "
                "local paper search, and a papers://list resource."
            )
        },
    },
]


def ensure_dataset(client: Client):
    """Create the eval dataset once, then add any missing examples."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Checkpoint 07 evaluation set for the Aegis medical research agent.",
        )

    existing_questions = {
        (example.inputs or {}).get("question")
        for example in client.list_examples(dataset_id=dataset.id)
    }

    for example in EVAL_EXAMPLES:
        question = example["inputs"]["question"]
        if question in existing_questions:
            continue

        client.create_example(
            dataset_id=dataset.id,
            inputs=example["inputs"],
            outputs=example["outputs"],
            metadata={"checkpoint": "cp07"},
        )

    return dataset


def agent_predict(inputs: dict[str, Any]) -> dict[str, str]:
    """Placeholder agent response function used to test the evaluation pipeline."""
    question = inputs["question"].lower()

    if "microhybridnet" in question:
        answer = "MicroHybridNet is a dual-branch medical imaging framework."
    elif "saif" in question or "answer" in question:
        answer = "Saif prefers concise bullet-point answers."
    elif "semantic memory" in question:
        answer = "Semantic memory saves lasting user profile facts and preferences."
    elif "episodic memory" in question:
        answer = "Episodic memory stores previous research notes in Chroma for later retrieval."
    elif "mcp" in question:
        answer = "The CP-06 MCP server exposes ArXiv and local paper-search tools."
    else:
        answer = "I do not know yet."

    return {"answer": answer}


judge_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


def _parse_judge_response(content: str) -> tuple[float, str]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        data = json.loads(match.group(0)) if match else {}

    score = float(data.get("score", 0.0))
    score = max(0.0, min(score, 1.0))
    rationale = str(data.get("rationale", content)).strip()
    return score, rationale


def llm_as_judge(run, example) -> dict[str, Any]:
    """Grade the predicted answer against the expected answer on a 0-1 scale."""
    question = (example.inputs or {}).get("question", "")
    expected = (example.outputs or {}).get("answer", "")
    actual = (run.outputs or {}).get("answer", "")

    prompt = f"""
You are a strict medical AI evaluation judge.

Question:
{question}

Expected answer:
{expected}

Actual answer:
{actual}

Score how well the actual answer matches the expected answer.
Use 1.0 for fully correct, 0.5 for partially correct, and 0.0 for incorrect.
Return only JSON in this exact shape:
{{"score": 0.0, "rationale": "brief reason"}}
"""

    response = judge_llm.invoke(prompt)
    score, rationale = _parse_judge_response(response.content)

    return {
        "key": "correctness",
        "score": score,
        "comment": rationale,
    }


if __name__ == "__main__":
    if not os.getenv("LANGCHAIN_API_KEY"):
        raise RuntimeError("LANGCHAIN_API_KEY is required for LangSmith evaluation.")
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("GROQ_API_KEY is required for the LLM-as-judge evaluator.")

    client = Client()
    dataset = ensure_dataset(client)

    results = evaluate(
        agent_predict,
        data=DATASET_NAME,
        evaluators=[llm_as_judge],
        experiment_prefix="cp07-medical-agent-eval",
        description="LLM-as-judge evaluation for the Aegis medical research agent.",
        client=client,
        max_concurrency=1,
    )

    print(f"Dataset ready: {dataset.name}")
    print(f"Experiment results: {results}")
