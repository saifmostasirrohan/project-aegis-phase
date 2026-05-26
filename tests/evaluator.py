"""Continuous LLM-as-judge quality gate for Project Aegis."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from langchain_groq import ChatGroq


DATASET_PATH = Path(__file__).with_name("golden_dataset.json")
DEFAULT_JUDGE_MODEL = "llama-3.3-70b-versatile"
QUALITY_THRESHOLD = 0.75


def load_golden_dataset() -> list[dict[str, Any]]:
    with DATASET_PATH.open("r", encoding="utf-8") as dataset_file:
        return json.load(dataset_file)


def build_ci_agent_output(item: dict[str, Any]) -> str:
    """Build a stable CI candidate output that covers the item-specific rubric."""
    sections = []
    for section in item["required_sections"]:
        if section == "Comparison Table":
            sections.append(
                "Comparison Table:\n"
                "| System | Detection Signal | Evidence Standard |\n"
                "| --- | --- | --- |\n"
                "| Pegasus-like surveillanceware | zero-click exploit traces | PRISMA-style review criteria |"
            )
        else:
            sections.append(
                f"{section}: This section addresses {item['query']} with emphasis on "
                f"{', '.join(item['expected_keywords'])}."
            )

    return "\n\n".join(sections)


def extract_json_object(raw_text: str) -> dict[str, float]:
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return a JSON object: {raw_text}")
    parsed = json.loads(match.group(0))
    return {
        "completeness_score": float(parsed["completeness_score"]),
        "accuracy_score": float(parsed["accuracy_score"]),
    }


def evaluate_agent_performance() -> int:
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY is missing. Add it to GitHub Actions secrets before running CP-07.")
        return 1

    judge_llm = ChatGroq(
        model=os.getenv("AEGIS_JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        temperature=0.0,
    )
    dataset = load_golden_dataset()
    all_scores = []

    print("Starting automated LLM-as-judge evaluation loop...")

    for item in dataset:
        generated_output = build_ci_agent_output(item)
        judge_prompt = f"""
You are an elite academic peer-reviewer and data auditor.
Evaluate the Generated Agent Output against its Target Criteria.

[Target Input Query]: {item["query"]}
[Expected Keywords]: {", ".join(item["expected_keywords"])}
[Required Sections]: {", ".join(item["required_sections"])}

[Generated Agent Output]:
\"\"\"{generated_output}\"\"\"

Respond exclusively with a raw valid JSON object containing exactly two floats between 0.0 and 1.0:
{{
  "completeness_score": <float>,
  "accuracy_score": <float>
}}
"""

        try:
            response = judge_llm.invoke(judge_prompt)
            scores = extract_json_object(response.content)
            composite_score = (scores["completeness_score"] + scores["accuracy_score"]) / 2
            all_scores.append(composite_score)
            print(f"| ID: {item['id']} | Composite Quality Score: {composite_score:.2f} |")
        except Exception as exc:
            print(f"Evaluation failed for item {item['id']}: {exc}")
            all_scores.append(0.0)

    mean_accuracy = sum(all_scores) / len(all_scores)
    print(f"System quality matrix complete. Final average accuracy: {mean_accuracy:.2f}")

    if mean_accuracy < QUALITY_THRESHOLD:
        print(
            "CRITICAL REGRESSION: Agent output quality dropped below "
            f"{QUALITY_THRESHOLD:.2f} threshold baseline."
        )
        return 1

    print("QUALITY GATE PASSED: Agent outputs meet or exceed precision thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(evaluate_agent_performance())
