"""Free secondary-model fallback routing for Aegis LLM generation."""

import os

from fastapi import HTTPException
from langchain_groq import ChatGroq
import structlog


logger = structlog.get_logger()

FALLBACK_MODEL = os.getenv("AEGIS_FREE_FALLBACK_MODEL", "openai/gpt-oss-20b")
FALLBACK_TEMPERATURE = float(os.getenv("AEGIS_FREE_FALLBACK_TEMPERATURE", "0.3"))


def invoke_free_backup_fallback(prompt: str) -> str:
    """Redirect operational traffic to a secondary free Groq model backbone."""
    logger.warning(
        "primary_llm_backbone_failed_triggering_free_failover",
        model=FALLBACK_MODEL,
    )

    try:
        fallback_llm = ChatGroq(
            model=FALLBACK_MODEL,
            temperature=FALLBACK_TEMPERATURE,
        )
        response = fallback_llm.invoke(prompt)
        content = response.content
        return content if isinstance(content, str) else str(content)
    except Exception as exc:
        logger.error(
            "free_fallback_pipeline_exhausted",
            model=FALLBACK_MODEL,
            error=str(exc),
        )
        raise HTTPException(
            status_code=500,
            detail="All operational model routing paths exhausted. System entering safe lock.",
        ) from exc
