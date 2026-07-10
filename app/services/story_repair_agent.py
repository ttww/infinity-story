"""Story Repair Agent (Spec §7.5, §14.3).

Improves a story graph based on a critic review report.
Uses the LLM service abstraction so it works with mock or real providers.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.config import get_settings
from app.services.llm_service import LLMService, get_llm_service
from app.story.prompts import REPAIR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class StoryRepairAgent:
    """Repairs graph issues identified by the critic."""

    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm

    @property
    def llm(self) -> LLMService:
        if self._llm is None:
            self._llm = get_llm_service(get_settings())
        return self._llm

    async def repair(
        self,
        graph: dict[str, Any],
        review_report: dict[str, Any],
    ) -> dict[str, Any]:
        """Return improved graph with documented changes.

        Returns:
            ``{"graph": {...}, "changes": [...], "summary": str}``
        """
        user_prompt = (
            "Current story graph:\n"
            f"{json.dumps(graph, ensure_ascii=False, indent=2)}\n\n"
            "Critic review report:\n"
            f"{json.dumps(review_report, ensure_ascii=False, indent=2)}\n\n"
            "Improve the story graph now. Return the full improved graph."
        )
        try:
            result = await self.llm.generate_json(
                system_prompt=REPAIR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.error("Repair generation failed: %s", exc)
            raise
        # The result may be the graph itself or a wrapper
        if "nodes" in result and "graph" not in result:
            result = {"graph": result, "changes": [], "summary": ""}
        result.setdefault("graph", graph)  # fallback to original
        result.setdefault("changes", [])
        result.setdefault("summary", "")
        return result
