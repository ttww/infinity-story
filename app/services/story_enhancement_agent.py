"""Story Enhancement Agent — Multi-Pass Story Enhancement.

Provides systematic story deepening after initial generation:
  1. Multi-Pass Enhancement — improve entire graph (atmosphere, characters, choices)
  2. Critic-Feedback-Based — batch-repair based on critic review issues
  3. Story-Arc-Expansion — expand thin acts with new nodes
  4. Character-Deepening — add relationships, secrets, character arcs
  5. Thematic-Deepening — add sub-plots, foreshadowing, recurring motifs

The agent delegates to the LLM service abstraction so it works with mock
or real providers.  Uses the same version-tracking and diff system as the
incremental editing pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.config import get_settings
from app.services.llm_service import LLMService, get_llm_service
from app.story.prompts import ENHANCEMENT_SYSTEM_PROMPT, build_enhancement_user_prompt

logger = logging.getLogger(__name__)


# ── Enhancement modes ─────────────────────────────────────────────────

ENHANCEMENT_MODES = (
    "atmosphere",
    "characters",
    "choices",
    "arc_expansion",
    "thematic",
    "critic_based",
)


class StoryEnhancementError(Exception):
    """Raised when the enhancement agent fails."""


class StoryEnhancementAgent:
    """Enhances a story graph using LLM-driven multi-pass improvement.

    Parameters
    ----------
    llm
        Optional injected ``LLMService``.  If ``None``, the service
        is created lazily via ``get_llm_service``.
    """

    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm

    @property
    def llm(self) -> LLMService:
        if self._llm is None:
            self._llm = get_llm_service(get_settings())
        return self._llm

    async def enhance(
        self,
        graph: dict[str, Any],
        mode: str,
        *,
        instruction: str = "",
        review_report: dict[str, Any] | None = None,
        target_act: int | None = None,
        add_node_count: int | None = None,
    ) -> dict[str, Any]:
        """Enhance a story graph and return the improved version.

        Parameters
        ----------
        graph
            The current story graph dict (nodes, start_node_id, ...).
        mode
            Enhancement mode. Must be one of ``ENHANCEMENT_MODES``.
        instruction
            Free-text instruction for the LLM.
        review_report
            Critic review report (required for ``critic_based`` mode).
        target_act
            Act to target (for ``arc_expansion`` mode).
        add_node_count
            Number of nodes to add (for ``arc_expansion`` mode).

        Returns
        -------
        dict
            ``{"graph": {...}, "changes": [...], "summary": str}``

        Raises
        ------
        StoryEnhancementError
            If the mode is invalid or the LLM fails.
        """
        if mode not in ENHANCEMENT_MODES:
            raise StoryEnhancementError(
                f"Invalid enhancement mode '{mode}'. "
                f"Valid modes: {', '.join(ENHANCEMENT_MODES)}"
            )

        if mode == "critic_based" and review_report is None:
            raise StoryEnhancementError(
                "critic_based mode requires a review_report"
            )

        user_prompt = build_enhancement_user_prompt(
            graph=graph,
            mode=mode,
            instruction=instruction,
            review_report=review_report,
            target_act=target_act,
            add_node_count=add_node_count,
        )

        try:
            result = await self.llm.generate_json(
                system_prompt=ENHANCEMENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.error("Enhancement generation failed (mode=%s): %s", mode, exc)
            raise StoryEnhancementError(
                f"Enhancement failed: {exc}"
            ) from exc

        # Normalise result: the LLM may return just the graph or a wrapper
        if "nodes" in result and "graph" not in result:
            # LLM returned the graph directly without a wrapper
            result = {
                "graph": result,
                "changes": [],
                "summary": "",
            }

        result.setdefault("graph", graph)  # fallback to original
        result.setdefault("changes", [])
        result.setdefault("summary", "")

        # Ensure the enhanced graph has a start_node_id
        enhanced_graph = result["graph"]
        if enhanced_graph.get("start_node_id") is None:
            for nid, node in enhanced_graph.get("nodes", {}).items():
                if isinstance(node, dict) and (
                    node.get("is_start") or node.get("type") == "start"
                ):
                    enhanced_graph["start_node_id"] = nid
                    break
            # Fall back to original start_node_id
            if enhanced_graph.get("start_node_id") is None:
                enhanced_graph["start_node_id"] = graph.get("start_node_id")

        return result
