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
            # With the diff-based prompt, the LLM only returns changed fields,
            # not the full graph. This dramatically reduces output size.
            # Estimate tokens needed based on graph size (for reasoning overhead).
            import json as _json
            graph_str = _json.dumps(graph, ensure_ascii=False)
            est_input_tokens = len(graph_str) // 4
            # LLM needs reasoning overhead + patch output.
            # Patches are typically much smaller than the full graph.
            # GLM-5.2 uses ~2x tokens for reasoning, so give generous room.
            needed_tokens = min(max(est_input_tokens, 4096), 16384)
            # Input budget: graph JSON + prompts. Use 3x input estimate
            # to give ample room for the full graph + system prompt.
            needed_input_tokens = min(max(est_input_tokens * 3, 8192), 65536)
            result = await self.llm.generate_json(
                system_prompt=ENHANCEMENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=needed_tokens,
                max_input_tokens=needed_input_tokens,
            )
        except Exception as exc:
            logger.error("Enhancement generation failed (mode=%s): %s", mode, exc)
            raise StoryEnhancementError(
                f"Enhancement failed: {exc}"
            ) from exc

        # Apply patches to build the enhanced graph
        enhanced_graph = self._apply_patches(graph, result)

        return {
            "graph": enhanced_graph,
            "changes": result.get("changes", []),
            "summary": result.get("summary", ""),
        }

    @staticmethod
    def _apply_patches(
        original: dict[str, Any],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Apply node patches from LLM result to the original graph.

        Handles both the new diff-based format (node_patches, new_nodes,
        deleted_nodes) and the legacy full-graph format (graph key).
        """
        import copy as _copy

        # Legacy fallback: if LLM returned a full graph, use it directly
        if "graph" in result and "node_patches" not in result:
            graph = result["graph"]
            # Ensure start_node_id
            if graph.get("start_node_id") is None:
                for nid, node in graph.get("nodes", {}).items():
                    if isinstance(node, dict) and (
                        node.get("is_start") or node.get("type") == "start"
                    ):
                        graph["start_node_id"] = nid
                        break
                if graph.get("start_node_id") is None:
                    graph["start_node_id"] = original.get("start_node_id")
            return graph

        # New diff-based format: deep-copy original and apply patches
        enhanced = _copy.deepcopy(original)
        nodes = enhanced.setdefault("nodes", {})

        # Apply field-level patches to existing nodes
        for nid, patch in result.get("node_patches", {}).items():
            if nid in nodes and isinstance(patch, dict):
                nodes[nid].update(patch)

        # Add new nodes
        for nid, node in result.get("new_nodes", {}).items():
            nodes[nid] = node

        # Remove deleted nodes (and fix dangling references)
        deleted = set(result.get("deleted_nodes", []))
        if deleted:
            for nid in deleted:
                nodes.pop(nid, None)
            # Remove choices pointing to deleted nodes
            for node in nodes.values():
                if isinstance(node, dict) and isinstance(node.get("choices"), list):
                    node["choices"] = [
                        c for c in node["choices"]
                        if not isinstance(c, dict)
                        or c.get("next_node_id") not in deleted
                    ]

        # Ensure start_node_id is still valid
        if enhanced.get("start_node_id") not in nodes:
            for nid, node in nodes.items():
                if isinstance(node, dict) and (
                    node.get("is_start") or node.get("type") == "start"
                ):
                    enhanced["start_node_id"] = nid
                    break
            if enhanced.get("start_node_id") not in nodes:
                # Fall back to first available node
                enhanced["start_node_id"] = next(iter(nodes), None)

        return enhanced
