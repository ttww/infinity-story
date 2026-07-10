"""Story Repair Agent (Spec §7.5, §14.3).

Improves a story graph based on a critic review report.
Uses the LLM service abstraction so it works with mock or real providers.
"""

from __future__ import annotations

import copy
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
            "Improve the story graph now. Return ONLY the changes as "
            "node_patches, new_nodes, and deleted_nodes — do NOT return "
            "the full graph."
        )

        # Estimate tokens needed for reasoning overhead
        graph_str = json.dumps(graph, ensure_ascii=False)
        est_input_tokens = len(graph_str) // 4
        needed_tokens = min(max(est_input_tokens, 4096), 16384)
        needed_input_tokens = min(max(est_input_tokens * 3, 8192), 65536)

        try:
            result = await self.llm.generate_json(
                system_prompt=REPAIR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=needed_tokens,
                max_input_tokens=needed_input_tokens,
            )
        except Exception as exc:
            logger.error("Repair generation failed: %s", exc)
            raise

        # Apply patches to build the repaired graph
        repaired_graph = self._apply_patches(graph, result)

        return {
            "graph": repaired_graph,
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
        # Legacy fallback: if LLM returned a full graph, use it directly
        if "graph" in result and "node_patches" not in result:
            graph = result["graph"]
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
        enhanced = copy.deepcopy(original)
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
                enhanced["start_node_id"] = next(iter(nodes), None)

        return enhanced
