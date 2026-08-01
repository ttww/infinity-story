"""Story Repair Agent (Spec §7.5, §14.3).

Improves a story graph based on a critic review report.
Uses the LLM service abstraction so it works with mock or real providers.
Added structured validation + retry to prevent broken graphs from
truncated LLM output from being saved.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from app.core.config import get_settings
from app.services.llm_service import LLMService, get_llm_service, LLMResponseError
from app.story.prompts import REPAIR_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class StoryRepairError(Exception):
    """Raised when repair fails after all retries."""


class StoryRepairAgent:
    """Repairs graph issues identified by the critic."""

    def __init__(self, llm: LLMService | None = None, max_retries: int = 2) -> None:
        self._llm = llm
        self._max_retries = max_retries

    @property
    def llm(self) -> LLMService:
        if self._llm is None:
            self._llm = get_llm_service(get_settings())
        return self._llm

    @property
    def llm_name(self) -> str:
        llm = self.llm
        model = getattr(llm, "_model", "?")
        ucase = llm.use_case or "default"
        return f"{model} ({ucase})"

    async def repair(
        self,
        graph: dict[str, Any],
        review_report: dict[str, Any],
    ) -> dict[str, Any]:
        """Return improved graph with documented changes.

        Retries up to ``self._max_retries`` times on JSON parse failure
        or invalid graph structure.  If all retries are exhausted, the
        **original graph** is returned unchanged rather than saving a
        truncated / broken graph to the database.

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

        last_error: str | None = None
        for attempt in range(self._max_retries + 1):
            try:
                result = await self.llm.generate_json(
                    system_prompt=REPAIR_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    max_tokens=needed_tokens,
                    max_input_tokens=needed_input_tokens,
                )

                # Apply patches to build the repaired graph
                repaired_graph = self._apply_patches(graph, result)

                # Validate structural integrity before accepting
                if self._validate_graph_structure(repaired_graph):
                    return {
                        "graph": repaired_graph,
                        "changes": result.get("changes", []),
                        "summary": result.get("summary", ""),
                    }

                last_error = "Graph validation failed (broken structure after patching)"
                logger.warning(
                    "Repair attempt %d/%d: %s",
                    attempt + 1, self._max_retries + 1, last_error,
                )

            except (LLMResponseError, json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Repair generation failed (attempt %d/%d): %s",
                    attempt + 1, self._max_retries + 1, exc,
                )

        # All retries exhausted — return original graph intact
        logger.error(
            "Repair failed after %d attempts. Returning original graph. "
            "Last error: %s",
            self._max_retries + 1, last_error,
        )
        model_info = f"[Model: {self.llm_name} / Provider: {self.llm.provider_name}]"
        return {
            "graph": graph,
            "changes": [],
            "summary": f"Reparatur fehlgeschlagen nach {self._max_retries + 1} Versuchen "
                        f"(letzter Fehler: {last_error}). {model_info} Original-Graph unverändert.",
        }

    @staticmethod
    def _validate_graph_structure(graph: dict[str, Any]) -> bool:
        """Check basic structural integrity of a story graph.

        Returns ``True`` if the graph is structurally sound:
          * ``nodes`` is a non-empty dict
          * ``start_node_id`` references a node that exists
          * every choice's ``next_node_id`` points to an existing node
          * node IDs are strings, nodes are dicts
        """
        nodes = graph.get("nodes")
        if not isinstance(nodes, dict):
            logger.warning("Graph validation: 'nodes' is not a dict")
            return False
        if len(nodes) == 0:
            logger.warning("Graph validation: 'nodes' is empty")
            return False

        # Validate each node
        for nid, node in nodes.items():
            if not isinstance(nid, str):
                logger.warning("Graph validation: node ID %r is not a string", nid)
                return False
            if not isinstance(node, dict):
                logger.warning("Graph validation: node %r is not a dict", nid)
                return False

        # Validate start_node_id
        start_id = graph.get("start_node_id")
        if start_id is None:
            logger.warning("Graph validation: 'start_node_id' is missing")
            return False
        if start_id not in nodes:
            logger.warning(
                "Graph validation: start_node_id %r not in nodes", start_id
            )
            return False

        # Validate choice references
        for nid, node in nodes.items():
            choices = node.get("choices") if isinstance(node, dict) else None
            if not isinstance(choices, list):
                continue
            for ci, choice in enumerate(choices):
                if not isinstance(choice, dict):
                    continue
                next_id = choice.get("next_node_id")
                if next_id is not None and next_id not in nodes:
                    logger.warning(
                        "Graph validation: choice %d of node %r "
                        "references non-existent node %r",
                        ci, nid, next_id,
                    )
                    return False

        return True

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