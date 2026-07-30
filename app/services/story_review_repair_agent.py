"""Combined Story Review + Repair Agent (Spec §7.4 + §7.5).

Instead of running separate critic and repair LLM calls (which produces
isolated, contradictory outputs), this agent does both in a single call:
- Reviews the graph against all 13 dramaturgy criteria
- Fixes every issue it finds directly in the graph
- Returns the complete repaired graph + score + issues

Iterative refinement: if the score is below 7.0, the agent re-runs with
its own repaired graph up to ``max_iterations`` times.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

from app.core.config import get_settings
from app.services.llm_service import LLMService, get_llm_service, LLMResponseError
from app.story.prompts import REVIEW_REPAIR_SYSTEM_PROMPT
from app.services.story_repair_agent import StoryRepairAgent

logger = logging.getLogger(__name__)


class ReviewRepairError(Exception):
    """Raised when review+repair fails after all iterations."""


class ReviewRepairAgent:
    """Combined review + repair agent with iterative refinement."""

    def __init__(
        self,
        llm: LLMService | None = None,
        max_iterations: int = 3,
    ) -> None:
        self._llm = llm
        self._max_iterations = max_iterations
        # Reuse the structural validator and patch applier from StoryRepairAgent
        self._validate_graph = StoryRepairAgent._validate_graph_structure
        self._apply_patches = StoryRepairAgent._apply_patches

    @property
    def llm(self) -> LLMService:
        if self._llm is None:
            self._llm = get_llm_service(get_settings())
        return self._llm

    async def review_and_repair(
        self,
        outline: dict[str, Any],
        graph: dict[str, Any],
    ) -> dict[str, Any]:
        """Review the graph against all criteria AND repair issues found.

        Iterates up to ``self._max_iterations`` times: each iteration
        feeds the *repaired* graph from the previous pass into the LLM
        again, so issues can be caught and fixed that were missed or
        introduced.

        Returns:
            ``{
                "score": float,
                "issues": [...],
                "repaired_graph": {...},
                "summary": str,
                "iterations_used": int,
            }``

        On catastrophic failure (all iterations produce invalid graphs),
        returns the last valid state with score 0.
        """
        user_prompt = (
            "=== STORY OUTLINE ===\n"
            f"{json.dumps(outline, ensure_ascii=False, indent=2)}\n\n"
            "=== STORY GRAPH ===\n"
            f"{json.dumps(graph, ensure_ascii=False, indent=2)}\n\n"
            "=== TASK ===\n"
            "Review the story graph against ALL 13 criteria from your "
            "instructions. Fix every issue by returning ONLY the patches "
            "(node_patches, new_nodes, deleted_nodes). Do NOT return "
            "the full graph."
        )

        current_graph = graph
        best_result: dict[str, Any] | None = None
        last_error: str | None = None

        for iteration in range(self._max_iterations):
            logger.info(
                "Review+Repair iteration %d/%d",
                iteration + 1, self._max_iterations,
            )

            try:
                result = await self.llm.generate_json(
                    system_prompt=REVIEW_REPAIR_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    max_tokens=65000,
                )

                score = result.get("score", 0.0)
                issues = result.get("issues", [])
                summary = result.get("summary", "")

                # Apply patches to the original graph (or keep unchanged)
                has_patches = bool(
                    result.get("node_patches")
                    or result.get("new_nodes")
                    or result.get("deleted_nodes")
                )
                if has_patches:
                    repaired_graph = self._apply_patches(current_graph, result)
                    logger.info(
                        "Iteration %d: applied patches → graph changed",
                        iteration + 1,
                    )
                else:
                    repaired_graph = current_graph
                    logger.info(
                        "Iteration %d: no patches — graph unchanged (score=%.1f)",
                        iteration + 1, score,
                    )

                # Validate structural integrity
                if not self._validate_graph(repaired_graph):
                    last_error = "Patched graph failed structural validation"
                    logger.warning("Iteration %d: %s", iteration + 1, last_error)
                    continue

                # Ensure start_node_id is set
                if repaired_graph.get("start_node_id") is None:
                    nodes = repaired_graph.get("nodes", {})
                    for nid, node in nodes.items():
                        if isinstance(node, dict) and (
                            node.get("is_start") or node.get("type") == "start"
                        ):
                            repaired_graph["start_node_id"] = nid
                            break
                    if repaired_graph.get("start_node_id") is None:
                        repaired_graph["start_node_id"] = next(iter(nodes), None)

                current_result = {
                    "score": score,
                    "issues": issues,
                    "repaired_graph": repaired_graph,
                    "summary": summary,
                    "iterations_used": iteration + 1,
                }

                # Track best by score
                if (best_result is None
                        or score > best_result.get("score", 0.0)):
                    best_result = current_result

                if score >= 7.0:
                    logger.info(
                        "Review+Repair converged at iteration %d (score %.1f)",
                        iteration + 1, score,
                    )
                    return current_result

                # Score < 7.0 — iterate with the repaired graph
                current_graph = repaired_graph
                user_prompt = (
                    "=== STORY OUTLINE ===\n"
                    f"{json.dumps(outline, ensure_ascii=False, indent=2)}\n\n"
                    "=== PREVIOUSLY REPAIRED GRAPH (still below threshold) ===\n"
                    f"{json.dumps(repaired_graph, ensure_ascii=False, indent=2)}\n\n"
                    f"=== PREVIOUS REVIEW ===\n"
                    f"Score: {score}/10\n"
                    f"Issues: {json.dumps(issues, ensure_ascii=False, indent=2)}\n\n"
                    "=== TASK ===\n"
                    "The story graph still needs improvement. Fix ALL remaining "
                    "issues listed above AND check for any new issues. "
                    "Return ONLY patches (node_patches, new_nodes, deleted_nodes)."
                )

            except (LLMResponseError, json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
                logger.warning(
                    "Review+Repair iteration %d failed: %s",
                    iteration + 1, exc,
                )
                # Keep current_graph unchanged for next iteration

        # All iterations exhausted — return best result or fallback
        if best_result is not None:
            logger.info(
                "Review+Repair exhausted %d iterations. Best score: %.1f",
                self._max_iterations, best_result["score"],
            )
            return best_result

        logger.error(
            "Review+Repair completely failed after %d iterations. "
            "Last error: %s",
            self._max_iterations, last_error,
        )
        return {
            "score": 0.0,
            "issues": [{
                "severity": "high",
                "node_id": None,
                "problem": f"Review+Repair fehlgeschlagen nach {self._max_iterations} Iterationen: {last_error}",
                "suggestion": "Manuelle Überprüfung erforderlich",
            }],
            "repaired_graph": graph,
            "summary": f"Review+Repair fehlgeschlagen (letzter Fehler: {last_error})",
            "iterations_used": self._max_iterations,
        }
