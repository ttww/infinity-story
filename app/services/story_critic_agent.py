"""Story Critic Agent (Spec §7.4, §14.2).

Reviews a story graph for dramaturgy, consistency, decisions, and
safety.  Uses the LLM service abstraction so it works with mock or
real providers.

The agent enforces a structured output schema via Pydantic
(``CriticReviewReport``).  If the LLM returns invalid JSON or a
schema-mismatched response, the agent retries up to
``max_schema_retries`` times before raising ``StoryCriticError``.

Output shape (Spec §7.4)::

    {
        "score": float,              # 0.0–10.0
        "issues": [                  # Spec §7.4 issues list
            {
                "severity": "high" | "medium" | "low" | "info",
                "node_id": str | None,
                "problem": str,
                "suggestion": str,
            },
            ...
        ],
        "repair_suggestions": [str, ...],  # Spec §14.2
        "summary": str,
    }

The 13 review criteria (Spec §7.4):
  1.  Premise clarity
  2.  Conflict
  3.  Turning points
  4.  Decision relevance
  5.  Consequences
  6.  Dead ends
  7.  End reachability
  8.  Secret reveal timing
  9.  Character consistency
  10. Logic errors
  11. Linearity
  12. Audience fit
  13. Safety
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.services.llm_service import LLMService, LLMResponseError, get_llm_service
from app.story.prompts import CRITIC_SYSTEM_PROMPT, build_critic_user_prompt
from app.story.schemas import CriticReviewReport

logger = logging.getLogger(__name__)


# ── Internal reference patterns ──────────────────────────────────────

@dataclass
class InternalRefFinding:
    """A single internal reference found in story text."""
    node_id: str
    field_name: str
    pattern: str
    match_text: str


# Compiled patterns that detect internal / programmatic references.
# Each tuple is (label, compiled_regex).
_INTERNAL_REF_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Node IDs: node_001, node_002, Node_003, etc.
    ("node_\\d+", re.compile(r"node_\d+", re.IGNORECASE)),
    # "(Teil 1 von 3)", "(Teil 2)", "(Teil xxx)"
    ("(Teil \\d+)", re.compile(r"(Teil\s+\d+)", re.IGNORECASE)),
    # "[node_002]", "<node_001>", etc.
    ("[node_\\d+]", re.compile(r"[\[<]node_\d+[\]>]", re.IGNORECASE)),
    # "NODE_001", "NODE_002" (all-caps variants)
    ("NODE_\\d+", re.compile(r"NODE_\d+")),
    # "node 1", "node 2" (space variant)
    ("node \\d+", re.compile(r"\bnode\s+\d+", re.IGNORECASE)),
]

# Text fields in a node that contain story-facing content
_STORY_TEXT_FIELDS: list[str] = [
    "scene_text",
    "scene_goal",
    "title",
    "mood",
    "quality_notes",
    "location",
    "reveals",
]


class StoryCriticError(Exception):
    """Raised when the critic agent fails after all retries."""


class StoryCriticAgent:
    """Reviews a story graph and returns a structured critique report.

    The agent delegates text generation to an ``LLMService`` and
    enforces the output schema via ``CriticReviewReport`` (Pydantic).

    Parameters
    ----------
    llm
        Optional injected ``LLMService``.  If ``None``, the service
        is created lazily via ``get_llm_service``.
    max_schema_retries
        Number of retry attempts when the LLM returns invalid JSON
        or a schema-mismatched response.  Default: 2.
    """

    def __init__(
        self,
        llm: LLMService | None = None,
        *,
        max_schema_retries: int = 2,
    ) -> None:
        self._llm = llm
        self._max_schema_retries = max_schema_retries

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

    async def review(
        self,
        outline: dict[str, Any],
        graph: dict[str, Any],
        *,
        settings: Settings | None = None,
    ) -> dict[str, Any]:
        """Review the story graph and return a structured critique.

        Parameters
        ----------
        outline
            The story outline dict (premise, main_conflict, etc.).
        graph
            The directed story graph dict (nodes, start_node_id, ...).
        settings
            Optional settings override (unused by the base LLMService
            generate_json signature, but kept for API compatibility
            with future providers that may accept it).

        Returns
        -------
        dict
            ``{"score": float, "issues": [...], "repair_suggestions": [...], "summary": str}``

        Raises
        ------
        StoryCriticError
            If the LLM fails after all retries.
        """
        # settings is accepted for API compatibility but the base
        # LLMService.generate_json does not take it as a kwarg.
        _ = settings  # noqa: F841
        user_prompt = build_critic_user_prompt(outline, graph)

        last_exc: Exception | None = None
        for attempt in range(self._max_schema_retries + 1):
            try:
                result = await self.llm.generate_json(
                    system_prompt=CRITIC_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                )
                return self._merge_internal_ref_findings(
                    self._validate_and_normalise(result), graph,
                )
            except (LLMResponseError, ValidationError) as exc:
                last_exc = exc
                logger.warning(
                    "Critic review schema validation failed "
                    "(attempt %d/%d): %s",
                    attempt + 1,
                    self._max_schema_retries + 1,
                    exc,
                )
            except Exception as exc:
                logger.error("Critic review failed: %s", exc)
                raise

        model_info = f"[Model: {self.llm_name} / Provider: {self.llm.provider_name}]"
        raise StoryCriticError(
            f"Critic review failed after {self._max_schema_retries + 1} "
            f"attempts. {model_info} Last error: {last_exc}"
        ) from last_exc

    def _validate_and_normalise(
        self, raw: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate LLM output against the Pydantic schema and
        return a normalised dict.

        This method enforces the ``CriticReviewReport`` schema and
        raises ``ValidationError`` on mismatch (which triggers a retry
        in the calling loop).
        """
        report = CriticReviewReport.model_validate(raw)
        return report.model_dump()

    # ── Internal reference scanning ─────────────────────────────────

    @staticmethod
    def scan_internal_references(
        graph: dict[str, Any],
    ) -> list[InternalRefFinding]:
        """Scan all story-facing text fields in *graph* for internal /
        programmatic references.

        Returns a list of :class:`InternalRefFinding` objects, one per
        match.  An empty list means no internal references were found.
        """
        findings: list[InternalRefFinding] = []
        nodes = graph.get("nodes", {})
        if not isinstance(nodes, dict):
            return findings

        for node_id, node in nodes.items():
            if not isinstance(node, dict):
                continue

            # ── Scalar text fields ────────────────────────────────
            for field_name in _STORY_TEXT_FIELDS:
                value = node.get(field_name)
                if not isinstance(value, str):
                    continue
                for label, pattern in _INTERNAL_REF_PATTERNS:
                    for m in pattern.finditer(value):
                        findings.append(InternalRefFinding(
                            node_id=node_id,
                            field_name=field_name,
                            pattern=label,
                            match_text=m.group(0),
                        ))

            # ── quality_notes (list of strings) ───────────────────
            qn = node.get("quality_notes")
            if isinstance(qn, list):
                for item in qn:
                    if not isinstance(item, str):
                        continue
                    for label, pattern in _INTERNAL_REF_PATTERNS:
                        for m in pattern.finditer(item):
                            findings.append(InternalRefFinding(
                                node_id=node_id,
                                field_name="quality_notes",
                                pattern=label,
                                match_text=m.group(0),
                            ))

            # ── choices[].label (player-facing choice text) ───────
            choices = node.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    label_val = choice.get("label", "")
                    if not isinstance(label_val, str):
                        continue
                    for plabel, pattern in _INTERNAL_REF_PATTERNS:
                        for m in pattern.finditer(label_val):
                            findings.append(InternalRefFinding(
                                node_id=node_id,
                                field_name="choices.label",
                                pattern=plabel,
                                match_text=m.group(0),
                            ))

        return findings

    @staticmethod
    def _merge_internal_ref_findings(
        review: dict[str, Any],
        graph: dict[str, Any],
    ) -> dict[str, Any]:
        """Append internal-reference findings to *review* issues.

        Each finding is added as a ``high`` severity issue of the form::

            INTERNAL_REFERENCE_FOUND: <field> contains "<match>"

        The score is left unchanged (the LLM's assessment stands); the
        deterministic findings supplement the LLM critique.
        """
        findings = StoryCriticAgent.scan_internal_references(graph)
        if not findings:
            return review

        issues = review.get("issues", [])
        for f in findings:
            issues.append({
                "severity": "high",
                "node_id": f.node_id,
                "problem": (
                    f'INTERNAL_REFERENCE_FOUND: {f.field_name} '
                    f'contains programmatic reference "{f.match_text}"'
                ),
                "suggestion": (
                    f'Remove the internal reference "{f.match_text}" '
                    f'from {f.field_name} in {f.node_id}. Replace with '
                    f'natural story text.'
                ),
            })
        review["issues"] = issues
        return review

    # ── Convenience helpers ───────────────────────────────────────

    @staticmethod
    def has_high_severity_issues(review: dict[str, Any]) -> bool:
        """Check if a review report contains any high-severity issues."""
        issues = review.get("issues", [])
        return any(
            isinstance(i, dict) and i.get("severity", "").lower() == "high"
            for i in issues
        )

    @staticmethod
    def is_publishable(
        review: dict[str, Any],
        *,
        min_score: float = 7.0,
    ) -> bool:
        """Check if a review meets publication thresholds (Spec §15).

        - score >= min_score (default 7.0)
        - no high-severity issues
        """
        score = review.get("score", 0.0)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0
        return score >= min_score and not StoryCriticAgent.has_high_severity_issues(review)
