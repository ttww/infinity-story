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
from typing import Any

from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.services.llm_service import LLMService, LLMResponseError, get_llm_service
from app.story.prompts import CRITIC_SYSTEM_PROMPT, build_critic_user_prompt
from app.story.schemas import CriticReviewReport

logger = logging.getLogger(__name__)


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
                return self._validate_and_normalise(result)
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

        raise StoryCriticError(
            f"Critic review failed after {self._max_schema_retries + 1} "
            f"attempts. Last error: {last_exc}"
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
