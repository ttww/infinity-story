"""Moderation / safety layer (Spec §5.8).

Simple filter with clear hook-points for future expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ModerationResult = Literal["pass", "flag", "block"]


@dataclass
class ModerationOutcome:
    result: ModerationResult
    reason: str = ""
    categories: list[str] = field(default_factory=list)


class ModerationService:
    """Basic content moderation with pluggable hooks."""

    # Simple keyword blocklist for MVP
    BLOCKED_PATTERNS: list[str] = [
        # Placeholder — real patterns injected from config in production
    ]

    async def check(self, text: str) -> ModerationOutcome:
        """Check text for policy violations."""
        text_lower = text.lower()
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in text_lower:
                return ModerationOutcome(
                    result="block",
                    reason=f"Blocked pattern matched: {pattern}",
                    categories=["keyword_filter"],
                )
        return ModerationOutcome(result="pass")
