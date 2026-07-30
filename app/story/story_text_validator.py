"""Regex-based validation of story text for internal reference markers.

Scans narrative prose for programmatic/internal patterns that should never
appear in user-facing story text — node IDs, ``(Teil ...)`` markers, raw
JSON field names, and similar technical artefacts.

The pattern list is module-level and extensible: add a new ``(name, regex)``
tuple to :data:`INTERNAL_MARKER_PATTERNS` and the function picks it up
automatically — no logic changes required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class MarkerMatch:
    """A single internal-marker match found in story text."""

    pattern_name: str
    pattern: str
    matched_text: str
    offset: int


@dataclass
class ValidationResult:
    """Structured result of :func:`validate_story_text_markers`.

    Attributes
    ----------
    passed
        ``True`` when no internal markers were found.
    matches
        List of :class:`MarkerMatch` instances, one per occurrence.
    """

    passed: bool
    matches: list[MarkerMatch] = field(default_factory=list)

    def as_dict(self) -> dict:
        """Convert to a plain dict for JSON serialisation."""
        return {
            "passed": self.passed,
            "matches": [
                {
                    "pattern_name": m.pattern_name,
                    "pattern": m.pattern,
                    "matched_text": m.matched_text,
                    "offset": m.offset,
                }
                for m in self.matches
            ],
        }


# ── Extensible pattern list ───────────────────────────────────────
# Each entry is (name, compiled_regex).  Add new patterns here — the
# validation function iterates over this list automatically.

INTERNAL_MARKER_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Node IDs: node_001, node_002, node_14, etc.
    ("node_id", re.compile(r"node_\d+")),
    # (Teil abc), (Teil XYZ), (Teil 1 von 3), etc.
    ("teil_marker", re.compile(r"\(Teil\s+[^)]+\)")),
    # Raw JSON field names that may leak into prose: "scene_goal:", "next_node_id:"
    ("json_field", re.compile(r"\b(?:scene_goal|next_node_id|suggested_next_node|state_updates|quality_notes|auto_advance_delay_ms)\s*:")),
    # Bare suggested_next_node references like "suggested_next_node: node_003"
    ("suggested_next_node", re.compile(r"\bsuggested_next_node\b")),
]


def validate_story_text_markers(text: str) -> ValidationResult:
    """Scan *text* for internal/programmatic reference markers.

    Returns a :class:`ValidationResult` with ``passed=True`` when the text
    is clean, or ``passed=False`` with a populated ``matches`` list when
    internal markers are found.

    Parameters
    ----------
    text
        The story prose to validate.

    Returns
    -------
    ValidationResult
        ``passed`` is ``True`` iff no internal markers were found.
        ``matches`` contains one :class:`MarkerMatch` per occurrence,
        each carrying the pattern name, the regex source, the matched
        text fragment, and the character offset in *text*.
    """
    matches: list[MarkerMatch] = []

    if not text:
        return ValidationResult(passed=True, matches=matches)

    for name, pattern in INTERNAL_MARKER_PATTERNS:
        for m in pattern.finditer(text):
            matches.append(
                MarkerMatch(
                    pattern_name=name,
                    pattern=pattern.pattern,
                    matched_text=m.group(),
                    offset=m.start(),
                )
            )

    # Sort by offset for deterministic, left-to-right ordering
    matches.sort(key=lambda x: x.offset)

    return ValidationResult(passed=len(matches) == 0, matches=matches)
