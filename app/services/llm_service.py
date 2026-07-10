"""LLM service abstraction (Spec §5.6).

Provides a pluggable provider interface for text and JSON generation:
  - mock        — deterministic, zero-cost, offline
  - openai      — OpenAI Chat Completions API (httpx)
  - azure_openai— Azure OpenAI Service (httpx)
  - ollama      — local Ollama REST API (httpx)

Features:
  * Structured prompt support (system + user messages)
  * Optional JSON output with schema instruction
  * Error handling with retries and exponential backoff
  * Token / cost tracking with per-call and cumulative accounting
  * Daily budget enforcement
  * Input token budget truncation
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timezone
from datetime import datetime
from typing import Any

import httpx

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Raw response from an LLM provider call."""

    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    raw: dict[str, Any] | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class UsageStats:
    """Cumulative token and cost statistics."""

    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def record(self, response: LLMResponse) -> None:
        self.total_calls += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_cost_usd += response.cost_usd
        self.calls.append(
            {
                "provider": response.provider,
                "model": response.model,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": round(response.cost_usd, 6),
                "elapsed_seconds": round(response.elapsed_seconds, 3),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


class LLMError(Exception):
    """Base error for LLM provider failures."""


class LLMTimeoutError(LLMError):
    """Request timed out."""


class LLMRateLimitError(LLMError):
    """Provider returned a rate-limit response."""


class LLMResponseError(LLMError):
    """Provider returned an unparseable or invalid response."""


class LLMBudgetExceededError(LLMError):
    """Daily token/cost budget exceeded."""


# ── Abstract interface ───────────────────────────────────────────

class LLMService(ABC):
    """Abstract LLM provider interface (Spec §5.6)."""

    provider_name: str

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.usage = UsageStats()

    # ── Core abstract methods ────────────────────────────

    @abstractmethod
    async def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Provider-specific completion call."""
        ...

    # ── Public API ───────────────────────────────────────

    async def generate_scene(
        self,
        story_context: Any,
        user_input: str | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Any:
        """Generate a story scene from a StoryContext.

        ``story_context`` is a StoryContext dataclass (from story_orchestrator);
        ``user_input`` overrides story_context.user_input if provided.

        Returns a GeneratedScene dataclass (from story_orchestrator).
        """
        # Import here to avoid circular imports at module load time
        from app.services.story_orchestrator import GeneratedScene, StoryContext
        from app.story.prompts import SCENE_SYSTEM_PROMPT, build_scene_user_prompt

        ctx: StoryContext = story_context
        effective_input = user_input if user_input is not None else ctx.user_input

        node = ctx.current_node or {}
        user_prompt = build_scene_user_prompt(
            node_id=node.get("id", "unknown"),
            scene_goal=node.get("scene_goal", ""),
            location=node.get("location", ""),
            characters=node.get("characters", []),
            world_state=ctx.world_state,
            user_input=effective_input,
            title=node.get("title", ""),
            genre=ctx.world_state.get("genre", ""),
            tone=ctx.world_state.get("tone", ""),
            language=ctx.world_state.get("language", "de"),
            reveals=node.get("reveals", []),
            predefined_choices=node.get("choices", []) or ctx.available_choices,
            history=ctx.history,
        )

        response = await self._complete(
            SCENE_SYSTEM_PROMPT,
            user_prompt,
            json_mode=True,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        scene = self._parse_scene_json(response.text)
        return scene

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Generate a JSON-structured response."""
        response = await self._complete(
            system_prompt,
            user_prompt,
            json_mode=True,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return self._parse_json(response.text)

    async def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate plain text."""
        response = await self._complete(
            system_prompt,
            user_prompt,
            json_mode=False,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.text

    # ── Shared helpers ───────────────────────────────────

    def _parse_json(self, text: str) -> dict[str, Any]:
        """Parse JSON from LLM output, tolerating markdown code fences."""
        if text is None:
            raise LLMResponseError("LLM returned None (no content). The model may have used all tokens for reasoning.")
        cleaned = text.strip()
        # Strip markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove first line (```json or ```) and last ``` if present
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Invalid JSON from LLM: {exc}\nText: {text[:500]}") from exc

    def _parse_scene_json(self, text: str) -> Any:
        """Parse a scene JSON response into a GeneratedScene."""
        from app.services.story_orchestrator import GeneratedScene

        data = self._parse_json(text)
        return GeneratedScene(
            scene_text=data.get("scene_text", ""),
            choices=data.get("choices", []),
            state_updates=data.get("state_updates", {}),
            suggested_next_node=data.get("suggested_next_node"),
        )

    def _truncate_input(self, prompt: str, max_input_tokens: int) -> str:
        """Truncate the input prompt to stay within the token budget.

        Uses a simple chars/4 approximation for tokens.
        """
        max_chars = max_input_tokens * 4
        if len(prompt) <= max_chars:
            return prompt
        logger.warning(
            "Input prompt truncated from %d to %d chars (budget: %d tokens)",
            len(prompt),
            max_chars,
            max_input_tokens,
        )
        return prompt[:max_chars]

    def _check_budget(self) -> None:
        """Raise if the daily cost budget has been exceeded."""
        budget = self.settings.llm_daily_budget_usd
        if budget > 0 and self.usage.total_cost_usd >= budget:
            raise LLMBudgetExceededError(
                f"Daily budget exceeded: ${self.usage.total_cost_usd:.4f} >= ${budget:.4f}"
            )

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost based on configured per-1k-token rates."""
        return (
            input_tokens / 1000 * self.settings.llm_cost_per_1k_input
            + output_tokens / 1000 * self.settings.llm_cost_per_1k_output
        )

    async def _retry_with_backoff(self, fn, *args, **kwargs) -> Any:
        """Call *fn* with exponential-backoff retries.

        Retries on LLMTimeoutError, LLMRateLimitError, and httpx transport errors.
        Does NOT retry on LLMResponseError (invalid output) or LLMBudgetExceededError.
        """
        max_retries = self.settings.llm_max_retries
        base = self.settings.llm_retry_backoff_base

        for attempt in range(max_retries + 1):
            try:
                return await fn(*args, **kwargs)
            except (LLMTimeoutError, LLMRateLimitError) as exc:
                if attempt == max_retries:
                    raise
                delay = base ** attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == max_retries:
                    raise LLMTimeoutError(str(exc)) from exc
                delay = base ** attempt
                logger.warning(
                    "HTTP transport error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    max_retries + 1,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

    def _record_usage(self, response: LLMResponse) -> None:
        """Record usage stats and check budget."""
        self.usage.record(response)
        self._check_budget()


# ── Mock provider ─────────────────────────────────────────────────

# ── Authoring mock data ──────────────────────────────────────────
# These are returned by MockLLMService for authoring pipeline prompts.

_MOCK_OUTLINE: dict[str, Any] = {
    "premise": (
        "Auf der Forschungsstation Helios auf dem Mars verschwinden "
        "Crewmitglieder unter mysteriösen Umständen."
    ),
    "main_conflict": (
        "Die Crew muss herausfinden, ob eine außerirdische Präsenz "
        "oder ein Saboteur unter ihnen für das Verschwinden verantwortlich ist."
    ),
    "core_mystery": (
        "Die verschwundenen Crewmitglieder tauchen wieder auf, "
        "aber sie verhalten sich anders."
    ),
    "main_characters": [
        {"name": "Dr. Lena Voss", "role": "Commander",
         "secret": "Hat Kontakt zum Mars-Mysterium"},
        {"name": "Kai Chen", "role": "Ingenieur",
         "secret": "Sabotiert heimlich Systeme"},
        {"name": "Mara Singh", "role": "Medizinerin",
         "secret": "Weiß von der Infektion"},
    ],
    "endings": [
        "Die Crew entdeckt die Wahrheit und kann die Station evakuieren.",
        "Der Saboteur wird entlarvt, aber die Station geht verloren.",
        "Alle werden Teil des Mars-Mysteriums.",
    ],
}

_MOCK_GRAPH: dict[str, Any] = {
    "nodes": {
        "node_001": {
            "id": "node_001",
            "title": "Ankunft auf Helios",
            "type": "start",
            "act": 1,
            "scene_goal": "Spieler trifft auf der Station ein.",
            "mood": "unheimlich",
            "location": "Docking Bay",
            "characters": ["Dr. Lena Voss"],
            "reveals": [],
            "choices": [
                {"id": "c1", "label": "Zum Briefing gehen",
                 "next_node_id": "node_002"},
                {"id": "c2", "label": "Station erkunden",
                 "next_node_id": "node_003"},
            ],
            "quality_notes": ["Startknoten mit starker Atmosphäre"],
            "is_start": True,
            "is_end": False,
        },
        "node_002": {
            "id": "node_002",
            "title": "Das Briefing",
            "type": "scene",
            "act": 1,
            "scene_goal": "Mission wird erklärt, erste Hinweise auf Verschwindungen.",
            "mood": "spannend",
            "location": "Konferenzraum",
            "characters": ["Dr. Lena Voss", "Kai Chen"],
            "reveals": ["Crewmitglied X verschwunden"],
            "choices": [
                {"id": "c3", "label": "Kai befragen",
                 "next_node_id": "node_004"},
                {"id": "c4", "label": "Maras Tagebuch lesen",
                 "next_node_id": "node_005"},
            ],
            "quality_notes": [],
            "is_start": False,
            "is_end": False,
        },
        "node_003": {
            "id": "node_003",
            "title": "Verlassener Korridor",
            "type": "scene",
            "act": 1,
            "scene_goal": "Spieler findet Spuren eines Kampfes.",
            "mood": "bedrohlich",
            "location": "Korridor C-7",
            "characters": [],
            "reveals": ["Kampfspuren"],
            "choices": [
                {"id": "c5", "label": "Zum Briefing eilen",
                 "next_node_id": "node_002"},
                {"id": "c6", "label": "Weiter erkunden",
                 "next_node_id": "node_005"},
            ],
            "quality_notes": [],
            "is_start": False,
            "is_end": False,
        },
        "node_004": {
            "id": "node_004",
            "title": "Kais Geheimnis",
            "type": "scene",
            "act": 2,
            "scene_goal": "Kai wird verdächtigt und verrät etwas.",
            "mood": "misstrauisch",
            "location": "Maschinenraum",
            "characters": ["Kai Chen"],
            "reveals": ["Kai saboteurt Systeme"],
            "choices": [
                {"id": "c7", "label": "Kai konfrontieren",
                 "next_node_id": "node_006"},
                {"id": "c8", "label": "Beweise sammeln",
                 "next_node_id": "node_005"},
            ],
            "quality_notes": [],
            "is_start": False,
            "is_end": False,
        },
        "node_005": {
            "id": "node_005",
            "title": "Maras Tagebuch",
            "type": "scene",
            "act": 2,
            "scene_goal": "Spieler entdeckt die Infektion.",
            "mood": "schockierend",
            "location": "Medbay",
            "characters": ["Mara Singh"],
            "reveals": ["Infektion durch Mars-Organismus"],
            "choices": [
                {"id": "c9", "label": "Quarantäne auslösen",
                 "next_node_id": "node_006"},
                {"id": "c10", "label": "Heilmittel suchen",
                 "next_node_id": "node_006"},
            ],
            "quality_notes": [],
            "is_start": False,
            "is_end": False,
        },
        "node_006": {
            "id": "node_006",
            "title": "Das Mars-Mysterium",
            "type": "scene",
            "act": 3,
            "scene_goal": "Endgültige Konfrontation mit der Wahrheit.",
            "mood": "enthüllend",
            "location": "Tiefste Ebene",
            "characters": ["Dr. Lena Voss", "Kai Chen", "Mara Singh"],
            "reveals": ["Die Station ist Teil einer größeren Anlage"],
            "choices": [
                {"id": "c11", "label": "Evakuierung einleiten",
                 "next_node_id": "node_007"},
                {"id": "c12", "label": "Saboteur ausliefern",
                 "next_node_id": "node_008"},
                {"id": "c13", "label": "Dem Mysterium beitreten",
                 "next_node_id": "node_009"},
            ],
            "quality_notes": [],
            "is_start": False,
            "is_end": False,
        },
        "node_007": {
            "id": "node_007",
            "title": "Evakuierung",
            "type": "end",
            "act": 3,
            "scene_goal": "Die Crew entkommt, Station geht verloren.",
            "mood": "bittersüß",
            "location": "Evakuierungsshuttle",
            "characters": [],
            "reveals": [],
            "choices": [],
            "quality_notes": [],
            "is_start": False,
            "is_end": True,
        },
        "node_008": {
            "id": "node_008",
            "title": "Gerechtigkeit",
            "type": "end",
            "act": 3,
            "scene_goal": "Saboteur wird überführt.",
            "mood": "gerecht",
            "location": "Brücke",
            "characters": [],
            "reveals": [],
            "choices": [],
            "quality_notes": [],
            "is_start": False,
            "is_end": True,
        },
        "node_009": {
            "id": "node_009",
            "title": "Transformation",
            "type": "end",
            "act": 3,
            "scene_goal": "Alle werden Teil des Mysteriums.",
            "mood": "fremdartig",
            "location": "Tiefste Ebene",
            "characters": [],
            "reveals": [],
            "choices": [],
            "quality_notes": [],
            "is_start": False,
            "is_end": True,
        },
    },
    "start_node_id": "node_001",
}

_MOCK_REVIEW: dict[str, Any] = {
    "score": 7.5,
    "issues": [
        {
            "severity": "medium",
            "node_id": "node_003",
            "problem": "Szene könnte mehr Spannung aufbauen.",
            "suggestion": "Füge ein Geräusch hinzu, das den Spieler alarmiert.",
        },
        {
            "severity": "info",
            "node_id": "node_006",
            "problem": "Die drei Enden gehen alle vom gleichen Knoten aus.",
            "suggestion": "Mehr Verzweigung im dritten Akt würde Tiefe geben.",
        },
    ],
    "repair_suggestions": [
        "node_003: Füge ein auditives Element hinzu, um die Spannung zu erhöhen.",
        "Akt 3: Verteile die Enden auf verschiedene Pfade statt alle von node_006.",
    ],
    "summary": (
        "Solider Graph mit gutem Pacing. Einige Szenen "
        "könnten dramaturgisch stärker sein."
    ),
}

_MOCK_ENHANCEMENT_GRAPH: dict[str, Any] = {
    "nodes": {
        "node_001": {
            "id": "node_001",
            "title": "Ankunft auf Helios",
            "type": "start",
            "act": 1,
            "scene_goal": "Spieler trifft auf der Station ein. Das kalte, flackernde Licht der Docking Bay empfängt ihn mit einem Summen, das nach Verlassenheit klingt.",
            "mood": "unheimlich",
            "location": "Docking Bay",
            "characters": ["Dr. Lena Voss"],
            "reveals": [],
            "choices": [
                {"id": "c1", "label": "Zum Briefing gehen", "next_node_id": "node_002"},
                {"id": "c2", "label": "Station erkunden", "next_node_id": "node_003"},
            ],
            "quality_notes": ["Startknoten mit starker Atmosphäre", "Enhanced: detaillierte Sinnesbeschreibungen"],
            "is_start": True,
            "is_end": False,
        },
        "node_002": {
            "id": "node_002",
            "title": "Das Briefing",
            "type": "scene",
            "act": 1,
            "scene_goal": "Mission wird erklärt, erste Hinweise auf Verschwindungen. Lena wirkt angespannt und vermeidet Blickkontakt.",
            "mood": "spannend",
            "location": "Konferenzraum",
            "characters": ["Dr. Lena Voss", "Kai Chen"],
            "reveals": ["Crewmitglied X verschwunden"],
            "choices": [
                {"id": "c3", "label": "Kai befragen — vorsichtig nachhaken", "next_node_id": "node_004"},
                {"id": "c4", "label": "Maras Tagebuch lesen — heimlich", "next_node_id": "node_005"},
            ],
            "quality_notes": ["Enhanced: komplexere Choice-Beschreibungen"],
            "is_start": False,
            "is_end": False,
        },
        "node_003": {
            "id": "node_003",
            "title": "Verlassener Korridor",
            "type": "scene",
            "act": 1,
            "scene_goal": "Spieler findet Spuren eines Kampfes und hört ein seltsames Geräusch aus den Lüftungsschachten.",
            "mood": "bedrohlich",
            "location": "Korridor C-7",
            "characters": [],
            "reveals": ["Kampfspuren", "Seltsames Geräusch"],
            "choices": [
                {"id": "c5", "label": "Zum Briefing eilen", "next_node_id": "node_002"},
                {"id": "c6", "label": "Weiter erkunden — trotz des Geräuschs", "next_node_id": "node_005"},
            ],
            "quality_notes": ["Enhanced: Geräusch hinzugefügt für mehr Spannung"],
            "is_start": False,
            "is_end": False,
        },
        "node_004": {
            "id": "node_004",
            "title": "Kais Geheimnis",
            "type": "scene",
            "act": 2,
            "scene_goal": "Kai wird verdächtigt und verrät etwas. Seine Hände zittern, als er von den Sabotagen spricht.",
            "mood": "misstrauisch",
            "location": "Maschinenraum",
            "characters": ["Kai Chen"],
            "reveals": ["Kai saboteurt Systeme"],
            "choices": [
                {"id": "c7", "label": "Kai konfrontieren — direkt und hart", "next_node_id": "node_006"},
                {"id": "c8", "label": "Beweise sammeln — still weiter beobachten", "next_node_id": "node_005"},
            ],
            "quality_notes": ["Enhanced: Charaktertiefe durch Körpersprache"],
            "is_start": False,
            "is_end": False,
        },
        "node_005": {
            "id": "node_005",
            "title": "Maras Tagebuch",
            "type": "scene",
            "act": 2,
            "scene_goal": "Spieler entdeckt die Infektion. Mara hatte ihre eigenen Notizen versteckt — voller Angst und Hoffnung.",
            "mood": "schockierend",
            "location": "Medbay",
            "characters": ["Mara Singh"],
            "reveals": ["Infektion durch Mars-Organismus"],
            "choices": [
                {"id": "c9", "label": "Quarantäne auslösen — die Station riskieren", "next_node_id": "node_006"},
                {"id": "c10", "label": "Heilmittel suchen — Zeit gegen Sicherheit", "next_node_id": "node_006"},
            ],
            "quality_notes": ["Enhanced: moralisches Dilemma in Choices"],
            "is_start": False,
            "is_end": False,
        },
        "node_006": {
            "id": "node_006",
            "title": "Das Mars-Mysterium",
            "type": "scene",
            "act": 3,
            "scene_goal": "Endgültige Konfrontation mit der Wahrheit. Die Station bebt, als sich tief unter dem Boden etwas regt.",
            "mood": "enthüllend",
            "location": "Tiefste Ebene",
            "characters": ["Dr. Lena Voss", "Kai Chen", "Mara Singh"],
            "reveals": ["Die Station ist Teil einer größeren Anlage"],
            "choices": [
                {"id": "c11", "label": "Evakuierung einleiten — die Crew retten", "next_node_id": "node_007"},
                {"id": "c12", "label": "Saboteur ausliefern — Gerechtigkeit vor Sicherheit", "next_node_id": "node_008"},
                {"id": "c13", "label": "Dem Mysterium beitreten — das Unbekannte wählen", "next_node_id": "node_009"},
            ],
            "quality_notes": ["Enhanced: drei Choices mit moralischen Konflikten"],
            "is_start": False,
            "is_end": False,
        },
        "node_007": {
            "id": "node_007",
            "title": "Evakuierung",
            "type": "end",
            "act": 3,
            "scene_goal": "Die Crew entkommt, Station geht verloren. Im Rückspiegel glüht Helios wie ein sterbender Stern.",
            "mood": "bittersüß",
            "location": "Evakuierungsshuttle",
            "characters": [],
            "reveals": [],
            "choices": [],
            "quality_notes": ["Enhanced: visuelle Abschlussmetapher"],
            "is_start": False,
            "is_end": True,
        },
        "node_008": {
            "id": "node_008",
            "title": "Gerechtigkeit",
            "type": "end",
            "act": 3,
            "scene_goal": "Saboteur wird überführt. Kai wird abgeführt, während Mara still weint.",
            "mood": "gerecht",
            "location": "Brücke",
            "characters": [],
            "reveals": [],
            "choices": [],
            "quality_notes": ["Enhanced: emotionale Tiefe im Ende"],
            "is_start": False,
            "is_end": True,
        },
        "node_009": {
            "id": "node_009",
            "title": "Transformation",
            "type": "end",
            "act": 3,
            "scene_goal": "Alle werden Teil des Mysteriums. Ein warmes Licht umschließt die Crew — und sie hören auf, zu zögern.",
            "mood": "fremdartig",
            "location": "Tiefste Ebene",
            "characters": [],
            "reveals": [],
            "choices": [],
            "quality_notes": ["Enhanced: atmosphärisches Ende"],
            "is_start": False,
            "is_end": True,
        },
    },
    "start_node_id": "node_001",
}

_MOCK_ENHANCEMENT_RESULT: dict[str, Any] = {
    "graph": _MOCK_ENHANCEMENT_GRAPH,
    "changes": [
        "Alle Knoten: Atmosphäre und Sinnesbeschreibungen erweitert",
        "node_002: Choices mit subtileren Beschreibungen versehen",
        "node_003: Auditives Element (Geräusch) hinzugefügt",
        "node_004: Körpersprache für Charaktertiefe ergänzt",
        "node_005: Moralisches Dilemma in den Choices verdeutlicht",
        "node_006: Drei Choices mit unterschiedlichen moralischen Konflikten",
        "Enden: Visuelle und emotionale Abschlussmetaphern hinzugefügt",
    ],
    "summary": (
        "Story-Graph vertieft: reichere Atmosphäre, komplexere Choices, "
        "tiefere Charakterbeschreibungen und moralische Dilemmata."
    ),
}


_MOCK_REVIEW_REPAIR: dict[str, Any] = {
    "graph": {
        "nodes": {
            "node_001": {
                "id": "node_001",
                "title": "Ankunft auf Helios",
                "type": "start",
                "act": 1,
                "scene_goal": "Spieler trifft auf der Station ein.",
                "mood": "unheimlich",
                "location": "Docking Bay",
                "characters": ["Dr. Lena Voss"],
                "reveals": [],
                "choices": [
                    {"id": "c1", "label": "Zum Briefing gehen",
                     "next_node_id": "node_002"},
                    {"id": "c2", "label": "Station erkunden",
                     "next_node_id": "node_003"},
                ],
                "quality_notes": ["Repariert: mehr Atmosphäre"],
                "is_start": True,
                "is_end": False,
            },
            "node_002": {
                "id": "node_002",
                "title": "Das Briefing",
                "type": "scene",
                "act": 1,
                "scene_goal": "Mission wird erklärt, erste Hinweise auf Verschwindungen.",
                "mood": "spannend",
                "location": "Konferenzraum",
                "characters": ["Dr. Lena Voss", "Kai Chen"],
                "reveals": ["Crewmitglied X verschwunden"],
                "choices": [
                    {"id": "c3", "label": "Kai befragen",
                     "next_node_id": "node_004"},
                    {"id": "c4", "label": "Maras Tagebuch lesen",
                     "next_node_id": "node_005"},
                ],
                "quality_notes": [],
                "is_start": False,
                "is_end": False,
            },
            "node_003": {
                "id": "node_003",
                "title": "Verlassener Korridor",
                "type": "scene",
                "act": 1,
                "scene_goal": "Spieler findet Spuren eines Kampfes und hört ein seltsames Geräusch.",
                "mood": "bedrohlich",
                "location": "Korridor C-7",
                "characters": [],
                "reveals": ["Kampfspuren", "Seltsames Geräusch"],
                "choices": [
                    {"id": "c5", "label": "Zum Briefing eilen",
                     "next_node_id": "node_002"},
                    {"id": "c6", "label": "Weiter erkunden",
                     "next_node_id": "node_005"},
                ],
                "quality_notes": ["Repariert: Geräusch hinzugefügt für mehr Spannung"],
                "is_start": False,
                "is_end": False,
            },
            "node_004": {
                "id": "node_004",
                "title": "Kais Geheimnis",
                "type": "scene",
                "act": 2,
                "scene_goal": "Kai wird verdächtigt und verrät etwas.",
                "mood": "misstrauisch",
                "location": "Maschinenraum",
                "characters": ["Kai Chen"],
                "reveals": ["Kai saboteurt Systeme"],
                "choices": [
                    {"id": "c7", "label": "Kai konfrontieren",
                     "next_node_id": "node_006"},
                    {"id": "c8", "label": "Beweise sammeln",
                     "next_node_id": "node_005"},
                ],
                "quality_notes": [],
                "is_start": False,
                "is_end": False,
            },
            "node_005": {
                "id": "node_005",
                "title": "Maras Tagebuch",
                "type": "scene",
                "act": 2,
                "scene_goal": "Spieler entdeckt die Infektion.",
                "mood": "schockierend",
                "location": "Medbay",
                "characters": ["Mara Singh"],
                "reveals": ["Infektion durch Mars-Organismus"],
                "choices": [
                    {"id": "c9", "label": "Quarantäne auslösen",
                     "next_node_id": "node_006"},
                    {"id": "c10", "label": "Heilmittel suchen",
                     "next_node_id": "node_006"},
                ],
                "quality_notes": [],
                "is_start": False,
                "is_end": False,
            },
            "node_006": {
                "id": "node_006",
                "title": "Das Mars-Mysterium",
                "type": "scene",
                "act": 3,
                "scene_goal": "Endgültige Konfrontation mit der Wahrheit.",
                "mood": "enthüllend",
                "location": "Tiefste Ebene",
                "characters": ["Dr. Lena Voss", "Kai Chen", "Mara Singh"],
                "reveals": ["Die Station ist Teil einer größeren Anlage"],
                "choices": [
                    {"id": "c11", "label": "Evakuierung einleiten",
                     "next_node_id": "node_007"},
                    {"id": "c12", "label": "Saboteur ausliefern",
                     "next_node_id": "node_008"},
                    {"id": "c13", "label": "Dem Mysterium beitreten",
                     "next_node_id": "node_009"},
                ],
                "quality_notes": [],
                "is_start": False,
                "is_end": False,
            },
            "node_007": {
                "id": "node_007",
                "title": "Evakuierung",
                "type": "end",
                "act": 3,
                "scene_goal": "Die Crew entkommt, Station geht verloren.",
                "mood": "bittersüß",
                "location": "Evakuierungsshuttle",
                "characters": [],
                "reveals": [],
                "choices": [],
                "quality_notes": [],
                "is_start": False,
                "is_end": True,
            },
            "node_008": {
                "id": "node_008",
                "title": "Gerechtigkeit",
                "type": "end",
                "act": 3,
                "scene_goal": "Saboteur wird überführt.",
                "mood": "gerecht",
                "location": "Brücke",
                "characters": [],
                "reveals": [],
                "choices": [],
                "quality_notes": [],
                "is_start": False,
                "is_end": True,
            },
            "node_009": {
                "id": "node_009",
                "title": "Transformation",
                "type": "end",
                "act": 3,
                "scene_goal": "Alle werden Teil des Mysteriums.",
                "mood": "fremdartig",
                "location": "Tiefste Ebene",
                "characters": [],
                "reveals": [],
                "choices": [],
                "quality_notes": [],
                "is_start": False,
                "is_end": True,
            },
        },
        "start_node_id": "node_001",
    },
    "changes": [
        "node_003: Geräusch hinzugefügt für mehr Spannung",
    ],
    "summary": "Verbesserungen an node_003 vorgenommen.",
}


class MockLLMService(LLMService):
    """Deterministic mock provider for development without API costs."""

    provider_name = "mock"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self._call_count = 0

    async def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._call_count += 1
        start = time.monotonic()
        # Simulate small processing delay
        await asyncio.sleep(0.001)

        if json_mode:
            sp = system_prompt.lower()
            # Check specific authoring prompts first (they may contain
            # generic words like "scene" or "outline")
            if "story critic" in sp:
                # Critic review for authoring pipeline
                text = json.dumps(_MOCK_REVIEW, ensure_ascii=False)
            elif "story repair" in sp:
                # Repair for authoring pipeline
                text = json.dumps(_MOCK_REVIEW_REPAIR, ensure_ascii=False)
            elif "story enhancement" in sp:
                # Enhancement for multi-pass story deepening
                text = json.dumps(_MOCK_ENHANCEMENT_RESULT, ensure_ascii=False)
            elif "directed story graph" in sp:
                # Graph generation for authoring pipeline
                text = json.dumps(_MOCK_GRAPH, ensure_ascii=False)
            elif "outline" in sp and "story authoring" in sp:
                # Outline generation for authoring pipeline
                text = json.dumps(_MOCK_OUTLINE, ensure_ascii=False)
            elif "scene" in sp or "narrator" in sp:
                # Dynamically generate a scene based on the current node
                # parsed from the user prompt (node_id, title, scene_goal).
                # suggested_next_node is set to None so the orchestrator
                # uses the choice's next_node_id instead.
                node_id = "unknown"
                title = "Unbekannter Ort"
                scene_goal = ""
                location = ""
                mock_choices: list[dict] = []
                in_exits = False
                for line in user_prompt.splitlines():
                    stripped = line.strip()
                    if stripped.startswith("Node ID:"):
                        node_id = stripped.split(":", 1)[1].strip()
                    elif stripped.startswith("Title:"):
                        title = stripped.split(":", 1)[1].strip()
                    elif stripped.startswith("Scene goal:"):
                        scene_goal = stripped.split(":", 1)[1].strip()
                    elif stripped.startswith("Location:"):
                        location = stripped.split(":", 1)[1].strip()
                    elif "Available exits" in stripped:
                        in_exits = True
                    elif in_exits:
                        # Lines look like:  1. [id] Label → node_002
                        if stripped.startswith("=== ") or stripped == "":
                            if mock_choices:
                                in_exits = False
                            continue
                        import re as _re
                        m = _re.match(
                            r"\d+\.\s*\[(\w+)\]\s*(.+?)\s*→\s*(\w+)",
                            stripped,
                        )
                        if m:
                            mock_choices.append({
                                "id": m.group(1),
                                "label": m.group(2),
                                "next_node_id": m.group(3),
                            })

                scene_text = (
                    f"[Mock scene] Knoten: {node_id} — {title}. "
                    f"Ort: {location}. Ziel: {scene_goal}. "
                    "Die Szene entfaltet sich..."
                )
                text = json.dumps(
                    {
                        "scene_text": scene_text,
                        "choices": mock_choices,
                        "state_updates": {},
                        "suggested_next_node": None,
                    },
                    ensure_ascii=False,
                )
            else:
                text = json.dumps({"mock": True, "prompt_echo": user_prompt[:200]})
        else:
            text = f"[Mock text response] {user_prompt[:200]}"

        elapsed = time.monotonic() - start
        # Rough token estimate for mock
        input_tokens = len(system_prompt) // 4 + len(user_prompt) // 4
        output_tokens = len(text) // 4
        cost = self._estimate_cost(input_tokens, output_tokens)  # 0 by default

        response = LLMResponse(
            text=text,
            provider=self.provider_name,
            model="mock-1.0",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            elapsed_seconds=elapsed,
        )
        self._record_usage(response)
        return response


# ── OpenAI-compatible provider ───────────────────────────────────

class OpenAICompatibleProvider(LLMService):
    """Shared implementation for OpenAI-compatible Chat Completions APIs.

    Subclasses set ``provider_name``, ``_base_url``, ``_model``, and ``_headers``.
    """

    _base_url: str
    _model: str

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }

    async def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        user_prompt = self._truncate_input(user_prompt, self.settings.llm_max_input_tokens)
        max_tok = max_tokens or self.settings.llm_max_tokens
        temp = temperature if temperature is not None else self.settings.llm_temperature

        body: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tok,
            "temperature": temp,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        url = f"{self._base_url.rstrip('/')}/chat/completions"
        timeout = httpx.Timeout(self.settings.llm_timeout_seconds)

        async def _do_call() -> LLMResponse:
            start = time.monotonic()
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=body, headers=self._headers())

            elapsed = time.monotonic() - start

            if resp.status_code == 429:
                raise LLMRateLimitError(f"Rate limited: {resp.text}")
            if resp.status_code == 408:
                raise LLMTimeoutError(f"Request timed out: {resp.text}")
            if resp.status_code >= 400:
                raise LLMResponseError(
                    f"Provider error {resp.status_code}: {resp.text[:500]}"
                )

            data = resp.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content")
            # GLM-5.2 and other reasoning models may return content=None
            # when all tokens are consumed by reasoning. Fall back to
            # reasoning text if available, or empty string.
            if content is None:
                content = message.get("reasoning") or message.get("reasoning_content") or ""
            usage_data = data.get("usage", {})

            input_tokens = usage_data.get("prompt_tokens", 0)
            output_tokens = usage_data.get("completion_tokens", 0)
            cost = self._estimate_cost(input_tokens, output_tokens)

            response = LLMResponse(
                text=content,
                provider=self.provider_name,
                model=data.get("model", self._model),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                elapsed_seconds=elapsed,
                raw=data,
            )
            self._record_usage(response)
            return response

        return await self._retry_with_backoff(_do_call)


class OpenAILLMService(OpenAICompatibleProvider):
    """OpenAI Chat Completions provider."""

    provider_name = "openai"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self._base_url = self.settings.openai_base_url
        self._model = self.settings.openai_model


# ── Azure OpenAI provider ────────────────────────────────────────

class AzureOpenAILLMService(OpenAICompatibleProvider):
    """Azure OpenAI Service provider.

    Uses the Azure-specific endpoint format:
      POST {endpoint}/openai/deployments/{deployment}/chat/completions?api-version={version}
    """

    provider_name = "azure_openai"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self._model = self.settings.azure_openai_deployment or self.settings.openai_model

    def _headers(self) -> dict[str, str]:
        return {
            "api-key": self.settings.azure_openai_api_key,
            "Content-Type": "application/json",
        }

    async def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        user_prompt = self._truncate_input(user_prompt, self.settings.llm_max_input_tokens)
        max_tok = max_tokens or self.settings.llm_max_tokens
        temp = temperature if temperature is not None else self.settings.llm_temperature

        body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tok,
            "temperature": temp,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        endpoint = self.settings.azure_openai_endpoint.rstrip("/")
        deployment = self.settings.azure_openai_deployment
        api_version = self.settings.azure_openai_api_version
        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        timeout = httpx.Timeout(self.settings.llm_timeout_seconds)

        async def _do_call() -> LLMResponse:
            start = time.monotonic()
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=body, headers=self._headers())

            elapsed = time.monotonic() - start

            if resp.status_code == 429:
                raise LLMRateLimitError(f"Rate limited: {resp.text}")
            if resp.status_code == 408:
                raise LLMTimeoutError(f"Request timed out: {resp.text}")
            if resp.status_code >= 400:
                raise LLMResponseError(
                    f"Azure error {resp.status_code}: {resp.text[:500]}"
                )

            data = resp.json()
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            usage_data = data.get("usage", {})

            input_tokens = usage_data.get("prompt_tokens", 0)
            output_tokens = usage_data.get("completion_tokens", 0)
            cost = self._estimate_cost(input_tokens, output_tokens)

            response = LLMResponse(
                text=content,
                provider=self.provider_name,
                model=deployment or "azure-openai",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                elapsed_seconds=elapsed,
                raw=data,
            )
            self._record_usage(response)
            return response

        return await self._retry_with_backoff(_do_call)


# ── Ollama provider ─────────────────────────────────────────────

class OllamaLLMService(LLMService):
    """Local Ollama REST API provider (no API key needed)."""

    provider_name = "ollama"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self._base_url = self.settings.ollama_base_url
        self._model = self.settings.ollama_model

    async def _complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        user_prompt = self._truncate_input(user_prompt, self.settings.llm_max_input_tokens)
        max_tok = max_tokens or self.settings.llm_max_tokens
        temp = temperature if temperature is not None else self.settings.llm_temperature

        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        body: dict[str, Any] = {
            "model": self._model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "num_predict": max_tok,
                "temperature": temp,
            },
        }
        if json_mode:
            body["format"] = "json"

        url = f"{self._base_url.rstrip('/')}/api/generate"
        timeout = httpx.Timeout(self.settings.llm_timeout_seconds)

        async def _do_call() -> LLMResponse:
            start = time.monotonic()
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=body)

            elapsed = time.monotonic() - start

            if resp.status_code == 429:
                raise LLMRateLimitError(f"Rate limited: {resp.text}")
            if resp.status_code == 408:
                raise LLMTimeoutError(f"Request timed out: {resp.text}")
            if resp.status_code >= 400:
                raise LLMResponseError(
                    f"Ollama error {resp.status_code}: {resp.text[:500]}"
                )

            data = resp.json()
            content = data.get("response", "")

            # Ollama provides eval_count as output token estimate
            input_tokens = data.get("prompt_eval_count", 0) or len(full_prompt) // 4
            output_tokens = data.get("eval_count", 0) or len(content) // 4
            cost = self._estimate_cost(input_tokens, output_tokens)

            response = LLMResponse(
                text=content,
                provider=self.provider_name,
                model=data.get("model", self._model),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                elapsed_seconds=elapsed,
                raw=data,
            )
            self._record_usage(response)
            return response

        return await self._retry_with_backoff(_do_call)


# ── Factory ──────────────────────────────────────────────────────

def get_llm_service(settings: Settings | None = None) -> LLMService:
    """Factory: return the configured LLM provider (Spec §5.6).

    Reads ``llm_provider`` from settings. Supported values:
    ``mock``, ``openai``, ``azure_openai``, ``ollama``.
    """
    s = settings or get_settings()
    provider = s.llm_provider
    if provider == "mock":
        return MockLLMService(s)
    elif provider == "openai":
        if not s.openai_api_key:
            raise LLMError("OpenAI provider requires OPENAI_API_KEY to be set")
        return OpenAILLMService(s)
    elif provider == "azure_openai":
        if not s.azure_openai_api_key:
            raise LLMError("Azure OpenAI provider requires AZURE_OPENAI_API_KEY to be set")
        if not s.azure_openai_endpoint:
            raise LLMError("Azure OpenAI provider requires AZURE_OPENAI_ENDPOINT to be set")
        if not s.azure_openai_deployment:
            raise LLMError("Azure OpenAI provider requires AZURE_OPENAI_DEPLOYMENT to be set")
        return AzureOpenAILLMService(s)
    elif provider == "ollama":
        return OllamaLLMService(s)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
