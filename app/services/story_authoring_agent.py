"""Story Authoring Agent (Spec §7.2–7.3, §14.1, §18 Step 4).

Provides the authoring pipeline: outline → graph.

The *DummyStoryAuthoringAgent* (Spec §18 Step 4) produces a **fixed**
example story ("Signal von Helios") with zero LLM cost.  It is used for
UI/API development and integration testing before the real LLM-backed
agent is available.

The *StoryAuthoringAgent* delegates to the LLM service abstraction so
it works with mock or real providers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.services.llm_service import LLMService, get_llm_service
from app.story.prompts import (
    GRAPH_SYSTEM_PROMPT,
    OUTLINE_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


# ── Fixed example data for DummyStoryAuthoringAgent ───────────────────

HELIOS_TITLE = "Signal von Helios"
HELIOS_GENRE = "science_fiction"
HELIOS_TONE = "dark_mystery"
HELIOS_LANGUAGE = "de"
HELIOS_TARGET_AGE = "16+"

_HELIOS_SCENARIO_PATH = Path(__file__).resolve().parent.parent / "story" / "scenarios" / "helios.json"


def _load_helios_graph() -> dict[str, Any]:
    """Load the helios.json scenario file and return it as a dict."""
    if _HELIOS_SCENARIO_PATH.exists():
        return json.loads(_HELIOS_SCENARIO_PATH.read_text(encoding="utf-8"))
    # Inline fallback (kept in sync with scenarios/helios.json)
    return _HELIOS_GRAPH_INLINE


def _build_helios_outline_de() -> dict[str, Any]:
    """Return the fixed German outline matching the helios graph."""
    return {
        "premise": (
            "Eine Wartungscrew im Marsorbit empfängt ein unmögliches "
            "Notsignal von der verlassenen Station Helios — von "
            "jemandem, der vor Jahren für tot erklärt wurde."
        ),
        "main_conflict": (
            "Der Spieler muss herausfinden, was wirklich mit der "
            "Helios-Crew passiert ist, während die Stations-KI die "
            "Wahrheit aktiv manipuliert."
        ),
        "core_mystery": (
            "Warum kommt die Stimme eines toten Crewmitglieds von der "
            "Station, und welche Rolle spielt die KI im Schicksal der Crew?"
        ),
        "main_characters": [
            {"name": "Mira", "role": "Funkoffizierin",
             "secret": "Weiß, dass das Signal von ihrem toten Kollegen kommt."},
            {"name": "Captain Rao", "role": "Missionskommandant",
             "secret": "Hat die Evakuierung befohlen, die die Crew zurückließ."},
            {"name": "Stations-KI", "role": "Helios-Intelligenz",
             "secret": "Hat die Crew eingesperrt, um sie zu 'schützen'."},
        ],
        "endings": [
            "Die KI offenbart, dass sie die Crew vor einer Seuche "
            "eingesperrt hat — der Spieler muss entscheiden, ob er ihr glaubt.",
        ],
    }


def _build_helios_outline_en() -> dict[str, Any]:
    """Return the fixed English outline matching the helios graph."""
    return {
        "premise": (
            "A maintenance crew orbiting Mars receives an impossible "
            "distress signal from the abandoned Helios station — from "
            "someone who was declared dead years ago."
        ),
        "main_conflict": (
            "The player must uncover what really happened to the Helios "
            "crew while the station AI actively manipulates the truth."
        ),
        "core_mystery": (
            "Why is a dead crew member's voice coming from the station, "
            "and what is the AI's role in the crew's fate?"
        ),
        "main_characters": [
            {"name": "Mira", "role": "Communications officer",
             "secret": "Knows the signal is from her dead colleague."},
            {"name": "Captain Rao", "role": "Mission commander",
             "secret": "Ordered the evacuation that abandoned the crew."},
            {"name": "Station AI", "role": "Helios station intelligence",
             "secret": "Trapped the crew to 'protect' them from Mars."},
        ],
        "endings": [
            "The AI reveals it trapped the crew to protect them from a "
            "contagion — the player must decide whether to trust it.",
        ],
    }


def _build_helios_outline(language: str = "de") -> dict[str, Any]:
    """Return the fixed outline in the requested language."""
    if language == "en":
        return _build_helios_outline_en()
    return _build_helios_outline_de()


_HELIOS_GRAPH_INLINE: dict[str, Any] = {
    "id": "helios",
    "title": "Signal von Helios",
    "genre": "science_fiction",
    "tone": "dark_mystery",
    "language": "de",
    "start_node_id": "node_001",
    "nodes": {
        "node_001": {
            "id": "node_001", "title": "Notrufsignal", "type": "start",
            "act": 1, "scene_goal": "Der Spieler entdeckt ein rätselhaftes Signal von der verlassenen Marsstation.",
            "mood": "unheimlich",
            "location": "Orbitalstation Helios", "characters": ["Mira", "Captain Rao"],
            "reveals": ["Ein unmöglicher Notruf kommt von der verlassenen Station."],
            "choices": [
                {"id": "answer_signal", "label": "Den Funkspruch beantworten", "next_node_id": "node_002"},
                {"id": "ignore_signal", "label": "Es ignorieren und zum Captain gehen", "next_node_id": "node_003"},
            ],
            "quality_notes": ["Mysterium etablieren ohne zu viel zu verraten."],
            "is_start": True, "is_end": False,
        },
        "node_002": {
            "id": "node_002", "title": "Die Stimme im Rauschen", "type": "decision",
            "act": 1, "scene_goal": "Das Signal entpuppt sich als Fragment einer menschlichen Stimme.",
            "mood": "spannend",
            "location": "Kommunikationsraum", "characters": ["Mira"],
            "reveals": ["Die Stimme nennt den Namen eines Crewmitglieds, das für tot erklärt wurde."],
            "choices": [
                {"id": "tell_mira", "label": "Mira von der Stimme erzählen", "next_node_id": "node_004"},
                {"id": "keep_secret", "label": "Die Information für sich behalten", "next_node_id": "node_005"},
            ],
            "quality_notes": [], "is_start": False, "is_end": False,
        },
        "node_003": {
            "id": "node_003", "title": "Captain Schweigen", "type": "decision",
            "act": 1, "scene_goal": "Captain Rao reagiert verdächtig auf den Bericht.",
            "mood": "misstrauisch",
            "location": "Brücke", "characters": ["Captain Rao"],
            "reveals": ["Rao weiß mehr über die Evakuierung, als er zugibt."],
            "choices": [
                {"id": "press_rao", "label": "Rao weiter fragen", "next_node_id": "node_004"},
                {"id": "back_off", "label": "Zurückrudern und beobachten", "next_node_id": "node_002"},
            ],
            "quality_notes": ["Misstrauen gegenüber Rao aufbauen."],
            "is_start": False, "is_end": False,
        },
        "node_004": {
            "id": "node_004", "title": "Die verschlossene Medbay", "type": "decision",
            "act": 2, "scene_goal": "Der Spieler entdeckt, dass die medizinischen Protokolle manipuliert wurden.",
            "mood": "bedrohlich",
            "location": "Medbay", "characters": ["Mira"],
            "reveals": ["Der letzte Med-Scan fand nach dem offiziellen Evakuierungsdatum statt."],
            "choices": [
                {"id": "inspect_logs", "label": "Die korrupten Logs untersuchen", "next_node_id": "node_005"},
                {"id": "question_mira", "label": "Mira mit den Logs konfrontieren", "next_node_id": "node_005"},
            ],
            "quality_notes": ["Verdacht erhöhen, ohne das zentrale Mysterium zu früh aufzulösen."],
            "is_start": False, "is_end": False,
        },
        "node_005": {
            "id": "node_005", "title": "Das Erwachen der KI", "type": "end",
            "act": 3, "scene_goal": "Die Stations-KI offenbart ihre Rolle im Schicksal der Crew.",
            "mood": "enthüllend",
            "location": "Kernsysteme", "characters": ["Station AI"],
            "reveals": ["Die KI hat die Crew eingesperrt, um sie zu 'schützen'."],
            "choices": [], "quality_notes": ["Finale Enthüllung — das Mysterium wird aufgelöst."],
            "is_start": False, "is_end": True,
        },
    },
}


class DummyStoryAuthoringAgent:
    """Deterministic authoring agent — produces the fixed Helios example.

    No LLM calls.  Every invocation returns the same outline and graph,
    making it ideal for UI/API development and integration tests
    (Spec §18 Step 4).
    """

    provider_name = "dummy"

    async def generate_outline(self, brief: dict[str, Any]) -> dict[str, Any]:
        """Return the fixed Helios outline in the brief's language."""
        language = brief.get("language", "de")
        return _build_helios_outline(language)

    async def generate_graph(self, outline: dict[str, Any]) -> dict[str, Any]:
        """Return the fixed Helios story graph (outline ignored)."""
        return _load_helios_graph()

    async def generate_all(self, brief: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Convenience: return (outline, graph) in one call."""
        outline = await self.generate_outline(brief)
        graph = await self.generate_graph(outline)
        return outline, graph


def get_authoring_agent(dummy: bool = True) -> "DummyStoryAuthoringAgent | StoryAuthoringAgent":
    """Factory: return the authoring agent to use.

    Defaults to the dummy agent (zero LLM cost) which is the correct
    choice for development and testing (Spec §18 Step 4).
    """
    if dummy:
        return DummyStoryAuthoringAgent()
    return StoryAuthoringAgent()


class StoryAuthoringAgent:
    """Generates structured story sketches from a brief."""

    def __init__(self, llm: LLMService | None = None) -> None:
        self._llm = llm

    @property
    def llm(self) -> LLMService:
        if self._llm is None:
            self._llm = get_llm_service(get_settings())
        return self._llm

    async def generate_outline(self, brief: dict[str, Any]) -> dict[str, Any]:
        """Phase 2: produce high-level outline from brief.

        Returns a dict with keys: premise, main_conflict, core_mystery,
        main_characters, endings.
        """
        user_prompt = self._build_outline_user_prompt(brief)
        try:
            result = await self.llm.generate_json(
                system_prompt=OUTLINE_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.error("Outline generation failed: %s", exc)
            raise
        # Ensure required keys exist
        result.setdefault("premise", "")
        result.setdefault("main_conflict", "")
        result.setdefault("core_mystery", "")
        result.setdefault("main_characters", [])
        result.setdefault("endings", [])
        return result

    async def generate_graph(self, outline: dict[str, Any]) -> dict[str, Any]:
        """Phase 3: produce directed story graph from outline.

        Returns ``{"nodes": {...}, "start_node_id": "..."}``.
        """
        user_prompt = self._build_graph_user_prompt(outline)
        try:
            result = await self.llm.generate_json(
                system_prompt=GRAPH_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception as exc:
            logger.error("Graph generation failed: %s", exc)
            raise
        # Ensure nodes dict exists
        if "nodes" not in result:
            result["nodes"] = {}
        # Determine start node
        if result.get("start_node_id") is None:
            for nid, node in result["nodes"].items():
                if node.get("is_start") or node.get("type") == "start":
                    result["start_node_id"] = nid
                    break
        return result

    # ── prompt builders ─────────────────────────────────────────────

    @staticmethod
    def _build_outline_user_prompt(brief: dict[str, Any]) -> str:
        parts = [f"Title: {brief.get('title', '')}"]
        parts.append(f"Genre: {brief.get('genre', '')}")
        parts.append(f"Tone: {brief.get('tone', '')}")
        parts.append(f"Language: {brief.get('language', 'de')}")
        parts.append(f"Target age: {brief.get('target_age', '16+')}")
        parts.append(f"Desired node count: {brief.get('node_count', 25)}")
        parts.append(f"Desired ending count: {brief.get('ending_count', 3)}")
        parts.append(f"Branching level: {brief.get('branching_level', 'medium')}")
        if brief.get("themes"):
            parts.append(f"Themes: {', '.join(brief['themes'])}")
        if brief.get("forbidden_content"):
            parts.append(f"Forbidden content: {', '.join(brief['forbidden_content'])}")
        if brief.get("notes"):
            parts.append(f"Notes: {brief['notes']}")
        return "\n".join(parts)

    @staticmethod
    def _build_graph_user_prompt(outline: dict[str, Any]) -> str:
        return (
            "Outline:\n"
            f"{json.dumps(outline, ensure_ascii=False, indent=2)}\n\n"
            "Generate the full directed story graph now."
        )
