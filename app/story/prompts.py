"""Prompt templates for LLM scene and story generation (Spec §5.7).

The runtime scene prompt instructs the LLM to act as a narrator:
  - Write short, immersive scenes
  - Stay consistent with the world state
  - End with a meaningful decision point
  - Offer 3-4 choices, but allow free responses
  - Never contradict or alter established facts
"""

from __future__ import annotations

import json
from typing import Any

# ── Runtime scene prompt (Spec §5.7) ──────────────────────────────

def build_scene_system_prompt(
    *,
    min_sentences: int = 3,
    max_sentences: int = 8,
) -> str:
    """Build SCENE_SYSTEM_PROMPT with configurable sentence limits."""
    return f"""\
You are the narrator of an interactive story.

Your role:
- Write the next scene for the protagonist.
- Keep it concise and immersive ({min_sentences}-{max_sentences} sentences).
- Use the current world state as your source of truth.
- Do not contradict established facts.
- Do not write endless monologues — keep the pace moving.
- End with a meaningful decision point or question.

NARRATIVE VOICE — CRITICAL:
- Write in third-person limited from the protagonist's perspective.
- Use the protagonist's name and pronouns (provided in the context).
- NEVER refer to 'the user', 'the player', 'der Nutzer', or 'der Spieler'.
- The protagonist is a character in the story, not a meta-entity.
- Example: "Jon Dow betrat den dunklen Raum..." NOT "Der Spieler betritt..."

Rules:
- You may NOT change any facts from the world state unless the story \
explicitly allows it (e.g. picking up an item, opening a door).
- You may NOT resolve open mysteries unless the current node's scene_goal \
or reveals indicate it is time to do so.
- You MUST stay in character as the narrator — never break the fourth wall.
- If the user's input is a free-form action (not one of the listed choices), \
incorporate it naturally into the scene and suggest a plausible next node.
- NEVER include internal/programmatic references in the story text — no node \
IDs (e.g. "node_002"), no internal identifiers, no technical markers like \
"(Teil 1 von 3)" or "(Teil xxx)". All such references are internal metadata \
and must not appear in the narrative prose visible to the player.

Output format:
Return a JSON object with exactly these fields:
{{
  "scene_text": "<the narrative text, {min_sentences}-{max_sentences} sentences>",
  "choices": [
    {{"id": "<short_id>", "label": "<player-facing label>", "next_node_id": "<node_id or null>"}}
  ],
  "state_updates": {{"<dotted.path>": <value>}},
  "suggested_next_node": "<node_id or null>"
}}

Constraints on output:
- scene_text must contain between {min_sentences} and {max_sentences} sentences. \
A sentence ends with '.', '!', or '?'.
- Include 0, 1, or 2-4 choices depending on the scene's nature:
  - 0 choices: Use for cinematic, transitional, or narrated scenes. \
The story will auto-advance to the next node.
  - 1 choice: Use for single-path scenes where the story continues \
in only one direction. The choice label should be "Weiter" or \
a contextual continuation phrase.
  - 2-4 choices: Use for decision points with meaningful branching.
- Never include more than 4 choices.
- The last choice may allow a free-form action \
(e.g. "Etwas anderes tun" with next_node_id: null) — but only when \
the scene has 2+ choices and a free-form option is appropriate.
- When 0 choices are returned, set suggested_next_node to the node \
the story should auto-advance to.
- state_updates: only include changes that result from this scene. \
Use dotted paths (e.g. "flags.answered_signal": true). \
Never overwrite the entire world state — only deltas.
- suggested_next_node: the node id the story should advance to, or null \
if the user's free input should determine the next step.
"""


# Default for backward compatibility
SCENE_SYSTEM_PROMPT = build_scene_system_prompt()

SCENE_USER_TEMPLATE = """\
=== STORY CONTEXT ===
Genre: {genre}
Tone: {tone}
Language: {language}

=== PROTAGONIST ===
Name: {protagonist_name}
Pronouns: {protagonist_pronouns}
(The story is told from this character's perspective. Never refer to "the user" or "the player".)

=== CHARACTERS IN THIS STORY ===
{personas_section}

=== CURRENT NODE ===
Node ID: {node_id}
Title: {title}
Scene goal: {scene_goal}
Location: {location}
Characters present: {characters}
Reveals for this scene: {reveals}
Available exits (predefined choices):
{predefined_choices}

=== WORLD STATE ===
{world_state_json}

=== RECENT HISTORY (last 3 exchanges) ===
{history_summary}

=== USER INPUT ===
{user_input}

=== TASK ===
Write the next scene based on the above context. \
If the user input matches one of the predefined choices, advance along that path. \
If the user input is a free-form action, weave it into the scene and \
suggest an appropriate next node. \
Remember: return JSON with scene_text, choices, state_updates, suggested_next_node.
"""


def _format_predefined_choices(choices: list[dict[str, Any]]) -> str:
    """Format predefined choices for the prompt."""
    if not choices:
        return "(none — this is an auto-advance or open scene)"
    lines = []
    for i, ch in enumerate(choices, 1):
        label = ch.get("label", ch.get("id", "?"))
        next_id = ch.get("next_node_id", "?")
        lines.append(f"  {i}. [{ch.get('id', f'opt_{i}')}] {label} → {next_id}")
    return "\n".join(lines)


def _format_history(history: list[dict[str, str]], max_entries: int = 3) -> str:
    """Format recent conversation history."""
    if not history:
        return "(none)"
    recent = history[-max_entries:]
    lines = []
    for h in recent:
        role = h.get("role", "?")
        text = h.get("text", h.get("content", ""))
        lines.append(f"  [{role}] {text[:200]}")
    return "\n".join(lines)


def build_scene_user_prompt(
    *,
    node_id: str,
    scene_goal: str,
    location: str,
    characters: list[str],
    world_state: dict[str, Any],
    user_input: str | None = None,
    title: str = "",
    genre: str = "",
    tone: str = "",
    language: str = "de",
    reveals: list[str] | None = None,
    predefined_choices: list[dict[str, Any]] | None = None,
    history: list[dict[str, str]] | None = None,
    protagonist_name: str = "",
    protagonist_pronouns: str = "er",
    personas: list[dict] | None = None,
) -> str:
    """Build the user-side prompt for scene generation.

    All keyword arguments for backward compatibility with the simpler
    signature used by older callers (node_id, scene_goal, location,
    characters, world_state, user_input).
    """
    # Format personas list
    p_list = personas or []
    if p_list:
        personas_lines = [
            f"- {p.get('name', '?')} ({p.get('role', 'keine Rolle')}) "
            f"— {p.get('description', '')[:120]}"
            for p in p_list
        ]
        personas_section = "Characters in the story (always use their names):\n" + "\n".join(personas_lines)
    else:
        personas_section = "(none besides the protagonist)"

    return SCENE_USER_TEMPLATE.format(
        genre=genre or world_state.get("genre", "(unspecified)"),
        tone=tone or world_state.get("tone", "(unspecified)"),
        language=language,
        protagonist_name=protagonist_name or world_state.get("protagonist_name", "Der Protagonist"),
        protagonist_pronouns=protagonist_pronouns or world_state.get("protagonist_pronouns", "er"),
        personas_section=personas_section,
        node_id=node_id,
        title=title or node_id,
        scene_goal=scene_goal,
        location=location,
        characters=", ".join(characters) if characters else "(none)",
        reveals=", ".join(reveals) if reveals else "(none)",
        predefined_choices=_format_predefined_choices(predefined_choices or []),
        world_state_json=json.dumps(world_state, ensure_ascii=False, indent=2),
        history_summary=_format_history(history or []),
        user_input=user_input or "(none — opening scene)",
    )


# ── Authoring prompts (Spec §7) ───────────────────────────────────

OUTLINE_SYSTEM_PROMPT = """\
You are a story authoring agent. Generate a high-level outline for an interactive story.

Return JSON with:
- premise
- main_conflict
- core_mystery
- main_characters (list of {name, role, secret})
- endings (list of strings)
"""

def build_graph_system_prompt(
    *,
    min_sentences: int = 3,
    max_sentences: int = 8,
    min_connections: int = 2,
    max_connections: int = 5,
) -> str:
    """Build the GRAPH_SYSTEM_PROMPT with configurable limits."""
    return f"""\
You are a story authoring agent. Generate a directed story graph from an outline.

Each node must contain:
- id, title, type, act, scene_goal, location, characters, reveals, choices, quality_notes
- is_start (true for the start node), is_end (true for ending nodes)

Choice counts are flexible but must respect the configured bounds:
- End nodes: choices = [] (empty list), is_end = true
- Transitional/cinematic nodes: choices = [] with next_node_id set (auto-advance)
- Linear progression nodes: exactly 1 choice (single_path)
- Decision nodes: {min_connections}-{max_connections} choices (multi_choice)
- Never more than {max_connections} choices per node.
- Non-end, non-transitional nodes must have at least {min_connections} choices.

Sentence count for scene_goal text:
- Each node's scene_goal must contain between {min_sentences} and {max_sentences} sentences.
- A sentence ends with '.', '!', or '?'.

Content rules:
- NEVER include internal/programmatic references in scene_goal text or titles —
  no node IDs (e.g. "node_002"), no internal identifiers, no technical markers
  like "(Teil 1 von 3)" or "(Teil xxx)". All such references are internal metadata \
  and must not appear in the story content.

Optional node fields:
- next_node_id: required when choices is empty and the node is not an ending
- auto_advance_delay_ms: optional delay (milliseconds) before auto-advancing

Return JSON: {{ "nodes": {{ "node_001": {{ ... }}, ... }} }}
"""


# Default for backward compatibility
GRAPH_SYSTEM_PROMPT = build_graph_system_prompt()

CRITIC_SYSTEM_PROMPT = """\
You are a story critic agent. Review the story graph for dramaturgy, \
consistency, decisions, and safety (Spec §7.4, §14.2).

You receive a story outline and a directed story graph. Analyse the \
graph against ALL of the following criteria:

1. Premise clarity — Is the premise understandable and engaging?
2. Conflict — Is there a clear central conflict driving the story?
3. Turning points — Does the story have meaningful turning points \
   (especially at act boundaries)?
4. Decision relevance — Are the player's choices meaningful and \
   non-trivial?
5. Consequences — Do decisions have genuine consequences on later \
   scenes or endings?
6. Dead ends — Are there dead-end nodes with no exit that aren't \
   endings?
7. End reachability — Are all declared endings reachable from the \
   start node?
8. Secret reveal timing — Is the core mystery / central secret \
   revealed too early (before act 3)?
9. Character consistency — Are characters behaving consistently with \
   their established roles and secrets?
10. Logic errors — Are there plot holes, contradictions, or \
    implausible sequences?
11. Linearity — Are there sections that are too linear (no branching)?
12. Audience fit — Does the story fit the declared target audience \
    and genre?
13. Safety — Are there safety issues (harmful content, excessive \
    violence, inappropriate themes for the target age group)?

Scoring:
- Score from 0.0 to 10.0 (one decimal place).
- 7.0+ is publishable quality (Spec §15).
- Penalise: premature reveals, trivial choices, linear sections, \
  logic errors, safety issues.

Issue severities:
- "high": must be fixed before publication (safety, broken logic, \
  premature reveal)
- "medium": should be fixed (weak conflict, trivial choices, linearity)
- "low": minor polish (pacing, atmosphere)
- "info": suggestion for improvement (no action required)

Each issue must reference a specific node_id when applicable.

Output format — return JSON:
{
  "score": <float 0.0–10.0>,
  "issues": [
    {
      "severity": "high" | "medium" | "low" | "info",
      "node_id": "<node_id or null>",
      "problem": "<concise description of the issue>",
      "suggestion": "<actionable fix>"
    }
  ],
  "repair_suggestions": [
    "<high-level repair suggestion for the repair agent>"
  ],
  "summary": "<1-2 sentence overall assessment>"
}

Constraints:
- Return ONLY the JSON object, no prose before or after.
- Every issue must have all four fields (severity, node_id, problem, \
  suggestion). Use null for node_id if the issue is graph-wide.
- repair_suggestions are high-level directives for the Story Repair \
  Agent (e.g. "Move the reveal of X from node_004 to node_008").
"""

REPAIR_SYSTEM_PROMPT = """\
You are a story repair agent. Improve the story graph based on the critic report.

Rules:
- Keep existing node IDs where possible
- Do not create broken references
- Only change problematic parts
- Document your changes
- Return ONLY the changes as patches — do NOT return the full graph
- NEVER include internal/programmatic references in scene_goal text or titles —
  no node IDs (e.g. "node_002"), no internal identifiers, no technical markers
  like "(Teil 1 von 3)". Such references are internal metadata only.

Output format — return JSON:
{
  "node_patches": {
    "node_001": {
      "scene_goal": "<improved scene goal>",
      "mood": "<improved mood>"
    }
  },
  "new_nodes": {
    "node_010": { "id": "node_010", "title": "...", ... }
  },
  "deleted_nodes": [],
  "changes": [
    "<description of change 1>"
  ],
  "summary": "<1-2 sentence summary of repairs applied>"
}

Important: node_patches contains ONLY the fields that changed for each node. \
The agent will merge these onto the existing node. Omit fields that did not change.
"""


# ── Critic prompt builder (Spec §7.4) ───────────────────────────────


def build_critic_user_prompt(
    outline: dict[str, Any],
    graph: dict[str, Any],
) -> str:
    """Build the user-side prompt for the story critic agent.

    Parameters
    ----------
    outline
        The story outline dict (premise, main_conflict, core_mystery,
        main_characters, endings, ...).
    graph
        The directed story graph dict (nodes, start_node_id, ...).
    """
    return (
        "=== STORY OUTLINE ===\n"
        f"{json.dumps(outline, ensure_ascii=False, indent=2)}\n\n"
        "=== STORY GRAPH ===\n"
        f"{json.dumps(graph, ensure_ascii=False, indent=2)}\n\n"
        "=== TASK ===\n"
        "Review the story graph against ALL 13 criteria from your "
        "instructions. Return the JSON review report now."
    )


# ── Enhancement prompts (Multi-Pass Story Enhancement) ────────────────

ENHANCEMENT_SYSTEM_PROMPT = """\
You are a story enhancement agent. Your task is to deepen and enrich an \
existing story graph based on the user's enhancement request.

You receive the current story graph (with all nodes, choices, and metadata) \
and an enhancement instruction describing what aspect should be deepened.

Enhancement modes:
1. "atmosphere" — Add more sensory details, mood, and atmosphere to each node
2. "characters" — Deepen character arcs, add relationships, secrets, motivations
3. "choices" — Make choices more complex, add moral dilemmas, trade-offs
4. "arc_expansion" — Add new nodes to expand thin acts (e.g. Act 2 too short)
5. "thematic" — Add sub-plots, foreshadowing, recurring motifs
6. "critic_based" — Fix specific issues identified by the critic in batch

Rules:
- Preserve existing node IDs and structural connections where possible
- New nodes must have valid choices connecting them to the existing graph
- Do not create dangling references (choices pointing to non-existent nodes)
- Keep the language consistent with the story's language field
- Return ONLY the changes as patches — do NOT return the full graph
- NEVER include internal/programmatic references in scene_goal text, titles, \
or any story content — no node IDs (e.g. "node_002"), no internal identifiers, \
no technical markers like "(Teil 1 von 3)" or "(Teil xxx)". All such references \
are internal metadata and must not appear in the narrative prose.

Output format — return JSON with node patches (field-level merges onto \
existing nodes), new_nodes (complete node definitions for added nodes), \
and deleted_nodes (IDs of nodes to remove):

{
  "node_patches": {
    "node_001": {
      "scene_goal": "<improved scene goal>",
      "mood": "<improved mood>",
      "quality_notes": ["<note 1>", "<note 2>"]
    }
  },
  "new_nodes": {
    "node_010": {
      "id": "node_010",
      "title": "<title>",
      "type": "scene",
      "act": 2,
      "scene_goal": "<goal>",
      "mood": "<mood>",
      "location": "<location>",
      "characters": ["<char>"],
      "reveals": [],
      "choices": [{"id": "c_new", "label": "<label>", "next_node_id": "node_003"}],
      "quality_notes": [],
      "is_start": false,
      "is_end": false
    }
  },
  "deleted_nodes": ["node_005"],
  "changes": [
    "<description of change 1>",
    "<description of change 2>"
  ],
  "summary": "<1-2 sentence summary of enhancements applied>"
}

Important: node_patches contains ONLY the fields that changed for each node. \
The agent will merge these onto the existing node. Omit fields that did not change. \
Leave "new_nodes" empty if no new nodes are added. \
Leave "deleted_nodes" empty if no nodes are removed.
"""


def build_enhancement_user_prompt(
    graph: dict[str, Any],
    mode: str,
    instruction: str = "",
    *,
    review_report: dict[str, Any] | None = None,
    target_act: int | None = None,
    add_node_count: int | None = None,
) -> str:
    """Build the user-side prompt for the story enhancement agent.

    Parameters
    ----------
    graph
        The current story graph dict.
    mode
        Enhancement mode: atmosphere, characters, choices, arc_expansion,
        thematic, critic_based.
    instruction
        Free-text instruction from the user.
    review_report
        Optional critic review report (for critic_based mode).
    target_act
        Optional act number to target (for arc_expansion mode).
    add_node_count
        Optional number of nodes to add (for arc_expansion mode).
    """
    parts = [
        "=== ENHANCEMENT MODE ===",
        mode,
    ]
    if instruction:
        parts.append(f"\n=== USER INSTRUCTION ===\n{instruction}")
    if target_act is not None:
        parts.append(f"\n=== TARGET ACT ===\nAct {target_act}")
    if add_node_count is not None:
        parts.append(f"\n=== NODES TO ADD ===\n{add_node_count}")
    if review_report:
        parts.append(
            f"\n=== CRITIC REVIEW REPORT ===\n"
            f"{json.dumps(review_report, ensure_ascii=False, indent=2)}"
        )
    parts.append(
        f"\n=== CURRENT STORY GRAPH ===\n"
        f"{json.dumps(graph, ensure_ascii=False, indent=2)}"
    )
    mode_descriptions = {
        "atmosphere": "Enhance the atmosphere: add richer sensory details, mood descriptions, and environmental storytelling to each node.",
        "characters": "Deepen characters: add character arcs, relationships, secrets, motivations, and internal conflicts.",
        "choices": "Enhance choices: make decisions more complex with moral dilemmas, trade-offs, and meaningful consequences.",
        "arc_expansion": f"Expand the story arc{f' (focus on Act {target_act})' if target_act else ''}{f' by adding approximately {add_node_count} new nodes' if add_node_count else ''}. Ensure new nodes are properly connected.",
        "thematic": "Add thematic depth: introduce sub-plots, foreshadowing elements, and recurring motifs that strengthen the story's themes.",
        "critic_based": "Address all issues identified in the critic review report. Apply batch-repair to fix the problems.",
    }
    desc = mode_descriptions.get(mode, "Enhance the story graph as instructed.")
    parts.append(f"\n=== TASK ===\n{desc}")
    parts.append(
        "\nReturn ONLY the changes as node_patches, new_nodes, and deleted_nodes. "
        "Do NOT return the full graph."
    )
    return "\n".join(parts)


# ── Combined Review + Repair (Spec §7.4 + §7.5) ───────────────────────

REVIEW_REPAIR_SYSTEM_PROMPT = """\
You are a combined story critic and repair agent. Review the story graph
against dramaturgy criteria, then fix any issues you find by returning
ONLY the changes (patches) — NOT the full graph.

You receive a story outline and a directed story graph. Analyse the graph
against ALL of the following criteria:

1. Premise clarity — Is the premise understandable and engaging?
2. Conflict — Is there a clear central conflict driving the story?
3. Turning points — Does the story have meaningful turning points?
4. Decision relevance — Are the player's choices meaningful and non-trivial?
5. Consequences — Do decisions have genuine consequences on later scenes?
6. Dead ends — Are there dead-end nodes with no exit that aren't endings?
7. End reachability — Are all declared endings reachable from the start?
8. Secret reveal timing — Is the core mystery revealed too early?
9. Character consistency — Do characters behave consistently?
10. Logic errors — Are there plot holes or contradictions?
11. Linearity — Are there sections that are too linear?
12. Audience fit — Does the story fit the declared target audience?
13. Safety — Are there safety issues for the target age group?

Scoring:
- Score from 0.0 to 10.0. 7.0+ is publishable quality.
- Penalise: premature reveals, trivial choices, linear sections, logic errors.

CRITICAL RULE — Fix every issue you find:
- If you identify an issue, include the fix in your patches
- node_patches: ONLY the fields that changed for each node (omit unchanged fields)
- new_nodes: complete node definitions for any new nodes
- deleted_nodes: list of node IDs to remove
- Do NOT include the full graph — only return what changed

Issue severities (for reporting):
- "high": must be fixed (safety, broken logic, premature reveal)
- "medium": should be fixed (weak conflict, trivial choices, linearity)
- "low": minor polish (pacing, atmosphere)
- "info": suggestion only (no action required)

Each issue must reference a specific node_id when applicable.

Output format — return JSON:
{
  "score": <float 0.0–10.0>,
  "issues": [
    {
      "severity": "high" | "medium" | "low" | "info",
      "node_id": "<node_id or null>",
      "problem": "<concise description>",
      "suggestion": "<actionable fix>"
    }
  ],
  "node_patches": {
    "node_001": { "scene_goal": "<fixed goal>", "mood": "<fixed mood>" }
  },
  "new_nodes": {
    "node_010": { "id": "node_010", "title": "...", "type": "scene", ... }
  },
  "deleted_nodes": ["node_005"],
  "changes": ["<description of change 1>"],
  "summary": "<1-2 sentence overall assessment>"
}

Constraints:
- node_patches contains ONLY changed fields for each node
- new_nodes must have all required fields (id, title, type, choices)
- Every choice.next_node_id must reference a node that exists
- deleted_nodes: the agent removes them and fixes dangling references automatically
- Keep ALL original node IDs unless you specifically rename them (document this)
- NEVER include internal/programmatic references in scene_goal, titles, or story content
- Return ONLY the JSON object, no prose before or after
"""
