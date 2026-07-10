"""Incremental story graph editing API (Spec §8.1.6 — inkrementelle Verbesserung).

Endpoints:
  PATCH  /api/admin/story-drafts/{id}/nodes/{node_id}          — edit a node
  POST   /api/admin/story-drafts/{id}/nodes                    — add a node
  DELETE /api/admin/story-drafts/{id}/nodes/{node_id}          — delete a node
  PATCH  /api/admin/story-drafts/{id}/nodes/{node_id}/choices/{choice_id}  — edit a choice
  POST   /api/admin/story-drafts/{id}/nodes/{node_id}/regenerate  — re-generate a node via LLM
  GET    /api/admin/story-drafts/{id}/diff                     — diff between last two versions
  GET    /api/admin/story-drafts/{id}/versions                 — list all versions with diffs
"""

from __future__ import annotations

import copy
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.database import get_session
from app.persistence.authoring_repositories import (
    StoryDraftRepository,
    StoryDraftVersionRepository,
)
from app.story.graph_diff import compute_graph_diff

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin-incremental"])


# ── Request schemas ─────────────────────────────────────────────────────

class NodeUpdateRequest(BaseModel):
    """Partial update for a story node. Only provided fields are applied."""
    title: str | None = None
    type: str | None = None
    act: int | None = None
    scene_goal: str | None = None
    scene_text: str | None = None
    location: str | None = None
    characters: list[str] | None = None
    mood: str | None = None
    known_facts: list[str] | None = None
    reveals: list[str] | None = None
    quality_notes: list[str] | None = None
    state_updates: dict[str, Any] | None = None


class NodeCreateRequest(BaseModel):
    """Create a new story node."""
    id: str | None = None
    title: str = "New Node"
    type: str = "scene"
    act: int = 1
    scene_goal: str = ""
    scene_text: str = ""
    location: str = ""
    characters: list[str] = Field(default_factory=list)
    mood: str = ""
    known_facts: list[str] = Field(default_factory=list)
    reveals: list[str] = Field(default_factory=list)
    choices: list[dict[str, Any]] = Field(default_factory=list)
    quality_notes: list[str] = Field(default_factory=list)
    state_updates: dict[str, Any] = Field(default_factory=dict)


class ChoiceUpdateRequest(BaseModel):
    """Partial update for a choice within a node."""
    label: str | None = None
    next_node_id: str | None = None
    state_updates: dict[str, Any] | None = None
    rationale: str | None = None


class NodeRegenerateRequest(BaseModel):
    """Re-generate a single node via LLM."""
    instruction: str | None = Field(
        default=None,
        description="Optional instruction for the LLM (e.g. 'make it more dramatic').",
    )


class NodeSplitRequest(BaseModel):
    """Split a node into multiple sub-nodes.

    The original node's scene_text/scene_goal is divided at the given
    split points, creating a chain of new nodes connected by choices.
    """
    split_points: list[int] = Field(
        ...,
        description="Character positions in scene_text where the text should be split into new nodes.",
    )
    titles: list[str] | None = Field(
        default=None,
        description="Optional titles for the new sub-nodes. Must have len = len(split_points) + 1.",
    )


# ── Response schemas ─────────────────────────────────────────────────────

class GraphMutationResponse(BaseModel):
    """Response after a graph mutation — returns the new version + diff."""
    draft_id: str
    version_id: str
    version_number: int
    node_id: str | None = None
    diff: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


class VersionDiffResponse(BaseModel):
    """Diff between two specific versions."""
    draft_id: str
    old_version_id: str
    new_version_id: str
    old_version_number: int
    new_version_number: int
    diff: dict[str, Any] = Field(default_factory=dict)


class VersionListResponse(BaseModel):
    """List of versions with increment diffs."""
    draft_id: str
    versions: list[dict[str, Any]] = Field(default_factory=list)


# ── Helpers ─────────────────────────────────────────────────────────────

async def _get_draft_version_graph(
    session: AsyncSession, draft_id: str
) -> tuple[StoryDraftVersionRepository, Any, dict[str, Any], dict[str, Any] | None]:
    """Fetch draft, latest version, graph, and outline. Raises 404/409 as needed."""
    draft_repo = StoryDraftRepository(session)
    version_repo = StoryDraftVersionRepository(session)

    draft = await draft_repo.get_by_id(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail=f"Draft '{draft_id}' not found")

    version = await version_repo.latest_for_draft(draft_id)
    if version is None:
        raise HTTPException(status_code=409, detail="Draft has no versions")

    graph = version_repo.parse_graph(version)
    outline = version_repo.parse_outline(version)
    # Return a deep copy as the old_graph snapshot for diff computation
    return version_repo, version, graph, outline, copy.deepcopy(graph)


async def _save_new_version(
    version_repo: StoryDraftVersionRepository,
    draft_id: str,
    graph: dict[str, Any],
    outline: dict[str, Any] | None,
    old_graph: dict[str, Any],
    created_by: str,
    notes: str,
) -> tuple[Any, dict[str, Any]]:
    """Create a new version with the modified graph and compute diff vs old."""
    new_version = await version_repo.create(
        draft_id=draft_id,
        graph=graph,
        outline=outline,
        created_by=created_by,
        notes=notes,
    )
    diff = compute_graph_diff(old_graph, graph)
    return new_version, diff


def _new_node_id() -> str:
    return f"node_{uuid.uuid4().hex[:8]}"


# ── Endpoints ───────────────────────────────────────────────────────────

@router.patch(
    "/story-drafts/{draft_id}/nodes/{node_id}",
    response_model=GraphMutationResponse,
)
async def update_node(
    draft_id: str,
    node_id: str,
    update: NodeUpdateRequest,
    session: AsyncSession = Depends(get_session),
) -> GraphMutationResponse:
    """Edit a single node's fields. Creates a new version with the changes."""
    version_repo, version, graph, outline, old_graph = await _get_draft_version_graph(session, draft_id)

    nodes = graph.get("nodes", {})
    if node_id not in nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    # Apply only non-None fields
    node = nodes[node_id]
    update_data = update.model_dump(exclude_none=True)
    for field, value in update_data.items():
        node[field] = value

    new_version, diff = await _save_new_version(
        version_repo, draft_id, graph, outline, old_graph,
        created_by="manual_edit",
        notes=f"Edited node '{node_id}': {', '.join(update_data.keys())}",
    )

    return GraphMutationResponse(
        draft_id=draft_id,
        version_id=new_version.id,
        version_number=new_version.version_number,
        node_id=node_id,
        diff=diff,
        message=f"Node '{node_id}' updated.",
    )


@router.post(
    "/story-drafts/{draft_id}/nodes",
    response_model=GraphMutationResponse,
    status_code=201,
)
async def add_node(
    draft_id: str,
    node: NodeCreateRequest,
    session: AsyncSession = Depends(get_session),
) -> GraphMutationResponse:
    """Add a new node to the graph. Creates a new version."""
    version_repo, version, graph, outline, old_graph = await _get_draft_version_graph(session, draft_id)

    node_id = node.id or _new_node_id()
    nodes = graph.get("nodes", {})
    if node_id in nodes:
        raise HTTPException(status_code=409, detail=f"Node '{node_id}' already exists")

    node_dict = node.model_dump(exclude_none=True)
    node_dict["id"] = node_id
    node_dict.setdefault("is_start", False)
    node_dict.setdefault("is_end", node.type in ("end", "ending"))
    nodes[node_id] = node_dict

    new_version, diff = await _save_new_version(
        version_repo, draft_id, graph, outline, old_graph,
        created_by="manual_edit",
        notes=f"Added node '{node_id}'",
    )

    return GraphMutationResponse(
        draft_id=draft_id,
        version_id=new_version.id,
        version_number=new_version.version_number,
        node_id=node_id,
        diff=diff,
        message=f"Node '{node_id}' added.",
    )


@router.delete(
    "/story-drafts/{draft_id}/nodes/{node_id}",
    response_model=GraphMutationResponse,
)
async def delete_node(
    draft_id: str,
    node_id: str,
    session: AsyncSession = Depends(get_session),
) -> GraphMutationResponse:
    """Delete a node from the graph. Also cleans up dangling choice references."""
    version_repo, version, graph, outline, old_graph = await _get_draft_version_graph(session, draft_id)

    nodes = graph.get("nodes", {})
    if node_id not in nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    # Prevent deleting the start node
    if graph.get("start_node_id") == node_id:
        raise HTTPException(status_code=409, detail="Cannot delete the start node")

    del nodes[node_id]

    # Clean up dangling choice references in remaining nodes
    for nid, n in nodes.items():
        if isinstance(n, dict):
            for choice in n.get("choices", []) or []:
                if isinstance(choice, dict) and choice.get("next_node_id") == node_id:
                    choice["next_node_id"] = None

    new_version, diff = await _save_new_version(
        version_repo, draft_id, graph, outline, old_graph,
        created_by="manual_edit",
        notes=f"Deleted node '{node_id}'",
    )

    return GraphMutationResponse(
        draft_id=draft_id,
        version_id=new_version.id,
        version_number=new_version.version_number,
        node_id=node_id,
        diff=diff,
        message=f"Node '{node_id}' deleted.",
    )


@router.patch(
    "/story-drafts/{draft_id}/nodes/{node_id}/choices/{choice_id}",
    response_model=GraphMutationResponse,
)
async def update_choice(
    draft_id: str,
    node_id: str,
    choice_id: str,
    update: ChoiceUpdateRequest,
    session: AsyncSession = Depends(get_session),
) -> GraphMutationResponse:
    """Edit a single choice within a node (label, next_node_id, state_updates)."""
    version_repo, version, graph, outline, old_graph = await _get_draft_version_graph(session, draft_id)

    nodes = graph.get("nodes", {})
    if node_id not in nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    node = nodes[node_id]
    choices = node.get("choices", []) or []

    # Find the choice by id
    found = None
    for choice in choices:
        if isinstance(choice, dict) and choice.get("id") == choice_id:
            found = choice
            break

    if found is None:
        raise HTTPException(
            status_code=404,
            detail=f"Choice '{choice_id}' not found in node '{node_id}'",
        )

    # Apply updates
    update_data = update.model_dump(exclude_none=True)
    for field, value in update_data.items():
        found[field] = value

    new_version, diff = await _save_new_version(
        version_repo, draft_id, graph, outline, old_graph,
        created_by="manual_edit",
        notes=f"Edited choice '{choice_id}' in node '{node_id}'",
    )

    return GraphMutationResponse(
        draft_id=draft_id,
        version_id=new_version.id,
        version_number=new_version.version_number,
        node_id=node_id,
        diff=diff,
        message=f"Choice '{choice_id}' updated.",
    )


@router.post(
    "/story-drafts/{draft_id}/nodes/{node_id}/regenerate",
    response_model=GraphMutationResponse,
)
async def regenerate_node(
    draft_id: str,
    node_id: str,
    req: NodeRegenerateRequest,
    session: AsyncSession = Depends(get_session),
) -> GraphMutationResponse:
    """Re-generate a single node's content via the LLM authoring agent.

    Only regenerates scene_goal, characters, reveals, mood, and quality_notes.
    Keeps the node's structural connections (choices, type, act) unchanged.
    Uses the dummy agent for now (zero LLM cost).
    """
    version_repo, version, graph, outline, old_graph = await _get_draft_version_graph(session, draft_id)

    nodes = graph.get("nodes", {})
    if node_id not in nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    node = nodes[node_id]

    # Use the authoring agent to re-generate node content
    from app.services.story_authoring_agent import get_authoring_agent
    agent = get_authoring_agent(dummy=True)

    # Build a mini-brief from the current node + outline context
    brief_context = {
        "title": node.get("title", ""),
        "genre": graph.get("genre", "science_fiction"),
        "tone": graph.get("tone", "dark_mystery"),
        "language": "de",
        "node_count": 1,
        "ending_count": 0,
        "branching_level": "none",
        "notes": req.instruction or f"Re-generate node '{node_id}' with fresh content.",
        "themes": [],
        "forbidden_content": [],
    }

    try:
        new_outline = await agent.generate_outline(brief_context)
        new_graph = await agent.generate_graph(new_outline)

        # Extract the first non-trivial node from the generated graph as replacement content
        new_nodes = new_graph.get("nodes", {})
        replacement_node = None
        for nid, n in new_nodes.items():
            if isinstance(n, dict) and n.get("scene_goal"):
                replacement_node = n
                break

        if replacement_node:
            # Only update content fields, keep structural fields
            node["scene_goal"] = replacement_node.get("scene_goal", node.get("scene_goal", ""))
            node["characters"] = replacement_node.get("characters", node.get("characters", []))
            node["reveals"] = replacement_node.get("reveals", node.get("reveals", []))
            node["mood"] = replacement_node.get("mood", node.get("mood", ""))
            node["quality_notes"] = replacement_node.get("quality_notes", node.get("quality_notes", []))
            node["known_facts"] = replacement_node.get("known_facts", node.get("known_facts", []))
    except Exception as exc:
        logger.error("Node regeneration failed for %s/%s: %s", draft_id, node_id, exc)
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {exc}")

    new_version, diff = await _save_new_version(
        version_repo, draft_id, graph, outline, old_graph,
        created_by="regenerate_agent",
        notes=f"Re-generated node '{node_id}'" + (f": {req.instruction}" if req.instruction else ""),
    )

    return GraphMutationResponse(
        draft_id=draft_id,
        version_id=new_version.id,
        version_number=new_version.version_number,
        node_id=node_id,
        diff=diff,
        message=f"Node '{node_id}' re-generated.",
    )


@router.get(
    "/story-drafts/{draft_id}/diff",
    response_model=VersionDiffResponse,
)
async def get_version_diff(
    draft_id: str,
    old_version_id: str | None = None,
    new_version_id: str | None = None,
    session: AsyncSession = Depends(get_session),
) -> VersionDiffResponse:
    """Get the diff between two versions.

    Defaults to the diff between the last two versions.
    """
    version_repo = StoryDraftVersionRepository(session)
    versions = await version_repo.list_by_draft(draft_id)

    if not versions:
        raise HTTPException(status_code=404, detail="No versions found")

    if old_version_id and new_version_id:
        old_ver = await version_repo.get_by_id(old_version_id)
        new_ver = await version_repo.get_by_id(new_version_id)
        if not old_ver or not new_ver:
            raise HTTPException(status_code=404, detail="Specified version(s) not found")
    elif len(versions) >= 2:
        old_ver = versions[-2]
        new_ver = versions[-1]
    else:
        # Only one version — diff against empty
        old_ver = None
        new_ver = versions[-1]

    old_graph = version_repo.parse_graph(old_ver) if old_ver else {}
    new_graph = version_repo.parse_graph(new_ver)
    diff = compute_graph_diff(old_graph, new_graph)

    return VersionDiffResponse(
        draft_id=draft_id,
        old_version_id=old_ver.id if old_ver else "",
        new_version_id=new_ver.id,
        old_version_number=old_ver.version_number if old_ver else 0,
        new_version_number=new_ver.version_number,
        diff=diff,
    )


@router.get(
    "/story-drafts/{draft_id}/versions",
    response_model=VersionListResponse,
)
async def list_versions_with_diffs(
    draft_id: str,
    session: AsyncSession = Depends(get_session),
) -> VersionListResponse:
    """List all versions with incremental diffs between consecutive versions."""
    version_repo = StoryDraftVersionRepository(session)
    versions = await version_repo.list_by_draft(draft_id)

    result = []
    prev_graph = None
    for v in versions:
        graph = version_repo.parse_graph(v)
        if prev_graph is not None:
            diff = compute_graph_diff(prev_graph, graph)
        else:
            diff = {"summary": "initial version", "added_nodes": list(graph.get("nodes", {}).keys())}
        result.append({
            "id": v.id,
            "version_number": v.version_number,
            "created_by": v.created_by,
            "created_at": v.created_at.isoformat() if v.created_at else None,
            "notes": v.notes,
            "diff_summary": diff.get("summary", ""),
            "diff": diff,
        })
        prev_graph = graph

    return VersionListResponse(
        draft_id=draft_id,
        versions=result,
    )


# ═════════════════════════════════════════════════════════════════════
# Node Split (Spec §8.1.6 — Knoten in Sub-Nodes aufspalten)
# ═════════════════════════════════════════════════════════════════════


class NodeSplitResponse(BaseModel):
    """Response after splitting a node."""
    draft_id: str
    version_id: str
    version_number: int
    original_node_id: str
    new_node_ids: list[str]
    diff: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


@router.post(
    "/story-drafts/{draft_id}/nodes/{node_id}/split",
    response_model=NodeSplitResponse,
)
async def split_node(
    draft_id: str,
    node_id: str,
    req: NodeSplitRequest,
    session: AsyncSession = Depends(get_session),
) -> NodeSplitResponse:
    """Split a node into multiple sub-nodes connected in a chain.

    The node's scene_text is divided at the given character positions.
    Each segment becomes a new sub-node. The original node keeps the
    first segment; new nodes get the subsequent segments.

    The original node's choices are moved to the last sub-node in the chain.
    New choices connecting the chain nodes are created automatically.

    If the node has no scene_text, scene_goal is used instead.
    """
    version_repo, version, graph, outline, old_graph = await _get_draft_version_graph(session, draft_id)

    nodes = graph.get("nodes", {})
    if node_id not in nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    node = nodes[node_id]

    # Determine the text to split: prefer scene_text, fall back to scene_goal
    source_text = node.get("scene_text", "") or node.get("scene_goal", "")
    if not source_text or not source_text.strip():
        raise HTTPException(
            status_code=400,
            detail=f"Node '{node_id}' has no scene_text or scene_goal to split",
        )

    # Validate split points
    split_points = sorted(set(req.split_points))
    for sp in split_points:
        if sp < 0 or sp >= len(source_text):
            raise HTTPException(
                status_code=400,
                detail=f"Split point {sp} is out of range (text length: {len(source_text)})",
            )

    if not split_points:
        raise HTTPException(status_code=400, detail="At least one split point is required")

    # Validate titles if provided
    expected_count = len(split_points) + 1
    titles = req.titles
    if titles is not None:
        if len(titles) != expected_count:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected_count} titles, got {len(titles)}",
            )
    else:
        titles = [node.get("title", node_id)] + [
            f"{node.get('title', node_id)} (Teil {i+2})" for i in range(len(split_points))
        ]

    # Split the text into segments
    segments = []
    prev = 0
    for sp in split_points:
        segments.append(source_text[prev:sp].strip())
        prev = sp
    segments.append(source_text[prev:].strip())

    # Save original choices to move to the last sub-node
    original_choices = node.get("choices", []) or []
    original_type = node.get("type", "scene")
    original_act = node.get("act", 1)
    original_location = node.get("location", "")
    original_mood = node.get("mood", "")
    original_characters = node.get("characters", [])
    original_state_updates = node.get("state_updates", {})

    # Update the original node to have only the first segment
    node["scene_text"] = segments[0]
    node["title"] = titles[0]
    node["choices"] = []  # Will be replaced by chain links

    # Create new sub-nodes for the remaining segments
    new_node_ids = [node_id]
    for i, segment in enumerate(segments[1:], start=1):
        sub_id = _new_node_id()
        sub_node = {
            "id": sub_id,
            "title": titles[i],
            "type": "scene",
            "act": original_act,
            "scene_goal": "",
            "scene_text": segment,
            "location": original_location,
            "characters": list(original_characters),
            "mood": original_mood,
            "known_facts": [],
            "reveals": [],
            "choices": [],
            "quality_notes": [],
            "state_updates": {},
            "is_start": False,
            "is_end": False,
        }
        nodes[sub_id] = sub_node
        new_node_ids.append(sub_id)

    # Connect the chain: each node gets a choice pointing to the next
    for i in range(len(new_node_ids) - 1):
        current_id = new_node_ids[i]
        next_id = new_node_ids[i + 1]
        nodes[current_id]["choices"] = [{
            "id": f"continue_{i+1}",
            "label": "Weiter" if i < len(new_node_ids) - 2 else "Fortsetzen",
            "next_node_id": next_id,
            "state_updates": {},
            "rationale": "Direkte Fortsetzung der Szene",
        }]

    # The last node gets the original choices back
    nodes[new_node_ids[-1]]["choices"] = original_choices

    new_version, diff = await _save_new_version(
        version_repo, draft_id, graph, outline, old_graph,
        created_by="manual_edit",
        notes=f"Split node '{node_id}' into {len(new_node_ids)} sub-nodes",
    )

    return NodeSplitResponse(
        draft_id=draft_id,
        version_id=new_version.id,
        version_number=new_version.version_number,
        original_node_id=node_id,
        new_node_ids=new_node_ids,
        diff=diff,
        message=f"Node '{node_id}' split into {len(new_node_ids)} sub-nodes: {', '.join(new_node_ids)}",
    )


# ═════════════════════════════════════════════════════════════════════
# Story Templates (Spec §8.1.6 — Vorgefertigte Story-Fragmente)
# ═════════════════════════════════════════════════════════════════════


class TemplateListResponse(BaseModel):
    """List of available story templates."""
    templates: list[dict[str, Any]] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


class TemplateApplyRequest(BaseModel):
    """Apply a template to a node."""
    template_id: str
    fields_override: dict[str, Any] | None = Field(
        default=None,
        description="Optional field overrides — e.g. {'mood': 'custom_mood'}",
    )


class TemplateApplyResponse(BaseModel):
    """Response after applying a template to a node."""
    draft_id: str
    version_id: str
    version_number: int
    node_id: str
    template_id: str
    applied_fields: dict[str, Any] = Field(default_factory=dict)
    diff: dict[str, Any] = Field(default_factory=dict)
    message: str = ""


@router.get(
    "/story-templates",
    response_model=TemplateListResponse,
)
async def list_story_templates() -> TemplateListResponse:
    """List all available story fragment templates."""
    from app.story.story_templates import list_templates, list_categories
    return TemplateListResponse(
        templates=list_templates(),
        categories=list_categories(),
    )


@router.get(
    "/story-templates/{template_id}",
)
async def get_story_template(template_id: str) -> JSONResponse:
    """Get a single story template by ID."""
    from app.story.story_templates import get_template
    tpl = get_template(template_id)
    if tpl is None:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
    return JSONResponse(content=tpl)


@router.post(
    "/story-drafts/{draft_id}/nodes/{node_id}/apply-template",
    response_model=TemplateApplyResponse,
)
async def apply_template_to_node(
    draft_id: str,
    node_id: str,
    req: TemplateApplyRequest,
    session: AsyncSession = Depends(get_session),
) -> TemplateApplyResponse:
    """Apply a story template to a node, merging template fields into the node.

    Only non-empty template fields are applied. If fields_override is provided,
    those override the template's fields.
    """
    from app.story.story_templates import apply_template_to_node as _apply

    version_repo, version, graph, outline, old_graph = await _get_draft_version_graph(session, draft_id)

    nodes = graph.get("nodes", {})
    if node_id not in nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    tpl_fields = _apply(req.template_id)
    if tpl_fields is None:
        raise HTTPException(status_code=404, detail=f"Template '{req.template_id}' not found")

    # Apply overrides
    if req.fields_override:
        tpl_fields.update(req.fields_override)

    node = nodes[node_id]
    applied = {}
    for field, value in tpl_fields.items():
        if value is not None and value != "":
            old_val = node.get(field)
            # For list fields, extend rather than replace if the node already has entries
            if isinstance(value, list) and isinstance(old_val, list) and old_val:
                # Append unique items
                merged = list(old_val)
                for item in value:
                    if item not in merged:
                        merged.append(item)
                node[field] = merged
                applied[field] = {"old": old_val, "new": merged}
            else:
                node[field] = value
                applied[field] = {"old": old_val, "new": value}

    new_version, diff = await _save_new_version(
        version_repo, draft_id, graph, outline, old_graph,
        created_by="manual_edit",
        notes=f"Applied template '{req.template_id}' to node '{node_id}'",
    )

    return TemplateApplyResponse(
        draft_id=draft_id,
        version_id=new_version.id,
        version_number=new_version.version_number,
        node_id=node_id,
        template_id=req.template_id,
        applied_fields=applied,
        diff=diff,
        message=f"Template '{req.template_id}' applied to node '{node_id}'.",
    )
