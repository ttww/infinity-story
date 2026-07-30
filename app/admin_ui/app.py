"""Admin UI — Story authoring interface.

Serves HTML pages for:
- Draft list (§8.1.1)
- Story brief form (§8.1.2)
- Graph visualization with zoom/pan, start/end/problematic nodes marked,
  clickable nodes for details (§8.1.3)
- Node detail panel: ID, Titel, Typ, Akt, Ort, Szenenziel, Figuren,
  Reveals, Choices, State-Updates, Kritiker-Kommentare, Warnings (§8.1.4)
- Simulation view (§8.1.5)

Spec Kapitel 8.
"""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from app.persistence.authoring_repositories import (
    StoryDraftRepository,
    StoryDraftVersionRepository,
    StoryReviewReportRepository,
    StoryValidationReportRepository,
)
from app.core.config import get_settings
from app.services.story_authoring_agent import get_authoring_agent
from app.services.story_critic_agent import StoryCriticAgent
from app.services.story_repair_agent import StoryRepairAgent
from app.services.story_validation_service import StoryValidationService
from app.services.publishing_service import PublishingService
from app.services.event_log import event_log, EventLogService
from app.persistence.database import get_session
from app.story.graph import load_graph_from_dict, graph_to_dict
from app.models import StoryBrief, ReviewReport, ReviewIssue
from app.admin_ui.graph_layout import compute_layout, enrich_node_detail
from app.admin_ui.simulation import SimulationEngine

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

admin_app = FastAPI(title="Infinity Story Admin", docs_url="/docs")

# Mount static files if the directory exists
if STATIC_DIR.exists():
    admin_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── Event Log API ────────────────────────────────────────────────────


@admin_app.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    """Log viewer page — shows background process events."""
    from app.core.config import get_settings
    return templates.TemplateResponse(request, "logs.html", {
        "is_mock_mode": get_settings().llm_provider == "mock",
    })


@admin_app.post("/api/events/clear")
async def clear_events():
    """Clear all events from the log."""
    from app.services.event_log import event_log
    count = event_log.clear()
    return {"cleared": count}


# ── Helper functions ──────────────────────────────────────────────────


async def _get_draft_and_version(session: AsyncSession, draft_id: str):
    """Fetch draft and its latest version, raise 404 if not found."""
    draft_repo = StoryDraftRepository(session)
    version_repo = StoryDraftVersionRepository(session)

    draft = await draft_repo.get_by_id(draft_id)
    if draft is None:
        raise HTTPException(status_code=404, detail=f"Draft '{draft_id}' not found")

    version = await version_repo.latest_for_draft(draft_id)
    return draft, version, draft_repo, version_repo


async def _get_graph_data(version, version_repo: StoryDraftVersionRepository) -> dict:
    """Extract and parse graph data from a version."""
    if version is None:
        return {}
    return version_repo.parse_graph(version)


async def _get_review_data(session: AsyncSession, draft_id: str) -> dict | None:
    """Get latest review report data for a draft."""
    review_repo = StoryReviewReportRepository(session)
    reviews = await review_repo.list_by_draft(draft_id)
    if not reviews:
        return None
    latest = reviews[-1]
    return {
        "id": latest.id,
        "score": latest.score,
        "issues": review_repo.parse_issues(latest),
        "summary": latest.summary or "",
    }


async def _get_validation_data(session: AsyncSession, draft_id: str) -> dict | None:
    """Get latest validation report data for a draft."""
    val_repo = StoryValidationReportRepository(session)
    validations = await val_repo.list_by_draft(draft_id)
    if not validations:
        return None
    latest = validations[-1]
    return {
        "is_valid": bool(latest.is_valid),
        "errors": json.loads(latest.errors_json) if isinstance(latest.errors_json, str) else latest.errors_json,
        "warnings": json.loads(latest.warnings_json) if isinstance(latest.warnings_json, str) else latest.warnings_json,
    }


async def _get_eligibility(session: AsyncSession, draft_id: str) -> dict:
    """Check publish eligibility for a draft."""
    pub_service = PublishingService(session)
    can, reasons = await pub_service.can_publish(draft_id)
    checks = {}
    for reason in reasons:
        # Simplify reason into check labels
        checks[reason] = False
    if can:
        checks["eligible"] = True
    return {"eligible": can, "checks": checks, "reasons": reasons}


# ── Routes ────────────────────────────────────────────────────────────


@admin_app.get("/", response_class=HTMLResponse)
async def draft_list(request: Request, session: AsyncSession = Depends(get_session)):
    """Draft Story List — Spec 8.1.1"""
    draft_repo = StoryDraftRepository(session)
    version_repo = StoryDraftVersionRepository(session)
    drafts = await draft_repo.list_all(limit=200)

    # Enrich with version info
    enriched = []
    for d in drafts:
        version = await version_repo.latest_for_draft(d.id)
        node_count = 0
        ending_count = 0
        if version:
            graph_data = version_repo.parse_graph(version)
            nodes = graph_data.get("nodes", {})
            node_count = len(nodes)
            ending_count = len([
                n for n in nodes.values()
                if isinstance(n, dict) and (n.get("type") in ("end", "ending") or n.get("is_end"))
            ])
        enriched.append({
            "id": d.id,
            "title": d.title,
            "genre": d.genre,
            "tone": d.tone,
            "language": d.language,
            "target_age": d.target_age,
            "status": d.status,
            "quality_score": d.quality_score,
            "created_at": d.created_at,
            "approved_at": d.approved_at,
            "published_at": d.published_at,
            "node_count": node_count,
            "ending_count": ending_count,
            "version_number": version.version_number if version else 0,
        })

    return templates.TemplateResponse(request, "draft_list.html", {
        "drafts": enriched,
        "is_mock_mode": get_settings().llm_provider == "mock",
    })


@admin_app.get("/new", response_class=HTMLResponse)
async def brief_form(request: Request):
    """Story Brief Form — Spec 8.1.2"""
    return templates.TemplateResponse(request, "brief_form.html", {
        "is_mock_mode": get_settings().llm_provider == "mock",
    })


@admin_app.post("/new")
async def create_draft(
    title: str = Form("Untitled"),
    genre: str = Form("science_fiction"),
    tone: str = Form("dark_mystery"),
    language: str = Form("de"),
    target_age: str = Form("16+"),
    node_count: int = Form(25),
    ending_count: int = Form(3),
    branching_level: str = Form("medium"),
    themes: str = Form(""),
    forbidden_content: str = Form(""),
    notes: str = Form(""),
    session: AsyncSession = Depends(get_session),
):
    """Handle form submission to create a new draft."""
    brief = StoryBrief(
        title=title,
        genre=genre,
        tone=tone,
        language=language,
        target_age=target_age,
        node_count=node_count,
        ending_count=ending_count,
        branching_level=branching_level,
        themes=[t.strip() for t in themes.split(",") if t.strip()],
        forbidden_content=[t.strip() for t in forbidden_content.split(",") if t.strip()],
        notes=notes,
    )

    draft_repo = StoryDraftRepository(session)
    version_repo = StoryDraftVersionRepository(session)

    # Create draft record
    draft = await draft_repo.create(
        title=brief.title,
        genre=brief.genre,
        tone=brief.tone,
        language=brief.language,
        target_age=brief.target_age,
        brief=brief.model_dump(),
    )

    # Generate graph using the appropriate agent
    # Use real LLM-backed agent when provider != mock; dummy for mock mode
    settings = get_settings()
    use_dummy = settings.llm_provider == "mock"
    agent = get_authoring_agent(dummy=use_dummy)
    brief_dict = brief.model_dump()

    event_log.emit_start("generation", "create_draft", f"Generating story '{title}'", draft_id=draft.id, detail={"title": title, "genre": genre})

    event_log.emit_start("generation", "outline_generation", f"Generating outline for '{title}'", draft_id=draft.id)
    outline = await agent.generate_outline(brief_dict)
    event_log.emit_done("generation", "outline_generation", f"Outline generated: {outline.get('premise', '')[:80]}", draft_id=draft.id, detail={"premise": outline.get("premise", "")[:200]})

    event_log.emit_start("generation", "graph_generation", f"Generating story graph for '{title}'", draft_id=draft.id)
    try:
        graph = await agent.generate_graph(
            outline,
            min_sentences=brief.min_sentences_per_node,
            max_sentences=brief.max_sentences_per_node,
            min_node_connections=brief.min_node_connections,
            max_node_connections=brief.max_node_connections,
        )
    except TypeError:
        graph = await agent.generate_graph(outline)
    node_count = len(graph.get("nodes", {}))
    event_log.emit_done("generation", "graph_generation", f"Graph generated with {node_count} nodes", draft_id=draft.id, detail={"node_count": node_count})

    # Store version
    await version_repo.create(
        draft_id=draft.id,
        graph=graph,
        outline=outline,
        created_by="dummy_agent" if use_dummy else f"{settings.llm_provider}_agent",
        notes="Initial generation via Admin UI",
    )

    from app.models.enums import DraftStatus
    await draft_repo.update_status(draft.id, DraftStatus.GENERATING)
    await draft_repo.update_status(draft.id, DraftStatus.NEEDS_REVIEW)

    event_log.emit_done("generation", "create_draft", f"Draft '{title}' created with {node_count} nodes", draft_id=draft.id, detail={"status": "needs_review"})

    return RedirectResponse(url=f"/admin/draft/{draft.id}", status_code=303)


@admin_app.get("/draft/{draft_id}", response_class=HTMLResponse)
async def draft_detail(request: Request, draft_id: str, session: AsyncSession = Depends(get_session)):
    """Draft detail with graph view and node panel — Spec 8.1.3, 8.1.4"""
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)

    graph_data = await _get_graph_data(version, version_repo)
    review_data = await _get_review_data(session, draft_id)
    validation_data = await _get_validation_data(session, draft_id)
    eligibility = await _get_eligibility(session, draft_id)

    # Compute SVG layout for graph visualization
    review_issues = review_data.get("issues", []) if review_data else []
    layout = compute_layout(graph_data, review_issues)

    # Prepare review issues and validation data for node enrichment
    review_issues = review_data.get("issues", []) if review_data else []
    val_errors = validation_data.get("errors", []) if validation_data else []
    val_warnings = validation_data.get("warnings", []) if validation_data else []

    # Enrich all nodes with detail data for the template
    nodes_detail = {}
    for nid, node in graph_data.get("nodes", {}).items():
        if isinstance(node, dict):
            nodes_detail[nid] = enrich_node_detail(
                nid, node, review_issues, val_errors, val_warnings
            )

    return templates.TemplateResponse(request, "draft_detail.html", {
        "draft": {
            "id": draft.id,
            "title": draft.title,
            "genre": draft.genre,
            "tone": draft.tone,
            "language": draft.language,
            "target_age": draft.target_age,
            "status": draft.status,
            "quality_score": draft.quality_score,
            "min_sentences_per_node": draft.min_sentences_per_node,
            "max_sentences_per_node": draft.max_sentences_per_node,
            "min_node_connections": draft.min_node_connections,
            "max_node_connections": draft.max_node_connections,
            "created_at": draft.created_at,
            "updated_at": draft.updated_at,
            "approved_at": draft.approved_at,
            "published_at": draft.published_at,
        },
        "version": {
            "id": version.id if version else None,
            "version_number": version.version_number if version else 0,
            "created_by": version.created_by if version else "",
            "notes": version.notes if version else "",
        } if version else None,
        "graph": graph_data,
        "graph_json": json.dumps(graph_data, ensure_ascii=False),
        "layout": layout,
        "layout_json": json.dumps(layout, ensure_ascii=False),
        "nodes_detail": nodes_detail,
        "nodes_detail_json": json.dumps(nodes_detail, ensure_ascii=False),
        "review": review_data,
        "validation": validation_data,
        "eligibility": eligibility,
    })


@admin_app.get("/draft/{draft_id}/node/{node_id}", response_class=JSONResponse)
async def get_node_detail(draft_id: str, node_id: str, session: AsyncSession = Depends(get_session)):
    """Get detailed data for a single node — Spec 8.1.4 API endpoint.

    Returns JSON with all node fields, review issues, and tech warnings.
    """
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    graph_data = await _get_graph_data(version, version_repo)

    nodes = graph_data.get("nodes", {})
    if node_id not in nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found in draft '{draft_id}'")

    node_data = nodes[node_id]
    if not isinstance(node_data, dict):
        raise HTTPException(status_code=400, detail=f"Node '{node_id}' data is invalid")

    review_data = await _get_review_data(session, draft_id)
    validation_data = await _get_validation_data(session, draft_id)

    review_issues = review_data.get("issues", []) if review_data else []
    val_errors = validation_data.get("errors", []) if validation_data else []
    val_warnings = validation_data.get("warnings", []) if validation_data else []

    detail = enrich_node_detail(node_id, node_data, review_issues, val_errors, val_warnings)
    return JSONResponse(content=detail)


@admin_app.post("/draft/{draft_id}/review")
async def run_review(draft_id: str, request: Request, session: AsyncSession = Depends(get_session)):
    """Start a critic review in the background and return a task_id.
    Frontend polls GET /draft/{draft_id}/task-status/{task_id} via EventSource."""
    from app.persistence.database import get_session_factory
    from app.models.enums import DraftStatus

    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    if version is None:
        event_log.emit_error("review", "review", f"No version found for draft {draft_id}", draft_id=draft_id)
        raise HTTPException(status_code=404, detail="No version found")

    task_id = f"review-{uuid4().hex[:8]}"
    _task_progress[task_id] = {"phase": "starting", "message": "⏳ Starte Review…", "task_id": task_id}

    asyncio.create_task(_run_review_background(
        task_id, draft_id,
        version.id, draft.title,
        get_session_factory(),
    ))

    return {"task_id": task_id, "ok": True}


async def _run_review_background(
    task_id: str, draft_id: str,
    version_id: str, draft_title: str,
    session_factory,
):
    """Run combined review + repair in a background asyncio task.
    Iterative refinement: if score < 7.0 the agent re-runs automatically."""
    from app.persistence.authoring_repositories import StoryDraftRepository, StoryDraftVersionRepository, StoryReviewReportRepository
    from app.services.story_review_repair_agent import ReviewRepairAgent
    from sqlalchemy import update as sa_update
    from app.models.story_draft import StoryDraft as _SD
    from app.models.enums import DraftStatus

    _set_progress(task_id, "loading", "📂 Lade Draft und Version…")

    async with session_factory() as session:
        try:
            draft_repo = StoryDraftRepository(session)
            version_repo = StoryDraftVersionRepository(session)
            review_repo = StoryReviewReportRepository(session)

            version = await version_repo.get_by_id(version_id)
            if version is None:
                _set_progress(task_id, "error", "Version nicht gefunden")
                event_log.emit_error("review", "review", f"No version found {version_id}", draft_id=draft_id)
                return

            event_log.emit_start("review", "critic_review",
                f"Running review+repair on '{draft_title}'",
                draft_id=draft_id, detail={"version_id": version_id},
            )

            _set_progress(task_id, "parsing", "📖 Lese Graph und Gliederung…")
            graph_data = version_repo.parse_graph(version)
            outline = version_repo.parse_outline(version) or {}

            _set_progress(task_id, "parsing", "📖 Analysiere Graph-Struktur ({})…".format(
                len(graph_data.get("nodes", {}))))

            # ── Combined Review + Repair mit Heartbeat ──
            agent = ReviewRepairAgent()
            result = await _run_with_heartbeat(
                task_id, "review_repair",
                [
                    "📊 Starte Review + Reparatur (Iteration 1)…",
                    "📖 Analysiere Story-Graph…",
                    "🧐 Prüfe narrative Qualität…",
                    "🔧 Repariere gefundene Issues…",
                    "✅ Erstelle verbesserten Graphen…",
                    "⏳ Noch am Arbeiten…",
                ],
                2.5,
                agent.review_and_repair(outline, graph_data),
            )

            score = result.get("score", 0)
            issues = result.get("issues", [])
            repaired_graph = result.get("repaired_graph", graph_data)
            iterations = result.get("iterations_used", 1)
            _set_progress(task_id, "review_repair_done",
                f"✅ Review+Reparatur abgeschlossen "
                f"(Score: {score}/10, {len(issues)} Issues, {iterations} Iterationen)")

            # Save the repaired graph as a new version
            _set_progress(task_id, "saving", "💾 Speichere reparierten Graphen als neue Version…")
            new_version = await version_repo.create(
                draft_id=draft_id, graph=repaired_graph, outline=outline,
                created_by="review_repair_agent",
                notes=f"Review+Repair: Score {score}/10, {len(issues)} Issues, {iterations} Iterationen",
            )

            _set_progress(task_id, "saving", "💾 Speichere Review-Bericht…")
            await review_repo.create(
                draft_id=draft_id, version_id=new_version.id,
                score=score, issues=issues,
                summary=result.get("summary"),
            )

            _set_progress(task_id, "saving", "💾 Aktualisiere Quality-Score…")
            await session.execute(
                sa_update(_SD)
                .where(_SD.id == draft_id)
                .values(quality_score=score)
            )

            if score < 7.0:
                await draft_repo.update_status(draft_id, DraftStatus.NEEDS_REVIEW)
            else:
                try:
                    await draft_repo.update_status(draft_id, DraftStatus.VALIDATED)
                except ValueError:
                    pass  # bereits validated — kein Fehler

            await session.commit()

            issue_count = len(issues)
            event_log.emit_done("review", "critic_review",
                f"Review+Repair completed: score {score}, {issue_count} issues, {iterations} iterations",
                draft_id=draft_id,
                detail={"score": score, "issues": issue_count, "iterations": iterations,
                        "summary": result.get("summary", "")},
            )

            _set_progress(task_id, "done", json.dumps({
                "ok": True, "score": score, "issues": issue_count, "iterations": iterations,
            }, ensure_ascii=False))

        except Exception as exc:
            event_log.emit_error("review", "review", f"Review+Repair failed: {exc}", draft_id=draft_id)
            _set_progress(task_id, "error", f"Fehler: {exc}")
            import traceback
            traceback.print_exc()


@admin_app.post("/draft/{draft_id}/repair")
async def run_repair(draft_id: str, session: AsyncSession = Depends(get_session)):
    """Start combined review+repair in background, return task_id."""
    from app.persistence.database import get_session_factory

    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    if version is None:
        event_log.emit_error("repair", "repair", f"No version found for draft {draft_id}", draft_id=draft_id)
        raise HTTPException(status_code=404, detail="No version found")

    task_id = f"repair-{uuid4().hex[:8]}"
    _task_progress[task_id] = {"phase": "starting", "message": "⏳ Starte Reparatur…", "task_id": task_id}

    asyncio.create_task(_run_repair_background(
        task_id, draft_id, version.id, draft.title,
        get_session_factory(),
    ))

    return {"task_id": task_id, "ok": True}


async def _run_repair_background(
    task_id: str, draft_id: str,
    version_id: str, draft_title: str,
    session_factory,
):
    """Run combined review+repair in background (Repair button)."""
    from app.persistence.authoring_repositories import StoryDraftRepository, StoryDraftVersionRepository
    from app.services.story_review_repair_agent import ReviewRepairAgent
    from sqlalchemy import update as sa_update
    from app.models.story_draft import StoryDraft as _SD
    from app.models.enums import DraftStatus

    _set_progress(task_id, "loading", "📂 Lade Draft und Version…")

    async with session_factory() as session:
        try:
            draft_repo = StoryDraftRepository(session)
            version_repo = StoryDraftVersionRepository(session)

            version = await version_repo.get_by_id(version_id)
            if version is None:
                _set_progress(task_id, "error", "Version nicht gefunden")
                event_log.emit_error("repair", "repair", f"No version {version_id}", draft_id=draft_id)
                return

            event_log.emit_start("repair", "repair_pass",
                f"Running repair on '{draft_title}'",
                draft_id=draft_id, detail={"version_id": version_id},
            )

            _set_progress(task_id, "parsing", "📖 Lese Graph und Gliederung…")
            graph_data = version_repo.parse_graph(version)
            outline = version_repo.parse_outline(version) or {}

            # ── Combined Review + Repair with Heartbeat ──
            agent = ReviewRepairAgent()
            result = await _run_with_heartbeat(
                task_id, "repair_iteration",
                [
                    "🔧 Starte Reparatur + Review (Iteration 1)…",
                    "📖 Analysiere Graph…",
                    "🧠 Repariere Issues…",
                    "✅ Prüffe Ergebnis…",
                    "⏳ Noch am Arbeiten…",
                ],
                2.5,
                agent.review_and_repair(outline, graph_data),
            )

            repaired_graph = result.get("repaired_graph", graph_data)
            score = result.get("score", 0)
            issues = result.get("issues", [])
            iterations = result.get("iterations_used", 1)

            _set_progress(task_id, "repair_done",
                f"✅ Reparatur abgeschlossen (Score: {score}/10, {len(issues)} Issues, {iterations} Iterationen)")

            _set_progress(task_id, "saving", "💾 Speichere neue Version…")
            new_version = await version_repo.create(
                draft_id=draft_id, graph=repaired_graph, outline=outline,
                created_by="review_repair_agent",
                notes=f"Reparatur: Score {score}/10, {iterations} Iterationen",
            )

            _set_progress(task_id, "saving", "💾 Aktualisiere Quality-Score…")
            await session.execute(
                sa_update(_SD)
                .where(_SD.id == draft_id)
                .values(quality_score=score)
            )

            if score < 7.0:
                await draft_repo.update_status(draft_id, DraftStatus.NEEDS_REVIEW)
            else:
                try:
                    await draft_repo.update_status(draft_id, DraftStatus.VALIDATED)
                except ValueError:
                    pass  # bereits validated — kein Fehler

            await session.commit()

            event_log.emit_done("repair", "repair_pass",
                f"Repair completed: score {score}, {len(issues)} issues, {iterations} iterations",
                draft_id=draft_id,
                detail={"score": score, "issues": len(issues), "iterations": iterations},
            )

            _set_progress(task_id, "done", json.dumps({
                "ok": True, "score": score, "issues": len(issues), "iterations": iterations,
            }, ensure_ascii=False))

        except Exception as exc:
            event_log.emit_error("repair", "repair", f"Repair failed: {exc}", draft_id=draft_id)
            _set_progress(task_id, "error", f"Fehler: {exc}")
            import traceback
            traceback.print_exc()


from uuid import uuid4

# In-memory progress store for background tasks (fix, review, etc.)
_task_progress: dict[str, dict] = {}


@admin_app.post("/draft/{draft_id}/fix-issue")
async def fix_issue(draft_id: str, request: Request, session: AsyncSession = Depends(get_session)):
    """Start a targeted FixThis repair in the background and return a task_id.
    Frontend polls GET /draft/{draft_id}/fix-status/{task_id} via EventSource."""
    from app.persistence.database import get_session_factory

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        payload = {}

    finding_id = payload.get("finding_id") or payload.get("id") or payload.get("issue_id")
    report_id = payload.get("report_id")
    try:
        issue_index = int(finding_id) - 1
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="finding_id must be a one-based integer")

    # Validate finding exists before spawning background task
    review_repo = StoryReviewReportRepository(session)
    source_report = None
    if report_id:
        source_report = await review_repo.get_by_id(report_id)
        if source_report is not None and source_report.draft_id != draft_id:
            source_report = None
    if source_report is None:
        reviews = await review_repo.list_by_draft(draft_id)
        if not reviews:
            raise HTTPException(status_code=400, detail="Run review first")
        source_report = reviews[-1]

    source_issues = review_repo.parse_issues(source_report)
    if issue_index < 0 or issue_index >= len(source_issues):
        raise HTTPException(status_code=404, detail=f"Finding {finding_id} not found")

    task_id = f"{draft_id}-{uuid4().hex[:8]}"
    _task_progress[task_id] = {"phase": "starting", "message": "⏳ Starte...", "task_id": task_id}

    # Spawn background task with a fresh DB session
    asyncio.create_task(_run_fix_background(
        task_id, draft_id, issue_index,
        source_report.id, source_report.score, source_report.summary,
        source_issues[issue_index],
        get_session_factory(),
    ))

    return {"task_id": task_id, "ok": True}


async def _run_with_heartbeat(
    task_id: str, phase: str, messages: list[str],
    interval: float, coro,
):
    """Run an async coroutine while sending periodic heartbeat progress updates.

    Iterates through ``messages`` cyclically every ``interval`` seconds
    during the wait, so the user sees changing status (e.g. "Analyziere Nodes…",
    "Verbinde mit Story-Schema…", "Noch am Überlegen…") instead of a frozen
    "Frage LLM…".
    """
    _set_progress(task_id, phase, messages[0])
    msg_index = 1

    async def heartbeat():
        nonlocal msg_index
        while True:
            await asyncio.sleep(interval)
            _set_progress(task_id, phase, messages[msg_index % len(messages)])
            msg_index += 1

    hb_task = asyncio.create_task(heartbeat())
    try:
        result = await coro
        return result
    finally:
        hb_task.cancel()
        try:
            await hb_task
        except asyncio.CancelledError:
            pass


async def _run_fix_background(
    task_id: str, draft_id: str,
    issue_index: int, source_report_id: str,
    source_score: float, source_summary: str | None,
    target_issue: dict,
    session_factory,
):
    """Run targeted FixThis repair using combined review+repair agent."""
    from app.persistence.authoring_repositories import StoryDraftRepository, StoryDraftVersionRepository, StoryReviewReportRepository
    from app.services.story_review_repair_agent import ReviewRepairAgent
    from sqlalchemy import update as sa_update
    from app.models.story_draft import StoryDraft as _SD

    _set_progress(task_id, "loading", "📂 Lade Draft und Version...")

    async with session_factory() as session:
        try:
            draft_repo = StoryDraftRepository(session)
            version_repo = StoryDraftVersionRepository(session)
            review_repo = StoryReviewReportRepository(session)

            draft = await draft_repo.get_by_id(draft_id)
            if draft is None:
                _set_progress(task_id, "error", f"Draft {draft_id} nicht gefunden")
                event_log.emit_error("repair", "fix_issue", f"Draft not found {draft_id}", draft_id=draft_id)
                return
            version = await version_repo.latest_for_draft(draft_id)
            if version is None:
                _set_progress(task_id, "error", "Keine Version gefunden")
                event_log.emit_error("repair", "fix_issue", f"No version for draft {draft_id}", draft_id=draft_id)
                return

            event_log.emit_start("repair", "fix_issue",
                f"FixThis finding #{issue_index + 1} on '{draft.title}'",
                draft_id=draft_id,
                detail={"report_id": source_report_id, "finding_id": issue_index + 1},
            )

            _set_progress(task_id, "loading", "📋 Markiere Finding als fix_requested...")
            await review_repo.update_issue_status(source_report_id, issue_index, "fix_requested")

            _set_progress(task_id, "parsing", "📖 Lese Graph und Gliederung…")
            graph_data = version_repo.parse_graph(version)
            outline = version_repo.parse_outline(version) or {}

            # Inject the specific issue into the outline so the combined
            # agent knows what to focus on
            focused_outline = dict(outline)
            focused_outline["_fix_this_issue"] = {
                "finding_id": issue_index + 1,
                "score": source_score,
                "issue": target_issue,
            }

            # ── Combined Review + Repair (targeted) with Heartbeat ──
            agent = ReviewRepairAgent()
            result = await _run_with_heartbeat(
                task_id, "fix_repair",
                [
                    f"🔧 FixThis #{issue_index + 1}: Repariere Issue…",
                    "📖 Analysiere betroffenen Node…",
                    "🧠 Wende Reparatur an…",
                    "✅ Prüffe Korrektur…",
                    "⏳ Noch am Arbeiten…",
                ],
                2.5,
                agent.review_and_repair(focused_outline, graph_data),
            )

            repaired_graph = result.get("repaired_graph", graph_data)
            score = result.get("score", 0)
            issues = result.get("issues", [])
            iterations = result.get("iterations_used", 1)

            _set_progress(task_id, "fix_done",
                f"✅ FixThis abgeschlossen (Score: {score}/10, {len(issues)} Issues, {iterations} Iterationen)")

            _set_progress(task_id, "saving", "💾 Speichere neue Version…")
            new_version = await version_repo.create(
                draft_id=draft_id, graph=repaired_graph, outline=outline,
                created_by="review_repair_agent",
                notes=f"FixThis #{issue_index + 1}: Score {score}/10, {iterations} Iterationen",
            )
            _set_progress(task_id, "saving", "💾 Neue Version v{} gespeichert".format(
                new_version.version_number if hasattr(new_version, 'version_number') else '?'))

            _set_progress(task_id, "saving", "💾 Speichere Re-Review-Bericht…")
            new_report = await review_repo.create(
                draft_id=draft_id, version_id=new_version.id,
                score=score, issues=issues,
                summary=result.get("summary"),
            )

            await review_repo.update_issue_status(source_report_id, issue_index, "fixed")

            await session.execute(
                sa_update(_SD)
                .where(_SD.id == draft_id)
                .values(quality_score=score)
            )
            await session.commit()

            event_log.emit_done("repair", "fix_issue",
                f"FixThis #{issue_index + 1} complete; score {score}, {len(issues)} issues, {iterations} iterations",
                draft_id=draft_id,
                detail={
                    "source_report_id": source_report_id,
                    "new_report_id": new_report.id,
                    "new_version_id": new_version.id,
                    "finding_id": issue_index + 1,
                    "score": score,
                    "issues": len(issues),
                    "iterations": iterations,
                },
            )

            _set_progress(task_id, "done", json.dumps({
                "ok": True, "finding_id": issue_index + 1,
                "fix_status": "fixed",
                "version_id": new_version.id,
                "report_id": new_report.id,
                "review": {
                    "score": score,
                    "issues": issues,
                    "summary": result.get("summary"),
                },
            }, ensure_ascii=False))

        except Exception as exc:
            event_log.emit_error("repair", "fix_issue", f"FixThis failed: {exc}", draft_id=draft_id)
            _set_progress(task_id, "error", f"Fehler: {exc}")
            import traceback
            traceback.print_exc()


def _set_progress(task_id: str, phase: str, message: str):
    """Update progress for a background FixThis task."""
    _task_progress[task_id] = {"phase": phase, "message": message, "done": phase in ("done", "error"), "task_id": task_id}


@admin_app.get("/draft/{draft_id}/task-status/{task_id}")
async def task_status(draft_id: str, task_id: str):
    """SSE endpoint: streams progress updates for any background task (fix, review, etc.)."""
    from starlette.responses import StreamingResponse

    async def event_stream():
        # Send initial state immediately
        task = _task_progress.get(task_id)
        if task:
            yield f"data: {json.dumps(task, ensure_ascii=False)}\n\n"

        # Poll for updates until done
        while True:
            task = _task_progress.get(task_id)
            if not task:
                yield f"data: {json.dumps({'phase': 'error', 'message': 'Task unbekannt', 'done': True})}\n\n"
                return
            if task.get("done"):
                # Send one final update and close
                yield f"data: {json.dumps(task, ensure_ascii=False)}\n\n"
                return
            await asyncio.sleep(0.5)
            yield f"data: {json.dumps(task, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@admin_app.post("/draft/{draft_id}/validate")
async def run_validation(draft_id: str, session: AsyncSession = Depends(get_session)):
    """Run validation."""
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    if version is None:
        event_log.emit_error("validation", "validate", f"No version found for draft {draft_id}", draft_id=draft_id)
        raise HTTPException(status_code=404, detail="No version found")

    graph_data = version_repo.parse_graph(version)

    event_log.emit_start("validation", "validate", f"Validating graph for '{draft.title}'", draft_id=draft_id, detail={"version_id": version.id})

    validator = StoryValidationService()
    result = await validator.validate(graph_data)

    val_repo = StoryValidationReportRepository(session)
    await val_repo.create(
        draft_id=draft_id,
        version_id=version.id,
        is_valid=result["is_valid"],
        errors=result["errors"],
        warnings=result["warnings"],
    )

    from app.models.enums import DraftStatus
    if result["is_valid"]:
        await draft_repo.update_status(draft_id, DraftStatus.VALIDATED)
    else:
        await draft_repo.update_status(draft_id, DraftStatus.NEEDS_REPAIR)
        await draft_repo.update_status(draft_id, DraftStatus.NEEDS_REVIEW)

    err_count = len(result.get("errors", []))
    warn_count = len(result.get("warnings", []))
    event_log.emit_done("validation", "validate", f"Validation: {'✅ valid' if result['is_valid'] else '❌ invalid'} — {err_count} errors, {warn_count} warnings", draft_id=draft_id, detail={"is_valid": result["is_valid"], "errors": err_count, "warnings": warn_count})

    return RedirectResponse(url=f"/admin/draft/{draft_id}", status_code=303)


@admin_app.post("/draft/{draft_id}/approve")
async def run_approve(draft_id: str, session: AsyncSession = Depends(get_session)):
    """Approve draft."""
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)

    from app.models.enums import DraftStatus
    if draft.status != DraftStatus.VALIDATED.value:
        event_log.emit_error("approve", "approve", f"Cannot approve draft {draft_id}: status is '{draft.status}', must be 'validated'", draft_id=draft_id, detail={"status": draft.status})
        raise HTTPException(status_code=409, detail=f"Draft must be 'validated' to approve, got '{draft.status}'")

    await draft_repo.update_status(draft_id, DraftStatus.APPROVED)
    event_log.emit_done("approve", "approve", f"Draft '{draft.title}' approved", draft_id=draft_id, detail={"status": "approved"})
    return RedirectResponse(url=f"/admin/draft/{draft_id}", status_code=303)


@admin_app.post("/draft/{draft_id}/publish")
async def run_publish(draft_id: str, session: AsyncSession = Depends(get_session)):
    """Publish draft."""
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)

    from app.models.enums import DraftStatus
    if draft.status != DraftStatus.APPROVED.value:
        event_log.emit_error("publish", "publish", f"Cannot publish draft {draft_id}: status is '{draft.status}', must be 'approved'", draft_id=draft_id, detail={"status": draft.status})
        raise HTTPException(status_code=409, detail=f"Draft must be 'approved' to publish, got '{draft.status}'")

    event_log.emit_start("publish", "publish", f"Publishing draft '{draft.title}'", draft_id=draft_id, detail={"status": draft.status})

    pub_service = PublishingService(session)
    try:
        result = await pub_service.publish(draft_id)
    except ValueError as e:
        event_log.emit_error("publish", "publish", f"Publishing failed: {e}", draft_id=draft_id, detail={"error": str(e)})
        raise HTTPException(status_code=400, detail=str(e))

    event_log.emit_done("publish", "publish", f"Draft '{draft.title}' published as '{result['scenario_id']}' ({result['nodes_published']} nodes)", draft_id=draft_id, detail=result)
    return RedirectResponse(url=f"/admin/draft/{draft_id}", status_code=303)


@admin_app.post("/draft/{draft_id}/settings", response_class=JSONResponse)
async def update_story_settings(
    draft_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Update story configuration parameters (sentence/connection bounds).

    Accepts JSON with min_sentences_per_node, max_sentences_per_node,
    min_node_connections, max_node_connections.  Validates that min <= max
    for each pair and that values are within allowed bounds.
    """
    from sqlalchemy import update as sa_update
    from app.models.story_draft import StoryDraft as _SD

    body = await request.json()
    errors: list[str] = []

    # Parse and validate each field
    fields = {
        "min_sentences_per_node": (1, 50),
        "max_sentences_per_node": (1, 100),
        "min_node_connections": (0, 20),
        "max_node_connections": (0, 50),
    }

    values: dict[str, int] = {}
    for field_name, (lo, hi) in fields.items():
        raw = body.get(field_name)
        if raw is None:
            errors.append(f"Missing field: {field_name}")
            continue
        try:
            val = int(raw)
        except (ValueError, TypeError):
            errors.append(f"Invalid integer for {field_name}: {raw!r}")
            continue
        if val < lo or val > hi:
            errors.append(f"{field_name} must be between {lo} and {hi}")
            continue
        values[field_name] = val

    # Cross-field validation: min <= max for each pair
    if not errors:
        if values["min_sentences_per_node"] > values["max_sentences_per_node"]:
            errors.append("min_sentences_per_node must be <= max_sentences_per_node")
        if values["min_node_connections"] > values["max_node_connections"]:
            errors.append("min_node_connections must be <= max_node_connections")

    if errors:
        return JSONResponse(
            status_code=422,
            content={"ok": False, "errors": errors},
        )

    # Persist to database
    stmt = (
        sa_update(_SD)
        .where(_SD.id == draft_id)
        .values(**values)
    )
    await session.execute(stmt)
    await session.commit()

    event_log.emit_done(
        "config", "update_settings",
        f"Story parameters updated for draft '{draft_id}'",
        draft_id=draft_id,
        detail=values,
    )

    return JSONResponse(
        status_code=200,
        content={"ok": True, "values": values},
    )


@admin_app.post("/draft/{draft_id}/check-limits", response_class=JSONResponse)
async def check_node_limits(
    draft_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Check which nodes would need adjustment given new limit parameters.

    Accepts JSON with the four limit fields and returns a preview of
    all sentence/connection changes that would be applied.
    """
    from app.story.limits import preview_limit_adjustments

    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    if version is None:
        return JSONResponse(status_code=404, content={"ok": False, "error": "No version found"})

    body = await request.json()
    min_s = int(body.get("min_sentences_per_node", 3))
    max_s = int(body.get("max_sentences_per_node", 8))
    min_c = int(body.get("min_node_connections", 2))
    max_c = int(body.get("max_node_connections", 4))

    graph_data = version_repo.parse_graph(version)
    preview = preview_limit_adjustments(graph_data, min_s, max_s, min_c, max_c)

    return JSONResponse(content={"ok": True, "preview": preview})


@admin_app.post("/draft/{draft_id}/apply-limits", response_class=JSONResponse)
async def apply_node_limits(
    draft_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Apply limit adjustments to all nodes, creating a new version.

    Accepts JSON with:
      - The four limit fields
      - mode: "auto" (default) | "sentences_only" | "connections_only"

    Returns the new version info.
    """
    from app.story.limits import apply_limit_adjustments, find_violating_nodes

    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    if version is None:
        return JSONResponse(status_code=404, content={"ok": False, "error": "No version found"})

    body = await request.json()
    min_s = int(body.get("min_sentences_per_node", 3))
    max_s = int(body.get("max_sentences_per_node", 8))
    min_c = int(body.get("min_node_connections", 2))
    max_c = int(body.get("max_node_connections", 4))
    mode = body.get("mode", "auto")

    graph_data = version_repo.parse_graph(version)
    new_graph = apply_limit_adjustments(graph_data, min_s, max_s, min_c, max_c, mode=mode)

    # Count remaining violations
    remaining = find_violating_nodes(new_graph, min_s, max_s, min_c, max_c)

    result = await _save_graph_as_new_version(
        session, draft_id, new_graph,
        f"Limit-Anpassung (mode={mode}): {min_s}-{max_s} Sätze, {min_c}-{max_c} Verbindungen",
        created_by="limit_adjustment",
    )

    event_log.emit_done(
        "config", "apply_limits",
        f"Node-Limits angewendet: {len(new_graph.get('nodes', {}))} Nodes angepasst",
        draft_id=draft_id,
        detail={"mode": mode, "remaining_violations": len(remaining)},
    )

    return JSONResponse(content={
        "ok": True,
        "version_id": result["version_id"],
        "version_number": result["version_number"],
        "remaining_violations": remaining,
    })


@admin_app.post("/draft/{draft_id}/delete")
async def delete_draft(draft_id: str, session: AsyncSession = Depends(get_session)):
    """Delete a draft and all cascade children (versions, reviews, validations, jobs).

    Published scenarios are NOT deleted — only the authoring draft data.
    Emits an event-log entry and redirects to the draft list.
    """
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    title = draft.title

    deleted = await draft_repo.delete(draft_id)
    if not deleted:
        event_log.emit_error("delete", "delete_draft", f"Failed to delete draft '{draft_id}'", draft_id=draft_id)
        raise HTTPException(status_code=500, detail="Failed to delete draft")

    event_log.emit_done("delete", "delete_draft", f"Draft '{title}' ({draft_id}) deleted with all versions, reviews, and jobs", draft_id=draft_id, detail={"title": title, "draft_id": draft_id})
    return RedirectResponse(url="/admin/", status_code=303)


@admin_app.get("/draft/{draft_id}/simulate", response_class=HTMLResponse)
async def simulate(request: Request, draft_id: str, session: AsyncSession = Depends(get_session)):
    """Simulation View — Spec §8.1.5

    Renders the simulation page with the graph data. The frontend
    JavaScript handles the interactive flow (start, choose, path, state-diff).
    """
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    graph_data = await _get_graph_data(version, version_repo)

    # Pre-compute the initial simulation state for the frontend
    sim_start = SimulationEngine.start(graph_data)

    return templates.TemplateResponse(request, "simulate.html", {
        "draft": {
            "id": draft.id,
            "title": draft.title,
            "status": draft.status,
        },
        "graph": graph_data,
        "graph_json": json.dumps(graph_data, ensure_ascii=False),
        "sim_start_json": json.dumps(sim_start, ensure_ascii=False),
    })


@admin_app.get("/draft/{draft_id}/simulate/start", response_class=JSONResponse)
async def simulate_start(draft_id: str, session: AsyncSession = Depends(get_session)):
    """Start a simulation — Spec §8.1.5 API endpoint.

    Returns JSON with the initial scene, choices, world state, and path.
    """
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    graph_data = await _get_graph_data(version, version_repo)
    result = SimulationEngine.start(graph_data)
    return JSONResponse(content=result)


@admin_app.post("/draft/{draft_id}/simulate/choose", response_class=JSONResponse)
async def simulate_choose(
    draft_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Process a choice in the simulation — Spec §8.1.5 API endpoint.

    Accepts JSON body: {
        "current_node_id": "...",
        "choice_id": "...",
        "world_state": {...},
        "path": ["node_001", ...],
        "step_count": N
    }

    Returns JSON with updated scene, choices, world_state, state_diff, path.
    """
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    graph_data = await _get_graph_data(version, version_repo)

    body = await request.json()
    current_node_id = body.get("current_node_id", "")
    choice_id = body.get("choice_id", "")
    world_state = body.get("world_state", {})
    path = body.get("path", [])
    step_count = body.get("step_count", 0)

    result = SimulationEngine.choose(
        graph_data, current_node_id, choice_id, world_state, path, step_count
    )
    return JSONResponse(content=result)


@admin_app.get("/draft/{draft_id}/simulate/state", response_class=JSONResponse)
async def simulate_state(
    draft_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Reconstruct full simulation state from a path — Spec §8.1.5 API endpoint.

    Query params: path (comma-separated node IDs), world_state (JSON string)
    """
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    graph_data = await _get_graph_data(version, version_repo)

    path_str = request.query_params.get("path", "")
    path = [p for p in path_str.split(",") if p] if path_str else []

    ws_str = request.query_params.get("world_state", "{}")
    try:
        world_state = json.loads(ws_str) if isinstance(ws_str, str) else ws_str
    except (json.JSONDecodeError, TypeError):
        world_state = {}

    step_count = int(request.query_params.get("step_count", "0"))

    result = SimulationEngine.get_full_state(graph_data, path, world_state, step_count)
    return JSONResponse(content=result)


# ═══════════════════════════════════════════════════════════════════════
# Event Log: REST + SSE + Logs Page (t_58834261)
# ═══════════════════════════════════════════════════════════════════════


@admin_app.get("/api/events", response_class=JSONResponse)
async def list_events(
    request: Request,
    limit: int = 100,
    offset: int = 0,
    category: str = "all",
    status: str = "all",
    draft_id: str | None = None,
):
    """REST endpoint: poll recent events with optional filters.

    Query params:
      limit   — max events to return (default 100)
      offset  — skip first N matching events (pagination)
      category — all | generation | review | repair | validation | publish | approve
      status   — all | running | done | error
      draft_id — filter by draft ID
    """
    return JSONResponse(content=event_log.list_events(
        limit=limit, offset=offset, category=category, status=status, draft_id=draft_id,
    ))


@admin_app.get("/api/events/stats", response_class=JSONResponse)
async def event_stats(request: Request):
    """REST endpoint: event log statistics."""
    return JSONResponse(content=event_log.get_stats())


@admin_app.get("/api/events/live")
async def events_live(request: Request):
    """SSE endpoint: stream live events.

    Returns Server-Sent Events.  Each event is a JSON object.
    Optional query params: category, draft_id for server-side filtering.
    """
    category = request.query_params.get("category")
    if category == "all":
        category = None
    draft_id = request.query_params.get("draft_id")

    async def event_stream():
        try:
            # Send a comment to establish the connection
            yield ": connected\n\n"
            # Stream events (subscribe_filtered sends initial snapshot + live updates)
            async for sse_data in event_log.subscribe_filtered(category=category, draft_id=draft_id):
                # Check if client has disconnected
                if await request.is_disconnected():
                    break
                yield sse_data
        except asyncio.CancelledError:
            # Client disconnected — normal, just exit
            raise
        except Exception:
            # Log the error but don't crash — send an error event to the client
            yield f"event: error\ndata: {{\"error\": \"stream_error\"}}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@admin_app.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    """Live event log page — shows background process events in real time.

    Features: SSE streaming, auto-scroll, category filter, JSON detail view.
    """
    return templates.TemplateResponse(request, "logs.html", {})


# ═══════════════════════════════════════════════════════════════════════
# Incremental Graph Editing (Spec §8.1.6 — inkrementelle Verbesserung)
# ═══════════════════════════════════════════════════════════════════════


async def _save_graph_as_new_version(
    session: AsyncSession, draft_id: str, graph: dict, notes: str, created_by: str = "manual_edit"
) -> dict:
    """Save a modified graph as a new version and return summary data."""
    version_repo = StoryDraftVersionRepository(session)
    latest = await version_repo.latest_for_draft(draft_id)
    outline = version_repo.parse_outline(latest) if latest else None
    old_graph = version_repo.parse_graph(latest) if latest else {}

    from app.story.graph_diff import compute_graph_diff
    new_version = await version_repo.create(
        draft_id=draft_id, graph=graph, outline=outline,
        created_by=created_by, notes=notes,
    )
    diff = compute_graph_diff(old_graph, graph)
    return {
        "version_id": new_version.id,
        "version_number": new_version.version_number,
        "diff": diff,
    }


@admin_app.post("/draft/{draft_id}/nodes/{node_id}/edit")
async def edit_node(
    draft_id: str,
    node_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Edit a node's fields via form submission. Creates a new version."""
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    graph_data = version_repo.parse_graph(version)
    nodes = graph_data.get("nodes", {})
    if node_id not in nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    form = await request.form()
    node = nodes[node_id]

    # Update fields from form data
    if "title" in form:
        node["title"] = form["title"]
    if "type" in form:
        node["type"] = form["type"]
    if "act" in form:
        try:
            node["act"] = int(form["act"])
        except (ValueError, TypeError):
            pass
    if "scene_goal" in form:
        node["scene_goal"] = form["scene_goal"]
    if "scene_text" in form:
        node["scene_text"] = form["scene_text"]
    if "location" in form:
        node["location"] = form["location"]
    if "mood" in form:
        node["mood"] = form["mood"]
    if "characters" in form:
        node["characters"] = [c.strip() for c in form["characters"].split(",") if c.strip()]
    if "reveals" in form:
        node["reveals"] = [r.strip() for r in form["reveals"].split("\n") if r.strip()]
    if "quality_notes" in form:
        node["quality_notes"] = [q.strip() for q in form["quality_notes"].split("\n") if q.strip()]

    await _save_graph_as_new_version(session, draft_id, graph_data, f"Edited node '{node_id}'")
    return RedirectResponse(url=f"/admin/draft/{draft_id}", status_code=303)


@admin_app.post("/draft/{draft_id}/nodes/add")
async def add_node(
    draft_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Add a new node to the graph via form submission."""
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    graph_data = version_repo.parse_graph(version)

    form = await request.form()
    import uuid as _uuid
    node_id = form.get("node_id") or f"node_{_uuid.uuid4().hex[:8]}"

    nodes = graph_data.get("nodes", {})
    if node_id in nodes:
        raise HTTPException(status_code=409, detail=f"Node '{node_id}' already exists")

    try:
        act = int(form.get("act", "1"))
    except (ValueError, TypeError):
        act = 1

    node_type = form.get("type", "scene")
    nodes[node_id] = {
        "id": node_id,
        "title": form.get("title", "New Node"),
        "type": node_type,
        "act": act,
        "scene_goal": form.get("scene_goal", ""),
        "scene_text": form.get("scene_text", ""),
        "location": form.get("location", ""),
        "characters": [c.strip() for c in form.get("characters", "").split(",") if c.strip()],
        "mood": form.get("mood", ""),
        "reveals": [],
        "choices": [],
        "quality_notes": [],
        "state_updates": {},
        "is_start": False,
        "is_end": node_type in ("end", "ending"),
    }

    await _save_graph_as_new_version(session, draft_id, graph_data, f"Added node '{node_id}'")
    return RedirectResponse(url=f"/admin/draft/{draft_id}", status_code=303)


@admin_app.post("/draft/{draft_id}/nodes/{node_id}/delete")
async def delete_node(
    draft_id: str,
    node_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete a node from the graph."""
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    graph_data = version_repo.parse_graph(version)
    nodes = graph_data.get("nodes", {})

    if node_id not in nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    if graph_data.get("start_node_id") == node_id:
        raise HTTPException(status_code=409, detail="Cannot delete the start node")

    del nodes[node_id]

    # Clean up dangling references
    for nid, n in nodes.items():
        if isinstance(n, dict):
            for choice in n.get("choices", []) or []:
                if isinstance(choice, dict) and choice.get("next_node_id") == node_id:
                    choice["next_node_id"] = None

    await _save_graph_as_new_version(session, draft_id, graph_data, f"Deleted node '{node_id}'")
    return RedirectResponse(url=f"/admin/draft/{draft_id}", status_code=303)


@admin_app.post("/draft/{draft_id}/nodes/{node_id}/choices/{choice_id}/edit")
async def edit_choice(
    draft_id: str,
    node_id: str,
    choice_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Edit a choice within a node."""
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    graph_data = version_repo.parse_graph(version)
    nodes = graph_data.get("nodes", {})

    if node_id not in nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    node = nodes[node_id]
    choices = node.get("choices", []) or []
    found = None
    for c in choices:
        if isinstance(c, dict) and c.get("id") == choice_id:
            found = c
            break
    if found is None:
        raise HTTPException(status_code=404, detail=f"Choice '{choice_id}' not found")

    form = await request.form()
    if "label" in form:
        found["label"] = form["label"]
    if "next_node_id" in form:
        found["next_node_id"] = form["next_node_id"] or None
    if "rationale" in form:
        found["rationale"] = form["rationale"]
    if "state_updates" in form:
        import json as _json
        try:
            found["state_updates"] = _json.loads(form["state_updates"]) if form["state_updates"].strip() else {}
        except _json.JSONDecodeError:
            pass  # ignore invalid JSON

    await _save_graph_as_new_version(
        session, draft_id, graph_data, f"Edited choice '{choice_id}' in node '{node_id}'"
    )
    return RedirectResponse(url=f"/admin/draft/{draft_id}", status_code=303)


@admin_app.post("/draft/{draft_id}/nodes/{node_id}/regenerate")
async def regenerate_node(
    draft_id: str,
    node_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Re-generate a single node's content via the authoring agent."""
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    graph_data = version_repo.parse_graph(version)
    nodes = graph_data.get("nodes", {})

    if node_id not in nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    node = nodes[node_id]
    form = await request.form()
    instruction = form.get("instruction", "")

    agent = get_authoring_agent(dummy=True)
    brief_context = {
        "title": node.get("title", ""),
        "genre": graph_data.get("genre", "science_fiction"),
        "tone": graph_data.get("tone", "dark_mystery"),
        "language": "de",
        "node_count": 1,
        "ending_count": 0,
        "branching_level": "none",
        "notes": instruction or f"Re-generate node '{node_id}'",
        "themes": [],
        "forbidden_content": [],
    }

    try:
        new_outline = await agent.generate_outline(brief_context)
        new_graph = await agent.generate_graph(new_outline)
        new_nodes = new_graph.get("nodes", {})
        replacement = None
        for nid, n in new_nodes.items():
            if isinstance(n, dict) and n.get("scene_goal"):
                replacement = n
                break
        if replacement:
            node["scene_goal"] = replacement.get("scene_goal", node.get("scene_goal", ""))
            node["characters"] = replacement.get("characters", node.get("characters", []))
            node["reveals"] = replacement.get("reveals", node.get("reveals", []))
            node["mood"] = replacement.get("mood", node.get("mood", ""))
            node["quality_notes"] = replacement.get("quality_notes", node.get("quality_notes", []))
            node["known_facts"] = replacement.get("known_facts", node.get("known_facts", []))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Regeneration failed: {exc}")

    await _save_graph_as_new_version(
        session, draft_id, graph_data,
        f"Re-generated node '{node_id}'" + (f": {instruction}" if instruction else ""),
        created_by="regenerate_agent",
    )
    return RedirectResponse(url=f"/admin/draft/{draft_id}", status_code=303)


# ═══════════════════════════════════════════════════════════════════════
# Story Enhancement (Multi-Pass Story Enhancement)
# ═══════════════════════════════════════════════════════════════════════


@admin_app.post("/draft/{draft_id}/enhance")
async def run_enhancement(
    draft_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Run story enhancement pass — Multi-Pass Story Enhancement.

    Accepts form data:
      mode: atmosphere | characters | choices | arc_expansion | thematic | critic_based
      instruction: optional free-text instruction
      target_act: optional, for arc_expansion
      add_node_count: optional, for arc_expansion
    """
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    if version is None:
        event_log.emit_error("enhancement", "enhance", f"No version found for draft {draft_id}", draft_id=draft_id)
        raise HTTPException(status_code=404, detail="No version found")

    form = await request.form()
    mode = form.get("mode", "atmosphere")
    instruction = form.get("instruction", "")
    target_act = None
    add_node_count = None
    try:
        target_act = int(form.get("target_act", "")) if form.get("target_act") else None
    except (ValueError, TypeError):
        pass
    try:
        add_node_count = int(form.get("add_node_count", "")) if form.get("add_node_count") else None
    except (ValueError, TypeError):
        pass

    from app.services.story_enhancement_agent import (
        ENHANCEMENT_MODES,
        StoryEnhancementAgent,
        StoryEnhancementError,
    )

    if mode not in ENHANCEMENT_MODES:
        raise HTTPException(status_code=422, detail=f"Invalid mode '{mode}'")

    graph_data = version_repo.parse_graph(version)
    outline = version_repo.parse_outline(version)

    # For critic_based mode, fetch latest review
    review_report = None
    if mode == "critic_based":
        review_data = await _get_review_data(session, draft_id)
        if not review_data:
            raise HTTPException(status_code=400, detail="Run review first (critic_based mode requires a review)")
        review_report = review_data

    event_log.emit_start(
        "enhancement",
        f"enhance_{mode}",
        f"Enhancing '{draft.title}' (mode: {mode})",
        draft_id=draft_id,
        detail={"mode": mode, "instruction": instruction[:200] if instruction else ""},
    )

    try:
        agent = StoryEnhancementAgent()
        result = await agent.enhance(
            graph=graph_data,
            mode=mode,
            instruction=instruction,
            review_report=review_report,
            target_act=target_act,
            add_node_count=add_node_count,
        )
    except StoryEnhancementError as exc:
        event_log.emit_error("enhancement", f"enhance_{mode}", f"Enhancement failed: {exc}", draft_id=draft_id)
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {exc}")

    new_graph = result.get("graph", graph_data)
    changes = result.get("changes", [])
    summary = result.get("summary", "")

    await version_repo.create(
        draft_id=draft_id,
        graph=new_graph,
        outline=outline,
        created_by="enhancement_agent",
        notes=f"Enhancement ({mode}): {summary}"[:500],
    )

    from app.models.enums import DraftStatus
    await draft_repo.update_status(draft_id, DraftStatus.NEEDS_REPAIR)
    await draft_repo.update_status(draft_id, DraftStatus.NEEDS_REVIEW)

    new_node_count = len(new_graph.get("nodes", {}))
    event_log.emit_done(
        "enhancement",
        f"enhance_{mode}",
        f"Enhancement ({mode}): {new_node_count} nodes, {len(changes)} changes. {summary[:80]}",
        draft_id=draft_id,
        detail={"mode": mode, "node_count": new_node_count, "changes_count": len(changes), "summary": summary[:200]},
    )

    return RedirectResponse(url=f"/admin/draft/{draft_id}", status_code=303)


@admin_app.post("/draft/{draft_id}/nodes/{node_id}/split")
async def split_node(
    draft_id: str,
    node_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Split a node into multiple sub-nodes via form submission.

    Accepts form fields:
      split_text: the full text to split (from the editor)
      split_positions: comma-separated character positions
      titles: optional comma-separated titles for sub-nodes
    """
    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    graph_data = version_repo.parse_graph(version)
    nodes = graph_data.get("nodes", {})

    if node_id not in nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    node = nodes[node_id]
    form = await request.form()

    split_text = form.get("split_text", "")
    if not split_text:
        split_text = node.get("scene_text", "") or node.get("scene_goal", "")
    if not split_text or not split_text.strip():
        raise HTTPException(status_code=400, detail="No text available to split")

    positions_str = form.get("split_positions", "")
    if not positions_str.strip():
        raise HTTPException(status_code=400, detail="Split positions required")
    try:
        split_points = sorted(set(int(p.strip()) for p in positions_str.split(",") if p.strip()))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid split positions — must be integers")

    titles_str = form.get("titles", "")
    titles = [t.strip() for t in titles_str.split(",") if t.strip()] if titles_str.strip() else None

    # Validate positions
    for sp in split_points:
        if sp < 0 or sp >= len(split_text):
            raise HTTPException(status_code=400, detail=f"Split point {sp} out of range")

    expected_count = len(split_points) + 1
    if titles and len(titles) != expected_count:
        raise HTTPException(status_code=400, detail=f"Expected {expected_count} titles, got {len(titles)}")

    if not titles:
        base_title = node.get("title", node_id)
        titles = [base_title] + [f"{base_title} (Teil {i+2})" for i in range(len(split_points))]

    # Split text into segments
    segments = []
    prev = 0
    for sp in split_points:
        segments.append(split_text[prev:sp].strip())
        prev = sp
    segments.append(split_text[prev:].strip())

    # Save original choices
    original_choices = node.get("choices", []) or []
    original_act = node.get("act", 1)
    original_location = node.get("location", "")
    original_mood = node.get("mood", "")
    original_characters = node.get("characters", [])

    # Update original node
    node["scene_text"] = segments[0]
    node["title"] = titles[0]
    node["choices"] = []

    # Create sub-nodes
    import uuid as _uuid
    new_node_ids = [node_id]
    for i, segment in enumerate(segments[1:], start=1):
        sub_id = f"node_{_uuid.uuid4().hex[:8]}"
        nodes[sub_id] = {
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
        new_node_ids.append(sub_id)

    # Chain connections
    for i in range(len(new_node_ids) - 1):
        nodes[new_node_ids[i]]["choices"] = [{
            "id": f"continue_{i+1}",
            "label": "Weiter" if i < len(new_node_ids) - 2 else "Fortsetzen",
            "next_node_id": new_node_ids[i + 1],
            "state_updates": {},
            "rationale": "Direkte Fortsetzung der Szene",
        }]
    nodes[new_node_ids[-1]]["choices"] = original_choices

    await _save_graph_as_new_version(
        session, draft_id, graph_data,
        f"Split node '{node_id}' into {len(new_node_ids)} sub-nodes",
        created_by="manual_edit",
    )
    return RedirectResponse(url=f"/admin/draft/{draft_id}", status_code=303)


@admin_app.get("/story-templates", response_class=JSONResponse)
async def list_story_templates_api():
    """List all available story fragment templates."""
    from app.story.story_templates import list_templates, list_categories
    return JSONResponse(content={
        "templates": list_templates(),
        "categories": list_categories(),
    })


@admin_app.get("/story-templates/{template_id}", response_class=JSONResponse)
async def get_story_template_api(template_id: str):
    """Get a single story template by ID."""
    from app.story.story_templates import get_template
    tpl = get_template(template_id)
    if tpl is None:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")
    return JSONResponse(content=tpl)


@admin_app.post("/draft/{draft_id}/nodes/{node_id}/apply-template")
async def apply_template_to_node(
    draft_id: str,
    node_id: str,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Apply a story template to a node via form submission."""
    from app.story.story_templates import apply_template_to_node as _apply

    draft, version, draft_repo, version_repo = await _get_draft_and_version(session, draft_id)
    graph_data = version_repo.parse_graph(version)
    nodes = graph_data.get("nodes", {})

    if node_id not in nodes:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")

    form = await request.form()
    template_id = form.get("template_id", "")
    if not template_id:
        raise HTTPException(status_code=400, detail="template_id required")

    tpl_fields = _apply(template_id)
    if tpl_fields is None:
        raise HTTPException(status_code=404, detail=f"Template '{template_id}' not found")

    node = nodes[node_id]
    for field, value in tpl_fields.items():
        if value is not None and value != "":
            old_val = node.get(field)
            if isinstance(value, list) and isinstance(old_val, list) and old_val:
                merged = list(old_val)
                for item in value:
                    if item not in merged:
                        merged.append(item)
                node[field] = merged
            else:
                node[field] = value

    await _save_graph_as_new_version(
        session, draft_id, graph_data,
        f"Applied template '{template_id}' to node '{node_id}'",
        created_by="manual_edit",
    )
    return RedirectResponse(url=f"/admin/draft/{draft_id}", status_code=303)


@admin_app.get("/draft/{draft_id}/diff", response_class=JSONResponse)
async def get_graph_diff(
    draft_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Get diff between the last two versions of a draft."""
    from app.story.graph_diff import compute_graph_diff
    version_repo = StoryDraftVersionRepository(session)
    versions = await version_repo.list_by_draft(draft_id)

    if not versions:
        return JSONResponse(content={"error": "No versions found"})

    if len(versions) >= 2:
        old_graph = version_repo.parse_graph(versions[-2])
        new_graph = version_repo.parse_graph(versions[-1])
        diff = compute_graph_diff(old_graph, new_graph)
        return JSONResponse(content={
            "old_version": versions[-2].version_number,
            "new_version": versions[-1].version_number,
            "diff": diff,
        })
    else:
        new_graph = version_repo.parse_graph(versions[-1])
        diff = compute_graph_diff({}, new_graph)
        return JSONResponse(content={
            "old_version": 0,
            "new_version": versions[-1].version_number,
            "diff": diff,
        })


@admin_app.get("/draft/{draft_id}/versions", response_class=JSONResponse)
async def list_versions_api(
    draft_id: str,
    session: AsyncSession = Depends(get_session),
):
    """List all versions with incremental diffs."""
    from app.story.graph_diff import compute_graph_diff
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
        })
        prev_graph = graph

    return JSONResponse(content={"draft_id": draft_id, "versions": result})
