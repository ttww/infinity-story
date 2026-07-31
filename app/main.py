"""FastAPI application entry point.

Runtime Story System: processes user messages, executes published stories.
"""

import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select, func

from app.core.config import settings, DATA_DIR, get_settings
from app.persistence.database import get_session
from sqlalchemy.ext.asyncio import AsyncSession

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB on startup, close on shutdown."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from app.persistence.database import init_db
    await init_db()

    # Register forced-exit signal handler for fast reloads
    # If uvicorn hangs during shutdown (open connections), this
    # kills the process after a timeout instead of blocking forever.
    def _force_exit(signum, frame):
        logger.warning("Forced exit on signal %d — skipping graceful shutdown", signum)
        os._exit(0)

    signal.signal(signal.SIGTERM, _force_exit)
    signal.signal(signal.SIGINT, _force_exit)

    yield


app = FastAPI(
    title=settings.app_name,
    description="WhatsApp-basiertes interaktives LLM-Story-System",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Dashboard / Startseite — kombiniert Admin + Chat + Szenarien.

    Shows: (1) Link 'Story spielen' → /chat/, (2) Link 'Story erstellen' → /admin/,
    (3) List of published scenarios with 'Spielen' buttons,
    (4) Statistics: drafts, published scenarios, active sessions.
    """
    from app.models.story_draft import StoryDraft
    from app.models.published_scenario import PublishedScenario
    from app.models.story_session import StorySession
    from app.models.enums import DraftStatus

    # ── Statistics ──
    # Draft count (non-published drafts)
    draft_stmt = (
        select(func.count(StoryDraft.id))
        .where(StoryDraft.status != DraftStatus.PUBLISHED.value)
    )
    drafts_count = (await session.execute(draft_stmt)).scalar() or 0

    # Published scenarios count
    pub_stmt = select(func.count(PublishedScenario.id))
    published_count = (await session.execute(pub_stmt)).scalar() or 0

    # Active sessions count
    active_stmt = (
        select(func.count(StorySession.id))
        .where(StorySession.status == "running")
    )
    active_sessions = (await session.execute(active_stmt)).scalar() or 0

    # ── Scenarios list (file-based + DB-published) ──
    from app.story.scenario_loader import list_all_scenarios
    scenarios = await list_all_scenarios(session)

    return templates.TemplateResponse(request, "dashboard.html", {
        "stats": {
            "drafts": drafts_count,
            "published": published_count,
            "active_sessions": active_sessions,
        },
        "scenarios": scenarios,
        "is_mock_mode": get_settings().llm_provider == "mock",
    })


@app.get("/health")
async def health():
    return {"status": "ok"}


# Import and include routers
from app.api.message_routes import router as message_router
from app.api.session_routes import router as session_router
from app.api.scenario_routes import router as scenario_router
from app.api.admin_story_draft_routes import router as admin_router
from app.api.incremental_edit_routes import router as incremental_router
from app.api.enhancement_routes import router as enhancement_router

app.include_router(message_router)
app.include_router(session_router)
app.include_router(scenario_router)
app.include_router(admin_router)
app.include_router(incremental_router)
app.include_router(enhancement_router)

# Mount Admin UI as a sub-application under /admin prefix (Spec §8)
from app.admin_ui.app import admin_app
app.mount("/admin", admin_app)

# Mount Runtime Chat UI under /chat prefix (Spec §13)
from app.runtime_ui.app import runtime_app
app.mount("/chat", runtime_app)
