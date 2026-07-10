"""Runtime Story Chat UI — interactive chat interface for end users.

Serves a WhatsApp-style chat page at /chat/ where users can:
  - Select a published scenario
  - Interact with the story via text input or choice buttons
  - See story responses in chat bubbles (user right, story left)
  - View session state and choices as clickable buttons

Connects to the runtime API (POST /api/message, GET /api/scenarios)
via fetch() from the browser.

Spec §13.2 (Runtime UI).
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

runtime_app = FastAPI(title="Infinity Story Runtime", docs_url=None, redoc_url=None)

if STATIC_DIR.exists():
    runtime_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@runtime_app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Render the chat page shell.

    All interactivity is client-side: the page fetches /api/scenarios
    and posts to /api/message via fetch().  We pass the API base URL
    so the frontend knows where to call (same origin = relative URLs).
    """
    return templates.TemplateResponse(request, "chat.html", {
        "api_base": "",  # same origin — relative URLs
    })
