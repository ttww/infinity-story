# Interactive Story System

WhatsApp-basiertes interaktives LLM-Story-System mit FastAPI.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
uvicorn app.main:app --reload
```

## Tests

```bash
pytest
```

## Struktur

Siehe Spec Kapitel 17 (`llm_interactive_story_system_requirements.md`).
