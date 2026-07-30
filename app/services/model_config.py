"""Model configuration — per-use-case model selection (Spec §5.6).

Stores which model/provider to use for each task type (review, repair,
scene generation, outline generation, etc.). Persisted as a JSON file
in the project data directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Known use cases ────────────────────────────────────────────────

USE_CASES = {
    "review": "Review + Reparatur (Kritiker)",
    "scene_generation": "Szenen-Generierung (Runtime-LLM)",
    "outline_generation": "Outline-Generierung (Autor)",
    "graph_generation": "Graph-Generierung (Autor)",
    "repair": "Reparatur-Agent (Patch)",
    "enhancement": "Story-Vertiefung (Enhancement)",
}

# ── Schema ─────────────────────────────────────────────────────────

class ModelAssignment(BaseModel):
    """A model assignment for one use case."""
    provider: str = Field(
        default="openrouter", pattern=r"^(openrouter|openai)$",
        description="Provider: 'openrouter' oder 'openai'",
    )
    model: str = Field(
        default="deepseek/deepseek-v4-flash", max_length=128,
        description="Model-Name/ID beim Provider",
    )
    notes: str = Field(default="", max_length=256)


class ModelConfig(BaseModel):
    """Full model configuration for all use cases."""
    assignments: dict[str, ModelAssignment] = Field(
        default_factory=lambda: {
            "review": ModelAssignment(
                provider="openrouter", model="deepseek/deepseek-v4-flash",
            ),
            "scene_generation": ModelAssignment(
                provider="openrouter", model="deepseek/deepseek-v4-flash",
            ),
            "outline_generation": ModelAssignment(
                provider="openrouter", model="deepseek/deepseek-v4-flash",
            ),
            "graph_generation": ModelAssignment(
                provider="openrouter", model="deepseek/deepseek-v4-flash",
            ),
            "repair": ModelAssignment(
                provider="openrouter", model="deepseek/deepseek-v4-flash",
            ),
            "enhancement": ModelAssignment(
                provider="openrouter", model="deepseek/deepseek-v4-flash",
            ),
        },
    )


# ── Storage ────────────────────────────────────────────────────────

CONFIG_FILE = "data/model_config.json"


def get_config_path() -> Path:
    """Return the absolute path to the model config JSON file."""
    from app.core.config import BASE_DIR
    return BASE_DIR / CONFIG_FILE


def load_config() -> ModelConfig:
    """Load model config from disk, or return defaults."""
    path = get_config_path()
    if not path.exists():
        logger.info("No model config found at %s, using defaults", path)
        return ModelConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ModelConfig(**data)
    except Exception as exc:
        logger.warning("Failed to load model config: %s", exc)
        return ModelConfig()


def save_config(config: ModelConfig) -> None:
    """Save model config to disk."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        config.model_dump_json(indent=2, exclude={"assignments": {"__all__": {"notes"}}}),
        encoding="utf-8",
    )
    logger.info("Model config saved to %s", path)


# ── Known OpenRouter models helper ─────────────────────────────────

KNOWN_OPENROUTER_MODELS = [
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-chat",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/o3-mini",
    "openai/o4-mini",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "z-ai/glm-5.2",
    "google/gemini-2.5-pro",
    "google/gemini-2.0-flash",
    "meta-llama/llama-4-maverick",
    "qwen/qwq-32b",
    "mistralai/mistral-large",
    "x-ai/grok-2",
]

KNOWN_OPENAI_MODELS = [
    "gpt-5.x",
    "gpt-5.0",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
    "o3-mini",
    "o4-mini",
]