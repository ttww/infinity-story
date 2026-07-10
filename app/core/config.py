"""Core configuration for Infinity Story."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from env vars / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    # App
    app_name: str = "Infinity Story"
    app_env: Literal["development", "staging", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8650
    debug: bool = True

    # Database
    database_path: str = "data/infinity_story.db"
    database_url: str = ""  # if set, overrides database_path

    # Ports
    runtime_api_port: int = 8650
    admin_ui_port: int = 8750
    admin_api_port: int = 8850
    whatsapp_mock_port: int = 8950

    # LLM
    llm_provider: Literal["mock", "openai", "ollama", "azure_openai"] = "mock"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # Azure OpenAI
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = ""
    azure_openai_api_version: str = "2024-02-15-preview"

    # LLM operational parameters
    llm_max_tokens: int = 2048
    llm_max_input_tokens: int = 6000
    llm_temperature: float = 0.7
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2
    llm_retry_backoff_base: float = 1.0
    llm_daily_budget_usd: float = 0.0
    llm_cost_per_1k_input: float = 0.0
    llm_cost_per_1k_output: float = 0.0

    # Moderation
    moderation_enabled: bool = True

    # Channel
    default_channel: Literal["whatsapp_mock", "cli_dev", "rest_dev"] = "whatsapp_mock"

    # Quality thresholds (Spec §15)
    min_quality_score: float = 7.0
    min_node_count: int = 15
    min_ending_count: int = 2
    max_authoring_rounds: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Singleton
settings = get_settings()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SCENARIOS_DIR = BASE_DIR / "app" / "story" / "scenarios"

