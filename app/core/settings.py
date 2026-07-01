"""Application settings and environment variable loading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from app.retrieval.vector_store import DEFAULT_INDEX_DIR
from app.scraper.catalog_scraper import DEFAULT_SOURCE_URL


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(slots=True)
class Settings:
    """Runtime settings derived from environment variables."""

    app_name: str = "shl-assessment-recommender"
    app_version: str = "1.0.0"
    catalog_source_url: str = DEFAULT_SOURCE_URL
    embedding_backend: str = os.getenv("SHL_EMBEDDING_BACKEND", "auto")
    llm_provider: str = os.getenv("SHL_LLM_PROVIDER", "auto")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    gemini_model: str = os.getenv("SHL_GEMINI_MODEL", "gemini-2.5-flash")
    anthropic_model: str = os.getenv("SHL_ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
    openrouter_model: str = os.getenv("SHL_OPENROUTER_MODEL", "openrouter/auto")
    max_recommendations: int = _env_int("SHL_MAX_RECOMMENDATIONS", 10)
    retrieval_candidate_pool: int = _env_int("SHL_RETRIEVAL_CANDIDATE_POOL", 50)
    cache_ttl_hours: int = _env_int("SHL_CACHE_TTL_HOURS", 24)
    vector_store_dir: str = os.getenv("SHL_VECTOR_STORE_DIR", str(DEFAULT_INDEX_DIR))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings object."""

    return Settings()
