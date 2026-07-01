"""Prompt loading utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


PROMPT_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=32)
def load_prompt(name: str) -> str:
    """Load a prompt template by basename without hardcoding prompt text."""

    path = PROMPT_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def prompt_path(name: str) -> Path:
    """Return the on-disk path for a prompt template."""

    return PROMPT_DIR / f"{name}.txt"
