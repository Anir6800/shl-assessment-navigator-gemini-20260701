"""Filesystem paths used across the project."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
DOCS_DIR = PROJECT_ROOT / "docs"
WORK_DIR = PROJECT_ROOT / "work"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def ensure_project_directories() -> None:
    """Create the standard project directories if they do not exist."""

    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        VECTOR_STORE_DIR,
        DOCS_DIR,
        WORK_DIR,
        OUTPUTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
