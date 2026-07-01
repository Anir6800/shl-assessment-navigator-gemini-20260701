"""Shared pytest fixtures for the SHL assistant test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.retrieval.engine import AssessmentRetrievalEngine
from app.scraper.catalog_scraper import scrape_catalog
from main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def engine() -> AssessmentRetrievalEngine:
    return AssessmentRetrievalEngine.from_dataset(scrape_catalog(force_refresh=False))
