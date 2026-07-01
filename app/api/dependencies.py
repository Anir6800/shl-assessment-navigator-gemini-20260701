"""FastAPI dependencies for application services."""

from __future__ import annotations

from fastapi import Request

from app.core.settings import get_settings
from app.retrieval.engine import AssessmentRetrievalEngine
from app.scraper.catalog_scraper import scrape_catalog
from app.services.chat_service import ChatService
from pathlib import Path


def get_chat_service(request: Request) -> ChatService:
    """Return the preloaded chat service from application state."""

    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        settings = getattr(request.app.state, "settings", None) or get_settings()
        dataset = scrape_catalog(force_refresh=False)
        retriever = AssessmentRetrievalEngine.from_dataset(
            dataset,
            index_dir=Path(settings.vector_store_dir),
        )
        service = ChatService.from_settings(settings=settings, retriever=retriever)
        request.app.state.settings = settings
        request.app.state.chat_service = service
    return service
