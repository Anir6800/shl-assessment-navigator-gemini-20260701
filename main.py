"""SHL assessment recommender entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.routes import router
from app.core.settings import get_settings
from app.retrieval.engine import AssessmentRetrievalEngine
from app.scraper.catalog_scraper import scrape_catalog
from app.services.chat_service import ChatService


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize cached catalog, index, and chat service on startup."""

    dataset = scrape_catalog(force_refresh=False)
    retriever = AssessmentRetrievalEngine.from_dataset(
        dataset,
        index_dir=Path(settings.vector_store_dir),
    )
    app.state.settings = settings
    app.state.chat_service = ChatService.from_settings(settings=settings, retriever=retriever)
    yield


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.include_router(router)
