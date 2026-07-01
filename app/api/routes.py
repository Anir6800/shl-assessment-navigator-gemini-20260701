"""HTTP routes for the SHL assessment assistant."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from app.api.dependencies import get_chat_service
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService


router = APIRouter()
UI_INDEX_PATH = Path(__file__).resolve().parents[1] / "ui" / "index.html"


@lru_cache(maxsize=1)
def _load_ui_html() -> str:
    return UI_INDEX_PATH.read_text(encoding="utf-8")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home() -> HTMLResponse:
    """Serve the single-page UI."""

    return HTMLResponse(_load_ui_html())


@router.get("/health")
async def health() -> dict[str, str]:
    """Readiness probe."""

    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Process a stateless conversation transcript and return the next response."""

    return await asyncio.to_thread(chat_service.handle, request.messages)
