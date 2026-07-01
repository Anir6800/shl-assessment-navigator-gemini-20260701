"""Pydantic schemas for the chat API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    """A single message in the stateless conversation history."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def _strip_content(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("content cannot be empty")
        return value


class ChatRequest(BaseModel):
    """POST /chat request body."""

    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(default_factory=list)

    @field_validator("messages")
    @classmethod
    def _non_empty(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if not value:
            raise ValueError("messages cannot be empty")
        return value


class Recommendation(BaseModel):
    """A single assessment recommendation."""

    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    """POST /chat response body."""

    model_config = ConfigDict(extra="forbid")

    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool
