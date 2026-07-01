"""Regression tests for Gemini reply normalization."""

from __future__ import annotations

from app.core.settings import Settings
from app.schemas.chat import ChatMessage
from app.services.chat_service import ChatService


class FakeGeminiProvider:
    name = "gemini"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return """```json
{
  "reply": "Are you also looking for assessments that measure professional skills like communication or collaboration, in addition to technical Java knowledge?",
  "recommendations": [],
  "end_of_conversation": false
}
```"""


def test_gemini_json_reply_is_normalized(engine):
    settings = Settings()
    service = ChatService.from_settings(settings=settings, retriever=engine, llm_provider=FakeGeminiProvider())

    response = service.handle(
        [
            ChatMessage(
                role="user",
                content="Hiring a Java developer who works with stakeholders, mid-level.",
            )
        ]
    )

    assert response.reply == (
        "Are you also looking for assessments that measure professional skills like communication "
        "or collaboration, in addition to technical Java knowledge?"
    )
    assert len(response.recommendations) == 10
    assert response.end_of_conversation is True
