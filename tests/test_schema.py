"""Schema compliance tests for the API contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest, ChatResponse, Recommendation


def test_chat_request_schema():
    request = ChatRequest.model_validate(
        {"messages": [{"role": "user", "content": "I need an assessment."}]}
    )
    assert len(request.messages) == 1
    assert request.messages[0].role == "user"


def test_chat_response_schema():
    response = ChatResponse.model_validate(
        {
            "reply": "Hello",
            "recommendations": [
                {"name": "OPQ32r", "url": "https://www.shl.com/x", "test_type": "Personality & Behavior"}
            ],
            "end_of_conversation": True,
        }
    )
    assert response.reply == "Hello"
    assert response.recommendations[0].name == "OPQ32r"


def test_recommendation_schema_rejects_extra_fields():
    with pytest.raises(ValidationError):
        Recommendation.model_validate(
            {
                "name": "OPQ32r",
                "url": "https://www.shl.com/x",
                "test_type": "Personality & Behavior",
                "unexpected": "value",
            }
        )
