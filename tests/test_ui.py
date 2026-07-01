"""UI smoke tests."""

from __future__ import annotations


def test_ui_root_serves_html(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "SHL Assessment Navigator" in response.text
    assert "textarea" in response.text
