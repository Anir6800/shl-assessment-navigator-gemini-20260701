"""Smoke tests for retrieval quality and catalog recall at 10."""

from __future__ import annotations


def _top_names(matches):
    return [match.item.name for match in matches]


def test_personality_recall_at_10(engine):
    names = _top_names(engine.search("personality test for managers", top_k=10))
    assert "Occupational Personality Questionnaire OPQ32r" in names


def test_java_recall_at_10(engine):
    names = _top_names(engine.search("Java developer assessment", top_k=10))
    assert any(name in names for name in ["Java 8 (New)", "Core Java (Advanced Level) (New)", "Java Frameworks (New)"])


def test_cognitive_recall_at_10(engine):
    names = _top_names(engine.search("cognitive ability test for graduates", top_k=10))
    assert any(name in names for name in ["SHL Verify Interactive G+", "Verify - G+"])
