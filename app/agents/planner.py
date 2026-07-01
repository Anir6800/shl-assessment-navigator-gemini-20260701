"""Conversation planning for the SHL assessment assistant."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from app.retrieval.engine import RetrievalFilters, infer_categories, infer_duration_preference, infer_job_levels
from app.schemas.chat import ChatMessage
from app.utils.text import normalize_whitespace


PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore (?:all |any )?previous instructions", re.IGNORECASE),
    re.compile(r"reveal (?:the )?system prompt", re.IGNORECASE),
    re.compile(r"show (?:me )?(?:the )?system prompt", re.IGNORECASE),
    re.compile(r"developer message", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"prompt injection", re.IGNORECASE),
    re.compile(r"sql injection", re.IGNORECASE),
)

OFF_TOPIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\blegal advice\b", re.IGNORECASE),
    re.compile(r"\bemployment law\b", re.IGNORECASE),
    re.compile(r"\bhow do i hire\b", re.IGNORECASE),
    re.compile(r"\bhow should i hire\b", re.IGNORECASE),
    re.compile(r"\bhiring advice\b", re.IGNORECASE),
    re.compile(r"\binterview tips\b", re.IGNORECASE),
    re.compile(r"\bwhat questions should i ask\b", re.IGNORECASE),
    re.compile(r"\bsalary negotiation\b", re.IGNORECASE),
)

ROLE_HINTS: tuple[str, ...] = (
    "developer",
    "engineer",
    "manager",
    "analyst",
    "consultant",
    "sales",
    "customer service",
    "contact center",
    "call center",
    "retail",
    "finance",
    "accounting",
    "technical support",
    "data",
    "scientist",
    "product manager",
    "project manager",
    "leader",
    "executive",
    "director",
    "supervisor",
    "nurse",
    "healthcare",
)

REFINEMENT_HINTS: tuple[str, ...] = (
    "actually",
    "instead",
    "also",
    "include",
    "add",
    "prefer",
    "more",
    "less",
    "change",
    "remove",
    "update",
    "refine",
)

COMPARE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?P<left>.+?)\s+(?:vs|versus|against)\s+(?P<right>.+)", re.IGNORECASE),
    re.compile(r"compare\s+(?P<left>.+?)\s+(?:and|vs|versus|against)\s+(?P<right>.+)", re.IGNORECASE),
    re.compile(r"difference between\s+(?P<left>.+?)\s+and\s+(?P<right>.+)", re.IGNORECASE),
)


class Intent(str, Enum):
    """High-level chat intent."""

    clarify = "clarify"
    recommend = "recommend"
    compare = "compare"
    refuse = "refuse"


@dataclass(slots=True)
class PlanResult:
    """Planner output consumed by the chat service."""

    intent: Intent
    query_text: str
    latest_user_text: str
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    compare_targets: tuple[str, str] | None = None
    clarification_question: str | None = None
    refusal_reason: str | None = None
    is_refinement: bool = False


class ConversationPlanner:
    """Infer intent and retrieval constraints from a stateless transcript."""

    def plan(self, messages: Sequence[ChatMessage]) -> PlanResult:
        user_messages = [message.content for message in messages if message.role == "user"]
        latest_user_text = normalize_whitespace(user_messages[-1] if user_messages else "")
        query_text = normalize_whitespace(" ".join(user_messages))

        refusal_reason = self._detect_refusal_reason(query_text)
        if refusal_reason:
            return PlanResult(
                intent=Intent.refuse,
                query_text=query_text,
                latest_user_text=latest_user_text,
                refusal_reason=refusal_reason,
            )

        compare_targets = self._extract_compare_targets(latest_user_text)
        if compare_targets is not None:
            return PlanResult(
                intent=Intent.compare,
                query_text=query_text,
                latest_user_text=latest_user_text,
                compare_targets=compare_targets,
                filters=self._build_filters(query_text),
            )

        is_refinement = self._is_refinement(latest_user_text)
        filters = self._build_filters(query_text)
        if self._needs_clarification(query_text, filters):
            return PlanResult(
                intent=Intent.clarify,
                query_text=query_text,
                latest_user_text=latest_user_text,
                filters=filters,
                clarification_question=self._build_clarification_question(query_text, filters),
                is_refinement=is_refinement,
            )

        return PlanResult(
            intent=Intent.recommend,
            query_text=query_text,
            latest_user_text=latest_user_text,
            filters=filters,
            is_refinement=is_refinement,
        )

    def _build_filters(self, text: str) -> RetrievalFilters:
        categories = infer_categories(text)
        job_levels = infer_job_levels(text)
        duration_preference = infer_duration_preference(text)
        needs_remote = self._detect_remote_preference(text)
        needs_adaptive = self._detect_adaptive_preference(text)
        explicit_names = self._extract_explicit_names(text)
        return RetrievalFilters(
            categories=categories,
            job_levels=job_levels,
            duration_preference=duration_preference,
            needs_remote=needs_remote,
            needs_adaptive=needs_adaptive,
            explicit_names=explicit_names,
        )

    @staticmethod
    def _detect_remote_preference(text: str) -> bool | None:
        lowered = normalize_whitespace(text).lower()
        if any(token in lowered for token in ("remote", "online", "virtual", "anywhere")):
            return True
        if any(token in lowered for token in ("onsite", "on-site", "in person")):
            return False
        return None

    @staticmethod
    def _detect_adaptive_preference(text: str) -> bool | None:
        lowered = normalize_whitespace(text).lower()
        if "adaptive" in lowered:
            return True
        if "non-adaptive" in lowered or "not adaptive" in lowered:
            return False
        return None

    @staticmethod
    def _extract_explicit_names(text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+.-]*", text)
        explicit: list[str] = []
        for token in tokens:
            if token.isupper() and len(token) >= 2:
                explicit.append(token)
            elif any(char.isdigit() for char in token):
                explicit.append(token)
        return explicit

    def _needs_clarification(self, query_text: str, filters: RetrievalFilters) -> bool:
        signal_count = 0
        signal_count += len(filters.categories)
        signal_count += len(filters.job_levels)
        signal_count += len(filters.explicit_names)
        if filters.duration_preference:
            signal_count += 1
        if filters.needs_remote is not None:
            signal_count += 1
        if filters.needs_adaptive is not None:
            signal_count += 1
        signal_count += self._count_role_hints(query_text)

        if signal_count == 0:
            return True
        if len(query_text.split()) <= 2 and signal_count < 2:
            return True
        return False

    @staticmethod
    def _count_role_hints(text: str) -> int:
        lowered = normalize_whitespace(text).lower()
        return sum(1 for hint in ROLE_HINTS if hint in lowered)

    def _build_clarification_question(self, query_text: str, filters: RetrievalFilters) -> str:
        missing_parts: list[str] = []
        if not filters.categories:
            missing_parts.append("assessment type (technical, personality, cognitive, or simulation)")
        if not filters.job_levels:
            missing_parts.append("seniority or experience level")
        if self._count_role_hints(query_text) == 0:
            missing_parts.append("role or job family")

        if not missing_parts:
            return "What should I optimize for?"
        if len(missing_parts) == 1:
            return f"What {missing_parts[0]} should I optimize for?"
        if len(missing_parts) == 2:
            return f"What {missing_parts[0]} and {missing_parts[1]} should I optimize for?"
        return f"What {missing_parts[0]}, {missing_parts[1]}, and {missing_parts[2]} should I optimize for?"

    @staticmethod
    def _detect_refusal_reason(text: str) -> str | None:
        lowered = normalize_whitespace(text).lower()
        if any(pattern.search(lowered) for pattern in PROMPT_INJECTION_PATTERNS):
            return "prompt_injection"
        if any(pattern.search(lowered) for pattern in OFF_TOPIC_PATTERNS):
            return "off_topic"
        if "legal" in lowered or "lawyer" in lowered or "liability" in lowered:
            return "legal"
        return None

    @staticmethod
    def _is_refinement(text: str) -> bool:
        lowered = normalize_whitespace(text).lower()
        return any(token in lowered for token in REFINEMENT_HINTS)

    @staticmethod
    def _extract_compare_targets(text: str) -> tuple[str, str] | None:
        candidate = normalize_whitespace(text)
        for pattern in COMPARE_PATTERNS:
            match = pattern.search(candidate)
            if not match:
                continue
            left = ConversationPlanner._clean_compare_target(match.group("left"))
            right = ConversationPlanner._clean_compare_target(match.group("right"))
            if left and right:
                return left, right
        return None

    @staticmethod
    def _clean_compare_target(value: str) -> str:
        cleaned = normalize_whitespace(value)
        cleaned = re.sub(r"^(?:what is the difference between|difference between|compare|which is better|compare the|between)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\b(?:please|assessment|test|product|catalog)\b", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip(" ?.,;")
        return cleaned
