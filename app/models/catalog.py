"""Normalized catalog models for SHL assessment records."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.utils.text import coerce_bool, join_text, normalize_whitespace, split_comma_list


RECOMMENDABLE_CATEGORIES: tuple[str, ...] = (
    "Knowledge & Skills",
    "Personality & Behavior",
    "Ability & Aptitude",
    "Biodata & Situational Judgment",
    "Assessment Exercises",
    "Simulations",
    "Competencies",
)

OUT_OF_SCOPE_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\breport\b", re.IGNORECASE),
    re.compile(r"\bguide\b", re.IGNORECASE),
    re.compile(r"\btraining\b", re.IGNORECASE),
    re.compile(r"\bsolution\b", re.IGNORECASE),
    re.compile(r"\bfeedback\b", re.IGNORECASE),
    re.compile(r"\b360\s*°?\b", re.IGNORECASE),
    re.compile(r"\bmfs\b", re.IGNORECASE),
    re.compile(r"profiler cards", re.IGNORECASE),
    re.compile(r"job profiling", re.IGNORECASE),
    re.compile(r"development tips", re.IGNORECASE),
    re.compile(r"interpretation report", re.IGNORECASE),
    re.compile(r"narrative report", re.IGNORECASE),
    re.compile(r"profile report", re.IGNORECASE),
    re.compile(r"standard report", re.IGNORECASE),
    re.compile(r"group report", re.IGNORECASE),
    re.compile(r"skills development report", re.IGNORECASE),
)

PRIMARY_TEST_TYPE_ORDER: tuple[str, ...] = (
    "Personality & Behavior",
    "Knowledge & Skills",
    "Ability & Aptitude",
    "Biodata & Situational Judgment",
    "Simulations",
    "Assessment Exercises",
    "Competencies",
)


def build_alias_text(categories: list[str]) -> str:
    """Expand SHL taxonomy labels into recruiter-friendly search aliases."""

    aliases: list[str] = []
    if "Personality & Behavior" in categories:
        aliases.extend(["personality test", "behavioral assessment", "behavioral style"])
    if "Knowledge & Skills" in categories:
        aliases.extend(["knowledge test", "skills test", "technical skills"])
    if "Ability & Aptitude" in categories:
        aliases.extend(["cognitive ability", "aptitude test", "reasoning test"])
    if "Biodata & Situational Judgment" in categories:
        aliases.extend(["sjt", "situational judgment", "biodata"])
    if "Simulations" in categories:
        aliases.extend(["simulation", "work simulation", "job simulation"])
    if "Assessment Exercises" in categories:
        aliases.extend(["assessment exercise", "exercise"])
    if "Competencies" in categories:
        aliases.extend(["competency assessment", "ucf", "competency framework"])
    return " ".join(aliases)


def infer_test_type(categories: list[str]) -> str:
    """Select a stable human-readable test type for a catalog item."""

    for category in PRIMARY_TEST_TYPE_ORDER:
        if category in categories:
            return category
    if categories:
        return " / ".join(categories)
    return "Unknown"


def infer_recommendable(name: str, categories: list[str]) -> bool:
    """Return True when the item looks like an individual assessment."""

    category_match = any(category in RECOMMENDABLE_CATEGORIES for category in categories)
    if not category_match:
        return False

    normalized_name = normalize_whitespace(name)
    return not any(pattern.search(normalized_name) for pattern in OUT_OF_SCOPE_NAME_PATTERNS)


def infer_duration_minutes(duration: str) -> int | None:
    """Extract the approximate duration in minutes when it is present."""

    match = re.search(r"(\d+)", duration or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def parse_scraped_at(value: object) -> datetime:
    """Parse ISO timestamps from the source payload."""

    if isinstance(value, datetime):
        return value
    text = normalize_whitespace(value)
    if not text:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


class CatalogItem(BaseModel):
    """Normalized SHL catalog record."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    entity_id: str
    name: str
    link: str
    scraped_at: datetime
    job_levels: list[str] = Field(default_factory=list)
    job_levels_raw: str = ""
    languages: list[str] = Field(default_factory=list)
    languages_raw: str = ""
    duration: str = ""
    duration_raw: str = ""
    duration_minutes: int | None = None
    status: str = ""
    remote: bool = False
    adaptive: bool = False
    description: str = ""
    keys: list[str] = Field(default_factory=list)
    competencies: list[str] = Field(default_factory=list)
    test_type: str = "Unknown"
    recommendable: bool = False
    search_text: str = ""

    @classmethod
    def from_source(cls, raw: dict[str, Any]) -> "CatalogItem":
        """Build a normalized record from the raw endpoint payload."""

        keys = split_comma_list(raw.get("keys", []))
        if not keys:
            keys = split_comma_list(raw.get("competencies", []))

        job_levels = split_comma_list(raw.get("job_levels", []))
        languages = split_comma_list(raw.get("languages", []))
        duration = normalize_whitespace(raw.get("duration"))
        duration_raw = normalize_whitespace(raw.get("duration_raw"))
        description = normalize_whitespace(raw.get("description"))
        name = normalize_whitespace(raw.get("name"))
        test_type = infer_test_type(keys)
        recommendable = infer_recommendable(name, keys)

        search_text = join_text(
            [
                name,
                description,
                " ".join(keys),
                build_alias_text(keys),
                " ".join(job_levels),
                " ".join(languages),
                duration,
                duration_raw,
            ]
        )

        return cls(
            entity_id=normalize_whitespace(raw.get("entity_id")),
            name=name,
            link=normalize_whitespace(raw.get("link")),
            scraped_at=parse_scraped_at(raw.get("scraped_at")),
            job_levels=job_levels,
            job_levels_raw=normalize_whitespace(raw.get("job_levels_raw")),
            languages=languages,
            languages_raw=normalize_whitespace(raw.get("languages_raw")),
            duration=duration,
            duration_raw=duration_raw,
            duration_minutes=infer_duration_minutes(duration_raw or duration),
            status=normalize_whitespace(raw.get("status")),
            remote=coerce_bool(raw.get("remote")),
            adaptive=coerce_bool(raw.get("adaptive")),
            description=description,
            keys=keys,
            competencies=list(keys),
            test_type=test_type,
            recommendable=recommendable,
            search_text=search_text,
        )


class CatalogDataset(BaseModel):
    """Serializable catalog dataset with metadata."""

    model_config = ConfigDict(extra="ignore")

    source_url: str
    fetched_at: datetime
    total_items: int
    recommendable_items: int
    items: list[CatalogItem]

    @classmethod
    def from_items(
        cls,
        *,
        source_url: str,
        items: list[CatalogItem],
        fetched_at: datetime | None = None,
    ) -> "CatalogDataset":
        return cls(
            source_url=source_url,
            fetched_at=fetched_at or datetime.now(timezone.utc),
            total_items=len(items),
            recommendable_items=sum(1 for item in items if item.recommendable),
            items=items,
        )
