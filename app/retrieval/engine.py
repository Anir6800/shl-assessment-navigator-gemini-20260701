"""Retrieval and ranking logic for SHL assessments."""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from app.models.catalog import CatalogDataset, CatalogItem
from app.retrieval.vector_store import (
    CatalogMatch,
    CatalogVectorStore,
    DEFAULT_INDEX_DIR,
    build_catalog_vector_store,
)
from app.utils.text import normalize_whitespace


CATEGORY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Personality & Behavior": (
        "personality",
        "behavior",
        "behaviour",
        "behavioral",
        "behavioural",
        "opq",
        "motivation",
        "leadership style",
        "teamwork",
        "work style",
    ),
    "Knowledge & Skills": (
        "knowledge",
        "skill",
        "skills",
        "technical",
        "developer",
        "coding",
        "programming",
        "software",
        "java",
        "python",
        "sql",
        "sales",
        "finance",
        "customer service",
        "business",
        "simulation",
    ),
    "Ability & Aptitude": (
        "cognitive",
        "aptitude",
        "ability",
        "reasoning",
        "logic",
        "numerical",
        "verbal",
        "abstract",
        "deductive",
        "inductive",
        "problem solving",
    ),
    "Biodata & Situational Judgment": (
        "sjt",
        "situational judgment",
        "situational judgement",
        "biodata",
        "scenario",
        "judgment",
        "judgement",
    ),
    "Simulations": (
        "simulation",
        "simulate",
        "work sample",
        "call center",
        "contact center",
        "role play",
        "video interview",
        "exercise",
    ),
    "Assessment Exercises": (
        "assessment exercise",
        "exercise",
        "in-basket",
        "in basket",
    ),
    "Competencies": (
        "competency",
        "competencies",
        "ucf",
        "framework",
        "skills framework",
    ),
}

JOB_LEVEL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Entry-Level": ("entry level", "entry-level", "new grad", "graduate"),
    "Graduate": ("graduate", "new grad"),
    "Mid-Professional": ("mid level", "mid-level", "mid professional", "mid-professional", "experienced"),
    "Professional Individual Contributor": ("individual contributor", "ic", "professional"),
    "Supervisor": ("supervisor", "team lead"),
    "Front Line Manager": ("front line manager", "line manager", "manager of managers"),
    "Manager": ("manager",),
    "Director": ("director",),
    "Executive": ("executive", "vp", "cxo"),
    "General Population": ("general population", "all employees"),
}

DURATION_HINTS_SHORT = ("short", "quick", "brief", "fast", "lightweight", "15 min", "20 min")
DURATION_HINTS_LONG = ("deep", "thorough", "long", "comprehensive", "30 min", "45 min", "60 min")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_whitespace(text).lower())


def _token_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _slugify(text: str) -> str:
    tokens = _tokenize(text)
    return "-".join(tokens)


def _acronym(text: str) -> str:
    tokens = [token for token in _tokenize(text) if token.isalpha()]
    if len(tokens) < 2:
        return ""
    return "".join(token[0] for token in tokens[:6])


def infer_categories(text: str) -> list[str]:
    """Infer likely test categories from free text."""

    lowered = normalize_whitespace(text).lower()
    matches: list[str] = []
    for category, synonyms in CATEGORY_SYNONYMS.items():
        if any(synonym in lowered for synonym in synonyms):
            matches.append(category)
    return matches


def infer_job_levels(text: str) -> list[str]:
    """Infer likely job levels from free text."""

    lowered = normalize_whitespace(text).lower()
    matches: list[str] = []
    for level, synonyms in JOB_LEVEL_SYNONYMS.items():
        if any(synonym in lowered for synonym in synonyms):
            matches.append(level)
    return matches


def infer_duration_preference(text: str) -> str | None:
    lowered = normalize_whitespace(text).lower()
    if any(hint in lowered for hint in DURATION_HINTS_SHORT):
        return "short"
    if any(hint in lowered for hint in DURATION_HINTS_LONG):
        return "long"
    return None


@dataclass(slots=True)
class RetrievalFilters:
    """Metadata preferences extracted from the conversation."""

    categories: list[str] = field(default_factory=list)
    job_levels: list[str] = field(default_factory=list)
    duration_preference: str | None = None
    needs_remote: bool | None = None
    needs_adaptive: bool | None = None
    explicit_names: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievedAssessment:
    """A candidate assessment plus scoring components."""

    item: CatalogItem
    final_score: float
    vector_score: float
    lexical_score: float
    category_score: float
    job_level_score: float
    metadata_score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ComparisonSummary:
    """Grounded comparison between two catalog assessments."""

    left: CatalogItem
    right: CatalogItem
    purpose: str
    measures: str
    duration: str
    use_case: str
    strengths: list[str]
    differences: list[str]


class AssessmentRetrievalEngine:
    """Hybrid retrieval engine combining FAISS similarity and metadata scoring."""

    def __init__(
        self,
        dataset: CatalogDataset,
        vector_store: CatalogVectorStore,
    ) -> None:
        self.dataset = dataset
        self.vector_store = vector_store
        self._items = list(vector_store.items)
        self._items_by_name = {self._normalize_identifier(item.name): item for item in self._items}
        self._items_by_slug = {self._slug_from_url(item.link): item for item in self._items if item.link}
        self._items_by_alias = self._build_alias_lookup(self._items)

    @classmethod
    def from_dataset(
        cls,
        dataset: CatalogDataset,
        *,
        index_dir: Path = DEFAULT_INDEX_DIR,
        refresh_index: bool = False,
    ) -> "AssessmentRetrievalEngine":
        vector_store = (
            build_catalog_vector_store(dataset, index_dir=index_dir)
            if refresh_index
            else CatalogVectorStore.load_or_build(dataset, index_dir=index_dir)
        )
        return cls(dataset=dataset, vector_store=vector_store)

    def search(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        top_k: int = 10,
        candidate_pool: int = 50,
    ) -> list[RetrievedAssessment]:
        """Return a ranked shortlist for the current query."""

        filters = filters or RetrievalFilters(
            categories=infer_categories(query),
            job_levels=infer_job_levels(query),
            duration_preference=infer_duration_preference(query),
        )
        vector_matches = self.vector_store.search(query, top_k=min(candidate_pool, len(self._items)))

        ranked: list[RetrievedAssessment] = []
        query_tokens = Counter(_tokenize(query))
        for match in vector_matches:
            item = match.item
            lexical_score, lexical_reason = self._lexical_score(query_tokens, item)
            category_score, category_reason = self._category_score(item, filters.categories)
            job_score, job_reason = self._job_level_score(item, filters.job_levels)
            metadata_score, metadata_reasons = self._metadata_score(
                item,
                filters=filters,
                category_score=category_score,
                job_score=job_score,
                lexical_score=lexical_score,
            )
            final_score = self._final_score(match.score, lexical_score, category_score, job_score, metadata_score)
            reasons = [reason for reason in [lexical_reason, category_reason, job_reason, *metadata_reasons] if reason]
            ranked.append(
                RetrievedAssessment(
                    item=item,
                    final_score=final_score,
                    vector_score=match.score,
                    lexical_score=lexical_score,
                    category_score=category_score,
                    job_level_score=job_score,
                    metadata_score=metadata_score,
                    reasons=reasons,
                )
            )

        ranked.sort(key=lambda candidate: candidate.final_score, reverse=True)
        return ranked[:top_k]

    def resolve_item(self, identifier: str) -> CatalogItem | None:
        """Resolve an assessment name, acronym, or slug back to a catalog record."""

        normalized = self._normalize_identifier(identifier)
        if not normalized:
            return None

        if normalized in self._items_by_name:
            return self._items_by_name[normalized]
        if normalized in self._items_by_slug:
            return self._items_by_slug[normalized]
        if normalized in self._items_by_alias:
            return self._items_by_alias[normalized]

        close = difflib.get_close_matches(
            normalized,
            list(self._items_by_name.keys()) + list(self._items_by_alias.keys()) + list(self._items_by_slug.keys()),
            n=1,
            cutoff=0.75,
        )
        if close:
            return (
                self._items_by_name.get(close[0])
                or self._items_by_alias.get(close[0])
                or self._items_by_slug.get(close[0])
            )
        return None

    def compare(self, left_identifier: str, right_identifier: str) -> ComparisonSummary | None:
        """Compare two assessments using only catalog data."""

        left = self.resolve_item(left_identifier)
        right = self.resolve_item(right_identifier)
        if left is None or right is None:
            return None
        if not left.recommendable or not right.recommendable:
            return None

        purpose = self._compare_purpose(left, right)
        measures = self._compare_measures(left, right)
        duration = self._compare_duration(left, right)
        use_case = self._compare_use_case(left, right)
        strengths = self._compare_strengths(left, right)
        differences = self._compare_differences(left, right)
        return ComparisonSummary(
            left=left,
            right=right,
            purpose=purpose,
            measures=measures,
            duration=duration,
            use_case=use_case,
            strengths=strengths,
            differences=differences,
        )

    def _lexical_score(self, query_tokens: Counter[str], item: CatalogItem) -> tuple[float, str]:
        item_tokens = Counter(_tokenize(item.search_text))
        if not query_tokens:
            return 0.0, ""

        overlap = sum(min(query_tokens[token], item_tokens[token]) for token in query_tokens)
        score = overlap / max(1, sum(query_tokens.values()))
        reason = ""
        if score > 0:
            reason = "Lexical overlap with the assessment description and taxonomy."
        return min(score, 1.0), reason

    def _category_score(self, item: CatalogItem, categories: Sequence[str]) -> tuple[float, str]:
        if not categories:
            return 0.0, ""
        matched = sum(1 for category in categories if category in item.keys or category == item.test_type)
        if matched == 0:
            return 0.0, ""
        score = matched / len(categories)
        return min(score, 1.0), "Matches the requested assessment category."

    def _job_level_score(self, item: CatalogItem, job_levels: Sequence[str]) -> tuple[float, str]:
        if not job_levels:
            return 0.0, ""
        item_levels = {normalize_whitespace(level).lower() for level in item.job_levels}
        matched = sum(1 for level in job_levels if normalize_whitespace(level).lower() in item_levels)
        if matched == 0:
            return 0.0, ""
        score = matched / len(job_levels)
        return min(score, 1.0), "Aligns with the requested job level."

    def _metadata_score(
        self,
        item: CatalogItem,
        *,
        filters: RetrievalFilters,
        category_score: float,
        job_score: float,
        lexical_score: float,
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []
        score = 0.0

        if filters.explicit_names:
            item_name = self._normalize_identifier(item.name)
            if item_name in {self._normalize_identifier(name) for name in filters.explicit_names}:
                score += 0.35
                reasons.append("Explicitly requested by name.")

        if filters.needs_remote is not None:
            remote_match = item.remote == filters.needs_remote
            if remote_match:
                score += 0.12
                reasons.append("Matches the remote testing preference.")

        if filters.needs_adaptive is not None:
            adaptive_match = item.adaptive == filters.needs_adaptive
            if adaptive_match:
                score += 0.08
                reasons.append("Matches the adaptive testing preference.")

        if filters.duration_preference == "short" and item.duration_minutes is not None:
            if item.duration_minutes <= 20:
                score += 0.12
                reasons.append("Keeps the assessment short.")
        elif filters.duration_preference == "long" and item.duration_minutes is not None:
            if item.duration_minutes >= 25:
                score += 0.10
                reasons.append("Fits a deeper assessment preference.")

        score += category_score * 0.35
        score += job_score * 0.20
        score += lexical_score * 0.10

        if item.test_type and item.test_type != "Unknown":
            score += 0.05
        if item.description:
            score += 0.05

        return min(score, 1.0), reasons

    @staticmethod
    def _final_score(
        vector_score: float,
        lexical_score: float,
        category_score: float,
        job_score: float,
        metadata_score: float,
    ) -> float:
        return (
            (vector_score * 0.45)
            + (lexical_score * 0.15)
            + (category_score * 0.15)
            + (job_score * 0.10)
            + (metadata_score * 0.15)
        )

    @staticmethod
    def _normalize_identifier(value: str) -> str:
        return normalize_whitespace(value).lower()

    @staticmethod
    def _slug_from_url(url: str) -> str:
        if not url:
            return ""
        slug = normalize_whitespace(url.rstrip("/").split("/")[-1])
        return slug.lower()

    @staticmethod
    def _build_alias_lookup(items: Sequence[CatalogItem]) -> dict[str, CatalogItem]:
        lookup: dict[str, CatalogItem] = {}
        for item in items:
            name_tokens = _tokenize(item.name)
            slug_tokens = _tokenize(item.link.rstrip("/").split("/")[-1] if item.link else "")
            aliases = {
                _acronym(item.name),
                "".join(token[0] for token in name_tokens if token.isalpha() and token),
                *name_tokens,
                *slug_tokens,
            }
            for alias in aliases:
                alias = normalize_whitespace(alias).lower()
                if alias:
                    lookup.setdefault(alias, item)
        return lookup

    @staticmethod
    def _first_sentence(text: str) -> str:
        normalized = normalize_whitespace(text)
        if not normalized:
            return ""
        parts = re.split(r"(?<=[.!?])\s+", normalized)
        return parts[0] if parts else normalized

    def _compare_purpose(self, left: CatalogItem, right: CatalogItem) -> str:
        left_focus = left.test_type or "assessment"
        right_focus = right.test_type or "assessment"
        if left_focus == right_focus:
            return f"Both assessments are in the {left_focus} family, but they target different content areas."
        return f"{left.name} is a {left_focus.lower()} assessment, while {right.name} is a {right_focus.lower()} assessment."

    def _compare_measures(self, left: CatalogItem, right: CatalogItem) -> str:
        left_measure = self._first_sentence(left.description) or "Not listed in the catalog."
        right_measure = self._first_sentence(right.description) or "Not listed in the catalog."
        return f"{left.name}: {left_measure} {right.name}: {right_measure}"

    def _compare_duration(self, left: CatalogItem, right: CatalogItem) -> str:
        left_duration = left.duration or "Not listed in the catalog."
        right_duration = right.duration or "Not listed in the catalog."
        if left_duration == right_duration:
            return f"Both are listed at {left_duration}."
        return f"{left.name}: {left_duration}. {right.name}: {right_duration}."

    def _compare_use_case(self, left: CatalogItem, right: CatalogItem) -> str:
        left_use = self._use_case_from_item(left)
        right_use = self._use_case_from_item(right)
        if left_use == right_use:
            return f"Both are useful for {left_use.lower()}."
        return f"{left.name} is better suited for {left_use.lower()}, while {right.name} is better suited for {right_use.lower()}."

    def _compare_strengths(self, left: CatalogItem, right: CatalogItem) -> list[str]:
        strengths: list[str] = []
        strengths.append(f"{left.name}: {self._strength_from_item(left)}")
        strengths.append(f"{right.name}: {self._strength_from_item(right)}")
        return strengths

    def _compare_differences(self, left: CatalogItem, right: CatalogItem) -> list[str]:
        differences: list[str] = []
        if left.test_type != right.test_type:
            differences.append(f"{left.name} is a {left.test_type}, while {right.name} is a {right.test_type}.")
        if left.duration != right.duration:
            differences.append(f"Duration differs: {left.duration or 'not listed'} vs {right.duration or 'not listed'}.")
        if left.remote != right.remote:
            differences.append(f"Remote testing support differs: {left.name}={'yes' if left.remote else 'no'}, {right.name}={'yes' if right.remote else 'no'}.")
        if left.adaptive != right.adaptive:
            differences.append(f"Adaptive flag differs: {left.name}={'yes' if left.adaptive else 'no'}, {right.name}={'yes' if right.adaptive else 'no'}.")
        if not differences:
            differences.append("The catalog metadata does not show a major difference beyond the content focus.")
        return differences

    def _strength_from_item(self, item: CatalogItem) -> str:
        if item.description:
            first_sentence = self._first_sentence(item.description)
            if len(first_sentence) > 160:
                first_sentence = first_sentence[:157].rstrip() + "..."
            return first_sentence
        return "Not listed in the catalog."

    def _use_case_from_item(self, item: CatalogItem) -> str:
        categories = set(item.keys)
        if "Personality & Behavior" in categories:
            return "evaluating behavioral fit and work style"
        if "Ability & Aptitude" in categories:
            return "screening problem-solving and reasoning ability"
        if "Biodata & Situational Judgment" in categories:
            return "assessing judgment in realistic job scenarios"
        if "Simulations" in categories:
            return "observing job-relevant performance in simulated tasks"
        if "Competencies" in categories:
            return "mapping candidate capability to a competency framework"
        if "Knowledge & Skills" in categories:
            return "checking role-specific knowledge and technical skills"
        return "screening candidates against the cataloged criteria"


def load_or_build_engine(
    dataset: CatalogDataset,
    *,
    index_dir: Path = DEFAULT_INDEX_DIR,
) -> AssessmentRetrievalEngine:
    """Load the vector store if possible, otherwise rebuild it."""

    return AssessmentRetrievalEngine.from_dataset(dataset, index_dir=index_dir)
