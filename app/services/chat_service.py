"""Chat orchestration for the SHL assessment assistant."""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.agents.planner import ConversationPlanner, Intent, PlanResult
from app.core.settings import Settings
from app.models.catalog import CatalogItem
from app.prompts.loader import load_prompt
from app.retrieval.engine import AssessmentRetrievalEngine, ComparisonSummary, RetrievedAssessment
from app.schemas.chat import ChatMessage, ChatResponse, Recommendation
from app.services.llm import LLMProvider, build_llm_provider
from app.utils.text import normalize_whitespace


@dataclass(slots=True)
class ChatService:
    """End-to-end service for the stateless chat endpoint."""

    planner: ConversationPlanner
    retriever: AssessmentRetrievalEngine
    settings: Settings
    llm_provider: LLMProvider

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        retriever: AssessmentRetrievalEngine,
        planner: ConversationPlanner | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> "ChatService":
        return cls(
            planner=planner or ConversationPlanner(),
            retriever=retriever,
            settings=settings,
            llm_provider=llm_provider or build_llm_provider(settings),
        )

    def handle(self, messages: list[ChatMessage]) -> ChatResponse:
        plan = self.planner.plan(messages)

        if plan.intent == Intent.refuse:
            return self._refusal_response(plan)
        if plan.intent == Intent.clarify:
            return self._clarification_response(plan)
        if plan.intent == Intent.compare:
            return self._comparison_response(plan)
        return self._recommendation_response(plan)

    def _refusal_response(self, plan: PlanResult) -> ChatResponse:
        reason = plan.refusal_reason or "off_topic"
        reply = self._compose_refusal_reply(reason)
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=True)

    def _clarification_response(self, plan: PlanResult) -> ChatResponse:
        reply = plan.clarification_question or "What role or assessment type should I optimize for?"
        if plan.is_refinement:
            reply = f"I can refine that. {reply}"
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    def _comparison_response(self, plan: PlanResult) -> ChatResponse:
        assert plan.compare_targets is not None
        left, right = plan.compare_targets
        comparison = self.retriever.compare(left, right)
        if comparison is None:
            reply = "I could not resolve both assessments from the SHL catalog. Please provide the exact assessment names."
            return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)
        reply = self._compose_comparison_reply(comparison)
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=True)

    def _recommendation_response(self, plan: PlanResult) -> ChatResponse:
        candidates = self.retriever.search(
            plan.query_text,
            filters=plan.filters,
            top_k=self.settings.max_recommendations,
            candidate_pool=self.settings.retrieval_candidate_pool,
        )
        recommendations = [
            Recommendation(name=match.item.name, url=match.item.link, test_type=match.item.test_type)
            for match in candidates[: self.settings.max_recommendations]
        ]
        reply = self._compose_recommendation_reply(plan, candidates)
        return ChatResponse(reply=reply, recommendations=recommendations, end_of_conversation=True)

    def _compose_refusal_reply(self, reason: str) -> str:
        if reason == "prompt_injection":
            fallback = "I can only help with SHL assessments, and I cannot follow prompt-injection instructions."
        elif reason == "legal":
            fallback = "I can only help with SHL assessments, not legal advice."
        elif reason == "off_topic":
            fallback = "I can only help with SHL assessments and cannot answer general hiring advice."
        else:
            fallback = "I can only help with SHL assessments."
        return self._maybe_polish_reply("guardrail", reason, fallback)

    def _compose_recommendation_reply(
        self,
        plan: PlanResult,
        candidates: list[RetrievedAssessment],
    ) -> str:
        if not candidates:
            return "I could not find a grounded SHL shortlist from the catalog. Please add a little more detail."

        lead = candidates[0].item
        prefix = "I refined the shortlist" if plan.is_refinement else "I found"
        summary_bits: list[str] = [f"{prefix} {len(candidates)} SHL assessment{'s' if len(candidates) != 1 else ''}"]
        if plan.filters.categories:
            summary_bits.append(f"focused on {', '.join(plan.filters.categories)}")
        if plan.filters.job_levels:
            summary_bits.append(f"for {', '.join(plan.filters.job_levels)}")
        if plan.filters.duration_preference:
            summary_bits.append(f"with a {plan.filters.duration_preference} assessment preference")
        summary = ", ".join(summary_bits)
        details = self._summarize_candidate_lead(lead)
        fallback = f"{summary}. {details}"
        context = self._format_recommendation_context(plan, candidates, fallback)
        return self._maybe_polish_reply("retrieval", context, fallback)

    def _compose_comparison_reply(self, comparison: ComparisonSummary) -> str:
        lines = [
            f"Comparison: {comparison.left.name} vs {comparison.right.name}",
            f"Purpose: {comparison.purpose}",
            f"Measures: {comparison.measures}",
            f"Duration: {comparison.duration}",
            f"Use case: {comparison.use_case}",
            "Strengths:",
        ]
        lines.extend(f"- {strength}" for strength in comparison.strengths)
        lines.append("Differences:")
        lines.extend(f"- {difference}" for difference in comparison.differences)
        fallback = "\n".join(lines)
        context = self._format_comparison_context(comparison, fallback)
        return self._maybe_polish_reply("comparison", context, fallback)

    @staticmethod
    def _summarize_candidate_lead(lead: CatalogItem) -> str:
        description = normalize_whitespace(lead.description)
        if not description:
            return f"The top match is {lead.name}."
        first_sentence = description.split(".")[0].strip()
        if len(first_sentence) > 180:
            first_sentence = first_sentence[:177].rstrip() + "..."
        return f"The top match is {lead.name}: {first_sentence}"

    @staticmethod
    def _format_recommendation_context(plan: PlanResult, candidates: list[RetrievedAssessment], fallback: str) -> str:
        candidate_lines = [
            f"- {candidate.item.name} | {candidate.item.test_type} | {candidate.item.link} | score={candidate.final_score:.3f}"
            for candidate in candidates[:10]
        ]
        return "\n".join(
            [
                f"Conversation query: {plan.query_text}",
                f"Filters: categories={plan.filters.categories}; job_levels={plan.filters.job_levels}; remote={plan.filters.needs_remote}; adaptive={plan.filters.needs_adaptive}; duration={plan.filters.duration_preference}",
                "Catalog candidates:",
                *candidate_lines,
                "Fallback reply:",
                fallback,
            ]
        )

    @staticmethod
    def _format_comparison_context(comparison: ComparisonSummary, fallback: str) -> str:
        return "\n".join(
            [
                f"Left: {comparison.left.name}",
                f"Right: {comparison.right.name}",
                f"Purpose: {comparison.purpose}",
                f"Measures: {comparison.measures}",
                f"Duration: {comparison.duration}",
                f"Use case: {comparison.use_case}",
                f"Strengths: {comparison.strengths}",
                f"Differences: {comparison.differences}",
                "Fallback reply:",
                fallback,
            ]
        )

    def _maybe_polish_reply(self, prompt_name: str, context: str, fallback: str) -> str:
        if getattr(self.llm_provider, "name", "template") == "template":
            return fallback
        try:
            system_prompt = "\n\n".join(
                [
                    load_prompt("system"),
                    load_prompt(prompt_name),
                load_prompt("formatting"),
            ]
        )
            candidate = self.llm_provider.generate(system_prompt=system_prompt, user_prompt=context).strip()
            return self._extract_reply_text(candidate) or fallback
        except Exception:
            return fallback

    @staticmethod
    def _extract_reply_text(candidate: str) -> str:
        text = candidate.strip()
        if not text:
            return ""

        if text.startswith("```"):
            lines = [line.rstrip() for line in text.splitlines()]
            if len(lines) >= 3 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
            else:
                text = text.strip("`").strip()

        json_blob = text
        if not json_blob.startswith("{"):
            start = json_blob.find("{")
            end = json_blob.rfind("}")
            if 0 <= start < end:
                json_blob = json_blob[start : end + 1]

        try:
            payload = json.loads(json_blob)
        except json.JSONDecodeError:
            return normalize_whitespace(text)

        if isinstance(payload, dict):
            reply = payload.get("reply")
            if isinstance(reply, str) and reply.strip():
                return normalize_whitespace(reply)

        return normalize_whitespace(text)
