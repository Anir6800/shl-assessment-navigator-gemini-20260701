"""LLM provider abstraction with optional external providers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

import requests

from app.core.settings import Settings


class LLMProviderError(RuntimeError):
    """Raised when a configured LLM provider fails."""


class LLMProvider(Protocol):
    """Interface for provider implementations."""

    name: str

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass(slots=True)
class TemplateLLMProvider:
    """Deterministic fallback that never calls an external API."""

    name: str = "template"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return user_prompt.strip()


@dataclass(slots=True)
class GeminiLLMProvider:
    """Gemini 2.5 Flash provider via the public generateContent endpoint."""

    api_key: str
    model: str = "gemini-2.5-flash"
    name: str = "gemini"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
        }
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code >= 400:
            raise LLMProviderError(f"Gemini request failed: {response.status_code} {response.text}")
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:  # pragma: no cover - provider failure is external
            raise LLMProviderError("Gemini response was missing text content") from exc


@dataclass(slots=True)
class AnthropicLLMProvider:
    """Claude provider via Anthropic's messages API."""

    api_key: str
    model: str = "claude-3-5-sonnet-20240620"
    name: str = "anthropic"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0.2,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=30)
        if response.status_code >= 400:
            raise LLMProviderError(f"Anthropic request failed: {response.status_code} {response.text}")
        data = response.json()
        try:
            content = data["content"][0]["text"]
        except Exception as exc:  # pragma: no cover - provider failure is external
            raise LLMProviderError("Anthropic response was missing text content") from exc
        return content


@dataclass(slots=True)
class OpenRouterLLMProvider:
    """OpenRouter chat-completions provider."""

    api_key: str
    model: str = "openrouter/auto"
    name: str = "openrouter"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30)
        if response.status_code >= 400:
            raise LLMProviderError(f"OpenRouter request failed: {response.status_code} {response.text}")
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception as exc:  # pragma: no cover - provider failure is external
            raise LLMProviderError("OpenRouter response was missing text content") from exc


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Choose the best available provider from environment variables."""

    preference = (settings.llm_provider or "auto").lower()
    if preference in {"gemini", "auto"} and settings.gemini_api_key:
        return GeminiLLMProvider(api_key=settings.gemini_api_key, model=settings.gemini_model)
    if preference in {"anthropic", "claude", "auto"} and settings.anthropic_api_key:
        return AnthropicLLMProvider(api_key=settings.anthropic_api_key, model=settings.anthropic_model)
    if preference in {"openrouter", "auto"} and settings.openrouter_api_key:
        return OpenRouterLLMProvider(api_key=settings.openrouter_api_key, model=settings.openrouter_model)
    return TemplateLLMProvider()
