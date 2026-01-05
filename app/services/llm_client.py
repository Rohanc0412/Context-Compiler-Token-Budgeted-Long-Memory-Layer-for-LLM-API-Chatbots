import hashlib
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from app.core.config import get_settings


class BaseLLMClient(ABC):
    @abstractmethod
    def chat(self, prompt: str, max_tokens: int) -> str:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    """Deterministic mock for tests and demos."""

    def chat(self, prompt: str, max_tokens: int) -> str:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        return f"[mock-llm-output {digest}]"


class OpenAIClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        import openai

        self.client = openai.OpenAI(api_key=api_key) if hasattr(openai, "OpenAI") else openai
        self.model = model

    def chat(self, prompt: str, max_tokens: int) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        if hasattr(resp, "choices"):
            return resp.choices[0].message.content
        return resp["choices"][0]["message"]["content"]


def get_llm_client() -> BaseLLMClient:
    settings = get_settings()
    if settings.enable_openai_client and settings.openai_api_key:
        return OpenAIClient(api_key=settings.openai_api_key)
    return MockLLMClient()

