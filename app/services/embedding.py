import hashlib
import os
import random
from abc import ABC, abstractmethod
from typing import List

from app.core.config import get_settings


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class MockEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dim: int = 128):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        rng = random.Random(digest)
        return [rng.random() for _ in range(self.dim)]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        import openai

        self.client = openai.OpenAI(api_key=api_key) if hasattr(openai, "OpenAI") else openai
        self.model = model

    def embed(self, text: str) -> List[float]:
        res = self.client.embeddings.create(model=self.model, input=text)
        data = res.data[0] if hasattr(res, "data") else res["data"][0]
        return list(data.embedding if hasattr(data, "embedding") else data["embedding"])


def get_embedding_provider() -> BaseEmbeddingProvider:
    settings = get_settings()
    if settings.enable_openai_client and settings.openai_api_key:
        return OpenAIEmbeddingProvider(api_key=settings.openai_api_key)
    return MockEmbeddingProvider()

