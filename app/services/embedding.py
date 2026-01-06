import hashlib
import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional

from app.core.config import get_settings


def embedding_dim_from_settings() -> int:
    settings = get_settings()
    model = getattr(settings, "embedding_model_name", "text-embedding-3-small")
    # Known OpenAI embedding dims; default to 1536 if unknown model, fallback 768 for mock-only
    known_dims = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    if settings.enable_openai_client:
        return known_dims.get(model, 1536)
    return known_dims.get(model, 768)


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class MockEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, dim: Optional[int] = None):
        self.dim = dim or embedding_dim_from_settings()

    def embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        rng = random.Random(digest)
        return [rng.random() for _ in range(self.dim)]


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str, model: Optional[str] = None):
        import openai

        self.client = openai.OpenAI(api_key=api_key) if hasattr(openai, "OpenAI") else openai
        settings = get_settings()
        self.model = model or settings.embedding_model_name

    def embed(self, text: str) -> List[float]:
        res = self.client.embeddings.create(model=self.model, input=text)
        data = res.data[0] if hasattr(res, "data") else res["data"][0]
        return list(data.embedding if hasattr(data, "embedding") else data["embedding"])


def get_embedding_provider() -> BaseEmbeddingProvider:
    settings = get_settings()
    if settings.enable_openai_client and settings.openai_api_key:
        try:
            return OpenAIEmbeddingProvider(api_key=settings.openai_api_key, model=settings.embedding_model_name)
        except ModuleNotFoundError:
            logging.getLogger(__name__).warning(
                "OpenAI client requested but not installed; falling back to mock embeddings."
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logging.getLogger(__name__).warning(
                "OpenAI embedding provider unavailable (%s); falling back to mock.", exc
            )
    return MockEmbeddingProvider()
