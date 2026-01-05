import hashlib
import time
from dataclasses import dataclass
from typing import Dict, Optional

from app.core.config import get_settings
from app.services.embedding import BaseEmbeddingProvider, get_embedding_provider


@dataclass
class CacheEntry:
    response_text: str
    created_at: float
    embedding: list[float]


class SemanticCache:
    def __init__(self, provider: Optional[BaseEmbeddingProvider] = None):
        self.provider = provider or get_embedding_provider()
        self.store: Dict[tuple[str, str], CacheEntry] = {}
        self.settings = get_settings()

    def _similarity(self, a: list[float], b: list[float]) -> float:
        # Cosine similarity
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(y * y for y in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get(self, user_id: str, query: str) -> Optional[str]:
        if not self.settings.enable_semantic_cache:
            return None
        key = (user_id, hashlib.sha256(query.lower().encode("utf-8")).hexdigest())
        if key not in self.store:
            return None
        entry = self.store[key]
        query_embed = self.provider.embed(query)
        score = self._similarity(query_embed, entry.embedding)
        if score >= self.settings.semantic_cache_threshold:
            return entry.response_text
        return None

    def put(self, user_id: str, query: str, response_text: str) -> None:
        if not self.settings.enable_semantic_cache:
            return
        key = (user_id, hashlib.sha256(query.lower().encode("utf-8")).hexdigest())
        self.store[key] = CacheEntry(
            response_text=response_text,
            created_at=time.time(),
            embedding=self.provider.embed(query),
        )

