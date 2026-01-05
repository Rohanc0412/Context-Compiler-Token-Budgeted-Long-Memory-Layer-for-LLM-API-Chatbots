import math
import time
from typing import List, Tuple

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import models
from app.services.embedding import get_embedding_provider, BaseEmbeddingProvider
from app.services.compiler import WhitespaceTokenCounter


class RetrievalService:
    def __init__(self, provider: BaseEmbeddingProvider | None = None):
        self.settings = get_settings()
        self.provider = provider or get_embedding_provider()
        self.token_counter = WhitespaceTokenCounter()
        self.alpha = 0.6
        self.beta = 0.3
        self.gamma = 0.1

    def _similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _recency_score(self, created_at) -> float:
        if not created_at:
            return 0.0
        age_seconds = time.time() - created_at.timestamp()
        return 1 / (1 + age_seconds / 3600)

    def retrieve(
        self, db: Session, user_id: str, session_id: str, query_text: str, top_k: int | None = None
    ) -> Tuple[List[models.EventChunk], List[models.Memory]]:
        k = top_k or self.settings.retrieval_top_k
        query_vec = self.provider.embed(query_text)
        q = (
            db.query(models.EventChunk, models.Embedding, models.Event)
            .join(models.Embedding, models.Embedding.chunk_id == models.EventChunk.chunk_id)
            .join(models.Event, models.Event.id == models.EventChunk.event_id)
            .filter(models.Event.user_id == user_id, models.Event.session_id == session_id)
        )
        candidates: List[Tuple[models.EventChunk, models.Embedding, models.Event]] = q.all()
        scored = []
        for chunk, emb, event in candidates:
            sim = self._similarity(query_vec, emb.embedding_vector)
            recency = self._recency_score(event.created_at)
            role_score = 0.1 if event.role == "user" else 0.0
            score = self.alpha * sim + self.beta * recency + self.gamma * role_score
            scored.append((score, chunk, event))
        scored.sort(key=lambda x: x[0], reverse=True)
        deduped_chunks: List[models.EventChunk] = []
        seen_hashes = set()
        for score, chunk, _ in scored:
            norm_text = chunk.chunk_text.strip().lower()
            if norm_text in seen_hashes:
                continue
            seen_hashes.add(norm_text)
            deduped_chunks.append(chunk)
            if len(deduped_chunks) >= k:
                break

        structured = (
            db.query(models.Memory)
            .filter(
                models.Memory.user_id == user_id,
                models.Memory.session_id == session_id,
                models.Memory.status == "active",
            )
            .all()
        )
        return deduped_chunks, structured

