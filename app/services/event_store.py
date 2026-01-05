import re
from typing import List

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.telemetry import record_step
from app.db import models
from app.services.compiler import WhitespaceTokenCounter
from app.services.embedding import get_embedding_provider
from app.services.memory_extractor import extract_and_store_memories

EMAIL_RE = re.compile(r"[\\w\\.]+@[\\w\\.]+")
PHONE_RE = re.compile(r"\\+?\\d[\\d\\-]{7,}\\d")


class EventStore:
    def __init__(self):
        self.settings = get_settings()
        self.embedder = get_embedding_provider()
        self.token_counter = WhitespaceTokenCounter()

    def is_memory_enabled(self, db: Session, user_id: str) -> bool:
        cfg = db.get(models.UserConfig, user_id)
        if cfg is None:
            cfg = models.UserConfig(user_id=user_id, enable_memory=True, redaction_enabled=True)
            db.add(cfg)
            db.commit()
            db.refresh(cfg)
        return cfg.enable_memory

    def redact(self, text: str) -> str:
        redacted = EMAIL_RE.sub("[redacted-email]", text)
        redacted = PHONE_RE.sub("[redacted-phone]", redacted)
        return redacted

    def chunk_text(self, text: str, target_tokens: int = 300) -> List[str]:
        words = text.split()
        chunks: List[str] = []
        current: List[str] = []
        current_tokens = 0
        for word in words:
            current.append(word)
            current_tokens += 1
            if current_tokens >= target_tokens:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
        if current:
            chunks.append(" ".join(current))
        return chunks or [text]

    def chunk_and_embed(self, db: Session, event: models.Event, enable_memory: bool = True) -> None:
        if not enable_memory:
            return
        text_for_embedding = self.redact(event.text) if self.settings.redact_before_embedding else event.text
        chunk_texts = self.chunk_text(text_for_embedding)
        for idx, chunk_text in enumerate(chunk_texts):
            token_count = self.token_counter.count(chunk_text)
            chunk = models.EventChunk(
                event_id=event.id,
                chunk_text=chunk_text,
                chunk_index=idx,
                token_count=token_count,
            )
            db.add(chunk)
            db.flush()
            embedding = self.embedder.embed(chunk_text)
            db.add(
                models.Embedding(
                    chunk_id=chunk.chunk_id,
                    embedding_vector=embedding,
                    model_name="mock-embedding",
                )
            )
        db.commit()

    def store_event(
        self,
        db: Session,
        user_id: str,
        session_id: str,
        role: str,
        text: str,
        metadata_json=None,
    ) -> models.Event:
        event = models.Event(
            user_id=user_id,
            session_id=session_id,
            role=role,
            text=text,
            metadata_json=metadata_json,
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        memory_enabled = self.is_memory_enabled(db, user_id)
        with record_step("chunk_and_embed"):
            self.chunk_and_embed(db, event, enable_memory=memory_enabled)
        with record_step("memory_extraction"):
            extract_and_store_memories(db, event, enabled=memory_enabled)
        return event
