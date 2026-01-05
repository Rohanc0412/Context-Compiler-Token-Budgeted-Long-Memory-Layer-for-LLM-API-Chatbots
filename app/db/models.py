import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
    UniqueConstraint,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy import text
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy import DDL

from app.db.session import Base

try:
    from pgvector.sqlalchemy import Vector  # type: ignore
except Exception:  # pragma: no cover - fallback if pgvector not installed
    from sqlalchemy.types import TypeDecorator, JSON

    class Vector(TypeDecorator):  # type: ignore
        impl = JSON
        cache_ok = True

        def __init__(self, *args, **kwargs):
            super().__init__()


class Event(Base):
    __tablename__ = "events"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, index=True, nullable=False)
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)  # user or assistant
    text = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    metadata_json = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)

    chunks = relationship("EventChunk", back_populates="event", cascade="all, delete-orphan")


class EventChunk(Base):
    __tablename__ = "event_chunks"
    chunk_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(UUID(as_uuid=True), ForeignKey("events.id"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)

    event = relationship("Event", back_populates="chunks")
    embedding = relationship("Embedding", back_populates="chunk", uselist=False, cascade="all, delete-orphan")


class Embedding(Base):
    __tablename__ = "embeddings"
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("event_chunks.chunk_id"), primary_key=True)
    embedding_vector = Column(Vector(768))
    model_name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    chunk = relationship("EventChunk", back_populates="embedding")


class Memory(Base):
    __tablename__ = "memories"
    memory_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, index=True, nullable=False)
    session_id = Column(String, index=True, nullable=False)
    type = Column(String, nullable=False)  # constraint, preference, decision, task
    value_json = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    status = Column(String, nullable=False, default="active")
    confidence = Column(Float, nullable=False, default=0.5)
    source_event_ids = Column(
        ARRAY(UUID(as_uuid=True)).with_variant(JSON(), "sqlite"),
        nullable=False,
        default=list,
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Summary(Base):
    __tablename__ = "summaries"
    summary_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, index=True, nullable=False)
    session_id = Column(String, index=True, nullable=False)
    summary_json = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    last_event_id = Column(UUID(as_uuid=True), ForeignKey("events.id"), nullable=True)
    verified_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class PromptTrace(Base):
    __tablename__ = "prompt_traces"
    request_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String, nullable=False)
    session_id = Column(String, nullable=False)
    compiled_prompt = Column(Text, nullable=False)
    section_token_counts = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    included_ids = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    dropped_items = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    total_input_tokens = Column(Integer, nullable=False)
    model_name = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class EvalRun(Base):
    __tablename__ = "eval_runs"
    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    suite_name = Column(String, nullable=False)
    metrics_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class UserConfig(Base):
    __tablename__ = "user_configs"
    user_id = Column(String, primary_key=True)
    enable_memory = Column(Boolean, nullable=False, default=True)
    redaction_enabled = Column(Boolean, nullable=False, default=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# Ensure pgvector extension is available
sqlalchemy_event.listen(
    Base.metadata,
    "before_create",
    DDL("CREATE EXTENSION IF NOT EXISTS vector").execute_if(dialect="postgresql"),
)
