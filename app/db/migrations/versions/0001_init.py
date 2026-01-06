"""initial schema

Revision ID: 0001_init
Revises:
Create Date: 2026-01-05
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()
    if conn.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False, index=True),
        sa.Column("session_id", sa.String(), nullable=False, index=True),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_table(
        "user_configs",
        sa.Column("user_id", sa.String(), primary_key=True),
        sa.Column("enable_memory", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("redaction_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_table(
        "event_chunks",
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("event_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("events.id"), nullable=False, index=True),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
    )

    try:
        from pgvector.sqlalchemy import Vector
    except Exception:
        Vector = sa.JSON  # type: ignore

    op.create_table(
        "embeddings",
        sa.Column("chunk_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("event_chunks.chunk_id"), primary_key=True),
        sa.Column("embedding_vector", Vector()),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "memories",
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False, index=True),
        sa.Column("session_id", sa.String(), nullable=False, index=True),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("value_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default=sa.text("0.5")),
        sa.Column("source_event_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "summaries",
        sa.Column("summary_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False, index=True),
        sa.Column("session_id", sa.String(), nullable=False, index=True),
        sa.Column("summary_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("last_event_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("events.id")),
        sa.Column("verified_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "prompt_traces",
        sa.Column("request_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("compiled_prompt", sa.Text(), nullable=False),
        sa.Column("section_token_counts", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("included_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("dropped_items", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("total_input_tokens", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "eval_runs",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("suite_name", sa.String(), nullable=False),
        sa.Column("metrics_json", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade():
    op.drop_table("eval_runs")
    op.drop_table("prompt_traces")
    op.drop_table("summaries")
    op.drop_table("memories")
    op.drop_table("embeddings")
    op.drop_table("event_chunks")
    op.drop_table("user_configs")
    op.drop_table("events")
