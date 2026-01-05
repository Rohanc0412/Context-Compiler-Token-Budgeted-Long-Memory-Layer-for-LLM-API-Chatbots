import uuid

from app.services.compiler import CompilerInput, ContextCompiler
from app.db import models


def test_compiler_trimming_order(db_session):
    compiler = ContextCompiler()
    user_id = "u1"
    session_id = "s1"
    # Create a long transcript to force trimming
    events = [
        models.Event(id=uuid.uuid4(), user_id=user_id, session_id=session_id, role="user", text="hello " * 10),
        models.Event(id=uuid.uuid4(), user_id=user_id, session_id=session_id, role="assistant", text="response " * 10),
    ]
    memories = [
        models.Memory(memory_id=uuid.uuid4(), user_id=user_id, session_id=session_id, type="constraint", value_json={"text": "I must be brief"}, status="active", source_event_ids=[]),
    ]
    chunks = [
        models.EventChunk(chunk_id=uuid.uuid4(), event_id=events[0].id, chunk_text="short fact", chunk_index=0, token_count=2)
    ]
    summary = models.Summary(
        summary_id=uuid.uuid4(),
        user_id=user_id,
        session_id=session_id,
        summary_json={"key": "value"},
    )
    payload = CompilerInput(
        user_id=user_id,
        session_id=session_id,
        latest_user_message="Current question",
        retrieved_chunks=chunks,
        structured_memories=memories,
        recent_events=events,
        summary=summary,
        input_budget_tokens=40,
        max_output_tokens=10,
    )
    compiled_prompt, trace = compiler.compile(db_session, payload)
    assert trace.section_token_counts["recent_transcript"] <= 10
    # retrieved memory should survive before summary is dropped
    assert "short fact" in compiled_prompt
    assert trace.total_input_tokens <= 40
    # trimming order should capture dropped transcript lines first
    assert trace.dropped_items.get("recent_transcript") is not None
