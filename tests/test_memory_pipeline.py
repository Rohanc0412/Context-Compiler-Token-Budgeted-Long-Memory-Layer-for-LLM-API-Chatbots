import uuid

import pytest

from app.core.config import get_settings
from app.db import models
from app.services.memory_gate import MemoryGate, GateResult
from app.services.memory_candidate_extractor import MemoryCandidateExtractor, Candidate, canonical_key
from app.services.memory_verifier import verify_candidates
from app.services.memory_pipeline import MemoryPipeline


def reset_settings(monkeypatch, env: dict | None = None):
    if env:
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
    get_settings.cache_clear()


def test_gate_predicts_worthy_for_constraints(monkeypatch):
    reset_settings(monkeypatch)
    gate = MemoryGate()
    res = gate.predict("I must avoid peanuts at all costs.")
    assert res.worthy
    assert res.top_type in ("constraint", "none")


def test_gate_rejects_greetings(monkeypatch):
    reset_settings(monkeypatch)
    gate = MemoryGate()
    res = gate.predict("Hello there, how are you?")
    assert res.worthy is False


def test_extractor_splits_and_extracts_two_memories_from_one_sentence(monkeypatch):
    reset_settings(monkeypatch)
    extractor = MemoryCandidateExtractor()
    cands = extractor.extract("I must avoid peanuts and I prefer vegan meals", str(uuid.uuid4()), "u:s")
    types = {c.type for c in cands}
    assert "constraint" in types
    assert "preference" in types


def test_role_gate_user_only(monkeypatch, db_session):
    reset_settings(monkeypatch, {"MEMORY_EXTRACT_ROLES": "user"})
    pipeline = MemoryPipeline()
    ev = models.Event(user_id="u1", session_id="s1", role="assistant", text="Task: finish report")
    db_session.add(ev)
    db_session.commit()
    db_session.refresh(ev)
    stored = pipeline.run(db_session, ev)
    assert stored == []
    assert db_session.query(models.Memory).count() == 0


def test_verifier_rejects_assistantish_phrases():
    cand = Candidate(
        type="constraint",
        value="got it",
        confidence=0.9,
        evidence="got it",
        source_event_id=str(uuid.uuid4()),
        canonical_key="constraint::got it",
    )
    verified = verify_candidates([cand], existing_keys=set())
    assert verified == []


def test_dedupe_upsert_updates_existing_memory(monkeypatch, db_session):
    reset_settings(monkeypatch)
    pipeline = MemoryPipeline()
    ev1 = models.Event(user_id="u1", session_id="s1", role="user", text="Task: finish the quarterly report by Friday.")
    db_session.add(ev1)
    db_session.commit()
    db_session.refresh(ev1)
    pipeline.run(db_session, ev1)
    ev2 = models.Event(user_id="u1", session_id="s1", role="user", text="Task: finish the quarterly report by Friday.")
    db_session.add(ev2)
    db_session.commit()
    db_session.refresh(ev2)
    pipeline.run(db_session, ev2)
    memories = db_session.query(models.Memory).all()
    assert len(memories) == 1
    assert len(memories[0].source_event_ids) == 2


def test_llm_called_only_when_low_confidence_and_flag_enabled(monkeypatch, db_session):
    reset_settings(
        monkeypatch,
        {
            "LLM_MEMORY_EXTRACTOR_ENABLED": "true",
            "LLM_MEMORY_EXTRACTOR_MIN_CONF": "0.8",
            "MEMORY_EXTRACT_ROLES": "user",
        },
    )
    pipeline = MemoryPipeline()

    # Force gate and extractor behavior
    pipeline.gate.predict = lambda text: GateResult(worthy=True, probs={"task": 0.6}, top_type="task", confidence=0.6)

    def fake_extract(text, source_event_id, scope):
        return [
            Candidate(
                type="task",
                value="finish report",
                confidence=0.5,
                evidence=text,
                source_event_id=source_event_id,
                canonical_key=canonical_key("task", "finish report", scope),
                extractor="rules",
            )
        ]

    pipeline.extractor.extract = fake_extract
    called = {}

    def fake_llm(text, user_id, session_id, source_event_id, existing_keys):
        called["hit"] = True
        return [
            Candidate(
                type="task",
                value="finish report",
                confidence=0.9,
                evidence=text,
                source_event_id=source_event_id,
                canonical_key=canonical_key("task", "finish report", f"{user_id}:{session_id}"),
                extractor="llm",
            )
        ]

    pipeline.llm_extractor.extract = fake_llm
    ev = models.Event(user_id="u2", session_id="s2", role="user", text="Can you remind me to finish the report?")
    db_session.add(ev)
    db_session.commit()
    db_session.refresh(ev)
    pipeline.run(db_session, ev)
    assert called.get("hit") is True
    memories = db_session.query(models.Memory).filter(models.Memory.user_id == "u2").all()
    assert len(memories) == 1
    assert memories[0].value_json.get("extractor") == "llm"

