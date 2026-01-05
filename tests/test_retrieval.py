from app.services.event_store import EventStore
from app.services.retrieval import RetrievalService


def test_retrieval_dedup(db_session):
    es = EventStore()
    rs = RetrievalService()
    user_id = "u2"
    session_id = "s2"
    es.store_event(db_session, user_id, session_id, "user", "Remember the code 1111.")
    es.store_event(db_session, user_id, session_id, "user", "Remember the code 1111.")  # duplicate
    chunks, memories = rs.retrieve(db_session, user_id, session_id, "code?")
    texts = {c.chunk_text for c in chunks}
    assert len(chunks) == len(texts)  # deduped
    assert len(chunks) == 1
