import re
from typing import List, Tuple

from sqlalchemy.orm import Session

from app.db import models

CONSTRAINT_PATTERNS = [
    r"\bi must\b",
    r"\bi cannot\b",
    r"\bdo not\b",
    r"\balways\b",
    r"\bnever\b",
    r"\bonly\b",
    r"\ballergic\b",
    r"\bavoid\b",
    r"\bno\b",
]
PREFERENCE_PATTERNS = [
    r"\bi prefer\b",
    r"\bi like\b",
    r"\bi do not like\b",
]
DECISION_PATTERNS = [
    r"\bwe decided\b",
    r"\bdecision\b",
    r"\bfinalized\b",
    r"\bchoose\b",
]
TASK_PATTERNS = [
    r"\btodo\b",
    r"\btask\b",
    r"\bnext step\b",
    r"\bremind me\b",
]


def _match_any(patterns: List[str], text: str) -> bool:
    return any(re.search(pat, text, flags=re.IGNORECASE) for pat in patterns)


def detect_memory_type(text: str) -> List[Tuple[str, str]]:
    findings: List[Tuple[str, str]] = []
    lower = text.lower()
    if _match_any(CONSTRAINT_PATTERNS, lower):
        findings.append(("constraint", text))
    if _match_any(PREFERENCE_PATTERNS, lower):
        findings.append(("preference", text))
    if _match_any(DECISION_PATTERNS, lower):
        findings.append(("decision", text))
    if _match_any(TASK_PATTERNS, lower):
        findings.append(("task", text))
    return findings


def extract_and_store_memories(
    db: Session,
    event: models.Event,
    enabled: bool = True,
) -> List[models.Memory]:
    if not enabled:
        return []
    memories: List[models.Memory] = []
    for type_, text in detect_memory_type(event.text):
        mem = models.Memory(
            user_id=event.user_id,
            session_id=event.session_id,
            type=type_,
            value_json={"text": text},
            status="active",
            confidence=0.7,
            source_event_ids=[event.id],
        )
        db.add(mem)
        memories.append(mem)
    db.commit()
    return memories

