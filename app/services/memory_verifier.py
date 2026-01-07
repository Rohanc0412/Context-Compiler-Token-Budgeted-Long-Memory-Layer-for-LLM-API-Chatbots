import re
import string
from typing import Dict, List, Set

from app.services.memory_candidate_extractor import Candidate


ASSISTANTISH = [
    "got it",
    "i will make sure",
    "feel free",
    "let me know",
    "sure thing",
    "noted",
    "i'll note that",
]


def _alnum_ratio(text: str) -> float:
    if not text:
        return 0.0
    alnum = sum(ch.isalnum() for ch in text)
    return alnum / max(len(text), 1)


def verify_candidates(candidates: List[Candidate], existing_keys: Set[str]) -> List[Candidate]:
    verified: List[Candidate] = []
    for cand in candidates:
        val = cand.value.strip()
        if len(val) < 3 or len(val) > 120:
            continue
        lower_val = val.lower()
        if any(phrase in lower_val for phrase in ASSISTANTISH):
            continue
        if _alnum_ratio(val) < 0.4:
            continue
        if cand.canonical_key in existing_keys:
            continue
        if cand.type == "task" and "update me on progress" in lower_val:
            continue
        if cand.type != "task" and "update me on progress" in lower_val:
            continue
        # Confidence adjustments
        if any(h in lower_val for h in ["maybe", "not sure", "if possible"]):
            cand.confidence *= 0.8
        if any(label in lower_val for label in ["task:", "decision:", "todo:"]):
            cand.confidence = min(1.0, cand.confidence + 0.1)
        verified.append(cand)
    return verified

