import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.core.config import get_settings


STOP_PUNCT = set(string.punctuation)


def _normalize_value(value: str) -> str:
    cleaned = value.strip().lower()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def canonical_key(type_: str, value: str, scope: Optional[str] = None) -> str:
    key = f"{type_.lower()}::{_normalize_value(value)}"
    if scope:
        key = f"{scope}::{key}"
    return key


@dataclass
class Candidate:
    type: str
    value: str
    confidence: float
    evidence: str
    source_event_id: str
    canonical_key: str
    metadata: Dict[str, str] = field(default_factory=dict)
    extractor: str = "rules"


class MemoryCandidateExtractor:
    def __init__(self):
        self.settings = get_settings()
        self.nlp = None
        if self.settings.spacy_matcher_enabled:
            try:
                import spacy
                from spacy.matcher import Matcher

                self.nlp = spacy.load("en_core_web_sm")
                self.matcher = Matcher(self.nlp.vocab)
                self._build_patterns()
            except Exception:
                self.nlp = None

    def _build_patterns(self):
        patterns = [
            ("PREFER", [{"LOWER": "i"}, {"LOWER": {"IN": ["prefer", "like", "love"]}}]),
            ("MUST", [{"LOWER": "i"}, {"LOWER": {"IN": ["must", "cannot", "can't", "should"]}}]),
            ("TASK", [{"LOWER": "task"}, {"ORTH": ":"}]),
            ("DECIDED", [{"LOWER": "we"}, {"LOWER": {"IN": ["decided", "chose", "agreed"]}}]),
        ]
        for name, pattern in patterns:
            self.matcher.add(name, [pattern])

    def _split_clauses(self, text: str) -> List[str]:
        parts = re.split(r"[.;]| and ", text)
        return [p.strip() for p in parts if p.strip()]

    def _extract_value_after_cue(self, clause: str, cues: List[str]) -> Optional[str]:
        lower = clause.lower()
        for cue in cues:
            if cue in lower:
                idx = lower.find(cue)
                value = clause[idx + len(cue) :].strip(" :,-")
                return value or None
        return None

    def _extract_okay_now(self, clause: str) -> Optional[str]:
        m = re.search(r"([A-Za-z0-9 ]+?) (?:is|are) ok(?:ay)? now", clause, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    def _parse_due_date(self, clause: str) -> Optional[str]:
        m = re.search(r"\bby (\w+day|\d{4}-\d{2}-\d{2}|tomorrow|today|next week)\b", clause, re.IGNORECASE)
        if m:
            token = m.group(1)
            return token
        return None

    def _spacy_candidates(self, text: str, source_event_id: str, scope: str) -> List[Candidate]:
        if not self.nlp:
            return []
        doc = self.nlp(text)
        matches = self.matcher(doc)
        cands: List[Candidate] = []
        for _, start, end in matches:
            span = doc[start:end].sent
            span_text = span.text.strip()
            label = doc.vocab.strings[_]
            type_map = {
                "PREFER": "preference",
                "MUST": "constraint",
                "TASK": "task",
                "DECIDED": "decision",
            }
            t = type_map.get(label, "preference")
            value = span_text
            if len(span) < len(span.sent):
                value = span_text.replace(span.text, "").strip() or span_text
            norm_value = _normalize_value(value)
            cands.append(
                Candidate(
                    type=t,
                    value=norm_value,
                    confidence=0.6,
                    evidence=span_text,
                    source_event_id=source_event_id,
                    canonical_key=canonical_key(t, norm_value, scope),
                    extractor="spacy",
                )
            )
        return cands

    def extract(self, text: str, source_event_id: str, scope: str) -> List[Candidate]:
        candidates: List[Candidate] = []
        seen_keys: set[str] = set()
        clauses = self._split_clauses(text)
        for clause in clauses:
            # Constraints
            value = self._extract_value_after_cue(
                clause, ["i must", "i cannot", "i can't", "i should not", "avoid", "never", "do not"]
            )
            if value:
                norm = _normalize_value(value)
                key = canonical_key("constraint", norm, scope)
                if key not in seen_keys:
                    seen_keys.add(key)
                    candidates.append(
                        Candidate(
                            type="constraint",
                            value=norm,
                            confidence=0.65,
                            evidence=clause,
                            source_event_id=source_event_id,
                            canonical_key=key,
                        )
                    )
            # Reversal of constraint
            value = self._extract_okay_now(clause)
            if value:
                norm = _normalize_value(f"{value} are okay now")
                key = canonical_key("constraint", norm, scope)
                if key not in seen_keys:
                    seen_keys.add(key)
                    candidates.append(
                        Candidate(
                            type="constraint",
                            value=norm,
                            confidence=0.65,
                            evidence=clause,
                            source_event_id=source_event_id,
                            canonical_key=key,
                        )
                    )
            value = self._extract_value_after_cue(clause, ["i prefer", "i like", "i love", "i do not like"])
            if value:
                norm = _normalize_value(value)
                key = canonical_key("preference", norm, scope)
                if key not in seen_keys:
                    seen_keys.add(key)
                    candidates.append(
                        Candidate(
                            type="preference",
                            value=norm,
                            confidence=0.6,
                            evidence=clause,
                            source_event_id=source_event_id,
                            canonical_key=key,
                        )
                    )
            value = self._extract_value_after_cue(clause, ["task", "todo", "next step", "remind me"])
            if value:
                norm = _normalize_value(value)
                metadata: Dict[str, str] = {}
                due = self._parse_due_date(clause)
                if due:
                    metadata["due_date"] = due
                key = canonical_key("task", norm, scope)
                if key not in seen_keys:
                    seen_keys.add(key)
                    candidates.append(
                        Candidate(
                            type="task",
                            value=norm,
                            confidence=0.65,
                            evidence=clause,
                            source_event_id=source_event_id,
                            canonical_key=key,
                            metadata=metadata,
                        )
                    )
            value = self._extract_value_after_cue(clause, ["we decided to", "we decided", "decision", "we chose"])
            if value:
                norm = _normalize_value(value)
                key = canonical_key("decision", norm, scope)
                if key not in seen_keys:
                    seen_keys.add(key)
                    candidates.append(
                        Candidate(
                            type="decision",
                            value=norm,
                            confidence=0.6,
                            evidence=clause,
                            source_event_id=source_event_id,
                            canonical_key=key,
                        )
                    )

        spacy_cands = self._spacy_candidates(text, source_event_id, scope)
        for c in spacy_cands:
            if c.canonical_key in seen_keys:
                continue
            seen_keys.add(c.canonical_key)
            candidates.append(c)

        # merge duplicates by canonical_key, keep max confidence
        deduped: Dict[str, Candidate] = {}
        for c in candidates:
            if c.canonical_key not in deduped or deduped[c.canonical_key].confidence < c.confidence:
                deduped[c.canonical_key] = c
        return list(deduped.values())
