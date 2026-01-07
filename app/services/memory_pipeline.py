import logging
from typing import List, Set

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import models
from app.services.memory_gate import MemoryGate, GateResult
from app.services.memory_candidate_extractor import MemoryCandidateExtractor, Candidate, canonical_key
from app.services.memory_verifier import verify_candidates
from app.services.llm_structured_memory_extractor import LLMStructuredMemoryExtractor


class MemoryPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.gate = MemoryGate()
        self.extractor = MemoryCandidateExtractor()
        self.llm_extractor = LLMStructuredMemoryExtractor()

    def _role_allowed(self, role: str) -> bool:
        roles = {r.strip().lower() for r in self.settings.memory_extract_roles.split(",")}
        return role.lower() in roles

    def _existing_active(self, db: Session, user_id: str, session_id: str) -> List[models.Memory]:
        return (
            db.query(models.Memory)
            .filter(
                models.Memory.user_id == user_id,
                models.Memory.session_id == session_id,
                models.Memory.status == "active",
            )
            .all()
        )

    def _store(self, db: Session, event: models.Event, candidates: List[Candidate]) -> List[models.Memory]:
        stored: List[models.Memory] = []
        existing = self._existing_active(db, event.user_id, event.session_id)
        existing_by_key = {
            m.value_json.get("canonical_key", canonical_key(m.type, m.value_json.get("text", ""), f"{event.user_id}:{event.session_id}")): m
            for m in existing
        }
        for cand in candidates:
            existing_mem = existing_by_key.get(cand.canonical_key)
            if existing_mem:
                # upsert: extend source_event_ids
                ids = set(existing_mem.source_event_ids or [])
                ids.add(event.id)
                existing_mem.source_event_ids = list(ids)
                existing_mem.value_json["text"] = cand.value
                existing_mem.value_json["canonical_key"] = cand.canonical_key
                existing_mem.value_json["metadata"] = cand.metadata
                existing_mem.confidence = cand.confidence
                db.add(existing_mem)
                stored.append(existing_mem)
                continue

            mem = models.Memory(
                user_id=event.user_id,
                session_id=event.session_id,
                type=cand.type,
                value_json={
                    "text": cand.value,
                    "canonical_key": cand.canonical_key,
                    "metadata": cand.metadata,
                    "extractor": cand.extractor,
                },
                status="active",
                confidence=cand.confidence,
                source_event_ids=[event.id],
            )

            conflict = self._find_conflict(existing, mem)
            if conflict:
                conflict.status = "outdated"
                mem.value_json["supersedes"] = str(conflict.memory_id)
                db.add(conflict)

            db.add(mem)
            stored.append(mem)
        db.commit()
        return stored

    def _find_conflict(self, existing: List[models.Memory], new_mem: models.Memory):
        if new_mem.type not in ("constraint", "preference"):
            return None
        new_text = (new_mem.value_json.get("text") or "").lower()
        neg_tokens = {"not", "no", "never", "avoid", "cannot", "can't", "dont", "don't"}
        new_has_neg = any(tok in new_text for tok in neg_tokens)
        new_words = set(new_text.replace(",", " ").split())
        for mem in existing:
            if mem.type != new_mem.type or mem.status != "active":
                continue
            old_text = (mem.value_json.get("text") or "").lower()
            old_has_neg = any(tok in old_text for tok in neg_tokens)
            overlap = new_words.intersection(set(old_text.replace(",", " ").split()))
            overlap = {w for w in overlap if len(w) > 2}
            if overlap and new_has_neg != old_has_neg:
                return mem
        return None

    def run(self, db: Session, event: models.Event) -> List[models.Memory]:
        if not self.settings.memory_extraction_enabled:
            return []
        if not self._role_allowed(event.role):
            return []
        gate_result: GateResult = self.gate.predict(event.text) if self.settings.memory_gate_enabled else GateResult(worthy=True)
        if not gate_result.worthy:
            return []

        scope = f"{event.user_id}:{event.session_id}"
        candidates = self.extractor.extract(event.text, str(event.id), scope)
        # adjust confidence by gate type probs if available
        for cand in candidates:
            if gate_result.probs:
                cand.confidence *= max(0.5, gate_result.probs.get(cand.type, 0.0))

        existing = self._existing_active(db, event.user_id, event.session_id)
        existing_keys: Set[str] = {
            m.value_json.get("canonical_key", canonical_key(m.type, m.value_json.get("text", ""), scope)) for m in existing
        }
        candidates = verify_candidates(candidates, existing_keys)

        # LLM fallback if enabled and needed
        if self.settings.llm_memory_extractor_enabled:
            best_conf = max([c.confidence for c in candidates], default=0.0)
            if (not candidates or best_conf < self.settings.llm_memory_extractor_min_conf) and gate_result.worthy:
                llm_cands = self.llm_extractor.extract(event.text, event.user_id, event.session_id, str(event.id), existing_keys)
                candidates.extend(llm_cands)

        if not candidates:
            return []
        # Deduplicate across all candidates by canonical_key, keeping highest confidence
        dedup: dict[str, Candidate] = {}
        for cand in candidates:
            if cand.canonical_key not in dedup or dedup[cand.canonical_key].confidence < cand.confidence:
                dedup[cand.canonical_key] = cand

        return self._store(db, event, list(dedup.values()))
