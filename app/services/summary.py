from collections import defaultdict
from typing import Dict, List, Optional, Any
import json
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import models
from app.services.llm_client import get_llm_client


class SummaryService:
    def __init__(self):
        self.settings = get_settings()
        self.llm = get_llm_client()

    def _stringify_uuids(self, obj: Any) -> Any:
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, list):
            return [self._stringify_uuids(x) for x in obj]
        if isinstance(obj, dict):
            return {k: self._stringify_uuids(v) for k, v in obj.items()}
        return obj

    def generate_summary(self, db: Session, user_id: str, session_id: str) -> models.Summary:
        if self.settings.use_llm_summary:
            return self.generate_summary_llm(db, user_id, session_id)
        memories = (
            db.query(models.Memory)
            .filter(
                models.Memory.user_id == user_id,
                models.Memory.session_id == session_id,
                models.Memory.status == "active",
            )
            .all()
        )
        grouped: Dict[str, List[dict]] = defaultdict(list)
        for mem in memories:
            grouped[mem.type].append(
                {"value": mem.value_json, "source_event_ids": mem.source_event_ids, "status": mem.status}
            )
        events = (
            db.query(models.Event)
            .filter(models.Event.user_id == user_id, models.Event.session_id == session_id)
            .order_by(models.Event.created_at.desc())
            .limit(5)
            .all()
        )
        key_context = [{"event_id": str(ev.id), "role": ev.role, "text": ev.text} for ev in events][::-1]
        summary_json = {
            "preferences": grouped.get("preference", []),
            "active_constraints": grouped.get("constraint", []),
            "open_tasks": grouped.get("task", []),
            "decisions": grouped.get("decision", []),
            "key_context": key_context,
        }
        summary_json = self._stringify_uuids(summary_json)
        last_event = events[0] if events else None
        summary = models.Summary(
            user_id=user_id,
            session_id=session_id,
            summary_json=summary_json,
            last_event_id=last_event.id if last_event else None,
        )
        db.add(summary)
        db.commit()
        db.refresh(summary)
        self._mark_outdated(db, user_id, session_id)
        return summary

    def generate_summary_llm(self, db: Session, user_id: str, session_id: str) -> models.Summary:
        # Pull recent events for context
        events = (
            db.query(models.Event)
            .filter(models.Event.user_id == user_id, models.Event.session_id == session_id)
            .order_by(models.Event.created_at.desc())
            .limit(self.settings.summary_window_events)
            .all()
        )
        events = list(reversed(events))
        memories = (
            db.query(models.Memory)
            .filter(
                models.Memory.user_id == user_id,
                models.Memory.session_id == session_id,
                models.Memory.status == "active",
            )
            .all()
        )

        transcript = "\n".join([f"{ev.role}: {ev.text}" for ev in events])
        memory_lines = "\n".join([f"{m.type}: {m.value_json.get('text')}" for m in memories])
        prompt = (
            "You are a summarizer. Extract a compact, factual JSON summary from the conversation.\n"
            "Return JSON with keys: preferences, active_constraints, open_tasks, decisions, key_context.\n"
            "Each list item should include a brief text and source_event_ids if clear. Do not invent facts.\n"
            f"Existing structured memory:\n{memory_lines or 'None'}\n\n"
            f"Transcript:\n{transcript}\n"
        )
        llm_text = self.llm.chat(prompt, max_tokens=self.settings.summary_max_output_tokens)
        summary_json: Dict[str, object]
        try:
            summary_json = json.loads(llm_text)
        except Exception:
            summary_json = {
                "model_summary": llm_text,
                "preferences": [],
                "active_constraints": [],
                "open_tasks": [],
                "decisions": [],
                "key_context": [
                    {"event_id": str(ev.id), "role": ev.role, "text": ev.text} for ev in events[-5:]
                ],
            }

        summary_json = self._stringify_uuids(summary_json)
        last_event = events[-1] if events else None
        summary = models.Summary(
            user_id=user_id,
            session_id=session_id,
            summary_json=summary_json,
            last_event_id=last_event.id if last_event else None,
        )
        db.add(summary)
        db.commit()
        db.refresh(summary)
        self._mark_outdated(db, user_id, session_id)
        return summary

    def _mark_outdated(self, db: Session, user_id: str, session_id: str) -> None:
        # If multiple memories of same type exist, keep most recent active.
        memories = (
            db.query(models.Memory)
            .filter(
                models.Memory.user_id == user_id,
                models.Memory.session_id == session_id,
            )
            .order_by(models.Memory.created_at.desc())
            .all()
        )
        latest_by_type: Dict[str, models.Memory] = {}
        for mem in memories:
            if mem.type not in latest_by_type:
                latest_by_type[mem.type] = mem
            else:
                if mem.status == "active":
                    mem.status = "outdated"
        db.commit()
