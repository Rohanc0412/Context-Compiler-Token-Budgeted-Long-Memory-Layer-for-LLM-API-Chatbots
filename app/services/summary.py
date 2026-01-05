from collections import defaultdict
from typing import Dict, List

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import models


class SummaryService:
    def __init__(self):
        self.settings = get_settings()

    def generate_summary(self, db: Session, user_id: str, session_id: str) -> models.Summary:
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

