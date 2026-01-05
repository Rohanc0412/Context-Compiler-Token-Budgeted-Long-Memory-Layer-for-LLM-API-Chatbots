from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db import models


class BaseTokenCounter(ABC):
    @abstractmethod
    def count(self, text: str) -> int:
        raise NotImplementedError


class WhitespaceTokenCounter(BaseTokenCounter):
    def count(self, text: str) -> int:
        return len(text.strip().split())


class ExactTokenCounter(BaseTokenCounter):  # pragma: no cover - optional dependency
    def __init__(self):
        import tiktoken

        self.enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.enc.encode(text))


def get_token_counter() -> BaseTokenCounter:
    try:
        return ExactTokenCounter()
    except Exception:
        return WhitespaceTokenCounter()


@dataclass
class CompilerInput:
    user_id: str
    session_id: str
    latest_user_message: str
    retrieved_chunks: List[models.EventChunk]
    structured_memories: List[models.Memory]
    recent_events: List[models.Event]
    summary: models.Summary | None
    model_name: str = "mock-llm"
    input_budget_tokens: int | None = None
    max_output_tokens: int | None = None


class ContextCompiler:
    def __init__(self):
        self.settings = get_settings()
        self.counter = get_token_counter()

    def _serialize_memories(self, memories: List[models.Memory], type_filter: List[str]) -> str:
        filtered = [m for m in memories if m.type in type_filter and m.status == "active"]
        return "\n".join(f"- ({m.type}) {m.value_json.get('text')}" for m in filtered) or "None"

    def _recent_transcript(self, events: List[models.Event]) -> List[str]:
        lines = []
        for ev in events[-10:]:  # last N events
            prefix = "User" if ev.role == "user" else "Assistant"
            lines.append(f"{prefix}: {ev.text}")
        return lines

    def _retrieved_memory_text(self, chunks: List[models.EventChunk]) -> List[str]:
        return [f"[chunk {c.chunk_id}] {c.chunk_text}" for c in chunks]

    def _budget_ok(self, sections: Dict[str, str], budget: int) -> Tuple[int, Dict[str, int]]:
        counts = {name: self.counter.count(text) for name, text in sections.items()}
        total = sum(counts.values())
        return total, counts

    def _trim_sections(
        self,
        sections: Dict[str, str],
        counts: Dict[str, int],
        budget: int,
        retrieved_ids: List[str],
        transcript_ids: List[str],
    ) -> Tuple[Dict[str, str], Dict[str, int], Dict[str, List[str]]]:
        dropped: Dict[str, List[str]] = {"recent_transcript": [], "retrieved_memories": [], "rolling_summary": []}
        total = sum(counts.values())
        if total <= budget:
            return sections, counts, dropped

        # Trim recent transcript (oldest first)
        if "recent_transcript" in sections and counts.get("recent_transcript", 0) > 0:
            lines = sections["recent_transcript"].splitlines()
            while total > budget and lines:
                dropped_line = lines.pop(0)
                dropped["recent_transcript"].append(dropped_line)
                sections["recent_transcript"] = "\n".join(lines) if lines else ""
                counts["recent_transcript"] = self.counter.count(sections["recent_transcript"])
                total = sum(counts.values())

        # Then retrieved memories (lowest score at end assumed)
        if total > budget and "retrieved_memories" in sections and counts.get("retrieved_memories", 0) > 0:
            lines = sections["retrieved_memories"].splitlines()
            while total > budget and lines:
                dropped_line = lines.pop()  # drop last deterministically
                dropped["retrieved_memories"].append(dropped_line)
                sections["retrieved_memories"] = "\n".join(lines) if lines else ""
                counts["retrieved_memories"] = self.counter.count(sections["retrieved_memories"])
                total = sum(counts.values())

        # Then rolling summary
        if total > budget and "rolling_summary" in sections:
            if sections["rolling_summary"]:
                dropped["rolling_summary"].append(sections["rolling_summary"])
            sections["rolling_summary"] = ""
            counts["rolling_summary"] = 0
            total = sum(counts.values())

        return sections, counts, dropped

    def compile(self, db: Session, payload: CompilerInput) -> Tuple[str, models.PromptTrace]:
        budget = payload.input_budget_tokens or self.settings.input_token_budget
        max_output = payload.max_output_tokens or self.settings.max_output_tokens
        # Build sections
        sections: Dict[str, str] = {
            "system_and_policy": "You are a careful assistant. Follow safety, privacy, and data minimization policies.",
            "developer_rules": "This service compiles context under a strict token budget. Be concise and respect constraints.",
            "active_constraints": self._serialize_memories(payload.structured_memories, ["constraint"]),
            "task_state": self._serialize_memories(payload.structured_memories, ["task", "decision"]),
            "rolling_summary": json.dumps(payload.summary.summary_json) if payload.summary else "",
            "retrieved_memories": "\n".join(self._retrieved_memory_text(payload.retrieved_chunks)),
            "recent_transcript": "\n".join(self._recent_transcript(payload.recent_events)),
            "current_user_message": payload.latest_user_message,
        }

        # Always include user message
        counts_total, counts = self._budget_ok(sections, budget)
        if counts["current_user_message"] > budget:
            # User message alone exhausts budget; keep only message
            sections = {
                "current_user_message": payload.latest_user_message,
            }
            counts = {"current_user_message": counts["current_user_message"]}
        else:
            sections, counts, dropped = self._trim_sections(
                sections,
                counts,
                budget,
                [str(c.chunk_id) for c in payload.retrieved_chunks],
                [str(e.id) for e in payload.recent_events],
            )

        compiled_parts = []
        order = [
            "system_and_policy",
            "developer_rules",
            "active_constraints",
            "task_state",
            "rolling_summary",
            "retrieved_memories",
            "recent_transcript",
            "current_user_message",
        ]
        included_ids = {
            "retrieved_chunk_ids": [str(c.chunk_id) for c in payload.retrieved_chunks],
            "memory_ids": [str(m.memory_id) for m in payload.structured_memories],
            "recent_event_ids": [str(e.id) for e in payload.recent_events],
            "summary_id": str(payload.summary.summary_id) if payload.summary else None,
        }
        dropped_items = dropped if "dropped" in locals() else {}
        for key in order:
            if key in sections and sections[key]:
                compiled_parts.append(f"## {key}\\n{sections[key]}")

        compiled_prompt = "\n\n".join(compiled_parts)
        total_input_tokens = sum(counts.values())
        trace = models.PromptTrace(
            user_id=payload.user_id,
            session_id=payload.session_id,
            compiled_prompt=compiled_prompt,
            section_token_counts=counts,
            included_ids=included_ids,
            dropped_items=dropped_items,
            total_input_tokens=total_input_tokens,
            model_name=payload.model_name,
        )
        db.add(trace)
        db.commit()
        db.refresh(trace)
        return compiled_prompt, trace

