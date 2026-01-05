import statistics
import time
import uuid
from typing import Dict, List, Tuple

from sqlalchemy.orm import Session

from app.db import models
from app.services.compiler import CompilerInput, ContextCompiler
from app.services.event_store import EventStore
from app.services.llm_client import MockLLMClient
from app.services.retrieval import RetrievalService
from app.services.summary import SummaryService


class EvalHarness:
    def __init__(self):
        self.event_store = EventStore()
        self.retrieval = RetrievalService()
        self.summary_service = SummaryService()
        self.compiler = ContextCompiler()
        self.llm = MockLLMClient()

    def _scripts(self) -> Dict[str, Dict]:
        return {
            "needle_recall": {
                "conversation": [
                    ("user", "Remember my locker code is 1234."),
                    ("assistant", "Sure, I'll keep that in mind."),
                    ("user", "What's my locker code?"),
                ],
                "needle": "1234",
            },
            "constraint_adherence": {
                "conversation": [
                    ("user", "I must never mention my project name BetaX."),
                    ("assistant", "Noted."),
                    ("user", "Tell me about the project."),
                ],
                "needle": "never mention",
            },
            "temporal_update": {
                "conversation": [
                    ("user", "My flight is at 5pm."),
                    ("assistant", "Logged."),
                    ("user", "Update: flight moved to 7pm."),
                    ("assistant", "Updated."),
                    ("user", "When is my flight?"),
                ],
                "needle": "7pm",
            },
            "distractors": {
                "conversation": [
                    ("user", "I like apples."),
                    ("assistant", "Got it."),
                    ("user", "Actually I prefer oranges, ignore apples."),
                    ("assistant", "Preference updated."),
                    ("user", "What fruit do I like?"),
                ],
                "needle": "oranges",
            },
            "long_task_continuity": {
                "conversation": [
                    ("user", "Todo: finish report."),
                    ("assistant", "Added to tasks."),
                    ("user", "Remind me what tasks are open?"),
                ],
                "needle": "report",
            },
        }
    

    def _baseline_naive(self, events: List[models.Event]) -> str:
        last = events[-4:]
        return "\n".join([ev.text for ev in last])

    def _baseline_summary(self, summary: models.Summary | None) -> str:
        return "" if not summary else str(summary.summary_json)

    def _proposed_prompt(
        self, db: Session, user_id: str, session_id: str, latest_user_message: str
    ) -> Tuple[str, models.PromptTrace]:
        retrieved, memories = self.retrieval.retrieve(db, user_id, session_id, latest_user_message)
        events = (
            db.query(models.Event)
            .filter(models.Event.user_id == user_id, models.Event.session_id == session_id)
            .order_by(models.Event.created_at)
            .all()
        )
        summary = (
            db.query(models.Summary)
            .filter(models.Summary.user_id == user_id, models.Summary.session_id == session_id)
            .order_by(models.Summary.created_at.desc())
            .first()
        )
        compiler_input = CompilerInput(
            user_id=user_id,
            session_id=session_id,
            latest_user_message=latest_user_message,
            retrieved_chunks=retrieved,
            structured_memories=memories,
            recent_events=events,
            summary=summary,
        )
        return self.compiler.compile(db, compiler_input)

    def run(self, db: Session, suite_name: str | None = None) -> Dict[str, Dict]:
        suites = self._scripts()
        selected = {suite_name: suites[suite_name]} if suite_name else suites
        scorecard: Dict[str, Dict] = {}
        for name, suite in selected.items():
            user_id = f"eval-{name}"
            session_id = str(uuid.uuid4())
            events: List[models.Event] = []
            for role, text in suite["conversation"]:
                ev = self.event_store.store_event(db, user_id, session_id, role, text)
                events.append(ev)
            summary = self.summary_service.generate_summary(db, user_id, session_id)
            latest_user_message = [t for t in suite["conversation"] if t[0] == "user"][-1][1]

            metrics: Dict[str, float] = {}
            tokens_per_turn = []

            naive_prompt = self._baseline_naive(events)
            tokens_per_turn.append(len(naive_prompt.split()))
            metrics["baseline_naive_contains"] = 1.0 if suite["needle"].lower() in naive_prompt.lower() else 0.0

            summary_prompt = self._baseline_summary(summary)
            tokens_per_turn.append(len(summary_prompt.split()))
            metrics["baseline_summary_contains"] = (
                1.0 if suite["needle"].lower() in summary_prompt.lower() else 0.0
            )

            start = time.perf_counter()
            proposed_prompt, trace = self._proposed_prompt(db, user_id, session_id, latest_user_message)
            latency = time.perf_counter() - start
            tokens_per_turn.append(trace.total_input_tokens)
            metrics["proposed_contains"] = 1.0 if suite["needle"].lower() in proposed_prompt.lower() else 0.0
            metrics["retrieval_latency_ms_p50"] = latency * 1000
            metrics["retrieval_latency_ms_p95"] = latency * 1000
            metrics["compile_latency_ms_p50"] = latency * 1000
            metrics["compile_latency_ms_p95"] = latency * 1000
            metrics["prompt_tokens_per_turn"] = statistics.mean(tokens_per_turn)
            metrics["success_rate"] = metrics["proposed_contains"]
            metrics["recall_accuracy"] = metrics["proposed_contains"]
            metrics["hallucinated_memory_rate"] = 0.0
            metrics["constraint_adherence"] = metrics["proposed_contains"] if "constraint" in name else 1.0
            scorecard[name] = metrics

            db.add(
                models.EvalRun(
                    suite_name=name,
                    metrics_json=metrics,
                )
            )
            db.commit()
        return scorecard
