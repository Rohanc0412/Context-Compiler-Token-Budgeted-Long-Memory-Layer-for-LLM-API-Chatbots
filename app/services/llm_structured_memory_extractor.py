import json
import time
from typing import List, Tuple

from app.core.config import get_settings
from app.schemas.memory_extraction import LLMMemoryResponse, LLMMemoryItem
from app.services.llm_client import get_llm_client, BaseLLMClient
from app.services.memory_candidate_extractor import Candidate, canonical_key
from app.services.memory_verifier import verify_candidates


class LLMStructuredMemoryExtractor:
    def __init__(self, llm_client: BaseLLMClient | None = None):
        self.settings = get_settings()
        self.llm = llm_client or get_llm_client()
        self.rate_limits = {}  # (user, session) -> list[timestamps]

    def _allow(self, user_id: str, session_id: str) -> bool:
        key = (user_id, session_id)
        now = time.time()
        horizon = now - 3600
        history = [t for t in self.rate_limits.get(key, []) if t >= horizon]
        if len(history) >= self.settings.llm_memory_extractor_max_calls_per_session_per_hour:
            return False
        history.append(now)
        self.rate_limits[key] = history
        return True

    def _prompt(self, text: str) -> str:
        return (
            "You are an information extractor. Extract actionable memories from the user message.\n"
            "Respond with JSON only, matching this schema:\n"
            '{ "memories": [ { "type": "constraint|preference|task|decision",'
            ' "text": "string", "confidence": 0.0-1.0, "due_date": "YYYY-MM-DD optional",'
            ' "status": "active|outdated|done optional" } ] }\n'
            "Do not include any extra keys or text. Base your answer only on the input.\n"
            f"Input:\n{text}\n"
        )

    def extract(
        self, text: str, user_id: str, session_id: str, source_event_id: str, existing_keys: set[str]
    ) -> List[Candidate]:
        if not self.settings.llm_memory_extractor_enabled:
            return []
        if not self._allow(user_id, session_id):
            return []
        prompt = self._prompt(text)
        llm_output = self.llm.chat(prompt, max_tokens=self.settings.summary_max_output_tokens)
        try:
            data = json.loads(llm_output)
            parsed = LLMMemoryResponse(**data)
        except Exception:
            return []
        candidates: List[Candidate] = []
        scope = f"{user_id}:{session_id}"
        for item in parsed.memories:
            cand = Candidate(
                type=item.type,
                value=item.text,
                confidence=item.confidence,
                evidence=text,
                source_event_id=source_event_id,
                canonical_key=canonical_key(item.type, item.text, scope),
                metadata={},
                extractor="llm",
            )
            if item.due_date:
                cand.metadata["due_date"] = item.due_date
            if item.status:
                cand.metadata["status"] = item.status
            candidates.append(cand)
        return verify_candidates(candidates, existing_keys)

