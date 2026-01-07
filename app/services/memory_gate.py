import logging
from dataclasses import dataclass, field
from typing import Dict

from app.core.config import get_settings


@dataclass
class GateResult:
    worthy: bool
    probs: Dict[str, float] = field(default_factory=dict)
    top_type: str = "none"
    confidence: float = 0.0


class _RuleGate:
    CUES = {
        "constraint": ["must", "cannot", "avoid", "never", "allergic", "do not", "only"],
        "preference": ["prefer", "like", "love", "do not like"],
        "task": ["task", "todo", "remind", "next step", "by ", "due"],
        "decision": ["decided", "finalized", "choose", "agreed"],
    }

    def predict(self, text: str, threshold: float) -> GateResult:
        lower = text.lower()
        probs: Dict[str, float] = {}
        for t, cues in self.CUES.items():
            hits = sum(1 for cue in cues if cue in lower)
            probs[t] = min(1.0, 0.3 * hits)
        top_type = max(probs, key=probs.get) if probs else "none"
        conf = probs.get(top_type, 0.0)
        worthy = conf >= threshold
        return GateResult(worthy=worthy, probs=probs, top_type=top_type, confidence=conf)


class MemoryGate:
    def __init__(self):
        self.settings = get_settings()
        self._pretrained = None
        self._rule = _RuleGate()
        if self.settings.memory_gate_enabled:
            self._load_or_fallback()

    def _load_or_fallback(self) -> None:
        # Try pretrained embeddings gate first if enabled
        if self.settings.memory_gate_pretrained_enabled:
            try:
                from sentence_transformers import SentenceTransformer
                import numpy as np

                class _PretrainedGate:
                    def __init__(self, model_name: str):
                        self.model = SentenceTransformer(model_name)
                        self.prototypes = {
                            "constraint": [
                                "must avoid",
                                "never do",
                                "cannot",
                                "do not allow",
                            ],
                            "preference": [
                                "I like",
                                "I prefer",
                                "I love",
                            ],
                            "task": [
                                "task:",
                                "todo",
                                "remind me to",
                                "by friday",
                            ],
                            "decision": [
                                "we decided",
                                "decision:",
                                "agreed to",
                            ],
                            "none": ["hello", "thanks"],
                        }
                        self.proto_embed = {
                            k: np.mean(self.model.encode(v, normalize_embeddings=True), axis=0) for k, v in self.prototypes.items()
                        }

                    def predict(self, text: str, threshold: float, type_threshold: float) -> GateResult:
                        vec = self.model.encode([text], normalize_embeddings=True)[0]
                        scores = {}
                        import numpy as np

                        for k, proto in self.proto_embed.items():
                            scores[k] = float(np.dot(vec, proto))
                        top_type = max(scores, key=scores.get)
                        conf = scores.get(top_type, 0.0)
                        worthy = conf >= threshold and top_type != "none"
                        if conf < type_threshold:
                            worthy = False
                        return GateResult(worthy=worthy, probs=scores, top_type=top_type, confidence=conf)

                self._pretrained = _PretrainedGate(self.settings.memory_gate_pretrained_model)
                logging.getLogger(__name__).info("Loaded pretrained memory gate model %s", self.settings.memory_gate_pretrained_model)
                self._loaded = True
                return
            except Exception as exc:  # pragma: no cover - defensive fallback
                logging.getLogger(__name__).warning("Pretrained gate unavailable (%s); using rule gate.", exc)

    def predict(self, text: str) -> GateResult:
        threshold = self.settings.memory_gate_threshold
        type_threshold = self.settings.memory_gate_type_threshold
        if self._pretrained:
            res = self._pretrained.predict(text, threshold, type_threshold)
            # Safety net: if pretrained says not worthy, fall back to rule gate to avoid false negatives
            if res.worthy:
                return res
        return self._rule.predict(text, threshold)
