from typing import List, Optional
from pydantic import BaseModel, validator


ALLOWED_TYPES = {"constraint", "preference", "task", "decision"}


class LLMMemoryItem(BaseModel):
    type: str
    text: str
    confidence: float
    due_date: Optional[str] = None
    status: Optional[str] = None

    @validator("type")
    def check_type(cls, v):
        if v not in ALLOWED_TYPES:
            raise ValueError("invalid type")
        return v

    @validator("text")
    def check_text(cls, v):
        if not (3 <= len(v) <= 200):
            raise ValueError("invalid text length")
        return v.strip()

    @validator("confidence")
    def check_conf(cls, v):
        if v < 0 or v > 1:
            raise ValueError("confidence out of range")
        return v


class LLMMemoryResponse(BaseModel):
    memories: List[LLMMemoryItem]
