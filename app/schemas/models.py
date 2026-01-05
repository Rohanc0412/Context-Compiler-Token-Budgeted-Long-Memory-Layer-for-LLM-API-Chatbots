from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class EventCreate(BaseModel):
    user_id: str
    session_id: str
    role: str
    text: str
    metadata_json: Optional[Dict[str, Any]] = None


class EventResponse(BaseModel):
    event_id: UUID


class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    input_budget_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None


class ChatResponse(BaseModel):
    response_text: str
    request_id: UUID


class CompileRequest(BaseModel):
    user_id: str
    session_id: str
    message: str
    input_budget_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None


class CompileResponse(BaseModel):
    compiled_prompt: str
    trace_id: UUID
    token_counts: Dict[str, int]
    total_input_tokens: int


class SummaryRunRequest(BaseModel):
    user_id: str
    session_id: str


class SummaryRunResponse(BaseModel):
    summary_id: UUID
    summary_json: Dict[str, Any]


class MemoryExportResponse(BaseModel):
    memories: List[Dict[str, Any]]
    summaries: List[Dict[str, Any]]


class MemoryDeleteRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None


class EvalRunRequest(BaseModel):
    suite_name: Optional[str] = None


class EvalRunResponse(BaseModel):
    scorecard: Dict[str, Dict[str, Any]]

