from fastapi import APIRouter, Depends, HTTPException
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.telemetry import record_request, record_step
from app.db.session import get_db
from app.db import models
from app.schemas import models as schemas
from app.services.cache import SemanticCache
from app.services.compiler import CompilerInput, ContextCompiler
from app.services.event_store import EventStore
from app.services.llm_client import get_llm_client
from app.services.retrieval import RetrievalService
from app.services.summary import SummaryService
from app.services.eval_harness import EvalHarness
from app.core.config import get_settings

router = APIRouter()

event_store = EventStore()
retrieval_service = RetrievalService()
summary_service = SummaryService()
compiler = ContextCompiler()
llm_client = get_llm_client()
semantic_cache = SemanticCache()
eval_harness = EvalHarness()
settings = get_settings()


@router.post("/events", response_model=schemas.EventResponse)
def post_event(payload: schemas.EventCreate, db: Session = Depends(get_db)):
    with record_request("/events"):
        with record_step("store_event"):
            ev = event_store.store_event(
                db,
                user_id=payload.user_id,
                session_id=payload.session_id,
                role=payload.role,
                text=payload.text,
                metadata_json=payload.metadata_json,
            )
        return schemas.EventResponse(event_id=ev.id)


@router.post("/chat", response_model=schemas.ChatResponse)
def chat(payload: schemas.ChatRequest, db: Session = Depends(get_db)):
    with record_request("/chat"):
        user_event = event_store.store_event(
            db,
            user_id=payload.user_id,
            session_id=payload.session_id,
            role="user",
            text=payload.message,
        )
        cached = semantic_cache.get(payload.user_id, payload.message)
        if cached:
            return schemas.ChatResponse(response_text=cached, request_id=user_event.id)

        with record_step("retrieval"):
            retrieved_chunks, memories = retrieval_service.retrieve(
                db, payload.user_id, payload.session_id, payload.message
            )
        recent_events = (
            db.query(models.Event)
            .filter(models.Event.user_id == payload.user_id, models.Event.session_id == payload.session_id)
            .order_by(models.Event.created_at)
            .all()
        )
        summary = (
            db.query(models.Summary)
            .filter(models.Summary.user_id == payload.user_id, models.Summary.session_id == payload.session_id)
            .order_by(models.Summary.created_at.desc())
            .first()
        )

        event_count = len(recent_events)
        if event_count % settings.summary_interval == 0:
            summary = summary_service.generate_summary(db, payload.user_id, payload.session_id)

        compiler_input = CompilerInput(
            user_id=payload.user_id,
            session_id=payload.session_id,
            latest_user_message=payload.message,
            retrieved_chunks=retrieved_chunks,
            structured_memories=memories,
            recent_events=recent_events,
            summary=summary,
            input_budget_tokens=payload.input_budget_tokens,
            max_output_tokens=payload.max_output_tokens,
        )
        with record_step("compile"):
            compiled_prompt, trace = compiler.compile(db, compiler_input)

        with record_step("llm_call"):
            assistant_response = llm_client.chat(compiled_prompt, payload.max_output_tokens or settings.max_output_tokens)

        assistant_event = event_store.store_event(
            db,
            user_id=payload.user_id,
            session_id=payload.session_id,
            role="assistant",
            text=assistant_response,
        )
        semantic_cache.put(payload.user_id, payload.message, assistant_response)
        return schemas.ChatResponse(response_text=assistant_response, request_id=trace.request_id)


@router.post("/compile", response_model=schemas.CompileResponse)
def compile_only(payload: schemas.CompileRequest, db: Session = Depends(get_db)):
    with record_request("/compile"):
        with record_step("retrieval"):
            retrieved_chunks, memories = retrieval_service.retrieve(
                db, payload.user_id, payload.session_id, payload.message
            )
        recent_events = (
            db.query(models.Event)
            .filter(models.Event.user_id == payload.user_id, models.Event.session_id == payload.session_id)
            .order_by(models.Event.created_at)
            .all()
        )
        summary = (
            db.query(models.Summary)
            .filter(models.Summary.user_id == payload.user_id, models.Summary.session_id == payload.session_id)
            .order_by(models.Summary.created_at.desc())
            .first()
        )
        compiler_input = CompilerInput(
            user_id=payload.user_id,
            session_id=payload.session_id,
            latest_user_message=payload.message,
            retrieved_chunks=retrieved_chunks,
            structured_memories=memories,
            recent_events=recent_events,
            summary=summary,
            input_budget_tokens=payload.input_budget_tokens,
            max_output_tokens=payload.max_output_tokens,
        )
        with record_step("compile"):
            compiled_prompt, trace = compiler.compile(db, compiler_input)
        return schemas.CompileResponse(
            compiled_prompt=compiled_prompt,
            trace_id=trace.request_id,
            token_counts=trace.section_token_counts,
            total_input_tokens=trace.total_input_tokens,
        )


@router.post("/summaries/run", response_model=schemas.SummaryRunResponse)
def run_summary(payload: schemas.SummaryRunRequest, db: Session = Depends(get_db)):
    summary = summary_service.generate_summary(db, payload.user_id, payload.session_id)
    return schemas.SummaryRunResponse(summary_id=summary.summary_id, summary_json=summary.summary_json)


@router.get("/memory/export", response_model=schemas.MemoryExportResponse)
def memory_export(user_id: str, db: Session = Depends(get_db)):
    memories = (
        db.query(models.Memory).filter(models.Memory.user_id == user_id).order_by(models.Memory.created_at).all()
    )
    summaries = db.query(models.Summary).filter(models.Summary.user_id == user_id).all()
    return schemas.MemoryExportResponse(
        memories=[{"memory_id": str(m.memory_id), "type": m.type, "value": m.value_json} for m in memories],
        summaries=[{"summary_id": str(s.summary_id), "summary": s.summary_json} for s in summaries],
    )


@router.post("/memory/delete")
def memory_delete(payload: schemas.MemoryDeleteRequest, db: Session = Depends(get_db)):
    filters = [models.Memory.user_id == payload.user_id]
    chunk_filters = [models.Event.user_id == payload.user_id]
    summary_filters = [models.Summary.user_id == payload.user_id]
    if payload.session_id:
        filters.append(models.Memory.session_id == payload.session_id)
        chunk_filters.append(models.Event.session_id == payload.session_id)
        summary_filters.append(models.Summary.session_id == payload.session_id)
    db.query(models.Memory).filter(*filters).delete(synchronize_session=False)
    db.query(models.Summary).filter(*summary_filters).delete(synchronize_session=False)
    # delete chunks and embeddings
    events = db.query(models.Event.id).filter(*chunk_filters).subquery()
    chunks = db.query(models.EventChunk.chunk_id).filter(models.EventChunk.event_id.in_(events)).subquery()
    db.query(models.Embedding).filter(models.Embedding.chunk_id.in_(chunks)).delete(synchronize_session=False)
    db.query(models.EventChunk).filter(models.EventChunk.event_id.in_(events)).delete(synchronize_session=False)
    db.commit()
    return {"status": "deleted"}


@router.post("/eval/run", response_model=schemas.EvalRunResponse)
def eval_run(payload: schemas.EvalRunRequest, db: Session = Depends(get_db)):
    result = eval_harness.run(db, suite_name=payload.suite_name)
    return schemas.EvalRunResponse(scorecard=result)


@router.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
