# LLM Context Compiler and Memory Layer

Production-oriented API layer that stores full chat history outside the prompt, retrieves long-term context, compiles a strict-budget prompt, and calls an LLM provider. Core components: Event Store, Retrieval Index, Rolling Summary, Context Compiler, Evaluation Harness.

## Quick start
- Copy `.env.example` to `.env` and adjust if needed.
- Run migrations: `alembic upgrade head` (inside the container or locally with `DATABASE_URL` set).
- Start stack: `docker compose up --build`.
- API available at `http://localhost:8000`.
- Metrics at `GET /metrics` (Prometheus format).

## Local (non-docker) dev
```bash
python -m venv .venv && source .venv/bin/activate  # or Scripts\\activate on Windows
pip install -r requirements.txt
export DATABASE_URL=postgresql+psycopg2://app:app@localhost:5432/app
alembic upgrade head
uvicorn app.main:app --reload
```

## Architecture highlights
- **Event Store**: Postgres (pgvector) stores events, chunks, embeddings. Optional redaction before embedding.
- **Retrieval Index**: vector similarity + recency/role rerank, dedupe near-duplicates, hybrid with structured memories.
- **Rolling Summary**: periodic JSON summary of constraints, preferences, open tasks, decisions; triggered every N turns or manually.
- **Context Compiler**: deterministic prompt assembly under strict input token budget; stores `prompt_traces` with per-section counts and dropped items.
- **Evaluation Harness**: scripted suites (needle recall, constraint adherence, temporal update, distractors, long task continuity) comparing naive baseline, summary baseline, and proposed pipeline.
- **Worker**: Dramatiq + Redis for async summary/chunk jobs (core flows are synchronous-safe).
- **Observability**: timings via Prometheus histograms/counters; `GET /metrics`.
- **Privacy**: per-user memory toggle; delete/export endpoints; optional semantic cache (off by default); optional redaction of emails/phones before embedding.

## Compiler sections and trimming
Order: system and policy → developer rules → active constraints → task state → rolling summary → retrieved memories → recent transcript → current user message.
Trimming order: recent transcript first, then retrieved memories, then rolling summary. Constraints and task state are preserved unless the user message alone exceeds the budget. Each compile stores `prompt_traces` with section token counts, included ids, dropped items, and the compiled prompt.

## Required endpoints (all implemented)
- `POST /events`
- `POST /chat`
- `POST /compile`
- `POST /summaries/run`
- `GET /memory/export`
- `POST /memory/delete`
- `POST /eval/run`
- `GET /metrics`

## Example calls
```bash
# Store an event
curl -X POST http://localhost:8000/events -H "Content-Type: application/json" \
  -d '{"user_id":"u1","session_id":"s1","role":"user","text":"I must avoid peanuts."}'

# Chat (compile + call LLM)
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"user_id":"u1","session_id":"s1","message":"Can I have satay?"}'

# Compile only (no LLM call)
curl -X POST http://localhost:8000/compile -H "Content-Type: application/json" \
  -d '{"user_id":"u1","session_id":"s1","message":"What did I ask?"}'

# Run summary
curl -X POST http://localhost:8000/summaries/run -H "Content-Type: application/json" \
  -d '{"user_id":"u1","session_id":"s1"}'

# Export and delete memory
curl "http://localhost:8000/memory/export?user_id=u1"
curl -X POST http://localhost:8000/memory/delete -H "Content-Type: application/json" \
  -d '{"user_id":"u1","session_id":"s1"}'

# Run eval suites
curl -X POST http://localhost:8000/eval/run -H "Content-Type: application/json" -d '{}'
```

## Data model (tables)
`events`, `event_chunks`, `embeddings`, `memories`, `summaries`, `prompt_traces`, `eval_runs`, plus `user_configs` for per-user privacy toggles.

## Evaluation harness and plots
- `/eval/run` executes scripted suites using the mock LLM; results saved to `eval_runs`.
- `python scripts/plot_eval.py` generates `prompt_tokens_per_turn.png` and `success_rate.png` from stored runs.

## Tests
- Pytest unit tests cover compiler trimming determinism, retrieval dedupe, and export/delete endpoints.
```bash
pytest
```

