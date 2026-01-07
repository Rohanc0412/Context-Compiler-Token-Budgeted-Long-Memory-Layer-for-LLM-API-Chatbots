# LLM Context Compiler and Memory Layer

Production-oriented API layer that stores full chat history outside the prompt, retrieves long-term context, compiles a strict-budget prompt, and calls an LLM provider. Core components: Event Store, Retrieval Index, Rolling Summary, Context Compiler, Evaluation Harness, and an upgraded memory pipeline.

## Quick start
- Copy `.env.example` to `.env` and adjust if needed.
- Run migrations: `alembic upgrade head` (inside the container or locally with `DATABASE_URL` set).
- Start stack: `docker compose up --build`.
- API at `http://localhost:8000`; metrics at `GET /metrics`.

## Local (non-docker) dev
```bash
python -m venv .venv && source .venv/bin/activate  # or Scripts\activate on Windows
pip install -r requirements.txt
export DATABASE_URL=postgresql+psycopg2://app:app@localhost:5432/app
alembic upgrade head
uvicorn app.main:app --reload
```

## Architecture highlights
- **Event Store**: Postgres (pgvector) stores events, chunks, embeddings. Optional redaction before embedding.
- **Retrieval Index**: vector similarity + recency/role rerank, dedupe near-duplicates, hybrid with structured memories.
- **Rolling Summary**: periodic JSON summary of constraints, preferences, open tasks, decisions; triggered every N turns, by token utilization, or manually.
- **Context Compiler**: deterministic prompt assembly under strict input token budget; stores `prompt_traces` with per-section counts and dropped items.
- **Memory pipeline (Fix set 4)**:
  - Stage 0: ML gate (TF-IDF + logistic regression, artifact at `app/assets/memory_gate.joblib`, fallback rule gate).
  - Stage 1: Rules + optional spaCy Matcher propose candidates with confidence.
  - Stage 2: Verifier rejects boilerplate/generic/noisy candidates, dedupes, and adjusts confidence.
  - Stage 3: Optional LLM structured extractor (flagged by `LLM_MEMORY_EXTRACTOR_ENABLED`) for low-confidence/no-candidate cases with per-session/hour rate limits and strict JSON validation.
  - Storage: dedupe/upsert by canonical key, conflict handling marks older constraints/preferences as outdated when contradicted. Role safety: default extracts only from `MEMORY_EXTRACT_ROLES=user`.
- **Evaluation Harness**: scripted suites comparing baselines and the compiler.
- **Worker**: Dramatiq + Redis for async summary/chunk jobs (core flows remain synchronous-safe).
- **Observability**: Prometheus histograms/counters; `GET /metrics`.
- **Privacy**: per-user memory toggle; delete/export endpoints; optional semantic cache; optional redaction.

## Compiler sections and trimming
Order: system and policy → developer rules → active constraints → task state → rolling summary → retrieved memories → recent transcript → current user message.
Trimming order: recent transcript first, then retrieved memories, then rolling summary. Constraints and task state are preserved unless the user message alone exceeds the budget. Each compile stores `prompt_traces` with section token counts, included ids, dropped items, and the compiled prompt.

## Memory pipeline usage
- `/chat` and `/events` automatically run the pipeline for allowed roles.
- LLM-based summary (optional) is controlled by `USE_LLM_SUMMARY`; memory LLM extractor is controlled by `LLM_MEMORY_EXTRACTOR_ENABLED`.
- spaCy matcher is optional; install the model with `python -m spacy download en_core_web_sm` for best results.

### LLM-based summarizer (optional)
- Enable with `USE_LLM_SUMMARY=true` (and OpenAI creds if using real LLM). Otherwise deterministic summary runs.
- Auto-triggers every `SUMMARY_INTERVAL` turns and on high prompt utilization (hysteresis via `SUMMARY_HIGH_UTILIZATION` / `SUMMARY_LOW_UTILIZATION`), or manually via `POST /summaries/run`.

### Example: clean memories via /chat
```bash
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" \
  -d '{"user_id":"u1","session_id":"s1","message":"I must avoid peanuts and I prefer vegan meals. Task: finish the quarterly report by Friday. We decided to use provider X for embeddings."}'

curl "http://localhost:8000/memory/export?user_id=u1"
# Expected memories:
# active_constraints: avoid peanuts
# preferences: vegan meals
# decisions: use provider x for embeddings
# tasks: finish the quarterly report by friday
```

## Required endpoints
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
- Pytest unit tests cover compiler trimming determinism, retrieval dedupe, memory pipeline (gate, verifier, dedupe, LLM trigger), and export/delete endpoints.
```bash
pytest
```

## Notes
- If you need spaCy matcher support, install the small English model: `python -m spacy download en_core_web_sm`.
- To use the memory LLM extractor, set `LLM_MEMORY_EXTRACTOR_ENABLED=true` and ensure an LLM client is configured; MockLLM remains default for tests.
