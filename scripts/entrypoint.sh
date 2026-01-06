#!/usr/bin/env sh
set -e

cd /app
export PYTHONPATH=/app

# Wait for Postgres to be ready before running migrations
python - <<'PY'
import os, time, psycopg2

raw_url = os.environ.get("DB_WAIT_DSN") or os.environ.get(
    "DATABASE_URL", "postgresql+psycopg2://app:app@db:5432/app"
)
# Normalize SQLAlchemy-style URLs to a psycopg2-friendly DSN
dsn = raw_url
dsn = dsn.replace("postgresql+psycopg2", "postgresql")
dsn = dsn.replace("postgres+psycopg2", "postgresql")
dsn = dsn.replace("postgresql+psycopg", "postgresql")
dsn = dsn.replace("postgres+psycopg", "postgresql")

for i in range(30):
    try:
        conn = psycopg2.connect(dsn)
        conn.close()
        break
    except Exception as exc:  # pragma: no cover - startup wait
        print(f"[entrypoint] DB not ready ({exc}); retrying...")
        time.sleep(2)
else:
    raise SystemExit("Database not available after retries")
PY

# Run migrations idempotently on container start
alembic upgrade head

# Start the API
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
