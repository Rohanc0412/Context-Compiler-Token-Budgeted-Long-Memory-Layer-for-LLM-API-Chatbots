import dramatiq
from dramatiq.brokers.redis import RedisBroker

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.db import models
from app.services.event_store import EventStore
from app.services.summary import SummaryService

settings = get_settings()
broker = RedisBroker(url=settings.worker_broker)
dramatiq.set_broker(broker)


@dramatiq.actor
def run_summary_job(user_id: str, session_id: str):
    with SessionLocal() as db:
        SummaryService().generate_summary(db, user_id, session_id)


@dramatiq.actor
def chunk_event_job(event_id: str):
    with SessionLocal() as db:
        event = db.get(models.Event, event_id)
        if event:
            EventStore().chunk_and_embed(db, event, enable_memory=True)

