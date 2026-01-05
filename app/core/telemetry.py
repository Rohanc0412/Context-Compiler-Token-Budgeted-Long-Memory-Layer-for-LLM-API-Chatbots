import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import Counter, Histogram

REQUEST_LATENCY = Histogram(
    "context_compiler_request_latency_seconds",
    "End-to-end request latency",
    ["endpoint"],
)
STEP_LATENCY = Histogram(
    "context_compiler_step_latency_seconds",
    "Latency per pipeline step",
    ["step"],
)
ERROR_COUNTER = Counter(
    "context_compiler_errors_total",
    "Errors by endpoint",
    ["endpoint"],
)


@contextmanager
def record_step(step: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        STEP_LATENCY.labels(step=step).observe(time.perf_counter() - start)


@contextmanager
def record_request(endpoint: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    except Exception:
        ERROR_COUNTER.labels(endpoint=endpoint).inc()
        raise
    finally:
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)

