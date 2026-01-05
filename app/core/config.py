import os
from functools import lru_cache
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    app_name: str = "LLM Context Compiler and Memory Layer"
    environment: str = Field("local", env="ENVIRONMENT")
    database_url: str = Field("postgresql+psycopg2://app:app@db:5432/app", env="DATABASE_URL")
    redis_url: str = Field("redis://redis:6379/0", env="REDIS_URL")
    enable_semantic_cache: bool = Field(False, env="ENABLE_SEMANTIC_CACHE")
    enable_openai_client: bool = Field(False, env="ENABLE_OPENAI_CLIENT")
    openai_api_key: str | None = Field(None, env="OPENAI_API_KEY")
    input_token_budget: int = Field(3000, env="INPUT_TOKEN_BUDGET")
    max_output_tokens: int = Field(512, env="MAX_OUTPUT_TOKENS")
    context_window: int = Field(4096, env="CONTEXT_WINDOW")
    retrieval_top_k: int = Field(12, env="RETRIEVAL_TOP_K")
    summary_interval: int = Field(10, env="SUMMARY_INTERVAL")
    tracing_enabled: bool = Field(True, env="TRACING_ENABLED")
    metrics_port: int = Field(8001, env="METRICS_PORT")
    worker_broker: str = Field("redis://redis:6379/0", env="WORKER_BROKER")
    worker_result_backend: str = Field("redis://redis:6379/1", env="WORKER_RESULT_BACKEND")
    redact_before_embedding: bool = Field(True, env="REDACT_BEFORE_EMBEDDING")
    semantic_cache_threshold: float = Field(0.92, env="SEMANTIC_CACHE_THRESHOLD")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()

