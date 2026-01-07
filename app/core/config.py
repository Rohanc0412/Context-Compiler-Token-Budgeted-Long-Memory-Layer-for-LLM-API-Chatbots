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
    summary_high_utilization: float = Field(0.9, env="SUMMARY_HIGH_UTILIZATION")
    summary_low_utilization: float = Field(0.7, env="SUMMARY_LOW_UTILIZATION")
    use_llm_summary: bool = Field(False, env="USE_LLM_SUMMARY")
    summary_model_name: str = Field("gpt-3.5-turbo", env="SUMMARY_MODEL_NAME")
    summary_max_output_tokens: int = Field(256, env="SUMMARY_MAX_OUTPUT_TOKENS")
    summary_window_events: int = Field(20, env="SUMMARY_WINDOW_EVENTS")
    memory_extraction_enabled: bool = Field(True, env="MEMORY_EXTRACTION_ENABLED")
    memory_extract_roles: str = Field("user", env="MEMORY_EXTRACT_ROLES")
    memory_gate_enabled: bool = Field(True, env="MEMORY_GATE_ENABLED")
    memory_gate_threshold: float = Field(0.55, env="MEMORY_GATE_THRESHOLD")
    memory_gate_type_threshold: float = Field(0.45, env="MEMORY_GATE_TYPE_THRESHOLD")
    memory_gate_pretrained_enabled: bool = Field(False, env="MEMORY_GATE_PRETRAINED_ENABLED")
    memory_gate_pretrained_model: str = Field("all-MiniLM-L6-v2", env="MEMORY_GATE_PRETRAINED_MODEL")
    spacy_matcher_enabled: bool = Field(True, env="SPACY_MATCHER_ENABLED")
    llm_memory_extractor_enabled: bool = Field(False, env="LLM_MEMORY_EXTRACTOR_ENABLED")
    llm_memory_extractor_min_conf: float = Field(0.75, env="LLM_MEMORY_EXTRACTOR_MIN_CONF")
    llm_memory_extractor_max_calls_per_session_per_hour: int = Field(
        10, env="LLM_MEMORY_EXTRACTOR_MAX_CALLS_PER_SESSION_PER_HOUR"
    )
    llm_memory_extractor_model_name: str = Field("mock", env="LLM_MEMORY_EXTRACTOR_MODEL_NAME")
    tracing_enabled: bool = Field(True, env="TRACING_ENABLED")
    metrics_port: int = Field(8001, env="METRICS_PORT")
    worker_broker: str = Field("redis://redis:6379/0", env="WORKER_BROKER")
    worker_result_backend: str = Field("redis://redis:6379/1", env="WORKER_RESULT_BACKEND")
    redact_before_embedding: bool = Field(True, env="REDACT_BEFORE_EMBEDDING")
    semantic_cache_threshold: float = Field(0.92, env="SEMANTIC_CACHE_THRESHOLD")
    embedding_model_name: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL_NAME")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
