"""
Application settings — load from environment / .env (no secrets committed).
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from tck_graphrag._paths import ENV_FILE


class Settings(BaseSettings):
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = ""

    # PostgreSQL (BasicRAG / pgvector)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "postgres"
    postgres_user: str = "postgres"
    postgres_password: str = ""

    # API keys — leave empty; set in your local .env only
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # LLM defaults (evaluation: gpt-4o vs claude-sonnet-4-6)
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4o"

    # BasicRAG retrieval (fixed top-10 matches published baseline)
    basic_rag_adaptive_retrieval: bool = False
    basic_rag_top_k_max: int = 25
    basic_rag_top_k_default: int = 10
    basic_rag_top_k_min: int = 2
    basic_rag_min_similarity: float = 0.42
    basic_rag_dynamic_k: bool = True
    basic_rag_dynamic_delta: float = 0.12

    app_name: str = "TCK-QA-150"
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()
