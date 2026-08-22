"""
Centralized configuration using Pydantic Settings.

All configurable parameters are defined here and loaded from environment
variables or .env file. No magic numbers should exist elsewhere in the codebase.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # --- API Keys ---
    google_api_key: str = Field(default="", description="Google AI Studio API key")
    groq_api_key: str = Field(default="", description="Groq API key")

    # --- Embedding ---
    embedding_model: str = Field(default="gemini-embedding-001")
    embedding_batch_size: int = Field(default=32, ge=1, le=100)
    embedding_dimension: int = Field(default=768)

    # --- Generation ---
    groq_model: str = Field(default="llama-3.3-70b-versatile")
    gemini_generation_model: str = Field(default="gemini-2.0-flash")
    generation_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    generation_max_tokens: int = Field(default=2048, ge=128, le=8192)

    # --- Chunking ---
    chunk_size_tokens: int = Field(default=400, ge=64, le=2048)
    chunk_overlap_tokens: int = Field(default=60, ge=0, le=512)

    # --- Retrieval ---
    dense_top_k: int = Field(default=50, ge=1, le=500)
    bm25_top_k: int = Field(default=50, ge=1, le=500)
    hybrid_candidates: int = Field(default=50, ge=1, le=500)
    rrf_k: int = Field(default=60, ge=1, le=200)

    # --- Reranking ---
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    reranker_top_k: int = Field(default=10, ge=1, le=100)
    reranker_threshold: float = Field(default=0.0)
    reranker_device: str = Field(default="cpu")

    # --- Agent ---
    max_retrieval_iterations: int = Field(default=3, ge=1, le=10)
    evidence_sufficiency_threshold: float = Field(default=0.4, ge=0.0, le=1.0)
    max_context_tokens: int = Field(default=6000, ge=512, le=32000)

    # --- Storage Paths ---
    chroma_persist_dir: str = Field(default="./chroma_data")
    bm25_index_dir: str = Field(default="./bm25_index")
    embedding_cache_dir: str = Field(default="./embedding_cache")
    upload_dir: str = Field(default="./uploads")

    # --- API ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    max_upload_size_mb: int = Field(default=50, ge=1, le=200)

    # --- Logging ---
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # --- Timeouts & Retries ---
    api_timeout_seconds: int = Field(default=30, ge=5, le=120)
    max_retries: int = Field(default=3, ge=1, le=10)
    retry_backoff_base: float = Field(default=2.0, ge=1.0, le=5.0)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }

    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir).resolve()

    @property
    def bm25_index_path(self) -> Path:
        return Path(self.bm25_index_dir).resolve()

    @property
    def embedding_cache_path(self) -> Path:
        return Path(self.embedding_cache_dir).resolve()

    @property
    def upload_path(self) -> Path:
        return Path(self.upload_dir).resolve()

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for path in [
            self.chroma_persist_path,
            self.bm25_index_path,
            self.embedding_cache_path,
            self.upload_path,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def validate_api_keys(self) -> list[str]:
        """Return a list of missing API key warnings."""
        warnings = []
        if not self.google_api_key:
            warnings.append("GOOGLE_API_KEY is not set — embeddings and Gemini fallback will fail")
        if not self.groq_api_key:
            warnings.append("GROQ_API_KEY is not set — Groq generation will fail")
        return warnings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
