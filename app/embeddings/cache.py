"""
Embedding cache with deterministic content hashing.

Prevents re-embedding unchanged content. Uses xxhash for fast
deterministic hashing and a file-backed JSON store.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

import xxhash

from app.observability.logging import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """
    File-backed embedding cache using content hashing.

    If a chunk's content has not changed (same hash), its cached embedding
    is returned instead of making an API call.
    """

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.cache_dir / "embedding_cache.json"
        self._lock = Lock()
        self._cache: dict[str, list[float]] = {}
        self._hits = 0
        self._misses = 0
        self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if self._cache_file.exists():
            try:
                with open(self._cache_file, "r") as f:
                    self._cache = json.load(f)
                logger.info("embedding_cache_loaded", entries=len(self._cache))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("embedding_cache_load_failed", error=str(e))
                self._cache = {}

    def _save(self) -> None:
        """Persist cache to disk atomically."""
        tmp_file = self._cache_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w") as f:
                json.dump(self._cache, f)
            tmp_file.replace(self._cache_file)
        except OSError as e:
            logger.error("embedding_cache_save_failed", error=str(e))

    @staticmethod
    def content_hash(text: str) -> str:
        """Compute deterministic hash of text content."""
        return xxhash.xxh64(text.encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        """Look up cached embedding for given text."""
        key = self.content_hash(text)
        with self._lock:
            embedding = self._cache.get(key)
            if embedding is not None:
                self._hits += 1
                return embedding
            self._misses += 1
            return None

    def put(self, text: str, embedding: list[float]) -> None:
        """Store an embedding in the cache."""
        key = self.content_hash(text)
        with self._lock:
            self._cache[key] = embedding

    def put_batch(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """Store a batch of embeddings."""
        with self._lock:
            for text, emb in zip(texts, embeddings):
                key = self.content_hash(text)
                self._cache[key] = emb

    def save(self) -> None:
        """Explicitly persist cache to disk."""
        with self._lock:
            self._save()

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache = {}
            if self._cache_file.exists():
                self._cache_file.unlink()
            self._hits = 0
            self._misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(self._hits + self._misses, 1) * 100, 1),
        }
