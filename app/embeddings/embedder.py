"""
Gemini embedding client with batching, caching, retry, and rate-limit handling.

Uses the google-genai SDK. Unchanged content is served from cache
instead of making API calls.
"""

from __future__ import annotations

import time
from typing import Sequence

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.embeddings.cache import EmbeddingCache
from app.observability.logging import get_logger

logger = get_logger(__name__)


class GeminiEmbedder:
    """
    Embeds text using Gemini Embedding API with caching and retry logic.

    Features:
    - Batch embedding to minimize API calls
    - Exponential backoff on rate limits
    - Content-hash-based caching to skip unchanged content
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        batch_size: int = 32,
        cache_dir: str = "./embedding_cache",
        max_retries: int = 3,
    ):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required for embeddings")

        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.cache = EmbeddingCache(cache_dir)
        self._api_calls = 0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def _embed_batch_api(self, texts: list[str]) -> list[list[float]]:
        """Call the Gemini embedding API for a batch of texts."""
        self._api_calls += 1

        result = self._client.models.embed_content(
            model=self.model,
            contents=texts,
        )

        return [list(emb.values) for emb in result.embeddings]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts, using cache where possible.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors in the same order.
        """
        if not texts:
            return []

        # Check cache first
        results: list[list[float] | None] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        if not uncached_texts:
            logger.debug("all_embeddings_cached", count=len(texts))
            return results  # type: ignore  — all are non-None

        # Batch API calls for uncached texts
        logger.info(
            "embedding_texts",
            total=len(texts),
            cached=len(texts) - len(uncached_texts),
            to_embed=len(uncached_texts),
        )

        all_embeddings: list[list[float]] = []
        for batch_start in range(0, len(uncached_texts), self.batch_size):
            batch = uncached_texts[batch_start : batch_start + self.batch_size]
            try:
                batch_embeddings = self._embed_batch_api(batch)
                all_embeddings.extend(batch_embeddings)

                # Cache the results
                self.cache.put_batch(batch, batch_embeddings)

                # Rate limit courtesy pause
                if batch_start + self.batch_size < len(uncached_texts):
                    time.sleep(0.5)

            except Exception as e:
                logger.error(
                    "embedding_batch_failed",
                    batch_start=batch_start,
                    batch_size=len(batch),
                    error=str(e),
                )
                raise

        # Fill in uncached results
        for i, idx in enumerate(uncached_indices):
            results[idx] = all_embeddings[i]

        # Persist cache
        self.cache.save()

        return results  # type: ignore

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        results = self.embed_texts([query])
        return results[0]

    @property
    def api_calls(self) -> int:
        return self._api_calls

    @property
    def cache_stats(self) -> dict:
        return self.cache.stats
