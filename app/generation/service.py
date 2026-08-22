"""Grounded answer generation with provider fallback."""
from __future__ import annotations
from app.generation.base import GenerationProvider
from app.models.generation import GenerationResult
from app.models.retrieval import ScoredChunk
from app.observability.logging import get_logger

logger = get_logger(__name__)
SYSTEM_PROMPT = """You answer only from the supplied document evidence. Never use outside knowledge,
invent facts, sources, pages, or citations. Every factual sentence must end with one of the exact
citations supplied in the evidence. If the evidence is insufficient, say so plainly."""

class GenerationService:
    def __init__(self, primary: GenerationProvider, fallback: GenerationProvider | None = None):
        self.primary, self.fallback = primary, fallback

    @staticmethod
    def _citation(chunk: ScoredChunk) -> str:
        meta = chunk.metadata
        location = f"p.{meta.page_number}" if meta.page_number is not None else (meta.section or "document")
        return f"[Source: {meta.filename}, {location}]"

    def generate(self, query: str, chunks: list[ScoredChunk]) -> GenerationResult:
        evidence = "\n\n".join(f"Evidence {i + 1} | citation: {self._citation(chunk)}\n{chunk.text}" for i, chunk in enumerate(chunks))
        prompt = f"Question: {query}\n\nDocument evidence:\n{evidence}\n\nAnswer the question using only this evidence."
        try:
            return self.primary.generate(SYSTEM_PROMPT, prompt)
        except Exception as primary_error:
            logger.warning("primary_generation_failed", error=str(primary_error))
            if not self.fallback:
                raise
            result = self.fallback.generate(SYSTEM_PROMPT, prompt)
            result.fallback_used = True
            return result
