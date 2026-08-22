"""
Evidence checker — determines whether retrieved evidence is sufficient.

Uses a combination of reranker scores and LLM assessment to decide
if the agent should proceed to generation or retry retrieval.
"""

from __future__ import annotations

import time

from app.generation.prompts import EVIDENCE_SUFFICIENCY_PROMPT, format_context
from app.generation.provider_manager import ProviderManager
from app.models.generation import EvidenceQuality
from app.models.retrieval import ScoredChunk
from app.observability.logging import get_logger

logger = get_logger(__name__)


class EvidenceChecker:
    """
    Evaluates whether retrieved evidence is sufficient to answer a query.

    Combines quantitative signals (reranker scores, chunk count) with
    LLM-based assessment for a holistic sufficiency decision.
    """

    def __init__(
        self,
        provider_manager: ProviderManager,
        sufficiency_threshold: float = 0.4,
    ):
        self.provider_manager = provider_manager
        self.sufficiency_threshold = sufficiency_threshold

    def check(
        self,
        query: str,
        chunks: list[ScoredChunk],
    ) -> EvidenceQuality:
        """
        Check evidence sufficiency for a query.

        Returns an EvidenceQuality assessment with a composite score.
        """
        if not chunks:
            return EvidenceQuality(
                score=0.0,
                is_sufficient=False,
                explanation="No evidence retrieved",
            )

        start = time.perf_counter()

        # Quantitative signals
        num_chunks = len(chunks)
        reranker_scores = [c.reranker_score or 0.0 for c in chunks if c.reranker_score]
        avg_reranker = sum(reranker_scores) / max(len(reranker_scores), 1)
        max_reranker = max(reranker_scores) if reranker_scores else 0.0

        # Compute quantitative score
        # Weighted: best chunk score (40%) + avg score (30%) + chunk count (30%)
        count_score = min(num_chunks / 5.0, 1.0)  # 5+ chunks = full score
        quant_score = (max_reranker * 0.4) + (avg_reranker * 0.3) + (count_score * 0.3)

        # Quick check: if quantitative score is very high, skip LLM
        if quant_score >= 0.7:
            elapsed = (time.perf_counter() - start) * 1000
            return EvidenceQuality(
                score=min(quant_score, 1.0),
                num_relevant_chunks=num_chunks,
                avg_reranker_score=round(avg_reranker, 4),
                evidence_coverage=quant_score,
                is_sufficient=True,
                explanation=f"Strong evidence: {num_chunks} chunks, avg reranker={avg_reranker:.3f}",
            )

        # LLM-based assessment for borderline cases
        llm_sufficient = self._llm_check(query, chunks)

        # Combine quantitative + LLM
        if llm_sufficient == "sufficient":
            final_score = max(quant_score, 0.6)
            is_sufficient = True
        elif llm_sufficient == "partial":
            final_score = max(quant_score, 0.4)
            is_sufficient = final_score >= self.sufficiency_threshold
        else:
            final_score = min(quant_score, 0.3)
            is_sufficient = False

        elapsed = (time.perf_counter() - start) * 1000

        result = EvidenceQuality(
            score=round(min(final_score, 1.0), 4),
            num_relevant_chunks=num_chunks,
            avg_reranker_score=round(avg_reranker, 4),
            evidence_coverage=round(quant_score, 4),
            is_sufficient=is_sufficient,
            explanation=(
                f"Evidence assessment: {llm_sufficient}, "
                f"{num_chunks} chunks, avg reranker={avg_reranker:.3f}"
            ),
        )

        logger.info(
            "evidence_check",
            score=result.score,
            sufficient=result.is_sufficient,
            chunks=num_chunks,
            avg_reranker=round(avg_reranker, 4),
            llm_assessment=llm_sufficient,
            latency_ms=round(elapsed, 2),
        )

        return result

    def _llm_check(self, query: str, chunks: list[ScoredChunk]) -> str:
        """LLM-based evidence sufficiency check. Returns 'sufficient'/'insufficient'/'partial'."""
        evidence_parts = []
        for chunk in chunks[:5]:  # Only check top 5 to save tokens
            evidence_parts.append({
                "text": chunk.text[:500],
                "filename": chunk.metadata.filename,
                "page_number": chunk.metadata.page_number,
                "section": chunk.metadata.section,
            })

        evidence_text = format_context(evidence_parts)

        prompt = EVIDENCE_SUFFICIENCY_PROMPT.format(
            query=query,
            evidence=evidence_text,
        )

        result = self.provider_manager.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=50,
        )

        if result.error:
            return "partial"  # Default on failure

        response = result.text.strip().lower()
        if "sufficient" in response and "insufficient" not in response:
            return "sufficient"
        elif "insufficient" in response:
            return "insufficient"
        elif "partial" in response:
            return "partial"
        return "partial"
