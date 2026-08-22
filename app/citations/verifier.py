"""
Answer verification — checks if the generated answer is grounded in evidence.

Uses LLM to verify claims against retrieved context. Returns a structured
verification result with unsupported claims, contradictions, and coverage.
"""

from __future__ import annotations

import re

from app.generation.provider_manager import ProviderManager
from app.generation.prompts import ANSWER_VERIFICATION_PROMPT, format_context
from app.models.generation import VerificationResult
from app.models.retrieval import ScoredChunk
from app.observability.logging import get_logger

logger = get_logger(__name__)


class AnswerVerifier:
    """
    Verifies that generated answers are grounded in retrieved evidence.

    Uses the LLM to check for unsupported claims, contradictions,
    and fabricated citations.
    """

    def __init__(self, provider_manager: ProviderManager):
        self.provider_manager = provider_manager

    def verify(
        self,
        query: str,
        answer: str,
        chunks: list[ScoredChunk],
    ) -> VerificationResult:
        """
        Verify answer against retrieved evidence.

        Args:
            query: The original query.
            answer: The generated answer to verify.
            chunks: The retrieved chunks used as evidence.

        Returns:
            VerificationResult with groundedness assessment.
        """
        if not answer or not chunks:
            return VerificationResult(
                is_grounded=False,
                evidence_coverage=0.0,
                verification_details="No answer or evidence to verify",
            )

        # Format evidence
        evidence_parts = []
        for chunk in chunks:
            evidence_parts.append({
                "text": chunk.text,
                "filename": chunk.metadata.filename,
                "page_number": chunk.metadata.page_number,
                "section": chunk.metadata.section,
            })
        evidence_text = format_context(evidence_parts)

        # Build verification prompt
        prompt = ANSWER_VERIFICATION_PROMPT.format(
            query=query,
            answer=answer,
            evidence=evidence_text,
        )

        result = self.provider_manager.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=512,
        )

        if result.error:
            logger.warning("verification_failed", error=result.error)
            return VerificationResult(
                is_grounded=True,  # Default to grounded if verification fails
                evidence_coverage=0.5,
                verification_details=f"Verification failed: {result.error}",
            )

        return self._parse_verification(result.text)

    def _parse_verification(self, response: str) -> VerificationResult:
        """Parse the structured verification response."""
        response_lower = response.lower()

        # Parse GROUNDED
        is_grounded = True
        grounded_match = re.search(r'grounded:\s*(yes|no)', response_lower)
        if grounded_match:
            is_grounded = grounded_match.group(1) == "yes"

        # Parse COVERAGE
        coverage = 0.5
        coverage_match = re.search(r'coverage:\s*(high|medium|low)', response_lower)
        if coverage_match:
            coverage_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
            coverage = coverage_map.get(coverage_match.group(1), 0.5)

        # Parse UNSUPPORTED_CLAIMS
        unsupported: list[str] = []
        unsupported_match = re.search(
            r'unsupported_claims:\s*\[?(.*?)\]?(?:\n|$)',
            response, re.IGNORECASE
        )
        if unsupported_match:
            claims_text = unsupported_match.group(1).strip()
            if claims_text.lower() not in ("none", '\"none\"', ""):
                unsupported = [c.strip().strip('"\'') for c in claims_text.split(",") if c.strip()]

        # Parse CONTRADICTIONS
        contradictions: list[str] = []
        contra_match = re.search(
            r'contradictions:\s*\[?(.*?)\]?(?:\n|$)',
            response, re.IGNORECASE
        )
        if contra_match:
            contra_text = contra_match.group(1).strip()
            if contra_text.lower() not in ("none", '\"none\"', ""):
                contradictions = [c.strip().strip('"\'') for c in contra_text.split(",") if c.strip()]

        return VerificationResult(
            is_grounded=is_grounded,
            evidence_coverage=coverage,
            unsupported_claims=unsupported,
            contradictions=contradictions,
            verification_details=response,
        )
