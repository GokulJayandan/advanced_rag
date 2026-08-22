"""Unit tests for EvidenceChecker."""

from app.agent.evidence_checker import EvidenceChecker


def test_evidence_checker_empty_chunks(mock_provider_manager):
    checker = EvidenceChecker(mock_provider_manager)
    quality = checker.check("Sample query", [])
    assert quality.score == 0.0
    assert quality.is_sufficient is False


def test_evidence_checker_strong_evidence(mock_provider_manager, sample_scored_chunks):
    checker = EvidenceChecker(mock_provider_manager, sufficiency_threshold=0.4)
    quality = checker.check("Sample query", sample_scored_chunks)
    assert quality.score > 0.4
    assert quality.is_sufficient is True
    assert quality.num_relevant_chunks == len(sample_scored_chunks)
