"""Unit tests for AnswerVerifier."""

from app.citations.verifier import AnswerVerifier


def test_answer_verifier_parsing(mock_provider_manager, sample_scored_chunks):
    mock_provider_manager.generate.return_value.text = (
        "GROUNDED: yes\n"
        "COVERAGE: high\n"
        "UNSUPPORTED_CLAIMS: none\n"
        "CONTRADICTIONS: none\n"
        "INVALID_CITATIONS: none"
    )

    verifier = AnswerVerifier(mock_provider_manager)
    res = verifier.verify("query", "answer", sample_scored_chunks)

    assert res.is_grounded is True
    assert res.evidence_coverage == 0.9
    assert res.unsupported_claims == []
    assert res.contradictions == []


def test_answer_verifier_unsupported_claims(mock_provider_manager, sample_scored_chunks):
    mock_provider_manager.generate.return_value.text = (
        "GROUNDED: no\n"
        "COVERAGE: low\n"
        "UNSUPPORTED_CLAIMS: [The company was founded in 1902]\n"
        "CONTRADICTIONS: none\n"
        "INVALID_CITATIONS: none"
    )

    verifier = AnswerVerifier(mock_provider_manager)
    res = verifier.verify("query", "answer", sample_scored_chunks)

    assert res.is_grounded is False
    assert res.evidence_coverage == 0.3
    assert len(res.unsupported_claims) == 1
