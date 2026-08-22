"""Integration tests for the full Agentic RAG pipeline."""

from unittest.mock import MagicMock
from app.agent.analyzer import QueryAnalyzer
from app.agent.context_optimizer import ContextOptimizer
from app.agent.evidence_checker import EvidenceChecker
from app.agent.orchestrator import RAGOrchestrator
from app.agent.planner import QueryPlanner
from app.citations.extractor import CitationExtractor
from app.citations.validator import CitationValidator
from app.citations.verifier import AnswerVerifier
from app.models.generation import GenerationResult, ProviderType
from app.retrieval.rrf import ReciprocalRankFusion


def test_full_rag_pipeline_execution(sample_scored_chunks):
    # Mock retrieval and reranking
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value.chunks = sample_scored_chunks

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = sample_scored_chunks

    # Mock provider
    mock_provider = MagicMock()
    mock_provider.generate.return_value = GenerationResult(
        text="The system uses a hybrid retrieval pipeline [Source: sample_guide.pdf, p.1].",
        provider=ProviderType.GROQ,
        model="llama-3.3-70b-versatile",
    )

    orchestrator = RAGOrchestrator(
        hybrid_retriever=mock_retriever,
        reranker=mock_reranker,
        provider_manager=mock_provider,
        query_analyzer=QueryAnalyzer(mock_provider),
        query_planner=QueryPlanner(mock_provider),
        evidence_checker=EvidenceChecker(mock_provider),
        context_optimizer=ContextOptimizer(),
        citation_extractor=CitationExtractor(),
        citation_validator=CitationValidator(),
        answer_verifier=AnswerVerifier(mock_provider),
        max_retrieval_iterations=2,
    )

    result = orchestrator.query("How does the hybrid retrieval pipeline work?")

    assert result.query_id.startswith("q-")
    assert "hybrid retrieval pipeline" in result.answer
    assert len(result.citations) > 0
    assert result.citations[0].filename == "sample_guide.pdf"
    assert result.evidence_quality.is_sufficient is True
    assert len(result.agent_decisions) > 0
    assert result.total_latency_ms > 0
