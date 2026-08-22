"""Unit tests for QueryAnalyzer."""

from app.agent.analyzer import QueryAnalyzer
from app.models.generation import QueryType


def test_heuristic_query_classification(mock_provider_manager):
    analyzer = QueryAnalyzer(mock_provider_manager)

    assert analyzer.classify("What is hybrid retrieval?") == QueryType.FACTUAL
    assert analyzer.classify("Compare dense retrieval vs BM25") == QueryType.COMPARISON
    assert analyzer.classify("How to configure ChromaDB step by step?") == QueryType.PROCEDURAL
    assert analyzer.classify("What is RRF? And how does BGE reranker work?") == QueryType.MULTI_PART
    assert analyzer.classify("Summarize the main architecture components.") == QueryType.SUMMARIZATION
