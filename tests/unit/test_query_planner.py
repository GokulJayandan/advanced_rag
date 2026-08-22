"""Unit tests for QueryPlanner."""

from app.agent.planner import QueryPlanner
from app.models.generation import QueryType


def test_query_planner_simple_strategy(mock_provider_manager):
    planner = QueryPlanner(mock_provider_manager)

    plan = planner.plan(query="What is BM25?", query_type=QueryType.FACTUAL)
    assert plan["strategy"] == "simple"
    assert len(plan["queries"]) == 1
    assert plan["queries"][0] == "What is BM25?"


def test_query_planner_decomposition(mock_provider_manager):
    mock_provider_manager.generate.return_value.text = "What is dense retrieval?\nWhat is sparse retrieval?\nHow do they differ?"
    planner = QueryPlanner(mock_provider_manager)

    plan = planner.plan(query="Compare dense and sparse retrieval", query_type=QueryType.COMPARISON)
    assert plan["strategy"] == "decompose"
    assert len(plan["queries"]) == 3
