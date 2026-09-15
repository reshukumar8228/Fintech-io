"""
Unit tests for Dataset-Aware Agentic Analytics Query Engine.
Tests natural language translation, query execution, dynamic chart generation,
fuzzy NLP typo handling, custom API configuration, and Developer Mode execution traces.
"""

import pytest
import pandas as pd
from src.agentic_query_engine import AgenticQueryEngine
from src.analytics_engine import AnalyticsEngine
from src.graph_agent import FraudGraphEngine, AgentIQAssistant


@pytest.fixture(scope="module")
def query_engine():
    return AgenticQueryEngine()


@pytest.fixture(scope="module")
def assistant():
    analytics = AnalyticsEngine()
    graph = FraudGraphEngine()
    return AgentIQAssistant(analytics, graph)


def test_bar_chart_comparison_query(query_engine):
    res = query_engine.answer_query("Compare Q3 sales by region")
    assert "title" in res
    assert "answer" in res
    assert isinstance(res["data"], pd.DataFrame)
    assert not res["data"].empty
    vis = res.get("visualization", {})
    assert vis.get("type") == "bar"
    assert "xField" in vis
    assert "yField" in vis
    assert "execution_trace" in res


def test_kyc_percentage_query(query_engine):
    res = query_engine.answer_query("How Much Percentage Kyc Verified.")
    assert "answer" in res
    assert isinstance(res["data"], pd.DataFrame)
    assert not res["data"].empty
    assert "kyc_status" in res["data"].columns or res["x"] == "kyc_status"
    assert "Verified" in res["answer"] or "VERIFIED" in str(res["data"])


def test_status_typo_comparison_query(query_engine):
    res = query_engine.answer_query("Comapre how much transaction successful, failed and pending.")
    assert "answer" in res
    assert isinstance(res["data"], pd.DataFrame)
    assert not res["data"].empty
    assert "status" in res["data"].columns or res["x"] == "status"
    assert "Successful" in res["answer"] or "SUCCESS" in str(res["data"])


def test_line_chart_trend_query(query_engine):
    res = query_engine.answer_query("Show monthly revenue trends")
    assert isinstance(res["data"], pd.DataFrame)
    assert not res["data"].empty
    vis = res.get("visualization", {})
    assert vis.get("type") == "line"
    assert vis.get("xField") in ["year_month", "date"]


def test_scatter_plot_query(query_engine):
    res = query_engine.answer_query("Show me a scatter plot of monthly income vs sales")
    assert isinstance(res["data"], pd.DataFrame)
    vis = res.get("visualization", {})
    assert vis.get("type") == "scatter"
    assert vis.get("xField") == "monthly_income"
    assert vis.get("yField") == "amount"


def test_factual_kpi_query(query_engine):
    res = query_engine.answer_query("What is total revenue in Q2?")
    assert "answer" in res
    assert "₹" in res["answer"] or "total" in res["answer"].lower()
    assert isinstance(res["data"], pd.DataFrame)


def test_dataset_discovery_query(query_engine):
    res = query_engine.answer_query("What data is available in this project?")
    assert res.get("metadata", {}).get("intent") == "discovery"
    assert "fact_unified_analytics" in res["answer"] or not res["data"].empty


def test_unanswerable_query(query_engine):
    res = query_engine.answer_query("What were sales in Tokyo?")
    assert res.get("metadata", {}).get("intent") == "unanswerable"
    assert "not available" in res["answer"].lower()
    assert res["data"].empty


def test_ambiguous_clarification_query(query_engine):
    res = query_engine.answer_query("Compare")
    assert res.get("metadata", {}).get("intent") == "clarification"
    assert "specify" in res["answer"].lower() or "clarify" in res["answer"].lower()


def test_custom_api_config(query_engine):
    query_engine.set_custom_api_config(api_key="TEST_KEY", provider="gemini", model_name="gemini-1.5-pro")
    assert query_engine.custom_api_key == "TEST_KEY"
    assert query_engine.custom_provider == "gemini"
    assert query_engine.custom_model == "gemini-1.5-pro"


def test_assistant_wrapper_integration(assistant):
    res = assistant.answer_query("Compare performance by merchant category")
    assert "answer" in res
    assert "text" in res
    assert "chart_type" in res
    assert res["chart_type"] == "bar"
    assert isinstance(res["data"], pd.DataFrame)
    assert not res["data"].empty
    assert "execution_trace" in res


def test_follow_up_context_handling(query_engine):
    history = [{"role": "user", "content": "Compare sales by merchant category"}]
    res = query_engine.answer_query("Now show that as a line chart", history=history)
    vis = res.get("visualization", {})
    assert vis.get("type") == "line"
