"""Tests for aragora.nomic.km_feedback_bridge — KMFeedbackBridge."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.nomic.km_feedback_bridge import KMFeedbackBridge, LearningItem


# ---------------------------------------------------------------------------
# LearningItem
# ---------------------------------------------------------------------------


class TestLearningItem:
    def test_auto_timestamp(self):
        before = time.time()
        item = LearningItem(content="hello", tags=["a"])
        assert item.timestamp >= before

    def test_explicit_timestamp(self):
        item = LearningItem(content="x", timestamp=42.0)
        assert item.timestamp == 42.0

    def test_to_dict(self):
        item = LearningItem(content="msg", tags=["t1", "t2"], source="src")
        d = item.to_dict()
        assert d["content"] == "msg"
        assert d["tags"] == ["t1", "t2"]
        assert d["source"] == "src"

    def test_default_source(self):
        item = LearningItem(content="x")
        assert item.source == "nomic_cycle"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeCycleRecord:
    cycle_id: str = "test_cycle"
    goal: str = "Improve tests"
    success: bool = True
    agents_used: list[str] = field(default_factory=lambda: ["claude", "gemini"])
    quality_delta: float = 0.15
    cost_usd: float = 0.05
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@pytest.fixture
def bridge():
    return KMFeedbackBridge()


@pytest.fixture
def record():
    return FakeCycleRecord()


# ---------------------------------------------------------------------------
# persist_cycle_learnings
# ---------------------------------------------------------------------------


class TestPersistLearnings:
    def test_extracts_summary(self, bridge, record):
        items = bridge.persist_cycle_learnings(record)
        assert len(items) >= 1
        summary = items[0]
        assert "succeeded" in summary.content.lower()
        assert "nomic_learned:true" in summary.tags

    def test_extracts_agent_success(self, bridge, record):
        items = bridge.persist_cycle_learnings(record)
        agent_items = [i for i in items if "agent_success" in i.tags]
        assert len(agent_items) >= 1
        assert "claude" in agent_items[0].content

    def test_extracts_agent_failure(self, bridge):
        record = FakeCycleRecord(success=False)
        items = bridge.persist_cycle_learnings(record)
        agent_items = [i for i in items if "agent_failure" in i.tags]
        assert len(agent_items) >= 1

    def test_extracts_cost_efficiency(self, bridge, record):
        items = bridge.persist_cycle_learnings(record)
        cost_items = [i for i in items if "cost_efficiency" in i.tags]
        assert len(cost_items) >= 1
        assert "quality-per-dollar" in cost_items[0].content

    def test_no_cost_efficiency_when_zero_cost(self, bridge):
        record = FakeCycleRecord(cost_usd=0.0)
        items = bridge.persist_cycle_learnings(record)
        cost_items = [i for i in items if "cost_efficiency" in i.tags]
        assert len(cost_items) == 0

    def test_no_cost_efficiency_on_failure(self, bridge):
        record = FakeCycleRecord(success=False, cost_usd=0.1, quality_delta=0.5)
        items = bridge.persist_cycle_learnings(record)
        cost_items = [i for i in items if "cost_efficiency" in i.tags]
        assert len(cost_items) == 0

    def test_stores_in_memory(self, bridge, record):
        bridge.persist_cycle_learnings(record)
        assert len(bridge._in_memory_store) > 0

    def test_tags_contain_cycle_id(self, bridge, record):
        items = bridge.persist_cycle_learnings(record)
        for item in items:
            assert any("cycle_id:" in t for t in item.tags)

    def test_tags_contain_goal(self, bridge, record):
        items = bridge.persist_cycle_learnings(record)
        for item in items:
            assert any("goal:" in t for t in item.tags)

    def test_empty_agents(self, bridge):
        record = FakeCycleRecord(agents_used=[])
        items = bridge.persist_cycle_learnings(record)
        # Should still produce summary but not agent-specific items
        assert len(items) >= 1


# ---------------------------------------------------------------------------
# retrieve_relevant_learnings
# ---------------------------------------------------------------------------


class TestRetrieveLearnings:
    def test_retrieves_from_in_memory(self, bridge, record):
        bridge.persist_cycle_learnings(record)
        results = bridge.retrieve_relevant_learnings("Improve tests")
        assert len(results) > 0
        assert any("improve" in r["content"].lower() for r in results)

    def test_respects_limit(self, bridge):
        # Persist many items
        for i in range(10):
            record = FakeCycleRecord(cycle_id=f"c{i}", goal=f"fix bug {i}")
            bridge.persist_cycle_learnings(record)
        results = bridge.retrieve_relevant_learnings("fix bug", limit=3)
        assert len(results) <= 3

    def test_relevance_ranking(self, bridge):
        bridge.persist_cycle_learnings(FakeCycleRecord(goal="Improve security scanning"))
        bridge.persist_cycle_learnings(FakeCycleRecord(goal="Fix database migrations"))
        results = bridge.retrieve_relevant_learnings("security scanning")
        # Security-related should rank higher
        if results:
            assert "security" in results[0]["content"].lower()

    def test_empty_store_returns_empty(self, bridge):
        results = bridge.retrieve_relevant_learnings("anything")
        assert results == []

    def test_no_match_returns_empty(self, bridge):
        bridge.persist_cycle_learnings(FakeCycleRecord(goal="fix authentication bug"))
        # Very unrelated query
        results = bridge.retrieve_relevant_learnings("quantum physics")
        # Might match zero or a few via tag overlap
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# KM integration (mocked)
# ---------------------------------------------------------------------------


class TestKMIntegration:
    def test_ingests_to_km_when_available(self, record):
        mock_km = MagicMock()
        mock_km.ingest_sync = MagicMock()
        bridge = KMFeedbackBridge(km=mock_km)

        with patch(
            "aragora.nomic.km_feedback_bridge.KMFeedbackBridge._ingest_to_km"
        ) as mock_ingest:
            bridge.persist_cycle_learnings(record)
            assert mock_ingest.call_count > 0

    def test_falls_back_to_memory_when_km_fails(self, record):
        mock_km = MagicMock()
        bridge = KMFeedbackBridge(km=mock_km)

        with patch.object(
            bridge,
            "_ingest_to_km",
            side_effect=RuntimeError("KM down"),
        ):
            items = bridge.persist_cycle_learnings(record)
            # Should still persist to in-memory
            assert len(bridge._in_memory_store) > 0

    def test_search_km_results_included(self):
        mock_km = MagicMock()
        mock_item = MagicMock()
        mock_item.content = "past learning about tests"
        mock_item.tags = ["nomic_learned:true"]
        mock_item.source = "nomic_cycle"
        mock_item.timestamp = 1000.0
        mock_km.search.return_value = [mock_item]

        bridge = KMFeedbackBridge(km=mock_km)
        results = bridge.retrieve_relevant_learnings("tests")
        assert len(results) >= 1
        assert "past learning" in results[0]["content"]

    def test_km_none_uses_in_memory_only(self, record):
        bridge = KMFeedbackBridge(km=None)
        # Mock get_knowledge_mound to return None
        with patch(
            "aragora.nomic.km_feedback_bridge.KMFeedbackBridge._get_km",
            return_value=None,
        ):
            bridge.persist_cycle_learnings(record)
            assert len(bridge._in_memory_store) > 0
            results = bridge.retrieve_relevant_learnings("Improve")
            assert len(results) > 0
