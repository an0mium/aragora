"""Tests for DecisionContextPreloader (Task 7, #811).

Covers:
- preload returns correct structure with all keys
- All None subsystems return empty defaults
- Mock knowledge bridge returns precedents
- Mock memory returns relevant knowledge
- Mock calibration returns agent scores
- Failing subsystem degrades gracefully (no exception, empty results)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.pipeline.km_bridge import PipelineKMBridge
from aragora.pipeline.decision_integrity import DecisionContextPreloader


class TestPreDebateContextPreloader:
    """Test DecisionContextPreloader.preload() behaviour."""

    def test_preload_returns_correct_structure(self) -> None:
        """preload() should always return a dict with the three expected keys."""
        preloader = DecisionContextPreloader()
        ctx = preloader.preload(task="Design a rate limiter", domain="software")
        assert "precedents" in ctx
        assert "relevant_knowledge" in ctx
        assert "agent_calibration" in ctx

    def test_preload_with_all_none_subsystems(self) -> None:
        """When all subsystems are None, preload() returns empty defaults."""
        preloader = DecisionContextPreloader(knowledge_bridge=None, memory=None, calibration=None)
        ctx = preloader.preload(task="Any task")
        assert ctx["precedents"] == []
        assert ctx["relevant_knowledge"] == []
        assert ctx["agent_calibration"] == {}

    def test_preload_with_knowledge_bridge(self) -> None:
        """A mock knowledge bridge should populate precedents."""
        mock_bridge = MagicMock()
        mock_bridge.query_precedents.return_value = [
            {"pipeline_id": "p1", "topic": "rate limiter"},
            {"pipeline_id": "p2", "topic": "token bucket"},
        ]

        preloader = DecisionContextPreloader(knowledge_bridge=mock_bridge)
        ctx = preloader.preload(task="Design a rate limiter")

        assert len(ctx["precedents"]) == 2
        assert ctx["precedents"][0]["pipeline_id"] == "p1"
        mock_bridge.query_precedents.assert_called_once_with("Design a rate limiter", limit=5)

    def test_preload_with_memory(self) -> None:
        """A mock continuum memory should populate relevant_knowledge."""
        entry1 = MagicMock()
        entry1.to_dict.return_value = {"content": "prior debate insight"}
        entry2 = MagicMock()
        entry2.to_dict.return_value = {"content": "second insight"}

        mock_memory = MagicMock()
        mock_memory.retrieve.return_value = [entry1, entry2]

        preloader = DecisionContextPreloader(memory=mock_memory)
        ctx = preloader.preload(task="Design a cache")

        assert len(ctx["relevant_knowledge"]) == 2
        assert ctx["relevant_knowledge"][0]["content"] == "prior debate insight"
        mock_memory.retrieve.assert_called_once_with(query="Design a cache", limit=10)

    def test_preload_with_memory_no_to_dict(self) -> None:
        """Memory entries without to_dict are stringified."""
        mock_memory = MagicMock()
        mock_memory.retrieve.return_value = ["plain string entry"]

        preloader = DecisionContextPreloader(memory=mock_memory)
        ctx = preloader.preload(task="task")

        assert len(ctx["relevant_knowledge"]) == 1
        assert ctx["relevant_knowledge"][0] == {"content": "plain string entry"}

    def test_preload_with_calibration(self) -> None:
        """A mock calibration source should populate agent_calibration."""
        agent1 = MagicMock()
        agent1.agent_name = "claude-sonnet"
        agent1.elo = 1520.0

        agent2 = MagicMock()
        agent2.agent_name = "gpt-4o"
        agent2.elo = 1480.0

        mock_calibration = MagicMock()
        mock_calibration.get_top_agents_for_domain.return_value = [agent1, agent2]

        preloader = DecisionContextPreloader(calibration=mock_calibration)
        ctx = preloader.preload(task="Code review", domain="software")

        assert ctx["agent_calibration"] == {
            "claude-sonnet": 1520.0,
            "gpt-4o": 1480.0,
        }
        mock_calibration.get_top_agents_for_domain.assert_called_once_with("software", limit=5)

    def test_preload_calibration_skipped_without_domain(self) -> None:
        """Calibration is not queried when domain is None."""
        mock_calibration = MagicMock()
        preloader = DecisionContextPreloader(calibration=mock_calibration)
        ctx = preloader.preload(task="Anything", domain=None)

        assert ctx["agent_calibration"] == {}
        mock_calibration.get_top_agents_for_domain.assert_not_called()

    def test_preload_knowledge_bridge_failure_degrades_gracefully(self) -> None:
        """If knowledge bridge raises, precedents stay empty and no exception."""
        mock_bridge = MagicMock()
        mock_bridge.query_precedents.side_effect = RuntimeError("KM unavailable")

        preloader = DecisionContextPreloader(knowledge_bridge=mock_bridge)
        ctx = preloader.preload(task="task")

        assert ctx["precedents"] == []

    def test_preload_memory_failure_degrades_gracefully(self) -> None:
        """If memory raises, relevant_knowledge stays empty."""
        mock_memory = MagicMock()
        mock_memory.retrieve.side_effect = ValueError("corrupt data")

        preloader = DecisionContextPreloader(memory=mock_memory)
        ctx = preloader.preload(task="task")

        assert ctx["relevant_knowledge"] == []

    def test_preload_calibration_failure_degrades_gracefully(self) -> None:
        """If calibration raises, agent_calibration stays empty."""
        mock_calibration = MagicMock()
        mock_calibration.get_top_agents_for_domain.side_effect = OSError("db error")

        preloader = DecisionContextPreloader(calibration=mock_calibration)
        ctx = preloader.preload(task="task", domain="legal")

        assert ctx["agent_calibration"] == {}

    def test_preload_all_subsystems_populated(self) -> None:
        """When all three subsystems are provided and working, all fields are filled."""
        mock_bridge = MagicMock()
        mock_bridge.query_precedents.return_value = [{"pipeline_id": "p1"}]

        entry = MagicMock()
        entry.to_dict.return_value = {"content": "insight"}
        mock_memory = MagicMock()
        mock_memory.retrieve.return_value = [entry]

        agent = MagicMock()
        agent.agent_name = "claude"
        agent.elo = 1500.0
        mock_calibration = MagicMock()
        mock_calibration.get_top_agents_for_domain.return_value = [agent]

        preloader = DecisionContextPreloader(
            knowledge_bridge=mock_bridge,
            memory=mock_memory,
            calibration=mock_calibration,
        )
        ctx = preloader.preload(task="Design API", domain="software")

        assert len(ctx["precedents"]) == 1
        assert len(ctx["relevant_knowledge"]) == 1
        assert ctx["agent_calibration"] == {"claude": 1500.0}

    def test_preload_with_pipeline_km_bridge_contract(self) -> None:
        """PipelineKMBridge should satisfy the preloader's sync precedent contract."""
        mock_km = MagicMock()
        pipeline_hit = MagicMock()
        pipeline_hit.content = "Pipeline cycle 42: rate limiter rollout"
        pipeline_hit.metadata = {
            "item_type": "pipeline_result",
            "cycle_id": "cycle-42",
            "success": True,
        }
        mock_km.search.return_value = [pipeline_hit]

        bridge = PipelineKMBridge(knowledge_mound=mock_km)
        with patch.object(
            bridge,
            "query_all_adapter_precedents",
            return_value={
                "receipts": [
                    {
                        "source": "receipt",
                        "receipt_id": "rcpt-9",
                        "summary": "Prior decision",
                    }
                ],
                "outcomes": [],
                "debates": [],
            },
        ):
            preloader = DecisionContextPreloader(knowledge_bridge=bridge)
            ctx = preloader.preload(task="Design a rate limiter")

        assert len(ctx["precedents"]) == 2
        assert ctx["precedents"][0]["source"] == "pipeline"
        assert ctx["precedents"][1]["source"] == "receipt"
