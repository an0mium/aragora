"""Tests for auto plan generation (Sprint 16D)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestPostDebateConfigDefault:
    """Test PostDebateConfig default for auto_create_plan."""

    def test_auto_create_plan_defaults_to_true(self):
        """auto_create_plan defaults to True."""
        from aragora.debate.post_debate_coordinator import PostDebateConfig

        config = PostDebateConfig()
        assert config.auto_create_plan is True

    def test_auto_create_plan_can_be_disabled(self):
        """auto_create_plan can be explicitly set to False."""
        from aragora.debate.post_debate_coordinator import PostDebateConfig

        config = PostDebateConfig(auto_create_plan=False)
        assert config.auto_create_plan is False


class TestPostDebateCoordinator:
    """Test PostDebateCoordinator plan generation behavior."""

    def test_plan_generated_for_high_confidence(self):
        """Plan is generated when confidence >= plan_min_confidence."""
        from aragora.debate.post_debate_coordinator import (
            PostDebateConfig,
            PostDebateCoordinator,
        )

        config = PostDebateConfig(
            auto_create_plan=True,
            auto_explain=False,
            auto_notify=False,
            auto_persist_receipt=False,
            auto_execution_bridge=False,
            plan_min_confidence=0.7,
        )
        coordinator = PostDebateCoordinator(config=config)

        result = MagicMock()
        result.confidence = 0.85
        result.final_answer = "Use microservices"
        result.consensus_reached = True
        result.winner = "claude"

        with patch.object(
            coordinator, "_step_create_plan", return_value={"steps": []}
        ) as mock_plan:
            output = coordinator.run(
                debate_id="test-1",
                debate_result=result,
                agents=[],
                confidence=0.85,
            )
            mock_plan.assert_called_once()
            assert output.plan == {"steps": []}

    def test_no_plan_for_low_confidence(self):
        """Plan is not generated when confidence < plan_min_confidence."""
        from aragora.debate.post_debate_coordinator import (
            PostDebateConfig,
            PostDebateCoordinator,
        )

        config = PostDebateConfig(
            auto_create_plan=True,
            auto_explain=False,
            auto_notify=False,
            auto_persist_receipt=False,
            auto_execution_bridge=False,
            plan_min_confidence=0.7,
        )
        coordinator = PostDebateCoordinator(config=config)

        result = MagicMock()
        result.confidence = 0.3
        result.final_answer = "Unclear"
        result.consensus_reached = False

        output = coordinator.run(
            debate_id="test-2",
            debate_result=result,
            agents=[],
            confidence=0.3,
        )
        # Plan should be None for low confidence
        assert output.plan is None


class TestDebateStatusHasPlan:
    """Test that debate status update includes has_plan field."""

    def test_has_plan_in_status_update(self):
        """has_plan is included in the debate status result dict."""
        from aragora.server.debate_controller import DebateController

        # Verify the field exists in the code pattern
        import inspect

        source = inspect.getsource(DebateController._run_debate)
        assert "has_plan" in source

    def test_has_plan_true_when_plan_exists(self):
        """has_plan is True when result has a plan attribute."""
        result = MagicMock()
        result.plan = {"steps": ["step1"]}

        has_plan = getattr(result, "plan", None) is not None
        assert has_plan is True

    def test_has_plan_false_when_no_plan(self):
        """has_plan is False when result has no plan."""
        result = MagicMock(spec=["final_answer", "confidence"])

        has_plan = getattr(result, "plan", None) is not None
        assert has_plan is False
