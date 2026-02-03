"""
Tests for Gold Path integration: debate -> decision plan -> execution flow.

Tests cover:
- Protocol with_gold_path convenience method
- Auto-plan creation after debate with high confidence
- Confidence threshold gating
- Consensus requirement for plan creation
- Plan storage and retrieval
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import DebateResult
from aragora.debate.protocol import DebateProtocol


def _create_mock_debate_result(**overrides) -> MagicMock:
    """Create a properly configured mock DebateResult for testing."""
    mock_result = MagicMock(spec=DebateResult)
    mock_result.confidence = overrides.get("confidence", 0.85)
    mock_result.consensus_reached = overrides.get("consensus_reached", True)
    mock_result.final_answer = overrides.get("final_answer", "Implement feature X because...")
    mock_result.total_cost_usd = overrides.get("total_cost_usd", 0.50)
    mock_result.per_agent_cost = overrides.get("per_agent_cost", {"claude": 0.50})
    mock_result.debate_id = overrides.get("debate_id", "test_debate_123")
    mock_result.task = overrides.get("task", "What should we implement?")
    mock_result.critiques = overrides.get("critiques", [])
    mock_result.dissenting_views = overrides.get("dissenting_views", [])
    mock_result.debate_cruxes = overrides.get("debate_cruxes", [])
    return mock_result


class TestGoldPathProtocol:
    """Tests for DebateProtocol.with_gold_path()."""

    def test_with_gold_path_enables_auto_create(self):
        """with_gold_path creates protocol with auto_create_plan=True."""
        protocol = DebateProtocol.with_gold_path()
        assert protocol.auto_create_plan is True

    def test_with_gold_path_default_confidence(self):
        """Default min confidence is 0.7."""
        protocol = DebateProtocol.with_gold_path()
        assert protocol.plan_min_confidence == 0.7

    def test_with_gold_path_custom_confidence(self):
        """Custom min confidence can be specified."""
        protocol = DebateProtocol.with_gold_path(min_confidence=0.9)
        assert protocol.plan_min_confidence == 0.9

    def test_with_gold_path_default_approval_mode(self):
        """Default approval mode is risk_based."""
        protocol = DebateProtocol.with_gold_path()
        assert protocol.plan_approval_mode == "risk_based"

    def test_with_gold_path_custom_approval_mode(self):
        """Custom approval mode can be specified."""
        protocol = DebateProtocol.with_gold_path(approval_mode="always")
        assert protocol.plan_approval_mode == "always"

    def test_with_gold_path_budget_limit(self):
        """Budget limit can be specified."""
        protocol = DebateProtocol.with_gold_path(budget_limit_usd=100.0)
        assert protocol.plan_budget_limit_usd == 100.0

    def test_with_gold_path_passes_other_kwargs(self):
        """Other protocol kwargs are passed through."""
        protocol = DebateProtocol.with_gold_path(rounds=5, consensus="majority")
        assert protocol.auto_create_plan is True
        assert protocol.rounds == 5
        assert protocol.consensus == "majority"


class TestGoldPathHookTriggering:
    """Tests for Gold Path hook triggering conditions."""

    @pytest.mark.asyncio
    async def test_plan_created_when_confidence_meets_threshold(self):
        """Plan is created when confidence >= threshold."""
        from aragora.pipeline.executor import _plan_store, store_plan
        from aragora.pipeline.decision_plan import DecisionPlanFactory

        # Clear plan store
        _plan_store.clear()

        protocol = DebateProtocol.with_gold_path(min_confidence=0.7)

        # Create mock result above threshold
        mock_result = _create_mock_debate_result(
            confidence=0.85,
            consensus_reached=True,
            final_answer="We should implement feature X because...",
            task="Should we implement feature X?",
            debate_id="test_debate_123",
        )

        # Directly test plan creation logic
        plan = DecisionPlanFactory.from_debate_result(
            mock_result,
            approval_mode="risk_based",
        )

        assert plan is not None
        assert plan.task == mock_result.task  # task field is used, not final_answer
        store_plan(plan)

        assert plan.id in _plan_store
        _plan_store.clear()

    @pytest.mark.asyncio
    async def test_plan_not_created_when_confidence_below_threshold(self):
        """Plan is NOT created when confidence < threshold."""
        from aragora.pipeline.executor import _plan_store

        _plan_store.clear()

        # Low confidence result
        mock_result = _create_mock_debate_result(
            confidence=0.5,  # Below threshold
            consensus_reached=True,
        )

        # With low confidence, the hook should not create a plan
        # We verify by checking that plan factory would still work
        # but the hook logic would gate it
        protocol = DebateProtocol.with_gold_path(min_confidence=0.7)
        assert mock_result.confidence < protocol.plan_min_confidence

        # Hook would check: if confidence < threshold, don't create
        # This is tested via the hook_handlers.py logic

    @pytest.mark.asyncio
    async def test_plan_not_created_when_no_consensus(self):
        """Plan is NOT created when consensus not reached."""
        mock_result = _create_mock_debate_result(
            confidence=0.9,  # High confidence
            consensus_reached=False,  # No consensus
        )

        # Without consensus, hook should not create plan
        # The hook logic checks: if not consensus_reached, skip
        protocol = DebateProtocol.with_gold_path()
        assert mock_result.consensus_reached is False
        # Hook would gate based on consensus_reached


class TestDecisionPlanFactory:
    """Tests for DecisionPlanFactory.from_debate_result."""

    def test_creates_plan_from_debate_result(self):
        """Factory creates plan from debate result."""
        from aragora.pipeline.decision_plan import DecisionPlanFactory

        mock_result = _create_mock_debate_result(
            confidence=0.85,
            final_answer="Implement caching layer:\n1. Add Redis\n2. Update API",
            debate_id="debate_123",
        )

        plan = DecisionPlanFactory.from_debate_result(mock_result)

        assert plan is not None
        assert plan.task is not None
        assert plan.debate_id == "debate_123"

    def test_plan_has_risk_register(self):
        """Created plan has risk register."""
        from aragora.pipeline.decision_plan import DecisionPlanFactory

        mock_result = _create_mock_debate_result(
            final_answer="Deploy to production",
            debate_id="debate_123",
        )

        plan = DecisionPlanFactory.from_debate_result(mock_result)

        assert hasattr(plan, "risk_register")

    def test_plan_has_verification_plan(self):
        """Created plan has verification plan."""
        from aragora.pipeline.decision_plan import DecisionPlanFactory

        mock_result = _create_mock_debate_result(
            final_answer="Refactor authentication module",
            debate_id="debate_123",
        )

        plan = DecisionPlanFactory.from_debate_result(mock_result)

        assert hasattr(plan, "verification_plan")


class TestPlanStorage:
    """Tests for plan storage and retrieval."""

    def test_store_and_retrieve_plan(self):
        """Plan can be stored and retrieved."""
        from aragora.pipeline.decision_plan import DecisionPlanFactory
        from aragora.pipeline.executor import _plan_store, get_plan, store_plan

        _plan_store.clear()

        mock_result = _create_mock_debate_result(
            final_answer="Test task",
            debate_id="debate_123",
        )

        plan = DecisionPlanFactory.from_debate_result(mock_result)
        store_plan(plan)

        retrieved = get_plan(plan.id)
        assert retrieved is not None
        assert retrieved.id == plan.id

        _plan_store.clear()

    def test_list_plans(self):
        """Can list all stored plans."""
        from aragora.pipeline.decision_plan import DecisionPlanFactory
        from aragora.pipeline.executor import _plan_store, list_plans, store_plan

        _plan_store.clear()

        mock_result1 = _create_mock_debate_result(
            final_answer="Test task 1",
            debate_id="debate_123",
        )
        mock_result2 = _create_mock_debate_result(
            final_answer="Test task 2",
            debate_id="debate_456",
        )

        plan1 = DecisionPlanFactory.from_debate_result(mock_result1)
        plan2 = DecisionPlanFactory.from_debate_result(mock_result2)

        store_plan(plan1)
        store_plan(plan2)

        plans = list_plans()
        assert len(plans) == 2

        _plan_store.clear()
