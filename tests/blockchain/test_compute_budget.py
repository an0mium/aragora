"""
Tests for ComputeBudgetManager -- budget compute resources based on epistemic track record.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.blockchain.compute_budget import (
    ACCURACY_REWARD_SCALE,
    BASELINE_ALLOCATION,
    ComputeBudget,
    ComputeBudgetManager,
    INACCURACY_PENALTY_SCALE,
)


# =============================================================================
# ComputeBudget dataclass tests
# =============================================================================


class TestComputeBudget:
    """Tests for the ComputeBudget dataclass."""

    def test_available_tokens_basic(self):
        """Available tokens = total + earned - used - penalty."""
        budget = ComputeBudget(
            agent_id="a1",
            total_tokens=500,
            used_tokens=100,
            earned_tokens=50,
            penalty_tokens=25,
        )
        assert budget.available_tokens == 425  # 500 + 50 - 100 - 25

    def test_available_tokens_never_negative(self):
        """Available tokens floor at zero."""
        budget = ComputeBudget(
            agent_id="a1",
            total_tokens=100,
            used_tokens=500,
            earned_tokens=0,
            penalty_tokens=0,
        )
        assert budget.available_tokens == 0

    def test_default_values(self):
        """Budget starts with all zeros."""
        budget = ComputeBudget(agent_id="a1")
        assert budget.total_tokens == 0
        assert budget.used_tokens == 0
        assert budget.earned_tokens == 0
        assert budget.penalty_tokens == 0
        assert budget.available_tokens == 0


# =============================================================================
# ComputeBudgetManager tests
# =============================================================================


class TestComputeBudgetManager:
    """Tests for ComputeBudgetManager."""

    @pytest.fixture
    def manager(self) -> ComputeBudgetManager:
        """Create a fresh ComputeBudgetManager."""
        return ComputeBudgetManager()

    def test_allocate_baseline(self, manager: ComputeBudgetManager):
        """Basic allocation returns baseline tokens."""
        tokens = manager.allocate("agent_1", task_complexity=1.0)
        assert tokens == BASELINE_ALLOCATION

    def test_allocate_scales_with_complexity(self, manager: ComputeBudgetManager):
        """Allocation scales with task complexity."""
        low = manager.allocate("agent_1", task_complexity=0.5)
        manager.reset("agent_1")
        high = manager.allocate("agent_1", task_complexity=2.0)
        assert high > low

    def test_allocate_minimum_one_token(self, manager: ComputeBudgetManager):
        """Allocation always returns at least 1 token."""
        tokens = manager.allocate("agent_1", task_complexity=0.0)
        assert tokens >= 1

    def test_charge_deducts_tokens(self, manager: ComputeBudgetManager):
        """Charging reduces available tokens."""
        manager.allocate("agent_1", task_complexity=1.0)
        manager.charge("agent_1", tokens_used=50)
        budget = manager.get_budget("agent_1")
        assert budget.used_tokens == 50
        assert budget.available_tokens == BASELINE_ALLOCATION - 50

    def test_charge_negative_raises(self, manager: ComputeBudgetManager):
        """Charging negative tokens raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            manager.charge("agent_1", tokens_used=-10)

    def test_reward_accuracy_perfect(self, manager: ComputeBudgetManager):
        """Perfect accuracy rewards max tokens."""
        bonus = manager.reward_accuracy("agent_1", epistemic_score=1.0)
        assert bonus == ACCURACY_REWARD_SCALE
        budget = manager.get_budget("agent_1")
        assert budget.earned_tokens == ACCURACY_REWARD_SCALE

    def test_reward_accuracy_zero(self, manager: ComputeBudgetManager):
        """Zero accuracy gives no reward."""
        bonus = manager.reward_accuracy("agent_1", epistemic_score=0.0)
        assert bonus == 0

    def test_reward_accuracy_partial(self, manager: ComputeBudgetManager):
        """Partial accuracy gives proportional reward."""
        bonus = manager.reward_accuracy("agent_1", epistemic_score=0.5)
        assert bonus == int(0.5 * ACCURACY_REWARD_SCALE)

    def test_penalize_inaccuracy_zero_score(self, manager: ComputeBudgetManager):
        """Zero epistemic score gives max penalty."""
        penalty = manager.penalize_inaccuracy("agent_1", epistemic_score=0.0)
        assert penalty == INACCURACY_PENALTY_SCALE

    def test_penalize_inaccuracy_perfect_score(self, manager: ComputeBudgetManager):
        """Perfect epistemic score gives no penalty."""
        penalty = manager.penalize_inaccuracy("agent_1", epistemic_score=1.0)
        assert penalty == 0

    def test_penalize_inaccuracy_partial(self, manager: ComputeBudgetManager):
        """Partial inaccuracy gives proportional penalty."""
        penalty = manager.penalize_inaccuracy("agent_1", epistemic_score=0.3)
        expected = int(0.7 * INACCURACY_PENALTY_SCALE)
        assert penalty == expected

    def test_get_budget_creates_empty(self, manager: ComputeBudgetManager):
        """Getting budget for unknown agent creates empty budget."""
        budget = manager.get_budget("new_agent")
        assert budget.agent_id == "new_agent"
        assert budget.available_tokens == 0

    def test_has_budget_true(self, manager: ComputeBudgetManager):
        """has_budget returns True when sufficient tokens available."""
        manager.allocate("agent_1", task_complexity=1.0)
        assert manager.has_budget("agent_1", tokens_needed=50) is True

    def test_has_budget_false(self, manager: ComputeBudgetManager):
        """has_budget returns False when insufficient tokens."""
        assert manager.has_budget("agent_1", tokens_needed=50) is False

    def test_reset_clears_budget(self, manager: ComputeBudgetManager):
        """Reset removes the agent's budget entirely."""
        manager.allocate("agent_1", task_complexity=1.0)
        manager.reset("agent_1")
        budget = manager.get_budget("agent_1")
        assert budget.available_tokens == 0

    def test_with_reputation_registry(self):
        """Allocation scales with reputation when registry is available."""
        mock_rep = MagicMock()
        mock_summary = MagicMock()
        mock_summary.normalized_value = 500.0  # Mid-range reputation
        mock_rep.get_summary.return_value = mock_summary

        manager_with_rep = ComputeBudgetManager(reputation_registry=mock_rep)
        tokens = manager_with_rep.allocate("42", task_complexity=1.0)

        # With reputation of 500/1000, should get boosted allocation
        assert tokens > 0
        mock_rep.get_summary.assert_called_once()

    def test_reputation_failure_graceful(self):
        """When reputation registry fails, allocation still works."""
        mock_rep = MagicMock()
        mock_rep.get_summary.side_effect = RuntimeError("Connection failed")

        manager = ComputeBudgetManager(reputation_registry=mock_rep)
        tokens = manager.allocate("42", task_complexity=1.0)
        assert tokens == BASELINE_ALLOCATION  # Falls back to neutral factor

    def test_full_lifecycle(self, manager: ComputeBudgetManager):
        """End-to-end lifecycle: allocate, charge, reward, penalize."""
        # Allocate
        allocated = manager.allocate("agent_1", task_complexity=1.0)
        assert allocated == BASELINE_ALLOCATION

        # Charge
        manager.charge("agent_1", tokens_used=30)

        # Reward accuracy
        bonus = manager.reward_accuracy("agent_1", epistemic_score=0.9)

        # Penalize inaccuracy
        penalty = manager.penalize_inaccuracy("agent_1", epistemic_score=0.3)

        budget = manager.get_budget("agent_1")
        expected = BASELINE_ALLOCATION + bonus - 30 - penalty
        assert budget.available_tokens == max(0, expected)
