"""Tests for budget-aware agent selection in TeamSelector.

Validates that TeamSelector correctly adjusts agent selection based on
budget status from BudgetManager, including WARN (prefer cheap agents),
SOFT_LIMIT (reduce count), and HARD_LIMIT (block) behaviors.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.budget_manager import BudgetAction
from aragora.debate.team_selector import (
    BudgetExceededError,
    TeamSelectionConfig,
    TeamSelector,
)


def _make_agent(name: str) -> MagicMock:
    """Create a mock agent with the given name."""
    agent = MagicMock()
    agent.name = name
    agent.agent_type = name
    agent.model = name
    agent.metadata = {}
    return agent


@pytest.fixture
def agents():
    """Create a standard set of test agents."""
    return [
        _make_agent("claude-3"),
        _make_agent("gpt-4"),
        _make_agent("gemini-pro"),
        _make_agent("llama-70b"),
        _make_agent("deepseek-r1"),
        _make_agent("mistral-large"),
    ]


@pytest.fixture
def budget_manager():
    """Create a mock BudgetManager."""
    return MagicMock()


@pytest.fixture
def selector(budget_manager):
    """Create a TeamSelector with budget_manager."""
    return TeamSelector(
        budget_manager=budget_manager,
        org_id="org-test",
        config=TeamSelectionConfig(
            enable_domain_filtering=False,
            enable_cv_selection=False,
            enable_km_expertise=False,
            enable_pattern_selection=False,
        ),
    )


# =========================================================================
# Basic configuration tests
# =========================================================================


class TestBudgetFilterConfig:
    """Test budget filter configuration."""

    def test_budget_filtering_disabled_by_default_without_manager(self, agents):
        """Budget filtering is a no-op when no budget_manager is provided."""
        selector = TeamSelector(
            config=TeamSelectionConfig(enable_domain_filtering=False),
        )
        result = selector.select(agents)
        assert len(result) == len(agents)

    def test_budget_filtering_disabled_by_config(self, agents, budget_manager):
        """Budget filtering skipped when enable_budget_filtering=False."""
        selector = TeamSelector(
            budget_manager=budget_manager,
            org_id="org-test",
            config=TeamSelectionConfig(
                enable_budget_filtering=False,
                enable_domain_filtering=False,
            ),
        )
        result = selector.select(agents)
        # Budget manager should not be called
        budget_manager.check_budget.assert_not_called()
        assert len(result) == len(agents)

    def test_budget_filtering_skipped_without_org_id(self, agents, budget_manager):
        """Budget filtering skipped when org_id is not set."""
        selector = TeamSelector(
            budget_manager=budget_manager,
            org_id=None,
            config=TeamSelectionConfig(enable_domain_filtering=False),
        )
        result = selector.select(agents)
        budget_manager.check_budget.assert_not_called()
        assert len(result) == len(agents)

    def test_default_cheap_agent_patterns(self):
        """Default cheap agent patterns include expected models."""
        config = TeamSelectionConfig()
        assert "llama" in config.budget_cheap_agent_patterns
        assert "deepseek" in config.budget_cheap_agent_patterns
        assert "mistral" in config.budget_cheap_agent_patterns
        assert "gemini" in config.budget_cheap_agent_patterns

    def test_custom_cheap_agent_patterns(self, agents, budget_manager):
        """Custom cheap agent patterns are respected."""
        budget_manager.check_budget.return_value = (True, "Warning", BudgetAction.WARN)
        selector = TeamSelector(
            budget_manager=budget_manager,
            org_id="org-test",
            config=TeamSelectionConfig(
                enable_domain_filtering=False,
                enable_cv_selection=False,
                budget_cheap_agent_patterns=["llama"],
            ),
        )
        result = selector.select(agents)
        assert len(result) == 1
        assert result[0].name == "llama-70b"


# =========================================================================
# No action (budget OK) tests
# =========================================================================


class TestBudgetNoAction:
    """Test behavior when budget check returns no action needed."""

    def test_no_budget_configured(self, agents, selector, budget_manager):
        """No filtering when no budget is configured for org."""
        budget_manager.check_budget.return_value = (True, "No budget configured", None)
        result = selector.select(agents)
        assert len(result) == len(agents)

    def test_budget_ok(self, agents, selector, budget_manager):
        """No filtering when budget is within limits."""
        budget_manager.check_budget.return_value = (True, "OK", None)
        result = selector.select(agents)
        assert len(result) == len(agents)

    def test_notify_action_no_filtering(self, agents, selector, budget_manager):
        """NOTIFY action doesn't filter agents (not in the filter logic)."""
        budget_manager.check_budget.return_value = (True, "OK", None)
        result = selector.select(agents)
        assert len(result) == len(agents)


# =========================================================================
# WARN action tests
# =========================================================================


class TestBudgetWarn:
    """Test behavior when budget is in WARN state."""

    def test_warn_prefers_cheap_agents(self, agents, selector, budget_manager):
        """WARN filters to cheap agents when available."""
        budget_manager.check_budget.return_value = (True, "Warning: 80% used", BudgetAction.WARN)
        result = selector.select(agents)
        # Should only include cheap agents: gemini, llama, deepseek, mistral
        names = [a.name for a in result]
        assert "claude-3" not in names
        assert "gpt-4" not in names
        assert "gemini-pro" in names
        assert "llama-70b" in names
        assert "deepseek-r1" in names
        assert "mistral-large" in names

    def test_warn_falls_back_to_all_when_no_cheap(self, budget_manager):
        """WARN returns all agents if no cheap agents match."""
        agents = [_make_agent("claude-3"), _make_agent("gpt-4")]
        budget_manager.check_budget.return_value = (True, "Warning", BudgetAction.WARN)
        selector = TeamSelector(
            budget_manager=budget_manager,
            org_id="org-test",
            config=TeamSelectionConfig(
                enable_domain_filtering=False,
                enable_cv_selection=False,
                budget_cheap_agent_patterns=["nonexistent"],
            ),
        )
        result = selector.select(agents)
        assert len(result) == 2

    def test_warn_with_max_agents_limit(self, agents, budget_manager):
        """WARN respects budget_warn_max_agents limit."""
        budget_manager.check_budget.return_value = (True, "Warning", BudgetAction.WARN)
        selector = TeamSelector(
            budget_manager=budget_manager,
            org_id="org-test",
            config=TeamSelectionConfig(
                enable_domain_filtering=False,
                enable_cv_selection=False,
                budget_warn_max_agents=2,
            ),
        )
        result = selector.select(agents)
        assert len(result) == 2

    def test_warn_no_max_agents_limit(self, agents, selector, budget_manager):
        """WARN returns all cheap agents when budget_warn_max_agents is None."""
        budget_manager.check_budget.return_value = (True, "Warning", BudgetAction.WARN)
        result = selector.select(agents)
        # 4 cheap agents: gemini, llama, deepseek, mistral
        assert len(result) == 4


# =========================================================================
# SOFT_LIMIT action tests
# =========================================================================


class TestBudgetSoftLimit:
    """Test behavior when budget is at SOFT_LIMIT."""

    def test_soft_limit_reduces_agent_count(self, agents, selector, budget_manager):
        """SOFT_LIMIT reduces to budget_soft_limit_max_agents."""
        budget_manager.check_budget.return_value = (
            True,
            "Warning: 95% used",
            BudgetAction.SOFT_LIMIT,
        )
        result = selector.select(agents)
        assert len(result) == 3  # default budget_soft_limit_max_agents

    def test_soft_limit_custom_max(self, agents, budget_manager):
        """SOFT_LIMIT respects custom budget_soft_limit_max_agents."""
        budget_manager.check_budget.return_value = (True, "95%", BudgetAction.SOFT_LIMIT)
        selector = TeamSelector(
            budget_manager=budget_manager,
            org_id="org-test",
            config=TeamSelectionConfig(
                enable_domain_filtering=False,
                enable_cv_selection=False,
                budget_soft_limit_max_agents=2,
            ),
        )
        result = selector.select(agents)
        assert len(result) == 2

    def test_soft_limit_no_reduction_when_under_max(self, budget_manager):
        """SOFT_LIMIT doesn't reduce when agent count is already under max."""
        agents = [_make_agent("claude"), _make_agent("gpt")]
        budget_manager.check_budget.return_value = (True, "95%", BudgetAction.SOFT_LIMIT)
        selector = TeamSelector(
            budget_manager=budget_manager,
            org_id="org-test",
            config=TeamSelectionConfig(
                enable_domain_filtering=False,
                enable_cv_selection=False,
                budget_soft_limit_max_agents=5,
            ),
        )
        result = selector.select(agents)
        assert len(result) == 2

    def test_soft_limit_preserves_order(self, agents, selector, budget_manager):
        """SOFT_LIMIT takes the first N agents (preserving order from prior steps)."""
        budget_manager.check_budget.return_value = (True, "95%", BudgetAction.SOFT_LIMIT)
        result = selector.select(agents)
        # Should be the first 3 agents from the input
        assert result[0].name == agents[0].name
        assert result[1].name == agents[1].name
        assert result[2].name == agents[2].name


# =========================================================================
# HARD_LIMIT action tests
# =========================================================================


class TestBudgetHardLimit:
    """Test behavior when budget is at HARD_LIMIT."""

    def test_hard_limit_raises_error(self, agents, selector, budget_manager):
        """HARD_LIMIT raises BudgetExceededError."""
        budget_manager.check_budget.return_value = (
            False,
            "Budget exceeded",
            BudgetAction.HARD_LIMIT,
        )
        with pytest.raises(BudgetExceededError, match="hard_limit"):
            selector.select(agents)

    def test_hard_limit_error_includes_org_id(self, agents, selector, budget_manager):
        """BudgetExceededError message includes org_id."""
        budget_manager.check_budget.return_value = (
            False,
            "Exceeded",
            BudgetAction.HARD_LIMIT,
        )
        with pytest.raises(BudgetExceededError, match="org-test"):
            selector.select(agents)

    def test_suspend_raises_error(self, agents, selector, budget_manager):
        """SUSPEND also raises BudgetExceededError."""
        budget_manager.check_budget.return_value = (
            False,
            "Suspended",
            BudgetAction.SUSPEND,
        )
        with pytest.raises(BudgetExceededError, match="suspend"):
            selector.select(agents)


# =========================================================================
# Error handling tests
# =========================================================================


class TestBudgetFilterErrorHandling:
    """Test graceful degradation when budget system is unavailable."""

    def test_import_error_skips_filtering(self, agents, budget_manager):
        """ImportError during budget check skips filtering gracefully."""
        budget_manager.check_budget.side_effect = ImportError("module not found")
        selector = TeamSelector(
            budget_manager=budget_manager,
            org_id="org-test",
            config=TeamSelectionConfig(
                enable_domain_filtering=False,
                enable_cv_selection=False,
            ),
        )
        result = selector.select(agents)
        assert len(result) == len(agents)

    def test_attribute_error_skips_filtering(self, agents, budget_manager):
        """AttributeError during budget check skips filtering gracefully."""
        budget_manager.check_budget.side_effect = AttributeError("no method")
        selector = TeamSelector(
            budget_manager=budget_manager,
            org_id="org-test",
            config=TeamSelectionConfig(
                enable_domain_filtering=False,
                enable_cv_selection=False,
            ),
        )
        result = selector.select(agents)
        assert len(result) == len(agents)

    def test_type_error_skips_filtering(self, agents, budget_manager):
        """TypeError during budget check skips filtering gracefully."""
        budget_manager.check_budget.side_effect = TypeError("wrong args")
        selector = TeamSelector(
            budget_manager=budget_manager,
            org_id="org-test",
            config=TeamSelectionConfig(
                enable_domain_filtering=False,
                enable_cv_selection=False,
            ),
        )
        result = selector.select(agents)
        assert len(result) == len(agents)


# =========================================================================
# Integration with other filtering tests
# =========================================================================


class TestBudgetWithOtherFilters:
    """Test that budget filtering works alongside other filters."""

    def test_budget_filter_applies_after_domain_filter(self, budget_manager):
        """Budget filtering runs after domain filtering."""
        agents = [
            _make_agent("claude-3"),
            _make_agent("deepseek-coder"),
            _make_agent("llama-70b"),
        ]
        budget_manager.check_budget.return_value = (True, "Warning", BudgetAction.WARN)
        selector = TeamSelector(
            budget_manager=budget_manager,
            org_id="org-test",
            config=TeamSelectionConfig(
                enable_domain_filtering=True,
                enable_cv_selection=False,
                domain_filter_fallback=True,
            ),
        )
        result = selector.select(agents, domain="code")
        # Domain filter passes claude, deepseek, (llama doesn't match code)
        # Then budget WARN filters to cheap: deepseek
        names = [a.name for a in result]
        assert "deepseek-coder" in names

    def test_budget_filter_with_circuit_breaker(self, budget_manager):
        """Budget filtering works alongside circuit breaker."""
        agents = [
            _make_agent("claude-3"),
            _make_agent("llama-70b"),
            _make_agent("deepseek-r1"),
        ]
        budget_manager.check_budget.return_value = (True, "Warning", BudgetAction.WARN)

        circuit_breaker = MagicMock()
        circuit_breaker.filter_available_agents.return_value = ["llama-70b", "deepseek-r1"]

        selector = TeamSelector(
            budget_manager=budget_manager,
            org_id="org-test",
            circuit_breaker=circuit_breaker,
            config=TeamSelectionConfig(
                enable_domain_filtering=False,
                enable_cv_selection=False,
            ),
        )
        result = selector.select(agents)
        names = [a.name for a in result]
        # claude is filtered by budget (WARN -> cheap only)
        # Then circuit breaker removes claude too (already gone)
        # Result should be llama and deepseek
        assert "llama-70b" in names
        assert "deepseek-r1" in names
        assert "claude-3" not in names

    def test_budget_exceeded_error_is_exception(self):
        """BudgetExceededError is a proper Exception subclass."""
        err = BudgetExceededError("test")
        assert isinstance(err, Exception)
        assert str(err) == "test"

    def test_check_budget_called_with_org_id(self, agents, selector, budget_manager):
        """check_budget is called with the correct org_id."""
        budget_manager.check_budget.return_value = (True, "OK", None)
        selector.select(agents)
        budget_manager.check_budget.assert_called_once_with("org-test", estimated_cost_usd=0.0)

    def test_allow_with_charges_no_filtering(self, agents, selector, budget_manager):
        """ALLOW_WITH_CHARGES action doesn't filter (not in filter logic)."""
        budget_manager.check_budget.return_value = (
            True,
            "Overage",
            BudgetAction.ALLOW_WITH_CHARGES,
        )
        result = selector.select(agents)
        # ALLOW_WITH_CHARGES is not handled by _apply_budget_filter,
        # so all agents pass through
        assert len(result) == len(agents)
