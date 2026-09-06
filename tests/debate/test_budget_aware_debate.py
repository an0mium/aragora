"""Tests for budget-aware debate configuration.

Covers:
- ArenaConfig accepts provider_budget field
- ProviderRouter.select_providers_with_details filters by budget
- Zero budget returns no providers
- High budget returns all eligible providers
"""

from __future__ import annotations

import pytest

from aragora.debate.arena_config import ArenaConfig
from aragora.routing.provider_router import ProviderRouter


# ===========================================================================
# Task 10: Budget-aware debate configuration
# ===========================================================================


class TestArenaConfigProviderBudget:
    """Test that ArenaConfig accepts and stores provider_budget."""

    def test_provider_budget_default_none(self):
        """provider_budget defaults to None."""
        config = ArenaConfig()
        assert config.provider_budget is None

    def test_provider_budget_set_value(self):
        """provider_budget can be set to a float."""
        config = ArenaConfig(provider_budget=5.0)
        assert config.provider_budget == 5.0

    def test_provider_budget_zero(self):
        """provider_budget can be set to 0."""
        config = ArenaConfig(provider_budget=0.0)
        assert config.provider_budget == 0.0

    def test_provider_budget_small_value(self):
        """provider_budget can be a very small float."""
        config = ArenaConfig(provider_budget=0.001)
        assert config.provider_budget == 0.001


class TestSelectProvidersWithDetails:
    """Test ProviderRouter.select_providers_with_details."""

    def _make_router_with_data(self) -> ProviderRouter:
        """Create a router with sufficient metrics data."""
        router = ProviderRouter()
        # Record enough data to exceed MIN_DEBATES_FOR_METRICS (10)
        for _ in range(5):
            router.record_outcome("claude-sonnet-5", quality=0.9, cost=0.15)
        for _ in range(5):
            router.record_outcome("gpt-4o", quality=0.85, cost=0.10)
        for _ in range(5):
            router.record_outcome("deepseek-r1", quality=0.7, cost=0.01)
        return router

    def test_returns_list_of_dicts(self):
        """Result should be a list of dicts with required keys."""
        router = self._make_router_with_data()
        result = router.select_providers_with_details(num_agents=3)

        assert isinstance(result, list)
        for entry in result:
            assert "provider" in entry
            assert "estimated_cost" in entry
            assert "quality_score" in entry

    def test_budget_filters_expensive_providers(self):
        """Providers exceeding per-agent budget should be excluded."""
        router = self._make_router_with_data()

        # Budget of 0.06 total for 3 agents = 0.02 per agent
        # Only deepseek-r1 (cost ~0.01) should fit
        result = router.select_providers_with_details(num_agents=3, budget=0.06)
        provider_names = [r["provider"] for r in result]
        assert "deepseek-r1" in provider_names
        # claude-sonnet-5 and gpt-4o cost more than 0.02 per agent
        assert "claude-sonnet-5" not in provider_names
        assert "gpt-4o" not in provider_names

    def test_zero_budget_no_providers(self):
        """Budget of 0 should return no providers (everything exceeds 0)."""
        router = self._make_router_with_data()
        result = router.select_providers_with_details(num_agents=3, budget=0.0)
        assert result == []

    def test_high_budget_all_providers(self):
        """Very high budget should include all tracked providers."""
        router = self._make_router_with_data()
        result = router.select_providers_with_details(num_agents=10, budget=1000.0)
        provider_names = {r["provider"] for r in result}
        assert "claude-sonnet-5" in provider_names
        assert "gpt-4o" in provider_names
        assert "deepseek-r1" in provider_names

    def test_no_budget_returns_all(self):
        """When budget is None, all providers should be eligible."""
        router = self._make_router_with_data()
        result = router.select_providers_with_details(num_agents=10, budget=None)
        assert len(result) >= 3

    def test_num_agents_limits_results(self):
        """Result length should be capped at num_agents."""
        router = self._make_router_with_data()
        result = router.select_providers_with_details(num_agents=1, budget=1000.0)
        assert len(result) == 1

    def test_min_quality_filters_low_quality(self):
        """Providers below min_quality should be excluded."""
        router = self._make_router_with_data()
        result = router.select_providers_with_details(num_agents=10, budget=1000.0, min_quality=0.8)
        provider_names = {r["provider"] for r in result}
        # deepseek-r1 has quality 0.7, below threshold
        assert "deepseek-r1" not in provider_names
        assert "claude-sonnet-5" in provider_names

    def test_quality_scores_in_range(self):
        """Quality scores should be in [0, 1]."""
        router = self._make_router_with_data()
        result = router.select_providers_with_details(num_agents=10)
        for entry in result:
            assert 0.0 <= entry["quality_score"] <= 1.0

    def test_estimated_cost_non_negative(self):
        """Estimated costs should be non-negative."""
        router = self._make_router_with_data()
        result = router.select_providers_with_details(num_agents=10)
        for entry in result:
            assert entry["estimated_cost"] >= 0.0


class TestSelectProvidersWithDetailsFallback:
    """Test fallback to pricing data when no metrics exist."""

    def test_no_metrics_uses_pricing(self):
        """When no metrics recorded, should fall back to static pricing."""
        router = ProviderRouter()
        result = router.select_providers_with_details(num_agents=3)

        assert isinstance(result, list)
        assert len(result) <= 3
        for entry in result:
            assert "provider" in entry
            assert entry["quality_score"] == 0.5  # Default neutral quality

    def test_no_metrics_budget_filters_pricing(self):
        """Budget constraint should also apply to pricing-based fallback."""
        router = ProviderRouter()
        # Very low budget should exclude expensive models
        result = router.select_providers_with_details(num_agents=10, budget=0.001)
        for entry in result:
            assert entry["estimated_cost"] <= 0.001 / 10

    def test_no_metrics_zero_budget(self):
        """Zero budget with no metrics should return empty list."""
        router = ProviderRouter()
        result = router.select_providers_with_details(num_agents=3, budget=0.0)
        assert result == []
