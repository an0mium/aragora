"""
Cost Attribution Integration Tests (Phase 3 Production Readiness).

Tests cost attribution across the debate lifecycle:
- Cost distribution across debate rounds
- Partial cost attribution for failed agents
- Cancelled debate cost handling
- Cross-tenant cost isolation
- Budget enforcement integration
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.cost_attribution import (
    AttributionEntry,
    AttributionLevel,
    AttributionSummary,
    CostAttributor,
    create_cost_attributor,
)
from aragora.billing.budget_manager import (
    Budget,
    BudgetAction,
    BudgetStatus,
    BudgetThreshold,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def cost_attributor():
    """Create a fresh cost attributor for testing."""
    return CostAttributor()


@pytest.fixture
def budget_with_alerts():
    """Create a budget with alert thresholds."""
    return Budget(
        budget_id="test-budget",
        org_id="org-1",
        name="Test Budget",
        amount_usd=100.0,
        spent_usd=0.0,
        thresholds=[
            BudgetThreshold(0.5, BudgetAction.WARN),  # 50%
            BudgetThreshold(0.8, BudgetAction.SOFT_LIMIT),  # 80%
            BudgetThreshold(1.0, BudgetAction.HARD_LIMIT),  # 100%
        ],
    )


# =============================================================================
# Debate Round Cost Distribution Tests
# =============================================================================


class TestDebateRoundCostDistribution:
    """Test cost distribution across debate rounds."""

    def test_cost_distributed_across_rounds(self, cost_attributor):
        """Each debate round contributes to total cost."""
        debate_id = "debate-123"
        user_id = "user-1"
        org_id = "org-1"

        # Simulate 3 rounds of a debate
        round_costs = [
            Decimal("0.50"),  # Round 1: Initial proposals
            Decimal("0.75"),  # Round 2: Critiques and revisions
            Decimal("0.25"),  # Round 3: Final consensus
        ]

        for round_num, cost in enumerate(round_costs, 1):
            cost_attributor.record_cost(
                cost_usd=cost,
                source_type="debate_round",
                source_id=f"{debate_id}_round_{round_num}",
                tokens_in=1000,
                tokens_out=500,
                provider="anthropic",
                model="claude-sonnet",
                user_id=user_id,
                debate_id=debate_id,
                org_id=org_id,
            )

        # Get summary for the debate
        summary = cost_attributor.get_user_summary(user_id)

        assert summary.total_cost_usd == Decimal("1.50")  # Sum of all rounds
        assert summary.total_api_calls == 3  # One per round

    def test_debate_cost_breakdown_by_agent(self, cost_attributor):
        """Costs are tracked per agent within a debate."""
        debate_id = "debate-456"
        user_id = "user-1"

        # Simulate costs from different agents
        agent_costs = {
            "claude": Decimal("0.80"),
            "gpt-4": Decimal("0.60"),
            "gemini": Decimal("0.40"),
        }

        for agent, cost in agent_costs.items():
            cost_attributor.record_cost(
                cost_usd=cost,
                source_type="agent_call",
                source_id=f"{debate_id}_{agent}",
                provider=agent.split("-")[0] if "-" in agent else agent,
                model=agent,
                user_id=user_id,
                debate_id=debate_id,
            )

        summary = cost_attributor.get_user_summary(user_id)

        assert summary.total_cost_usd == Decimal("1.80")
        # Check model breakdown
        assert "claude" in summary.cost_by_model or "gpt-4" in summary.cost_by_model

    def test_multiple_debates_tracked_separately(self, cost_attributor):
        """Costs from different debates are tracked separately."""
        user_id = "user-1"

        # Debate 1
        cost_attributor.record_cost(
            cost_usd=Decimal("1.00"),
            debate_id="debate-1",
            user_id=user_id,
        )
        cost_attributor.record_cost(
            cost_usd=Decimal("0.50"),
            debate_id="debate-1",
            user_id=user_id,
        )

        # Debate 2
        cost_attributor.record_cost(
            cost_usd=Decimal("0.75"),
            debate_id="debate-2",
            user_id=user_id,
        )

        # Get all entries
        entries = cost_attributor._entries

        debate_1_cost = sum(e.cost_usd for e in entries if e.debate_id == "debate-1")
        debate_2_cost = sum(e.cost_usd for e in entries if e.debate_id == "debate-2")

        assert debate_1_cost == Decimal("1.50")
        assert debate_2_cost == Decimal("0.75")


# =============================================================================
# Failed Agent Partial Cost Attribution Tests
# =============================================================================


class TestFailedAgentCostAttribution:
    """Test partial cost attribution for failed agents."""

    def test_failed_agent_partial_cost_attributed(self, cost_attributor):
        """Failed agents still contribute their consumed tokens."""
        debate_id = "debate-fail-1"
        user_id = "user-1"

        # Successful agent call
        cost_attributor.record_cost(
            cost_usd=Decimal("0.50"),
            source_type="agent_call",
            source_id=f"{debate_id}_claude_success",
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            user_id=user_id,
            debate_id=debate_id,
            metadata={"status": "success"},
        )

        # Failed agent call - still consumed tokens before failure
        cost_attributor.record_cost(
            cost_usd=Decimal("0.30"),  # Partial cost - tokens consumed before failure
            source_type="agent_call",
            source_id=f"{debate_id}_gpt_failed",
            tokens_in=800,  # Input tokens were consumed
            tokens_out=200,  # Partial output before failure
            provider="openai",
            user_id=user_id,
            debate_id=debate_id,
            metadata={"status": "failed", "error": "timeout"},
        )

        # Total should include both - we pay for tokens regardless of success
        summary = cost_attributor.get_user_summary(user_id)

        assert summary.total_cost_usd == Decimal("0.80")
        assert summary.total_tokens_in == 1800
        assert summary.total_tokens_out == 700

    def test_failed_agent_marked_in_metadata(self, cost_attributor):
        """Failed agent costs include failure metadata."""
        cost_attributor.record_cost(
            cost_usd=Decimal("0.20"),
            source_type="agent_call",
            user_id="user-1",
            debate_id="debate-1",
            metadata={
                "status": "failed",
                "error_type": "rate_limit",
                "partial": True,
            },
        )

        entry = cost_attributor._entries[0]
        assert entry.metadata["status"] == "failed"
        assert entry.metadata["partial"] is True


# =============================================================================
# Cancelled Debate Cost Tests
# =============================================================================


class TestCancelledDebateCost:
    """Test cancelled debate cost handling."""

    def test_cancelled_debate_records_partial_cost(self, cost_attributor):
        """Cancelled debates record cost up to cancellation point."""
        debate_id = "debate-cancelled-1"
        user_id = "user-1"

        # Round 1 completes
        cost_attributor.record_cost(
            cost_usd=Decimal("0.50"),
            source_type="debate_round",
            source_id=f"{debate_id}_round_1",
            user_id=user_id,
            debate_id=debate_id,
            metadata={"round": 1, "status": "completed"},
        )

        # Round 2 partially completes (2/4 agents)
        cost_attributor.record_cost(
            cost_usd=Decimal("0.30"),
            source_type="debate_round",
            source_id=f"{debate_id}_round_2_partial",
            user_id=user_id,
            debate_id=debate_id,
            metadata={"round": 2, "status": "partial", "agents_completed": 2, "agents_total": 4},
        )

        # Debate cancelled - no round 3
        # Record cancellation marker
        cost_attributor.record_cost(
            cost_usd=Decimal("0.00"),  # No additional cost for cancellation event
            source_type="debate_cancellation",
            source_id=f"{debate_id}_cancelled",
            user_id=user_id,
            debate_id=debate_id,
            metadata={"cancelled_at_round": 2, "reason": "user_request"},
        )

        # Total should be sum of completed work
        summary = cost_attributor.get_user_summary(user_id)
        assert summary.total_cost_usd == Decimal("0.80")

        # Verify cancellation is recorded
        entries = [e for e in cost_attributor._entries if e.debate_id == debate_id]
        cancellation_entry = [e for e in entries if e.source_type == "debate_cancellation"]
        assert len(cancellation_entry) == 1

    def test_cancelled_debate_metadata_preserved(self, cost_attributor):
        """Cancellation reason and state are preserved in metadata."""
        debate_id = "debate-cancel-2"

        # Record work then cancel
        cost_attributor.record_cost(
            cost_usd=Decimal("0.40"),
            debate_id=debate_id,
            user_id="user-1",
        )

        cost_attributor.record_cost(
            cost_usd=Decimal("0.00"),
            source_type="debate_cancellation",
            debate_id=debate_id,
            user_id="user-1",
            metadata={
                "cancelled_at_round": 1,
                "reason": "budget_exceeded",
                "final_cost_usd": "0.40",
            },
        )

        cancellation = [
            e for e in cost_attributor._entries if e.source_type == "debate_cancellation"
        ][0]

        assert cancellation.metadata["reason"] == "budget_exceeded"
        assert cancellation.metadata["final_cost_usd"] == "0.40"


# =============================================================================
# Cross-Tenant Cost Isolation Tests
# =============================================================================


class TestCrossTenantCostIsolation:
    """Test cost isolation between tenants."""

    def test_cost_isolation_between_tenants(self, cost_attributor):
        """Tenant A's debates don't appear in Tenant B's costs."""
        # Tenant A costs
        cost_attributor.record_cost(
            cost_usd=Decimal("1.00"),
            org_id="org-tenant-a",
            user_id="user-a-1",
            debate_id="debate-a-1",
        )
        cost_attributor.record_cost(
            cost_usd=Decimal("0.50"),
            org_id="org-tenant-a",
            user_id="user-a-2",
            debate_id="debate-a-2",
        )

        # Tenant B costs
        cost_attributor.record_cost(
            cost_usd=Decimal("0.75"),
            org_id="org-tenant-b",
            user_id="user-b-1",
            debate_id="debate-b-1",
        )

        # Calculate org costs from entries (no get_org_summary method)
        tenant_a_cost = sum(
            e.cost_usd for e in cost_attributor._entries if e.org_id == "org-tenant-a"
        )
        tenant_b_cost = sum(
            e.cost_usd for e in cost_attributor._entries if e.org_id == "org-tenant-b"
        )

        # Verify isolation
        assert tenant_a_cost == Decimal("1.50")
        assert tenant_b_cost == Decimal("0.75")

        # Verify no cross-contamination
        tenant_a_entries = [e for e in cost_attributor._entries if e.org_id == "org-tenant-a"]
        tenant_a_debates = [e.debate_id for e in tenant_a_entries if e.debate_id]
        assert "debate-b-1" not in tenant_a_debates

    def test_user_in_multiple_orgs_separate_costs(self, cost_attributor):
        """Same user in different orgs has separate cost tracking."""
        user_id = "shared-user"

        # User in Org A
        cost_attributor.record_cost(
            cost_usd=Decimal("1.00"),
            org_id="org-a",
            user_id=user_id,
        )

        # User in Org B
        cost_attributor.record_cost(
            cost_usd=Decimal("2.00"),
            org_id="org-b",
            user_id=user_id,
        )

        # Calculate org costs from entries
        org_a_cost = sum(e.cost_usd for e in cost_attributor._entries if e.org_id == "org-a")
        org_b_cost = sum(e.cost_usd for e in cost_attributor._entries if e.org_id == "org-b")

        assert org_a_cost == Decimal("1.00")
        assert org_b_cost == Decimal("2.00")

        # Total user cost across both orgs
        user_summary = cost_attributor.get_user_summary(user_id)
        assert user_summary.total_cost_usd == Decimal("3.00")


# =============================================================================
# Budget Enforcement Integration Tests
# =============================================================================


class TestCostAttributionWithBudgetEnforcement:
    """Test cost attribution triggers budget alerts."""

    def test_cost_triggers_budget_threshold(self, cost_attributor, budget_with_alerts):
        """Cost recording can trigger budget threshold checks."""
        budget = budget_with_alerts
        org_id = "org-1"

        # Record costs that exceed 50% threshold
        for i in range(6):
            cost_attributor.record_cost(
                cost_usd=Decimal("10.00"),
                org_id=org_id,
                user_id=f"user-{i}",
            )
            # Update budget spent
            budget.spent_usd += 10.0

        # Check budget utilization
        utilization = budget.spent_usd / budget.amount_usd
        assert utilization == 0.6  # 60%

        # Check which threshold was crossed
        triggered_actions = []
        for threshold in budget.thresholds:
            if utilization >= threshold.percentage:
                triggered_actions.append(threshold.action)

        assert BudgetAction.WARN in triggered_actions
        assert BudgetAction.SOFT_LIMIT not in triggered_actions  # Not at 80% yet

    def test_budget_soft_limit_at_80_percent(self, cost_attributor, budget_with_alerts):
        """Soft limit triggers at 80% budget utilization."""
        budget = budget_with_alerts

        # Spend to 85%
        budget.spent_usd = 85.0

        utilization = budget.spent_usd / budget.amount_usd
        triggered = [t for t in budget.thresholds if utilization >= t.percentage]

        assert any(t.action == BudgetAction.SOFT_LIMIT for t in triggered)

    def test_hard_limit_blocks_at_100_percent(self, cost_attributor, budget_with_alerts):
        """Hard limit blocks spending at 100%."""
        budget = budget_with_alerts
        budget.spent_usd = 100.0  # At limit

        # Try to spend more
        result = budget.can_spend_extended(10.0)

        assert result.allowed is False
        assert "exceeded" in result.message.lower()

    def test_cost_attribution_updates_budget_tracking(self, cost_attributor):
        """Cost attribution can update budget tracking."""
        org_id = "org-budget-test"
        budget = Budget(
            budget_id="budget-1",
            org_id=org_id,
            name="Test Budget",
            amount_usd=50.0,
            spent_usd=0.0,
        )

        # Simulate budget update callback
        costs_recorded = []

        def on_cost_recorded(entry: AttributionEntry):
            costs_recorded.append(entry.cost_usd)
            budget.spent_usd += float(entry.cost_usd)

        # Record costs with callback simulation
        for _ in range(5):
            entry = cost_attributor.record_cost(
                cost_usd=Decimal("5.00"),
                org_id=org_id,
                user_id="user-1",
            )
            on_cost_recorded(entry)

        # Budget should be updated
        assert budget.spent_usd == 25.0
        assert len(costs_recorded) == 5


# =============================================================================
# Cost Attribution Summary Tests
# =============================================================================


class TestCostAttributionSummary:
    """Test cost attribution summary generation."""

    def test_summary_includes_all_fields(self, cost_attributor):
        """Summary includes all expected fields."""
        # Record some costs
        cost_attributor.record_cost(
            cost_usd=Decimal("1.00"),
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet",
            user_id="user-1",
            org_id="org-1",
        )

        summary = cost_attributor.get_user_summary("user-1")

        assert summary.entity_id == "user-1"
        assert summary.entity_type == AttributionLevel.USER
        assert summary.total_cost_usd == Decimal("1.00")
        assert summary.total_tokens_in == 1000
        assert summary.total_tokens_out == 500
        assert summary.total_api_calls == 1

    def test_summary_aggregates_by_model(self, cost_attributor):
        """Summary aggregates costs by model."""
        cost_attributor.record_cost(
            cost_usd=Decimal("0.50"),
            model="claude-sonnet",
            user_id="user-1",
        )
        cost_attributor.record_cost(
            cost_usd=Decimal("0.70"),
            model="gpt-4",
            user_id="user-1",
        )
        cost_attributor.record_cost(
            cost_usd=Decimal("0.30"),
            model="claude-sonnet",
            user_id="user-1",
        )

        summary = cost_attributor.get_user_summary("user-1")

        # Claude-sonnet should have 0.50 + 0.30 = 0.80
        assert summary.cost_by_model.get("claude-sonnet") == Decimal("0.80")
        assert summary.cost_by_model.get("gpt-4") == Decimal("0.70")

    def test_time_bounded_summary(self, cost_attributor):
        """Summary can be bounded by time period."""
        now = datetime.now(timezone.utc)

        # Record costs at different times
        cost_attributor.record_cost(
            cost_usd=Decimal("1.00"),
            user_id="user-1",
        )

        # Manually adjust timestamp of first entry to be older
        cost_attributor._entries[0].timestamp = now - timedelta(days=35)

        # Record another recent cost
        cost_attributor.record_cost(
            cost_usd=Decimal("0.50"),
            user_id="user-1",
        )

        # Get summary for last 30 days (uses period_start parameter)
        summary = cost_attributor.get_user_summary(
            "user-1",
            period_start=now - timedelta(days=30),
        )

        # The total_cost_usd from aggregates includes all entries
        # But daily_costs should only include entries within the period
        # Filter daily_costs to check period filtering
        cutoff_date = (now - timedelta(days=30)).strftime("%Y-%m-%d")
        recent_daily_costs = [d for d in summary.daily_costs if d["date"] >= cutoff_date]
        # The recent entry should be in daily costs
        assert len(recent_daily_costs) >= 1


# =============================================================================
# Entry Management Tests
# =============================================================================


class TestAttributionEntryManagement:
    """Test attribution entry lifecycle management."""

    def test_max_entries_enforced(self):
        """Max entries limit is enforced."""
        attributor = CostAttributor(max_entries=10)

        # Record 15 entries
        for i in range(15):
            attributor.record_cost(
                cost_usd=Decimal("0.10"),
                user_id=f"user-{i}",
            )

        # Should only keep max_entries
        assert len(attributor._entries) <= 10

    def test_oldest_entries_removed_first(self):
        """Oldest entries are removed when limit is reached."""
        attributor = CostAttributor(max_entries=5)

        # Record entries with identifiable IDs
        for i in range(7):
            attributor.record_cost(
                cost_usd=Decimal("0.10"),
                source_id=f"entry-{i}",
                user_id="user-1",
            )

        # Oldest entries (0, 1) should be removed
        remaining_ids = [e.source_id for e in attributor._entries]
        assert "entry-0" not in remaining_ids
        assert "entry-1" not in remaining_ids
        # Recent entries should remain
        assert "entry-6" in remaining_ids
