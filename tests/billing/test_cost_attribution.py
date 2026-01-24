"""Tests for cost attribution system."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from aragora.billing.cost_attribution import (
    AllocationMethod,
    AttributionEntry,
    AttributionLevel,
    AttributionSummary,
    ChargebackReport,
    CostAllocation,
    CostAttributor,
    create_cost_attributor,
)


class TestAttributionEntry:
    """Tests for AttributionEntry dataclass."""

    def test_creation(self):
        """Test AttributionEntry creation."""
        entry = AttributionEntry(
            source_type="api_call",
            source_id="call-123",
            cost_usd=Decimal("0.50"),
            tokens_in=100,
            tokens_out=50,
            provider="anthropic",
            model="claude-sonnet",
            user_id="user-1",
            task_id="task-1",
        )

        assert entry.source_type == "api_call"
        assert entry.cost_usd == Decimal("0.50")
        assert entry.user_id == "user-1"
        assert entry.id is not None

    def test_to_dict(self):
        """Test AttributionEntry serialization."""
        entry = AttributionEntry(
            source_type="debate_round",
            cost_usd=Decimal("1.00"),
            user_id="user-1",
            debate_id="debate-1",
        )

        data = entry.to_dict()

        assert data["source_type"] == "debate_round"
        assert data["cost_usd"] == "1.00"
        assert data["user_id"] == "user-1"
        assert data["debate_id"] == "debate-1"
        assert "timestamp" in data


class TestAttributionSummary:
    """Tests for AttributionSummary dataclass."""

    def test_to_dict(self):
        """Test AttributionSummary serialization."""
        now = datetime.now(timezone.utc)
        summary = AttributionSummary(
            entity_id="user-1",
            entity_type=AttributionLevel.USER,
            period_start=now - timedelta(days=30),
            period_end=now,
            total_cost_usd=Decimal("100.00"),
            total_tokens_in=10000,
            total_tokens_out=5000,
            total_api_calls=50,
            cost_by_model={"claude-sonnet": Decimal("80.00")},
        )

        data = summary.to_dict()

        assert data["entity_id"] == "user-1"
        assert data["entity_type"] == "user"
        assert data["total_cost_usd"] == "100.00"
        assert data["cost_by_model"]["claude-sonnet"] == "80.00"


class TestChargebackReport:
    """Tests for ChargebackReport dataclass."""

    def test_to_dict(self):
        """Test ChargebackReport serialization."""
        report = ChargebackReport(
            org_id="org-1",
            total_cost_usd=Decimal("1000.00"),
            shared_costs_usd=Decimal("100.00"),
        )

        report.allocations_by_user["user-1"] = CostAllocation(
            entity_id="user-1",
            entity_type=AttributionLevel.USER,
            cost_usd=Decimal("500.00"),
            api_calls=100,
        )

        data = report.to_dict()

        assert data["org_id"] == "org-1"
        assert data["total_cost_usd"] == "1000.00"
        assert data["shared_costs_usd"] == "100.00"
        assert "user-1" in data["allocations_by_user"]
        assert data["allocations_by_user"]["user-1"]["cost_usd"] == "500.00"


class TestCostAttributor:
    """Tests for CostAttributor class."""

    def test_init(self):
        """Test CostAttributor initialization."""
        attributor = CostAttributor()

        assert attributor._max_entries == 10000
        assert len(attributor._entries) == 0

    def test_record_cost(self):
        """Test recording a cost."""
        attributor = CostAttributor()

        entry = attributor.record_cost(
            cost_usd=Decimal("0.50"),
            tokens_in=100,
            tokens_out=50,
            provider="anthropic",
            model="claude-sonnet",
            user_id="user-1",
            task_id="task-1",
            workspace_id="ws-1",
        )

        assert entry.cost_usd == Decimal("0.50")
        assert len(attributor._entries) == 1
        assert attributor._user_costs["user-1"]["total_cost"] == Decimal("0.50")

    def test_record_multiple_costs(self):
        """Test recording multiple costs."""
        attributor = CostAttributor()

        attributor.record_cost(cost_usd=Decimal("0.50"), user_id="user-1")
        attributor.record_cost(cost_usd=Decimal("0.30"), user_id="user-1")
        attributor.record_cost(cost_usd=Decimal("0.20"), user_id="user-2")

        assert attributor._user_costs["user-1"]["total_cost"] == Decimal("0.80")
        assert attributor._user_costs["user-1"]["api_calls"] == 2
        assert attributor._user_costs["user-2"]["total_cost"] == Decimal("0.20")

    def test_record_cost_with_team_mapping(self):
        """Test that team costs are aggregated when user-team mapping exists."""
        attributor = CostAttributor()

        # Set up team mapping
        attributor.set_user_team("user-1", "team-a")
        attributor.set_user_team("user-2", "team-a")

        attributor.record_cost(cost_usd=Decimal("0.50"), user_id="user-1")
        attributor.record_cost(cost_usd=Decimal("0.30"), user_id="user-2")

        assert attributor._team_costs["team-a"]["total_cost"] == Decimal("0.80")

    def test_record_cost_with_project_mapping(self):
        """Test that project costs are aggregated when task-project mapping exists."""
        attributor = CostAttributor()

        # Set up project mapping
        attributor.set_task_project("task-1", "project-x")
        attributor.set_task_project("task-2", "project-x")

        attributor.record_cost(cost_usd=Decimal("0.50"), task_id="task-1")
        attributor.record_cost(cost_usd=Decimal("0.30"), task_id="task-2")

        assert attributor._project_costs["project-x"]["total_cost"] == Decimal("0.80")

    def test_max_entries_limit(self):
        """Test that entries are bounded by max_entries."""
        attributor = CostAttributor(max_entries=10)

        # Record more than max
        for i in range(20):
            attributor.record_cost(
                cost_usd=Decimal("0.01"),
                user_id=f"user-{i}",
            )

        assert len(attributor._entries) == 10

    def test_get_user_summary(self):
        """Test getting user cost summary."""
        attributor = CostAttributor()

        attributor.record_cost(
            cost_usd=Decimal("0.50"),
            tokens_in=100,
            tokens_out=50,
            model="claude-sonnet",
            agent_name="claude",
            user_id="user-1",
        )
        attributor.record_cost(
            cost_usd=Decimal("0.30"),
            tokens_in=80,
            tokens_out=40,
            model="claude-haiku",
            agent_name="haiku",
            user_id="user-1",
        )

        summary = attributor.get_user_summary("user-1")

        assert summary.entity_id == "user-1"
        assert summary.entity_type == AttributionLevel.USER
        assert summary.total_cost_usd == Decimal("0.80")
        assert summary.total_tokens_in == 180
        assert summary.total_tokens_out == 90
        assert summary.total_api_calls == 2
        assert summary.cost_by_model["claude-sonnet"] == Decimal("0.50")
        assert summary.cost_by_model["claude-haiku"] == Decimal("0.30")

    def test_get_user_summary_averages(self):
        """Test that user summary calculates averages correctly."""
        attributor = CostAttributor()

        attributor.record_cost(
            cost_usd=Decimal("1.00"),
            tokens_in=100,
            tokens_out=100,
            user_id="user-1",
        )
        attributor.record_cost(
            cost_usd=Decimal("2.00"),
            tokens_in=200,
            tokens_out=200,
            user_id="user-1",
        )

        summary = attributor.get_user_summary("user-1")

        assert summary.avg_cost_per_call == Decimal("1.50")  # 3.00 / 2
        assert summary.avg_tokens_per_call == 300.0  # 600 / 2

    def test_get_task_summary(self):
        """Test getting task cost summary."""
        attributor = CostAttributor()

        attributor.record_cost(
            cost_usd=Decimal("0.50"),
            tokens_in=100,
            tokens_out=50,
            task_id="task-1",
            user_id="user-1",
            workspace_id="ws-1",
        )
        attributor.record_cost(
            cost_usd=Decimal("0.30"),
            tokens_in=80,
            tokens_out=40,
            task_id="task-1",
        )

        summary = attributor.get_task_summary("task-1")

        assert summary["task_id"] == "task-1"
        assert summary["total_cost_usd"] == "0.80"
        assert summary["tokens_in"] == 180
        assert summary["api_calls"] == 2
        assert summary["user_id"] == "user-1"
        assert summary["workspace_id"] == "ws-1"

    def test_get_team_summary(self):
        """Test getting team cost summary."""
        attributor = CostAttributor()

        attributor.set_user_team("user-1", "team-a")
        attributor.set_user_team("user-2", "team-a")

        attributor.record_cost(cost_usd=Decimal("0.50"), user_id="user-1")
        attributor.record_cost(cost_usd=Decimal("0.30"), user_id="user-2")

        summary = attributor.get_team_summary("team-a")

        assert summary["team_id"] == "team-a"
        assert summary["total_cost_usd"] == "0.80"
        assert summary["member_count"] == 2
        assert "user-1" in summary["member_costs"]
        assert summary["member_costs"]["user-1"]["cost_usd"] == "0.50"

    def test_get_project_summary(self):
        """Test getting project cost summary."""
        attributor = CostAttributor()

        attributor.set_task_project("task-1", "project-x")
        attributor.set_task_project("task-2", "project-x")

        attributor.record_cost(cost_usd=Decimal("0.50"), task_id="task-1")
        attributor.record_cost(cost_usd=Decimal("0.30"), task_id="task-2")

        summary = attributor.get_project_summary("project-x")

        assert summary["project_id"] == "project-x"
        assert summary["total_cost_usd"] == "0.80"
        assert summary["task_count"] == 2

    def test_generate_chargeback_report_direct(self):
        """Test generating chargeback report with direct allocation."""
        attributor = CostAttributor()

        attributor.record_cost(
            cost_usd=Decimal("0.50"),
            user_id="user-1",
            workspace_id="ws-1",
        )
        attributor.record_cost(
            cost_usd=Decimal("0.30"),
            user_id="user-2",
            workspace_id="ws-1",
        )

        report = attributor.generate_chargeback_report(
            workspace_id="ws-1",
            allocation_method=AllocationMethod.DIRECT,
        )

        assert report.total_cost_usd == Decimal("0.80")
        assert len(report.allocations_by_user) == 2
        assert report.allocations_by_user["user-1"].cost_usd == Decimal("0.50")
        assert report.allocations_by_user["user-2"].cost_usd == Decimal("0.30")

    def test_generate_chargeback_report_proportional(self):
        """Test generating chargeback report with proportional allocation."""
        attributor = CostAttributor()

        # Direct costs
        attributor.record_cost(
            cost_usd=Decimal("0.75"),
            user_id="user-1",
            workspace_id="ws-1",
        )
        attributor.record_cost(
            cost_usd=Decimal("0.25"),
            user_id="user-2",
            workspace_id="ws-1",
        )

        # Shared cost (no user)
        attributor.record_cost(
            cost_usd=Decimal("1.00"),
            workspace_id="ws-1",
        )

        report = attributor.generate_chargeback_report(
            workspace_id="ws-1",
            allocation_method=AllocationMethod.PROPORTIONAL,
        )

        assert report.total_cost_usd == Decimal("2.00")
        assert report.shared_costs_usd == Decimal("1.00")

        # User 1 had 75% of direct costs, should get 75% of shared
        # 0.75 + 0.75 = 1.50
        assert report.allocations_by_user["user-1"].cost_usd == Decimal("1.50")
        # User 2 had 25% of direct costs, should get 25% of shared
        # 0.25 + 0.25 = 0.50
        assert report.allocations_by_user["user-2"].cost_usd == Decimal("0.50")

    def test_generate_chargeback_report_equal(self):
        """Test generating chargeback report with equal allocation."""
        attributor = CostAttributor()

        # Direct costs
        attributor.record_cost(
            cost_usd=Decimal("0.50"),
            user_id="user-1",
            workspace_id="ws-1",
        )
        attributor.record_cost(
            cost_usd=Decimal("0.50"),
            user_id="user-2",
            workspace_id="ws-1",
        )

        # Shared cost
        attributor.record_cost(
            cost_usd=Decimal("1.00"),
            workspace_id="ws-1",
        )

        report = attributor.generate_chargeback_report(
            workspace_id="ws-1",
            allocation_method=AllocationMethod.EQUAL,
        )

        # Each user gets 0.50 + 0.50 = 1.00
        assert report.allocations_by_user["user-1"].cost_usd == Decimal("1.00")
        assert report.allocations_by_user["user-2"].cost_usd == Decimal("1.00")

    def test_generate_chargeback_report_with_teams(self):
        """Test that chargeback report includes team aggregations."""
        attributor = CostAttributor()

        attributor.set_user_team("user-1", "team-a")
        attributor.set_user_team("user-2", "team-a")
        attributor.set_user_team("user-3", "team-b")

        attributor.record_cost(cost_usd=Decimal("0.50"), user_id="user-1")
        attributor.record_cost(cost_usd=Decimal("0.30"), user_id="user-2")
        attributor.record_cost(cost_usd=Decimal("0.20"), user_id="user-3")

        report = attributor.generate_chargeback_report()

        assert len(report.allocations_by_team) == 2
        assert report.allocations_by_team["team-a"].cost_usd == Decimal("0.80")
        assert report.allocations_by_team["team-b"].cost_usd == Decimal("0.20")

    def test_get_top_users_by_cost(self):
        """Test getting top users by cost."""
        attributor = CostAttributor()

        attributor.record_cost(cost_usd=Decimal("0.10"), user_id="user-1")
        attributor.record_cost(cost_usd=Decimal("0.50"), user_id="user-2")
        attributor.record_cost(cost_usd=Decimal("0.30"), user_id="user-3")

        top_users = attributor.get_top_users_by_cost(limit=2)

        assert len(top_users) == 2
        assert top_users[0]["user_id"] == "user-2"
        assert top_users[0]["total_cost_usd"] == "0.50"
        assert top_users[1]["user_id"] == "user-3"

    def test_get_top_users_by_cost_filtered(self):
        """Test getting top users filtered by workspace."""
        attributor = CostAttributor()

        attributor.record_cost(
            cost_usd=Decimal("0.50"),
            user_id="user-1",
            workspace_id="ws-1",
        )
        attributor.record_cost(
            cost_usd=Decimal("0.30"),
            user_id="user-2",
            workspace_id="ws-2",
        )

        top_users = attributor.get_top_users_by_cost(workspace_id="ws-1", limit=10)

        assert len(top_users) == 1
        assert top_users[0]["user_id"] == "user-1"

    def test_get_cost_trends(self):
        """Test getting cost trends over time."""
        attributor = CostAttributor()

        # Create entries across multiple days
        now = datetime.now(timezone.utc)

        entry1 = attributor.record_cost(cost_usd=Decimal("0.50"), user_id="user-1")
        entry1.timestamp = now - timedelta(days=2)

        entry2 = attributor.record_cost(cost_usd=Decimal("0.30"), user_id="user-1")
        entry2.timestamp = now - timedelta(days=1)

        entry3 = attributor.record_cost(cost_usd=Decimal("0.20"), user_id="user-1")
        # entry3 stays at now

        trends = attributor.get_cost_trends(
            entity_type=AttributionLevel.USER,
            entity_id="user-1",
            period_days=7,
        )

        assert len(trends) == 3

    def test_create_cost_attributor(self):
        """Test factory function."""
        attributor = create_cost_attributor()

        assert isinstance(attributor, CostAttributor)


class TestAllocationMethod:
    """Tests for AllocationMethod enum."""

    def test_values(self):
        """Test allocation method values."""
        assert AllocationMethod.DIRECT.value == "direct"
        assert AllocationMethod.PROPORTIONAL.value == "proportional"
        assert AllocationMethod.EQUAL.value == "equal"
        assert AllocationMethod.WEIGHTED.value == "weighted"


class TestAttributionLevel:
    """Tests for AttributionLevel enum."""

    def test_values(self):
        """Test attribution level values."""
        assert AttributionLevel.USER.value == "user"
        assert AttributionLevel.TASK.value == "task"
        assert AttributionLevel.DEBATE.value == "debate"
        assert AttributionLevel.WORKSPACE.value == "workspace"
        assert AttributionLevel.ORGANIZATION.value == "organization"
        assert AttributionLevel.TEAM.value == "team"
        assert AttributionLevel.PROJECT.value == "project"
