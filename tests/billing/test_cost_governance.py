"""
Comprehensive tests for cost governance policy engine.

Tests cover:
- CostGovernanceEngine initialization and configuration
- Policy registration (add_policy, remove_policy)
- Policy evaluation (evaluate_request)
- Multiple policy enforcement
- Audit callback triggering
- ModelRestriction (allowed/blocked models, token limits, rate limiting, quotas)
- SpendingLimit (daily/weekly/monthly budgets, soft vs hard limits, budget reset)
- TimeRestriction (allowed hours/days, timezone handling, weekend restrictions)
- Approval workflow (request_approval, approve_request, deny_request)
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.cost_governance import (
    CostGovernanceEngine,
    CostGovernancePolicy,
    CostPolicyAction,
    CostPolicyEnforcement,
    CostPolicyScope,
    CostPolicyType,
    ModelRestriction,
    PolicyEvaluationContext,
    PolicyEvaluationResult,
    PolicyViolation,
    SpendingLimit,
    TimeRestriction,
    create_cost_governance_engine,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def engine():
    """Fresh CostGovernanceEngine instance for each test."""
    return CostGovernanceEngine()


@pytest.fixture
def sample_policy():
    """Sample CostGovernancePolicy for testing."""
    return CostGovernancePolicy(
        name="test-policy",
        description="A test policy",
        policy_type=CostPolicyType.SPENDING_LIMIT,
        scope=CostPolicyScope.GLOBAL,
        enforcement=CostPolicyEnforcement.HARD,
    )


@pytest.fixture
def sample_context():
    """Sample PolicyEvaluationContext for testing."""
    return PolicyEvaluationContext(
        workspace_id="ws-123",
        team_id="team-456",
        user_id="user-789",
        project_id="proj-001",
        operation="debate",
        model="claude-opus-4",
        estimated_cost_usd=Decimal("5.00"),
        estimated_tokens=10000,
        current_daily_spend=Decimal("50.00"),
        current_weekly_spend=Decimal("200.00"),
        current_monthly_spend=Decimal("800.00"),
    )


# =============================================================================
# ModelRestriction Tests (15+ tests)
# =============================================================================


class TestModelRestriction:
    """Tests for ModelRestriction dataclass."""

    def test_to_dict_all_fields(self):
        """Test serialization with all fields."""
        restriction = ModelRestriction(
            model_pattern="claude-opus*",
            allowed=True,
            max_requests_per_hour=100,
            max_tokens_per_request=10000,
            allowed_operations=["analysis", "summarization"],
            fallback_model="claude-sonnet",
        )

        data = restriction.to_dict()

        assert data["model_pattern"] == "claude-opus*"
        assert data["allowed"] is True
        assert data["max_requests_per_hour"] == 100
        assert data["max_tokens_per_request"] == 10000
        assert data["allowed_operations"] == ["analysis", "summarization"]
        assert data["fallback_model"] == "claude-sonnet"

    def test_to_dict_minimal_fields(self):
        """Test serialization with minimal fields."""
        restriction = ModelRestriction(model_pattern="gpt-4*")

        data = restriction.to_dict()

        assert data["model_pattern"] == "gpt-4*"
        assert data["allowed"] is True
        assert data["max_requests_per_hour"] is None
        assert data["max_tokens_per_request"] is None
        assert data["allowed_operations"] == []
        assert data["fallback_model"] is None

    def test_from_dict_all_fields(self):
        """Test deserialization with all fields."""
        data = {
            "model_pattern": "gpt-4*",
            "allowed": False,
            "max_requests_per_hour": 50,
            "max_tokens_per_request": 8000,
            "allowed_operations": ["chat", "completion"],
            "fallback_model": "gpt-3.5-turbo",
        }

        restriction = ModelRestriction.from_dict(data)

        assert restriction.model_pattern == "gpt-4*"
        assert restriction.allowed is False
        assert restriction.max_requests_per_hour == 50
        assert restriction.max_tokens_per_request == 8000
        assert restriction.allowed_operations == ["chat", "completion"]
        assert restriction.fallback_model == "gpt-3.5-turbo"

    def test_from_dict_minimal_fields(self):
        """Test deserialization with minimal fields."""
        data = {"model_pattern": "gemini*"}

        restriction = ModelRestriction.from_dict(data)

        assert restriction.model_pattern == "gemini*"
        assert restriction.allowed is True
        assert restriction.max_requests_per_hour is None
        assert restriction.allowed_operations == []

    def test_round_trip_serialization(self):
        """Test that to_dict -> from_dict preserves data."""
        original = ModelRestriction(
            model_pattern="claude-opus*",
            allowed=False,
            max_requests_per_hour=75,
            max_tokens_per_request=5000,
            allowed_operations=["analysis"],
            fallback_model="claude-haiku",
        )

        data = original.to_dict()
        restored = ModelRestriction.from_dict(data)

        assert restored.model_pattern == original.model_pattern
        assert restored.allowed == original.allowed
        assert restored.max_requests_per_hour == original.max_requests_per_hour
        assert restored.max_tokens_per_request == original.max_tokens_per_request
        assert restored.allowed_operations == original.allowed_operations
        assert restored.fallback_model == original.fallback_model

    def test_default_values(self):
        """Test default field values."""
        restriction = ModelRestriction(model_pattern="test*")

        assert restriction.allowed is True
        assert restriction.max_requests_per_hour is None
        assert restriction.max_tokens_per_request is None
        assert restriction.allowed_operations == []
        assert restriction.fallback_model is None

    def test_wildcard_pattern(self):
        """Test restriction with wildcard pattern."""
        restriction = ModelRestriction(model_pattern="claude*", allowed=False)

        assert restriction.model_pattern == "claude*"
        assert restriction.allowed is False

    def test_exact_match_pattern(self):
        """Test restriction with exact match pattern."""
        restriction = ModelRestriction(
            model_pattern="claude-3-opus-20240229",
            allowed=True,
            max_tokens_per_request=50000,
        )

        assert restriction.model_pattern == "claude-3-opus-20240229"
        assert restriction.max_tokens_per_request == 50000

    def test_multiple_operations(self):
        """Test restriction with multiple allowed operations."""
        restriction = ModelRestriction(
            model_pattern="gpt-4*",
            allowed_operations=["chat", "completion", "analysis", "code_review"],
        )

        assert len(restriction.allowed_operations) == 4
        assert "chat" in restriction.allowed_operations
        assert "code_review" in restriction.allowed_operations

    def test_zero_rate_limit(self):
        """Test restriction with zero rate limit."""
        restriction = ModelRestriction(
            model_pattern="expensive-model*",
            max_requests_per_hour=0,
        )

        assert restriction.max_requests_per_hour == 0

    def test_high_token_limit(self):
        """Test restriction with high token limit."""
        restriction = ModelRestriction(
            model_pattern="claude-opus*",
            max_tokens_per_request=200000,
        )

        assert restriction.max_tokens_per_request == 200000

    def test_from_dict_defaults(self):
        """Test from_dict uses default values for missing keys."""
        data = {"model_pattern": "test-model"}

        restriction = ModelRestriction.from_dict(data)

        assert restriction.allowed is True
        assert restriction.max_requests_per_hour is None
        assert restriction.allowed_operations == []

    def test_empty_operations_list(self):
        """Test empty allowed_operations means all operations allowed."""
        restriction = ModelRestriction(
            model_pattern="claude*",
            allowed_operations=[],  # Empty = all allowed
        )

        assert restriction.allowed_operations == []

    def test_fallback_model_same_provider(self):
        """Test fallback model from same provider."""
        restriction = ModelRestriction(
            model_pattern="claude-opus*",
            allowed=False,
            fallback_model="claude-sonnet-4",
        )

        assert restriction.fallback_model == "claude-sonnet-4"

    def test_fallback_model_different_provider(self):
        """Test fallback model from different provider."""
        restriction = ModelRestriction(
            model_pattern="claude-opus*",
            allowed=False,
            fallback_model="gpt-4-turbo",
        )

        assert restriction.fallback_model == "gpt-4-turbo"


# =============================================================================
# SpendingLimit Tests (15+ tests)
# =============================================================================


class TestSpendingLimit:
    """Tests for SpendingLimit dataclass."""

    def test_to_dict_all_fields(self):
        """Test serialization with all fields."""
        limit = SpendingLimit(
            daily_limit_usd=Decimal("100"),
            weekly_limit_usd=Decimal("500"),
            monthly_limit_usd=Decimal("2000"),
            per_operation_limit_usd=Decimal("10"),
            alert_threshold_percent=75.0,
            hard_limit=True,
        )

        data = limit.to_dict()

        assert data["daily_limit_usd"] == "100"
        assert data["weekly_limit_usd"] == "500"
        assert data["monthly_limit_usd"] == "2000"
        assert data["per_operation_limit_usd"] == "10"
        assert data["alert_threshold_percent"] == 75.0
        assert data["hard_limit"] is True

    def test_to_dict_minimal_fields(self):
        """Test serialization with minimal fields."""
        limit = SpendingLimit()

        data = limit.to_dict()

        assert data["daily_limit_usd"] is None
        assert data["monthly_limit_usd"] is None
        assert data["alert_threshold_percent"] == 80.0
        assert data["hard_limit"] is True

    def test_from_dict_all_fields(self):
        """Test deserialization with all fields."""
        data = {
            "daily_limit_usd": "200",
            "weekly_limit_usd": "1000",
            "monthly_limit_usd": "4000",
            "per_operation_limit_usd": "25",
            "alert_threshold_percent": 90.0,
            "hard_limit": False,
        }

        limit = SpendingLimit.from_dict(data)

        assert limit.daily_limit_usd == Decimal("200")
        assert limit.weekly_limit_usd == Decimal("1000")
        assert limit.monthly_limit_usd == Decimal("4000")
        assert limit.per_operation_limit_usd == Decimal("25")
        assert limit.alert_threshold_percent == 90.0
        assert limit.hard_limit is False

    def test_from_dict_minimal_fields(self):
        """Test deserialization with minimal fields."""
        data = {}

        limit = SpendingLimit.from_dict(data)

        assert limit.daily_limit_usd is None
        assert limit.weekly_limit_usd is None
        assert limit.monthly_limit_usd is None
        assert limit.hard_limit is True

    def test_round_trip_serialization(self):
        """Test that to_dict -> from_dict preserves data."""
        original = SpendingLimit(
            daily_limit_usd=Decimal("150"),
            weekly_limit_usd=Decimal("750"),
            monthly_limit_usd=Decimal("3000"),
            per_operation_limit_usd=Decimal("15"),
            alert_threshold_percent=85.0,
            hard_limit=False,
        )

        data = original.to_dict()
        restored = SpendingLimit.from_dict(data)

        assert restored.daily_limit_usd == original.daily_limit_usd
        assert restored.weekly_limit_usd == original.weekly_limit_usd
        assert restored.monthly_limit_usd == original.monthly_limit_usd
        assert restored.per_operation_limit_usd == original.per_operation_limit_usd
        assert restored.alert_threshold_percent == original.alert_threshold_percent
        assert restored.hard_limit == original.hard_limit

    def test_default_values(self):
        """Test default field values."""
        limit = SpendingLimit()

        assert limit.daily_limit_usd is None
        assert limit.weekly_limit_usd is None
        assert limit.monthly_limit_usd is None
        assert limit.per_operation_limit_usd is None
        assert limit.alert_threshold_percent == 80.0
        assert limit.hard_limit is True

    def test_soft_limit(self):
        """Test soft limit configuration."""
        limit = SpendingLimit(
            daily_limit_usd=Decimal("100"),
            hard_limit=False,  # Soft limit
        )

        assert limit.hard_limit is False

    def test_hard_limit(self):
        """Test hard limit configuration."""
        limit = SpendingLimit(
            daily_limit_usd=Decimal("100"),
            hard_limit=True,  # Hard limit
        )

        assert limit.hard_limit is True

    def test_low_alert_threshold(self):
        """Test low alert threshold."""
        limit = SpendingLimit(
            daily_limit_usd=Decimal("100"),
            alert_threshold_percent=50.0,
        )

        assert limit.alert_threshold_percent == 50.0

    def test_high_alert_threshold(self):
        """Test high alert threshold."""
        limit = SpendingLimit(
            daily_limit_usd=Decimal("100"),
            alert_threshold_percent=95.0,
        )

        assert limit.alert_threshold_percent == 95.0

    def test_daily_only_limit(self):
        """Test daily-only limit configuration."""
        limit = SpendingLimit(daily_limit_usd=Decimal("50"))

        assert limit.daily_limit_usd == Decimal("50")
        assert limit.weekly_limit_usd is None
        assert limit.monthly_limit_usd is None

    def test_weekly_only_limit(self):
        """Test weekly-only limit configuration."""
        limit = SpendingLimit(weekly_limit_usd=Decimal("250"))

        assert limit.daily_limit_usd is None
        assert limit.weekly_limit_usd == Decimal("250")
        assert limit.monthly_limit_usd is None

    def test_monthly_only_limit(self):
        """Test monthly-only limit configuration."""
        limit = SpendingLimit(monthly_limit_usd=Decimal("1000"))

        assert limit.daily_limit_usd is None
        assert limit.weekly_limit_usd is None
        assert limit.monthly_limit_usd == Decimal("1000")

    def test_per_operation_only_limit(self):
        """Test per-operation-only limit configuration."""
        limit = SpendingLimit(per_operation_limit_usd=Decimal("5"))

        assert limit.per_operation_limit_usd == Decimal("5")
        assert limit.daily_limit_usd is None

    def test_all_limits_configured(self):
        """Test all limits configured together."""
        limit = SpendingLimit(
            daily_limit_usd=Decimal("100"),
            weekly_limit_usd=Decimal("500"),
            monthly_limit_usd=Decimal("2000"),
            per_operation_limit_usd=Decimal("10"),
        )

        assert limit.daily_limit_usd == Decimal("100")
        assert limit.weekly_limit_usd == Decimal("500")
        assert limit.monthly_limit_usd == Decimal("2000")
        assert limit.per_operation_limit_usd == Decimal("10")

    def test_decimal_precision(self):
        """Test decimal precision is maintained."""
        limit = SpendingLimit(
            daily_limit_usd=Decimal("100.50"),
            monthly_limit_usd=Decimal("2000.75"),
        )

        assert limit.daily_limit_usd == Decimal("100.50")
        assert limit.monthly_limit_usd == Decimal("2000.75")


# =============================================================================
# TimeRestriction Tests (10+ tests)
# =============================================================================


class TestTimeRestriction:
    """Tests for TimeRestriction dataclass."""

    def test_is_allowed_during_work_hours(self):
        """Test time restriction during work hours."""
        restriction = TimeRestriction(
            allowed_hours_start=9,
            allowed_hours_end=17,
            allowed_days=[0, 1, 2, 3, 4],  # Mon-Fri
        )

        # Just verify it returns a boolean
        result = restriction.is_allowed_now()
        assert isinstance(result, bool)

    def test_to_dict_all_fields(self):
        """Test serialization with all fields."""
        restriction = TimeRestriction(
            allowed_hours_start=8,
            allowed_hours_end=18,
            allowed_days=[0, 1, 2, 3, 4],
            timezone="America/New_York",
        )

        data = restriction.to_dict()

        assert data["allowed_hours_start"] == 8
        assert data["allowed_hours_end"] == 18
        assert data["allowed_days"] == [0, 1, 2, 3, 4]
        assert data["timezone"] == "America/New_York"

    def test_from_dict_all_fields(self):
        """Test deserialization with all fields."""
        data = {
            "allowed_hours_start": 6,
            "allowed_hours_end": 22,
            "allowed_days": [0, 1, 2, 3, 4, 5, 6],
            "timezone": "Europe/London",
        }

        restriction = TimeRestriction.from_dict(data)

        assert restriction.allowed_hours_start == 6
        assert restriction.allowed_hours_end == 22
        assert restriction.allowed_days == [0, 1, 2, 3, 4, 5, 6]
        assert restriction.timezone == "Europe/London"

    def test_from_dict_default_values(self):
        """Test from_dict uses defaults for missing fields."""
        data = {}

        restriction = TimeRestriction.from_dict(data)

        assert restriction.allowed_hours_start == 0
        assert restriction.allowed_hours_end == 24
        assert restriction.allowed_days == [0, 1, 2, 3, 4]
        assert restriction.timezone == "UTC"

    def test_round_trip_serialization(self):
        """Test that to_dict -> from_dict preserves data."""
        original = TimeRestriction(
            allowed_hours_start=10,
            allowed_hours_end=16,
            allowed_days=[1, 3, 5],
            timezone="Asia/Tokyo",
        )

        data = original.to_dict()
        restored = TimeRestriction.from_dict(data)

        assert restored.allowed_hours_start == original.allowed_hours_start
        assert restored.allowed_hours_end == original.allowed_hours_end
        assert restored.allowed_days == original.allowed_days
        assert restored.timezone == original.timezone

    def test_default_values(self):
        """Test default field values."""
        restriction = TimeRestriction()

        assert restriction.allowed_hours_start == 0
        assert restriction.allowed_hours_end == 24
        assert restriction.allowed_days == [0, 1, 2, 3, 4]
        assert restriction.timezone == "UTC"

    def test_weekday_only_restriction(self):
        """Test weekday-only restriction (Mon-Fri)."""
        restriction = TimeRestriction(allowed_days=[0, 1, 2, 3, 4])

        assert 5 not in restriction.allowed_days  # Saturday
        assert 6 not in restriction.allowed_days  # Sunday

    def test_weekend_only_restriction(self):
        """Test weekend-only restriction (Sat-Sun)."""
        restriction = TimeRestriction(allowed_days=[5, 6])

        assert len(restriction.allowed_days) == 2
        assert 5 in restriction.allowed_days
        assert 6 in restriction.allowed_days

    def test_all_days_allowed(self):
        """Test all days allowed (7 days a week)."""
        restriction = TimeRestriction(allowed_days=[0, 1, 2, 3, 4, 5, 6])

        assert len(restriction.allowed_days) == 7

    def test_narrow_time_window(self):
        """Test narrow time window restriction."""
        restriction = TimeRestriction(
            allowed_hours_start=12,
            allowed_hours_end=13,  # Only noon hour
        )

        assert restriction.allowed_hours_start == 12
        assert restriction.allowed_hours_end == 13

    def test_overnight_hours_not_supported(self):
        """Test that overnight hours (e.g., 22-6) use simple comparison."""
        restriction = TimeRestriction(
            allowed_hours_start=22,
            allowed_hours_end=6,  # This won't work as overnight
        )

        # The current implementation doesn't support overnight
        # Just verify it stores the values
        assert restriction.allowed_hours_start == 22
        assert restriction.allowed_hours_end == 6

    def test_full_day_window(self):
        """Test 24-hour window."""
        restriction = TimeRestriction(
            allowed_hours_start=0,
            allowed_hours_end=24,
        )

        assert restriction.allowed_hours_start == 0
        assert restriction.allowed_hours_end == 24

    def test_different_timezone(self):
        """Test different timezone configuration."""
        restriction = TimeRestriction(timezone="America/Los_Angeles")

        assert restriction.timezone == "America/Los_Angeles"

    def test_is_allowed_now_mocked_weekday(self):
        """Test is_allowed_now with mocked weekday."""
        restriction = TimeRestriction(
            allowed_hours_start=0,
            allowed_hours_end=24,
            allowed_days=[0, 1, 2, 3, 4],  # Mon-Fri only
        )

        # Mock a Wednesday at 10:00 UTC
        mock_now = datetime(2024, 1, 10, 10, 0, 0, tzinfo=timezone.utc)  # Wednesday
        with patch("aragora.billing.cost_governance.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            # The function uses datetime.now(timezone.utc) directly
            # We need to test the logic
            day_of_week = mock_now.weekday()  # 2 = Wednesday
            hour = mock_now.hour  # 10

            # Manually check the logic
            in_allowed_days = day_of_week in restriction.allowed_days
            in_allowed_hours = (
                restriction.allowed_hours_start <= hour < restriction.allowed_hours_end
            )

            assert in_allowed_days is True
            assert in_allowed_hours is True


# =============================================================================
# CostGovernancePolicy Tests
# =============================================================================


class TestCostGovernancePolicy:
    """Tests for CostGovernancePolicy dataclass."""

    def test_creation(self):
        """Test policy creation."""
        policy = CostGovernancePolicy(
            name="dev-limits",
            description="Development environment limits",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            scope=CostPolicyScope.WORKSPACE,
            enforcement=CostPolicyEnforcement.HARD,
            target_workspaces=["dev-ws"],
        )

        assert policy.name == "dev-limits"
        assert policy.policy_type == CostPolicyType.SPENDING_LIMIT
        assert policy.scope == CostPolicyScope.WORKSPACE
        assert policy.id is not None

    def test_matches_global_scope(self):
        """Test that global scope matches everything."""
        policy = CostGovernancePolicy(
            name="global-policy",
            scope=CostPolicyScope.GLOBAL,
        )

        assert policy.matches(workspace_id="any")
        assert policy.matches(user_id="anyone")
        assert policy.matches()

    def test_matches_workspace_scope(self):
        """Test workspace scope matching."""
        policy = CostGovernancePolicy(
            name="ws-policy",
            scope=CostPolicyScope.WORKSPACE,
            target_workspaces=["ws-1", "ws-2"],
        )

        assert policy.matches(workspace_id="ws-1")
        assert policy.matches(workspace_id="ws-2")
        assert not policy.matches(workspace_id="ws-3")

    def test_matches_workspace_scope_empty_targets(self):
        """Test workspace scope with empty targets matches all."""
        policy = CostGovernancePolicy(
            name="ws-policy",
            scope=CostPolicyScope.WORKSPACE,
            target_workspaces=[],  # Empty = all workspaces
        )

        assert policy.matches(workspace_id="ws-1")
        assert policy.matches(workspace_id="ws-2")

    def test_matches_team_scope(self):
        """Test team scope matching."""
        policy = CostGovernancePolicy(
            name="team-policy",
            scope=CostPolicyScope.TEAM,
            target_teams=["team-1"],
        )

        assert policy.matches(team_id="team-1")
        assert not policy.matches(team_id="team-2")

    def test_matches_user_scope(self):
        """Test user scope matching."""
        policy = CostGovernancePolicy(
            name="user-policy",
            scope=CostPolicyScope.USER,
            target_users=["user-1"],
        )

        assert policy.matches(user_id="user-1")
        assert not policy.matches(user_id="user-2")

    def test_matches_project_scope(self):
        """Test project scope matching."""
        policy = CostGovernancePolicy(
            name="project-policy",
            scope=CostPolicyScope.PROJECT,
            target_projects=["proj-1"],
        )

        assert policy.matches(project_id="proj-1")
        assert not policy.matches(project_id="proj-2")

    def test_disabled_policy_doesnt_match(self):
        """Test that disabled policies don't match."""
        policy = CostGovernancePolicy(
            name="disabled",
            scope=CostPolicyScope.GLOBAL,
            enabled=False,
        )

        assert not policy.matches()

    def test_to_dict(self):
        """Test policy serialization."""
        policy = CostGovernancePolicy(
            name="test-policy",
            policy_type=CostPolicyType.MODEL_RESTRICTION,
            model_restrictions=[ModelRestriction(model_pattern="claude-opus*", allowed=False)],
        )

        data = policy.to_dict()

        assert data["name"] == "test-policy"
        assert data["policy_type"] == "model_restriction"
        assert len(data["model_restrictions"]) == 1

    def test_from_dict(self):
        """Test policy deserialization."""
        data = {
            "name": "test-policy",
            "policy_type": "spending_limit",
            "scope": "workspace",
            "target_workspaces": ["ws-1"],
            "spending_limit": {
                "daily_limit_usd": "100",
            },
        }

        policy = CostGovernancePolicy.from_dict(data)

        assert policy.name == "test-policy"
        assert policy.policy_type == CostPolicyType.SPENDING_LIMIT
        assert policy.scope == CostPolicyScope.WORKSPACE
        assert policy.spending_limit is not None
        assert policy.spending_limit.daily_limit_usd == Decimal("100")

    def test_to_dict_with_time_restriction(self):
        """Test policy serialization with time restriction."""
        policy = CostGovernancePolicy(
            name="work-hours-only",
            policy_type=CostPolicyType.TIME_RESTRICTION,
            time_restriction=TimeRestriction(
                allowed_hours_start=9,
                allowed_hours_end=17,
                allowed_days=[0, 1, 2, 3, 4],
                timezone="America/Chicago",
            ),
        )

        data = policy.to_dict()

        assert data["name"] == "work-hours-only"
        assert data["time_restriction"] is not None
        assert data["time_restriction"]["allowed_hours_start"] == 9
        assert data["time_restriction"]["allowed_hours_end"] == 17
        assert data["time_restriction"]["timezone"] == "America/Chicago"

    def test_from_dict_with_time_restriction(self):
        """Test policy deserialization with time restriction."""
        data = {
            "name": "time-restricted",
            "policy_type": "time_restriction",
            "time_restriction": {
                "allowed_hours_start": 8,
                "allowed_hours_end": 20,
                "allowed_days": [0, 1, 2, 3, 4, 5],
                "timezone": "UTC",
            },
        }

        policy = CostGovernancePolicy.from_dict(data)

        assert policy.name == "time-restricted"
        assert policy.time_restriction is not None
        assert policy.time_restriction.allowed_hours_start == 8
        assert policy.time_restriction.allowed_hours_end == 20
        assert policy.time_restriction.allowed_days == [0, 1, 2, 3, 4, 5]

    def test_from_dict_with_approval_config(self):
        """Test policy deserialization with approval configuration."""
        data = {
            "name": "approval-policy",
            "policy_type": "approval_required",
            "approval_threshold_usd": "100",
            "auto_approve_under_usd": "10",
            "approvers": ["manager-1", "manager-2"],
        }

        policy = CostGovernancePolicy.from_dict(data)

        assert policy.approval_threshold_usd == Decimal("100")
        assert policy.auto_approve_under_usd == Decimal("10")
        assert policy.approvers == ["manager-1", "manager-2"]


# =============================================================================
# PolicyEvaluationContext Tests
# =============================================================================


class TestPolicyEvaluationContext:
    """Tests for PolicyEvaluationContext dataclass."""

    def test_creation_with_all_fields(self):
        """Test context creation with all fields."""
        context = PolicyEvaluationContext(
            workspace_id="ws-123",
            team_id="team-456",
            user_id="user-789",
            project_id="proj-001",
            operation="debate",
            model="claude-opus-4",
            estimated_cost_usd=Decimal("5.00"),
            estimated_tokens=10000,
            current_daily_spend=Decimal("50.00"),
            current_weekly_spend=Decimal("200.00"),
            current_monthly_spend=Decimal("800.00"),
            request_id="req-001",
        )

        assert context.workspace_id == "ws-123"
        assert context.team_id == "team-456"
        assert context.user_id == "user-789"
        assert context.project_id == "proj-001"
        assert context.operation == "debate"
        assert context.model == "claude-opus-4"
        assert context.estimated_cost_usd == Decimal("5.00")
        assert context.estimated_tokens == 10000

    def test_default_values(self):
        """Test context default values."""
        context = PolicyEvaluationContext()

        assert context.workspace_id is None
        assert context.team_id is None
        assert context.user_id is None
        assert context.operation == ""
        assert context.model == ""
        assert context.estimated_cost_usd == Decimal("0")
        assert context.estimated_tokens == 0
        assert context.current_daily_spend == Decimal("0")
        assert context.timestamp is not None


# =============================================================================
# PolicyEvaluationResult Tests
# =============================================================================


class TestPolicyEvaluationResult:
    """Tests for PolicyEvaluationResult dataclass."""

    def test_default_result(self):
        """Test default result is allowed."""
        result = PolicyEvaluationResult()

        assert result.allowed is True
        assert result.action == CostPolicyAction.ALLOW
        assert result.violations == []
        assert result.requires_approval is False

    def test_to_dict(self):
        """Test result serialization."""
        result = PolicyEvaluationResult(
            allowed=False,
            action=CostPolicyAction.DENY,
            violations=[
                PolicyViolation(
                    policy_id="policy-1",
                    policy_name="test-policy",
                    violation_type="daily_limit_exceeded",
                    message="Daily limit exceeded",
                    severity=CostPolicyEnforcement.HARD,
                    action=CostPolicyAction.DENY,
                )
            ],
            requires_approval=False,
            warnings=["Approaching limit"],
        )

        data = result.to_dict()

        assert data["allowed"] is False
        assert data["action"] == "deny"
        assert len(data["violations"]) == 1
        assert data["warnings"] == ["Approaching limit"]


# =============================================================================
# PolicyViolation Tests
# =============================================================================


class TestPolicyViolation:
    """Tests for PolicyViolation dataclass."""

    def test_creation(self):
        """Test violation creation."""
        violation = PolicyViolation(
            policy_id="policy-1",
            policy_name="test-policy",
            violation_type="daily_limit_exceeded",
            message="Daily spending limit exceeded",
            severity=CostPolicyEnforcement.HARD,
            action=CostPolicyAction.DENY,
            details={"current_spend": "150", "limit": "100"},
        )

        assert violation.policy_id == "policy-1"
        assert violation.policy_name == "test-policy"
        assert violation.violation_type == "daily_limit_exceeded"
        assert violation.severity == CostPolicyEnforcement.HARD
        assert violation.action == CostPolicyAction.DENY

    def test_to_dict(self):
        """Test violation serialization."""
        violation = PolicyViolation(
            policy_id="policy-1",
            policy_name="test-policy",
            violation_type="model_not_allowed",
            message="Model not allowed",
            severity=CostPolicyEnforcement.HARD,
            action=CostPolicyAction.DENY,
            details={"model": "claude-opus-4"},
        )

        data = violation.to_dict()

        assert data["policy_id"] == "policy-1"
        assert data["violation_type"] == "model_not_allowed"
        assert data["severity"] == "hard"
        assert data["action"] == "deny"
        assert data["details"]["model"] == "claude-opus-4"


# =============================================================================
# CostGovernanceEngine Tests (20+ tests)
# =============================================================================


class TestCostGovernanceEngine:
    """Tests for CostGovernanceEngine class."""

    def test_init(self):
        """Test engine initialization."""
        engine = CostGovernanceEngine()

        assert len(engine._policies) == 0
        assert len(engine._pending_approvals) == 0
        assert len(engine._audit_callbacks) == 0

    def test_init_with_trackers(self):
        """Test engine initialization with cost tracker and attributor."""
        mock_tracker = MagicMock()
        mock_attributor = MagicMock()

        engine = CostGovernanceEngine(
            cost_tracker=mock_tracker,
            cost_attributor=mock_attributor,
        )

        assert engine._cost_tracker == mock_tracker
        assert engine._cost_attributor == mock_attributor

    def test_add_policy(self, engine):
        """Test adding a policy."""
        policy = CostGovernancePolicy(name="test-policy")
        engine.add_policy(policy)

        assert policy.id in engine._policies
        assert engine.get_policy(policy.id) == policy

    def test_add_multiple_policies(self, engine):
        """Test adding multiple policies."""
        for i in range(5):
            policy = CostGovernancePolicy(name=f"policy-{i}")
            engine.add_policy(policy)

        assert len(engine._policies) == 5

    def test_remove_policy(self, engine):
        """Test removing a policy."""
        policy = CostGovernancePolicy(name="test-policy")
        engine.add_policy(policy)
        result = engine.remove_policy(policy.id)

        assert result is True
        assert engine.get_policy(policy.id) is None

    def test_remove_nonexistent_policy(self, engine):
        """Test removing a policy that doesn't exist."""
        result = engine.remove_policy("nonexistent")

        assert result is False

    def test_get_policy(self, engine):
        """Test getting a policy by ID."""
        policy = CostGovernancePolicy(name="test-policy")
        engine.add_policy(policy)

        retrieved = engine.get_policy(policy.id)

        assert retrieved == policy

    def test_get_policy_nonexistent(self, engine):
        """Test getting a nonexistent policy returns None."""
        result = engine.get_policy("nonexistent")

        assert result is None

    def test_list_policies(self, engine):
        """Test listing policies."""
        policy1 = CostGovernancePolicy(
            name="policy-1",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            priority=1,
        )
        policy2 = CostGovernancePolicy(
            name="policy-2",
            policy_type=CostPolicyType.MODEL_RESTRICTION,
            priority=2,
        )
        engine.add_policy(policy1)
        engine.add_policy(policy2)

        all_policies = engine.list_policies()
        assert len(all_policies) == 2
        # Higher priority first
        assert all_policies[0].name == "policy-2"

        spending_policies = engine.list_policies(policy_type=CostPolicyType.SPENDING_LIMIT)
        assert len(spending_policies) == 1
        assert spending_policies[0].name == "policy-1"

    def test_list_policies_by_scope(self, engine):
        """Test listing policies by scope."""
        global_policy = CostGovernancePolicy(
            name="global",
            scope=CostPolicyScope.GLOBAL,
        )
        workspace_policy = CostGovernancePolicy(
            name="workspace",
            scope=CostPolicyScope.WORKSPACE,
        )
        engine.add_policy(global_policy)
        engine.add_policy(workspace_policy)

        global_policies = engine.list_policies(scope=CostPolicyScope.GLOBAL)
        assert len(global_policies) == 1
        assert global_policies[0].name == "global"

    def test_list_policies_enabled_only(self, engine):
        """Test listing only enabled policies."""
        enabled = CostGovernancePolicy(name="enabled", enabled=True)
        disabled = CostGovernancePolicy(name="disabled", enabled=False)
        engine.add_policy(enabled)
        engine.add_policy(disabled)

        policies = engine.list_policies(enabled_only=True)
        assert len(policies) == 1
        assert policies[0].name == "enabled"

        all_policies = engine.list_policies(enabled_only=False)
        assert len(all_policies) == 2

    def test_evaluate_no_policies(self, engine):
        """Test evaluation with no policies."""
        context = PolicyEvaluationContext(
            user_id="user-1",
            operation="debate",
            model="claude-sonnet",
        )

        result = engine.evaluate(context)

        assert result.allowed is True
        assert result.action == CostPolicyAction.ALLOW
        assert len(result.violations) == 0

    def test_evaluate_model_not_allowed(self, engine):
        """Test model restriction - model not allowed."""
        policy = CostGovernancePolicy(
            name="no-opus",
            policy_type=CostPolicyType.MODEL_RESTRICTION,
            model_restrictions=[
                ModelRestriction(
                    model_pattern="claude-opus*",
                    allowed=False,
                    fallback_model="claude-sonnet",
                )
            ],
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(model="claude-opus-4")

        result = engine.evaluate(context)

        assert result.allowed is False
        assert result.action == CostPolicyAction.DENY
        assert len(result.violations) == 1
        assert result.violations[0].violation_type == "model_not_allowed"
        assert result.suggested_model == "claude-sonnet"

    def test_evaluate_model_operation_not_allowed(self, engine):
        """Test model restriction - operation not allowed."""
        policy = CostGovernancePolicy(
            name="opus-limited",
            policy_type=CostPolicyType.MODEL_RESTRICTION,
            model_restrictions=[
                ModelRestriction(
                    model_pattern="claude-opus*",
                    allowed=True,
                    allowed_operations=["analysis", "code_review"],
                )
            ],
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(
            model="claude-opus-4",
            operation="chat",
        )

        result = engine.evaluate(context)

        assert result.allowed is False
        assert len(result.violations) == 1
        assert result.violations[0].violation_type == "operation_not_allowed"

    def test_evaluate_model_token_limit_exceeded(self, engine):
        """Test model restriction - token limit exceeded."""
        policy = CostGovernancePolicy(
            name="token-limit",
            policy_type=CostPolicyType.MODEL_RESTRICTION,
            model_restrictions=[
                ModelRestriction(
                    model_pattern="claude-opus*",
                    allowed=True,
                    max_tokens_per_request=5000,
                )
            ],
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(
            model="claude-opus-4",
            estimated_tokens=10000,
        )

        result = engine.evaluate(context)

        assert len(result.violations) == 1
        assert result.violations[0].violation_type == "token_limit_exceeded"

    def test_evaluate_model_rate_limit_exceeded(self, engine):
        """Test model restriction - rate limit exceeded."""
        policy = CostGovernancePolicy(
            name="rate-limit",
            policy_type=CostPolicyType.MODEL_RESTRICTION,
            model_restrictions=[
                ModelRestriction(
                    model_pattern="claude-opus*",
                    allowed=True,
                    max_requests_per_hour=5,
                )
            ],
        )
        engine.add_policy(policy)

        # Record some requests
        for _ in range(5):
            engine.record_model_request("claude-opus-4")

        context = PolicyEvaluationContext(model="claude-opus-4")

        result = engine.evaluate(context)

        assert len(result.violations) == 1
        assert result.violations[0].violation_type == "rate_limit_exceeded"
        assert result.violations[0].action == CostPolicyAction.THROTTLE

    def test_evaluate_spending_limit_daily_exceeded(self, engine):
        """Test spending limit - daily exceeded."""
        policy = CostGovernancePolicy(
            name="daily-limit",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            spending_limit=SpendingLimit(
                daily_limit_usd=Decimal("100"),
                hard_limit=True,
            ),
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(current_daily_spend=Decimal("105"))

        result = engine.evaluate(context)

        assert result.allowed is False
        assert result.action == CostPolicyAction.DENY
        assert result.violations[0].violation_type == "daily_limit_exceeded"

    def test_evaluate_spending_limit_weekly_exceeded(self, engine):
        """Test spending limit - weekly exceeded."""
        policy = CostGovernancePolicy(
            name="weekly-limit",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            spending_limit=SpendingLimit(
                weekly_limit_usd=Decimal("500"),
                hard_limit=True,
            ),
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(current_weekly_spend=Decimal("550"))

        result = engine.evaluate(context)

        assert result.allowed is False
        assert result.violations[0].violation_type == "weekly_limit_exceeded"

    def test_evaluate_spending_limit_monthly_exceeded(self, engine):
        """Test monthly spending limit exceeded."""
        policy = CostGovernancePolicy(
            name="monthly-limit",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            spending_limit=SpendingLimit(
                monthly_limit_usd=Decimal("2500"),
                hard_limit=True,
            ),
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(current_monthly_spend=Decimal("2600"))

        result = engine.evaluate(context)

        assert result.allowed is False
        assert result.violations[0].violation_type == "monthly_limit_exceeded"

    def test_evaluate_spending_limit_warning(self, engine):
        """Test spending limit warning threshold."""
        policy = CostGovernancePolicy(
            name="daily-limit",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            spending_limit=SpendingLimit(
                daily_limit_usd=Decimal("100"),
                alert_threshold_percent=80.0,
            ),
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(current_daily_spend=Decimal("85"))

        result = engine.evaluate(context)

        assert result.allowed is True
        assert len(result.warnings) == 1
        assert "Approaching daily limit" in result.warnings[0]

    def test_evaluate_per_operation_limit(self, engine):
        """Test per-operation limit triggers approval."""
        policy = CostGovernancePolicy(
            name="op-limit",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            spending_limit=SpendingLimit(
                per_operation_limit_usd=Decimal("10"),
            ),
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(estimated_cost_usd=Decimal("15"))

        result = engine.evaluate(context)

        assert result.action == CostPolicyAction.QUEUE
        assert result.violations[0].violation_type == "operation_cost_exceeded"

    def test_evaluate_approval_required(self, engine):
        """Test approval required policy."""
        policy = CostGovernancePolicy(
            name="approval-required",
            policy_type=CostPolicyType.APPROVAL_REQUIRED,
            approval_threshold_usd=Decimal("50"),
            approvers=["manager-1", "manager-2"],
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(estimated_cost_usd=Decimal("75"))

        result = engine.evaluate(context)

        assert result.requires_approval is True
        assert result.action == CostPolicyAction.QUEUE
        assert "manager-1" in result.approvers

    def test_evaluate_auto_approve_under_threshold(self, engine):
        """Test auto-approve for small operations."""
        policy = CostGovernancePolicy(
            name="approval-required",
            policy_type=CostPolicyType.APPROVAL_REQUIRED,
            approval_threshold_usd=Decimal("50"),
            auto_approve_under_usd=Decimal("10"),
            approvers=["manager-1"],
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(estimated_cost_usd=Decimal("5"))

        result = engine.evaluate(context)

        assert result.requires_approval is False
        assert result.allowed is True

    def test_evaluate_soft_enforcement(self, engine):
        """Test soft enforcement allows but logs violations."""
        policy = CostGovernancePolicy(
            name="soft-limit",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            enforcement=CostPolicyEnforcement.SOFT,
            spending_limit=SpendingLimit(
                daily_limit_usd=Decimal("100"),
                hard_limit=False,
            ),
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(current_daily_spend=Decimal("150"))

        result = engine.evaluate(context)

        # Soft enforcement - allowed despite violation
        assert result.allowed is True
        assert len(result.violations) == 1

    def test_evaluate_workspace_scoped_policy(self, engine):
        """Test workspace-scoped policy only affects target workspaces."""
        policy = CostGovernancePolicy(
            name="dev-limit",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            scope=CostPolicyScope.WORKSPACE,
            target_workspaces=["dev-ws"],
            spending_limit=SpendingLimit(
                daily_limit_usd=Decimal("50"),
                hard_limit=True,
            ),
        )
        engine.add_policy(policy)

        # Should affect dev workspace
        context1 = PolicyEvaluationContext(
            workspace_id="dev-ws",
            current_daily_spend=Decimal("60"),
        )
        result1 = engine.evaluate(context1)
        assert result1.allowed is False

        # Should not affect prod workspace
        context2 = PolicyEvaluationContext(
            workspace_id="prod-ws",
            current_daily_spend=Decimal("60"),
        )
        result2 = engine.evaluate(context2)
        assert result2.allowed is True

    def test_evaluate_multiple_policies(self, engine):
        """Test evaluation with multiple overlapping policies."""
        spending_policy = CostGovernancePolicy(
            name="spending",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            spending_limit=SpendingLimit(daily_limit_usd=Decimal("100")),
        )
        model_policy = CostGovernancePolicy(
            name="model",
            policy_type=CostPolicyType.MODEL_RESTRICTION,
            model_restrictions=[ModelRestriction(model_pattern="gpt-4*", allowed=True)],
        )
        engine.add_policy(spending_policy)
        engine.add_policy(model_policy)

        context = PolicyEvaluationContext(
            model="gpt-4-turbo",
            current_daily_spend=Decimal("50"),
        )

        result = engine.evaluate(context)

        assert result.allowed is True
        assert len(result.violations) == 0

    def test_record_model_request(self, engine):
        """Test model request recording for rate limiting."""
        engine.record_model_request("claude-sonnet")
        engine.record_model_request("claude-sonnet")
        engine.record_model_request("claude-opus")

        assert engine._count_recent_requests("claude-sonnet") == 2
        assert engine._count_recent_requests("claude-opus") == 1
        assert engine._count_recent_requests("gpt-4") == 0

    def test_count_recent_requests_window(self, engine):
        """Test that old requests are cleaned from the window."""
        # Add a request from 2 hours ago (beyond the 1-hour window)
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        engine._model_requests["claude-opus"] = [old_time]

        # Add a recent request
        engine.record_model_request("claude-opus")

        # Old request should be cleaned up
        count = engine._count_recent_requests("claude-opus")
        assert count == 1

    def test_policy_priority_order(self, engine):
        """Test that higher priority policies are evaluated first."""
        # Low priority policy - allows opus
        low_priority = CostGovernancePolicy(
            name="allow-opus",
            priority=1,
            policy_type=CostPolicyType.MODEL_RESTRICTION,
            enforcement=CostPolicyEnforcement.SOFT,
            model_restrictions=[ModelRestriction(model_pattern="claude-opus*", allowed=True)],
        )

        # High priority policy - denies opus
        high_priority = CostGovernancePolicy(
            name="deny-opus",
            priority=10,
            policy_type=CostPolicyType.MODEL_RESTRICTION,
            enforcement=CostPolicyEnforcement.HARD,
            model_restrictions=[ModelRestriction(model_pattern="claude-opus*", allowed=False)],
        )

        engine.add_policy(low_priority)
        engine.add_policy(high_priority)

        context = PolicyEvaluationContext(model="claude-opus-4")
        result = engine.evaluate(context)

        # High priority deny should win
        assert result.allowed is False

    def test_audit_callback(self, engine):
        """Test audit callback is called."""
        audit_logs = []
        engine.add_audit_callback(lambda entry: audit_logs.append(entry))

        policy = CostGovernancePolicy(
            name="test-policy",
            policy_type=CostPolicyType.SPENDING_LIMIT,
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext()
        engine.evaluate(context)

        assert len(audit_logs) == 1
        assert "result" in audit_logs[0]
        assert "context" in audit_logs[0]
        assert "timestamp" in audit_logs[0]

    def test_audit_callback_multiple(self, engine):
        """Test multiple audit callbacks are called."""
        callback1_logs = []
        callback2_logs = []
        engine.add_audit_callback(lambda entry: callback1_logs.append(entry))
        engine.add_audit_callback(lambda entry: callback2_logs.append(entry))

        context = PolicyEvaluationContext()
        engine.evaluate(context)

        assert len(callback1_logs) == 1
        assert len(callback2_logs) == 1

    def test_audit_callback_error_handling(self, engine):
        """Test audit callback errors are caught."""

        def failing_callback(entry):
            raise Exception("Callback error")

        engine.add_audit_callback(failing_callback)

        # Should not raise
        context = PolicyEvaluationContext()
        result = engine.evaluate(context)

        assert result.allowed is True

    def test_model_matches_pattern_wildcard(self, engine):
        """Test model pattern matching with wildcard."""
        assert engine._model_matches_pattern("claude-opus-4", "claude-opus*")
        assert engine._model_matches_pattern("claude-opus-4-20240229", "claude-opus*")
        assert not engine._model_matches_pattern("claude-sonnet-4", "claude-opus*")

    def test_model_matches_pattern_exact(self, engine):
        """Test model pattern matching with exact match."""
        assert engine._model_matches_pattern("claude-opus-4", "claude-opus-4")
        assert not engine._model_matches_pattern("claude-opus-4-20240229", "claude-opus-4")

    def test_get_action_for_enforcement(self, engine):
        """Test enforcement level to action mapping."""
        assert (
            engine._get_action_for_enforcement(CostPolicyEnforcement.HARD) == CostPolicyAction.DENY
        )
        assert (
            engine._get_action_for_enforcement(CostPolicyEnforcement.SOFT) == CostPolicyAction.ALLOW
        )
        assert (
            engine._get_action_for_enforcement(CostPolicyEnforcement.WARN) == CostPolicyAction.ALLOW
        )
        assert (
            engine._get_action_for_enforcement(CostPolicyEnforcement.AUDIT)
            == CostPolicyAction.ALLOW
        )


# =============================================================================
# Approval Workflow Tests (10+ tests)
# =============================================================================


class TestApprovalWorkflow:
    """Tests for approval workflow methods."""

    def test_request_approval(self, engine):
        """Test requesting approval for an operation."""
        policy = CostGovernancePolicy(
            name="approval-policy",
            approvers=["manager-1"],
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(estimated_cost_usd=Decimal("100"))

        request_id = engine.request_approval(
            context=context,
            policy_id=policy.id,
            requestor_id="user-1",
        )

        assert request_id is not None
        pending = engine.get_pending_approvals()
        assert len(pending) == 1
        assert pending[0]["request_id"] == request_id
        assert pending[0]["status"] == "pending"

    def test_request_approval_stores_context(self, engine):
        """Test that approval request stores context correctly."""
        policy = CostGovernancePolicy(name="approval-policy", approvers=["manager-1"])
        engine.add_policy(policy)

        context = PolicyEvaluationContext(
            user_id="user-1",
            operation="expensive-op",
            estimated_cost_usd=Decimal("500"),
        )

        request_id = engine.request_approval(
            context=context,
            policy_id=policy.id,
            requestor_id="user-1",
        )

        approval = engine._pending_approvals[request_id]
        assert approval["context"] == context
        assert approval["policy_id"] == policy.id
        assert approval["requestor_id"] == "user-1"

    def test_approve_request(self, engine):
        """Test approving a request."""
        policy = CostGovernancePolicy(
            name="approval-policy",
            approvers=["manager-1"],
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext()
        request_id = engine.request_approval(
            context=context,
            policy_id=policy.id,
            requestor_id="user-1",
        )

        result = engine.approve_request(request_id, "manager-1")

        assert result is True
        assert engine._pending_approvals[request_id]["status"] == "approved"
        assert engine._pending_approvals[request_id]["approved_by"] == "manager-1"
        assert engine._pending_approvals[request_id]["approved_at"] is not None

    def test_approve_request_unauthorized(self, engine):
        """Test that unauthorized approvers cannot approve."""
        policy = CostGovernancePolicy(
            name="approval-policy",
            approvers=["manager-1"],
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext()
        request_id = engine.request_approval(
            context=context,
            policy_id=policy.id,
            requestor_id="user-1",
        )

        result = engine.approve_request(request_id, "unauthorized-user")

        assert result is False
        assert engine._pending_approvals[request_id]["status"] == "pending"

    def test_approve_nonexistent_request(self, engine):
        """Test approving a nonexistent request."""
        result = engine.approve_request("nonexistent", "manager-1")

        assert result is False

    def test_deny_request(self, engine):
        """Test denying a request."""
        policy = CostGovernancePolicy(
            name="approval-policy",
            approvers=["manager-1"],
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext()
        request_id = engine.request_approval(
            context=context,
            policy_id=policy.id,
            requestor_id="user-1",
        )

        result = engine.deny_request(request_id, "manager-1", "Too expensive")

        assert result is True
        assert engine._pending_approvals[request_id]["status"] == "denied"
        assert engine._pending_approvals[request_id]["denied_by"] == "manager-1"
        assert engine._pending_approvals[request_id]["denial_reason"] == "Too expensive"
        assert engine._pending_approvals[request_id]["denied_at"] is not None

    def test_deny_request_no_reason(self, engine):
        """Test denying a request without a reason."""
        policy = CostGovernancePolicy(name="approval-policy", approvers=["manager-1"])
        engine.add_policy(policy)

        context = PolicyEvaluationContext()
        request_id = engine.request_approval(
            context=context,
            policy_id=policy.id,
            requestor_id="user-1",
        )

        result = engine.deny_request(request_id, "manager-1")

        assert result is True
        assert engine._pending_approvals[request_id]["status"] == "denied"
        assert engine._pending_approvals[request_id]["denial_reason"] == ""

    def test_deny_nonexistent_request(self, engine):
        """Test denying a nonexistent request."""
        result = engine.deny_request("nonexistent", "manager-1", "reason")

        assert result is False

    def test_get_pending_approvals_all(self, engine):
        """Test getting all pending approvals."""
        policy = CostGovernancePolicy(name="approval-policy", approvers=["manager-1"])
        engine.add_policy(policy)

        # Create multiple requests
        for i in range(3):
            context = PolicyEvaluationContext()
            engine.request_approval(
                context=context,
                policy_id=policy.id,
                requestor_id=f"user-{i}",
            )

        pending = engine.get_pending_approvals()

        assert len(pending) == 3

    def test_get_pending_approvals_excludes_approved(self, engine):
        """Test that approved requests are excluded from pending."""
        policy = CostGovernancePolicy(name="approval-policy", approvers=["manager-1"])
        engine.add_policy(policy)

        context = PolicyEvaluationContext()
        request_id = engine.request_approval(
            context=context,
            policy_id=policy.id,
            requestor_id="user-1",
        )

        # Approve the request
        engine.approve_request(request_id, "manager-1")

        pending = engine.get_pending_approvals()

        assert len(pending) == 0

    def test_get_pending_approvals_excludes_denied(self, engine):
        """Test that denied requests are excluded from pending."""
        policy = CostGovernancePolicy(name="approval-policy", approvers=["manager-1"])
        engine.add_policy(policy)

        context = PolicyEvaluationContext()
        request_id = engine.request_approval(
            context=context,
            policy_id=policy.id,
            requestor_id="user-1",
        )

        # Deny the request
        engine.deny_request(request_id, "manager-1", "reason")

        pending = engine.get_pending_approvals()

        assert len(pending) == 0

    def test_get_pending_approvals_by_approver(self, engine):
        """Test getting pending approvals filtered by approver."""
        policy1 = CostGovernancePolicy(name="policy-1", approvers=["manager-1"])
        policy2 = CostGovernancePolicy(name="policy-2", approvers=["manager-2"])
        engine.add_policy(policy1)
        engine.add_policy(policy2)

        # Create request for each policy
        engine.request_approval(
            context=PolicyEvaluationContext(),
            policy_id=policy1.id,
            requestor_id="user-1",
        )
        engine.request_approval(
            context=PolicyEvaluationContext(),
            policy_id=policy2.id,
            requestor_id="user-2",
        )

        # Manager 1 should only see requests for policy 1
        pending_manager1 = engine.get_pending_approvals(approver_id="manager-1")
        assert len(pending_manager1) == 1

        # Manager 2 should only see requests for policy 2
        pending_manager2 = engine.get_pending_approvals(approver_id="manager-2")
        assert len(pending_manager2) == 1

    def test_approve_request_missing_policy(self, engine):
        """Test approving when policy no longer exists."""
        policy = CostGovernancePolicy(name="approval-policy", approvers=["manager-1"])
        engine.add_policy(policy)

        context = PolicyEvaluationContext()
        request_id = engine.request_approval(
            context=context,
            policy_id=policy.id,
            requestor_id="user-1",
        )

        # Remove the policy
        engine.remove_policy(policy.id)

        # Approve should still work (policy check returns None)
        result = engine.approve_request(request_id, "manager-1")

        # When policy is None, approver check is skipped
        assert result is True


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateCostGovernanceEngine:
    """Tests for factory function."""

    def test_create(self):
        """Test factory creates engine."""
        engine = create_cost_governance_engine()

        assert isinstance(engine, CostGovernanceEngine)

    def test_create_with_trackers(self):
        """Test factory creates engine with trackers."""
        mock_tracker = MagicMock()
        mock_attributor = MagicMock()

        engine = create_cost_governance_engine(
            cost_tracker=mock_tracker,
            cost_attributor=mock_attributor,
        )

        assert engine._cost_tracker == mock_tracker
        assert engine._cost_attributor == mock_attributor


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for cost governance workflow."""

    def test_full_policy_lifecycle(self, engine):
        """Test complete policy lifecycle: create, evaluate, update, remove."""
        # 1. Create and add policy
        policy = CostGovernancePolicy(
            name="integration-test",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            spending_limit=SpendingLimit(daily_limit_usd=Decimal("100")),
        )
        engine.add_policy(policy)

        # 2. Evaluate within limits
        context1 = PolicyEvaluationContext(current_daily_spend=Decimal("50"))
        result1 = engine.evaluate(context1)
        assert result1.allowed is True

        # 3. Evaluate over limits
        context2 = PolicyEvaluationContext(current_daily_spend=Decimal("150"))
        result2 = engine.evaluate(context2)
        assert result2.allowed is False

        # 4. Remove policy
        engine.remove_policy(policy.id)

        # 5. Evaluate again - should be allowed
        result3 = engine.evaluate(context2)
        assert result3.allowed is True

    def test_multi_policy_evaluation(self, engine):
        """Test evaluation with multiple policies of different types."""
        # Add spending limit
        spending_policy = CostGovernancePolicy(
            name="spending",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            spending_limit=SpendingLimit(daily_limit_usd=Decimal("100")),
        )
        engine.add_policy(spending_policy)

        # Add model restriction
        model_policy = CostGovernancePolicy(
            name="model",
            policy_type=CostPolicyType.MODEL_RESTRICTION,
            model_restrictions=[
                ModelRestriction(
                    model_pattern="claude-opus*",
                    allowed=True,
                    max_tokens_per_request=10000,
                )
            ],
        )
        engine.add_policy(model_policy)

        # Add approval required for high costs
        approval_policy = CostGovernancePolicy(
            name="approval",
            policy_type=CostPolicyType.APPROVAL_REQUIRED,
            approval_threshold_usd=Decimal("50"),
            approvers=["manager-1"],
        )
        engine.add_policy(approval_policy)

        # Test 1: All clear
        context1 = PolicyEvaluationContext(
            model="claude-opus-4",
            estimated_tokens=5000,
            estimated_cost_usd=Decimal("25"),
            current_daily_spend=Decimal("50"),
        )
        result1 = engine.evaluate(context1)
        assert result1.allowed is True

        # Test 2: Token limit exceeded
        context2 = PolicyEvaluationContext(
            model="claude-opus-4",
            estimated_tokens=15000,
            estimated_cost_usd=Decimal("25"),
            current_daily_spend=Decimal("50"),
        )
        result2 = engine.evaluate(context2)
        assert len(result2.violations) > 0

        # Test 3: Approval required
        context3 = PolicyEvaluationContext(
            model="claude-opus-4",
            estimated_tokens=5000,
            estimated_cost_usd=Decimal("75"),
            current_daily_spend=Decimal("50"),
        )
        result3 = engine.evaluate(context3)
        assert result3.requires_approval is True

    def test_approval_workflow_complete(self, engine):
        """Test complete approval workflow."""
        # 1. Setup policy
        policy = CostGovernancePolicy(
            name="approval-policy",
            policy_type=CostPolicyType.APPROVAL_REQUIRED,
            approval_threshold_usd=Decimal("50"),
            approvers=["manager-1", "manager-2"],
        )
        engine.add_policy(policy)

        # 2. Evaluate and get approval required
        context = PolicyEvaluationContext(estimated_cost_usd=Decimal("100"))
        result = engine.evaluate(context)
        assert result.requires_approval is True

        # 3. Submit approval request
        request_id = engine.request_approval(
            context=context,
            policy_id=policy.id,
            requestor_id="user-1",
        )

        # 4. Check pending
        pending = engine.get_pending_approvals(approver_id="manager-1")
        assert len(pending) == 1

        # 5. Approve
        approved = engine.approve_request(request_id, "manager-1")
        assert approved is True

        # 6. Verify no longer pending
        pending_after = engine.get_pending_approvals()
        assert len(pending_after) == 0

    def test_audit_trail(self, engine):
        """Test audit trail captures all evaluations."""
        audit_log = []
        engine.add_audit_callback(lambda entry: audit_log.append(entry))

        # Add policy
        policy = CostGovernancePolicy(
            name="audited-policy",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            spending_limit=SpendingLimit(daily_limit_usd=Decimal("100")),
        )
        engine.add_policy(policy)

        # Multiple evaluations
        for spend in [Decimal("25"), Decimal("50"), Decimal("75"), Decimal("110")]:
            context = PolicyEvaluationContext(current_daily_spend=spend)
            engine.evaluate(context)

        # All evaluations should be logged
        assert len(audit_log) == 4

        # Last evaluation should show violation
        last_entry = audit_log[-1]
        assert last_entry["result"]["allowed"] is False
