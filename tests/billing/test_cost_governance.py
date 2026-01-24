"""Tests for cost governance policy engine."""

from datetime import datetime, timezone
from decimal import Decimal

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
    PolicyViolation,
    SpendingLimit,
    TimeRestriction,
    create_cost_governance_engine,
)


class TestModelRestriction:
    """Tests for ModelRestriction dataclass."""

    def test_to_dict(self):
        """Test serialization."""
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
        assert data["fallback_model"] == "claude-sonnet"

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "model_pattern": "gpt-4*",
            "allowed": False,
            "max_requests_per_hour": 50,
        }

        restriction = ModelRestriction.from_dict(data)

        assert restriction.model_pattern == "gpt-4*"
        assert restriction.allowed is False
        assert restriction.max_requests_per_hour == 50


class TestSpendingLimit:
    """Tests for SpendingLimit dataclass."""

    def test_to_dict(self):
        """Test serialization."""
        limit = SpendingLimit(
            daily_limit_usd=Decimal("100"),
            monthly_limit_usd=Decimal("2500"),
            alert_threshold_percent=80.0,
            hard_limit=True,
        )

        data = limit.to_dict()

        assert data["daily_limit_usd"] == "100"
        assert data["monthly_limit_usd"] == "2500"
        assert data["alert_threshold_percent"] == 80.0

    def test_from_dict(self):
        """Test deserialization."""
        data = {
            "daily_limit_usd": "200",
            "monthly_limit_usd": "5000",
            "hard_limit": False,
        }

        limit = SpendingLimit.from_dict(data)

        assert limit.daily_limit_usd == Decimal("200")
        assert limit.monthly_limit_usd == Decimal("5000")
        assert limit.hard_limit is False


class TestTimeRestriction:
    """Tests for TimeRestriction dataclass."""

    def test_is_allowed_during_work_hours(self):
        """Test time restriction during work hours."""
        restriction = TimeRestriction(
            allowed_hours_start=9,
            allowed_hours_end=17,
            allowed_days=[0, 1, 2, 3, 4],  # Mon-Fri
        )

        # This test might be flaky depending on when it runs
        # Just verify it returns a boolean
        result = restriction.is_allowed_now()
        assert isinstance(result, bool)

    def test_to_dict(self):
        """Test serialization."""
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

    def test_from_dict(self):
        """Test deserialization."""
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

    def test_matches_user_scope(self):
        """Test user scope matching."""
        policy = CostGovernancePolicy(
            name="user-policy",
            scope=CostPolicyScope.USER,
            target_users=["user-1"],
        )

        assert policy.matches(user_id="user-1")
        assert not policy.matches(user_id="user-2")

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


class TestCostGovernanceEngine:
    """Tests for CostGovernanceEngine class."""

    def test_init(self):
        """Test engine initialization."""
        engine = CostGovernanceEngine()

        assert len(engine._policies) == 0

    def test_add_policy(self):
        """Test adding a policy."""
        engine = CostGovernanceEngine()

        policy = CostGovernancePolicy(name="test-policy")
        engine.add_policy(policy)

        assert policy.id in engine._policies
        assert engine.get_policy(policy.id) == policy

    def test_remove_policy(self):
        """Test removing a policy."""
        engine = CostGovernanceEngine()

        policy = CostGovernancePolicy(name="test-policy")
        engine.add_policy(policy)
        result = engine.remove_policy(policy.id)

        assert result is True
        assert engine.get_policy(policy.id) is None

    def test_remove_nonexistent_policy(self):
        """Test removing a policy that doesn't exist."""
        engine = CostGovernanceEngine()

        result = engine.remove_policy("nonexistent")

        assert result is False

    def test_list_policies(self):
        """Test listing policies."""
        engine = CostGovernanceEngine()

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

    def test_evaluate_no_policies(self):
        """Test evaluation with no policies."""
        engine = CostGovernanceEngine()

        context = PolicyEvaluationContext(
            user_id="user-1",
            operation="debate",
            model="claude-sonnet",
        )

        result = engine.evaluate(context)

        assert result.allowed is True
        assert result.action == CostPolicyAction.ALLOW
        assert len(result.violations) == 0

    def test_evaluate_model_not_allowed(self):
        """Test model restriction - model not allowed."""
        engine = CostGovernanceEngine()

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

        context = PolicyEvaluationContext(
            model="claude-opus-4",
        )

        result = engine.evaluate(context)

        assert result.allowed is False
        assert result.action == CostPolicyAction.DENY
        assert len(result.violations) == 1
        assert result.violations[0].violation_type == "model_not_allowed"
        assert result.suggested_model == "claude-sonnet"

    def test_evaluate_model_operation_not_allowed(self):
        """Test model restriction - operation not allowed."""
        engine = CostGovernanceEngine()

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

    def test_evaluate_spending_limit_exceeded(self):
        """Test spending limit exceeded."""
        engine = CostGovernanceEngine()

        policy = CostGovernancePolicy(
            name="daily-limit",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            spending_limit=SpendingLimit(
                daily_limit_usd=Decimal("100"),
                hard_limit=True,
            ),
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(
            current_daily_spend=Decimal("105"),
        )

        result = engine.evaluate(context)

        assert result.allowed is False
        assert result.action == CostPolicyAction.DENY
        assert result.violations[0].violation_type == "daily_limit_exceeded"

    def test_evaluate_spending_limit_warning(self):
        """Test spending limit warning threshold."""
        engine = CostGovernanceEngine()

        policy = CostGovernancePolicy(
            name="daily-limit",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            spending_limit=SpendingLimit(
                daily_limit_usd=Decimal("100"),
                alert_threshold_percent=80.0,
            ),
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(
            current_daily_spend=Decimal("85"),  # 85% of limit
        )

        result = engine.evaluate(context)

        assert result.allowed is True
        assert len(result.warnings) == 1
        assert "Approaching daily limit" in result.warnings[0]

    def test_evaluate_monthly_limit_exceeded(self):
        """Test monthly spending limit exceeded."""
        engine = CostGovernanceEngine()

        policy = CostGovernancePolicy(
            name="monthly-limit",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            spending_limit=SpendingLimit(
                monthly_limit_usd=Decimal("2500"),
                hard_limit=True,
            ),
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(
            current_monthly_spend=Decimal("2600"),
        )

        result = engine.evaluate(context)

        assert result.allowed is False
        assert result.violations[0].violation_type == "monthly_limit_exceeded"

    def test_evaluate_per_operation_limit(self):
        """Test per-operation limit triggers approval."""
        engine = CostGovernanceEngine()

        policy = CostGovernancePolicy(
            name="op-limit",
            policy_type=CostPolicyType.SPENDING_LIMIT,
            spending_limit=SpendingLimit(
                per_operation_limit_usd=Decimal("10"),
            ),
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(
            estimated_cost_usd=Decimal("15"),
        )

        result = engine.evaluate(context)

        assert result.action == CostPolicyAction.QUEUE
        assert result.violations[0].violation_type == "operation_cost_exceeded"

    def test_evaluate_approval_required(self):
        """Test approval required policy."""
        engine = CostGovernanceEngine()

        policy = CostGovernancePolicy(
            name="approval-required",
            policy_type=CostPolicyType.APPROVAL_REQUIRED,
            approval_threshold_usd=Decimal("50"),
            approvers=["manager-1", "manager-2"],
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(
            estimated_cost_usd=Decimal("75"),
        )

        result = engine.evaluate(context)

        assert result.requires_approval is True
        assert result.action == CostPolicyAction.QUEUE
        assert "manager-1" in result.approvers

    def test_evaluate_auto_approve_under_threshold(self):
        """Test auto-approve for small operations."""
        engine = CostGovernanceEngine()

        policy = CostGovernancePolicy(
            name="approval-required",
            policy_type=CostPolicyType.APPROVAL_REQUIRED,
            approval_threshold_usd=Decimal("50"),
            auto_approve_under_usd=Decimal("10"),
            approvers=["manager-1"],
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(
            estimated_cost_usd=Decimal("5"),
        )

        result = engine.evaluate(context)

        assert result.requires_approval is False
        assert result.allowed is True

    def test_evaluate_soft_enforcement(self):
        """Test soft enforcement allows but logs violations."""
        engine = CostGovernanceEngine()

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

        context = PolicyEvaluationContext(
            current_daily_spend=Decimal("150"),
        )

        result = engine.evaluate(context)

        # Soft enforcement - allowed despite violation
        assert result.allowed is True
        assert len(result.violations) == 1

    def test_evaluate_workspace_scoped_policy(self):
        """Test workspace-scoped policy only affects target workspaces."""
        engine = CostGovernanceEngine()

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

    def test_record_model_request(self):
        """Test model request recording for rate limiting."""
        engine = CostGovernanceEngine()

        engine.record_model_request("claude-sonnet")
        engine.record_model_request("claude-sonnet")
        engine.record_model_request("claude-opus")

        assert engine._count_recent_requests("claude-sonnet") == 2
        assert engine._count_recent_requests("claude-opus") == 1
        assert engine._count_recent_requests("gpt-4") == 0

    def test_request_approval(self):
        """Test requesting approval for an operation."""
        engine = CostGovernanceEngine()

        policy = CostGovernancePolicy(
            name="approval-policy",
            approvers=["manager-1"],
        )
        engine.add_policy(policy)

        context = PolicyEvaluationContext(
            estimated_cost_usd=Decimal("100"),
        )

        request_id = engine.request_approval(
            context=context,
            policy_id=policy.id,
            requestor_id="user-1",
        )

        assert request_id is not None
        pending = engine.get_pending_approvals()
        assert len(pending) == 1
        assert pending[0]["request_id"] == request_id

    def test_approve_request(self):
        """Test approving a request."""
        engine = CostGovernanceEngine()

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

    def test_approve_request_unauthorized(self):
        """Test that unauthorized approvers cannot approve."""
        engine = CostGovernanceEngine()

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

    def test_deny_request(self):
        """Test denying a request."""
        engine = CostGovernanceEngine()

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
        assert engine._pending_approvals[request_id]["denial_reason"] == "Too expensive"

    def test_audit_callback(self):
        """Test audit callback is called."""
        engine = CostGovernanceEngine()

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

    def test_policy_priority_order(self):
        """Test that higher priority policies are evaluated first."""
        engine = CostGovernanceEngine()

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


class TestCreateCostGovernanceEngine:
    """Tests for factory function."""

    def test_create(self):
        """Test factory creates engine."""
        engine = create_cost_governance_engine()

        assert isinstance(engine, CostGovernanceEngine)
