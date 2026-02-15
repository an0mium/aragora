"""Tests for Agent Fabric data models."""

from __future__ import annotations

from datetime import datetime, timezone

from aragora.fabric.models import (
    AgentConfig,
    AgentHandle,
    AgentInfo,
    ApprovalRequest,
    BudgetConfig,
    BudgetStatus,
    HealthStatus,
    IsolationConfig,
    Policy,
    PolicyContext,
    PolicyDecision,
    PolicyEffect,
    PolicyRule,
    Priority,
    ResourceUsage,
    Task,
    TaskHandle,
    TaskStatus,
    Usage,
    UsageReport,
)


class TestPriority:
    def test_ordering(self):
        assert Priority.CRITICAL.value < Priority.HIGH.value
        assert Priority.HIGH.value < Priority.NORMAL.value
        assert Priority.NORMAL.value < Priority.LOW.value

    def test_values(self):
        assert Priority.CRITICAL.value == 0
        assert Priority.LOW.value == 3


class TestTaskStatus:
    def test_all_states(self):
        for status in TaskStatus:
            assert isinstance(status.value, str)

    def test_expected_states(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestTask:
    def test_create_basic(self):
        task = Task(id="t1", type="debate", payload={"topic": "test"})
        assert task.id == "t1"
        assert task.type == "debate"
        assert task.payload == {"topic": "test"}

    def test_defaults(self):
        task = Task(id="t1", type="test", payload={})
        assert isinstance(task.created_at, datetime)
        assert task.timeout_seconds is None
        assert task.metadata == {}
        assert task.depends_on == []

    def test_with_dependencies(self):
        task = Task(id="t2", type="test", payload={}, depends_on=["t1"])
        assert task.depends_on == ["t1"]


class TestTaskHandle:
    def test_create(self):
        handle = TaskHandle(
            task_id="t1",
            agent_id="a1",
            status=TaskStatus.RUNNING,
            scheduled_at=datetime.now(timezone.utc),
        )
        assert handle.task_id == "t1"
        assert handle.agent_id == "a1"
        assert handle.result is None
        assert handle.error is None


class TestAgentConfig:
    def test_create_minimal(self):
        config = AgentConfig(id="a1", model="claude-3-opus")
        assert config.id == "a1"
        assert config.model == "claude-3-opus"
        assert config.tools == []
        assert config.pool_id is None
        assert config.max_concurrent_tasks == 1

    def test_create_full(self):
        config = AgentConfig(
            id="a1",
            model="gpt-4",
            tools=["shell", "browser"],
            isolation=IsolationConfig(level="container", memory_mb=1024),
            budget=BudgetConfig(max_tokens_per_day=100_000),
            policies=["default"],
            pool_id="debate-pool",
            max_concurrent_tasks=3,
        )
        assert config.tools == ["shell", "browser"]
        assert config.isolation.level == "container"
        assert config.budget.max_tokens_per_day == 100_000
        assert config.pool_id == "debate-pool"


class TestIsolationConfig:
    def test_defaults(self):
        config = IsolationConfig()
        assert config.level == "process"
        assert config.memory_mb == 512
        assert config.cpu_cores == 1.0
        assert config.network_egress == []


class TestBudgetConfig:
    def test_defaults(self):
        config = BudgetConfig()
        assert config.max_tokens_per_day is None
        assert config.hard_limit is True
        assert config.alert_threshold_percent == 80.0

    def test_custom(self):
        config = BudgetConfig(
            max_tokens_per_day=50_000,
            max_cost_per_day_usd=10.0,
            hard_limit=False,
        )
        assert config.max_tokens_per_day == 50_000
        assert config.max_cost_per_day_usd == 10.0
        assert config.hard_limit is False


class TestAgentHandle:
    def test_create(self):
        config = AgentConfig(id="a1", model="claude-3-opus")
        handle = AgentHandle(
            agent_id="a1",
            config=config,
            spawned_at=datetime.now(timezone.utc),
        )
        assert handle.status == HealthStatus.HEALTHY
        assert handle.tasks_completed == 0
        assert handle.tasks_failed == 0


class TestAgentInfo:
    def test_create(self):
        info = AgentInfo(
            agent_id="a1",
            model="claude-3-opus",
            status=HealthStatus.HEALTHY,
            spawned_at=datetime.now(timezone.utc),
            last_heartbeat=datetime.now(timezone.utc),
            tasks_pending=2,
            tasks_running=1,
            tasks_completed=10,
            tasks_failed=0,
            budget_usage_percent=45.0,
        )
        assert info.agent_id == "a1"
        assert info.budget_usage_percent == 45.0


class TestPolicyModels:
    def test_policy_rule(self):
        rule = PolicyRule(
            action_pattern="tool:shell:*",
            effect=PolicyEffect.DENY,
            description="Block shell access",
        )
        assert rule.action_pattern == "tool:shell:*"
        assert rule.effect == PolicyEffect.DENY

    def test_policy(self):
        policy = Policy(
            id="p1",
            name="Security Policy",
            rules=[
                PolicyRule(action_pattern="*", effect=PolicyEffect.ALLOW),
            ],
            priority=100,
        )
        assert policy.id == "p1"
        assert policy.enabled is True
        assert len(policy.rules) == 1

    def test_policy_context(self):
        ctx = PolicyContext(
            agent_id="a1",
            user_id="u1",
            tenant_id="t1",
            action="tool:browser:navigate",
            resource="https://example.com",
        )
        assert ctx.agent_id == "a1"
        assert ctx.action == "tool:browser:navigate"

    def test_policy_decision_allowed(self):
        decision = PolicyDecision(allowed=True, effect=PolicyEffect.ALLOW)
        assert decision.allowed
        assert not decision.requires_approval

    def test_policy_decision_denied(self):
        decision = PolicyDecision(allowed=False, effect=PolicyEffect.DENY, reason="blocked")
        assert not decision.allowed

    def test_policy_decision_approval(self):
        decision = PolicyDecision(
            allowed=True,
            effect=PolicyEffect.REQUIRE_APPROVAL,
            requires_approval=True,
            approvers=["admin"],
        )
        assert decision.requires_approval
        assert decision.approvers == ["admin"]


class TestUsageModels:
    def test_usage(self):
        usage = Usage(
            agent_id="a1",
            tokens_input=500,
            tokens_output=200,
            cost_usd=0.01,
            model="claude-3-opus",
        )
        assert usage.agent_id == "a1"
        assert usage.tokens_input == 500

    def test_budget_status(self):
        status = BudgetStatus(
            entity_id="a1",
            entity_type="agent",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            usage_percent=75.0,
            over_limit=False,
        )
        assert not status.over_limit

    def test_usage_report(self):
        report = UsageReport(
            entity_id="a1",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            total_tokens=10000,
            total_cost_usd=0.50,
        )
        assert report.total_tokens == 10000

    def test_resource_usage(self):
        ru = ResourceUsage(agent_id="a1", memory_mb=256.0, cpu_percent=50.0)
        assert ru.memory_mb == 256.0


class TestApprovalModels:
    def test_approval_request(self):
        req = ApprovalRequest(
            id="req1",
            action="deploy",
            context=PolicyContext(agent_id="a1"),
            requested_at=datetime.now(timezone.utc),
            requested_by="user1",
            approvers=["admin1"],
        )
        assert req.status == "pending"
        assert req.approved_by is None
