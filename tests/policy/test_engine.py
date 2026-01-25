"""Tests for policy engine."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from aragora.policy.engine import (
    DEFAULT_POLICIES,
    Policy,
    PolicyDecision,
    PolicyEngine,
    PolicyResult,
    PolicyViolation,
    create_default_engine,
)
from aragora.policy.risk import BlastRadius, RiskBudget, RiskLevel
from aragora.policy.tools import Tool, ToolCapability, ToolCategory, ToolRegistry


@pytest.fixture
def tool_registry():
    """Create a mock tool registry."""
    registry = ToolRegistry()

    # Add a file writer tool
    file_writer = Tool(
        name="file_writer",
        description="File operations",
        category=ToolCategory.WRITE,
        capabilities=[
            ToolCapability(
                name="write_file",
                description="Write to file",
                risk_level=RiskLevel.HIGH,
                blast_radius=BlastRadius.LOCAL,
            ),
            ToolCapability(
                name="delete_file",
                description="Delete file",
                risk_level=RiskLevel.CRITICAL,
                blast_radius=BlastRadius.SHARED,
                requires_human_approval=True,
            ),
            ToolCapability(
                name="read_file",
                description="Read file",
                risk_level=RiskLevel.LOW,
                blast_radius=BlastRadius.DRAFT,
            ),
        ],
    )
    registry.register(file_writer)

    # Add a git tool
    git_tool = Tool(
        name="git",
        description="Git operations",
        category=ToolCategory.WRITE,
        capabilities=[
            ToolCapability(
                name="git_commit",
                description="Git commit",
                risk_level=RiskLevel.MEDIUM,
                blast_radius=BlastRadius.DRAFT,
            ),
            ToolCapability(
                name="git_push",
                description="Git push",
                risk_level=RiskLevel.CRITICAL,
                blast_radius=BlastRadius.PRODUCTION,
                requires_human_approval=True,
            ),
        ],
    )
    registry.register(git_tool)

    # Add code executor
    executor = Tool(
        name="code_executor",
        description="Execute code",
        category=ToolCategory.EXECUTE,
        capabilities=[
            ToolCapability(
                name="run_shell",
                description="Run shell command",
                risk_level=RiskLevel.CRITICAL,
                blast_radius=BlastRadius.PRODUCTION,
            ),
        ],
    )
    registry.register(executor)

    return registry


@pytest.fixture
def engine(tool_registry):
    """Create a policy engine with test registry."""
    return PolicyEngine(tool_registry=tool_registry)


class TestPolicyDecision:
    """Tests for PolicyDecision enum."""

    def test_all_decisions(self):
        """Test all decision types exist."""
        assert PolicyDecision.ALLOW.value == "allow"
        assert PolicyDecision.DENY.value == "deny"
        assert PolicyDecision.ESCALATE.value == "escalate"
        assert PolicyDecision.BUDGET_EXCEEDED.value == "budget_exceeded"


class TestPolicyResult:
    """Tests for PolicyResult dataclass."""

    def test_create_allow_result(self):
        """Test creating an allow result."""
        result = PolicyResult(
            decision=PolicyDecision.ALLOW,
            allowed=True,
            reason="Action permitted",
            agent="claude",
            tool="file_writer",
            capability="read_file",
        )

        assert result.decision == PolicyDecision.ALLOW
        assert result.allowed is True
        assert result.reason == "Action permitted"
        assert result.requires_human_approval is False

    def test_create_deny_result(self):
        """Test creating a deny result."""
        result = PolicyResult(
            decision=PolicyDecision.DENY,
            allowed=False,
            reason="Denied by policy",
            agent="claude",
            tool="file_writer",
            capability="write_file",
        )

        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False

    def test_create_escalate_result(self):
        """Test creating an escalate result."""
        result = PolicyResult(
            decision=PolicyDecision.ESCALATE,
            allowed=False,
            requires_human_approval=True,
            reason="Requires approval",
        )

        assert result.decision == PolicyDecision.ESCALATE
        assert result.requires_human_approval is True


class TestPolicyViolation:
    """Tests for PolicyViolation exception."""

    def test_exception_contains_result(self):
        """Test that exception contains the result."""
        result = PolicyResult(
            decision=PolicyDecision.DENY,
            allowed=False,
            reason="Test denial",
        )

        exc = PolicyViolation(result)

        assert exc.result == result
        assert "Test denial" in str(exc)


class TestPolicy:
    """Tests for Policy class."""

    def test_policy_matches_all(self):
        """Test policy that matches all actions."""
        policy = Policy(
            name="allow_all",
            description="Allow all actions",
            allow=True,
        )

        assert policy.matches("claude", "file_writer", "write_file", {})
        assert policy.matches("gpt", "git", "commit", {})

    def test_policy_matches_specific_agent(self):
        """Test policy that matches specific agent."""
        policy = Policy(
            name="claude_only",
            description="Only applies to Claude",
            agents=["claude"],
            allow=True,
        )

        assert policy.matches("claude", "file_writer", "write_file", {})
        assert not policy.matches("gpt", "file_writer", "write_file", {})

    def test_policy_matches_specific_tool(self):
        """Test policy that matches specific tool."""
        policy = Policy(
            name="file_tool_only",
            description="Only applies to file_writer",
            tools=["file_writer"],
            allow=True,
        )

        assert policy.matches("claude", "file_writer", "write_file", {})
        assert not policy.matches("claude", "git", "commit", {})

    def test_policy_matches_specific_capability(self):
        """Test policy that matches specific capability."""
        policy = Policy(
            name="write_only",
            description="Only applies to write",
            capabilities=["write_file"],
            allow=True,
        )

        assert policy.matches("claude", "file_writer", "write_file", {})
        assert not policy.matches("claude", "file_writer", "read_file", {})

    def test_policy_disabled(self):
        """Test disabled policy doesn't match."""
        policy = Policy(
            name="disabled",
            description="Disabled policy",
            enabled=False,
        )

        assert not policy.matches("claude", "file_writer", "write_file", {})

    def test_policy_with_condition(self):
        """Test policy with condition."""
        policy = Policy(
            name="protect_core",
            description="Protect core files",
            conditions=["'core.py' in file_path"],
        )

        # Matches when condition is true
        assert policy.matches("claude", "file_writer", "write_file", {"file_path": "src/core.py"})

        # Doesn't match when condition is false
        assert not policy.matches(
            "claude", "file_writer", "write_file", {"file_path": "src/utils.py"}
        )

    def test_policy_with_multiple_conditions(self):
        """Test policy with multiple conditions (AND)."""
        policy = Policy(
            name="complex",
            description="Multiple conditions",
            conditions=["size > 100", "'production' in env"],
        )

        # Both conditions must be true
        assert policy.matches("claude", "tool", "cap", {"size": 200, "env": "production"})
        assert not policy.matches("claude", "tool", "cap", {"size": 50, "env": "production"})
        assert not policy.matches("claude", "tool", "cap", {"size": 200, "env": "development"})

    def test_policy_condition_comparison_operators(self):
        """Test policy conditions with various comparison operators."""
        policy_eq = Policy(name="eq", description="Equals", conditions=["status == 'active'"])
        policy_neq = Policy(
            name="neq", description="Not equals", conditions=["status != 'deleted'"]
        )
        policy_lt = Policy(name="lt", description="Less than", conditions=["count < 10"])
        policy_gt = Policy(name="gt", description="Greater than", conditions=["count > 5"])
        policy_in = Policy(name="in", description="In list", conditions=["env in ['dev', 'test']"])

        assert policy_eq.matches("a", "t", "c", {"status": "active"})
        assert not policy_eq.matches("a", "t", "c", {"status": "inactive"})

        assert policy_neq.matches("a", "t", "c", {"status": "active"})
        assert not policy_neq.matches("a", "t", "c", {"status": "deleted"})

        assert policy_lt.matches("a", "t", "c", {"count": 5})
        assert not policy_lt.matches("a", "t", "c", {"count": 15})

        assert policy_gt.matches("a", "t", "c", {"count": 10})
        assert not policy_gt.matches("a", "t", "c", {"count": 3})

        assert policy_in.matches("a", "t", "c", {"env": "dev"})
        assert not policy_in.matches("a", "t", "c", {"env": "prod"})

    def test_policy_condition_boolean_operators(self):
        """Test policy conditions with AND/OR."""
        policy_and = Policy(
            name="and",
            description="Both conditions",
            conditions=["x > 0 and y > 0"],
        )
        policy_or = Policy(
            name="or",
            description="Either condition",
            conditions=["x > 0 or y > 0"],
        )
        policy_not = Policy(
            name="not",
            description="Negation",
            conditions=["not is_disabled"],
        )

        assert policy_and.matches("a", "t", "c", {"x": 1, "y": 1})
        assert not policy_and.matches("a", "t", "c", {"x": 1, "y": -1})

        assert policy_or.matches("a", "t", "c", {"x": 1, "y": -1})
        assert not policy_or.matches("a", "t", "c", {"x": -1, "y": -1})

        assert policy_not.matches("a", "t", "c", {"is_disabled": False})
        assert not policy_not.matches("a", "t", "c", {"is_disabled": True})

    def test_policy_condition_blocks_dangerous_expressions(self):
        """Test that policy conditions block dangerous expressions."""
        # Attribute access is blocked
        policy_attr = Policy(
            name="attr",
            description="Tries attribute access",
            conditions=["obj.__class__"],
        )
        # Should not match because condition evaluation fails
        assert not policy_attr.matches("a", "t", "c", {"obj": object()})

        # Function calls are blocked
        policy_call = Policy(
            name="call",
            description="Tries function call",
            conditions=["eval('1+1')"],
        )
        assert not policy_call.matches("a", "t", "c", {})


class TestPolicyEngine:
    """Tests for PolicyEngine class."""

    def test_add_policy(self, engine):
        """Test adding a policy."""
        policy = Policy(name="test", description="Test policy")

        engine.add_policy(policy)

        assert len(engine.policies) == 1
        assert engine.policies[0].name == "test"

    def test_remove_policy(self, engine):
        """Test removing a policy."""
        policy = Policy(name="test", description="Test policy")
        engine.add_policy(policy)

        removed = engine.remove_policy("test")

        assert removed is True
        assert len(engine.policies) == 0

    def test_remove_nonexistent_policy(self, engine):
        """Test removing a policy that doesn't exist."""
        removed = engine.remove_policy("nonexistent")
        assert removed is False

    def test_policy_priority(self, engine):
        """Test that policies are sorted by priority."""
        low_priority = Policy(name="low", description="Low priority", priority=1)
        high_priority = Policy(name="high", description="High priority", priority=100)
        medium_priority = Policy(name="medium", description="Medium priority", priority=50)

        engine.add_policy(low_priority)
        engine.add_policy(high_priority)
        engine.add_policy(medium_priority)

        # Should be sorted descending by priority
        assert engine.policies[0].name == "high"
        assert engine.policies[1].name == "medium"
        assert engine.policies[2].name == "low"

    def test_check_action_unknown_tool(self, engine):
        """Test checking action with unknown tool."""
        result = engine.check_action(
            agent="claude",
            tool="unknown_tool",
            capability="something",
        )

        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False
        assert "Unknown tool" in result.reason

    def test_check_action_unknown_capability(self, engine):
        """Test checking action with unknown capability."""
        result = engine.check_action(
            agent="claude",
            tool="file_writer",
            capability="unknown_capability",
        )

        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False
        assert "does not have capability" in result.reason

    def test_check_action_allowed(self, engine):
        """Test checking allowed action."""
        result = engine.check_action(
            agent="claude",
            tool="file_writer",
            capability="read_file",
            context={"file_path": "src/utils.py"},
        )

        assert result.decision == PolicyDecision.ALLOW
        assert result.allowed is True

    def test_check_action_denied_by_policy(self, engine):
        """Test checking action denied by policy."""
        deny_policy = Policy(
            name="deny_writes",
            description="Deny all writes",
            capabilities=["write_file"],
            allow=False,
            priority=100,
        )
        engine.add_policy(deny_policy)

        result = engine.check_action(
            agent="claude",
            tool="file_writer",
            capability="write_file",
        )

        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False
        assert "deny_writes" in result.reason

    def test_check_action_requires_approval_by_policy(self, engine):
        """Test checking action that requires approval by policy."""
        approval_policy = Policy(
            name="approve_writes",
            description="Writes need approval",
            capabilities=["write_file"],
            require_human_approval=True,
            priority=100,
        )
        engine.add_policy(approval_policy)

        result = engine.check_action(
            agent="claude",
            tool="file_writer",
            capability="write_file",
        )

        assert result.decision == PolicyDecision.ESCALATE
        assert result.requires_human_approval is True

    def test_check_action_requires_approval_by_capability(self, engine):
        """Test checking action where capability requires approval."""
        result = engine.check_action(
            agent="claude",
            tool="file_writer",
            capability="delete_file",  # Has requires_human_approval=True
        )

        assert result.decision == PolicyDecision.ESCALATE
        assert result.requires_human_approval is True

    def test_check_action_budget_exceeded(self, engine):
        """Test checking action when budget exceeded."""
        # Create a small budget
        budget = RiskBudget(total=5.0)
        engine._budgets["test_session"] = budget

        # Exhaust the budget
        budget.spend(5.0, "previous action")

        result = engine.check_action(
            agent="claude",
            tool="file_writer",
            capability="write_file",  # Has some risk cost
            session_id="test_session",
        )

        assert result.decision == PolicyDecision.BUDGET_EXCEEDED
        assert result.allowed is False

    def test_budget_tracking(self, engine):
        """Test that budget is tracked across actions."""
        # Perform some actions
        engine.check_action(
            agent="claude",
            tool="file_writer",
            capability="read_file",
            session_id="session1",
        )

        summary = engine.get_session_summary("session1")

        assert summary["session_id"] == "session1"
        assert summary["budget"]["spent"] > 0

    def test_audit_log(self, engine):
        """Test that actions are logged."""
        engine.check_action(
            agent="claude",
            tool="file_writer",
            capability="read_file",
        )

        audit_log = engine.get_audit_log()

        assert len(audit_log) > 0
        assert audit_log[-1]["agent"] == "claude"
        assert audit_log[-1]["tool"] == "file_writer"
        assert audit_log[-1]["capability"] == "read_file"


class TestDefaultPolicies:
    """Tests for default policies."""

    def test_protect_core_files(self):
        """Test the protect_core_files policy."""
        policy = None
        for p in DEFAULT_POLICIES:
            if p.name == "protect_core_files":
                policy = p
                break

        assert policy is not None
        assert policy.allow is False

        # Should match core file writes
        assert policy.matches(
            "claude", "file_writer", "write_file", {"file_path": "aragora/core.py"}
        )

        # Should not match other files
        assert not policy.matches(
            "claude", "file_writer", "write_file", {"file_path": "src/utils.py"}
        )

    def test_require_approval_for_push(self):
        """Test the require_approval_for_push policy."""
        policy = None
        for p in DEFAULT_POLICIES:
            if p.name == "require_approval_for_push":
                policy = p
                break

        assert policy is not None
        assert policy.require_human_approval is True

        # Should match git push
        assert policy.matches("claude", "git", "git_push", {})

        # Should not match git commit
        assert not policy.matches("claude", "git", "git_commit", {})


class TestCreateDefaultEngine:
    """Tests for create_default_engine function."""

    def test_creates_engine_with_policies(self):
        """Test that default engine has policies."""
        engine = create_default_engine()

        assert len(engine.policies) > 0
        policy_names = [p.name for p in engine.policies]
        assert "protect_core_files" in policy_names


class TestRiskBudgetIntegration:
    """Tests for risk budget integration."""

    def test_budget_per_session(self, engine):
        """Test that budgets are per-session."""
        engine.check_action("claude", "file_writer", "read_file", session_id="session1")
        engine.check_action("claude", "file_writer", "read_file", session_id="session2")

        summary1 = engine.get_session_summary("session1")
        summary2 = engine.get_session_summary("session2")

        # Each session should have its own budget
        assert summary1["session_id"] != summary2["session_id"]

    def test_action_spends_budget(self, engine):
        """Test that allowed actions spend budget."""
        budget_before = engine.get_budget("test_session").remaining

        engine.check_action(
            agent="claude",
            tool="file_writer",
            capability="read_file",
            session_id="test_session",
        )

        budget_after = engine.get_budget("test_session").remaining

        assert budget_after < budget_before
