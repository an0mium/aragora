"""
Tests for the policy engine - Aragora's trust infrastructure.

Tests cover:
- Risk levels and blast radius calculations
- Risk budget management and spending
- Policy matching and evaluation
- PolicyEngine action checks
- Tool registry and capability lookups
- Default policies
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from aragora.policy.risk import (
    RiskLevel,
    BlastRadius,
    RiskBudget,
    get_risk_color,
    get_blast_radius_color,
)
from aragora.policy.tools import (
    Tool,
    ToolCapability,
    ToolCategory,
    ToolRegistry,
)
from aragora.policy.engine import (
    Policy,
    PolicyDecision,
    PolicyEngine,
    PolicyResult,
    PolicyViolation,
    DEFAULT_POLICIES,
    create_default_engine,
)


# =============================================================================
# Risk Level and Blast Radius Tests
# =============================================================================


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_risk_levels_ordering(self):
        """Risk levels should be ordered from NONE (0) to CRITICAL (4)."""
        assert RiskLevel.NONE < RiskLevel.LOW
        assert RiskLevel.LOW < RiskLevel.MEDIUM
        assert RiskLevel.MEDIUM < RiskLevel.HIGH
        assert RiskLevel.HIGH < RiskLevel.CRITICAL

    def test_risk_level_values(self):
        """Risk levels should have expected integer values."""
        assert RiskLevel.NONE.value == 0
        assert RiskLevel.LOW.value == 1
        assert RiskLevel.MEDIUM.value == 2
        assert RiskLevel.HIGH.value == 3
        assert RiskLevel.CRITICAL.value == 4

    def test_risk_colors(self):
        """Risk levels should have appropriate colors."""
        assert get_risk_color(RiskLevel.NONE) == "gray"
        assert get_risk_color(RiskLevel.LOW) == "green"
        assert get_risk_color(RiskLevel.MEDIUM) == "yellow"
        assert get_risk_color(RiskLevel.HIGH) == "orange"
        assert get_risk_color(RiskLevel.CRITICAL) == "red"


class TestBlastRadius:
    """Tests for BlastRadius enum."""

    def test_blast_radius_ordering(self):
        """Blast radius should be ordered from READ_ONLY (0) to PRODUCTION (4)."""
        assert BlastRadius.READ_ONLY < BlastRadius.DRAFT
        assert BlastRadius.DRAFT < BlastRadius.LOCAL
        assert BlastRadius.LOCAL < BlastRadius.SHARED
        assert BlastRadius.SHARED < BlastRadius.PRODUCTION

    def test_blast_radius_values(self):
        """Blast radius should have expected integer values."""
        assert BlastRadius.READ_ONLY.value == 0
        assert BlastRadius.DRAFT.value == 1
        assert BlastRadius.LOCAL.value == 2
        assert BlastRadius.SHARED.value == 3
        assert BlastRadius.PRODUCTION.value == 4

    def test_blast_radius_colors(self):
        """Blast radius should have appropriate colors."""
        assert get_blast_radius_color(BlastRadius.READ_ONLY) == "gray"
        assert get_blast_radius_color(BlastRadius.DRAFT) == "blue"
        assert get_blast_radius_color(BlastRadius.LOCAL) == "green"
        assert get_blast_radius_color(BlastRadius.SHARED) == "yellow"
        assert get_blast_radius_color(BlastRadius.PRODUCTION) == "red"


# =============================================================================
# Risk Budget Tests
# =============================================================================


class TestRiskBudget:
    """Tests for RiskBudget class."""

    def test_budget_creation_defaults(self):
        """Budget should have sensible defaults."""
        budget = RiskBudget()
        assert budget.total == 100.0
        assert budget.spent == 0.0
        assert budget.human_approval_threshold == 80.0
        assert budget.max_single_action == 30.0
        assert budget.remaining == 100.0

    def test_budget_custom_values(self):
        """Budget should accept custom values."""
        budget = RiskBudget(
            total=50.0,
            human_approval_threshold=40.0,
            max_single_action=15.0,
        )
        assert budget.total == 50.0
        assert budget.human_approval_threshold == 40.0
        assert budget.max_single_action == 15.0

    def test_remaining_calculation(self):
        """Remaining should be total minus spent."""
        budget = RiskBudget(total=100.0)
        budget.spent = 30.0
        assert budget.remaining == 70.0

    def test_remaining_never_negative(self):
        """Remaining should never be negative."""
        budget = RiskBudget(total=100.0)
        budget.spent = 150.0
        assert budget.remaining == 0

    def test_utilization_calculation(self):
        """Utilization should be spent/total."""
        budget = RiskBudget(total=100.0)
        budget.spent = 50.0
        assert budget.utilization == 0.5

    def test_utilization_zero_total(self):
        """Utilization with zero total should be 1.0 (fully utilized)."""
        budget = RiskBudget(total=0.0)
        assert budget.utilization == 1.0

    def test_cost_calculation_read_only(self):
        """Read-only actions should be free."""
        budget = RiskBudget()
        cost = budget.calculate_cost(RiskLevel.NONE, BlastRadius.READ_ONLY)
        assert cost == 0.0

    def test_cost_calculation_medium_local(self):
        """Medium risk + local blast should have moderate cost."""
        budget = RiskBudget()
        # Cost = risk_level * (blast_radius + 1) = 2 * (2 + 1) = 6
        cost = budget.calculate_cost(RiskLevel.MEDIUM, BlastRadius.LOCAL)
        assert cost == 6.0

    def test_cost_calculation_critical_production(self):
        """Critical risk + production blast should be expensive."""
        budget = RiskBudget()
        # Cost = risk_level * (blast_radius + 1) = 4 * (4 + 1) = 20
        cost = budget.calculate_cost(RiskLevel.CRITICAL, BlastRadius.PRODUCTION)
        assert cost == 20.0

    def test_cost_calculation_with_multiplier(self):
        """Cost multiplier should scale the cost."""
        budget = RiskBudget()
        cost = budget.calculate_cost(RiskLevel.MEDIUM, BlastRadius.LOCAL, multiplier=2.0)
        assert cost == 12.0  # 6.0 * 2

    def test_can_afford_within_budget(self):
        """Should return True when cost is within budget."""
        budget = RiskBudget(total=100.0, max_single_action=60.0)
        assert budget.can_afford(50.0) is True

    def test_can_afford_exceeds_budget(self):
        """Should return False when cost exceeds remaining budget."""
        budget = RiskBudget(total=100.0)
        budget.spent = 90.0
        assert budget.can_afford(20.0) is False

    def test_can_afford_exceeds_max_single_action(self):
        """Should return False when cost exceeds max single action."""
        budget = RiskBudget(total=100.0, max_single_action=10.0)
        assert budget.can_afford(15.0) is False

    def test_can_afford_without_approval(self):
        """Should return True when cost stays below approval threshold."""
        budget = RiskBudget(total=100.0, human_approval_threshold=80.0, max_single_action=60.0)
        assert budget.can_afford_without_approval(50.0) is True

    def test_can_afford_without_approval_exceeds_threshold(self):
        """Should return False when cost would exceed approval threshold."""
        budget = RiskBudget(total=100.0, human_approval_threshold=80.0)
        budget.spent = 70.0
        assert budget.can_afford_without_approval(15.0) is False

    def test_spend_tracks_action(self):
        """Spending should track the action."""
        budget = RiskBudget(total=100.0)
        budget.spend(10.0, "write config", agent="claude", tool="file_writer")

        assert budget.spent == 10.0
        assert len(budget.actions) == 1
        assert budget.actions[0]["cost"] == 10.0
        assert budget.actions[0]["description"] == "write config"
        assert budget.actions[0]["agent"] == "claude"
        assert budget.actions[0]["tool"] == "file_writer"

    def test_spend_returns_true_when_within_budget(self):
        """Spending should return True when within budget."""
        budget = RiskBudget(total=100.0)
        result = budget.spend(10.0, "test action")
        assert result is True

    def test_spend_returns_false_when_exceeds_budget(self):
        """Spending should return False when exceeding budget."""
        budget = RiskBudget(total=100.0, max_single_action=50.0)
        budget.spent = 95.0
        result = budget.spend(10.0, "test action")
        assert result is False

    def test_requires_human_approval_below_threshold(self):
        """Should not require approval below threshold."""
        budget = RiskBudget(total=100.0, human_approval_threshold=80.0)
        budget.spent = 50.0
        assert budget.requires_human_approval is False

    def test_requires_human_approval_at_threshold(self):
        """Should require approval at or above threshold."""
        budget = RiskBudget(total=100.0, human_approval_threshold=80.0)
        budget.spent = 80.0
        assert budget.requires_human_approval is True

    def test_to_dict(self):
        """Budget should serialize to dict correctly."""
        budget = RiskBudget(total=100.0)
        budget.spend(25.0, "test")

        d = budget.to_dict()
        assert d["total"] == 100.0
        assert d["spent"] == 25.0
        assert d["remaining"] == 75.0
        assert d["utilization"] == 0.25
        assert d["action_count"] == 1


# =============================================================================
# Tool and ToolCapability Tests
# =============================================================================


class TestToolCapability:
    """Tests for ToolCapability dataclass."""

    def test_capability_defaults(self):
        """Capability should have sensible defaults."""
        cap = ToolCapability(name="read_file", description="Read a file")
        assert cap.risk_level == RiskLevel.LOW
        assert cap.blast_radius == BlastRadius.LOCAL
        assert cap.requires_human_approval is False
        assert cap.max_uses_per_session is None
        assert cap.cooldown_seconds == 0.0

    def test_capability_custom_values(self):
        """Capability should accept custom risk values."""
        cap = ToolCapability(
            name="delete_file",
            description="Delete a file",
            risk_level=RiskLevel.HIGH,
            blast_radius=BlastRadius.SHARED,
            requires_human_approval=True,
        )
        assert cap.risk_level == RiskLevel.HIGH
        assert cap.blast_radius == BlastRadius.SHARED
        assert cap.requires_human_approval is True


class TestTool:
    """Tests for Tool dataclass."""

    def test_tool_creation(self):
        """Tool should be creatable with required fields."""
        tool = Tool(
            name="file_writer",
            description="Write files",
            category=ToolCategory.WRITE,
        )
        assert tool.name == "file_writer"
        assert tool.category == ToolCategory.WRITE

    def test_tool_with_capabilities(self):
        """Tool should accept a list of capabilities."""
        tool = Tool(
            name="file_writer",
            description="Write files",
            category=ToolCategory.WRITE,
            capabilities=[
                ToolCapability("write_file", "Write to file"),
                ToolCapability("delete_file", "Delete file", RiskLevel.HIGH),
            ],
        )
        assert len(tool.capabilities) == 2

    def test_get_capability_found(self):
        """get_capability should return matching capability."""
        cap = ToolCapability("write_file", "Write to file")
        tool = Tool(
            name="file_writer",
            description="Write files",
            category=ToolCategory.WRITE,
            capabilities=[cap],
        )
        assert tool.get_capability("write_file") == cap

    def test_get_capability_not_found(self):
        """get_capability should return None for unknown capability."""
        tool = Tool(
            name="file_writer",
            description="Write files",
            category=ToolCategory.WRITE,
        )
        assert tool.get_capability("nonexistent") is None

    def test_has_capability(self):
        """has_capability should check for capability existence."""
        tool = Tool(
            name="file_writer",
            description="Write files",
            category=ToolCategory.WRITE,
            capabilities=[ToolCapability("write_file", "Write")],
        )
        assert tool.has_capability("write_file") is True
        assert tool.has_capability("delete_file") is False

    def test_to_dict(self):
        """Tool should serialize to dict correctly."""
        tool = Tool(
            name="file_writer",
            description="Write files",
            category=ToolCategory.WRITE,
            capabilities=[ToolCapability("write_file", "Write")],
            risk_level=RiskLevel.MEDIUM,
        )
        d = tool.to_dict()
        assert d["name"] == "file_writer"
        assert d["category"] == "write"
        assert d["risk_level"] == "MEDIUM"
        assert len(d["capabilities"]) == 1


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_tool(self):
        """Should be able to register a tool."""
        registry = ToolRegistry()
        tool = Tool(
            name="test_tool",
            description="Test",
            category=ToolCategory.READ,
        )
        registry.register(tool)
        assert registry.get("test_tool") == tool

    def test_register_overwrites_existing(self):
        """Registering same name should overwrite."""
        registry = ToolRegistry()
        tool1 = Tool(name="test", description="First", category=ToolCategory.READ)
        tool2 = Tool(name="test", description="Second", category=ToolCategory.WRITE)

        registry.register(tool1)
        registry.register(tool2)

        assert registry.get("test").description == "Second"

    def test_unregister_tool(self):
        """Should be able to unregister a tool."""
        registry = ToolRegistry()
        tool = Tool(name="test", description="Test", category=ToolCategory.READ)
        registry.register(tool)

        result = registry.unregister("test")
        assert result is True
        assert registry.get("test") is None

    def test_unregister_nonexistent(self):
        """Unregistering nonexistent tool should return False."""
        registry = ToolRegistry()
        result = registry.unregister("nonexistent")
        assert result is False

    def test_list_tools(self):
        """Should list all registered tools."""
        registry = ToolRegistry()
        tool1 = Tool(name="tool1", description="First", category=ToolCategory.READ)
        tool2 = Tool(name="tool2", description="Second", category=ToolCategory.WRITE)

        registry.register(tool1)
        registry.register(tool2)

        tools = registry.list_tools()
        assert len(tools) == 2
        assert tool1 in tools
        assert tool2 in tools

    def test_capability_indexing(self):
        """Registry should index capabilities for lookup."""
        registry = ToolRegistry()
        tool = Tool(
            name="file_tool",
            description="File ops",
            category=ToolCategory.WRITE,
            capabilities=[
                ToolCapability("read_file", "Read"),
                ToolCapability("write_file", "Write"),
            ],
        )
        registry.register(tool)

        tools = registry.find_tools_with_capability("write_file")
        assert len(tools) == 1
        assert tools[0].name == "file_tool"

    def test_find_tools_multiple(self):
        """Should find all tools with a capability."""
        registry = ToolRegistry()
        tool1 = Tool(
            name="tool1",
            description="First",
            category=ToolCategory.WRITE,
            capabilities=[ToolCapability("write_file", "Write")],
        )
        tool2 = Tool(
            name="tool2",
            description="Second",
            category=ToolCategory.WRITE,
            capabilities=[ToolCapability("write_file", "Write")],
        )

        registry.register(tool1)
        registry.register(tool2)

        tools = registry.find_tools_with_capability("write_file")
        assert len(tools) == 2


# =============================================================================
# Policy Tests
# =============================================================================


class TestPolicy:
    """Tests for Policy matching logic."""

    def test_policy_matches_all(self):
        """Empty lists should match all agents/tools/capabilities."""
        policy = Policy(name="allow_all", description="Allow everything")
        assert policy.matches("any_agent", "any_tool", "any_cap", {}) is True

    def test_policy_matches_specific_agent(self):
        """Policy should match only specified agents."""
        policy = Policy(
            name="claude_only",
            description="Only Claude",
            agents=["claude"],
        )
        assert policy.matches("claude", "tool", "cap", {}) is True
        assert policy.matches("gpt", "tool", "cap", {}) is False

    def test_policy_matches_specific_tool(self):
        """Policy should match only specified tools."""
        policy = Policy(
            name="file_ops",
            description="File operations",
            tools=["file_writer"],
        )
        assert policy.matches("agent", "file_writer", "cap", {}) is True
        assert policy.matches("agent", "network", "cap", {}) is False

    def test_policy_matches_specific_capability(self):
        """Policy should match only specified capabilities."""
        policy = Policy(
            name="write_only",
            description="Write only",
            capabilities=["write_file"],
        )
        assert policy.matches("agent", "tool", "write_file", {}) is True
        assert policy.matches("agent", "tool", "read_file", {}) is False

    def test_policy_disabled(self):
        """Disabled policies should not match."""
        policy = Policy(
            name="disabled",
            description="Disabled policy",
            enabled=False,
        )
        assert policy.matches("agent", "tool", "cap", {}) is False

    def test_policy_condition_simple(self):
        """Policy should evaluate simple conditions."""
        policy = Policy(
            name="large_files",
            description="Large files require approval",
            conditions=["file_size > 1000"],
        )
        assert policy.matches("agent", "tool", "cap", {"file_size": 2000}) is True
        assert policy.matches("agent", "tool", "cap", {"file_size": 500}) is False

    def test_policy_condition_string_match(self):
        """Policy should evaluate string conditions."""
        policy = Policy(
            name="core_files",
            description="Core files protected",
            conditions=["'core.py' in file_path"],
        )
        assert policy.matches("agent", "tool", "cap", {"file_path": "aragora/core.py"}) is True
        assert policy.matches("agent", "tool", "cap", {"file_path": "utils.py"}) is False

    def test_policy_condition_multiple(self):
        """All conditions must be satisfied."""
        policy = Policy(
            name="multi_cond",
            description="Multiple conditions",
            conditions=["size > 100", "extension == '.py'"],
        )
        assert policy.matches("agent", "tool", "cap", {"size": 200, "extension": ".py"}) is True
        assert policy.matches("agent", "tool", "cap", {"size": 200, "extension": ".js"}) is False
        assert policy.matches("agent", "tool", "cap", {"size": 50, "extension": ".py"}) is False

    def test_policy_condition_invalid(self):
        """Invalid conditions should not match."""
        policy = Policy(
            name="bad_cond",
            description="Bad condition",
            conditions=["undefined_var > 0"],
        )
        assert policy.matches("agent", "tool", "cap", {}) is False


# =============================================================================
# PolicyEngine Tests
# =============================================================================


class TestPolicyEngine:
    """Tests for PolicyEngine class."""

    @pytest.fixture
    def registry(self):
        """Create a tool registry with test tools."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="file_writer",
                description="Write files",
                category=ToolCategory.WRITE,
                capabilities=[
                    ToolCapability("read_file", "Read", RiskLevel.NONE, BlastRadius.READ_ONLY),
                    ToolCapability("write_file", "Write", RiskLevel.MEDIUM, BlastRadius.LOCAL),
                    ToolCapability("delete_file", "Delete", RiskLevel.HIGH, BlastRadius.LOCAL),
                ],
            )
        )
        registry.register(
            Tool(
                name="git",
                description="Git operations",
                category=ToolCategory.EXECUTE,
                capabilities=[
                    ToolCapability("git_status", "Status", RiskLevel.NONE, BlastRadius.READ_ONLY),
                    ToolCapability("git_commit", "Commit", RiskLevel.MEDIUM, BlastRadius.LOCAL),
                    ToolCapability("git_push", "Push", RiskLevel.HIGH, BlastRadius.SHARED),
                ],
            )
        )
        return registry

    @pytest.fixture
    def engine(self, registry):
        """Create an engine with the test registry."""
        return PolicyEngine(tool_registry=registry)

    def test_check_action_unknown_tool(self, engine):
        """Should deny unknown tools."""
        result = engine.check_action("agent", "unknown_tool", "cap")
        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False
        assert "Unknown tool" in result.reason

    def test_check_action_unknown_capability(self, engine):
        """Should deny unknown capabilities."""
        result = engine.check_action("agent", "file_writer", "unknown_cap")
        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False
        assert "does not have capability" in result.reason

    def test_check_action_allowed(self, engine):
        """Should allow valid actions within budget."""
        result = engine.check_action("agent", "file_writer", "read_file")
        assert result.decision == PolicyDecision.ALLOW
        assert result.allowed is True

    def test_check_action_denied_by_policy(self, engine):
        """Should deny actions blocked by policy."""
        engine.add_policy(
            Policy(
                name="block_delete",
                description="Block all deletes",
                capabilities=["delete_file"],
                allow=False,
                priority=100,
            )
        )

        result = engine.check_action("agent", "file_writer", "delete_file")
        assert result.decision == PolicyDecision.DENY
        assert result.allowed is False
        assert "block_delete" in result.reason

    def test_check_action_escalate_by_policy(self, engine):
        """Should escalate actions requiring approval."""
        engine.add_policy(
            Policy(
                name="approve_push",
                description="Require approval for push",
                capabilities=["git_push"],
                require_human_approval=True,
                priority=100,
            )
        )

        result = engine.check_action("agent", "git", "git_push")
        assert result.decision == PolicyDecision.ESCALATE
        assert result.allowed is False
        assert result.requires_human_approval is True

    def test_check_action_budget_exceeded(self, engine):
        """Should deny when budget exceeded."""
        # Set a very small budget
        budget = engine.get_budget("test_session")
        budget.total = 5.0
        budget.spent = 4.0

        # write_file has cost > 1
        result = engine.check_action(
            "agent", "file_writer", "write_file", session_id="test_session"
        )
        assert result.decision == PolicyDecision.BUDGET_EXCEEDED
        assert result.allowed is False

    def test_check_action_spends_budget(self, engine):
        """Should spend budget on allowed actions."""
        budget = engine.get_budget("test_session")
        initial_remaining = budget.remaining

        engine.check_action("agent", "file_writer", "write_file", session_id="test_session")

        assert budget.remaining < initial_remaining

    def test_add_policy_sorted_by_priority(self, engine):
        """Policies should be sorted by priority (descending)."""
        engine.add_policy(Policy(name="low", description="Low", priority=10))
        engine.add_policy(Policy(name="high", description="High", priority=100))
        engine.add_policy(Policy(name="medium", description="Medium", priority=50))

        assert engine.policies[0].name == "high"
        assert engine.policies[1].name == "medium"
        assert engine.policies[2].name == "low"

    def test_remove_policy(self, engine):
        """Should be able to remove policies."""
        engine.add_policy(Policy(name="test", description="Test"))
        assert len(engine.policies) == 1

        result = engine.remove_policy("test")
        assert result is True
        assert len(engine.policies) == 0

    def test_remove_policy_not_found(self, engine):
        """Removing nonexistent policy should return False."""
        result = engine.remove_policy("nonexistent")
        assert result is False

    def test_get_audit_log(self, engine):
        """Audit log should track all checks."""
        engine.check_action("agent", "file_writer", "read_file")
        engine.check_action("agent", "file_writer", "write_file")

        log = engine.get_audit_log()
        assert len(log) >= 2

    def test_get_session_summary(self, engine):
        """Should provide session summary."""
        engine.check_action("agent", "file_writer", "write_file", session_id="test_session")

        summary = engine.get_session_summary("test_session")
        assert summary["session_id"] == "test_session"
        assert "budget" in summary
        assert summary["budget"]["spent"] > 0

    def test_policy_risk_multiplier(self, engine):
        """Policy risk multiplier should affect cost calculation."""
        engine.add_policy(
            Policy(
                name="dangerous_agent",
                description="Risky agent",
                agents=["risky_agent"],
                risk_multiplier=2.0,
                priority=100,
            )
        )

        # Check a normal agent first
        result1 = engine.check_action("normal_agent", "file_writer", "write_file")
        cost1 = result1.risk_cost

        # Reset budget
        engine._budgets = {}

        # Check risky agent - should have higher cost
        result2 = engine.check_action("risky_agent", "file_writer", "write_file")
        cost2 = result2.risk_cost

        assert cost2 == cost1 * 2.0


class TestPolicyViolation:
    """Tests for PolicyViolation exception."""

    def test_policy_violation_creation(self):
        """PolicyViolation should wrap a PolicyResult."""
        result = PolicyResult(
            decision=PolicyDecision.DENY,
            allowed=False,
            reason="Test violation",
        )
        exc = PolicyViolation(result)
        assert exc.result == result
        assert "Test violation" in str(exc)


class TestDefaultPolicies:
    """Tests for default policy configurations."""

    def test_default_policies_exist(self):
        """Should have default policies defined."""
        assert len(DEFAULT_POLICIES) > 0

    def test_protect_core_files_policy(self):
        """Should have policy protecting core files."""
        policy = next((p for p in DEFAULT_POLICIES if p.name == "protect_core_files"), None)
        assert policy is not None
        assert policy.allow is False

        # Should match core.py
        assert (
            policy.matches("agent", "file_writer", "write_file", {"file_path": "aragora/core.py"})
            is True
        )

        # Should not match other files
        assert (
            policy.matches("agent", "file_writer", "write_file", {"file_path": "aragora/utils.py"})
            is False
        )

    def test_require_approval_for_push(self):
        """Should have policy requiring approval for git push."""
        policy = next((p for p in DEFAULT_POLICIES if p.name == "require_approval_for_push"), None)
        assert policy is not None
        assert policy.require_human_approval is True

    def test_create_default_engine(self):
        """create_default_engine should include default policies."""
        engine = create_default_engine()
        assert len(engine.policies) == len(DEFAULT_POLICIES)


# =============================================================================
# Integration Tests
# =============================================================================


class TestPolicyEngineIntegration:
    """Integration tests for the policy engine."""

    def test_full_workflow(self):
        """Test a complete policy evaluation workflow."""
        # Create registry with tools
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="code_editor",
                description="Edit code files",
                category=ToolCategory.WRITE,
                capabilities=[
                    ToolCapability("edit_file", "Edit", RiskLevel.MEDIUM, BlastRadius.LOCAL),
                ],
            )
        )

        # Create engine with custom budget
        engine = PolicyEngine(
            tool_registry=registry,
            default_budget=RiskBudget(total=50.0, human_approval_threshold=40.0),
        )

        # Add a policy
        engine.add_policy(
            Policy(
                name="trusted_agents",
                description="Trusted agents get lower risk",
                agents=["trusted"],
                risk_multiplier=0.5,
                priority=50,
            )
        )

        # Check actions
        result1 = engine.check_action("trusted", "code_editor", "edit_file", session_id="s1")
        assert result1.allowed is True

        result2 = engine.check_action("untrusted", "code_editor", "edit_file", session_id="s2")
        assert result2.allowed is True

        # Trusted agent should have lower cost
        assert result1.risk_cost < result2.risk_cost

    def test_budget_exhaustion(self):
        """Test that budget exhaustion properly blocks actions."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="expensive_tool",
                description="Expensive operations",
                category=ToolCategory.EXECUTE,
                cost_multiplier=3.0,  # 3x cost
                capabilities=[
                    # Cost = HIGH(3) * (SHARED(3) + 1) * 3.0 = 36
                    ToolCapability("run", "Run", RiskLevel.HIGH, BlastRadius.SHARED),
                ],
            )
        )

        engine = PolicyEngine(
            tool_registry=registry,
            # First action costs 36, so total=50 allows first but not second
            default_budget=RiskBudget(total=50.0, max_single_action=50.0),
        )

        # First action should succeed
        result1 = engine.check_action("agent", "expensive_tool", "run", session_id="s1")
        assert result1.allowed is True

        # Second action should fail (budget exceeded)
        result2 = engine.check_action("agent", "expensive_tool", "run", session_id="s1")
        assert result2.allowed is False
        assert result2.decision == PolicyDecision.BUDGET_EXCEEDED
