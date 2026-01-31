"""
Tests for OpenClaw capability filtering.
"""

from __future__ import annotations

import pytest

from aragora.gateway.openclaw.capabilities import (
    CapabilityCategory,
    CapabilityCheckResult,
    CapabilityFilter,
    CapabilityRule,
    DEFAULT_CAPABILITY_RULES,
)


class TestCapabilityFilter:
    """Tests for CapabilityFilter."""

    def test_always_allowed_capability(self) -> None:
        """Test that always-allowed capabilities pass."""
        filter = CapabilityFilter()
        result = filter.check("text_generation")

        assert result.allowed is True
        assert result.category == CapabilityCategory.ALWAYS_ALLOWED
        assert "always allowed" in result.reason.lower()

    def test_blocked_by_default_capability(self) -> None:
        """Test that blocked-by-default capabilities are blocked."""
        filter = CapabilityFilter()
        result = filter.check("shell_execute")

        assert result.allowed is False
        assert result.category == CapabilityCategory.BLOCKED_BY_DEFAULT
        assert "blocked" in result.reason.lower()

    def test_approval_required_capability(self) -> None:
        """Test that approval-required capabilities return requires_approval."""
        filter = CapabilityFilter()
        result = filter.check("file_system_write")

        assert result.allowed is False
        assert result.category == CapabilityCategory.APPROVAL_REQUIRED
        assert result.requires_approval is True
        assert result.approval_gate is not None

    def test_tenant_configurable_disabled(self) -> None:
        """Test tenant-configurable capability when not enabled."""
        filter = CapabilityFilter()
        result = filter.check("browser_automation")

        assert result.allowed is False
        assert result.category == CapabilityCategory.TENANT_CONFIGURABLE
        assert "not enabled" in result.reason.lower()

    def test_tenant_configurable_enabled(self) -> None:
        """Test tenant-configurable capability when enabled."""
        filter = CapabilityFilter(tenant_enabled={"browser_automation"})
        result = filter.check("browser_automation")

        assert result.allowed is True
        assert result.category == CapabilityCategory.TENANT_CONFIGURABLE
        assert "enabled for this tenant" in result.reason.lower()

    def test_unknown_capability_blocked(self) -> None:
        """Test that unknown capabilities are blocked."""
        filter = CapabilityFilter()
        result = filter.check("unknown_capability_xyz")

        assert result.allowed is False
        assert result.category == CapabilityCategory.BLOCKED_BY_DEFAULT
        assert "unknown" in result.reason.lower()

    def test_blocked_override(self) -> None:
        """Test blocked override takes precedence."""
        filter = CapabilityFilter(blocked_override={"text_generation"})
        result = filter.check("text_generation")

        # Even though text_generation is normally always allowed,
        # the blocked_override should block it
        assert result.allowed is False
        assert "blocked by policy override" in result.reason.lower()

    def test_check_multiple(self) -> None:
        """Test checking multiple capabilities."""
        filter = CapabilityFilter()
        results = filter.check_multiple(["text_generation", "shell_execute"])

        assert "text_generation" in results
        assert "shell_execute" in results
        assert results["text_generation"].allowed is True
        assert results["shell_execute"].allowed is False

    def test_get_blocked_capabilities(self) -> None:
        """Test getting list of blocked capabilities."""
        filter = CapabilityFilter()
        blocked = filter.get_blocked_capabilities(
            [
                "text_generation",
                "shell_execute",
                "admin_escalate",
            ]
        )

        assert "text_generation" not in blocked
        assert "shell_execute" in blocked
        assert "admin_escalate" in blocked

    def test_enable_for_tenant(self) -> None:
        """Test enabling capability for tenant."""
        filter = CapabilityFilter()

        # Initially disabled
        result = filter.check("browser_automation")
        assert result.allowed is False

        # Enable it
        filter.enable_for_tenant("browser_automation")

        # Now allowed
        result = filter.check("browser_automation")
        assert result.allowed is True

    def test_disable_for_tenant(self) -> None:
        """Test disabling capability for tenant."""
        filter = CapabilityFilter(tenant_enabled={"browser_automation"})

        # Initially enabled
        result = filter.check("browser_automation")
        assert result.allowed is True

        # Disable it
        filter.disable_for_tenant("browser_automation")

        # Now blocked
        result = filter.check("browser_automation")
        assert result.allowed is False

    def test_add_block_override(self) -> None:
        """Test adding a block override."""
        filter = CapabilityFilter()

        # Initially allowed
        result = filter.check("search_internal")
        assert result.allowed is True

        # Add override
        filter.add_block_override("search_internal")

        # Now blocked
        result = filter.check("search_internal")
        assert result.allowed is False

    def test_remove_block_override(self) -> None:
        """Test removing a block override."""
        filter = CapabilityFilter(blocked_override={"text_generation"})

        # Initially blocked
        result = filter.check("text_generation")
        assert result.allowed is False

        # Remove override
        filter.remove_block_override("text_generation")

        # Now allowed again
        result = filter.check("text_generation")
        assert result.allowed is True


class TestDefaultCapabilityRules:
    """Tests for default capability rules."""

    def test_default_rules_exist(self) -> None:
        """Test that default rules are defined."""
        assert len(DEFAULT_CAPABILITY_RULES) > 0

    def test_always_allowed_rules(self) -> None:
        """Test that some rules are always allowed."""
        always_allowed = [
            name
            for name, rule in DEFAULT_CAPABILITY_RULES.items()
            if rule.category == CapabilityCategory.ALWAYS_ALLOWED
        ]
        assert "text_generation" in always_allowed
        assert "search_internal" in always_allowed

    def test_blocked_by_default_rules(self) -> None:
        """Test that some rules are blocked by default."""
        blocked = [
            name
            for name, rule in DEFAULT_CAPABILITY_RULES.items()
            if rule.category == CapabilityCategory.BLOCKED_BY_DEFAULT
        ]
        assert "shell_execute" in blocked
        assert "admin_escalate" in blocked

    def test_approval_required_have_gates(self) -> None:
        """Test that approval-required capabilities have gates."""
        for name, rule in DEFAULT_CAPABILITY_RULES.items():
            if rule.category == CapabilityCategory.APPROVAL_REQUIRED:
                assert rule.approval_gate is not None, f"{name} missing approval_gate"


class TestCapabilityCheckResult:
    """Tests for CapabilityCheckResult dataclass."""

    def test_result_fields(self) -> None:
        """Test result has all expected fields."""
        result = CapabilityCheckResult(
            allowed=True,
            capability="test",
            category=CapabilityCategory.ALWAYS_ALLOWED,
            reason="test reason",
        )

        assert result.allowed is True
        assert result.capability == "test"
        assert result.category == CapabilityCategory.ALWAYS_ALLOWED
        assert result.reason == "test reason"
        assert result.requires_approval is False  # Default
        assert result.approval_gate is None  # Default

    def test_result_with_approval(self) -> None:
        """Test result with approval requirement."""
        result = CapabilityCheckResult(
            allowed=False,
            capability="file_write",
            category=CapabilityCategory.APPROVAL_REQUIRED,
            reason="requires approval",
            requires_approval=True,
            approval_gate="computer_use.file_write",
        )

        assert result.requires_approval is True
        assert result.approval_gate == "computer_use.file_write"
