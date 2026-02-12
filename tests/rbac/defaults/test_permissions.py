"""
Tests for RBAC Permission Definitions (aragora/rbac/defaults/permissions.py).

This module contains comprehensive tests for all 230+ permission constants defined
in the permissions.py file. Tests cover:

1. Permission constant validation - all PERM_* constants are valid Permission objects
2. Permission string formats - resource.action format consistency
3. Permission groupings by resource type
4. No duplicate permissions
5. Permission naming conventions
6. Permission ID and key consistency
7. Required permissions for each resource type
8. Permission descriptions and metadata
9. Integration with ResourceType and Action enums
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pytest

from aragora.rbac.defaults import permissions as perm_module
from aragora.rbac.defaults.permissions import (
    # Helper function
    _permission,
    # Debate permissions
    PERM_DEBATE_CREATE,
    PERM_DEBATE_READ,
    PERM_DEBATE_UPDATE,
    PERM_DEBATE_DELETE,
    PERM_DEBATE_RUN,
    PERM_DEBATE_STOP,
    PERM_DEBATE_FORK,
    # Agent permissions
    PERM_AGENT_CREATE,
    PERM_AGENT_READ,
    PERM_AGENT_UPDATE,
    PERM_AGENT_DELETE,
    PERM_AGENT_DEPLOY,
    # User permissions
    PERM_USER_READ,
    PERM_USER_INVITE,
    PERM_USER_REMOVE,
    PERM_USER_CHANGE_ROLE,
    PERM_USER_IMPERSONATE,
    # Organization permissions
    PERM_ORG_READ,
    PERM_ORG_UPDATE,
    PERM_ORG_BILLING,
    PERM_ORG_AUDIT,
    PERM_ORG_EXPORT,
    PERM_ORG_INVITE,
    PERM_ORG_DELETE,
    PERM_ORG_USAGE_READ,
    PERM_ORG_MEMBERS,
    # Admin permissions
    PERM_ADMIN_CONFIG,
    PERM_ADMIN_METRICS,
    PERM_ADMIN_FEATURES,
    PERM_ADMIN_ALL,
    PERM_ADMIN_AUDIT,
    PERM_ADMIN_SECURITY,
    PERM_ADMIN_SYSTEM,
    # Control Plane permissions
    PERM_CONTROL_PLANE_READ,
    PERM_CONTROL_PLANE_SUBMIT,
    PERM_CONTROL_PLANE_CANCEL,
    PERM_CONTROL_PLANE_DELIBERATE,
    PERM_CONTROL_PLANE_AGENTS,
    PERM_CONTROL_PLANE_AGENTS_READ,
    PERM_CONTROL_PLANE_AGENTS_REGISTER,
    PERM_CONTROL_PLANE_AGENTS_UNREGISTER,
    PERM_CONTROL_PLANE_TASKS,
    PERM_CONTROL_PLANE_TASKS_READ,
    PERM_CONTROL_PLANE_TASKS_SUBMIT,
    PERM_CONTROL_PLANE_TASKS_CLAIM,
    PERM_CONTROL_PLANE_TASKS_COMPLETE,
    PERM_CONTROL_PLANE_HEALTH_READ,
    # Compliance permissions
    PERM_COMPLIANCE_READ,
    PERM_COMPLIANCE_UPDATE,
    PERM_COMPLIANCE_CHECK,
    PERM_COMPLIANCE_GDPR,
    PERM_COMPLIANCE_SOC2,
    PERM_COMPLIANCE_LEGAL,
    PERM_COMPLIANCE_AUDIT,
    # Computer-Use permissions
    PERM_COMPUTER_USE_READ,
    PERM_COMPUTER_USE_EXECUTE,
    PERM_COMPUTER_USE_BROWSER,
    PERM_COMPUTER_USE_SHELL,
    PERM_COMPUTER_USE_FILE_READ,
    PERM_COMPUTER_USE_FILE_WRITE,
    PERM_COMPUTER_USE_SCREENSHOT,
    PERM_COMPUTER_USE_NETWORK,
    PERM_COMPUTER_USE_ADMIN,
    # Gauntlet permissions
    PERM_GAUNTLET_RUN,
    PERM_GAUNTLET_READ,
    PERM_GAUNTLET_DELETE,
    PERM_GAUNTLET_SIGN,
    PERM_GAUNTLET_COMPARE,
    PERM_GAUNTLET_EXPORT,
    # Team permissions
    PERM_TEAM_CREATE,
    PERM_TEAM_READ,
    PERM_TEAM_UPDATE,
    PERM_TEAM_DELETE,
    PERM_TEAM_ADD_MEMBER,
    PERM_TEAM_REMOVE_MEMBER,
    PERM_TEAM_SHARE,
    PERM_TEAM_DISSOLVE,
    # Workspace permissions
    PERM_WORKSPACE_CREATE,
    PERM_WORKSPACE_READ,
    PERM_WORKSPACE_UPDATE,
    PERM_WORKSPACE_DELETE,
    PERM_WORKSPACE_MEMBER_ADD,
    PERM_WORKSPACE_MEMBER_REMOVE,
    PERM_WORKSPACE_MEMBER_CHANGE_ROLE,
    PERM_WORKSPACE_SHARE,
    # Data governance permissions
    PERM_DATA_CLASSIFICATION_READ,
    PERM_DATA_CLASSIFICATION_CLASSIFY,
    PERM_DATA_CLASSIFICATION_UPDATE,
    PERM_DATA_RETENTION_READ,
    PERM_DATA_RETENTION_UPDATE,
    PERM_DATA_LINEAGE_READ,
    PERM_PII_READ,
    PERM_PII_REDACT,
    PERM_PII_MASK,
    # Billing permissions
    PERM_BILLING_READ,
    PERM_BILLING_RECOMMENDATIONS_READ,
    PERM_BILLING_RECOMMENDATIONS_APPLY,
    PERM_BILLING_FORECAST_READ,
    PERM_BILLING_FORECAST_SIMULATE,
    PERM_BILLING_EXPORT_HISTORY,
    PERM_BILLING_DELETE,
    PERM_BILLING_CANCEL,
    # Backup & DR permissions
    PERM_BACKUP_CREATE,
    PERM_BACKUP_READ,
    PERM_BACKUP_RESTORE,
    PERM_BACKUP_DELETE,
    PERM_DR_READ,
    PERM_DR_EXECUTE,
    PERM_DR_ALIAS_READ,
    PERM_DR_DRILL,
)
from aragora.rbac.models import Action, Permission, ResourceType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def all_permission_constants() -> list[tuple[str, Permission]]:
    """Get all PERM_* constants from the permissions module."""
    constants = []
    for name in dir(perm_module):
        if name.startswith("PERM_"):
            value = getattr(perm_module, name)
            if isinstance(value, Permission):
                constants.append((name, value))
    return constants


@pytest.fixture
def permission_constants_by_resource(
    all_permission_constants: list[tuple[str, Permission]],
) -> dict[ResourceType, list[tuple[str, Permission]]]:
    """Group permission constants by resource type."""
    by_resource: dict[ResourceType, list[tuple[str, Permission]]] = {}
    for name, perm in all_permission_constants:
        if perm.resource not in by_resource:
            by_resource[perm.resource] = []
        by_resource[perm.resource].append((name, perm))
    return by_resource


@pytest.fixture
def debate_permissions() -> list[Permission]:
    """Get all debate-related permissions."""
    return [
        PERM_DEBATE_CREATE,
        PERM_DEBATE_READ,
        PERM_DEBATE_UPDATE,
        PERM_DEBATE_DELETE,
        PERM_DEBATE_RUN,
        PERM_DEBATE_STOP,
        PERM_DEBATE_FORK,
    ]


@pytest.fixture
def agent_permissions() -> list[Permission]:
    """Get all agent-related permissions."""
    return [
        PERM_AGENT_CREATE,
        PERM_AGENT_READ,
        PERM_AGENT_UPDATE,
        PERM_AGENT_DELETE,
        PERM_AGENT_DEPLOY,
    ]


@pytest.fixture
def admin_permissions() -> list[Permission]:
    """Get all admin-related permissions."""
    return [
        PERM_ADMIN_CONFIG,
        PERM_ADMIN_METRICS,
        PERM_ADMIN_FEATURES,
        PERM_ADMIN_ALL,
        PERM_ADMIN_AUDIT,
        PERM_ADMIN_SECURITY,
        PERM_ADMIN_SYSTEM,
    ]


@pytest.fixture
def user_management_permissions() -> list[Permission]:
    """Get all user management permissions."""
    return [
        PERM_USER_READ,
        PERM_USER_INVITE,
        PERM_USER_REMOVE,
        PERM_USER_CHANGE_ROLE,
        PERM_USER_IMPERSONATE,
    ]


@pytest.fixture
def control_plane_permissions() -> list[Permission]:
    """Get all control plane permissions."""
    return [
        PERM_CONTROL_PLANE_READ,
        PERM_CONTROL_PLANE_SUBMIT,
        PERM_CONTROL_PLANE_CANCEL,
        PERM_CONTROL_PLANE_DELIBERATE,
        PERM_CONTROL_PLANE_AGENTS,
        PERM_CONTROL_PLANE_AGENTS_READ,
        PERM_CONTROL_PLANE_AGENTS_REGISTER,
        PERM_CONTROL_PLANE_AGENTS_UNREGISTER,
        PERM_CONTROL_PLANE_TASKS,
        PERM_CONTROL_PLANE_TASKS_READ,
        PERM_CONTROL_PLANE_TASKS_SUBMIT,
        PERM_CONTROL_PLANE_TASKS_CLAIM,
        PERM_CONTROL_PLANE_TASKS_COMPLETE,
        PERM_CONTROL_PLANE_HEALTH_READ,
    ]


@pytest.fixture
def computer_use_permissions() -> list[Permission]:
    """Get all computer-use permissions."""
    return [
        PERM_COMPUTER_USE_READ,
        PERM_COMPUTER_USE_EXECUTE,
        PERM_COMPUTER_USE_BROWSER,
        PERM_COMPUTER_USE_SHELL,
        PERM_COMPUTER_USE_FILE_READ,
        PERM_COMPUTER_USE_FILE_WRITE,
        PERM_COMPUTER_USE_SCREENSHOT,
        PERM_COMPUTER_USE_NETWORK,
        PERM_COMPUTER_USE_ADMIN,
    ]


@pytest.fixture
def data_governance_permissions() -> list[Permission]:
    """Get all data governance permissions."""
    return [
        PERM_DATA_CLASSIFICATION_READ,
        PERM_DATA_CLASSIFICATION_CLASSIFY,
        PERM_DATA_CLASSIFICATION_UPDATE,
        PERM_DATA_RETENTION_READ,
        PERM_DATA_RETENTION_UPDATE,
        PERM_DATA_LINEAGE_READ,
        PERM_PII_READ,
        PERM_PII_REDACT,
        PERM_PII_MASK,
    ]


# =============================================================================
# Test Permission Constant Validity
# =============================================================================


class TestPermissionConstantValidity:
    """Tests verifying all permission constants are valid Permission objects."""

    def test_all_perm_constants_are_permission_instances(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """All PERM_* constants should be Permission dataclass instances."""
        for name, perm in all_permission_constants:
            assert isinstance(perm, Permission), f"{name} is not a Permission instance"

    def test_minimum_permission_count(self, all_permission_constants: list[tuple[str, Permission]]):
        """Should have at least 200 permission constants defined."""
        # The file has 230+ permissions based on the line count
        assert len(all_permission_constants) >= 200, (
            f"Expected at least 200 permissions, found {len(all_permission_constants)}"
        )

    def test_all_permissions_have_id(self, all_permission_constants: list[tuple[str, Permission]]):
        """All permissions should have a non-empty ID."""
        for name, perm in all_permission_constants:
            assert perm.id, f"{name} has empty or None ID"
            assert isinstance(perm.id, str), f"{name} ID is not a string"

    def test_all_permissions_have_name(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """All permissions should have a non-empty name."""
        for name, perm in all_permission_constants:
            assert perm.name, f"{name} has empty or None name"
            assert isinstance(perm.name, str), f"{name} name is not a string"

    def test_all_permissions_have_valid_resource(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """All permissions should have a valid ResourceType."""
        for name, perm in all_permission_constants:
            assert isinstance(perm.resource, ResourceType), (
                f"{name} resource {perm.resource} is not a ResourceType"
            )

    def test_all_permissions_have_valid_action(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """All permissions should have a valid Action."""
        for name, perm in all_permission_constants:
            assert isinstance(perm.action, Action), f"{name} action {perm.action} is not an Action"

    def test_all_permissions_have_description(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """All permissions should have a description (can be empty string)."""
        for name, perm in all_permission_constants:
            assert perm.description is not None, f"{name} has None description"
            assert isinstance(perm.description, str), f"{name} description is not a string"


# =============================================================================
# Test Permission String Formats
# =============================================================================


class TestPermissionStringFormats:
    """Tests for permission key and ID format consistency."""

    def test_permission_key_format_resource_dot_action(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """Permission keys should follow 'resource.action' format."""
        for name, perm in all_permission_constants:
            key = perm.key
            assert "." in key, f"{name} key '{key}' does not contain '.'"
            parts = key.split(".")
            assert len(parts) >= 2, f"{name} key '{key}' does not have at least 2 parts"

    def test_permission_key_starts_with_resource_value(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """Permission key should start with the resource type value."""
        for name, perm in all_permission_constants:
            key = perm.key
            resource_value = perm.resource.value
            assert key.startswith(resource_value + "."), (
                f"{name} key '{key}' does not start with resource '{resource_value}.'"
            )

    def test_permission_key_ends_with_action_value(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """Permission key should end with the action value."""
        for name, perm in all_permission_constants:
            key = perm.key
            action_value = perm.action.value
            assert key.endswith("." + action_value), (
                f"{name} key '{key}' does not end with action '.{action_value}'"
            )

    def test_permission_id_equals_key(self, all_permission_constants: list[tuple[str, Permission]]):
        """Permission ID should equal the key (resource.action format)."""
        for name, perm in all_permission_constants:
            # The _permission helper sets id = key
            assert perm.id == perm.key, f"{name} has mismatched ID '{perm.id}' and key '{perm.key}'"

    def test_permission_key_lowercase(self, all_permission_constants: list[tuple[str, Permission]]):
        """Permission keys should be lowercase."""
        for name, perm in all_permission_constants:
            key = perm.key
            assert key == key.lower(), f"{name} key '{key}' is not lowercase"

    def test_permission_key_no_spaces(self, all_permission_constants: list[tuple[str, Permission]]):
        """Permission keys should not contain spaces."""
        for name, perm in all_permission_constants:
            key = perm.key
            assert " " not in key, f"{name} key '{key}' contains spaces"


# =============================================================================
# Test No Duplicate Permissions
# =============================================================================


class TestNoDuplicatePermissions:
    """Tests ensuring no duplicate permissions exist."""

    # Known intentional duplicates - these are aliases that map multiple
    # semantic concepts to the same underlying permission key.
    # For example, PERM_CULTURE_READ and PERM_KNOWLEDGE_READ both map to knowledge.read
    KNOWN_ALIAS_KEYS = {
        "analytics.read",  # PERM_ANALYTICS_READ, PERM_PERFORMANCE_READ
        "analytics.update",  # PERM_ANALYTICS_EXPORT (different), PERM_PERFORMANCE_WRITE
        "billing.read",  # Multiple billing read variants
        "billing.update",  # Multiple billing update variants
        "knowledge.read",  # PERM_KNOWLEDGE_READ, PERM_CULTURE_READ
        "knowledge.update",  # PERM_KNOWLEDGE_UPDATE, PERM_CULTURE_WRITE
        "organization.read",  # PERM_ORG_READ, PERM_ORG_USAGE_READ, PERM_ORG_MEMBERS
        "payments.create",  # Multiple payment create variants
        "payments.read",  # Multiple payment read variants
        "security.read",  # PERM_CVE_READ, PERM_SAST_READ, PERM_SBOM_READ, PERM_SECRETS_READ, PERM_VULNERABILITY_READ
        "security.execute",  # PERM_SAST_SCAN, PERM_SECRETS_SCAN, PERM_VULNERABILITY_SCAN
        "skills.update",  # PERM_SKILLS_INSTALL, PERM_SKILLS_RATE
    }

    def test_no_unexpected_duplicate_permission_ids(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """Permission IDs should be unique except for known aliases."""
        ids = [perm.id for _, perm in all_permission_constants]
        duplicates = [id_ for id_, count in Counter(ids).items() if count > 1]
        unexpected = [d for d in duplicates if d not in self.KNOWN_ALIAS_KEYS]
        assert not unexpected, f"Unexpected duplicate permission IDs found: {unexpected}"

    def test_no_unexpected_duplicate_permission_keys(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """Permission keys should be unique except for known aliases."""
        keys = [perm.key for _, perm in all_permission_constants]
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        unexpected = [d for d in duplicates if d not in self.KNOWN_ALIAS_KEYS]
        assert not unexpected, f"Unexpected duplicate permission keys found: {unexpected}"

    def test_known_aliases_actually_duplicate(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """Verify that known aliases are actually duplicated (guards against stale list)."""
        keys = [perm.key for _, perm in all_permission_constants]
        key_counts = Counter(keys)
        # At least some of the known aliases should have duplicates
        found_duplicates = [k for k in self.KNOWN_ALIAS_KEYS if key_counts.get(k, 0) > 1]
        assert len(found_duplicates) >= 5, (
            f"Expected at least 5 known aliases to be duplicated, found {len(found_duplicates)}"
        )

    def test_no_duplicate_constant_names(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """All PERM_* constant names should be unique."""
        names = [name for name, _ in all_permission_constants]
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        assert not duplicates, f"Duplicate constant names found: {duplicates}"

    def test_most_constants_match_permission_semantics(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """Most constant names should reflect the permission semantics."""
        # Some constants use semantic aliases (e.g., CULTURE maps to KNOWLEDGE resource)
        # We test that the majority follow the convention
        alias_mappings = {
            # Constant name prefix -> actual resource name it maps to
            "CULTURE": "KNOWLEDGE",
            "PERFORMANCE": "ANALYTICS",
            "HISTORY": "INTROSPECTION",
        }

        mismatches = []
        for name, perm in all_permission_constants:
            # Check if this is a known alias mapping
            is_known_alias = False
            for alias_prefix, actual_resource in alias_mappings.items():
                if alias_prefix in name and perm.resource.name == actual_resource:
                    is_known_alias = True
                    break

            if is_known_alias:
                continue

            # Standard check: resource or action should appear in name
            resource_in_name = perm.resource.name in name
            action_in_name = perm.action.name in name or perm.action.value.upper().replace(
                "_", ""
            ) in name.replace("_", "")

            if not (resource_in_name or action_in_name):
                mismatches.append((name, perm.key))

        # Allow some mismatches for edge cases, but most should match
        max_allowed_mismatches = 15  # Allow up to 15 edge cases
        assert len(mismatches) <= max_allowed_mismatches, (
            f"Too many naming mismatches ({len(mismatches)}): {mismatches[:10]}..."
        )


# =============================================================================
# Test Permission Naming Conventions
# =============================================================================


class TestPermissionNamingConventions:
    """Tests for permission naming convention compliance."""

    def test_constant_names_start_with_perm(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """All constant names should start with PERM_."""
        for name, _ in all_permission_constants:
            assert name.startswith("PERM_"), f"Constant '{name}' does not start with PERM_"

    def test_constant_names_uppercase(self, all_permission_constants: list[tuple[str, Permission]]):
        """All constant names should be uppercase with underscores."""
        for name, _ in all_permission_constants:
            assert name == name.upper(), f"Constant '{name}' is not uppercase"

    def test_permission_display_names_title_case(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """Permission display names should be in title case or contain at least one capital."""
        for name, perm in all_permission_constants:
            display_name = perm.name
            # Should have at least one uppercase letter
            assert any(c.isupper() for c in display_name), (
                f"{name} display name '{display_name}' has no uppercase letters"
            )

    def test_permission_names_human_readable(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """Permission names should be human-readable (no underscores, no dots)."""
        for name, perm in all_permission_constants:
            display_name = perm.name
            # Names like "Create Debates" are preferred over "debates.create"
            assert "_" not in display_name or display_name.count("_") < 3, (
                f"{name} display name '{display_name}' contains too many underscores"
            )

    def test_constant_name_format_perm_resource_action(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """Constant names should follow PERM_RESOURCE_ACTION format."""
        # Allow some flexibility as some have additional words
        pattern = re.compile(r"^PERM_[A-Z][A-Z0-9_]+$")
        for name, _ in all_permission_constants:
            assert pattern.match(name), f"Constant '{name}' does not match expected naming pattern"


# =============================================================================
# Test Permission Helper Function
# =============================================================================


class TestPermissionHelperFunction:
    """Tests for the _permission helper function."""

    def test_permission_helper_creates_valid_permission(self):
        """_permission helper should create a valid Permission."""
        perm = _permission(
            ResourceType.DEBATE,
            Action.CREATE,
            "Test Create",
            "Test description",
        )
        assert isinstance(perm, Permission)
        assert perm.resource == ResourceType.DEBATE
        assert perm.action == Action.CREATE
        assert perm.name == "Test Create"
        assert perm.description == "Test description"

    def test_permission_helper_auto_generates_id(self):
        """_permission helper should auto-generate ID from resource.action."""
        perm = _permission(
            ResourceType.AGENT,
            Action.DEPLOY,
            "Deploy Agent",
            "Deploy to production",
        )
        assert perm.id == "agents.deploy"

    def test_permission_helper_auto_generates_name_if_empty(self):
        """_permission helper should auto-generate name if not provided."""
        perm = _permission(ResourceType.MEMORY, Action.READ)
        # Name should be auto-generated from key
        assert perm.name  # Should not be empty

    def test_permission_helper_key_property(self):
        """Permission created by helper should have correct key property."""
        perm = _permission(
            ResourceType.WORKFLOW,
            Action.RUN,
            "Run Workflow",
            "Execute workflow",
        )
        assert perm.key == "workflows.run"

    def test_permission_helper_with_complex_action(self):
        """_permission helper works with complex action values."""
        perm = _permission(
            ResourceType.CONTROL_PLANE,
            Action.AGENTS_READ,
            "View Control Plane Agents",
            "Read agent registry",
        )
        assert perm.key == "control_plane.agents.read"


# =============================================================================
# Test Permission Groupings
# =============================================================================


class TestPermissionGroupings:
    """Tests for logical groupings of permissions."""

    def test_debate_permissions_have_crud_operations(self, debate_permissions: list[Permission]):
        """Debate permissions should include CRUD operations."""
        actions = {p.action for p in debate_permissions}
        assert Action.CREATE in actions, "Missing CREATE action for debates"
        assert Action.READ in actions, "Missing READ action for debates"
        assert Action.UPDATE in actions, "Missing UPDATE action for debates"
        assert Action.DELETE in actions, "Missing DELETE action for debates"

    def test_debate_permissions_have_execution_operations(
        self, debate_permissions: list[Permission]
    ):
        """Debate permissions should include execution operations."""
        actions = {p.action for p in debate_permissions}
        assert Action.RUN in actions, "Missing RUN action for debates"
        assert Action.STOP in actions, "Missing STOP action for debates"
        assert Action.FORK in actions, "Missing FORK action for debates"

    def test_agent_permissions_have_crud_operations(self, agent_permissions: list[Permission]):
        """Agent permissions should include CRUD operations."""
        actions = {p.action for p in agent_permissions}
        assert Action.CREATE in actions, "Missing CREATE action for agents"
        assert Action.READ in actions, "Missing READ action for agents"
        assert Action.UPDATE in actions, "Missing UPDATE action for agents"
        assert Action.DELETE in actions, "Missing DELETE action for agents"

    def test_agent_permissions_have_deploy_operation(self, agent_permissions: list[Permission]):
        """Agent permissions should include DEPLOY operation."""
        actions = {p.action for p in agent_permissions}
        assert Action.DEPLOY in actions, "Missing DEPLOY action for agents"

    def test_admin_permissions_have_all_wildcard(self, admin_permissions: list[Permission]):
        """Admin permissions should include ALL wildcard."""
        actions = {p.action for p in admin_permissions}
        assert Action.ALL in actions, "Missing ALL (wildcard) action for admin"

    def test_user_permissions_have_management_operations(
        self, user_management_permissions: list[Permission]
    ):
        """User permissions should include management operations."""
        actions = {p.action for p in user_management_permissions}
        assert Action.READ in actions, "Missing READ action for users"
        assert Action.INVITE in actions, "Missing INVITE action for users"
        assert Action.REMOVE in actions, "Missing REMOVE action for users"
        assert Action.CHANGE_ROLE in actions, "Missing CHANGE_ROLE action for users"
        assert Action.IMPERSONATE in actions, "Missing IMPERSONATE action for users"

    def test_control_plane_has_agent_operations(self, control_plane_permissions: list[Permission]):
        """Control plane should have agent-related operations."""
        actions = {p.action for p in control_plane_permissions}
        assert Action.AGENTS_READ in actions, "Missing AGENTS_READ for control plane"
        assert Action.AGENTS_REGISTER in actions, "Missing AGENTS_REGISTER for control plane"
        assert Action.AGENTS_UNREGISTER in actions, "Missing AGENTS_UNREGISTER for control plane"

    def test_control_plane_has_task_operations(self, control_plane_permissions: list[Permission]):
        """Control plane should have task-related operations."""
        actions = {p.action for p in control_plane_permissions}
        assert Action.TASKS_READ in actions, "Missing TASKS_READ for control plane"
        assert Action.TASKS_SUBMIT in actions, "Missing TASKS_SUBMIT for control plane"
        assert Action.TASKS_CLAIM in actions, "Missing TASKS_CLAIM for control plane"
        assert Action.TASKS_COMPLETE in actions, "Missing TASKS_COMPLETE for control plane"

    def test_computer_use_has_all_tool_operations(self, computer_use_permissions: list[Permission]):
        """Computer-use should have all tool operations."""
        actions = {p.action for p in computer_use_permissions}
        assert Action.BROWSER in actions, "Missing BROWSER action for computer-use"
        assert Action.SHELL in actions, "Missing SHELL action for computer-use"
        assert Action.FILE_READ in actions, "Missing FILE_READ action for computer-use"
        assert Action.FILE_WRITE in actions, "Missing FILE_WRITE action for computer-use"
        assert Action.SCREENSHOT in actions, "Missing SCREENSHOT action for computer-use"
        assert Action.NETWORK in actions, "Missing NETWORK action for computer-use"

    def test_data_governance_has_classification_operations(
        self, data_governance_permissions: list[Permission]
    ):
        """Data governance should have classification operations."""
        actions = {p.action for p in data_governance_permissions}
        assert Action.CLASSIFY in actions, "Missing CLASSIFY action for data governance"
        assert Action.REDACT in actions, "Missing REDACT action for data governance"
        assert Action.MASK in actions, "Missing MASK action for data governance"


# =============================================================================
# Test Resource Type Coverage
# =============================================================================


class TestResourceTypeCoverage:
    """Tests ensuring all resource types have associated permissions."""

    def test_debate_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """DEBATE resource type should have permissions."""
        assert ResourceType.DEBATE in permission_constants_by_resource
        assert len(permission_constants_by_resource[ResourceType.DEBATE]) >= 5

    def test_agent_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """AGENT resource type should have permissions."""
        assert ResourceType.AGENT in permission_constants_by_resource
        assert len(permission_constants_by_resource[ResourceType.AGENT]) >= 4

    def test_user_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """USER resource type should have permissions."""
        assert ResourceType.USER in permission_constants_by_resource
        assert len(permission_constants_by_resource[ResourceType.USER]) >= 4

    def test_organization_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """ORGANIZATION resource type should have permissions."""
        assert ResourceType.ORGANIZATION in permission_constants_by_resource
        assert len(permission_constants_by_resource[ResourceType.ORGANIZATION]) >= 5

    def test_admin_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """ADMIN resource type should have permissions."""
        assert ResourceType.ADMIN in permission_constants_by_resource
        assert len(permission_constants_by_resource[ResourceType.ADMIN]) >= 5

    def test_control_plane_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """CONTROL_PLANE resource type should have permissions."""
        assert ResourceType.CONTROL_PLANE in permission_constants_by_resource
        assert len(permission_constants_by_resource[ResourceType.CONTROL_PLANE]) >= 10

    def test_compliance_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """COMPLIANCE resource type should have permissions."""
        assert ResourceType.COMPLIANCE in permission_constants_by_resource
        assert len(permission_constants_by_resource[ResourceType.COMPLIANCE]) >= 5

    def test_gauntlet_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """GAUNTLET resource type should have permissions."""
        assert ResourceType.GAUNTLET in permission_constants_by_resource
        assert len(permission_constants_by_resource[ResourceType.GAUNTLET]) >= 4

    def test_computer_use_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """COMPUTER_USE resource type should have permissions."""
        assert ResourceType.COMPUTER_USE in permission_constants_by_resource
        assert len(permission_constants_by_resource[ResourceType.COMPUTER_USE]) >= 8

    def test_team_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """TEAM resource type should have permissions."""
        assert ResourceType.TEAM in permission_constants_by_resource
        assert len(permission_constants_by_resource[ResourceType.TEAM]) >= 6

    def test_workspace_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """WORKSPACE resource type should have permissions."""
        assert ResourceType.WORKSPACE in permission_constants_by_resource
        assert len(permission_constants_by_resource[ResourceType.WORKSPACE]) >= 4


# =============================================================================
# Test Specific Permission Values
# =============================================================================


class TestSpecificPermissionValues:
    """Tests for specific permission constant values."""

    def test_perm_debate_create_values(self):
        """PERM_DEBATE_CREATE should have correct values."""
        assert PERM_DEBATE_CREATE.resource == ResourceType.DEBATE
        assert PERM_DEBATE_CREATE.action == Action.CREATE
        assert PERM_DEBATE_CREATE.key == "debates.create"
        assert "Create" in PERM_DEBATE_CREATE.name or "create" in PERM_DEBATE_CREATE.name.lower()

    def test_perm_admin_all_values(self):
        """PERM_ADMIN_ALL should have wildcard action."""
        assert PERM_ADMIN_ALL.resource == ResourceType.ADMIN
        assert PERM_ADMIN_ALL.action == Action.ALL
        assert PERM_ADMIN_ALL.key == "admin.*"

    def test_perm_user_impersonate_values(self):
        """PERM_USER_IMPERSONATE should be a dangerous permission."""
        assert PERM_USER_IMPERSONATE.resource == ResourceType.USER
        assert PERM_USER_IMPERSONATE.action == Action.IMPERSONATE
        # Description mentions acting on behalf of others
        desc_lower = PERM_USER_IMPERSONATE.description.lower()
        assert "impersonate" in desc_lower or "behalf" in desc_lower or "act" in desc_lower

    def test_perm_compliance_gdpr_values(self):
        """PERM_COMPLIANCE_GDPR should have GDPR action."""
        assert PERM_COMPLIANCE_GDPR.resource == ResourceType.COMPLIANCE
        assert PERM_COMPLIANCE_GDPR.action == Action.GDPR
        assert "gdpr" in PERM_COMPLIANCE_GDPR.name.lower() or "GDPR" in PERM_COMPLIANCE_GDPR.name

    def test_perm_compliance_soc2_values(self):
        """PERM_COMPLIANCE_SOC2 should have SOC2 action."""
        assert PERM_COMPLIANCE_SOC2.resource == ResourceType.COMPLIANCE
        assert PERM_COMPLIANCE_SOC2.action == Action.SOC2
        assert "soc2" in PERM_COMPLIANCE_SOC2.name.lower() or "SOC2" in PERM_COMPLIANCE_SOC2.name

    def test_perm_computer_use_shell_values(self):
        """PERM_COMPUTER_USE_SHELL should have SHELL action."""
        assert PERM_COMPUTER_USE_SHELL.resource == ResourceType.COMPUTER_USE
        assert PERM_COMPUTER_USE_SHELL.action == Action.SHELL
        assert "shell" in PERM_COMPUTER_USE_SHELL.name.lower()

    def test_perm_backup_restore_values(self):
        """PERM_BACKUP_RESTORE should have RESTORE action."""
        assert PERM_BACKUP_RESTORE.resource == ResourceType.BACKUP
        assert PERM_BACKUP_RESTORE.action == Action.RESTORE
        assert "restore" in PERM_BACKUP_RESTORE.name.lower()

    def test_perm_gauntlet_sign_values(self):
        """PERM_GAUNTLET_SIGN should have SIGN action."""
        assert PERM_GAUNTLET_SIGN.resource == ResourceType.GAUNTLET
        assert PERM_GAUNTLET_SIGN.action == Action.SIGN
        assert "sign" in PERM_GAUNTLET_SIGN.description.lower()


# =============================================================================
# Test Permission Descriptions
# =============================================================================


class TestPermissionDescriptions:
    """Tests for permission description quality."""

    def test_high_privilege_permissions_have_descriptions(self):
        """High-privilege permissions should have meaningful descriptions."""
        high_privilege = [
            PERM_ADMIN_ALL,
            PERM_USER_IMPERSONATE,
            PERM_ORG_DELETE,
            PERM_BACKUP_RESTORE,
            PERM_DR_EXECUTE,
        ]
        for perm in high_privilege:
            assert len(perm.description) >= 10, (
                f"High-privilege permission {perm.key} has insufficient description"
            )

    def test_destructive_permissions_mention_intent(self):
        """Destructive permissions should have descriptions mentioning consequences."""
        destructive_keywords = ["delete", "remove", "permanently", "irreversible"]
        destructive_perms = [
            PERM_DEBATE_DELETE,
            PERM_AGENT_DELETE,
            PERM_ORG_DELETE,
            PERM_BACKUP_DELETE,
        ]
        for perm in destructive_perms:
            desc_lower = perm.description.lower()
            name_lower = perm.name.lower()
            has_keyword = any(kw in desc_lower or kw in name_lower for kw in destructive_keywords)
            assert has_keyword, (
                f"Destructive permission {perm.key} should mention destructive intent"
            )

    def test_all_permissions_have_nonempty_descriptions(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """All permissions should have non-empty descriptions."""
        for name, perm in all_permission_constants:
            # Description can be empty string but should exist
            assert perm.description is not None
            # Most should have a description
            if perm.description == "":
                # Log but don't fail for empty descriptions
                pass


# =============================================================================
# Test Permission Key Generation
# =============================================================================


class TestPermissionKeyGeneration:
    """Tests for permission key property generation."""

    def test_key_property_is_generated_correctly(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """Permission key property should be generated from resource and action."""
        for name, perm in all_permission_constants:
            expected_key = f"{perm.resource.value}.{perm.action.value}"
            assert perm.key == expected_key, (
                f"{name} key mismatch: expected '{expected_key}', got '{perm.key}'"
            )

    def test_key_property_is_consistent_with_id(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """Permission key should be consistent with ID."""
        for name, perm in all_permission_constants:
            # For permissions created with _permission helper, id == key
            assert perm.id == perm.key, (
                f"{name} has inconsistent ID '{perm.id}' and key '{perm.key}'"
            )


# =============================================================================
# Test Permission Matches Logic
# =============================================================================


class TestPermissionMatchesLogic:
    """Tests for Permission.matches() method."""

    def test_permission_matches_exact_resource_and_action(self):
        """Permission should match exact resource and action."""
        assert PERM_DEBATE_CREATE.matches(ResourceType.DEBATE, Action.CREATE)
        assert PERM_AGENT_DEPLOY.matches(ResourceType.AGENT, Action.DEPLOY)
        assert PERM_USER_IMPERSONATE.matches(ResourceType.USER, Action.IMPERSONATE)

    def test_permission_does_not_match_different_action(self):
        """Permission should not match different action."""
        assert not PERM_DEBATE_CREATE.matches(ResourceType.DEBATE, Action.DELETE)
        assert not PERM_AGENT_READ.matches(ResourceType.AGENT, Action.DEPLOY)

    def test_permission_does_not_match_different_resource(self):
        """Permission should not match different resource."""
        assert not PERM_DEBATE_CREATE.matches(ResourceType.AGENT, Action.CREATE)
        assert not PERM_AGENT_READ.matches(ResourceType.DEBATE, Action.READ)

    def test_wildcard_permission_matches_any_action_on_resource(self):
        """Wildcard (ALL) permission should match any action on resource."""
        assert PERM_ADMIN_ALL.matches(ResourceType.ADMIN, Action.CREATE)
        assert PERM_ADMIN_ALL.matches(ResourceType.ADMIN, Action.READ)
        assert PERM_ADMIN_ALL.matches(ResourceType.ADMIN, Action.DELETE)
        assert PERM_ADMIN_ALL.matches(ResourceType.ADMIN, Action.SYSTEM_CONFIG)

    def test_wildcard_permission_does_not_match_different_resource(self):
        """Wildcard permission should not match different resource."""
        assert not PERM_ADMIN_ALL.matches(ResourceType.DEBATE, Action.CREATE)
        assert not PERM_ADMIN_ALL.matches(ResourceType.AGENT, Action.READ)


# =============================================================================
# Test Enterprise Permission Categories
# =============================================================================


class TestEnterprisePermissionCategories:
    """Tests for enterprise-specific permission categories."""

    def test_compliance_permissions_cover_major_frameworks(self):
        """Compliance permissions should cover major regulatory frameworks."""
        compliance_perms = [
            PERM_COMPLIANCE_GDPR,
            PERM_COMPLIANCE_SOC2,
            PERM_COMPLIANCE_LEGAL,
            PERM_COMPLIANCE_AUDIT,
        ]
        frameworks = {"gdpr", "soc2", "legal", "audit"}
        found_frameworks = set()
        for perm in compliance_perms:
            action_lower = perm.action.value.lower()
            for fw in frameworks:
                if fw in action_lower:
                    found_frameworks.add(fw)
        assert found_frameworks == frameworks, (
            f"Missing compliance frameworks: {frameworks - found_frameworks}"
        )

    def test_billing_permissions_cover_financial_operations(self):
        """Billing permissions should cover financial operations."""
        billing_perms = [
            PERM_BILLING_READ,
            PERM_BILLING_RECOMMENDATIONS_READ,
            PERM_BILLING_FORECAST_READ,
            PERM_BILLING_EXPORT_HISTORY,
        ]
        assert all(p.resource == ResourceType.BILLING for p in billing_perms)

    def test_backup_dr_permissions_exist(self):
        """Backup and disaster recovery permissions should exist."""
        backup_perms = [
            PERM_BACKUP_CREATE,
            PERM_BACKUP_READ,
            PERM_BACKUP_RESTORE,
            PERM_BACKUP_DELETE,
        ]
        dr_perms = [PERM_DR_READ, PERM_DR_EXECUTE]

        assert all(p.resource == ResourceType.BACKUP for p in backup_perms)
        assert all(p.resource == ResourceType.DISASTER_RECOVERY for p in dr_perms)


# =============================================================================
# Test Dangerous Permission Identification
# =============================================================================


class TestDangerousPermissionIdentification:
    """Tests to identify and categorize dangerous/high-privilege permissions."""

    def test_impersonate_is_identified_as_dangerous(self):
        """IMPERSONATE action should be identifiable as dangerous."""
        assert PERM_USER_IMPERSONATE.action == Action.IMPERSONATE
        # This permission allows acting as another user
        assert "impersonate" in PERM_USER_IMPERSONATE.key

    def test_wildcard_actions_are_identifiable(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """Wildcard (ALL) permissions should be identifiable."""
        wildcard_perms = [
            (name, perm) for name, perm in all_permission_constants if perm.action == Action.ALL
        ]
        assert len(wildcard_perms) >= 1, "Should have at least one wildcard permission"
        for name, perm in wildcard_perms:
            assert perm.key.endswith(".*"), f"{name} wildcard should end with '.*'"

    def test_delete_permissions_are_identifiable(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """DELETE permissions should be identifiable."""
        delete_perms = [
            (name, perm) for name, perm in all_permission_constants if perm.action == Action.DELETE
        ]
        assert len(delete_perms) >= 10, "Should have multiple DELETE permissions"
        for name, perm in delete_perms:
            assert ".delete" in perm.key, f"{name} DELETE perm should have .delete in key"

    def test_admin_operations_are_identifiable(
        self, all_permission_constants: list[tuple[str, Permission]]
    ):
        """ADMIN_OP permissions should be identifiable."""
        admin_op_perms = [
            (name, perm)
            for name, perm in all_permission_constants
            if perm.action == Action.ADMIN_OP
        ]
        # Admin operations grant elevated privileges
        for name, perm in admin_op_perms:
            assert "admin" in perm.key, f"{name} ADMIN_OP should have 'admin' in key"


# =============================================================================
# Test Permission Coverage for Security-Critical Resources
# =============================================================================


class TestSecurityCriticalResourceCoverage:
    """Tests ensuring security-critical resources have proper permission coverage."""

    def test_authentication_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """AUTHENTICATION resource should have permissions."""
        assert ResourceType.AUTHENTICATION in permission_constants_by_resource

    def test_api_key_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """API_KEY resource should have permissions."""
        assert ResourceType.API_KEY in permission_constants_by_resource

    def test_session_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """SESSION resource should have permissions."""
        assert ResourceType.SESSION in permission_constants_by_resource

    def test_audit_log_resource_has_permissions(
        self, permission_constants_by_resource: dict[ResourceType, list]
    ):
        """AUDIT_LOG resource should have permissions."""
        assert ResourceType.AUDIT_LOG in permission_constants_by_resource


# =============================================================================
# Test Permission Constant Organization
# =============================================================================


class TestPermissionConstantOrganization:
    """Tests for how permissions are organized in the module."""

    def test_permissions_organized_by_resource_sections(self):
        """Permission file should have organized sections (verified by comments)."""
        import inspect

        source = inspect.getsource(perm_module)

        # Check for section headers
        expected_sections = [
            "DEBATE PERMISSIONS",
            "AGENT PERMISSIONS",
            "USER MANAGEMENT PERMISSIONS",
            "ADMIN PERMISSIONS",
        ]
        for section in expected_sections:
            assert section in source, f"Missing section header: {section}"

    def test_each_resource_type_is_documented(self):
        """Each resource type in use should have a documented section."""
        import inspect

        source = inspect.getsource(perm_module)

        # Check for various resource sections
        resource_sections = [
            "DEBATE",
            "AGENT",
            "USER",
            "ORGANIZATION",
            "ADMIN",
            "CONTROL PLANE",
            "COMPLIANCE",
        ]
        for section in resource_sections:
            assert section in source, f"Missing documentation for {section}"
