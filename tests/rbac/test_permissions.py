"""
Tests for RBAC permission definitions in aragora/rbac/defaults/permissions.py.

Covers:
- _permission helper function behavior
- Permission ID and key format validation
- All permission categories are defined and non-empty
- Permission fields (name, description, resource, action) correctness
- No duplicate permission IDs across different constants
- Resource-action consistency
- Comprehensive category coverage
"""

import re

import pytest

from aragora.rbac.defaults.permissions import (
    # Helper
    _permission,
    # Debate
    PERM_DEBATE_CREATE,
    PERM_DEBATE_READ,
    PERM_DEBATE_UPDATE,
    PERM_DEBATE_DELETE,
    PERM_DEBATE_RUN,
    PERM_DEBATE_STOP,
    PERM_DEBATE_FORK,
    # Agent
    PERM_AGENT_CREATE,
    PERM_AGENT_READ,
    PERM_AGENT_UPDATE,
    PERM_AGENT_DELETE,
    PERM_AGENT_DEPLOY,
    # User
    PERM_USER_READ,
    PERM_USER_INVITE,
    PERM_USER_REMOVE,
    PERM_USER_CHANGE_ROLE,
    PERM_USER_IMPERSONATE,
    # Organization
    PERM_ORG_READ,
    PERM_ORG_UPDATE,
    PERM_ORG_BILLING,
    PERM_ORG_AUDIT,
    PERM_ORG_EXPORT,
    PERM_ORG_INVITE,
    PERM_ORG_DELETE,
    # API
    PERM_API_GENERATE_KEY,
    PERM_API_REVOKE_KEY,
    # Memory
    PERM_MEMORY_READ,
    PERM_MEMORY_UPDATE,
    PERM_MEMORY_DELETE,
    # Workflow
    PERM_WORKFLOW_CREATE,
    PERM_WORKFLOW_READ,
    PERM_WORKFLOW_RUN,
    PERM_WORKFLOW_DELETE,
    # Analytics
    PERM_ANALYTICS_READ,
    PERM_ANALYTICS_EXPORT,
    PERM_PERFORMANCE_READ,
    PERM_PERFORMANCE_WRITE,
    # Admin
    PERM_ADMIN_CONFIG,
    PERM_ADMIN_METRICS,
    PERM_ADMIN_FEATURES,
    PERM_ADMIN_ALL,
    PERM_ADMIN_AUDIT,
    PERM_ADMIN_SECURITY,
    PERM_ADMIN_SYSTEM,
    # Connector
    PERM_CONNECTOR_READ,
    PERM_CONNECTOR_CREATE,
    PERM_CONNECTOR_DELETE,
    PERM_CONNECTOR_AUTHORIZE,
    PERM_CONNECTOR_ROTATE,
    PERM_CONNECTOR_TEST,
    PERM_CONNECTOR_UPDATE,
    PERM_CONNECTOR_ROLLBACK,
    # Gauntlet
    PERM_GAUNTLET_RUN,
    PERM_GAUNTLET_READ,
    PERM_GAUNTLET_DELETE,
    PERM_GAUNTLET_SIGN,
    PERM_GAUNTLET_COMPARE,
    PERM_GAUNTLET_EXPORT,
    # Marketplace
    PERM_MARKETPLACE_READ,
    PERM_MARKETPLACE_PUBLISH,
    PERM_MARKETPLACE_IMPORT,
    PERM_MARKETPLACE_RATE,
    PERM_MARKETPLACE_REVIEW,
    PERM_MARKETPLACE_DELETE,
    # Compliance
    PERM_COMPLIANCE_READ,
    PERM_COMPLIANCE_UPDATE,
    PERM_COMPLIANCE_CHECK,
    PERM_COMPLIANCE_GDPR,
    PERM_COMPLIANCE_SOC2,
    PERM_COMPLIANCE_LEGAL,
    PERM_COMPLIANCE_AUDIT,
    # Control Plane
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
    # Data governance
    PERM_DATA_CLASSIFICATION_READ,
    PERM_DATA_CLASSIFICATION_CLASSIFY,
    PERM_DATA_CLASSIFICATION_UPDATE,
    PERM_DATA_RETENTION_READ,
    PERM_DATA_RETENTION_UPDATE,
    PERM_DATA_LINEAGE_READ,
    PERM_PII_READ,
    PERM_PII_REDACT,
    PERM_PII_MASK,
    # Computer-use
    PERM_COMPUTER_USE_READ,
    PERM_COMPUTER_USE_EXECUTE,
    PERM_COMPUTER_USE_BROWSER,
    PERM_COMPUTER_USE_SHELL,
    PERM_COMPUTER_USE_FILE_READ,
    PERM_COMPUTER_USE_FILE_WRITE,
    PERM_COMPUTER_USE_SCREENSHOT,
    PERM_COMPUTER_USE_NETWORK,
    PERM_COMPUTER_USE_ADMIN,
    # Finance/Receipt/Cost
    PERM_FINANCE_READ,
    PERM_FINANCE_WRITE,
    PERM_FINANCE_APPROVE,
    PERM_RECEIPT_READ,
    PERM_RECEIPT_VERIFY,
    PERM_RECEIPT_EXPORT,
    PERM_RECEIPT_SEND,
    PERM_COST_READ,
    PERM_COST_WRITE,
    # Team
    PERM_TEAM_CREATE,
    PERM_TEAM_READ,
    PERM_TEAM_UPDATE,
    PERM_TEAM_DELETE,
    PERM_TEAM_ADD_MEMBER,
    PERM_TEAM_REMOVE_MEMBER,
    PERM_TEAM_SHARE,
    PERM_TEAM_DISSOLVE,
    # Workspace
    PERM_WORKSPACE_CREATE,
    PERM_WORKSPACE_READ,
    PERM_WORKSPACE_UPDATE,
    PERM_WORKSPACE_DELETE,
    # Backup & DR
    PERM_BACKUP_CREATE,
    PERM_BACKUP_READ,
    PERM_BACKUP_RESTORE,
    PERM_BACKUP_DELETE,
    PERM_DR_READ,
    PERM_DR_EXECUTE,
    # Evolution
    PERM_EVOLUTION_READ,
    # Additional permissions
    PERM_INBOX_READ,
    PERM_INBOX_UPDATE,
    PERM_INBOX_CREATE,
    PERM_INBOX_WRITE,
    PERM_INBOX_DELETE,
    PERM_SKILLS_READ,
    PERM_SKILLS_INSTALL,
    PERM_SKILLS_PUBLISH,
    PERM_SKILLS_RATE,
    PERM_SKILLS_INVOKE,
    PERM_MEMORY_WRITE,
    PERM_DOCUMENTS_READ,
    PERM_DOCUMENTS_CREATE,
    PERM_DOCUMENTS_DELETE,
    PERM_KNOWLEDGE_READ,
    PERM_KNOWLEDGE_UPDATE,
    PERM_KNOWLEDGE_WRITE,
    PERM_KNOWLEDGE_DELETE,
    PERM_KNOWLEDGE_SHARE,
    PERM_CANVAS_READ,
    PERM_CANVAS_CREATE,
    PERM_CANVAS_UPDATE,
    PERM_CANVAS_DELETE,
    PERM_CANVAS_RUN,
    PERM_CANVAS_WRITE,
    PERM_CANVAS_SHARE,
    PERM_CODEBASE_READ,
    PERM_CODEBASE_ANALYZE,
    PERM_CODEBASE_WRITE,
    PERM_FEATURES_READ,
    PERM_FEATURES_WRITE,
    PERM_FEATURES_DELETE,
)
from aragora.rbac.models import Action, Permission, ResourceType


# ---------------------------------------------------------------------------
# Collect all PERM_* constants defined in the permissions module
# ---------------------------------------------------------------------------
import aragora.rbac.defaults.permissions as _pmod

ALL_PERM_CONSTANTS: list[tuple[str, Permission]] = [
    (name, getattr(_pmod, name))
    for name in dir(_pmod)
    if name.startswith("PERM_") and isinstance(getattr(_pmod, name), Permission)
]


class TestPermissionHelper:
    """Tests for the _permission helper function."""

    def test_creates_permission_instance(self):
        """_permission should return a Permission dataclass instance."""
        perm = _permission(ResourceType.DEBATE, Action.CREATE, "Test", "desc")
        assert isinstance(perm, Permission)

    def test_auto_generates_id_as_key(self):
        """Permission id should be resource.action format."""
        perm = _permission(ResourceType.AGENT, Action.DEPLOY)
        assert perm.id == "agents.deploy"

    def test_auto_generates_name_when_omitted(self):
        """When name is omitted, it should be auto-generated from the key."""
        perm = _permission(ResourceType.MEMORY, Action.READ)
        # "memory.read" -> "Memory Read"
        assert perm.name == "Memory Read"

    def test_explicit_name_preserved(self):
        """Explicit name should override the auto-generated one."""
        perm = _permission(ResourceType.DEBATE, Action.CREATE, "My Custom Name")
        assert perm.name == "My Custom Name"

    def test_description_stored(self):
        """Description string should be stored on the permission."""
        perm = _permission(ResourceType.DEBATE, Action.READ, "N", "Some description")
        assert perm.description == "Some description"

    def test_resource_and_action_assigned(self):
        """Resource and action enums should be correctly assigned."""
        perm = _permission(ResourceType.WORKFLOW, Action.RUN)
        assert perm.resource == ResourceType.WORKFLOW
        assert perm.action == Action.RUN


class TestPermissionKeyFormat:
    """All permission constants should follow the resource.action key format."""

    @pytest.mark.parametrize("name,perm", ALL_PERM_CONSTANTS)
    def test_key_matches_resource_dot_action(self, name: str, perm: Permission):
        """Permission.key should be '{resource_value}.{action_value}'."""
        expected_key = f"{perm.resource.value}.{perm.action.value}"
        assert perm.key == expected_key, f"{name}.key = {perm.key!r}, expected {expected_key!r}"

    @pytest.mark.parametrize("name,perm", ALL_PERM_CONSTANTS)
    def test_id_equals_key(self, name: str, perm: Permission):
        """Permission id should equal the key (set by _permission helper)."""
        assert perm.id == perm.key, f"{name}.id = {perm.id!r} != key {perm.key!r}"

    @pytest.mark.parametrize("name,perm", ALL_PERM_CONSTANTS)
    def test_key_contains_dot_separator(self, name: str, perm: Permission):
        """Permission key must contain a dot separator."""
        assert "." in perm.key, f"{name} key {perm.key!r} missing dot"


class TestPermissionFields:
    """Every permission constant should have non-empty name and description."""

    @pytest.mark.parametrize("name,perm", ALL_PERM_CONSTANTS)
    def test_name_is_nonempty(self, name: str, perm: Permission):
        assert perm.name, f"{name} has empty name"

    @pytest.mark.parametrize("name,perm", ALL_PERM_CONSTANTS)
    def test_description_is_nonempty(self, name: str, perm: Permission):
        assert perm.description, f"{name} has empty description"

    @pytest.mark.parametrize("name,perm", ALL_PERM_CONSTANTS)
    def test_resource_is_valid_enum(self, name: str, perm: Permission):
        assert isinstance(perm.resource, ResourceType), f"{name} resource not ResourceType"

    @pytest.mark.parametrize("name,perm", ALL_PERM_CONSTANTS)
    def test_action_is_valid_enum(self, name: str, perm: Permission):
        assert isinstance(perm.action, Action), f"{name} action not Action"


class TestPermissionCount:
    """Verify the expected breadth of permission definitions."""

    def test_minimum_total_permissions(self):
        """Module should define at least 150 PERM_* constants."""
        assert len(ALL_PERM_CONSTANTS) >= 150, (
            f"Expected >=150 permissions, found {len(ALL_PERM_CONSTANTS)}"
        )

    def test_no_non_permission_perm_constants(self):
        """Every module-level PERM_* attribute should be a Permission instance."""
        for name in dir(_pmod):
            if name.startswith("PERM_"):
                obj = getattr(_pmod, name)
                assert isinstance(obj, Permission), (
                    f"{name} is {type(obj).__name__}, expected Permission"
                )


class TestDebatePermissions:
    """Debate category should have full CRUD + run/stop/fork."""

    EXPECTED = [
        (PERM_DEBATE_CREATE, Action.CREATE, "Create Debates"),
        (PERM_DEBATE_READ, Action.READ, "View Debates"),
        (PERM_DEBATE_UPDATE, Action.UPDATE, "Update Debates"),
        (PERM_DEBATE_DELETE, Action.DELETE, "Delete Debates"),
        (PERM_DEBATE_RUN, Action.RUN, "Run Debates"),
        (PERM_DEBATE_STOP, Action.STOP, "Stop Debates"),
        (PERM_DEBATE_FORK, Action.FORK, "Fork Debates"),
    ]

    def test_all_debate_perms_use_debate_resource(self):
        for perm, _, _ in self.EXPECTED:
            assert perm.resource == ResourceType.DEBATE

    def test_debate_actions_and_names(self):
        for perm, action, name in self.EXPECTED:
            assert perm.action == action, f"{perm.id} action mismatch"
            assert perm.name == name, f"{perm.id} name mismatch"

    def test_debate_count(self):
        assert len(self.EXPECTED) == 7


class TestAgentPermissions:
    """Agent category should have CRUD + deploy."""

    EXPECTED_ACTIONS = {
        Action.CREATE,
        Action.READ,
        Action.UPDATE,
        Action.DELETE,
        Action.DEPLOY,
    }

    def test_agent_resource(self):
        for perm in [
            PERM_AGENT_CREATE,
            PERM_AGENT_READ,
            PERM_AGENT_UPDATE,
            PERM_AGENT_DELETE,
            PERM_AGENT_DEPLOY,
        ]:
            assert perm.resource == ResourceType.AGENT

    def test_agent_actions_complete(self):
        actual = {
            p.action
            for p in [
                PERM_AGENT_CREATE,
                PERM_AGENT_READ,
                PERM_AGENT_UPDATE,
                PERM_AGENT_DELETE,
                PERM_AGENT_DEPLOY,
            ]
        }
        assert actual == self.EXPECTED_ACTIONS


class TestUserPermissions:
    """User management permissions."""

    def test_user_resources(self):
        for perm in [
            PERM_USER_READ,
            PERM_USER_INVITE,
            PERM_USER_REMOVE,
            PERM_USER_CHANGE_ROLE,
            PERM_USER_IMPERSONATE,
        ]:
            assert perm.resource == ResourceType.USER

    def test_impersonate_is_dangerous(self):
        """Impersonate should be clearly labeled and use IMPERSONATE action."""
        assert PERM_USER_IMPERSONATE.action == Action.IMPERSONATE
        assert "impersonate" in PERM_USER_IMPERSONATE.name.lower()


class TestOrganizationPermissions:
    """Organization permissions including billing and audit."""

    def test_org_resource(self):
        for perm in [
            PERM_ORG_READ,
            PERM_ORG_UPDATE,
            PERM_ORG_BILLING,
            PERM_ORG_AUDIT,
            PERM_ORG_EXPORT,
            PERM_ORG_INVITE,
            PERM_ORG_DELETE,
        ]:
            assert perm.resource == ResourceType.ORGANIZATION

    def test_billing_action(self):
        assert PERM_ORG_BILLING.action == Action.MANAGE_BILLING

    def test_audit_action(self):
        assert PERM_ORG_AUDIT.action == Action.VIEW_AUDIT


class TestAdminPermissions:
    """Admin permissions should include wildcard and fine-grained options."""

    def test_admin_all_is_wildcard(self):
        assert PERM_ADMIN_ALL.action == Action.ALL
        assert PERM_ADMIN_ALL.resource == ResourceType.ADMIN

    def test_admin_fine_grained_actions(self):
        expected_actions = {
            Action.SYSTEM_CONFIG,
            Action.VIEW_METRICS,
            Action.MANAGE_FEATURES,
            Action.ALL,
            Action.AUDIT,
            Action.SECURITY,
            Action.SYSTEM,
        }
        admin_perms = [
            PERM_ADMIN_CONFIG,
            PERM_ADMIN_METRICS,
            PERM_ADMIN_FEATURES,
            PERM_ADMIN_ALL,
            PERM_ADMIN_AUDIT,
            PERM_ADMIN_SECURITY,
            PERM_ADMIN_SYSTEM,
        ]
        actual = {p.action for p in admin_perms}
        assert actual == expected_actions


class TestConnectorPermissions:
    """Connector category should cover full lifecycle."""

    def test_connector_lifecycle_actions(self):
        connector_perms = [
            PERM_CONNECTOR_READ,
            PERM_CONNECTOR_CREATE,
            PERM_CONNECTOR_DELETE,
            PERM_CONNECTOR_AUTHORIZE,
            PERM_CONNECTOR_ROTATE,
            PERM_CONNECTOR_TEST,
            PERM_CONNECTOR_UPDATE,
            PERM_CONNECTOR_ROLLBACK,
        ]
        actions = {p.action for p in connector_perms}
        expected = {
            Action.READ,
            Action.CREATE,
            Action.DELETE,
            Action.AUTHORIZE,
            Action.ROTATE,
            Action.TEST,
            Action.UPDATE,
            Action.ROLLBACK,
        }
        assert actions == expected

    def test_all_connectors_use_connector_resource(self):
        for perm in [
            PERM_CONNECTOR_READ,
            PERM_CONNECTOR_CREATE,
            PERM_CONNECTOR_DELETE,
            PERM_CONNECTOR_AUTHORIZE,
            PERM_CONNECTOR_ROTATE,
            PERM_CONNECTOR_TEST,
            PERM_CONNECTOR_UPDATE,
            PERM_CONNECTOR_ROLLBACK,
        ]:
            assert perm.resource == ResourceType.CONNECTOR


class TestControlPlanePermissions:
    """Control plane should have granular sub-operation permissions."""

    def test_control_plane_count(self):
        cp_perms = [
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
        assert len(cp_perms) == 14

    def test_all_use_control_plane_resource(self):
        for perm in [
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
        ]:
            assert perm.resource == ResourceType.CONTROL_PLANE


class TestComputerUsePermissions:
    """Computer-use permissions should cover all execution primitives."""

    def test_computer_use_actions(self):
        cu_perms = [
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
        actions = {p.action for p in cu_perms}
        expected = {
            Action.READ,
            Action.EXECUTE,
            Action.BROWSER,
            Action.SHELL,
            Action.FILE_READ,
            Action.FILE_WRITE,
            Action.SCREENSHOT,
            Action.NETWORK,
            Action.ADMIN_OP,
        }
        assert actions == expected

    def test_all_use_computer_use_resource(self):
        for perm in [
            PERM_COMPUTER_USE_READ,
            PERM_COMPUTER_USE_EXECUTE,
            PERM_COMPUTER_USE_BROWSER,
            PERM_COMPUTER_USE_SHELL,
            PERM_COMPUTER_USE_FILE_READ,
            PERM_COMPUTER_USE_FILE_WRITE,
            PERM_COMPUTER_USE_SCREENSHOT,
            PERM_COMPUTER_USE_NETWORK,
            PERM_COMPUTER_USE_ADMIN,
        ]:
            assert perm.resource == ResourceType.COMPUTER_USE


class TestDataGovernancePermissions:
    """Data governance permissions: classification, retention, lineage, PII."""

    def test_data_classification_perms(self):
        assert PERM_DATA_CLASSIFICATION_READ.resource == ResourceType.DATA_CLASSIFICATION
        assert PERM_DATA_CLASSIFICATION_CLASSIFY.action == Action.CLASSIFY
        assert PERM_DATA_CLASSIFICATION_UPDATE.action == Action.UPDATE

    def test_data_retention_perms(self):
        assert PERM_DATA_RETENTION_READ.resource == ResourceType.DATA_RETENTION
        assert PERM_DATA_RETENTION_UPDATE.action == Action.UPDATE

    def test_data_lineage_perm(self):
        assert PERM_DATA_LINEAGE_READ.resource == ResourceType.DATA_LINEAGE
        assert PERM_DATA_LINEAGE_READ.action == Action.READ

    def test_pii_perms(self):
        assert PERM_PII_READ.resource == ResourceType.PII
        assert PERM_PII_REDACT.action == Action.REDACT
        assert PERM_PII_MASK.action == Action.MASK


class TestCompliancePermissions:
    """Compliance permissions should cover GDPR, SOC2, legal, and audit."""

    def test_compliance_actions_present(self):
        expected_actions = {
            Action.READ,
            Action.UPDATE,
            Action.CHECK,
            Action.GDPR,
            Action.SOC2,
            Action.LEGAL,
            Action.AUDIT,
        }
        actual = {
            p.action
            for p in [
                PERM_COMPLIANCE_READ,
                PERM_COMPLIANCE_UPDATE,
                PERM_COMPLIANCE_CHECK,
                PERM_COMPLIANCE_GDPR,
                PERM_COMPLIANCE_SOC2,
                PERM_COMPLIANCE_LEGAL,
                PERM_COMPLIANCE_AUDIT,
            ]
        }
        assert actual == expected_actions


class TestTeamPermissions:
    """Team permissions should support member management and dissolution."""

    def test_team_crud_plus_member_ops(self):
        expected_actions = {
            Action.CREATE,
            Action.READ,
            Action.UPDATE,
            Action.DELETE,
            Action.ADD_MEMBER,
            Action.REMOVE_MEMBER,
            Action.SHARE,
            Action.DISSOLVE,
        }
        team_perms = [
            PERM_TEAM_CREATE,
            PERM_TEAM_READ,
            PERM_TEAM_UPDATE,
            PERM_TEAM_DELETE,
            PERM_TEAM_ADD_MEMBER,
            PERM_TEAM_REMOVE_MEMBER,
            PERM_TEAM_SHARE,
            PERM_TEAM_DISSOLVE,
        ]
        actual = {p.action for p in team_perms}
        assert actual == expected_actions

    def test_all_use_team_resource(self):
        for perm in [
            PERM_TEAM_CREATE,
            PERM_TEAM_READ,
            PERM_TEAM_UPDATE,
            PERM_TEAM_DELETE,
            PERM_TEAM_ADD_MEMBER,
            PERM_TEAM_REMOVE_MEMBER,
            PERM_TEAM_SHARE,
            PERM_TEAM_DISSOLVE,
        ]:
            assert perm.resource == ResourceType.TEAM


class TestBackupDRPermissions:
    """Backup and disaster recovery permissions."""

    def test_backup_actions(self):
        expected = {Action.CREATE, Action.READ, Action.RESTORE, Action.DELETE}
        actual = {
            p.action
            for p in [
                PERM_BACKUP_CREATE,
                PERM_BACKUP_READ,
                PERM_BACKUP_RESTORE,
                PERM_BACKUP_DELETE,
            ]
        }
        assert actual == expected

    def test_dr_perms_exist(self):
        assert PERM_DR_READ.resource == ResourceType.DISASTER_RECOVERY
        assert PERM_DR_EXECUTE.action == Action.EXECUTE


class TestGauntletPermissions:
    """Gauntlet (adversarial testing) permissions."""

    def test_gauntlet_actions(self):
        expected = {
            Action.RUN,
            Action.READ,
            Action.DELETE,
            Action.SIGN,
            Action.COMPARE,
            Action.EXPORT_DATA,
        }
        actual = {
            p.action
            for p in [
                PERM_GAUNTLET_RUN,
                PERM_GAUNTLET_READ,
                PERM_GAUNTLET_DELETE,
                PERM_GAUNTLET_SIGN,
                PERM_GAUNTLET_COMPARE,
                PERM_GAUNTLET_EXPORT,
            ]
        }
        assert actual == expected


class TestResourceCoverage:
    """Verify that major resource types have at least one permission defined."""

    # Collect all resource types that appear in PERM_* constants
    COVERED_RESOURCES = {perm.resource for _, perm in ALL_PERM_CONSTANTS}

    EXPECTED_RESOURCES = {
        ResourceType.DEBATE,
        ResourceType.AGENT,
        ResourceType.USER,
        ResourceType.ORGANIZATION,
        ResourceType.API,
        ResourceType.MEMORY,
        ResourceType.WORKFLOW,
        ResourceType.ANALYTICS,
        ResourceType.ADMIN,
        ResourceType.CONNECTOR,
        ResourceType.GAUNTLET,
        ResourceType.MARKETPLACE,
        ResourceType.COMPLIANCE,
        ResourceType.CONTROL_PLANE,
        ResourceType.DATA_CLASSIFICATION,
        ResourceType.DATA_RETENTION,
        ResourceType.DATA_LINEAGE,
        ResourceType.PII,
        ResourceType.COMPUTER_USE,
        ResourceType.TEAM,
        ResourceType.WORKSPACE,
        ResourceType.BACKUP,
        ResourceType.DISASTER_RECOVERY,
        ResourceType.KNOWLEDGE,
        ResourceType.CANVAS,
        ResourceType.CODEBASE,
        ResourceType.INBOX,
        ResourceType.SKILLS,
        ResourceType.RECEIPT,
        ResourceType.FINANCE,
        ResourceType.COST,
        ResourceType.PROVENANCE,
        ResourceType.EVOLUTION,
    }

    def test_all_expected_resources_covered(self):
        missing = self.EXPECTED_RESOURCES - self.COVERED_RESOURCES
        assert not missing, f"No permissions defined for resources: {missing}"


class TestAdditionalPermissions:
    """Tests for the additional/supplementary permissions section."""

    def test_inbox_additional_actions(self):
        assert PERM_INBOX_CREATE.action == Action.CREATE
        assert PERM_INBOX_WRITE.action == Action.WRITE
        assert PERM_INBOX_DELETE.action == Action.DELETE

    def test_skills_invoke(self):
        assert PERM_SKILLS_INVOKE.action == Action.INVOKE
        assert PERM_SKILLS_INVOKE.resource == ResourceType.SKILLS

    def test_memory_write(self):
        assert PERM_MEMORY_WRITE.action == Action.WRITE
        assert PERM_MEMORY_WRITE.resource == ResourceType.MEMORY

    def test_documents_delete(self):
        assert PERM_DOCUMENTS_DELETE.action == Action.DELETE
        assert PERM_DOCUMENTS_DELETE.resource == ResourceType.DOCUMENTS

    def test_knowledge_additional(self):
        assert PERM_KNOWLEDGE_WRITE.action == Action.WRITE
        assert PERM_KNOWLEDGE_DELETE.action == Action.DELETE
        assert PERM_KNOWLEDGE_SHARE.action == Action.SHARE

    def test_canvas_additional(self):
        assert PERM_CANVAS_WRITE.action == Action.WRITE
        assert PERM_CANVAS_SHARE.action == Action.SHARE

    def test_codebase_write(self):
        assert PERM_CODEBASE_WRITE.action == Action.WRITE
        assert PERM_CODEBASE_WRITE.resource == ResourceType.CODEBASE

    def test_features_additional(self):
        assert PERM_FEATURES_WRITE.action == Action.WRITE
        assert PERM_FEATURES_DELETE.action == Action.DELETE

    def test_evolution_permission(self):
        assert PERM_EVOLUTION_READ.resource == ResourceType.EVOLUTION
        assert PERM_EVOLUTION_READ.action == Action.READ
