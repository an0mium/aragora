"""Tests for Secure Gateway RBAC permissions."""

from __future__ import annotations

import pytest

from aragora.rbac.defaults.permissions import (
    PERM_GATEWAY_AGENT_CREATE,
    PERM_GATEWAY_AGENT_DELETE,
    PERM_GATEWAY_AGENT_READ,
    PERM_GATEWAY_CREDENTIAL_CREATE,
    PERM_GATEWAY_CREDENTIAL_DELETE,
    PERM_GATEWAY_CREDENTIAL_READ,
    PERM_GATEWAY_CREDENTIAL_ROTATE,
    PERM_GATEWAY_HEALTH,
    PERM_GATEWAY_HYBRID_DEBATE,
)
from aragora.rbac.defaults.registry import SYSTEM_PERMISSIONS
from aragora.rbac.defaults.roles import ROLE_ADMIN, ROLE_MEMBER, ROLE_OWNER, ROLE_VIEWER
from aragora.rbac.models import Permission


# --------------------------------------------------------------------------
# Permission constants exist and are Permission instances
# --------------------------------------------------------------------------

ALL_GATEWAY_SECURE_PERMS = [
    PERM_GATEWAY_AGENT_CREATE,
    PERM_GATEWAY_AGENT_READ,
    PERM_GATEWAY_AGENT_DELETE,
    PERM_GATEWAY_CREDENTIAL_CREATE,
    PERM_GATEWAY_CREDENTIAL_READ,
    PERM_GATEWAY_CREDENTIAL_DELETE,
    PERM_GATEWAY_CREDENTIAL_ROTATE,
    PERM_GATEWAY_HYBRID_DEBATE,
    PERM_GATEWAY_HEALTH,
]


@pytest.mark.parametrize("perm", ALL_GATEWAY_SECURE_PERMS, ids=lambda p: p.name)
def test_permission_is_permission_instance(perm: Permission) -> None:
    """Each new gateway permission should be a Permission dataclass."""
    assert isinstance(perm, Permission)


@pytest.mark.parametrize("perm", ALL_GATEWAY_SECURE_PERMS, ids=lambda p: p.name)
def test_permission_key_is_string(perm: Permission) -> None:
    """Permission keys must be strings."""
    assert isinstance(perm.key, str)
    assert len(perm.key) > 0


@pytest.mark.parametrize("perm", ALL_GATEWAY_SECURE_PERMS, ids=lambda p: p.name)
def test_permission_key_starts_with_gateway(perm: Permission) -> None:
    """All secure gateway permissions should be under the gateway resource."""
    assert perm.key.startswith("gateway."), (
        f"Expected key starting with 'gateway.', got '{perm.key}'"
    )


# --------------------------------------------------------------------------
# Permissions are registered in SYSTEM_PERMISSIONS
# --------------------------------------------------------------------------


@pytest.mark.parametrize("perm", ALL_GATEWAY_SECURE_PERMS, ids=lambda p: p.name)
def test_permission_registered_in_system_permissions(perm: Permission) -> None:
    """Every new gateway permission must appear in the SYSTEM_PERMISSIONS registry."""
    assert perm.key in SYSTEM_PERMISSIONS, (
        f"Permission '{perm.key}' not found in SYSTEM_PERMISSIONS"
    )


# --------------------------------------------------------------------------
# Admin role has ALL new gateway permissions
# --------------------------------------------------------------------------


def test_admin_has_all_gateway_permissions() -> None:
    """The admin role must have every secure gateway permission."""
    for perm in ALL_GATEWAY_SECURE_PERMS:
        assert perm.key in ROLE_ADMIN.permissions, f"Admin role missing permission '{perm.key}'"


# --------------------------------------------------------------------------
# Owner role has ALL new gateway permissions (owner gets everything)
# --------------------------------------------------------------------------


def test_owner_has_all_gateway_permissions() -> None:
    """The owner role has all system permissions, including gateway."""
    for perm in ALL_GATEWAY_SECURE_PERMS:
        assert perm.key in ROLE_OWNER.permissions, f"Owner role missing permission '{perm.key}'"


# --------------------------------------------------------------------------
# Member (operator) role has expected gateway permissions
# --------------------------------------------------------------------------

MEMBER_EXPECTED_PERMS = [
    PERM_GATEWAY_AGENT_READ,
    PERM_GATEWAY_CREDENTIAL_READ,
    PERM_GATEWAY_HEALTH,
    PERM_GATEWAY_HYBRID_DEBATE,
]

MEMBER_UNEXPECTED_PERMS = [
    PERM_GATEWAY_AGENT_CREATE,
    PERM_GATEWAY_AGENT_DELETE,
    PERM_GATEWAY_CREDENTIAL_CREATE,
    PERM_GATEWAY_CREDENTIAL_DELETE,
    PERM_GATEWAY_CREDENTIAL_ROTATE,
]


def test_member_has_expected_gateway_permissions() -> None:
    """The member role should have read/health/hybrid_debate gateway permissions."""
    for perm in MEMBER_EXPECTED_PERMS:
        assert perm.key in ROLE_MEMBER.permissions, (
            f"Member role missing expected permission '{perm.key}'"
        )


def test_member_does_not_have_write_gateway_permissions() -> None:
    """The member role should NOT have create/delete/rotate gateway permissions."""
    for perm in MEMBER_UNEXPECTED_PERMS:
        assert perm.key not in ROLE_MEMBER.permissions, (
            f"Member role should not have permission '{perm.key}'"
        )


# --------------------------------------------------------------------------
# Viewer role has only health permission
# --------------------------------------------------------------------------


def test_viewer_has_gateway_health_permission() -> None:
    """The viewer role should have the gateway health permission."""
    assert PERM_GATEWAY_HEALTH.key in ROLE_VIEWER.permissions


def test_viewer_has_only_health_gateway_permission() -> None:
    """The viewer role should not have any gateway permission other than health."""
    gateway_perms_in_viewer = [
        perm for perm in ALL_GATEWAY_SECURE_PERMS if perm.key in ROLE_VIEWER.permissions
    ]
    assert len(gateway_perms_in_viewer) == 1
    assert gateway_perms_in_viewer[0] is PERM_GATEWAY_HEALTH
