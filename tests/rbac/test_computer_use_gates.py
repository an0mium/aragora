"""
Tests for Computer-Use RBAC Gates.

Tests cover:
- Computer-use permission definitions
- Permission key format
- Role assignments for computer-use
- Admin vs limited access
"""

from __future__ import annotations

import pytest

from aragora.rbac.defaults import (
    PERM_COMPUTER_USE_ADMIN,
    PERM_COMPUTER_USE_BROWSER,
    PERM_COMPUTER_USE_EXECUTE,
    PERM_COMPUTER_USE_FILE_READ,
    PERM_COMPUTER_USE_FILE_WRITE,
    PERM_COMPUTER_USE_NETWORK,
    PERM_COMPUTER_USE_READ,
    PERM_COMPUTER_USE_SCREENSHOT,
    PERM_COMPUTER_USE_SHELL,
    ROLE_ADMIN,
    ROLE_OWNER,
    ROLE_MEMBER,
    ROLE_VIEWER,
    SYSTEM_PERMISSIONS,
    get_permission,
    get_role_permissions,
)
from aragora.rbac.models import Action, ResourceType


# =============================================================================
# Permission Definition Tests
# =============================================================================


class TestComputerUsePermissions:
    """Tests for computer-use permission definitions."""

    def test_permission_keys_format(self):
        """Verify permission keys follow resource.action format."""
        perms = [
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

        for perm in perms:
            assert perm.key.startswith("computer_use.")
            assert perm.resource == ResourceType.COMPUTER_USE

    def test_permissions_in_system_registry(self):
        """Verify all computer-use permissions are in the system registry."""
        expected_keys = [
            "computer_use.read",
            "computer_use.execute",
            "computer_use.browser",
            "computer_use.shell",
            "computer_use.file_read",
            "computer_use.file_write",
            "computer_use.screenshot",
            "computer_use.network",
            "computer_use.admin",
        ]

        for key in expected_keys:
            assert key in SYSTEM_PERMISSIONS, f"Missing permission: {key}"
            perm = SYSTEM_PERMISSIONS[key]
            assert perm.resource == ResourceType.COMPUTER_USE

    def test_permission_actions(self):
        """Verify permissions have correct actions."""
        assert PERM_COMPUTER_USE_READ.action == Action.READ
        assert PERM_COMPUTER_USE_EXECUTE.action == Action.EXECUTE
        assert PERM_COMPUTER_USE_BROWSER.action == Action.BROWSER
        assert PERM_COMPUTER_USE_SHELL.action == Action.SHELL
        assert PERM_COMPUTER_USE_FILE_READ.action == Action.FILE_READ
        assert PERM_COMPUTER_USE_FILE_WRITE.action == Action.FILE_WRITE
        assert PERM_COMPUTER_USE_SCREENSHOT.action == Action.SCREENSHOT
        assert PERM_COMPUTER_USE_NETWORK.action == Action.NETWORK
        assert PERM_COMPUTER_USE_ADMIN.action == Action.ADMIN_OP

    def test_permission_descriptions(self):
        """Verify permissions have meaningful descriptions."""
        perms = [
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

        for perm in perms:
            assert perm.description, f"Missing description for {perm.key}"
            assert len(perm.description) > 10, f"Description too short for {perm.key}"


# =============================================================================
# Permission Lookup Tests
# =============================================================================


class TestComputerUsePermissionLookup:
    """Tests for permission lookup."""

    def test_get_permission_by_key(self):
        """Test looking up permissions by key."""
        perm = get_permission("computer_use.execute")
        assert perm is not None
        assert perm.action == Action.EXECUTE
        assert perm.resource == ResourceType.COMPUTER_USE

    def test_get_permission_with_colon_format(self):
        """Test looking up permissions with colon format (handler compatibility)."""
        # Colon format should also work due to aliasing
        perm = get_permission("computer_use:execute")
        assert perm is not None
        assert perm.action == Action.EXECUTE

    def test_nonexistent_permission(self):
        """Test looking up nonexistent permission."""
        perm = get_permission("computer_use.nonexistent")
        assert perm is None


# =============================================================================
# Role Assignment Tests
# =============================================================================


class TestComputerUseRoleAssignments:
    """Tests for computer-use permissions in roles."""

    def test_admin_has_all_computer_use_permissions(self):
        """Admin role should have all computer-use permissions."""
        admin_perms = get_role_permissions("admin")

        expected = [
            PERM_COMPUTER_USE_READ.key,
            PERM_COMPUTER_USE_EXECUTE.key,
            PERM_COMPUTER_USE_BROWSER.key,
            PERM_COMPUTER_USE_SHELL.key,
            PERM_COMPUTER_USE_FILE_READ.key,
            PERM_COMPUTER_USE_FILE_WRITE.key,
            PERM_COMPUTER_USE_SCREENSHOT.key,
            PERM_COMPUTER_USE_NETWORK.key,
            PERM_COMPUTER_USE_ADMIN.key,
        ]

        for perm_key in expected:
            assert perm_key in admin_perms, f"Admin missing: {perm_key}"

    def test_owner_has_all_computer_use_permissions(self):
        """Owner role should have all permissions (including computer-use)."""
        owner_perms = get_role_permissions("owner")

        # Owner has all permissions
        for perm_key in SYSTEM_PERMISSIONS:
            if perm_key.startswith("computer_use."):
                assert perm_key in owner_perms, f"Owner missing: {perm_key}"

    def test_viewer_no_computer_use_permissions(self):
        """Viewer role should not have computer-use permissions."""
        viewer_perms = get_role_permissions("viewer")

        for perm_key in viewer_perms:
            assert not perm_key.startswith("computer_use."), f"Viewer should not have: {perm_key}"

    def test_member_no_computer_use_by_default(self):
        """Member role should not have computer-use permissions by default."""
        member_perms = get_role_permissions("member")

        # Member should not have dangerous computer-use permissions
        dangerous = [
            PERM_COMPUTER_USE_SHELL.key,
            PERM_COMPUTER_USE_FILE_WRITE.key,
            PERM_COMPUTER_USE_ADMIN.key,
        ]

        for perm_key in dangerous:
            assert perm_key not in member_perms, f"Member should not have: {perm_key}"


# =============================================================================
# Permission Granularity Tests
# =============================================================================


class TestComputerUsePermissionGranularity:
    """Tests for granular permission control."""

    def test_read_only_access(self):
        """Test that read permission is separate from execute."""
        read_perm = PERM_COMPUTER_USE_READ
        exec_perm = PERM_COMPUTER_USE_EXECUTE

        assert read_perm.key != exec_perm.key
        assert read_perm.action == Action.READ
        assert exec_perm.action == Action.EXECUTE

    def test_file_read_vs_write_separation(self):
        """Test that file read and write are separate permissions."""
        read_perm = PERM_COMPUTER_USE_FILE_READ
        write_perm = PERM_COMPUTER_USE_FILE_WRITE

        assert read_perm.key != write_perm.key
        assert read_perm.action == Action.FILE_READ
        assert write_perm.action == Action.FILE_WRITE

    def test_shell_is_high_privilege(self):
        """Shell execution should be a distinct, high-privilege permission."""
        shell_perm = PERM_COMPUTER_USE_SHELL

        assert shell_perm.action == Action.SHELL
        assert (
            "shell" in shell_perm.description.lower() or "command" in shell_perm.description.lower()
        )

    def test_browser_separate_from_shell(self):
        """Browser automation should be separate from shell execution."""
        browser_perm = PERM_COMPUTER_USE_BROWSER
        shell_perm = PERM_COMPUTER_USE_SHELL

        assert browser_perm.key != shell_perm.key
        assert browser_perm.action == Action.BROWSER
        assert shell_perm.action == Action.SHELL


# =============================================================================
# Permission Key Aliasing Tests
# =============================================================================


class TestComputerUseKeyAliasing:
    """Tests for permission key format aliases."""

    def test_dot_and_colon_aliases(self):
        """Both dot and colon formats should resolve to same permission."""
        dot_key = "computer_use.execute"
        colon_key = "computer_use:execute"

        dot_perm = SYSTEM_PERMISSIONS.get(dot_key)
        colon_perm = SYSTEM_PERMISSIONS.get(colon_key)

        assert dot_perm is not None
        assert colon_perm is not None
        # They should be the same permission object
        assert dot_perm.key == colon_perm.key

    def test_all_computer_use_have_colon_aliases(self):
        """All computer-use permissions should have colon aliases."""
        base_keys = [
            "computer_use.read",
            "computer_use.execute",
            "computer_use.browser",
            "computer_use.shell",
            "computer_use.file_read",
            "computer_use.file_write",
            "computer_use.screenshot",
            "computer_use.network",
            "computer_use.admin",
        ]

        for base_key in base_keys:
            colon_key = base_key.replace(".", ":", 1)
            assert colon_key in SYSTEM_PERMISSIONS, f"Missing colon alias: {colon_key}"
