"""Tests for RBAC v2 CLI commands.

Tests the offline RBAC CLI integration:
- aragora rbac list-roles
- aragora rbac list-permissions
- aragora rbac check-local <user> <permission>
"""

from __future__ import annotations

import argparse
import json

import pytest

from aragora.cli.commands.rbac_ops import (
    _cmd_check_local,
    _cmd_list_permissions_local,
    _cmd_list_roles_local,
    add_rbac_ops_parser,
    cmd_rbac_ops,
)


# ---------------------------------------------------------------------------
# Parser registration tests
# ---------------------------------------------------------------------------


class TestRBACParserRegistration:
    """Verify that RBAC subcommands are registered in the CLI parser."""

    def test_rbac_parser_registered(self):
        """The RBAC parser registers with subparsers."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_rbac_ops_parser(subparsers)

        args = parser.parse_args(["rbac", "list-roles"])
        assert args.rbac_command == "list-roles"

    def test_list_roles_parses(self):
        """'rbac list-roles' parses correctly."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_rbac_ops_parser(subparsers)

        args = parser.parse_args(["rbac", "list-roles", "--json"])
        assert args.rbac_command == "list-roles"
        assert args.json is True

    def test_list_roles_verbose_parses(self):
        """'rbac list-roles --verbose' parses correctly."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_rbac_ops_parser(subparsers)

        args = parser.parse_args(["rbac", "list-roles", "--verbose"])
        assert args.verbose is True

    def test_list_permissions_parses(self):
        """'rbac list-permissions' parses correctly."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_rbac_ops_parser(subparsers)

        args = parser.parse_args(["rbac", "list-permissions"])
        assert args.rbac_command == "list-permissions"

    def test_list_permissions_with_group(self):
        """'rbac list-permissions --group debates' parses correctly."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_rbac_ops_parser(subparsers)

        args = parser.parse_args(["rbac", "list-permissions", "--group", "debates"])
        assert args.group == "debates"

    def test_check_local_parses(self):
        """'rbac check-local user1 debates:read' parses correctly."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_rbac_ops_parser(subparsers)

        args = parser.parse_args(
            [
                "rbac",
                "check-local",
                "user1",
                "debates:read",
                "--role",
                "admin",
            ]
        )
        assert args.rbac_command == "check-local"
        assert args.user_id == "user1"
        assert args.permission == "debates:read"
        assert args.role == "admin"

    def test_check_local_json_parses(self):
        """'rbac check-local --json' parses correctly."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        add_rbac_ops_parser(subparsers)

        args = parser.parse_args(
            [
                "rbac",
                "check-local",
                "user1",
                "debates:read",
                "--json",
            ]
        )
        assert args.json is True


# ---------------------------------------------------------------------------
# List roles tests
# ---------------------------------------------------------------------------


class TestRBACListRoles:
    """Tests for 'aragora rbac list-roles'."""

    def test_list_roles_text_output(self, capsys):
        """List roles command prints role information."""
        args = argparse.Namespace(
            rbac_command="list-roles",
            json=False,
            verbose=False,
        )
        _cmd_list_roles_local(args)
        output = capsys.readouterr().out

        assert "System Roles" in output
        # Should list at least some well-known roles
        # Check for the = separator
        assert "=" * 60 in output

    def test_list_roles_json_output(self, capsys):
        """List roles command outputs valid JSON."""
        args = argparse.Namespace(
            rbac_command="list-roles",
            json=True,
            verbose=False,
        )
        _cmd_list_roles_local(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        assert "roles" in data
        assert "total" in data
        assert data["total"] > 0

        role = data["roles"][0]
        assert "name" in role
        assert "display_name" in role
        assert "description" in role
        assert "permission_count" in role
        assert "is_system" in role
        assert "priority" in role

    def test_list_roles_json_no_permissions_by_default(self, capsys):
        """Without --verbose, JSON output should not include permissions list."""
        args = argparse.Namespace(
            rbac_command="list-roles",
            json=True,
            verbose=False,
        )
        _cmd_list_roles_local(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        for role in data["roles"]:
            assert "permissions" not in role

    def test_list_roles_json_verbose_includes_permissions(self, capsys):
        """With --verbose, JSON output should include permissions list."""
        args = argparse.Namespace(
            rbac_command="list-roles",
            json=True,
            verbose=True,
        )
        _cmd_list_roles_local(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        # At least one role should have permissions
        has_perms = any("permissions" in role for role in data["roles"])
        assert has_perms

    def test_list_roles_verbose_text(self, capsys):
        """Verbose text output shows individual permissions."""
        args = argparse.Namespace(
            rbac_command="list-roles",
            json=False,
            verbose=True,
        )
        _cmd_list_roles_local(args)
        output = capsys.readouterr().out

        # Verbose mode should show individual permissions with "-" prefix
        assert "- " in output


# ---------------------------------------------------------------------------
# List permissions tests
# ---------------------------------------------------------------------------


class TestRBACListPermissions:
    """Tests for 'aragora rbac list-permissions'."""

    def test_list_permissions_text_output(self, capsys):
        """List permissions command prints grouped permissions."""
        args = argparse.Namespace(
            rbac_command="list-permissions",
            json=False,
            group=None,
        )
        _cmd_list_permissions_local(args)
        output = capsys.readouterr().out

        assert "Permissions" in output
        assert "=" * 60 in output

    def test_list_permissions_json_output(self, capsys):
        """List permissions command outputs valid JSON."""
        args = argparse.Namespace(
            rbac_command="list-permissions",
            json=True,
            group=None,
        )
        _cmd_list_permissions_local(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        assert "permissions" in data
        assert "total" in data
        assert data["total"] > 0

        perm = data["permissions"][0]
        assert "key" in perm
        assert "name" in perm
        assert "resource" in perm
        assert "action" in perm

    def test_list_permissions_group_filter(self, capsys):
        """List permissions command filters by resource group."""
        args = argparse.Namespace(
            rbac_command="list-permissions",
            json=True,
            group="debates",
        )
        _cmd_list_permissions_local(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        # All returned permissions should be in the debates group
        for perm in data["permissions"]:
            assert perm["resource"] == "debates"

    def test_list_permissions_unknown_group(self, capsys):
        """List permissions with non-matching group shows nothing."""
        args = argparse.Namespace(
            rbac_command="list-permissions",
            json=False,
            group="nonexistent_group",
        )
        _cmd_list_permissions_local(args)
        output = capsys.readouterr().out

        assert "No permissions found" in output


# ---------------------------------------------------------------------------
# Check local tests
# ---------------------------------------------------------------------------


class TestRBACCheckLocal:
    """Tests for 'aragora rbac check-local'."""

    def test_check_local_with_admin_role(self, capsys):
        """Admin role should have broad permissions."""
        args = argparse.Namespace(
            rbac_command="check-local",
            user_id="test-user-1",
            permission="debates:read",
            role="admin",
            json=False,
        )
        _cmd_check_local(args)
        output = capsys.readouterr().out

        assert "test-user-1" in output
        assert "debates:read" in output

    def test_check_local_json_output(self, capsys):
        """Check local command outputs valid JSON."""
        args = argparse.Namespace(
            rbac_command="check-local",
            user_id="test-user-1",
            permission="debates:read",
            role="admin",
            json=True,
        )
        _cmd_check_local(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        assert "allowed" in data
        assert "reason" in data

    def test_check_local_no_role(self, capsys):
        """Check with no role should deny access."""
        args = argparse.Namespace(
            rbac_command="check-local",
            user_id="test-user-1",
            permission="debates:read",
            role=None,
            json=True,
        )
        _cmd_check_local(args)
        output = capsys.readouterr().out

        data = json.loads(output)
        assert data["allowed"] is False

    def test_check_local_missing_user_id(self, capsys):
        """Check with no user_id shows error."""
        args = argparse.Namespace(
            rbac_command="check-local",
            user_id=None,
            permission="debates:read",
            role=None,
            json=False,
        )
        _cmd_check_local(args)
        output = capsys.readouterr().out

        assert "Error" in output

    def test_check_local_missing_permission(self, capsys):
        """Check with no permission shows error."""
        args = argparse.Namespace(
            rbac_command="check-local",
            user_id="test-user-1",
            permission=None,
            role=None,
            json=False,
        )
        _cmd_check_local(args)
        output = capsys.readouterr().out

        assert "Error" in output

    def test_check_local_shows_role_info(self, capsys):
        """Check local shows role information when role is provided."""
        args = argparse.Namespace(
            rbac_command="check-local",
            user_id="test-user-1",
            permission="debates:read",
            role="viewer",
            json=False,
        )
        _cmd_check_local(args)
        output = capsys.readouterr().out

        assert "Role: viewer" in output
        assert "permission(s)" in output


# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------


class TestRBACDispatcher:
    """Tests for the cmd_rbac_ops dispatcher."""

    def test_dispatches_to_list_roles(self, capsys):
        """Dispatcher routes 'list-roles' to _cmd_list_roles_local."""
        args = argparse.Namespace(
            rbac_command="list-roles",
            json=False,
            verbose=False,
        )
        cmd_rbac_ops(args)
        output = capsys.readouterr().out
        assert "System Roles" in output

    def test_dispatches_to_list_permissions(self, capsys):
        """Dispatcher routes 'list-permissions' to _cmd_list_permissions_local."""
        args = argparse.Namespace(
            rbac_command="list-permissions",
            json=True,
            group=None,
        )
        cmd_rbac_ops(args)
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "permissions" in data

    def test_dispatches_to_check_local(self, capsys):
        """Dispatcher routes 'check-local' to _cmd_check_local."""
        args = argparse.Namespace(
            rbac_command="check-local",
            user_id="user1",
            permission="debates:read",
            role="admin",
            json=True,
        )
        cmd_rbac_ops(args)
        output = capsys.readouterr().out
        data = json.loads(output)
        assert "allowed" in data

    def test_no_subcommand_shows_help(self, capsys):
        """Dispatcher shows help when no subcommand is given."""
        args = argparse.Namespace(rbac_command=None)
        cmd_rbac_ops(args)
        output = capsys.readouterr().out
        assert "list-roles" in output
        assert "list-permissions" in output
        assert "check-local" in output
