"""
RBAC operations CLI commands.

Provides CLI access to the role-based access control system via server API endpoints.
Commands:
- aragora rbac roles                        - List roles
- aragora rbac permissions                  - List permissions
- aragora rbac assign <user_id> <role>      - Assign role to user
- aragora rbac check <user_id> <permission> - Check user permission
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

import httpx

from aragora.config.settings import get_settings

logger = logging.getLogger(__name__)


def _get_api_base() -> str:
    """Get the API base URL from settings."""
    settings = get_settings()
    host = getattr(settings, "server_host", "localhost")
    port = getattr(settings, "server_port", 8000)
    return f"http://{host}:{port}"


def _get_auth_headers() -> dict[str, str]:
    """Get authentication headers if available."""
    settings = get_settings()
    token = getattr(settings, "api_token", None)
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


async def _api_get(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make GET request to API."""
    base = _get_api_base()
    url = f"{base}{endpoint}"
    headers = _get_auth_headers()
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


async def _api_post(endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Make POST request to API."""
    base = _get_api_base()
    url = f"{base}{endpoint}"
    headers = _get_auth_headers()
    headers["Content-Type"] = "application/json"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=data or {}, headers=headers)
        response.raise_for_status()
        return response.json()


def cmd_rbac_ops(args: argparse.Namespace) -> None:
    """Handle 'rbac' command - dispatch to subcommands."""
    subcommand = getattr(args, "rbac_command", None)

    if subcommand == "roles":
        asyncio.run(_cmd_roles(args))
    elif subcommand == "permissions":
        asyncio.run(_cmd_permissions(args))
    elif subcommand == "assign":
        asyncio.run(_cmd_assign(args))
    elif subcommand == "check":
        asyncio.run(_cmd_check(args))
    else:
        print("\nUsage: aragora rbac <command>")
        print("\nCommands:")
        print("  roles                          List all roles")
        print("  permissions                    List all permissions")
        print("  assign <user_id> <role>        Assign role to user")
        print("  check <user_id> <permission>   Check user permission")


async def _cmd_roles(args: argparse.Namespace) -> None:
    """List all roles."""
    as_json = getattr(args, "json", False)

    try:
        result = await _api_get("/api/v1/rbac/roles")
        if as_json:
            print(json.dumps(result, indent=2))
            return

        roles = result.get("roles", [])
        print(f"\nRoles ({len(roles)}):\n")
        for role in roles:
            name = role.get("name", "unknown")
            description = role.get("description", "")
            perm_count = len(role.get("permissions", []))
            print(f"  {name:20} ({perm_count} permissions)")
            if description:
                print(f"    {description}")

        if not roles:
            print("  No roles defined.")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_permissions(args: argparse.Namespace) -> None:
    """List all permissions."""
    as_json = getattr(args, "json", False)
    group = getattr(args, "group", None)

    params: dict[str, Any] = {}
    if group:
        params["group"] = group

    try:
        result = await _api_get("/api/v1/rbac/permissions", params=params)
        if as_json:
            print(json.dumps(result, indent=2))
            return

        permissions = result.get("permissions", [])
        print(f"\nPermissions ({len(permissions)}):\n")

        # Group by resource prefix
        grouped: dict[str, list[str]] = {}
        for perm in permissions:
            pname = perm if isinstance(perm, str) else perm.get("name", "unknown")
            resource = pname.split(":")[0] if ":" in pname else "general"
            grouped.setdefault(resource, []).append(pname)

        for resource, perms in sorted(grouped.items()):
            print(f"  {resource}:")
            for p in sorted(perms):
                print(f"    {p}")

        if not permissions:
            print("  No permissions defined.")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_assign(args: argparse.Namespace) -> None:
    """Assign a role to a user."""
    user_id = getattr(args, "user_id", None)
    role = getattr(args, "role", None)
    as_json = getattr(args, "json", False)

    if not user_id:
        print("Error: User ID is required")
        print("Usage: aragora rbac assign <user_id> <role>")
        return

    if not role:
        print("Error: Role name is required")
        print("Usage: aragora rbac assign <user_id> <role>")
        return

    data: dict[str, Any] = {
        "user_id": user_id,
        "role": role,
    }

    try:
        result = await _api_post("/api/v1/rbac/assignments", data)
        if as_json:
            print(json.dumps(result, indent=2))
            return

        if result.get("success", True):
            print(f"\nRole '{role}' assigned to user '{user_id}'.")
        else:
            error = result.get("error", "Unknown error")
            print(f"\nAssignment failed: {error}")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_check(args: argparse.Namespace) -> None:
    """Check if a user has a specific permission."""
    user_id = getattr(args, "user_id", None)
    permission = getattr(args, "permission", None)
    as_json = getattr(args, "json", False)

    if not user_id:
        print("Error: User ID is required")
        print("Usage: aragora rbac check <user_id> <permission>")
        return

    if not permission:
        print("Error: Permission name is required")
        print("Usage: aragora rbac check <user_id> <permission>")
        return

    params: dict[str, Any] = {
        "user_id": user_id,
        "permission": permission,
    }

    try:
        result = await _api_get("/api/v1/rbac/check", params=params)
        if as_json:
            print(json.dumps(result, indent=2))
            return

        allowed = result.get("allowed", False)
        reason = result.get("reason", "")
        if allowed:
            print(f"\nUser '{user_id}' HAS permission '{permission}'.")
        else:
            print(f"\nUser '{user_id}' DOES NOT have permission '{permission}'.")
        if reason:
            print(f"  Reason: {reason}")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        _print_api_error(e)
    except Exception as e:
        print(f"\nError: {e}")


def _print_api_error(e: httpx.HTTPStatusError) -> None:
    """Print a human-readable API error message."""
    try:
        error_data = e.response.json()
        print(f"  {error_data.get('error', error_data.get('detail', 'Unknown error'))}")
    except (ValueError, KeyError):
        print(f"  {e.response.text}")


def add_rbac_ops_parser(subparsers: Any) -> None:
    """Add RBAC operations subparser to CLI."""
    rp = subparsers.add_parser(
        "rbac",
        help="RBAC management commands",
        description="Manage roles, permissions, and access assignments.",
    )
    rp.set_defaults(func=cmd_rbac_ops)

    rp_sub = rp.add_subparsers(dest="rbac_command")

    # roles
    roles_p = rp_sub.add_parser("roles", help="List all roles")
    roles_p.add_argument("--json", action="store_true", help="Output as JSON")

    # permissions
    perms_p = rp_sub.add_parser("permissions", help="List all permissions")
    perms_p.add_argument("--group", "-g", help="Filter by permission group")
    perms_p.add_argument("--json", action="store_true", help="Output as JSON")

    # assign
    assign_p = rp_sub.add_parser("assign", help="Assign role to user")
    assign_p.add_argument("user_id", help="User ID")
    assign_p.add_argument("role", help="Role name to assign")
    assign_p.add_argument("--json", action="store_true", help="Output as JSON")

    # check
    check_p = rp_sub.add_parser("check", help="Check user permission")
    check_p.add_argument("user_id", help="User ID")
    check_p.add_argument("permission", help="Permission to check (e.g. debates:read)")
    check_p.add_argument("--json", action="store_true", help="Output as JSON")


__all__ = [
    "cmd_rbac_ops",
    "add_rbac_ops_parser",
]
