"""
RBAC operations CLI commands.

Provides CLI access to the role-based access control system.

API-backed commands (require running server):
- aragora rbac roles                        - List roles (via API)
- aragora rbac permissions                  - List permissions (via API)
- aragora rbac assign <user_id> <role>      - Assign role to user (via API)
- aragora rbac check <user_id> <permission> - Check user permission (via API)

Offline commands (work without server, use local RBAC module):
- aragora rbac list-roles                   - List all system roles
- aragora rbac list-permissions             - List all permissions
- aragora rbac check-local <user> <perm>    - Check permission locally
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

try:
    from aragora.config.settings import get_settings
except ImportError:
    get_settings = None  # type: ignore[assignment,misc]

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

    # Offline commands (no server required)
    if subcommand == "list-roles":
        _cmd_list_roles_local(args)
    elif subcommand == "list-permissions":
        _cmd_list_permissions_local(args)
    elif subcommand == "check-local":
        _cmd_check_local(args)
    # API-backed commands
    elif subcommand == "roles":
        asyncio.run(_cmd_roles(args))
    elif subcommand == "permissions":
        asyncio.run(_cmd_permissions(args))
    elif subcommand == "assign":
        asyncio.run(_cmd_assign(args))
    elif subcommand == "check":
        asyncio.run(_cmd_check(args))
    else:
        print("\nUsage: aragora rbac <command>")
        print("\nOffline commands (no server required):")
        print("  list-roles                     List all system roles and permissions")
        print("  list-permissions               List all available permissions")
        print("  check-local <user> <permission> Check permission locally")
        print("\nAPI commands (require running server):")
        print("  roles                          List all roles (via API)")
        print("  permissions                    List all permissions (via API)")
        print("  assign <user_id> <role>        Assign role to user (via API)")
        print("  check <user_id> <permission>   Check user permission (via API)")


def _cmd_list_roles_local(args: argparse.Namespace) -> None:
    """List all system roles (offline, no server required)."""
    from aragora.rbac.defaults.roles import SYSTEM_ROLES

    as_json = getattr(args, "json", False)
    verbose = getattr(args, "verbose", False)

    if as_json:
        data = {
            "roles": [
                {
                    "name": role.name,
                    "display_name": role.display_name,
                    "description": role.description,
                    "permission_count": len(role.permissions),
                    "is_system": role.is_system,
                    "priority": role.priority,
                    "parent_roles": role.parent_roles,
                    **({"permissions": sorted(role.permissions)} if verbose else {}),
                }
                for role in SYSTEM_ROLES.values()
            ],
            "total": len(SYSTEM_ROLES),
        }
        print(json.dumps(data, indent=2))
        return

    print(f"\nSystem Roles ({len(SYSTEM_ROLES)}):")
    print("=" * 60)

    for name, role in sorted(SYSTEM_ROLES.items(), key=lambda x: -x[1].priority):
        perm_count = len(role.permissions)
        parents = f" (inherits: {', '.join(role.parent_roles)})" if role.parent_roles else ""
        print(f"\n  {role.display_name} [{name}]")
        print(f"    {role.description}")
        print(f"    Permissions: {perm_count}  |  Priority: {role.priority}{parents}")

        if verbose:
            for perm in sorted(role.permissions):
                print(f"      - {perm}")

    print()


def _cmd_list_permissions_local(args: argparse.Namespace) -> None:
    """List all available permissions (offline, no server required)."""
    from aragora.rbac.defaults.registry import SYSTEM_PERMISSIONS

    as_json = getattr(args, "json", False)
    group_filter = getattr(args, "group", None)

    if as_json:
        data = []
        for key, perm in sorted(SYSTEM_PERMISSIONS.items()):
            resource = key.split(".")[0] if "." in key else "general"
            if group_filter and resource != group_filter:
                continue
            data.append({
                "key": perm.key,
                "name": perm.name,
                "resource": perm.resource.value,
                "action": perm.action.value,
                "description": perm.description,
            })
        print(json.dumps({"permissions": data, "total": len(data)}, indent=2))
        return

    # Group by resource prefix
    grouped: dict[str, list[tuple[str, Any]]] = {}
    for key, perm in sorted(SYSTEM_PERMISSIONS.items()):
        resource = perm.resource.value
        if group_filter and resource != group_filter:
            continue
        grouped.setdefault(resource, []).append((key, perm))

    total = sum(len(v) for v in grouped.values())
    if group_filter:
        print(f"\nPermissions for '{group_filter}' ({total}):")
    else:
        print(f"\nAll Permissions ({total}):")
    print("=" * 60)

    if not grouped:
        print("  No permissions found.")
        if group_filter:
            print(f"  No permissions matching group '{group_filter}'.")
        print()
        return

    for resource, perms in sorted(grouped.items()):
        print(f"\n  {resource} ({len(perms)}):")
        for key, perm in perms:
            desc = f"  -- {perm.description}" if perm.description else ""
            print(f"    {perm.key}{desc}")

    print()


def _cmd_check_local(args: argparse.Namespace) -> None:
    """Check if a user has a specific permission (offline, no server required)."""
    from aragora.rbac.checker import PermissionChecker
    from aragora.rbac.defaults.helpers import get_role_permissions
    from aragora.rbac.models import AuthorizationContext

    user_id = getattr(args, "user_id", None)
    permission = getattr(args, "permission", None)
    role_name = getattr(args, "role", None)
    as_json = getattr(args, "json", False)

    if not user_id or not permission:
        print("Error: user_id and permission are required.")
        print("Usage: aragora rbac check-local <user_id> <permission> --role <role>")
        return

    # Build context with the specified role
    roles = {role_name} if role_name else set()
    context = AuthorizationContext(
        user_id=user_id,
        roles=roles,
    )

    checker = PermissionChecker(enable_cache=False)
    decision = checker.check_permission(context, permission)

    if as_json:
        print(json.dumps(decision.to_dict(), indent=2, default=str))
        return

    if decision.allowed:
        print(f"\nUser '{user_id}' HAS permission '{permission}'.")
    else:
        print(f"\nUser '{user_id}' DOES NOT have permission '{permission}'.")
    print(f"  Reason: {decision.reason}")
    if role_name:
        print(f"  Role: {role_name}")
        perms = get_role_permissions(role_name, include_inherited=True)
        print(f"  Role has {len(perms)} permission(s)")
    print()


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
    except (OSError, ConnectionError, RuntimeError, ValueError) as e:
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
    except (OSError, ConnectionError, RuntimeError, ValueError) as e:
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
    except (OSError, ConnectionError, RuntimeError, ValueError) as e:
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
    except (OSError, ConnectionError, RuntimeError, ValueError) as e:
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
        description=(
            "Manage roles, permissions, and access assignments.\n\n"
            "Offline commands (no server required):\n"
            "  list-roles        List all system roles\n"
            "  list-permissions   List all available permissions\n"
            "  check-local       Check permission locally\n\n"
            "API commands (require running server):\n"
            "  roles             List roles via API\n"
            "  permissions       List permissions via API\n"
            "  assign            Assign role to user via API\n"
            "  check             Check permission via API"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    rp.set_defaults(func=cmd_rbac_ops)

    rp_sub = rp.add_subparsers(dest="rbac_command")

    # -- Offline commands --

    # list-roles (offline)
    list_roles_p = rp_sub.add_parser(
        "list-roles",
        help="List all system roles (offline, no server required)",
    )
    list_roles_p.add_argument("--json", action="store_true", help="Output as JSON")
    list_roles_p.add_argument(
        "--verbose", "-v", action="store_true", help="Show all permissions per role"
    )

    # list-permissions (offline)
    list_perms_p = rp_sub.add_parser(
        "list-permissions",
        help="List all available permissions (offline, no server required)",
    )
    list_perms_p.add_argument("--group", "-g", help="Filter by resource group")
    list_perms_p.add_argument("--json", action="store_true", help="Output as JSON")

    # check-local (offline)
    check_local_p = rp_sub.add_parser(
        "check-local",
        help="Check user permission locally (offline, no server required)",
    )
    check_local_p.add_argument("user_id", help="User ID")
    check_local_p.add_argument(
        "permission", help="Permission to check (e.g. debates:read or debates.read)"
    )
    check_local_p.add_argument(
        "--role",
        "-r",
        help="Role to assign to the user for the check (e.g. admin, viewer, editor)",
    )
    check_local_p.add_argument("--json", action="store_true", help="Output as JSON")

    # -- API commands --

    # roles
    roles_p = rp_sub.add_parser("roles", help="List all roles (via API)")
    roles_p.add_argument("--json", action="store_true", help="Output as JSON")

    # permissions
    perms_p = rp_sub.add_parser("permissions", help="List all permissions (via API)")
    perms_p.add_argument("--group", "-g", help="Filter by permission group")
    perms_p.add_argument("--json", action="store_true", help="Output as JSON")

    # assign
    assign_p = rp_sub.add_parser("assign", help="Assign role to user (via API)")
    assign_p.add_argument("user_id", help="User ID")
    assign_p.add_argument("role", help="Role name to assign")
    assign_p.add_argument("--json", action="store_true", help="Output as JSON")

    # check
    check_p = rp_sub.add_parser("check", help="Check user permission (via API)")
    check_p.add_argument("user_id", help="User ID")
    check_p.add_argument("permission", help="Permission to check (e.g. debates:read)")
    check_p.add_argument("--json", action="store_true", help="Output as JSON")


__all__ = [
    "cmd_rbac_ops",
    "add_rbac_ops_parser",
    "_cmd_list_roles_local",
    "_cmd_list_permissions_local",
    "_cmd_check_local",
]
