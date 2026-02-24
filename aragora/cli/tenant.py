"""
Aragora Tenant CLI.

Commands for managing multi-tenant deployments:
- List, create, delete tenants
- Manage quotas and limits
- Export tenant data
- Migration support

Usage:
    aragora tenant list
    aragora tenant create --name "Acme Corp" --tier starter
    aragora tenant quota-get --tenant acme-corp
    aragora tenant quota-set --tenant acme-corp --debates 1000
    aragora tenant export --tenant acme-corp --output acme-data.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from aragora.security.safe_http import safe_request

DEFAULT_API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")


def get_api_token() -> str | None:
    """Get API token from environment."""
    return os.environ.get("ARAGORA_API_TOKEN") or os.environ.get("ARAGORA_API_KEY")


def api_request(
    method: str,
    path: str,
    data: dict[str, Any] | None = None,
    server_url: str = DEFAULT_API_URL,
) -> dict[str, Any]:
    """Make authenticated API request."""
    url = f"{server_url.rstrip('/')}{path}"
    token = get_api_token()

    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = safe_request(method, url, json=data, headers=headers, timeout=30)
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result
    except httpx.HTTPStatusError as e:
        error_body = e.response.text
        raise RuntimeError(f"API error ({e.response.status_code}): {error_body}")
    except httpx.RequestError as e:
        raise RuntimeError(f"Connection error: {e}")


# =============================================================================
# List Tenants
# =============================================================================


def cmd_list(args: argparse.Namespace) -> int:
    """List all tenants."""
    print("\n" + "=" * 70)
    print("ARAGORA TENANTS")
    print("=" * 70)

    try:
        params = []
        if args.status:
            params.append(f"status={args.status}")
        if args.tier:
            params.append(f"tier={args.tier}")
        if args.limit:
            params.append(f"limit={args.limit}")

        query = f"?{'&'.join(params)}" if params else ""
        result = api_request("GET", f"/api/v1/tenants{query}", server_url=args.server)

        tenants = result.get("tenants", [])
        total = result.get("total", len(tenants))

        if not tenants:
            print("\nNo tenants found.")
            return 0

        print(f"\nTotal: {total} tenant(s)")
        print("-" * 70)
        print(f"{'ID':<25} {'Name':<20} {'Tier':<12} {'Status':<10}")
        print("-" * 70)

        for tenant in tenants:
            tid = tenant.get("id", "")[:25]
            name = tenant.get("name", "")[:20]
            tier = tenant.get("tier", "unknown")[:12]
            status = tenant.get("status", "unknown")[:10]
            print(f"{tid:<25} {name:<20} {tier:<12} {status:<10}")

        print("-" * 70)
        return 0

    except RuntimeError as e:
        if "Connection" in str(e):
            return cmd_list_local(args)
        print(f"\nError: {e}")
        return 1


def cmd_list_local(args: argparse.Namespace) -> int:
    """List tenants from local configuration (offline mode)."""
    print("\n(Running in offline mode - showing local configuration)")

    try:
        from aragora.tenancy import TenantManager

        manager = TenantManager()
        tenants = manager.list_tenants(
            status=args.status,
            tier=args.tier,
        )
        limit = args.limit or 100
        tenants = tenants[:limit]

        if not tenants:
            print("\nNo tenants configured locally.")
            print("Create one with: aragora tenant create --name 'My Tenant'")
            return 0

        print(f"\nTotal: {len(tenants)} tenant(s)")
        print("-" * 70)
        print(f"{'ID':<25} {'Name':<20} {'Tier':<12} {'Status':<10}")
        print("-" * 70)

        for tenant in tenants:
            print(
                f"{tenant.id[:24]:<25} "
                f"{tenant.name[:19]:<20} "
                f"{tenant.tier.value:<12} "
                f"{tenant.status.value:<10}"
            )

        return 0

    except (OSError, RuntimeError, ValueError, KeyError) as e:
        print(f"\nError: {e}")
        return 1


# =============================================================================
# Create Tenant
# =============================================================================


def cmd_create(args: argparse.Namespace) -> int:
    """Create a new tenant."""
    print("\n" + "=" * 60)
    print("CREATE TENANT")
    print("=" * 60)

    try:
        data = {
            "name": args.name,
            "tier": args.tier,
        }

        if args.domain:
            data["domain"] = args.domain
        if args.admin_email:
            data["admin_email"] = args.admin_email

        result = api_request("POST", "/api/v1/tenants", data=data, server_url=args.server)

        tenant = result.get("tenant", result)
        print("\nTenant created successfully!")
        print(f"  ID:     {tenant.get('id')}")
        print(f"  Name:   {tenant.get('name')}")
        print(f"  Tier:   {tenant.get('tier')}")
        print(f"  Status: {tenant.get('status')}")

        if tenant.get("api_key"):
            print(f"\n  API Key: {tenant.get('api_key')}")
            print("  (Save this - it won't be shown again)")

        return 0

    except RuntimeError as e:
        if "Connection" in str(e):
            return cmd_create_local(args)
        print(f"\nError: {e}")
        return 1


def cmd_create_local(args: argparse.Namespace) -> int:
    """Create tenant locally (offline mode)."""
    print("\n(Running in offline mode - creating local tenant)")

    try:
        from aragora.tenancy import Tenant, TenantManager, TenantTier

        tier = TenantTier(args.tier)
        tenant = Tenant.create(
            name=args.name,
            owner_email=getattr(args, "admin_email", "") or "",
            tier=tier,
        )

        manager = TenantManager()
        manager.register_tenant(tenant)

        print("\nTenant created successfully!")
        print(f"  ID:     {tenant.id}")
        print(f"  Name:   {tenant.name}")
        print(f"  Tier:   {tenant.tier.value}")
        print(f"  Status: {tenant.status.value}")

        api_key = tenant.generate_api_key()
        print(f"\n  API Key: {api_key}")
        print("  (Save this - it won't be shown again)")

        return 0

    except (OSError, RuntimeError, ValueError, KeyError) as e:
        print(f"\nError: {e}")
        return 1


# =============================================================================
# Delete Tenant
# =============================================================================


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete a tenant."""
    tenant_id = args.tenant

    if not args.force:
        confirm = input(f"Are you sure you want to delete tenant '{tenant_id}'? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return 0

    try:
        api_request("DELETE", f"/api/v1/tenants/{tenant_id}", server_url=args.server)
        print(f"\nTenant '{tenant_id}' deleted successfully.")
        return 0

    except RuntimeError as e:
        if "Connection" in str(e):
            return cmd_delete_local(args)
        print(f"\nError: {e}")
        return 1


def cmd_delete_local(args: argparse.Namespace) -> int:
    """Delete tenant locally (offline mode)."""
    try:
        from aragora.tenancy import TenantManager

        manager = TenantManager()
        tenant = manager.unregister_tenant(args.tenant)

        if tenant:
            print(f"\nTenant '{args.tenant}' deleted.")
            return 0
        else:
            print(f"\nTenant '{args.tenant}' not found.")
            return 1

    except (OSError, RuntimeError, ValueError, KeyError) as e:
        print(f"\nError: {e}")
        return 1


# =============================================================================
# Quota Management
# =============================================================================


def cmd_quota_get(args: argparse.Namespace) -> int:
    """Get tenant quotas and usage."""
    tenant_id = args.tenant

    print("\n" + "=" * 60)
    print(f"QUOTAS FOR: {tenant_id}")
    print("=" * 60)

    try:
        result = api_request("GET", f"/api/v1/tenants/{tenant_id}/quotas", server_url=args.server)

        quotas = result.get("quotas", {})
        usage = result.get("usage", {})

        print("\nResource Limits:")
        print("-" * 40)
        print(f"  Debates/day:         {quotas.get('max_debates_per_day', 'N/A')}")
        print(f"  Concurrent debates:  {quotas.get('max_concurrent_debates', 'N/A')}")
        print(f"  Agents/debate:       {quotas.get('max_agents_per_debate', 'N/A')}")
        print(f"  Rounds/debate:       {quotas.get('max_rounds_per_debate', 'N/A')}")
        print(f"  Users:               {quotas.get('max_users', 'N/A')}")
        print(f"  Connectors:          {quotas.get('max_connectors', 'N/A')}")

        print("\nToken Limits:")
        print("-" * 40)
        print(f"  Tokens/month:        {quotas.get('tokens_per_month', 'N/A'):,}")
        print(f"  Tokens/debate:       {quotas.get('tokens_per_debate', 'N/A'):,}")

        print("\nAPI Rate Limits:")
        print("-" * 40)
        print(f"  Requests/minute:     {quotas.get('api_requests_per_minute', 'N/A')}")
        print(f"  Requests/day:        {quotas.get('api_requests_per_day', 'N/A')}")

        if usage:
            print("\nCurrent Usage:")
            print("-" * 40)
            print(f"  Debates today:       {usage.get('debates_today', 0)}")
            print(f"  Tokens this month:   {usage.get('tokens_this_month', 0):,}")
            print(f"  Storage used:        {usage.get('storage_bytes', 0) / (1024 * 1024):.1f} MB")

        return 0

    except RuntimeError as e:
        if "Connection" in str(e):
            return cmd_quota_get_local(args)
        print(f"\nError: {e}")
        return 1


def cmd_quota_get_local(args: argparse.Namespace) -> int:
    """Get tenant quotas locally (offline mode)."""
    try:
        from aragora.tenancy import TenantManager

        manager = TenantManager()
        tenant = manager.get_tenant(args.tenant)

        if not tenant:
            print(f"\nTenant '{args.tenant}' not found.")
            return 1

        config = tenant.config

        print("\nResource Limits:")
        print("-" * 40)
        print(f"  Debates/day:         {config.max_debates_per_day}")
        print(f"  Concurrent debates:  {config.max_concurrent_debates}")
        print(f"  Agents/debate:       {config.max_agents_per_debate}")
        print(f"  Rounds/debate:       {config.max_rounds_per_debate}")
        print(f"  Users:               {config.max_users}")
        print(f"  Connectors:          {config.max_connectors}")

        print("\nToken Limits:")
        print("-" * 40)
        print(f"  Tokens/month:        {config.tokens_per_month:,}")
        print(f"  Tokens/debate:       {config.tokens_per_debate:,}")

        return 0

    except (OSError, RuntimeError, ValueError, KeyError) as e:
        print(f"\nError: {e}")
        return 1


def cmd_quota_set(args: argparse.Namespace) -> int:
    """Set tenant quotas."""
    tenant_id = args.tenant

    # Build quota update
    updates: dict[str, Any] = {}
    if args.debates is not None:
        updates["max_debates_per_day"] = args.debates
    if args.concurrent is not None:
        updates["max_concurrent_debates"] = args.concurrent
    if args.agents is not None:
        updates["max_agents_per_debate"] = args.agents
    if args.rounds is not None:
        updates["max_rounds_per_debate"] = args.rounds
    if args.users is not None:
        updates["max_users"] = args.users
    if args.tokens_month is not None:
        updates["tokens_per_month"] = args.tokens_month
    if args.tokens_debate is not None:
        updates["tokens_per_debate"] = args.tokens_debate

    if not updates:
        print("No quotas specified. Use --debates, --tokens-month, etc.")
        return 1

    try:
        api_request(
            "PATCH",
            f"/api/v1/tenants/{tenant_id}/quotas",
            data=updates,
            server_url=args.server,
        )

        print(f"\nQuotas updated for tenant '{tenant_id}':")
        for key, value in updates.items():
            print(f"  {key}: {value}")

        return 0

    except RuntimeError as e:
        print(f"\nError: {e}")
        return 1


# =============================================================================
# Tenant Data Export
# =============================================================================


def cmd_export(args: argparse.Namespace) -> int:
    """Export tenant data."""
    tenant_id = args.tenant
    output_path = Path(args.output)
    export_format = args.format

    print("\n" + "=" * 60)
    print(f"EXPORTING DATA FOR: {tenant_id}")
    print("=" * 60)

    try:
        # Request export from API
        params = f"?format={export_format}"
        if args.include_debates:
            params += "&include_debates=true"
        if args.include_knowledge:
            params += "&include_knowledge=true"
        if args.include_audit:
            params += "&include_audit=true"

        result = api_request(
            "GET",
            f"/api/v1/tenants/{tenant_id}/export{params}",
            server_url=args.server,
        )

        # Write output
        if export_format == "json":
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, default=str)
        else:
            # For parquet, the API returns a download URL
            download_url = result.get("download_url")
            if download_url:
                print(f"Download from: {download_url}")
            else:
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2, default=str)

        print("\nExport complete!")
        print(f"  Output: {output_path}")
        print(f"  Format: {export_format}")

        if "statistics" in result:
            stats = result["statistics"]
            print("\nExported:")
            print(f"  Debates:    {stats.get('debates', 0)}")
            print(f"  Documents:  {stats.get('documents', 0)}")
            print(f"  Users:      {stats.get('users', 0)}")

        return 0

    except RuntimeError as e:
        if "Connection" in str(e):
            return cmd_export_local(args)
        print(f"\nError: {e}")
        return 1


def cmd_export_local(args: argparse.Namespace) -> int:
    """Export tenant data locally (offline mode)."""
    print("\n(Running in offline mode - exporting local data)")

    try:
        from aragora.tenancy import TenantManager

        manager = TenantManager()
        tenant = manager.get_tenant(args.tenant)

        if not tenant:
            print(f"\nTenant '{args.tenant}' not found.")
            return 1

        # Build export data
        export_data = {
            "tenant": tenant.to_dict(),
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "format_version": "1.0",
        }

        # Write output
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        print("\nExport complete!")
        print(f"  Output: {output_path}")

        return 0

    except (OSError, RuntimeError, ValueError, KeyError) as e:
        print(f"\nError: {e}")
        return 1


# =============================================================================
# Tenant Status Management
# =============================================================================


def cmd_suspend(args: argparse.Namespace) -> int:
    """Suspend a tenant."""
    tenant_id = args.tenant
    reason = args.reason or "Administrative action"

    try:
        api_request(
            "POST",
            f"/api/v1/tenants/{tenant_id}/suspend",
            data={"reason": reason},
            server_url=args.server,
        )

        print(f"\nTenant '{tenant_id}' suspended.")
        print(f"  Reason: {reason}")
        return 0

    except RuntimeError as e:
        print(f"\nError: {e}")
        return 1


def cmd_activate(args: argparse.Namespace) -> int:
    """Activate a suspended tenant."""
    tenant_id = args.tenant

    try:
        api_request(
            "POST",
            f"/api/v1/tenants/{tenant_id}/activate",
            server_url=args.server,
        )

        print(f"\nTenant '{tenant_id}' activated.")
        return 0

    except RuntimeError as e:
        print(f"\nError: {e}")
        return 1


# =============================================================================
# Parser Creation
# =============================================================================


def create_tenant_parser(subparsers: argparse._SubParsersAction) -> None:
    """Create tenant subcommand parser."""
    tenant_parser = subparsers.add_parser(
        "tenant",
        help="Manage multi-tenant deployments",
        description="""
Manage tenants for multi-tenant Aragora deployments.

Examples:
    aragora tenant list                           # List all tenants
    aragora tenant create --name "Acme Corp"      # Create new tenant
    aragora tenant quota-get --tenant acme-corp   # View quotas
    aragora tenant quota-set --tenant acme-corp --debates 1000
    aragora tenant export --tenant acme-corp --output acme.json
    aragora tenant suspend --tenant acme-corp --reason "Non-payment"
    aragora tenant activate --tenant acme-corp
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    tenant_subparsers = tenant_parser.add_subparsers(dest="tenant_action")

    # List tenants
    list_parser = tenant_subparsers.add_parser("list", help="List all tenants")
    list_parser.add_argument("--status", "-s", help="Filter by status")
    list_parser.add_argument("--tier", "-t", help="Filter by tier")
    list_parser.add_argument("--limit", "-l", type=int, help="Max results")
    list_parser.add_argument("--server", default=DEFAULT_API_URL, help="API server URL")
    list_parser.set_defaults(func=cmd_list)

    # Create tenant
    create_parser = tenant_subparsers.add_parser("create", help="Create a new tenant")
    create_parser.add_argument("--name", "-n", required=True, help="Tenant name")
    create_parser.add_argument(
        "--tier",
        "-t",
        default="starter",
        choices=["free", "starter", "professional", "enterprise"],
        help="Subscription tier",
    )
    create_parser.add_argument("--domain", "-d", help="Custom domain")
    create_parser.add_argument("--admin-email", "-e", help="Admin email")
    create_parser.add_argument("--server", default=DEFAULT_API_URL, help="API server URL")
    create_parser.set_defaults(func=cmd_create)

    # Delete tenant
    delete_parser = tenant_subparsers.add_parser("delete", help="Delete a tenant")
    delete_parser.add_argument("--tenant", "-t", required=True, help="Tenant ID")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    delete_parser.add_argument("--server", default=DEFAULT_API_URL, help="API server URL")
    delete_parser.set_defaults(func=cmd_delete)

    # Get quotas
    quota_get_parser = tenant_subparsers.add_parser("quota-get", help="Get tenant quotas")
    quota_get_parser.add_argument("--tenant", "-t", required=True, help="Tenant ID")
    quota_get_parser.add_argument("--server", default=DEFAULT_API_URL, help="API server URL")
    quota_get_parser.set_defaults(func=cmd_quota_get)

    # Set quotas
    quota_set_parser = tenant_subparsers.add_parser("quota-set", help="Set tenant quotas")
    quota_set_parser.add_argument("--tenant", "-t", required=True, help="Tenant ID")
    quota_set_parser.add_argument("--debates", type=int, help="Max debates per day")
    quota_set_parser.add_argument("--concurrent", type=int, help="Max concurrent debates")
    quota_set_parser.add_argument("--agents", type=int, help="Max agents per debate")
    quota_set_parser.add_argument("--rounds", type=int, help="Max rounds per debate")
    quota_set_parser.add_argument("--users", type=int, help="Max users")
    quota_set_parser.add_argument("--tokens-month", type=int, help="Tokens per month")
    quota_set_parser.add_argument("--tokens-debate", type=int, help="Tokens per debate")
    quota_set_parser.add_argument("--server", default=DEFAULT_API_URL, help="API server URL")
    quota_set_parser.set_defaults(func=cmd_quota_set)

    # Export data
    export_parser = tenant_subparsers.add_parser("export", help="Export tenant data")
    export_parser.add_argument("--tenant", "-t", required=True, help="Tenant ID")
    export_parser.add_argument("--output", "-o", required=True, help="Output file path")
    export_parser.add_argument(
        "--format",
        "-f",
        default="json",
        choices=["json", "parquet"],
        help="Export format",
    )
    export_parser.add_argument(
        "--include-debates", action="store_true", help="Include debate history"
    )
    export_parser.add_argument(
        "--include-knowledge", action="store_true", help="Include knowledge base"
    )
    export_parser.add_argument("--include-audit", action="store_true", help="Include audit logs")
    export_parser.add_argument("--server", default=DEFAULT_API_URL, help="API server URL")
    export_parser.set_defaults(func=cmd_export)

    # Suspend tenant
    suspend_parser = tenant_subparsers.add_parser("suspend", help="Suspend a tenant")
    suspend_parser.add_argument("--tenant", "-t", required=True, help="Tenant ID")
    suspend_parser.add_argument("--reason", "-r", help="Suspension reason")
    suspend_parser.add_argument("--server", default=DEFAULT_API_URL, help="API server URL")
    suspend_parser.set_defaults(func=cmd_suspend)

    # Activate tenant
    activate_parser = tenant_subparsers.add_parser("activate", help="Activate a suspended tenant")
    activate_parser.add_argument("--tenant", "-t", required=True, help="Tenant ID")
    activate_parser.add_argument("--server", default=DEFAULT_API_URL, help="API server URL")
    activate_parser.set_defaults(func=cmd_activate)

    # Default handler for 'aragora tenant' without subcommand
    tenant_parser.set_defaults(func=lambda args: cmd_list(args))


def main(args: argparse.Namespace) -> int:
    """Main entry point for tenant CLI."""
    action = getattr(args, "tenant_action", None)

    if not action:
        # No subcommand, show list by default
        return cmd_list(args)

    # Dispatch to handler
    handler = getattr(args, "func", None)
    if handler:
        return handler(args)
    else:
        print(f"Unknown tenant action: {action}")
        return 1


__all__ = ["create_tenant_parser", "main"]
