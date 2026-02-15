"""CLI command for listing registered HTTP handlers and routes.

Usage:
    aragora handlers list [--tier <name>] [--json]
    aragora handlers routes [--tier <name>] [--json]
"""

from __future__ import annotations

import argparse
import json as json_mod
import logging

logger = logging.getLogger(__name__)


def add_handlers_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``handlers`` subcommand on the CLI parser."""
    parser = subparsers.add_parser(
        "handlers",
        help="List registered HTTP handlers and routes",
        description="Discover available API endpoints without reading source code.",
    )
    sub = parser.add_subparsers(dest="handlers_action")

    # handlers list
    list_p = sub.add_parser("list", help="List registered handler classes")
    list_p.add_argument("--tier", help="Filter by handler tier (core, extended, enterprise, experimental)")
    list_p.add_argument("--json", action="store_true", help="Output as JSON")
    list_p.set_defaults(func=_cmd_list_handlers)

    # handlers routes
    routes_p = sub.add_parser("routes", help="List all registered routes")
    routes_p.add_argument("--tier", help="Filter by handler tier")
    routes_p.add_argument("--json", action="store_true", help="Output as JSON")
    routes_p.set_defaults(func=_cmd_list_routes)

    # Default: show list
    parser.set_defaults(func=_cmd_list_handlers)


def _get_registry_data(tier_filter: str | None = None) -> list[dict]:
    """Collect handler metadata from the registry."""
    from aragora.server.handler_registry.core import HANDLER_TIERS, get_active_tiers
    from aragora.server.handler_registry import HANDLER_REGISTRY

    active_tiers = get_active_tiers()
    results: list[dict] = []

    for attr_name, handler_cls in HANDLER_REGISTRY:
        tier = HANDLER_TIERS.get(attr_name, "unknown")
        if tier_filter and tier != tier_filter:
            continue

        cls_name = handler_cls.__name__ if handler_cls else "(unavailable)"
        routes: list[str] = []
        if handler_cls and hasattr(handler_cls, "ROUTES"):
            routes = list(handler_cls.ROUTES)

        results.append({
            "attr": attr_name,
            "class": cls_name,
            "tier": tier,
            "routes": routes,
            "route_count": len(routes),
            "active": tier in active_tiers,
        })

    return results


def _cmd_list_handlers(args: argparse.Namespace) -> int:
    """List handler classes with tier and route count."""
    tier_filter = getattr(args, "tier", None)
    as_json = getattr(args, "json", False)
    data = _get_registry_data(tier_filter)

    if as_json:
        print(json_mod.dumps(data, indent=2))
        return 0

    # Table output
    print(f"{'Handler':<45} {'Tier':<14} {'Routes':>6}  {'Active'}")
    print("-" * 78)
    for entry in data:
        active = "yes" if entry["active"] else "no"
        print(f"{entry['class']:<45} {entry['tier']:<14} {entry['route_count']:>6}  {active}")
    print(f"\nTotal: {len(data)} handlers, {sum(e['route_count'] for e in data)} routes")
    return 0


def _cmd_list_routes(args: argparse.Namespace) -> int:
    """List all routes sorted by path."""
    tier_filter = getattr(args, "tier", None)
    as_json = getattr(args, "json", False)
    data = _get_registry_data(tier_filter)

    # Flatten to route-level entries
    route_entries: list[dict] = []
    for entry in data:
        for route in entry["routes"]:
            route_entries.append({
                "path": route,
                "handler": entry["class"],
                "tier": entry["tier"],
                "active": entry["active"],
            })

    route_entries.sort(key=lambda r: r["path"])

    if as_json:
        print(json_mod.dumps(route_entries, indent=2))
        return 0

    print(f"{'Path':<55} {'Handler':<30} {'Tier'}")
    print("-" * 95)
    for r in route_entries:
        print(f"{r['path']:<55} {r['handler']:<30} {r['tier']}")
    print(f"\nTotal: {len(route_entries)} routes")
    return 0
