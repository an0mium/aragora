#!/usr/bin/env python3
"""Generate contract drift backlog and burndown targets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OWNER_MAP = {
    "analytics": "@team-analytics",
    "audit": "@team-risk",
    "auth": "@team-platform",
    "billing": "@team-finops",
    "chat": "@team-integrations",
    "connectors": "@team-integrations",
    "control-plane": "@team-platform",
    "costs": "@team-finops",
    "crm": "@team-integrations",
    "debates": "@team-core",
    "ecommerce": "@team-integrations",
    "email": "@team-integrations",
    "gateway": "@team-platform",
    "integrations": "@team-integrations",
    "knowledge": "@team-core",
    "learning": "@team-core",
    "marketplace": "@team-growth",
    "memory": "@team-core",
    "notifications": "@team-integrations",
    "outlook": "@team-integrations",
    "partners": "@team-growth",
    "rbac": "@team-platform",
    "sme": "@team-platform",
    "sso": "@team-platform",
    "workflows": "@team-core",
}


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _route_domain(path: str) -> str:
    parts = [p for p in path.strip("/").split("/") if p]
    if parts[:2] == ["api", "v1"] and len(parts) > 2:
        return parts[2]
    if parts and parts[0] == "api" and len(parts) > 1:
        return parts[1]
    return parts[0] if parts else "root"


def _entry_domain(entry: str) -> str:
    # verify_sdk_contracts baseline entries are "METHOD /api/..."
    if " " in entry:
        _, path = entry.split(" ", 1)
        return _route_domain(path)
    return _route_domain(entry)


def _owner(domain: str) -> str:
    return OWNER_MAP.get(domain, "@team-platform")


def _targets(start_total: int, weeks: int = 8, weekly_reduction: float = 0.1) -> list[dict[str, Any]]:
    rows = []
    current = start_total
    start = date.today()
    for w in range(1, weeks + 1):
        current = max(0, int(round(current * (1.0 - weekly_reduction))))
        rows.append(
            {
                "week": w,
                "date": (start + timedelta(days=7 * w)).isoformat(),
                "target_max_open_items": current,
            }
        )
    return rows


def build_backlog() -> dict[str, Any]:
    verify = _load(PROJECT_ROOT / "scripts/baselines/verify_sdk_contracts.json")
    routes = _load(PROJECT_ROOT / "scripts/baselines/validate_openapi_routes.json")
    parity = _load(PROJECT_ROOT / "scripts/baselines/check_sdk_parity.json")

    items_by_source: dict[str, list[str]] = {
        "verify_python_sdk_drift": list(verify.get("python_sdk_drift", [])),
        "verify_typescript_sdk_drift": list(verify.get("typescript_sdk_drift", [])),
        "routes_missing_in_spec": list(routes.get("missing_in_spec", [])),
        "routes_orphaned_in_spec": list(routes.get("orphaned_in_spec", [])),
        "sdk_missing_from_both": list(parity.get("missing_from_both_sdks", [])),
    }

    domain_counts: dict[str, Counter[str]] = {}
    owner_totals: Counter[str] = Counter()
    total_items = 0
    for source, items in items_by_source.items():
        c: Counter[str] = Counter()
        for item in items:
            domain = _entry_domain(item)
            c[domain] += 1
            owner_totals[_owner(domain)] += 1
        domain_counts[source] = c
        total_items += len(items)

    backlog_rows: list[dict[str, Any]] = []
    ticket_id = 1
    for source, counter in domain_counts.items():
        for domain, count in counter.most_common():
            backlog_rows.append(
                {
                    "ticket": f"CD-{ticket_id:03d}",
                    "source": source,
                    "domain": domain,
                    "owner": _owner(domain),
                    "open_items": count,
                    "target_weekly_burn": max(1, int(round(count * 0.1))),
                }
            )
            ticket_id += 1

    summary = {
        "generated_on": date.today().isoformat(),
        "total_items": total_items,
        "counts_by_source": {k: len(v) for k, v in items_by_source.items()},
        "counts_by_owner": dict(owner_totals),
        "weekly_targets": _targets(total_items, weeks=8, weekly_reduction=0.1),
        "tickets": backlog_rows,
    }
    return summary


def to_markdown(data: dict[str, Any]) -> str:
    lines = ["# Contract Drift Backlog Program", ""]
    lines.append(f"Generated: `{data['generated_on']}`")
    lines.append(f"Total open items: **{data['total_items']}**")
    lines.append("")
    lines.append("## Source Counts")
    lines.append("")
    lines.append("| Source | Open Items |")
    lines.append("|---|---:|")
    for source, count in data["counts_by_source"].items():
        lines.append(f"| `{source}` | {count} |")
    lines.append("")
    lines.append("## Weekly Burndown Targets (10%/week)")
    lines.append("")
    lines.append("| Week | Date | Target Max Open Items |")
    lines.append("|---:|---|---:|")
    for row in data["weekly_targets"]:
        lines.append(f"| {row['week']} | {row['date']} | {row['target_max_open_items']} |")
    lines.append("")
    lines.append("## Ownership Summary")
    lines.append("")
    lines.append("| Owner | Open Items |")
    lines.append("|---|---:|")
    for owner, count in sorted(data["counts_by_owner"].items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"| {owner} | {count} |")
    lines.append("")
    lines.append("## Ticket Seeds")
    lines.append("")
    lines.append("| Ticket | Source | Domain | Owner | Open Items | Weekly Burn Target |")
    lines.append("|---|---|---|---|---:|---:|")
    for row in data["tickets"][:80]:
        lines.append(
            f"| {row['ticket']} | `{row['source']}` | `{row['domain']}` | {row['owner']} | {row['open_items']} | {row['target_weekly_burn']} |"
        )
    lines.append("")
    lines.append("_Note: prioritize top open-item domains per source first._")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate contract drift backlog artifacts")
    parser.add_argument(
        "--markdown-out",
        default="docs/status/CONTRACT_DRIFT_BACKLOG.md",
        help="Path to markdown output",
    )
    parser.add_argument(
        "--json-out",
        default="docs/status/CONTRACT_DRIFT_BACKLOG.json",
        help="Path to JSON output",
    )
    args = parser.parse_args()

    data = build_backlog()
    md = to_markdown(data)

    md_path = PROJECT_ROOT / args.markdown_out
    json_path = PROJECT_ROOT / args.json_out
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md)
    json_path.write_text(json.dumps(data, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
