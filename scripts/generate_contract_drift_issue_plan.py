#!/usr/bin/env python3
"""Generate owner-prioritized issue plan from contract drift backlog."""

from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _issue_title(ticket: dict[str, Any]) -> str:
    source = str(ticket.get("source", "")).replace("_", " ")
    domain = ticket.get("domain", "unknown")
    count = ticket.get("open_items", 0)
    return f"Contract drift: {source} - {domain} ({count} items)"


def build_plan(backlog: dict[str, Any], max_tickets: int = 40) -> dict[str, Any]:
    tickets = list(backlog.get("tickets", []))
    tickets = sorted(
        tickets,
        key=lambda t: (
            -int(t.get("open_items", 0)),
            str(t.get("owner", "")),
            str(t.get("source", "")),
            str(t.get("domain", "")),
        ),
    )[:max_tickets]

    owners = sorted(
        backlog.get("counts_by_owner", {}).items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    sources = sorted(
        backlog.get("counts_by_source", {}).items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    waves: list[dict[str, Any]] = []
    wave_count = 4
    if tickets:
        wave_size = int(math.ceil(len(tickets) / wave_count))
        for idx in range(wave_count):
            start = idx * wave_size
            end = min(len(tickets), (idx + 1) * wave_size)
            wave_tickets = tickets[start:end]
            if not wave_tickets:
                continue
            wave_items = sum(int(t.get("open_items", 0)) for t in wave_tickets)
            wave_owners: dict[str, int] = {}
            for ticket in wave_tickets:
                owner = str(ticket.get("owner", "@team-platform"))
                wave_owners[owner] = wave_owners.get(owner, 0) + int(ticket.get("open_items", 0))

            waves.append(
                {
                    "wave": idx + 1,
                    "total_open_items": wave_items,
                    "focus_owners": [owner for owner, _ in sorted(wave_owners.items(), key=lambda kv: kv[1], reverse=True)[:3]],
                    "tickets": [
                        {
                            **ticket,
                            "issue_title": _issue_title(ticket),
                        }
                        for ticket in wave_tickets
                    ],
                }
            )

    return {
        "generated_on": date.today().isoformat(),
        "program_total_items": backlog.get("total_items", 0),
        "max_seed_tickets": max_tickets,
        "owner_priority": [{"owner": owner, "open_items": count} for owner, count in owners],
        "source_priority": [{"source": source, "open_items": count} for source, count in sources],
        "waves": waves,
    }


def to_markdown(plan: dict[str, Any]) -> str:
    lines = ["# Contract Drift Issue Plan", ""]
    lines.append(f"Generated: `{plan['generated_on']}`")
    lines.append(f"Program total open items: **{plan['program_total_items']}**")
    lines.append("")

    lines.append("## Owner Priority")
    lines.append("")
    lines.append("| Owner | Open Items |")
    lines.append("|---|---:|")
    for row in plan.get("owner_priority", [])[:10]:
        lines.append(f"| {row['owner']} | {row['open_items']} |")
    lines.append("")

    lines.append("## Source Priority")
    lines.append("")
    lines.append("| Source | Open Items |")
    lines.append("|---|---:|")
    for row in plan.get("source_priority", []):
        lines.append(f"| `{row['source']}` | {row['open_items']} |")
    lines.append("")

    lines.append("## Wave Plan")
    lines.append("")
    for wave in plan.get("waves", []):
        focus = ", ".join(wave.get("focus_owners", []))
        lines.append(f"### Wave {wave['wave']}")
        lines.append("")
        lines.append(f"Focus owners: {focus}")
        lines.append(f"Open items in wave: **{wave['total_open_items']}**")
        lines.append("")
        lines.append("| Ticket | Title | Owner | Source | Domain | Open Items | Weekly Burn |")
        lines.append("|---|---|---|---|---|---:|---:|")
        for ticket in wave.get("tickets", []):
            lines.append(
                f"| {ticket['ticket']} | {ticket['issue_title']} | {ticket['owner']} | "
                f"`{ticket['source']}` | `{ticket['domain']}` | {ticket['open_items']} | "
                f"{ticket.get('target_weekly_burn', 1)} |"
            )
        lines.append("")

    lines.append("## Execution Notes")
    lines.append("")
    lines.append("1. Work tickets in wave order; complete Wave 1 before opening new Wave 2 work.")
    lines.append("2. Each merged PR must not regress strict drift baselines and should reduce at least one source bucket when touching contract surfaces.")
    lines.append("3. Re-run backlog and issue plan generation weekly to refresh priorities.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate contract drift issue plan")
    parser.add_argument(
        "--backlog-json",
        default="docs/status/CONTRACT_DRIFT_BACKLOG.json",
        help="Path to backlog JSON",
    )
    parser.add_argument(
        "--markdown-out",
        default="docs/status/CONTRACT_DRIFT_ISSUE_PLAN.md",
        help="Path to markdown output",
    )
    parser.add_argument(
        "--json-out",
        default="docs/status/CONTRACT_DRIFT_ISSUE_PLAN.json",
        help="Path to JSON output",
    )
    parser.add_argument("--max-tickets", type=int, default=40, help="Maximum issue seeds")
    args = parser.parse_args()

    backlog = _load(Path(args.backlog_json))
    if not backlog:
        raise SystemExit(f"Backlog file missing or empty: {args.backlog_json}")

    plan = build_plan(backlog, max_tickets=args.max_tickets)
    md = to_markdown(plan)

    md_path = Path(args.markdown_out)
    json_path = Path(args.json_out)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md)
    json_path.write_text(json.dumps(plan, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
