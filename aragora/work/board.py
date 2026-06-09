"""Read-only WorkItem collection and graph assembly."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Callable

from aragora.work.models import SCHEMA_VERSION, WorkGraph, WorkItem
from aragora.work.scoring import build_recommendations, score_work_item
from aragora.work.sources import (
    collect_automation_outbox,
    collect_automation_receipts,
    collect_beads_and_convoys,
    collect_broker_runs,
    collect_github_prs,
    collect_mission_files,
    enrich_with_agent_bridge_lanes,
    resolve_work_state_root,
)

WORK_SOURCE_ALIASES = {
    "github-pr": "github_pr",
    "github_pr": "github_pr",
    "pr": "github_pr",
    "prs": "github_pr",
    "automation-outbox": "automation_outbox",
    "automation_outbox": "automation_outbox",
    "outbox": "automation_outbox",
    "automation-receipt": "automation_receipt",
    "automation-receipts": "automation_receipt",
    "automation_receipt": "automation_receipt",
    "automation_receipts": "automation_receipt",
    "receipt": "automation_receipt",
    "receipts": "automation_receipt",
    "broker-run": "broker_run",
    "broker-runs": "broker_run",
    "broker_run": "broker_run",
    "broker_runs": "broker_run",
    "bead": "bead",
    "beads": "bead",
    "convoy": "convoy",
    "convoys": "convoy",
    "bead-convoy": "bead_convoy",
    "beads-convoys": "bead_convoy",
    "bead_convoy": "bead_convoy",
    "beads_convoys": "bead_convoy",
    "mission": "mission_file",
    "missions": "mission_file",
    "mission-file": "mission_file",
    "mission-files": "mission_file",
    "mission_file": "mission_file",
    "mission_files": "mission_file",
}

WORK_SOURCE_CHOICES = tuple(sorted(set(WORK_SOURCE_ALIASES.values())))


def normalize_work_sources(sources: Iterable[str] | None) -> set[str] | None:
    """Normalize user-facing work-board source names to internal source ids."""
    if not sources:
        return None
    normalized: set[str] = set()
    unknown: list[str] = []
    for raw in sources:
        key = raw.strip().lower().replace(" ", "_")
        value = WORK_SOURCE_ALIASES.get(key)
        if value is None:
            unknown.append(raw)
        else:
            normalized.add(value)
    if unknown:
        choices = ", ".join(WORK_SOURCE_CHOICES)
        raise ValueError(f"unknown work source {unknown[0]!r}; expected one of: {choices}")
    return normalized or None


Collector = tuple[str, set[str], Callable[[Path], tuple[list[WorkItem], dict]], Path]


def collect_work_items(
    repo_root: Path | str,
    *,
    scope: str = "current",
    sources: Iterable[str] | None = None,
) -> tuple[list[WorkItem], list[dict]]:
    """Collect work items from all known read-only sources."""
    root = Path(repo_root).expanduser().resolve()
    state_root, state_health = resolve_work_state_root(root)
    source_filter = normalize_work_sources(sources)
    health: list[dict] = []
    items: list[WorkItem] = []
    collectors: list[Collector] = [
        ("github_pr", {"github_pr"}, collect_github_prs, root),
        ("automation_outbox", {"automation_outbox"}, collect_automation_outbox, state_root),
        (
            "automation_receipt",
            {"automation_receipt"},
            lambda r: collect_automation_receipts(r, scope=scope),
            state_root,
        ),
        ("broker_run", {"broker_run"}, lambda r: collect_broker_runs(r, scope=scope), state_root),
        (
            "bead_convoy",
            {"bead", "convoy", "bead_convoy"},
            lambda r: collect_beads_and_convoys(r, scope=scope),
            root,
        ),
        (
            "mission_file",
            {"mission_file"},
            lambda r: collect_mission_files(r, scope=scope),
            root,
        ),
    ]
    health.append(state_health)
    for collector_source, item_sources, collector, collector_root in collectors:
        if source_filter is not None and not source_filter.intersection(
            {collector_source, *item_sources}
        ):
            continue
        collected, source_health = collector(collector_root)
        if (
            source_filter is not None
            and collector_source == "bead_convoy"
            and "bead_convoy" not in source_filter
        ):
            collected = [item for item in collected if item.source in source_filter]
        items.extend(collected)
        health.append(source_health)
    if source_filter is None or "github_pr" in source_filter:
        items, lane_health = enrich_with_agent_bridge_lanes(state_root, items)
        health.append(lane_health)

    if scope == "current":
        items = [item for item in items if item.scope == "current"]

    # Keep IDs stable and unique if two legacy stores expose the same raw id.
    seen: dict[str, int] = {}
    for item in items:
        count = seen.get(item.id, 0)
        seen[item.id] = count + 1
        if count:
            item.id = f"{item.id}#{count + 1}"

    for item in items:
        item.score = score_work_item(item)
    items.sort(
        key=lambda it: (it.score.total if it.score else 0.0, it.updated_at or "", it.id),
        reverse=True,
    )
    return items, health


def build_work_graph(
    repo_root: Path | str,
    *,
    scope: str = "current",
    root_id: str | None = None,
    sources: Iterable[str] | None = None,
) -> WorkGraph:
    items, health = collect_work_items(repo_root, scope=scope, sources=sources)
    by_id = {item.id: item for item in items}
    edges: list[dict[str, str]] = []
    for item in items:
        for dep in item.dependencies:
            edges.append({"from": item.id, "to": dep, "relation": "depends_on"})
        branch = item.branch
        if branch:
            for other in items:
                if other is item:
                    continue
                if other.branch == branch:
                    edges.append({"from": item.id, "to": other.id, "relation": "same_branch"})

    if root_id:
        neighbors = {root_id}
        for edge in edges:
            if edge["from"] == root_id:
                neighbors.add(edge["to"])
            if edge["to"] == root_id:
                neighbors.add(edge["from"])
        items = [item for item in items if item.id in neighbors]
        edges = [edge for edge in edges if edge["from"] in neighbors and edge["to"] in neighbors]
        if root_id not in by_id:
            health.append(
                {"source": "work_graph", "status": "missing", "detail": f"{root_id} not found"}
            )

    return WorkGraph(
        items=items,
        edges=edges,
        source_health=health,
        root_id=root_id,
        schema_version=SCHEMA_VERSION,
    )


def build_robot_recommendations(
    repo_root: Path | str,
    *,
    scope: str = "current",
    sources: Iterable[str] | None = None,
) -> tuple[list, list[dict]]:
    items, health = collect_work_items(repo_root, scope=scope, sources=sources)
    return build_recommendations(items), health
