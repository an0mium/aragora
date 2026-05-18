"""Read-only WorkItem collection and graph assembly."""

from __future__ import annotations

from pathlib import Path

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
)


def collect_work_items(
    repo_root: Path | str, *, scope: str = "current"
) -> tuple[list[WorkItem], list[dict]]:
    """Collect work items from all known read-only sources."""
    root = Path(repo_root)
    health: list[dict] = []
    items: list[WorkItem] = []
    collectors = [
        collect_github_prs,
        collect_automation_outbox,
        lambda r: collect_automation_receipts(r, scope=scope),
        lambda r: collect_broker_runs(r, scope=scope),
        lambda r: collect_beads_and_convoys(r, scope=scope),
        lambda r: collect_mission_files(r, scope=scope),
    ]
    for collector in collectors:
        collected, source_health = collector(root)
        items.extend(collected)
        health.append(source_health)
    items, lane_health = enrich_with_agent_bridge_lanes(root, items)
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
) -> WorkGraph:
    items, health = collect_work_items(repo_root, scope=scope)
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
    repo_root: Path | str, *, scope: str = "current"
) -> tuple[list, list[dict]]:
    items, health = collect_work_items(repo_root, scope=scope)
    return build_recommendations(items), health
