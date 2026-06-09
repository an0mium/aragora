"""Read-only ``aragora work`` commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from aragora.work.board import build_robot_recommendations, build_work_graph, collect_work_items
from aragora.work.models import SCHEMA_VERSION


def _repo_root(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "repo", ".")).expanduser().resolve()


def _render_human(payload: dict[str, Any]) -> str:
    """Render a work-board payload as a compact, human-readable summary.

    Mirrors the JSON payload's information without raw braces. The ``--json``
    flag selects the machine-stable JSON form; the default is this view.
    """
    lines: list[str] = []
    schema = payload.get("schema_version")
    if schema:
        lines.append(f"schema: {schema}")

    def _fmt_item(item: dict[str, Any], *, indent: str = "  ") -> None:
        item_id = item.get("id", "?")
        title = item.get("title") or ""
        status = item.get("status") or "unknown"
        header = f"{indent}- {item_id} [{status}]"
        if title:
            header += f" {title}"
        lines.append(header)
        owner = item.get("owner")
        if owner:
            lines.append(f"{indent}  owner: {owner}")
        url = item.get("url")
        if url:
            lines.append(f"{indent}  url: {url}")
        deps = item.get("dependencies") or []
        if deps:
            lines.append(f"{indent}  depends_on: {', '.join(deps)}")

    # work list / robot share a top-level scope + count.
    if "scope" in payload:
        lines.append(f"scope: {payload['scope']}")
    if "count" in payload:
        lines.append(f"count: {payload['count']}")
    if "emitted_count" in payload and payload.get("emitted_count") != payload.get("count"):
        limit = payload.get("limit")
        suffix = f" (limit: {limit})" if limit is not None else ""
        lines.append(f"emitted_count: {payload['emitted_count']}{suffix}")

    if "items" in payload:
        items = payload.get("items") or []
        if not items:
            lines.append("items: (none)")
        else:
            lines.append("items:")
            for item in items:
                _fmt_item(item)

    if "recommendations" in payload:
        recs = payload.get("recommendations") or []
        if not recs:
            lines.append("recommendations: (none)")
        else:
            lines.append("recommendations:")
            for rec in recs:
                lines.append(
                    f"  #{rec.get('rank')} {rec.get('item_id')} "
                    f"[{rec.get('classification')}] -> {rec.get('action')} "
                    f"({rec.get('priority')})"
                )
                blockers = rec.get("blockers") or []
                if blockers:
                    lines.append(f"    blockers: {', '.join(blockers)}")

    # work show emits a single item under "found"/"item".
    if "found" in payload:
        lines.append(f"id: {payload.get('id')}")
        lines.append(f"found: {payload.get('found')}")
        item = payload.get("item")
        if item:
            _fmt_item(item)

    # work graph emits edges.
    if "edges" in payload:
        edges = payload.get("edges") or []
        if edges:
            lines.append("edges:")
            for edge in edges:
                lines.append(f"  {edge.get('from')} --{edge.get('relation')}--> {edge.get('to')}")
        else:
            lines.append("edges: (none)")

    for health in payload.get("source_health") or []:
        status = health.get("status")
        if status and status != "ok":
            lines.append(f"source {health.get('source')}: {status}")

    return "\n".join(lines) if lines else "(no work)"


def _emit(payload: dict[str, Any], *, as_json: bool) -> int:
    output = json.dumps(payload, sort_keys=True, indent=2) if as_json else _render_human(payload)
    try:
        print(output)
    except BrokenPipeError:
        _mute_stdout_after_broken_pipe()
    return 0


def _select_limited_items(
    items: list[dict[str, Any]], *, limit: int | None, root_id: str | None
) -> list[dict[str, Any]]:
    if limit is None:
        return list(items)
    selected = list(items[:limit])
    if limit <= 0 or not root_id or any(item.get("id") == root_id for item in selected):
        return selected
    root_item = next((item for item in items if item.get("id") == root_id), None)
    if not root_item:
        return selected
    if len(selected) >= limit:
        selected = selected[: max(limit - 1, 0)]
    selected.append(root_item)
    return selected


def _edge_in_selected(edge: dict[str, Any], selected_ids: set[str]) -> bool:
    return edge.get("from") in selected_ids and edge.get("to") in selected_ids


def _limit_graph_payload(payload: dict[str, Any], *, limit: int | None) -> dict[str, Any]:
    items = list(payload.get("items") or [])
    edges = list(payload.get("edges") or [])
    root_id = payload.get("root_id")
    selected_items = _select_limited_items(items, limit=limit, root_id=root_id)
    selected_ids = {str(item.get("id")) for item in selected_items}
    selected_edges = [edge for edge in edges if _edge_in_selected(edge, selected_ids)]
    if limit is None:
        selected_edges = edges

    limited = dict(payload)
    limited.update(
        {
            "item_count": len(items),
            "edge_count": len(edges),
            "emitted_item_count": len(selected_items),
            "emitted_edge_count": len(selected_edges),
            "limit": limit,
            "items": selected_items,
            "edges": selected_edges,
            "items_omitted": len(selected_items) < len(items),
            "edges_omitted": len(selected_edges) < len(edges),
        }
    )
    return limited


def _compact_work_item(item: dict[str, Any]) -> dict[str, Any]:
    score = item.get("score") or {}
    compact = {
        "id": item.get("id"),
        "source": item.get("source"),
        "item_type": item.get("item_type"),
        "status": item.get("status"),
        "scope": item.get("scope"),
        "owner": item.get("owner"),
        "branch": item.get("branch"),
        "title": item.get("title"),
    }
    if "total" in score:
        compact["score_total"] = score.get("total")
    return {key: value for key, value in compact.items() if value not in (None, "", [])}


def _summary_graph_payload(payload: dict[str, Any], *, limit: int | None) -> dict[str, Any]:
    example_limit = 10 if limit is None else limit
    items = list(payload.get("items") or [])
    edges = list(payload.get("edges") or [])
    item_examples = _select_limited_items(
        items,
        limit=example_limit,
        root_id=payload.get("root_id"),
    )
    edge_examples = edges[:example_limit]
    return {
        "schema_version": payload.get("schema_version"),
        "root_id": payload.get("root_id"),
        "item_count": len(items),
        "edge_count": len(edges),
        "emitted_item_count": len(item_examples),
        "emitted_edge_count": len(edge_examples),
        "limit": example_limit,
        "item_examples": [_compact_work_item(item) for item in item_examples],
        "edge_examples": list(edge_examples),
        "items_omitted": len(item_examples) < len(items),
        "edges_omitted": len(edge_examples) < len(edges),
        "source_health": list(payload.get("source_health") or []),
        "details_omitted": True,
    }


def _mute_stdout_after_broken_pipe() -> None:
    """Avoid interpreter-shutdown tracebacks after downstream pipes close."""
    try:
        sys.stdout.close()
    except OSError:
        pass
    sys.stdout = open(os.devnull, "w", encoding="utf-8")


def cmd_work_list(args: argparse.Namespace) -> int:
    items, health = collect_work_items(_repo_root(args), scope=args.scope)
    limit = getattr(args, "limit", None)
    emitted_items = items[:limit] if limit is not None else items
    return _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "scope": args.scope,
            "count": len(items),
            "emitted_count": len(emitted_items),
            "limit": limit,
            "items": [item.to_dict() for item in emitted_items],
            "source_health": health,
        },
        as_json=getattr(args, "json", False),
    )


def cmd_work_show(args: argparse.Namespace) -> int:
    items, health = collect_work_items(_repo_root(args), scope="all")
    item = next((candidate for candidate in items if candidate.id == args.work_id), None)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "id": args.work_id,
        "found": item is not None,
        "item": item.to_dict() if item else None,
        "source_health": health,
    }
    return _emit(payload, as_json=getattr(args, "json", False))


def cmd_work_graph(args: argparse.Namespace) -> int:
    graph = build_work_graph(_repo_root(args), scope="all", root_id=getattr(args, "work_id", None))
    payload = graph.to_dict()
    limit = getattr(args, "limit", None)
    if getattr(args, "summary_only", False):
        payload = _summary_graph_payload(payload, limit=limit)
    elif limit is not None:
        payload = _limit_graph_payload(payload, limit=limit)
    return _emit(payload, as_json=getattr(args, "json", False))


def cmd_work_robot(args: argparse.Namespace) -> int:
    recommendations, health = build_robot_recommendations(_repo_root(args), scope="current")
    limit = getattr(args, "limit", None)
    emitted_recommendations = recommendations[:limit] if limit is not None else recommendations
    return _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "scope": "current",
            "count": len(recommendations),
            "emitted_count": len(emitted_recommendations),
            "limit": limit,
            "recommendations": [rec.to_dict() for rec in emitted_recommendations],
            "source_health": health,
            "mutations": [],
        },
        as_json=getattr(args, "json", False),
    )
