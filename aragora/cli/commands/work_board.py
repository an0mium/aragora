"""Read-only ``aragora work`` commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from aragora.work.board import build_robot_recommendations, build_work_graph, collect_work_items
from aragora.work.models import SCHEMA_VERSION

SUMMARY_ONLY_DEFAULT_TOP_LIMIT = 20


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

    if "recommendation_summary" in payload:
        summary = payload.get("recommendation_summary") or {}
        lines.append("recommendation_summary:")
        for key in ("by_classification", "by_action", "by_priority"):
            values = summary.get(key) or {}
            if values:
                rendered = ", ".join(f"{name}={count}" for name, count in values.items())
                lines.append(f"  {key}: {rendered}")
        top = summary.get("top") or []
        if top:
            lines.append("  top:")
            for rec in top:
                lines.append(
                    f"    #{rec.get('rank')} {rec.get('item_id')} "
                    f"[{rec.get('classification')}] -> {rec.get('action')}"
                )

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


def _mute_stdout_after_broken_pipe() -> None:
    """Avoid interpreter-shutdown tracebacks after downstream pipes close."""
    try:
        sys.stdout.close()
    except OSError:
        pass
    sys.stdout = open(os.devnull, "w", encoding="utf-8")


def _counts(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(value for value in values if value).items()))


def _compact_recommendation(rec: Any) -> dict[str, Any]:
    item = rec.item
    row: dict[str, Any] = {
        "rank": rec.rank,
        "item_id": rec.item_id,
        "classification": rec.classification,
        "action": rec.action,
        "priority": rec.priority,
    }
    if item is not None:
        if item.source:
            row["source"] = item.source
        if item.branch:
            row["branch"] = item.branch
    if rec.blockers:
        row["blocker_count"] = len(rec.blockers)
    return row


def _recommendation_summary(recommendations: list[Any], emitted: list[Any]) -> dict[str, Any]:
    return {
        "by_classification": _counts([rec.classification for rec in recommendations]),
        "by_action": _counts([rec.action for rec in recommendations]),
        "by_priority": _counts([rec.priority for rec in recommendations]),
        "top": [_compact_recommendation(rec) for rec in emitted],
    }


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
    return _emit(graph.to_dict(), as_json=getattr(args, "json", False))


def cmd_work_robot(args: argparse.Namespace) -> int:
    recommendations, health = build_robot_recommendations(_repo_root(args), scope="current")
    limit = getattr(args, "limit", None)
    summary_only = getattr(args, "summary_only", False)
    summary_limit = SUMMARY_ONLY_DEFAULT_TOP_LIMIT if summary_only and limit is None else None
    effective_limit = limit if limit is not None else summary_limit
    emitted_recommendations = (
        recommendations[:effective_limit] if effective_limit is not None else recommendations
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scope": "current",
        "count": len(recommendations),
        "emitted_count": len(emitted_recommendations),
        "limit": limit,
        "source_health": health,
        "mutations": [],
    }
    if summary_limit is not None:
        payload["summary_limit"] = summary_limit
    if summary_only:
        payload["recommendation_summary"] = _recommendation_summary(
            recommendations, emitted_recommendations
        )
    else:
        payload["recommendations"] = [rec.to_dict() for rec in emitted_recommendations]
    return _emit(payload, as_json=getattr(args, "json", False))
