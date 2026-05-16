"""Read-only ``aragora work`` commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aragora.work.board import build_robot_recommendations, build_work_graph, collect_work_items
from aragora.work.models import SCHEMA_VERSION


def _repo_root(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "repo", ".")).expanduser().resolve()


def _emit(payload: dict[str, Any], *, as_json: bool) -> int:
    if as_json:
        print(json.dumps(payload, sort_keys=True, indent=2))
        return 0
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


def cmd_work_list(args: argparse.Namespace) -> int:
    items, health = collect_work_items(_repo_root(args), scope=args.scope)
    return _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "scope": args.scope,
            "count": len(items),
            "items": [item.to_dict() for item in items],
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
    return _emit(
        {
            "schema_version": SCHEMA_VERSION,
            "scope": "current",
            "count": len(recommendations),
            "recommendations": [rec.to_dict() for rec in recommendations],
            "source_health": health,
            "mutations": [],
        },
        as_json=getattr(args, "json", False),
    )
