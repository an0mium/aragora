#!/usr/bin/env python3
"""Publish a recurring worktree value inventory snapshot.

Reads the JSON payload produced by ``scripts/codex_worktree_value_inventory.py``
or a previously published report from this script, then writes a small,
repo-tracked summary plus the full snapshot under
``docs/status/generated/worktree_value_inventory/`` together with a
``latest.json`` pointer. The companion status surface lives at
``docs/status/WORKTREE_VALUE_INVENTORY_STATUS.md``.

Read-only by default: ``--dry-run`` prints the summary without writing
anything. ``--input`` lets an operator hand in a pre-captured inventory
artifact (the inventory script can be slow when the legacy ``~/.codex/worktrees``
root has thousands of entries, so capturing once and publishing afterwards is
the recommended flow).

This publisher is intentionally additive:

- it does not run cleanup, harvest, or any GitHub-mutating action
- it does not import any aragora subpackage; it stays a pure stdlib script
- it never deletes prior dated artifacts (the directory acts as an audit
  trail, mirroring the rescue-productization and B0-truth conventions)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUBLISH_DIR = REPO_ROOT / "docs" / "status" / "generated" / "worktree_value_inventory"
DEFAULT_STATUS_DOC = REPO_ROOT / "docs" / "status" / "WORKTREE_VALUE_INVENTORY_STATUS.md"
SCHEMA_VERSION = 1


def _coerce_utc_datetime(value: str | None = None) -> dt.datetime:
    if value:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        parsed = dt.datetime.now(dt.UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC).replace(microsecond=0)


def normalize_generated_at(value: str | None = None) -> str:
    return _coerce_utc_datetime(value).isoformat().replace("+00:00", "Z")


def _stable_repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def load_inventory_payload(path: Path) -> dict[str, Any]:
    """Load a worktree value-inventory JSON document.

    The expected schema matches ``scripts/codex_worktree_value_inventory.py
    --json`` output: a top-level object with at least ``candidates`` (list)
    and ``roots`` (list of strings). The function also accepts a previously
    published report from this script and unwraps its preserved ``inventory``
    payload, which prevents ``latest.json`` dry-runs from silently reporting an
    empty inventory. Missing keys default to empty collections so the publisher
    never raises on partial or future-schema inventories.
    """
    if not path.exists():
        raise FileNotFoundError(f"inventory JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"inventory JSON at {path} must be a JSON object")
    nested_inventory = payload.get("inventory")
    if isinstance(nested_inventory, dict) and (
        "candidates" in nested_inventory or "roots" in nested_inventory
    ):
        return nested_inventory
    return payload


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _summarize_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    classifications: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    active_sessions = 0
    registered_worktrees = 0
    dirty_count = 0
    has_open_pr = 0
    cleanup_candidates: list[dict[str, Any]] = []
    harvest_candidates: list[dict[str, Any]] = []
    preserves: list[dict[str, Any]] = []

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        classification = str(cand.get("classification") or "").strip() or "unknown"
        decision = str(cand.get("decision") or "").strip() or "unknown"
        classifications[classification] += 1
        decisions[decision] += 1

        if bool(cand.get("active_session")):
            active_sessions += 1
        git_block = cand.get("git") or {}
        if isinstance(git_block, dict):
            if bool(git_block.get("registered_worktree")):
                registered_worktrees += 1
            if bool(git_block.get("dirty")):
                dirty_count += 1
        if cand.get("links") and isinstance(cand["links"], dict):
            if cand["links"].get("open_prs"):
                has_open_pr += 1

        row = {
            "path": str(cand.get("path") or ""),
            "branch": (git_block.get("branch") if isinstance(git_block, dict) else None),
            "classification": classification,
            "decision": decision,
            "ahead": (_coerce_int(git_block.get("ahead")) if isinstance(git_block, dict) else None),
            "behind": (
                _coerce_int(git_block.get("behind")) if isinstance(git_block, dict) else None
            ),
            "dirty": bool(git_block.get("dirty")) if isinstance(git_block, dict) else False,
            "active_session": bool(cand.get("active_session")),
            "mtime": str(cand.get("mtime") or ""),
        }
        if decision == "cleanup_candidate":
            cleanup_candidates.append(row)
        elif decision == "harvest_candidate":
            harvest_candidates.append(row)
        elif decision == "preserve":
            preserves.append(row)

    return {
        "total_candidates": len(candidates),
        "active_sessions": active_sessions,
        "registered_worktrees": registered_worktrees,
        "dirty_count": dirty_count,
        "candidates_with_open_pr": has_open_pr,
        "classifications": dict(classifications),
        "decisions": dict(decisions),
        "cleanup_candidates": cleanup_candidates,
        "harvest_candidates": harvest_candidates,
        "preserves": preserves,
    }


def build_published_report(
    *,
    inventory_path: Path,
    generated_at: str | None = None,
) -> dict[str, Any]:
    inventory = load_inventory_payload(inventory_path)
    candidates = inventory.get("candidates") or []
    if not isinstance(candidates, list):
        candidates = []
    summary = _summarize_candidates([c for c in candidates if isinstance(c, dict)])
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": normalize_generated_at(generated_at),
        "source_inventory_path": _stable_repo_path(inventory_path),
        "base": str(inventory.get("base") or ""),
        "base_sha": str(inventory.get("base_sha") or ""),
        "roots": [str(r) for r in (inventory.get("roots") or [])],
        "summary": summary,
        "inventory": inventory,
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def resolve_published_report_path(*, publish_dir: Path, generated_at: str) -> Path:
    timestamp = _coerce_utc_datetime(generated_at).strftime("%Y%m%dT%H%M%SZ")
    return publish_dir / f"worktree-value-inventory-{timestamp}.json"


def resolve_latest_report_path(*, publish_dir: Path) -> Path:
    return publish_dir / "latest.json"


def publish_report_bundle(
    *,
    publish_dir: Path,
    payload: dict[str, Any],
) -> dict[str, Path]:
    timestamped = write_json(
        resolve_published_report_path(
            publish_dir=publish_dir,
            generated_at=str(payload.get("generated_at") or ""),
        ),
        payload,
    )
    latest = write_json(
        resolve_latest_report_path(publish_dir=publish_dir),
        payload,
    )
    return {"timestamped": timestamped, "latest": latest}


def render_status_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    classifications = summary.get("classifications") or {}
    decisions = summary.get("decisions") or {}
    cleanup_candidates = summary.get("cleanup_candidates") or []
    harvest_candidates = summary.get("harvest_candidates") or []
    preserves = summary.get("preserves") or []

    lines: list[str] = []
    lines.append("# Worktree Value Inventory Status\n")
    lines.append(f"Last updated: {payload.get('generated_at', '')}\n")
    lines.append(
        "This repo-tracked status surface captures the most recent run of "
        "`scripts/codex_worktree_value_inventory.py` against the canonical "
        "and legacy Aragora worktree roots, classifying each checkout as "
        "preserve / harvest_candidate / cleanup_candidate.\n"
    )
    lines.append("## Summary\n")
    lines.append(f"- Source inventory: `{payload.get('source_inventory_path', '')}`")
    lines.append(f"- Base ref: `{payload.get('base', '')}`")
    lines.append(f"- Roots: `{', '.join(payload.get('roots') or []) or '(none)'}`")
    lines.append(f"- Total candidates: `{summary.get('total_candidates', 0)}`")
    lines.append(f"- Active sessions: `{summary.get('active_sessions', 0)}`")
    lines.append(f"- Registered worktrees: `{summary.get('registered_worktrees', 0)}`")
    lines.append(f"- Candidates with open PR: `{summary.get('candidates_with_open_pr', 0)}`")
    lines.append(f"- Dirty checkouts: `{summary.get('dirty_count', 0)}`\n")

    lines.append("## Classifications\n")
    if classifications:
        for key in sorted(classifications):
            lines.append(f"- `{key}`: {classifications[key]}")
    else:
        lines.append("- (no candidates)")
    lines.append("")

    lines.append("## Decisions\n")
    if decisions:
        for key in sorted(decisions):
            lines.append(f"- `{key}`: {decisions[key]}")
    else:
        lines.append("- (no candidates)")
    lines.append("")

    def _emit_rows(rows: list[dict[str, Any]], heading: str) -> None:
        lines.append(f"## {heading}\n")
        if not rows:
            lines.append("- (none)\n")
            return
        lines.append("| Path | Branch | Classification | Ahead | Behind | Dirty | Active |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for row in rows[:25]:
            path = str(row.get("path") or "").replace("|", "/")
            branch = str(row.get("branch") or "-").replace("|", "/")
            cls = str(row.get("classification") or "-")
            ahead = row.get("ahead")
            behind = row.get("behind")
            dirty = "yes" if row.get("dirty") else "no"
            active = "yes" if row.get("active_session") else "no"
            lines.append(
                f"| `{path[-60:]}` | `{branch}` | `{cls}` | "
                f"{ahead if ahead is not None else '-'} | "
                f"{behind if behind is not None else '-'} | "
                f"{dirty} | {active} |"
            )
        if len(rows) > 25:
            lines.append(f"\n*Truncated: {len(rows) - 25} additional rows omitted.*")
        lines.append("")

    _emit_rows(harvest_candidates, "Harvest Candidates")
    _emit_rows(cleanup_candidates, "Cleanup Candidates")
    _emit_rows(preserves, "Preserved (active or dirty)")

    lines.append("## Provenance\n")
    lines.append(
        "- Generator: `scripts/codex_worktree_value_inventory.py` "
        "(see PR #7250 for canonical+legacy smart roots; PR #7253/#7254 for "
        "managed-session lookup and foreign-worktree preservation)."
    )
    lines.append(
        "- Publisher: `scripts/publish_worktree_value_inventory.py` "
        "(this file). Read-only; never deletes worktrees or branches.\n"
    )
    return "\n".join(lines).rstrip() + "\n"


def write_status_markdown(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to an inventory JSON snapshot (output of "
        "scripts/codex_worktree_value_inventory.py --json) or a previously "
        "published worktree-value-inventory report.",
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=DEFAULT_PUBLISH_DIR,
        help=f"Directory for latest/timestamped inventory artifacts "
        f"(default: {DEFAULT_PUBLISH_DIR})",
    )
    parser.add_argument(
        "--status-doc",
        type=Path,
        default=DEFAULT_STATUS_DOC,
        help=f"Path to the human-readable status surface (default: {DEFAULT_STATUS_DOC})",
    )
    parser.add_argument(
        "--generated-at",
        default=None,
        help="Override the generated_at timestamp (ISO-8601 UTC).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the summary payload but do not write any artifacts.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_published_report(
        inventory_path=args.input.resolve(),
        generated_at=args.generated_at,
    )
    published: dict[str, Path] | None = None
    status_path: Path | None = None
    if not args.dry_run:
        published = publish_report_bundle(
            publish_dir=args.publish_dir.resolve(),
            payload=payload,
        )
        status_path = write_status_markdown(
            args.status_doc.resolve(),
            render_status_markdown(payload),
        )

    if args.json:
        out = {
            "schema_version": payload["schema_version"],
            "generated_at": payload["generated_at"],
            "summary": payload["summary"],
        }
        if published is not None:
            out["timestamped"] = _stable_repo_path(published["timestamped"])
            out["latest"] = _stable_repo_path(published["latest"])
        if status_path is not None:
            out["status_doc"] = _stable_repo_path(status_path)
        print(json.dumps(out, indent=2))
    else:
        summary = payload["summary"]
        print(
            f"worktree-value-inventory: total={summary['total_candidates']} "
            f"harvest={len(summary['harvest_candidates'])} "
            f"cleanup={len(summary['cleanup_candidates'])} "
            f"preserve={len(summary['preserves'])}"
        )
        if published is None:
            print("dry-run: artifacts not written")
        else:
            print(f"timestamped: {published['timestamped']}")
            print(f"latest:      {published['latest']}")
        if status_path is not None:
            print(f"status_doc:  {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
