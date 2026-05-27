#!/usr/bin/env python3
"""Local-only harvest report for sandboxed Codex automation output."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.publish_automation_handoffs import (  # noqa: E402
    DEFAULT_OUTBOX_DIR,
    DEFAULT_RECEIPT_DIR,
    TERMINAL_RECEIPT_STATUSES,
    Handoff,
    load_outbox_handoffs,
)

UTC = timezone.utc
SCHEMA_VERSION = 1
DEFAULT_PUBLISHER_CACHE = Path(".aragora/automation-github-status/latest.json")
DEFAULT_WORKTREE_HARVEST = Path(".aragora/worktree-harvest/latest.json")
PRODUCT_PROOF_TOKENS = (
    "aavt",
    "aft",
    "ask",
    "benchmark",
    "doctor",
    "dogfood",
    "eu ai",
    "external",
    "grok",
    "provider",
    "receipt",
    "validate-env",
)
META_TOOLING_TOKENS = (
    "agent_bridge",
    "automation",
    "handoff",
    "harvest",
    "merge",
    "outbox",
    "publisher",
    "queue",
    "settle",
    "steward",
)
CLASSIFICATION_ORDER = {
    "product_proof_candidate": 0,
    "ready_for_pr": 1,
    "needs_review": 2,
    "meta_tooling_candidate": 3,
    "issue_only": 4,
    "needs_rebase": 5,
    "protected_active_work": 6,
    "superseded_candidate": 7,
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _run(
    args: list[str],
    *,
    cwd: Path,
    timeout: int = 10,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout=stdout,
            stderr=stderr or f"command timed out after {timeout}s: {' '.join(args)}",
        )


def repo_root(path: Path) -> Path:
    proc = _run(["git", "rev-parse", "--show-toplevel"], cwd=path)
    if proc.returncode != 0:
        raise SystemExit(proc.stderr.strip() or "not a git repository")
    return Path(proc.stdout.strip()).resolve()


def _same_git_origin(left: Path, right: Path) -> bool:
    left_proc = _run(["git", "config", "--get", "remote.origin.url"], cwd=left)
    right_proc = _run(["git", "config", "--get", "remote.origin.url"], cwd=right)
    return (
        left_proc.returncode == 0
        and right_proc.returncode == 0
        and bool(left_proc.stdout.strip())
        and left_proc.stdout.strip() == right_proc.stdout.strip()
    )


def automation_state_root(root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        resolved = explicit.expanduser().resolve()
        return resolved if resolved.name == ".aragora" else resolved / ".aragora"
    if (root / ".aragora").is_dir():
        return root / ".aragora"

    configured = os.environ.get("ARAGORA_AUTOMATION_STATE_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.append(Path.home() / "Development" / "aragora")
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.name == ".aragora" and resolved.is_dir():
            return resolved
        if (resolved / ".aragora").is_dir() and _same_git_origin(root, resolved):
            return resolved / ".aragora"
    return root / ".aragora"


def _state_path(state_root: Path, explicit: Path | None, default: Path) -> Path:
    if explicit is not None:
        expanded = explicit.expanduser()
        return expanded.resolve() if expanded.is_absolute() else (state_root / expanded).resolve()
    if default.parts[:1] == (".aragora",):
        return state_root.joinpath(*default.parts[1:]).resolve()
    return (state_root / default).resolve()


def _json_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix == ".json")


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _terminal_receipt_count(receipt_dir: Path) -> int:
    count = 0
    for path in _json_files(receipt_dir):
        payload = _load_json(path)
        if payload is None:
            continue
        if str(payload.get("status") or "").strip().lower() in TERMINAL_RECEIPT_STATUSES:
            count += 1
    return count


def _branch_exists(root: Path, branch: str | None) -> bool:
    if not branch:
        return False
    proc = _run(["git", "rev-parse", "--verify", branch], cwd=root)
    return proc.returncode == 0


def _text_blob(handoff: Handoff) -> str:
    # The rendered body contains generic words such as "proof" and "receipt" for
    # most handoffs. Classify from durable intent fields to avoid over-ranking
    # routine meta-tooling as product-proof work.
    return f"{handoff.task_title}\n{handoff.branch or ''}\n{handoff.priority}".lower()


def _contains_signal(text: str, tokens: tuple[str, ...]) -> bool:
    for token in tokens:
        if " " in token or "-" in token or "_" in token:
            if token in text:
                return True
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text):
            return True
    return False


def classify_handoff(root: Path, handoff: Handoff) -> str:
    """Return one harvest classification for a handoff."""

    if handoff.source_kind != "outbox":
        return "needs_review"
    if not handoff.branch:
        return "issue_only"
    if not _branch_exists(root, handoff.branch):
        return "needs_rebase"

    text = _text_blob(handoff)
    if _contains_signal(text, PRODUCT_PROOF_TOKENS):
        return "product_proof_candidate"
    if _contains_signal(text, META_TOOLING_TOKENS):
        return "meta_tooling_candidate"
    return "ready_for_pr"


def _handoff_record(root: Path, handoff: Handoff) -> dict[str, Any]:
    classification = classify_handoff(root, handoff)
    return {
        "task": handoff.task_title,
        "source_file": handoff.source_file,
        "source_kind": handoff.source_kind,
        "branch": handoff.branch,
        "desired_head": handoff.desired_head,
        "priority": handoff.priority,
        "idempotency_key": handoff.idempotency_key,
        "classification": classification,
        "rank": CLASSIFICATION_ORDER.get(classification, 99),
    }


def _load_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return _load_json(path) or {}


def _run_json(args: list[str], *, cwd: Path, timeout: int) -> dict[str, Any]:
    proc = _run(args, cwd=cwd, timeout=timeout)
    if proc.returncode != 0:
        return {
            "error": proc.stderr.strip() or proc.stdout.strip() or "command failed",
            "returncode": proc.returncode,
        }
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {"error": "invalid json", "returncode": proc.returncode}
    return payload if isinstance(payload, dict) else {"payload": payload}


def _local_branch_count(root: Path, prefix: str = "codex/") -> int:
    proc = _run(
        ["git", "for-each-ref", "--format=%(refname:short)", f"refs/heads/{prefix}"],
        cwd=root,
    )
    if proc.returncode != 0:
        return 0
    return sum(1 for line in proc.stdout.splitlines() if line.strip())


def _int_from_mapping(value: Any, key: str) -> int:
    if not isinstance(value, Mapping):
        return 0
    raw = value.get(key)
    return raw if isinstance(raw, int) else 0


def _coordination_counts(operator_snapshot: Mapping[str, Any]) -> dict[str, int]:
    pending = operator_snapshot.get("pending_steering_messages")
    heartbeats = operator_snapshot.get("agent_heartbeats")
    return {
        "pending_steering_messages": _int_from_mapping(pending, "count"),
        "unread_steering_messages": _int_from_mapping(pending, "unread_message_count"),
        "fresh_agent_heartbeats": _int_from_mapping(heartbeats, "fresh_count"),
        "stale_agent_heartbeats": _int_from_mapping(heartbeats, "stale_count"),
    }


def _compact_publisher_cache(payload: Mapping[str, Any]) -> dict[str, Any]:
    local_queue = payload.get("local_queue")
    github_health = payload.get("github_health")
    return {
        "generated_at": payload.get("generated_at"),
        "local_queue": dict(local_queue) if isinstance(local_queue, Mapping) else {},
        "github_health": dict(github_health) if isinstance(github_health, Mapping) else {},
    }


def _compact_operator_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    pending = payload.get("pending_steering_messages")
    heartbeats = payload.get("agent_heartbeats")
    health = payload.get("health")
    summary = payload.get("summary")
    conflicts = payload.get("lane_conflicts")
    return {
        "health": dict(health) if isinstance(health, Mapping) else {},
        "lane_conflict_count": len(conflicts) if isinstance(conflicts, list) else 0,
        "pending_steering_messages": {
            "count": _int_from_mapping(pending, "count"),
            "unread_message_count": _int_from_mapping(pending, "unread_message_count"),
        },
        "agent_heartbeats": {
            "fresh_count": _int_from_mapping(heartbeats, "fresh_count"),
            "stale_count": _int_from_mapping(heartbeats, "stale_count"),
        },
        "summary": dict(summary) if isinstance(summary, Mapping) else {},
    }


def _compact_worktree_harvest(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    return {
        "schema": payload.get("schema"),
        "generated_at": payload.get("generated_at"),
        "summary": dict(summary) if isinstance(summary, Mapping) else {},
    }


def build_report(
    *,
    repo_root: Path,
    state_root: Path,
    outbox_dir: Path | None = None,
    receipt_dir: Path | None = None,
    publisher_cache: Mapping[str, Any] | None = None,
    operator_snapshot: Mapping[str, Any] | None = None,
    worktree_harvest: Mapping[str, Any] | None = None,
    top_limit: int = 20,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    state_root = state_root.resolve()
    outbox_root = _state_path(state_root, outbox_dir, DEFAULT_OUTBOX_DIR)
    receipt_root = _state_path(state_root, receipt_dir, DEFAULT_RECEIPT_DIR)
    handoffs = load_outbox_handoffs(
        repo_root,
        outbox_dir=outbox_root,
        receipt_dir=receipt_root,
    )
    records = [_handoff_record(repo_root, handoff) for handoff in handoffs]
    records.sort(key=lambda item: (item["rank"], item["task"], item["source_file"]))
    class_counts = Counter(str(item["classification"]) for item in records)
    coordination = _coordination_counts(operator_snapshot or {})
    counts = {
        "outbox_total_count": len(_json_files(outbox_root)),
        "active_handoff_count": len(handoffs),
        "terminal_receipt_count": _terminal_receipt_count(receipt_root),
        "local_codex_branch_count": _local_branch_count(repo_root),
        **coordination,
    }
    recommended = records[0] if records else None
    return {
        "automation_harvest_schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "repo": str(repo_root),
        "state_root": str(state_root),
        "outbox_dir": str(outbox_root),
        "receipt_dir": str(receipt_root),
        "counts": counts,
        "classification_counts": dict(sorted(class_counts.items())),
        "recommended_next_target": recommended,
        "top_next_handoffs": records[: max(top_limit, 0)],
        "publisher_cache": _compact_publisher_cache(publisher_cache or {}),
        "operator_snapshot_summary": _compact_operator_snapshot(operator_snapshot or {}),
        "worktree_harvest_summary": _compact_worktree_harvest(worktree_harvest or {}),
    }


def write_latest_report(report: Mapping[str, Any], *, state_root: Path) -> Path:
    state_root = state_root.expanduser().resolve()
    if state_root.name != ".aragora":
        state_root = state_root / ".aragora"
    output_dir = state_root / "automation-harvest"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "latest.json"
    output_path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def print_markdown(report: Mapping[str, Any]) -> None:
    raw_counts = report.get("counts")
    counts: Mapping[str, Any] = raw_counts if isinstance(raw_counts, Mapping) else {}
    print("# Codex Automation Harvest Report\n")
    print(f"- Repo: `{report.get('repo')}`")
    print(f"- State root: `{report.get('state_root')}`")
    print(f"- Active handoffs: `{counts.get('active_handoff_count', 0)}`")
    print(f"- Total outbox files: `{counts.get('outbox_total_count', 0)}`")
    print(f"- Terminal receipts: `{counts.get('terminal_receipt_count', 0)}`")
    print(f"- Local codex branches: `{counts.get('local_codex_branch_count', 0)}`")
    print(f"- Pending steering messages: `{counts.get('pending_steering_messages', 0)}`")
    print(f"- Fresh agent heartbeats: `{counts.get('fresh_agent_heartbeats', 0)}`")
    target = report.get("recommended_next_target")
    if isinstance(target, Mapping):
        print("\n## Recommended Next Target\n")
        print(f"- Task: `{target.get('task')}`")
        print(f"- Classification: `{target.get('classification')}`")
        print(f"- Branch: `{target.get('branch')}`")
        print(f"- Source: `{target.get('source_file')}`")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Path inside the repository")
    parser.add_argument(
        "--state-root",
        type=Path,
        default=None,
        help="Automation state root, either repo root or .aragora directory.",
    )
    parser.add_argument("--outbox-dir", type=Path, default=None)
    parser.add_argument("--receipt-dir", type=Path, default=None)
    parser.add_argument("--publisher-cache", type=Path, default=None)
    parser.add_argument("--operator-snapshot-json", type=Path, default=None)
    parser.add_argument("--worktree-harvest-json", type=Path, default=None)
    parser.add_argument("--top-limit", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    parser.add_argument("--markdown", action="store_true", help="Emit Markdown")
    parser.add_argument(
        "--write-latest",
        action="store_true",
        help="Write .aragora/automation-harvest/latest.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = repo_root(Path(args.repo))
    state_root = automation_state_root(root, args.state_root)
    publisher_cache_path = _state_path(
        state_root,
        args.publisher_cache,
        DEFAULT_PUBLISHER_CACHE,
    )
    operator_snapshot = (
        _load_optional_json(args.operator_snapshot_json)
        if args.operator_snapshot_json
        else _run_json(
            [
                "python3",
                "scripts/agent_bridge.py",
                "operator-snapshot",
                "--json",
                "--summary-only",
            ],
            cwd=root,
            timeout=15,
        )
    )
    worktree_harvest_path = _state_path(
        state_root,
        args.worktree_harvest_json,
        DEFAULT_WORKTREE_HARVEST,
    )
    report = build_report(
        repo_root=root,
        state_root=state_root,
        outbox_dir=args.outbox_dir,
        receipt_dir=args.receipt_dir,
        publisher_cache=_load_optional_json(publisher_cache_path),
        operator_snapshot=operator_snapshot,
        worktree_harvest=_load_optional_json(worktree_harvest_path),
        top_limit=args.top_limit,
    )
    if args.write_latest:
        report["latest_path"] = str(write_latest_report(report, state_root=state_root))
    if args.markdown and not args.json:
        print_markdown(report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
