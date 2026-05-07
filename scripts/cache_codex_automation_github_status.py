#!/usr/bin/env python3
"""Write a local GitHub queue-status cache for sandboxed automations.

The Codex desktop automation sandbox can do useful local repair work but often
cannot reach GitHub. A normal user-shell publisher can run this helper and leave
fresh queue evidence in ``.aragora/automation-github-status/latest.json`` so
watcher/shepherd automations do not treat sandboxed ``gh`` failures as no-op
truth.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.github_cli_health import check_github_cli_health
from scripts.publish_automation_handoffs import _open_boss_ready_count
from scripts.publish_codex_automation_branches import (
    _open_codex_pr_is_unhealthy,
    _open_codex_prs,
)

UTC = timezone.utc
DEFAULT_REPO = "synaptent/aragora"
DEFAULT_LABELS = ("boss-ready",)
DEFAULT_MAX_OPEN_ISSUES = 16
DEFAULT_MAX_OPEN_PRS = 12
DEFAULT_OUTBOX_DIR = Path(".aragora/automation-outbox")
DEFAULT_RECEIPT_DIR = Path(".aragora/automation-receipts")
DEFAULT_OUTPUT = Path(".aragora/automation-github-status/latest.json")
TERMINAL_RECEIPT_STATUSES = {"published", "already_satisfied", "completed", "skipped"}
DUPLICATE_OUTBOX_EXAMPLE_LIMIT = 20


def _repo_root(path: Path) -> Path:
    import subprocess

    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "not a git repository")
    return Path(proc.stdout.strip()).resolve()


def _shared_state_root(repo_root: Path) -> Path:
    if (repo_root / ".aragora").is_dir():
        return repo_root
    configured = os.environ.get("ARAGORA_AUTOMATION_STATE_ROOT")
    if configured:
        configured_path = Path(configured).expanduser()
        try:
            resolved = configured_path.resolve()
        except OSError:
            resolved = configured_path
        if resolved.name == ".aragora" and resolved.is_dir():
            return resolved.parent
        if (resolved / ".aragora").is_dir():
            return resolved
    fallback = Path.home() / "Development" / "aragora"
    if (fallback / ".aragora").is_dir():
        return fallback
    return repo_root


def _resolve_state_path(repo_root: Path, value: Path | None, default: Path) -> Path:
    if value is not None:
        return value if value.is_absolute() else repo_root / value
    return _shared_state_root(repo_root) / default


def _json_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.glob("*.json") if item.is_file())


def _queue_file_key(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return path.stem
    if isinstance(payload, dict):
        key = str(payload.get("idempotency_key") or "").strip()
        if key:
            return key
    return path.stem


def _mapping_from_action(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        return parsed
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, Mapping):
        return parsed
    return None


def _local_evidence_mappings(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _queue_file_branch(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    for local_evidence in _local_evidence_mappings(payload.get("local_evidence")):
        branch = str(local_evidence.get("branch") or "").strip()
        if branch:
            return branch
    branch = str(payload.get("branch") or "").strip()
    if branch:
        return branch
    requested_action = _mapping_from_action(payload.get("requested_action"))
    if requested_action is None:
        return ""
    return str(requested_action.get("branch") or "").strip()


def _duplicate_key_summaries(keys: Sequence[str]) -> list[dict[str, Any]]:
    counts = Counter(keys)
    return [
        {"idempotency_key": key, "count": count}
        for key, count in sorted(counts.items())
        if count > 1
    ][:DUPLICATE_OUTBOX_EXAMPLE_LIMIT]


def _duplicate_branch_summaries(
    files: Sequence[tuple[Path, str, str]],
) -> list[dict[str, Any]]:
    by_branch: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for path, key, branch in files:
        if branch:
            by_branch[branch].append((path, key))
    summaries: list[dict[str, Any]] = []
    for branch, entries in sorted(by_branch.items()):
        if len(entries) <= 1:
            continue
        summaries.append(
            {
                "branch": branch,
                "count": len(entries),
                "files": [path.name for path, _key in entries],
                "idempotency_keys": [key for _path, key in entries],
            }
        )
    return summaries[:DUPLICATE_OUTBOX_EXAMPLE_LIMIT]


def _terminal_receipt_key(path: Path) -> str | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    status = str(payload.get("status") or "").strip().lower()
    if status not in TERMINAL_RECEIPT_STATUSES:
        return None
    key = str(payload.get("idempotency_key") or "").strip()
    return key or path.stem


def _receipt_state(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "file": path.name,
            "idempotency_key": path.stem,
            "status": "unparseable",
        }
    if not isinstance(payload, dict):
        return {
            "file": path.name,
            "idempotency_key": path.stem,
            "status": "invalid",
        }
    key = str(payload.get("idempotency_key") or path.stem).strip() or path.stem
    status = str(payload.get("status") or "").strip().lower() or "missing"
    return {
        "file": path.name,
        "idempotency_key": key,
        "status": status,
    }


def _local_queue_state(
    *,
    repo_root: Path,
    outbox_dir: Path | None,
    receipt_dir: Path | None,
) -> dict[str, Any]:
    outbox = _resolve_state_path(repo_root, outbox_dir, DEFAULT_OUTBOX_DIR)
    receipts = _resolve_state_path(repo_root, receipt_dir, DEFAULT_RECEIPT_DIR)
    outbox_files = _json_files(outbox)
    receipt_files = _json_files(receipts)
    receipt_states = [_receipt_state(item) for item in receipt_files]
    terminal_receipt_keys = {
        item["idempotency_key"]
        for item in receipt_states
        if item["status"] in TERMINAL_RECEIPT_STATUSES
    }
    nonterminal_receipts = [
        item for item in receipt_states if item["status"] not in TERMINAL_RECEIPT_STATUSES
    ]
    outbox_items = [
        (item, _queue_file_key(item), _queue_file_branch(item)) for item in outbox_files
    ]
    outbox_keys = [key for _item, key, _branch in outbox_items]
    outbox_branch_count = sum(1 for _item, _key, branch in outbox_items if branch)
    unique_outbox_branches = {branch for _item, _key, branch in outbox_items if branch}
    outbox_key_counts = Counter(outbox_keys)
    outbox_branch_counts = Counter(branch for _item, _key, branch in outbox_items if branch)
    duplicate_idempotency_keys = _duplicate_key_summaries(outbox_keys)
    duplicate_outbox_branches = _duplicate_branch_summaries(outbox_items)
    terminal_receipted_outbox_count = sum(1 for key in outbox_keys if key in terminal_receipt_keys)
    return {
        "outbox_dir": str(outbox),
        "receipt_dir": str(receipts),
        "outbox_count": len(outbox_files),
        "outbox_unique_idempotency_count": len(set(outbox_keys)),
        "outbox_duplicate_idempotency_count": sum(
            count - 1 for count in outbox_key_counts.values() if count > 1
        ),
        "outbox_duplicate_idempotency_keys": duplicate_idempotency_keys,
        "outbox_branch_count": outbox_branch_count,
        "outbox_unique_branch_count": len(unique_outbox_branches),
        "outbox_duplicate_branch_count": sum(
            count - 1 for count in outbox_branch_counts.values() if count > 1
        ),
        "outbox_duplicate_branches": duplicate_outbox_branches,
        "receipt_count": len(receipt_files),
        "terminal_receipt_count": len(terminal_receipt_keys),
        "nonterminal_receipt_count": len(nonterminal_receipts),
        "nonterminal_receipts": nonterminal_receipts,
        "terminal_receipted_outbox_count": terminal_receipted_outbox_count,
        "unreceipted_outbox_count": len(outbox_keys) - terminal_receipted_outbox_count,
    }


def _merge_state_counts(open_prs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in open_prs:
        state = str(item.get("mergeStateStatus") or "UNKNOWN").upper()
        counts[state] = counts.get(state, 0) + 1
    return counts


def build_status(
    *,
    repo_root: Path,
    github_repo: str,
    labels: Sequence[str],
    max_open_prs: int,
    max_open_issues: int,
    outbox_dir: Path | None = None,
    receipt_dir: Path | None = None,
) -> dict[str, Any]:
    health = check_github_cli_health(repo_root)
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "repo_root": str(repo_root),
        "github_repo": github_repo,
        "github_health": health.to_dict(),
        "limits": {
            "max_open_prs": max_open_prs,
            "max_open_issues": max_open_issues,
        },
        "local_queue": _local_queue_state(
            repo_root=repo_root,
            outbox_dir=outbox_dir,
            receipt_dir=receipt_dir,
        ),
    }

    if not health.ready:
        payload["github_queue"] = {
            "available": False,
            "reason": health.mode,
        }
        return payload

    open_prs = _open_codex_prs(repo_root, github_repo)
    unhealthy_open_pr_count = sum(1 for item in open_prs if _open_codex_pr_is_unhealthy(item))
    open_issue_count = _open_boss_ready_count(repo_root, github_repo, list(labels))
    payload["github_queue"] = {
        "available": True,
        "open_codex_pr_count": len(open_prs),
        "unhealthy_open_pr_count": unhealthy_open_pr_count,
        "all_open_prs_unhealthy": bool(open_prs) and unhealthy_open_pr_count == len(open_prs),
        "merge_state_counts": _merge_state_counts(open_prs),
        "open_issue_count": open_issue_count,
        "labels": list(labels),
        "pressure": {
            "open_pr_cap_reached": len(open_prs) >= max_open_prs,
            "open_issue_cap_reached": open_issue_count >= max_open_issues,
        },
        "open_pr_heads": [
            item.get("headRefName") for item in open_prs if isinstance(item.get("headRefName"), str)
        ],
    }
    return payload


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(path.parent),
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache GitHub queue status for local-only Codex automations."
    )
    parser.add_argument("--repo", default=".", help="Path inside the target repository")
    parser.add_argument("--github-repo", default=DEFAULT_REPO)
    parser.add_argument("--label", action="append", dest="labels", default=list(DEFAULT_LABELS))
    parser.add_argument("--max-open-prs", type=int, default=DEFAULT_MAX_OPEN_PRS)
    parser.add_argument("--max-open-issues", type=int, default=DEFAULT_MAX_OPEN_ISSUES)
    parser.add_argument("--outbox-dir", type=Path, default=None)
    parser.add_argument("--receipt-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", help="Print the cached payload")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = _repo_root(Path(args.repo))
    output = args.output if args.output.is_absolute() else repo_root / args.output
    payload = build_status(
        repo_root=repo_root,
        github_repo=args.github_repo,
        labels=args.labels,
        max_open_prs=args.max_open_prs,
        max_open_issues=args.max_open_issues,
        outbox_dir=args.outbox_dir,
        receipt_dir=args.receipt_dir,
    )
    write_status(output, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
