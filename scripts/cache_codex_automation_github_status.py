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
import subprocess
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


def _has_queue_state_dirs(state_root: Path) -> bool:
    state_dir = state_root if state_root.name == ".aragora" else state_root / ".aragora"
    return (state_dir / "automation-outbox").is_dir() or (
        state_dir / "automation-receipts"
    ).is_dir()


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
    if _has_queue_state_dirs(repo_root):
        return repo_root
    configured = os.environ.get("ARAGORA_AUTOMATION_STATE_ROOT")
    if configured:
        configured_root = Path(configured).expanduser()
        if configured_root.name == ".aragora" and configured_root.is_dir():
            return configured_root.resolve()
        if (configured_root / ".aragora").is_dir():
            return configured_root.resolve()
    fallback = Path.home() / "Development" / "aragora"
    if _has_queue_state_dirs(fallback):
        return fallback
    return repo_root


def _automation_state_default_path(state_root: Path, default: Path) -> Path:
    expanded = state_root.expanduser()
    if default.parts[:1] == (".aragora",) and expanded.name == ".aragora":
        return expanded.joinpath(*default.parts[1:])
    return expanded / default


def _resolve_state_path(repo_root: Path, value: Path | None, default: Path) -> Path:
    if value is not None:
        return value if value.is_absolute() else repo_root / value
    return _automation_state_default_path(_shared_state_root(repo_root), default)


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


def _payload_head(payload: Mapping[str, Any], keys: Sequence[str]) -> str:
    for local_evidence in _local_evidence_mappings(payload.get("local_evidence")):
        for key in keys:
            head = str(local_evidence.get(key) or "").strip()
            if head:
                return head

    requested_action = _mapping_from_action(payload.get("requested_action"))
    if requested_action is not None:
        for key in keys:
            head = str(requested_action.get(key) or "").strip()
            if head:
                return head

    for key in keys:
        head = str(payload.get(key) or "").strip()
        if head:
            return head
    return ""


def _desired_head_from_outbox(payload: Mapping[str, Any]) -> str:
    return _payload_head(
        payload,
        ("desired_head_sha", "target_head_sha", "head_sha", "head", "commit"),
    )


def _head_from_receipt(payload: Mapping[str, Any]) -> str:
    for key in (
        "published_head_sha",
        "target_pr_head_sha",
        "headRefOid",
        "head_ref_oid",
        "head_sha",
        "head",
        "commit",
    ):
        head = str(payload.get(key) or "").strip()
        if head:
            return head
    return ""


def _requests_target_pr(payload: Mapping[str, Any]) -> bool:
    if str(payload.get("target_pr") or "").strip():
        return True
    requested_action = _mapping_from_action(payload.get("requested_action"))
    if requested_action is None:
        return False
    return bool(str(requested_action.get("target_pr") or "").strip())


def _remote_tracking_head(repo_root: Path, branch: str) -> str:
    branch = branch.strip()
    if not branch:
        return ""
    remote_ref = branch if branch.startswith("origin/") else f"origin/{branch}"
    proc = subprocess.run(
        [
            "git",
            "rev-parse",
            "--verify",
            "--quiet",
            "--end-of-options",
            f"{remote_ref}^{{commit}}",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _stale_target_pr_receipt_evidence(
    *,
    repo_root: Path,
    outbox_payload: Mapping[str, Any],
    branch: str,
    receipt_payload: Mapping[str, Any],
) -> dict[str, str] | None:
    reason = str(receipt_payload.get("reason") or "").strip().lower()
    if reason != "target_open_pr" or not _requests_target_pr(outbox_payload):
        return None

    desired_head = _desired_head_from_outbox(outbox_payload)
    if not desired_head:
        return None

    receipt_head = _head_from_receipt(receipt_payload)
    if receipt_head and receipt_head != desired_head:
        return {
            "desired_head_sha": desired_head,
            "receipt_head_sha": receipt_head,
            "reason": "receipt_head_mismatch",
        }

    remote_head = _remote_tracking_head(repo_root, branch)
    if remote_head and remote_head != desired_head:
        return {
            "desired_head_sha": desired_head,
            "remote_head_sha": remote_head,
            "reason": "remote_tracking_head_mismatch",
        }
    return None


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


def _terminal_receipts_by_key(
    receipt_files: Sequence[Path],
) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    by_key: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for path in receipt_files:
        payload = _load_json_mapping(path)
        if payload is None:
            continue
        status = str(payload.get("status") or "").strip().lower()
        if status not in TERMINAL_RECEIPT_STATUSES:
            continue
        key = str(payload.get("idempotency_key") or path.stem).strip()
        if key:
            by_key[key].append((path, payload))
    return by_key


def _load_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


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
    terminal_receipts_by_key = _terminal_receipts_by_key(receipt_files)
    terminal_receipt_keys = set(terminal_receipts_by_key)
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
    terminal_receipted_outbox_count = 0
    stale_target_pr_receipted_outbox: list[dict[str, str]] = []
    for path, key, branch in outbox_items:
        receipt_payloads = terminal_receipts_by_key.get(key, [])
        if not receipt_payloads:
            continue
        outbox_payload = _load_json_mapping(path)
        if outbox_payload is None:
            terminal_receipted_outbox_count += 1
            continue

        stale_evidence: dict[str, str] | None = None
        for receipt_path, receipt_payload in receipt_payloads:
            stale_evidence = _stale_target_pr_receipt_evidence(
                repo_root=repo_root,
                outbox_payload=outbox_payload,
                branch=branch,
                receipt_payload=receipt_payload,
            )
            if stale_evidence is None:
                terminal_receipted_outbox_count += 1
                break
            stale_evidence = {
                **stale_evidence,
                "branch": branch,
                "file": path.name,
                "idempotency_key": key,
                "receipt_file": receipt_path.name,
            }
        else:
            if stale_evidence is not None:
                stale_target_pr_receipted_outbox.append(stale_evidence)

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
        "stale_target_pr_receipted_outbox_count": len(stale_target_pr_receipted_outbox),
        "stale_target_pr_receipted_outbox": stale_target_pr_receipted_outbox[
            :DUPLICATE_OUTBOX_EXAMPLE_LIMIT
        ],
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
    parser.add_argument(
        "--state-root",
        type=Path,
        default=None,
        help=(
            "Shared automation state root for default output and queue dirs. "
            "Accepts either a repo root containing .aragora or the .aragora directory itself."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print the cached payload")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = _repo_root(Path(args.repo))
    state_root = args.state_root.expanduser() if args.state_root is not None else None
    output_base = state_root if state_root is not None else _shared_state_root(repo_root)
    output = (
        args.output
        if args.output.is_absolute()
        else _automation_state_default_path(output_base, args.output)
    )
    payload = build_status(
        repo_root=repo_root,
        github_repo=args.github_repo,
        labels=args.labels,
        max_open_prs=args.max_open_prs,
        max_open_issues=args.max_open_issues,
        outbox_dir=args.outbox_dir
        or (
            _automation_state_default_path(state_root, DEFAULT_OUTBOX_DIR)
            if state_root is not None
            else None
        ),
        receipt_dir=args.receipt_dir
        or (
            _automation_state_default_path(state_root, DEFAULT_RECEIPT_DIR)
            if state_root is not None
            else None
        ),
    )
    write_status(output, payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
