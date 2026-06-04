"""GitHub + ``evidence-lint`` I/O for merge-quorum reconciliation CLIs.

Thin subprocess wrappers shared by ``scripts/reconcile_merge_quorum.py`` (A1)
and ``scripts/settle_status.py`` (A2). Kept separate from the pure decision
logic in :mod:`aragora.swarm.merge_quorum_reconcile` so the decision logic stays
network-free and unit-testable. Counting is always delegated to the canonical
``review-queue evidence-lint`` subcommand so these tools match the gate's parser.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Any

from aragora.swarm.merge_quorum_reconcile import EvidenceComment, QuorumRun

QUORUM_CHECK_NAME = "aragora-merge-quorum"
QUORUM_WORKFLOW_FILE = "aragora-merge-quorum.yml"
HUMAN_SETTLEMENT_CONTEXT = "aragora/human-settlement"
_SHADOW_MARKERS = ("shadow", "advisory")
_FAILED_CONCLUSIONS = {"FAILURE", "TIMED_OUT", "ERROR", "STARTUP_FAILURE"}
_MERGE_PACKET_TIMEOUT = 120


def aragora_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("ARAGORA_USE_SECRETS_MANAGER", "false")
    return env


def run(
    args: list[str], *, env: dict[str, str] | None = None, timeout: int | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        args, capture_output=True, text=True, check=False, env=env, timeout=timeout
    )


def run_json(args: list[str], *, env: dict[str, str] | None = None) -> Any:
    proc = run(args, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(args)}\n{proc.stderr}")
    return json.loads(proc.stdout or "null")


def list_open_prs(repo: str, *, limit: int, author: str | None) -> list[int]:
    rows = (
        run_json(
            [
                "gh",
                "pr",
                "list",
                "--repo",
                repo,
                "--state",
                "open",
                "--limit",
                str(limit),
                "--json",
                "number,isDraft,author",
            ]
        )
        or []
    )
    numbers: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("isDraft"):
            continue
        if author and str((row.get("author") or {}).get("login", "")) != author:
            continue
        num = row.get("number")
        if isinstance(num, int):
            numbers.append(num)
    return numbers


def fetch_pr_context(repo: str, pr: int) -> dict[str, Any]:
    """Head SHA, head committedDate, quorum conclusion, and real-failure flag."""
    data = run_json(
        [
            "gh",
            "pr",
            "view",
            str(pr),
            "--repo",
            repo,
            "--json",
            "headRefOid,commits,statusCheckRollup",
        ]
    )
    head_sha = str(data.get("headRefOid") or "").strip()
    commits = data.get("commits") or []
    head_committed_at = ""
    for entry in commits:
        if isinstance(entry, dict) and str(entry.get("oid") or "") == head_sha:
            head_committed_at = str(entry.get("committedDate") or "")
            break
    if not head_committed_at and commits:
        last = commits[-1]
        if isinstance(last, dict):
            head_committed_at = str(last.get("committedDate") or "")

    real_failure = False
    quorum_conclusion = ""
    for check in data.get("statusCheckRollup") or []:
        if not isinstance(check, dict):
            continue
        name = str(check.get("name") or check.get("context") or "")
        conclusion = str(check.get("conclusion") or check.get("state") or "").upper()
        if name == QUORUM_CHECK_NAME:
            quorum_conclusion = conclusion
            continue
        if conclusion not in _FAILED_CONCLUSIONS:
            continue
        is_required = check.get("isRequired")
        if is_required is True:
            real_failure = True
        elif is_required is None and not any(m in name.lower() for m in _SHADOW_MARKERS):
            real_failure = True
    return {
        "head_sha": head_sha,
        "head_committed_at": head_committed_at,
        "quorum_conclusion": quorum_conclusion,
        "has_real_required_failure": real_failure,
    }


def fetch_latest_quorum_run(repo: str, head_sha: str) -> QuorumRun | None:
    if not head_sha:
        return None
    data = run_json(
        [
            "gh",
            "api",
            f"repos/{repo}/actions/workflows/{QUORUM_WORKFLOW_FILE}/runs?head_sha={head_sha}&per_page=1",
        ]
    )
    runs = (data or {}).get("workflow_runs") or []
    if not runs:
        return None
    run_obj = runs[0]
    return QuorumRun(
        run_id=int(run_obj.get("id")),
        created_at=str(run_obj.get("created_at") or ""),
        conclusion=str(run_obj.get("conclusion") or "").upper(),
        head_sha=str(run_obj.get("head_sha") or ""),
    )


def fetch_human_settlement_present(repo: str, head_sha: str) -> bool:
    if not head_sha:
        return False
    try:
        statuses = run_json(["gh", "api", f"repos/{repo}/commits/{head_sha}/statuses"]) or []
    except RuntimeError:
        return False
    for status in statuses:
        if not isinstance(status, dict):
            continue
        if str(status.get("context") or "") == HUMAN_SETTLEMENT_CONTEXT:
            return str(status.get("state") or "").lower() == "success"
    return False


def fetch_pr_tier(repo: str, pr: int) -> int | None:
    """Best-effort tier from ``review-queue merge-packet``; ``None`` if unknown."""
    try:
        proc = run(
            [
                sys.executable,
                "-m",
                "aragora.cli.main",
                "review-queue",
                "merge-packet",
                "--pr",
                str(pr),
                "--repo",
                repo,
                "--json",
            ],
            env=aragora_env(),
            timeout=_MERGE_PACKET_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    entries = payload if isinstance(payload, list) else [payload]
    for entry in entries:
        if isinstance(entry, dict) and entry.get("tier") is not None:
            try:
                return int(entry["tier"])
            except (TypeError, ValueError):
                return None
    return None


def fetch_evidence_comments(
    repo: str, pr: int, head_sha: str, head_committed_at: str
) -> list[EvidenceComment]:
    comments = run_json(["gh", "api", f"repos/{repo}/issues/{pr}/comments", "--paginate"]) or []
    env = aragora_env()
    results: list[EvidenceComment] = []
    for comment in comments:
        if not isinstance(comment, dict):
            continue
        body = str(comment.get("body") or "")
        if not body.strip():
            continue
        author = str((comment.get("user") or {}).get("login") or "")
        lint = lint_comment(repo, pr, head_sha, head_committed_at, author, body, env)
        counted = lint.get("counted_reviewer_ids") or []
        results.append(
            EvidenceComment(
                created_at=str(comment.get("created_at") or ""),
                would_count=bool(lint.get("would_count")),
                reviewer_id=str(counted[0]) if counted else "",
                is_dogfood=bool(lint.get("dogfood_evidence")),
            )
        )
    return results


def lint_comment(
    repo: str,
    pr: int,
    head_sha: str,
    head_committed_at: str,
    author: str,
    body: str,
    env: dict[str, str],
) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(body)
        body_file = fh.name
    try:
        args = [
            sys.executable,
            "-m",
            "aragora.cli.main",
            "review-queue",
            "evidence-lint",
            "--pr",
            str(pr),
            "--head-sha",
            head_sha,
            "--author",
            author,
            "--body-file",
            body_file,
            "--repo",
            repo,
            "--json",
        ]
        if head_committed_at:
            args.extend(["--head-committed-at", head_committed_at])
        proc = run(args, env=env)
        if proc.returncode != 0 or not proc.stdout.strip():
            return {}
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {}
    finally:
        try:
            os.unlink(body_file)
        except OSError:
            pass
