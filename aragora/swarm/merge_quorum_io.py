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
import re
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
_EVIDENCE_LINT_TIMEOUT = 90
_GH_TIMEOUT = 60
_GITHUB_ACTIONS_AUTHOR = "github-actions[bot]"
_MIN_EVIDENCE_BODY = 40


def _could_count(author: str, body: str) -> bool:
    """Cheap pre-check mirroring evidence-lint's hard rejections.

    A countable comment must have a non-github-actions author and enough text to
    carry a 7-char head citation plus a reviewer heading. Used only to skip
    obviously-uncountable comments before spawning the lint subprocess; the lint
    CLI remains the source of truth for everything that passes this filter.
    """
    if author == _GITHUB_ACTIONS_AUTHOR:
        return False
    return len(body.strip()) >= _MIN_EVIDENCE_BODY


def _looks_like_shadow(name: str) -> bool:
    """Whether a check name is a non-required shadow/advisory check.

    Shadow checks are named with a trailing marker word (e.g. ``... Shadow``),
    so this matches only the *last* token. A required check that merely contains
    a marker earlier in its name (e.g. ``aragora-shadow-deploy-required``, whose
    last token is ``required``) is therefore not misclassified as a shadow. This
    heuristic is only consulted when GitHub does not report ``isRequired``.
    """
    tokens = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
    return bool(tokens) and tokens[-1] in _SHADOW_MARKERS


def aragora_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("ARAGORA_USE_SECRETS_MANAGER", "false")
    return env


def run(
    args: list[str], *, env: dict[str, str] | None = None, timeout: int | None = _GH_TIMEOUT
) -> subprocess.CompletedProcess:
    # Default to a bounded timeout so no bare call can hang the reconciler;
    # callers that need longer (model-review CLIs) pass an explicit timeout.
    return subprocess.run(
        args, capture_output=True, text=True, check=False, env=env, timeout=timeout
    )


def run_json(
    args: list[str], *, env: dict[str, str] | None = None, timeout: int = _GH_TIMEOUT
) -> Any:
    try:
        proc = run(args, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"command timed out after {timeout}s: {' '.join(args)}") from exc
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
    if not head_committed_at and head_sha:
        # The head may be beyond the returned commit slice; fetch its date
        # directly rather than guessing from the last listed commit.
        try:
            commit = run_json(["gh", "api", f"repos/{repo}/commits/{head_sha}"]) or {}
        except RuntimeError:
            commit = {}
        committer = (commit.get("commit") or {}).get("committer") or {}
        head_committed_at = str(committer.get("date") or "")

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
        elif is_required is None and not _looks_like_shadow(name):
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
    raw_id = run_obj.get("id")
    if raw_id is None:
        return None
    try:
        run_id = int(raw_id)
    except (TypeError, ValueError):
        return None
    return QuorumRun(
        run_id=run_id,
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
        author = str((comment.get("user") or {}).get("login") or "")
        if not _could_count(author, body):
            # Cheap pre-filter: avoid spawning evidence-lint for comments that
            # cannot possibly count (evidence-lint rejects github-actions
            # authors and anything too short for a head citation + heading).
            continue
        lint = lint_comment(pr, head_sha, head_committed_at, author, body, env)
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


def _evidence_lint_args(
    pr: int, head_sha: str, head_committed_at: str, author: str, body_file: str
) -> list[str]:
    """Build the ``review-queue evidence-lint`` argv.

    Note: ``evidence-lint`` infers the repo from the current context and does
    *not* accept ``--repo``; passing it makes argparse reject the call. With
    ``--head-committed-at`` supplied the lint is fully offline.
    """
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
        "--json",
    ]
    if head_committed_at:
        args.extend(["--head-committed-at", head_committed_at])
    return args


def lint_comment(
    pr: int,
    head_sha: str,
    head_committed_at: str,
    author: str,
    body: str,
    env: dict[str, str],
) -> dict[str, Any]:
    body_file = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as fh:
            # Capture the path before writing so a write failure still cleans up.
            body_file = fh.name
            fh.write(body)
        args = _evidence_lint_args(pr, head_sha, head_committed_at, author, body_file)
        try:
            proc = run(args, env=env, timeout=_EVIDENCE_LINT_TIMEOUT)
        except subprocess.TimeoutExpired:
            return {}
        if proc.returncode != 0 or not proc.stdout.strip():
            return {}
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {}
    finally:
        if body_file:
            try:
                os.unlink(body_file)
            except OSError:
                pass
