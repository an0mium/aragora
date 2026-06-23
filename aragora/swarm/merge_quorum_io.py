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

from aragora.swarm import github_app_auth
from aragora.swarm.merge_quorum_reconcile import (
    EvidenceComment,
    PacketClassification,
    QuorumRun,
    parse_ci_packet_classification,
)

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


def _read_env() -> dict[str, str]:
    """Environment for *read-only* ``gh`` calls.

    Prefer the GitHub App installation token so reads draw on the App's separate
    API budget instead of starving the operator's shared per-user PAT quota (the
    chronic GraphQL-exhaustion failure mode). Degrades transparently to the
    ambient/operator auth when no App config is present. Read-only by design --
    the App installation 403s on classic branch-protection writes, so write call
    sites deliberately keep ``run()``'s default (PAT) env instead of this helper.
    """
    return github_app_auth.github_cli_env(aragora_env())


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
            ],
            env=_read_env(),
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


def _gh_json_for_rest_fallback(args: list[str]) -> Any:
    """Adapter for review-queue REST helpers that preserves read-token routing."""
    from aragora.cli.commands.review_queue_transport import _GhError

    try:
        return run_json(["gh", *args], env=_read_env())
    except RuntimeError as exc:
        raise _GhError(str(exc)) from exc


def _fetch_pr_context_with_rest_fallback(repo: str, pr: int, source_error: str) -> dict[str, Any]:
    """Fallback to REST PR/check metadata after GraphQL-backed ``gh pr view`` fails."""
    from aragora.cli.commands.review_queue_rest_fallback import _hydrate_pr_with_rest_fallback

    return _hydrate_pr_with_rest_fallback(
        number=pr,
        repo_slug=repo,
        source_error=source_error,
        gh_json=_gh_json_for_rest_fallback,
    )


def _head_committed_at(data: dict[str, Any], head_sha: str, repo: str) -> str:
    commits = data.get("commits") or []
    for entry in commits:
        if isinstance(entry, dict) and str(entry.get("oid") or "") == head_sha:
            return str(entry.get("committedDate") or "")
    if not head_sha:
        return ""
    # The head may be beyond the returned commit slice; fetch its date directly
    # rather than guessing from the last listed commit.
    try:
        commit = run_json(["gh", "api", f"repos/{repo}/commits/{head_sha}"], env=_read_env()) or {}
    except RuntimeError:
        commit = {}
    committer = (commit.get("commit") or {}).get("committer") or {}
    return str(committer.get("date") or "")


def _latest_by_name(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, tuple[str, int, dict[str, Any]]] = {}
    for index, item in enumerate(items):
        name = str(item.get("name") or item.get("context") or "").strip()
        if not name:
            continue
        timestamp = str(
            item.get("completed_at")
            or item.get("started_at")
            or item.get("updatedAt")
            or item.get("updated_at")
            or item.get("created_at")
            or item.get("createdAt")
            or ""
        )
        previous = latest.get(name)
        if (
            previous is None
            or timestamp > previous[0]
            or (timestamp == previous[0] and index < previous[1])
        ):
            latest[name] = (timestamp, index, item)
    return {name: row[2] for name, row in latest.items()}


def _direct_check_rollup(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a conservative rollup from REST direct checks when PR rollup is absent."""
    rollup = data.get("statusCheckRollup") or []
    if rollup:
        return [item for item in rollup if isinstance(item, dict)]

    direct_runs = [item for item in data.get("directCheckRuns") or [] if isinstance(item, dict)]
    commit_statuses = [item for item in data.get("commitStatuses") or [] if isinstance(item, dict)]
    required_payload = data.get("requiredStatusChecks")
    required_checks = []
    if isinstance(required_payload, dict):
        required_checks = [
            item for item in required_payload.get("checks") or [] if isinstance(item, dict)
        ]

    direct_by_name = _latest_by_name(direct_runs)
    status_by_name = _latest_by_name(commit_statuses)
    synthetic: list[dict[str, Any]] = []

    for required in required_checks:
        context = str(required.get("context") or "").strip()
        if not context:
            continue
        run = direct_by_name.get(context)
        if run is not None:
            synthetic.append(
                {
                    "name": context,
                    "conclusion": str(run.get("conclusion") or "").upper(),
                    "state": str(run.get("status") or "").upper(),
                    "isRequired": True,
                }
            )
            continue
        status = status_by_name.get(context)
        if status is not None:
            synthetic.append(
                {
                    "context": context,
                    "state": str(status.get("state") or "").upper(),
                    "isRequired": True,
                }
            )

    if QUORUM_CHECK_NAME in direct_by_name and all(
        str(item.get("name") or item.get("context") or "") != QUORUM_CHECK_NAME
        for item in synthetic
    ):
        run = direct_by_name[QUORUM_CHECK_NAME]
        synthetic.append(
            {
                "name": QUORUM_CHECK_NAME,
                "conclusion": str(run.get("conclusion") or "").upper(),
                "state": str(run.get("status") or "").upper(),
                "isRequired": False,
            }
        )
    return synthetic


def _context_from_pr_payload(repo: str, data: dict[str, Any]) -> dict[str, Any]:
    head_sha = str(data.get("headRefOid") or "").strip()
    head_committed_at = _head_committed_at(data, head_sha, repo)

    real_failure = False
    quorum_conclusion = ""
    for check in _direct_check_rollup(data):
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
        "rest_fallback": data.get("_rest_fallback") or {},
    }


def fetch_pr_context(repo: str, pr: int) -> dict[str, Any]:
    """Head SHA, head committedDate, quorum conclusion, and real-failure flag."""
    try:
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
            ],
            env=_read_env(),
        )
    except RuntimeError as exc:
        data = _fetch_pr_context_with_rest_fallback(repo, pr, str(exc))
    if not isinstance(data, dict):
        data = {}
    return _context_from_pr_payload(repo, data)


def fetch_pr_head_sha(repo: str, pr: int) -> str:
    """Resolve a PR head SHA, using REST fallback when ``gh pr view`` is blocked."""
    try:
        data = run_json(
            ["gh", "pr", "view", str(pr), "--repo", repo, "--json", "headRefOid"],
            env=_read_env(),
        )
        return str((data or {}).get("headRefOid") or "").strip()
    except RuntimeError as exc:
        data = _fetch_pr_context_with_rest_fallback(repo, pr, str(exc))
    return str(data.get("headRefOid") or "").strip()


def fetch_latest_quorum_run(repo: str, head_sha: str) -> QuorumRun | None:
    if not head_sha:
        return None
    data = run_json(
        [
            "gh",
            "api",
            f"repos/{repo}/actions/workflows/{QUORUM_WORKFLOW_FILE}/runs?head_sha={head_sha}&per_page=1",
        ],
        env=_read_env(),
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
        statuses = (
            run_json(["gh", "api", f"repos/{repo}/commits/{head_sha}/statuses"], env=_read_env())
            or []
        )
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
    packet = fetch_merge_packet_classification(repo, pr)
    return packet.tier if packet is not None else None


def fetch_merge_packet_classification(repo: str, pr: int) -> PacketClassification | None:
    """Best-effort current local merge-packet classification for one PR."""
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
            env=_read_env(),
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
    # merge-packet --json returns a top-level object with the per-PR rows under
    # "entries"; tier lives on each entry, not the envelope. Accept a bare list
    # or single entry too for forward-compatibility.
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        nested = payload.get("entries")
        entries = nested if isinstance(nested, list) else [payload]
    else:
        entries = []

    def _coerce(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _packet(entry: dict[str, Any]) -> PacketClassification | None:
        tier = _coerce(entry.get("tier"))
        pr_value = _coerce(entry.get("pr_number"))
        if pr_value is None:
            pr_value = pr
        if pr_value != pr or entry.get("tier") is None:
            return None
        return PacketClassification(
            source="local",
            pr_number=pr_value,
            head_sha=str(entry.get("head_sha") or ""),
            tier=tier,
            status=str(entry.get("status") or ""),
            verdict=str(entry.get("verdict") or ""),
            requires_human_risk_settlement=bool(entry.get("requires_human_risk_settlement")),
        )

    # Prefer the row whose pr_number matches the requested PR. The normal
    # single-PR --json shape always discloses pr_number, so a multi-PR envelope
    # can never resolve the wrong PR's tier (which would mis-gate posting).
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        packet = _packet(entry)
        if packet is not None and _coerce(entry.get("pr_number")) == pr:
            return packet
    # Fall back to the first disclosed tier only when NO row carries a pr_number
    # (forward-compat shapes such as a bare list or single entry).
    if not any(isinstance(e, dict) and e.get("pr_number") is not None for e in entries):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            packet = _packet(entry)
            if packet is not None:
                return packet
    return None


def fetch_quorum_run_packet_classification(
    repo: str, *, run_id: int, pr: int, head_sha: str
) -> PacketClassification | None:
    """Best-effort CI packet classification parsed from a merge-quorum run log."""
    try:
        proc = run(
            ["gh", "run", "view", str(run_id), "--repo", repo, "--log"],
            timeout=_GH_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return parse_ci_packet_classification(proc.stdout, pr_number=pr, head_sha=head_sha)


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
