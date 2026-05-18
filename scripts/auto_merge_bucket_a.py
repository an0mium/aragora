"""Stage 2 of the operator-delegation rollout — auto-merge Bucket A PRs.

Implements ``docs/roadmap/OPERATOR_DELEGATION_ROLLOUT.md`` Stage 2:

  - Reads the Stage 1 classifier output from ``scripts/triage_open_prs.py``.
  - For every PR classified as Bucket A, decides whether to ``gh pr merge
    --squash`` it.
  - Default ``--dry-run``: prints the intended merges, never mutates.
  - With ``--apply``: merges only PRs in Bucket A; never touches B/C/D.
  - Settling window: skip any PR whose most recent commit is younger
    than ``--settling-minutes`` (default 30) — gives sibling tooling
    and CI a chance to catch late breakage.
  - Defense-in-depth tripwires: a second, independent re-check of the
    PR's files, labels, draft status, mergeable status, and CI rollup.
    Any hit aborts that PR (with ``--apply``, exits the run non-zero
    overall to make the failure loud).
  - Writes a receipt to ``docs/status/AUTO_MERGE_RECEIPT_<utc>.md``
    on every ``--apply`` run, with the PR list, the policy version,
    and the sha256 of the classifier output.

Pure stdlib + ``gh`` subprocess. No ``aragora.*`` imports. No third-
party deps.

CLI::

    python3 scripts/auto_merge_bucket_a.py [--apply]
                                           [--settling-minutes N]
                                           [--only-pr N]
                                           [--delete-branch-on-merge]
                                           [--json]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
TRIAGE_SCRIPT = REPO_ROOT / "scripts" / "triage_open_prs.py"
RECEIPT_DIR = REPO_ROOT / "docs" / "status"
POLICY_DOC = REPO_ROOT / "docs" / "governance" / "OPERATOR_DELEGATION_POLICY.md"

BUCKET_A = "A"

DEFAULT_SETTLING_MINUTES = 30

# Defense-in-depth tripwires. These intentionally duplicate the
# triage classifier's most dangerous checks so a future bug in the
# classifier can't accidentally bless an unsafe PR.
PROTECTED_PATHS: frozenset[str] = frozenset(
    {
        "CLAUDE.md",
        "AGENTS.md",
        "aragora/__init__.py",
        ".env",
        ".envrc",
        "scripts/nomic_loop.py",
        "docs/AGENT_OPERATING_CONTRACT.md",
        "docs/governance/OPERATOR_DELEGATION_POLICY.md",
        "automation.toml",
    }
)

PROTECTED_PREFIXES: tuple[str, ...] = (
    ".github/workflows/",
    "secrets/",
)

TRUSTED_AUTHORS: frozenset[str] = frozenset({"an0mium"})


@dataclasses.dataclass(frozen=True)
class MergeDecision:
    pr_number: int
    title: str
    decision: str  # "merge", "skip-settling", "skip-tripwire", "skip-non-bucket-a"
    reason: str
    head_sha: str | None = None
    applied: bool = False


# ---------------------------------------------------------------------------
# Subprocess helpers (testable: callers can inject a custom runner)
# ---------------------------------------------------------------------------


Runner = Callable[[list[str]], "subprocess.CompletedProcess[str]"]


def _default_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, check=False)


def run_triage(*, runner: Runner | None = None) -> dict[str, Any]:
    """Invoke the Stage 1 classifier and return its JSON output."""
    runner = runner or _default_runner
    proc = runner(["python3", str(TRIAGE_SCRIPT), "--json"])
    if proc.returncode != 0:
        raise RuntimeError(
            f"triage_open_prs.py --json failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    return json.loads(proc.stdout or "{}")


def fetch_pr_commit_metadata(pr_number: int, *, runner: Runner | None = None) -> dict[str, Any]:
    """Fetch the gh metadata we need for the defense-in-depth re-check.

    Returns a dict with: ``commits`` (list, newest last), ``files``
    (list of {path, additions, deletions}), ``isDraft`` (bool),
    ``mergeable`` (str), ``mergeStateStatus`` (str), ``statusCheckRollup``
    (list), ``author.login`` (str), ``labels`` (list), ``headRefOid``
    (str), ``title`` (str).
    """
    runner = runner or _default_runner
    proc = runner(
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--json",
            (
                "number,title,author,isDraft,mergeable,mergeStateStatus,"
                "labels,files,commits,statusCheckRollup,headRefOid"
            ),
        ]
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh pr view {pr_number} failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    return json.loads(proc.stdout or "{}")


def gh_pr_merge_squash(
    pr_number: int,
    head_sha: str,
    delete_branch_on_merge: bool = False,
    *,
    runner: Runner | None = None,
) -> subprocess.CompletedProcess[str]:
    """Execute ``gh pr merge --squash`` for the given PR and exact head.

    Raises ``RuntimeError`` on non-zero exit. Returns the completed
    process so callers can inspect stdout/stderr.
    """
    runner = runner or _default_runner
    args = [
        "gh",
        "pr",
        "merge",
        str(pr_number),
        "--squash",
        "--match-head-commit",
        head_sha,
    ]
    if delete_branch_on_merge:
        args.append("--delete-branch")
    proc = runner(args)
    if proc.returncode != 0:
        raise RuntimeError(
            f"gh pr merge --squash {pr_number} failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout).strip()}"
        )
    return proc


# ---------------------------------------------------------------------------
# Defense-in-depth tripwire (independent of Stage 1 classifier)
# ---------------------------------------------------------------------------


def _file_paths(metadata: dict[str, Any]) -> list[str]:
    files = metadata.get("files") or []
    out: list[str] = []
    for entry in files:
        if isinstance(entry, dict):
            path = str(entry.get("path") or "")
            if path:
                out.append(path)
    return out


def _is_protected_path(path: str) -> bool:
    if path in PROTECTED_PATHS:
        return True
    for prefix in PROTECTED_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def defense_in_depth_tripwire(metadata: dict[str, Any]) -> str | None:
    """Re-check the dangerous gates Stage 1 already checked.

    Returns a tripwire reason string on hit, or ``None`` if the PR
    clears every defense-in-depth check.
    """

    if bool(metadata.get("isDraft")):
        return "draft (defense-in-depth)"

    mergeable = str(metadata.get("mergeable") or "")
    if mergeable != "MERGEABLE":
        return f"not mergeable (mergeable={mergeable or '(unknown)'})"

    merge_state = str(metadata.get("mergeStateStatus") or "")
    if merge_state not in {"CLEAN", "UNSTABLE", "HAS_HOOKS"}:
        # CLEAN = ready; UNSTABLE = required checks passed but optional are red;
        # HAS_HOOKS = passes required + has required hook checks. BLOCKED /
        # BEHIND / DIRTY / UNKNOWN are all bail-out states for Stage 2.
        return f"merge state not clean (mergeStateStatus={merge_state or '(unknown)'})"

    author_raw = metadata.get("author") or {}
    author = author_raw.get("login", "") if isinstance(author_raw, dict) else str(author_raw)
    if author not in TRUSTED_AUTHORS:
        return f"non-trusted author ({author or '(unknown)'})"

    for path in _file_paths(metadata):
        if _is_protected_path(path):
            return f"edits protected path ({path})"

    checks = metadata.get("statusCheckRollup") or []
    ci_failure = sum(1 for c in checks if isinstance(c, dict) and c.get("conclusion") == "FAILURE")
    if ci_failure:
        return f"CI red ({ci_failure} failures)"

    ci_pending = sum(
        1 for c in checks if isinstance(c, dict) and c.get("status") in ("IN_PROGRESS", "QUEUED")
    )
    if ci_pending:
        return f"CI pending ({ci_pending} in-flight)"

    return None


# ---------------------------------------------------------------------------
# Settling window
# ---------------------------------------------------------------------------


def _parse_commit_timestamp(raw: str) -> datetime.datetime | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.datetime.fromisoformat(raw)
    except ValueError:
        return None


def latest_commit_age_minutes(
    metadata: dict[str, Any],
    *,
    now: datetime.datetime | None = None,
) -> float | None:
    """Return age in minutes of the most recent commit on the PR.

    Returns ``None`` if no commit timestamp can be parsed.
    """
    if now is None:
        now = datetime.datetime.now(datetime.timezone.utc)
    commits = metadata.get("commits") or []
    latest: datetime.datetime | None = None
    for entry in commits:
        if not isinstance(entry, dict):
            continue
        ts = _parse_commit_timestamp(str(entry.get("committedDate") or ""))
        if ts is None:
            continue
        if latest is None or ts > latest:
            latest = ts
    if latest is None:
        return None
    return (now - latest).total_seconds() / 60.0


def settling_window_skip_reason(
    metadata: dict[str, Any],
    *,
    settling_minutes: int,
    now: datetime.datetime | None = None,
) -> str | None:
    age = latest_commit_age_minutes(metadata, now=now)
    if age is None:
        return "could not parse latest commit timestamp"
    if age < settling_minutes:
        return (
            f"settling window not met (last commit {age:.1f}min ago, need ≥ {settling_minutes}min)"
        )
    return None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def decide(
    triage_payload: dict[str, Any],
    *,
    apply: bool,
    settling_minutes: int,
    only_pr: int | None = None,
    metadata_provider: Callable[[int], dict[str, Any]] | None = None,
    merger: Callable[[int, str, bool], subprocess.CompletedProcess[str]] | None = None,
    delete_branch_on_merge: bool = False,
    now: datetime.datetime | None = None,
) -> tuple[list[MergeDecision], int]:
    """Decide what to do with each PR; return decisions + tripwire exit code.

    Tripwire exit code is 0 if every Bucket A PR was either merged
    (with --apply) or cleanly deferred (settling window). Any
    defense-in-depth tripwire hit forces a non-zero exit code; that
    is the policy's "abort and exit non-zero on any single tripwire"
    requirement.
    """
    metadata_provider = metadata_provider or fetch_pr_commit_metadata
    merger = merger or gh_pr_merge_squash

    decisions: list[MergeDecision] = []
    tripwire_exit = 0

    for entry in triage_payload.get("results") or []:
        if not isinstance(entry, dict):
            continue
        bucket = str(entry.get("bucket") or "")
        pr_number = int(entry.get("pr_number") or 0)
        title = str(entry.get("title") or "")
        if only_pr is not None and pr_number != only_pr:
            continue
        if bucket != BUCKET_A:
            decisions.append(
                MergeDecision(
                    pr_number=pr_number,
                    title=title,
                    decision="skip-non-bucket-a",
                    reason=f"bucket={bucket} (Stage 2 only acts on A)",
                    applied=False,
                )
            )
            continue

        try:
            metadata = metadata_provider(pr_number)
        except Exception as exc:
            decisions.append(
                MergeDecision(
                    pr_number=pr_number,
                    title=title,
                    decision="metadata-fetch-failed",
                    reason=f"metadata fetch failed: {exc}",
                    applied=False,
                )
            )
            tripwire_exit = 1
            continue
        head_sha = str(metadata.get("headRefOid") or "")

        try:
            metadata_number = int(metadata.get("number") or 0)
        except (TypeError, ValueError):
            metadata_number = 0
        if metadata_number != pr_number:
            decisions.append(
                MergeDecision(
                    pr_number=pr_number,
                    title=title,
                    decision="skip-tripwire",
                    reason=f"PR number mismatch (classifier={pr_number}, gh={metadata_number})",
                    head_sha=head_sha or None,
                    applied=False,
                )
            )
            tripwire_exit = 1
            continue

        tripwire = defense_in_depth_tripwire(metadata)
        if tripwire is not None:
            decisions.append(
                MergeDecision(
                    pr_number=pr_number,
                    title=title,
                    decision="skip-tripwire",
                    reason=tripwire,
                    head_sha=head_sha or None,
                    applied=False,
                )
            )
            tripwire_exit = 1
            continue

        settling = settling_window_skip_reason(
            metadata,
            settling_minutes=settling_minutes,
            now=now,
        )
        if settling is not None:
            decisions.append(
                MergeDecision(
                    pr_number=pr_number,
                    title=title,
                    decision="skip-settling",
                    reason=settling,
                    head_sha=head_sha or None,
                    applied=False,
                )
            )
            continue

        if apply:
            try:
                merger(pr_number, head_sha, delete_branch_on_merge)
            except RuntimeError as exc:
                decisions.append(
                    MergeDecision(
                        pr_number=pr_number,
                        title=title,
                        decision="merge-failed",
                        reason=str(exc),
                        head_sha=head_sha or None,
                        applied=False,
                    )
                )
                tripwire_exit = 1
                continue
            decisions.append(
                MergeDecision(
                    pr_number=pr_number,
                    title=title,
                    decision="merge",
                    reason="bucket A + tripwire clear + settling window met",
                    head_sha=head_sha or None,
                    applied=True,
                )
            )
        else:
            decisions.append(
                MergeDecision(
                    pr_number=pr_number,
                    title=title,
                    decision="merge",
                    reason="bucket A + tripwire clear + settling window met (dry-run)",
                    head_sha=head_sha or None,
                    applied=False,
                )
            )

    return decisions, tripwire_exit


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def _classifier_sha256(triage_payload: dict[str, Any]) -> str:
    serialized = json.dumps(triage_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _policy_version() -> str:
    if not POLICY_DOC.is_file():
        return "unknown"
    text = POLICY_DOC.read_text(encoding="utf-8")
    match = re.search(r"(?im)^\s*version\s*:\s*(?P<version>.+?)\s*$", text)
    if match is not None:
        return match.group("version").strip()
    return "tracked-doc"


def render_receipt(
    decisions: list[MergeDecision],
    *,
    triage_payload: dict[str, Any],
    apply: bool,
    settling_minutes: int,
    now: datetime.datetime | None = None,
) -> str:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    lines: list[str] = []
    lines.append("# Auto-merge Bucket A receipt")
    lines.append("")
    lines.append(f"- Generated: `{now.strftime('%Y-%m-%dT%H:%M:%SZ')}`")
    lines.append(f"- Mode: `{'apply' if apply else 'dry-run'}`")
    lines.append(f"- Settling window: `{settling_minutes} min`")
    lines.append(f"- Policy version: `{_policy_version()}`")
    lines.append(f"- Classifier output sha256: `{_classifier_sha256(triage_payload)}`")
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    if not decisions:
        lines.append("No Bucket A PRs available; nothing to do.")
        lines.append("")
        return "\n".join(lines)
    lines.append("| PR | Decision | Applied | Head SHA | Reason |")
    lines.append("|---|---|---|---|---|")
    for d in decisions:
        head = (d.head_sha or "—")[:12]
        lines.append(
            f"| #{d.pr_number} | `{d.decision}` | {'yes' if d.applied else 'no'} | "
            f"`{head}` | {d.reason} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_receipt(
    receipt_md: str,
    *,
    now: datetime.datetime | None = None,
    receipt_dir: Path | None = None,
) -> Path:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    receipt_dir = receipt_dir or RECEIPT_DIR
    receipt_dir.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    path = receipt_dir / f"AUTO_MERGE_RECEIPT_{stamp}.md"
    path.write_text(receipt_md, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def emit_json(
    decisions: list[MergeDecision],
    *,
    triage_payload: dict[str, Any],
    apply: bool,
    settling_minutes: int,
    receipt_path: Path | None,
) -> str:
    payload = {
        "mode": "apply" if apply else "dry-run",
        "settling_minutes": settling_minutes,
        "policy_version": _policy_version(),
        "classifier_sha256": _classifier_sha256(triage_payload),
        "receipt_path": str(receipt_path) if receipt_path else None,
        "decisions": [dataclasses.asdict(d) for d in decisions],
    }
    return json.dumps(payload, indent=2)


def emit_table(decisions: list[MergeDecision], *, apply: bool) -> str:
    lines: list[str] = []
    lines.append(f"auto_merge_bucket_a — mode={'apply' if apply else 'dry-run'}")
    if not decisions:
        lines.append("  (no Bucket A PRs)")
        return "\n".join(lines)
    for d in decisions:
        applied = "[applied]" if d.applied else "[skipped]"
        head = (d.head_sha or "—")[:10]
        lines.append(f"  #{d.pr_number:<5d} {applied:9s} {d.decision:24s} head={head} — {d.reason}")
        lines.append(f"      title: {d.title}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Stage 2 of operator-delegation rollout — auto-merge Bucket A PRs.")
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually merge. Without this, runs dry-run by default.",
    )
    parser.add_argument(
        "--settling-minutes",
        type=int,
        default=DEFAULT_SETTLING_MINUTES,
        help=(
            "Skip PRs whose latest commit is younger than this "
            f"(default {DEFAULT_SETTLING_MINUTES})."
        ),
    )
    parser.add_argument(
        "--only-pr",
        type=int,
        default=None,
        help="Process only this PR number (still bucket-A gated).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable table.",
    )
    parser.add_argument(
        "--delete-branch-on-merge",
        action="store_true",
        help="Delete merged PR branches. Default keeps branches for auditability.",
    )

    args = parser.parse_args(argv)

    if args.settling_minutes < 0:
        print("error: --settling-minutes must be non-negative", file=sys.stderr)
        return 2

    try:
        triage_payload = run_triage()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    decisions, tripwire_exit = decide(
        triage_payload,
        apply=args.apply,
        settling_minutes=args.settling_minutes,
        only_pr=args.only_pr,
        delete_branch_on_merge=args.delete_branch_on_merge,
    )

    receipt_path: Path | None = None
    if args.apply:
        receipt_md = render_receipt(
            decisions,
            triage_payload=triage_payload,
            apply=True,
            settling_minutes=args.settling_minutes,
        )
        receipt_path = write_receipt(receipt_md)

    if args.json:
        print(
            emit_json(
                decisions,
                triage_payload=triage_payload,
                apply=args.apply,
                settling_minutes=args.settling_minutes,
                receipt_path=receipt_path,
            )
        )
    else:
        print(emit_table(decisions, apply=args.apply))
        if receipt_path is not None:
            print(f"receipt: {receipt_path}")

    return tripwire_exit


if __name__ == "__main__":
    sys.exit(main())
