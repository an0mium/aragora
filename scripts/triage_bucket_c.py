"""Stage 3 of the operator-delegation rollout — Bucket C batcher.

Implements ``docs/roadmap/OPERATOR_DELEGATION_ROLLOUT.md`` Stage 3:

  - Reads the Stage 1 classifier output from ``scripts/triage_open_prs.py``.
  - Filters to Bucket C only.
  - Accepts a one-character response per PR via stdin (``--interactive``)
    or via a JSON response file (``--responses FILE``):
      ``y`` — advance: ``gh pr ready`` (if draft) + comment + optional label
      ``n`` — close:    ``gh pr close --comment``
      ``d`` — defer:    no-op
  - Defaults to dry-run; ``--apply`` is required to mutate.
  - Hard-codes the operator-delegation policy hold list — held PRs
    are never advanced or closed regardless of the operator's
    response.
  - Defense-in-depth: also skips PRs that touch protected paths even
    if the classifier somehow missed them.
  - Writes a receipt to ``docs/status/BUCKET_C_RECEIPT_<utc>.md`` on
    every ``--apply`` run.

Pure stdlib + ``gh`` subprocess. No ``aragora.*`` imports. No third-
party deps. The response file is JSON (``{"7251": "y", "7252": "d"}``)
to stay stdlib-only; YAML is a superset of JSON and the same payloads
are accepted as ``.yaml`` files with no loader change required.

CLI::

    python3 scripts/triage_bucket_c.py [--interactive]
                                       [--responses FILE]
                                       [--apply]
                                       [--json]
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parent.parent
GH_REPO = "synaptent/aragora"
TRIAGE_SCRIPT = REPO_ROOT / "scripts" / "triage_open_prs.py"
RECEIPT_DIR = REPO_ROOT / "docs" / "status"
POLICY_DOC = REPO_ROOT / "docs" / "governance" / "OPERATOR_DELEGATION_POLICY.md"

BUCKET_C = "C"

# Held PR numbers — kept in sync with the policy doc's canonical hold
# list and the parallel constant in scripts/triage_open_prs.py +
# scripts/apply_operator_decisions.py. Stage 3 hard-skips these
# regardless of the operator's response.
HELD_PR_NUMBERS: frozenset[int] = frozenset({4990, 7173, 7215, 7240, 7243, 7245, 7249, 7252})

# Defense-in-depth protected paths. Even if Stage 1 somehow classified
# a PR touching one of these as Bucket C (rather than the held list),
# Stage 3 still refuses to advance it.
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

RESPONSE_ADVANCE = "y"
RESPONSE_CLOSE = "n"
RESPONSE_DEFER = "d"
VALID_RESPONSES: frozenset[str] = frozenset({RESPONSE_ADVANCE, RESPONSE_CLOSE, RESPONSE_DEFER})

# Status codes for the result table.
STATUS_ADVANCED = "advanced"
STATUS_CLOSED = "closed"
STATUS_DEFERRED = "deferred"
STATUS_HELD = "held-skipped"
STATUS_PROTECTED = "protected-skipped"
STATUS_NOT_BUCKET_C = "not-bucket-c-skipped"
STATUS_NO_RESPONSE = "no-response-skipped"
STATUS_INVALID_RESPONSE = "invalid-response"
STATUS_WOULD_ADVANCE = "would-advance"
STATUS_WOULD_CLOSE = "would-close"
STATUS_GH_FAILED = "gh-failed"
STATUS_LIVE_CHECK_FAILED = "live-check-failed"


@dataclasses.dataclass(frozen=True)
class EntryResult:
    pr_number: int
    title: str
    response: str | None
    status: str
    reason: str
    gh_commands: tuple[tuple[str, ...], ...] = ()


@dataclasses.dataclass(frozen=True)
class PrSnapshot:
    state: str
    is_draft: bool
    head_sha: str
    files: tuple[str, ...]


class LivePrCheckError(RuntimeError):
    """Raised when a live PR guard cannot verify the current GitHub state."""


# ---------------------------------------------------------------------------
# Subprocess helpers (injectable for testing)
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


def fetch_pr_snapshot(pr_number: int, *, runner: Runner | None = None) -> PrSnapshot:
    """Return the live PR state needed by apply-mode safety guards."""
    runner = runner or _default_runner
    proc = runner(
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--repo",
            GH_REPO,
            "--json",
            "state,isDraft,headRefOid,files",
        ]
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip() or f"exit {proc.returncode}"
        raise LivePrCheckError(f"gh pr view failed: {detail}")
    try:
        payload = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise LivePrCheckError(f"gh pr view returned non-JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LivePrCheckError("gh pr view returned non-object JSON")
    state = str(payload.get("state") or "")
    head_sha = str(payload.get("headRefOid") or "")
    if not state:
        raise LivePrCheckError("gh pr view response missing state")
    if not head_sha:
        raise LivePrCheckError("gh pr view response missing headRefOid")
    files_payload = payload.get("files")
    if not isinstance(files_payload, list):
        raise LivePrCheckError("gh pr view response missing files list")
    out: list[str] = []
    for idx, entry in enumerate(files_payload):
        if not isinstance(entry, dict):
            raise LivePrCheckError(f"gh pr view response has malformed file entry at index {idx}")
        path = entry.get("path")
        if not isinstance(path, str) or not path:
            raise LivePrCheckError(f"gh pr view response missing file path at index {idx}")
        out.append(path)
    if not out:
        raise LivePrCheckError("gh pr view response had no files to verify")
    return PrSnapshot(
        state=state,
        is_draft=bool(payload.get("isDraft")),
        head_sha=head_sha,
        files=tuple(out),
    )


def fetch_pr_files(pr_number: int, *, runner: Runner | None = None) -> list[str]:
    """Return the file paths touched by the given PR.

    Used by the defense-in-depth protected-path tripwire.
    """
    return list(fetch_pr_snapshot(pr_number, runner=runner).files)


def gh_pr_ready(
    pr_number: int, *, runner: Runner | None = None
) -> subprocess.CompletedProcess[str]:
    runner = runner or _default_runner
    return runner(["gh", "pr", "ready", str(pr_number), "--repo", GH_REPO])


def gh_pr_close(
    pr_number: int, body: str, *, runner: Runner | None = None
) -> subprocess.CompletedProcess[str]:
    runner = runner or _default_runner
    return runner(["gh", "pr", "close", str(pr_number), "--repo", GH_REPO, "--comment", body])


def gh_pr_comment(
    pr_number: int, body: str, *, runner: Runner | None = None
) -> subprocess.CompletedProcess[str]:
    runner = runner or _default_runner
    return runner(["gh", "pr", "comment", str(pr_number), "--repo", GH_REPO, "--body", body])


# ---------------------------------------------------------------------------
# Response loading
# ---------------------------------------------------------------------------


def load_responses_file(path: Path) -> dict[int, str]:
    """Load a JSON response file mapping PR number → y/n/d.

    Accepts integer or string keys. Raises ``ValueError`` for any
    response not in ``{y, n, d}``.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"response file {path} must be a JSON object mapping PR number to y/n/d")
    parsed: dict[int, str] = {}
    for key, value in raw.items():
        try:
            pr_num = int(str(key).lstrip("#"))
        except ValueError as exc:
            raise ValueError(f"response file {path}: invalid PR key {key!r}") from exc
        response = str(value).strip().lower()
        if response not in VALID_RESPONSES:
            raise ValueError(
                f"response file {path}: PR #{pr_num} has invalid"
                f" response {value!r}; expected one of"
                f" {sorted(VALID_RESPONSES)}"
            )
        parsed[pr_num] = response
    return parsed


def collect_responses_interactive(
    bucket_c_entries: list[dict[str, Any]],
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> dict[int, str]:
    """Prompt for y/n/d per Bucket C PR, reading from stdin."""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    responses: dict[int, str] = {}
    for entry in bucket_c_entries:
        pr_number = int(entry.get("pr_number") or 0)
        title = str(entry.get("title") or "")
        reason = str(entry.get("reason") or "")
        stdout.write(f"\nPR #{pr_number}\n  title: {title}\n  reason: {reason}\n")
        stdout.write("  [y]advance / [n]close / [d]defer: ")
        stdout.flush()
        raw = (stdin.readline() or "").strip().lower()
        if raw == "":
            # Empty input is a defer (skip).
            responses[pr_number] = RESPONSE_DEFER
        elif raw in VALID_RESPONSES:
            responses[pr_number] = raw
        else:
            stdout.write(f"  invalid response {raw!r}; treating as defer.\n")
            responses[pr_number] = RESPONSE_DEFER
    return responses


# ---------------------------------------------------------------------------
# Defense-in-depth
# ---------------------------------------------------------------------------


def _is_protected_path(path: str) -> bool:
    if path in PROTECTED_PATHS:
        return True
    for prefix in PROTECTED_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


def protected_path_tripwire(
    pr_number: int,
    *,
    runner: Runner | None = None,
    files_provider: Callable[[int], list[str]] | None = None,
) -> str | None:
    """Return a tripwire reason if the PR touches any protected path."""
    fetch = files_provider or (lambda n: fetch_pr_files(n, runner=runner))
    try:
        paths = fetch(pr_number)
    except LivePrCheckError as exc:
        return f"could not verify protected paths ({exc})"
    for path in paths:
        if _is_protected_path(path):
            return f"edits protected path ({path})"
    return None


def _verify_live_pr_before_mutation(
    pr_number: int,
    *,
    expected_head: str,
    runner: Runner | None = None,
) -> str | None:
    """Return a fail-closed reason if the live PR is not safe to mutate."""
    try:
        snapshot = fetch_pr_snapshot(pr_number, runner=runner)
    except LivePrCheckError as exc:
        return f"could not verify live PR before mutation ({exc})"
    if snapshot.state != "OPEN":
        return f"live PR state is {snapshot.state!r}, expected 'OPEN'"
    if snapshot.head_sha != expected_head:
        return f"live PR head changed before mutation ({snapshot.head_sha} != {expected_head})"
    tripwire = protected_path_tripwire(
        pr_number,
        runner=runner,
        files_provider=lambda _n: list(snapshot.files),
    )
    if tripwire is not None:
        return f"live PR tripwire changed before mutation ({tripwire})"
    try:
        live_triage = run_triage(runner=runner)
    except (RuntimeError, json.JSONDecodeError) as exc:
        return f"could not verify live Bucket C classification before mutation ({exc})"
    for entry in live_triage.get("results") or []:
        if not isinstance(entry, dict):
            continue
        if int(entry.get("pr_number") or 0) != pr_number:
            continue
        live_bucket = str(entry.get("bucket") or "")
        if live_bucket != BUCKET_C:
            return f"live PR bucket is {live_bucket!r}, expected {BUCKET_C!r}"
        return None
    return "live PR was not present in current triage output"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


_COMMENT_HEADER = (
    "Applied via `scripts/triage_bucket_c.py` (Stage 3 of the operator-delegation rollout)."
)


def _advance_comment_body() -> str:
    return (
        f"{_COMMENT_HEADER}\n\nOperator response: **y** — advancing this"
        " Bucket C PR (mark-ready). Stage 2 will pick up the merge once"
        " CI settles and the classifier flips to Bucket A.\n"
    )


def _close_comment_body() -> str:
    return (
        f"{_COMMENT_HEADER}\n\nOperator response: **n** — closing this"
        " Bucket C PR. Reopen if this was applied in error.\n"
    )


def decide(
    triage_payload: dict[str, Any],
    responses: dict[int, str],
    *,
    apply: bool,
    runner: Runner | None = None,
    files_provider: Callable[[int], list[str]] | None = None,
) -> list[EntryResult]:
    """Walk the Bucket C subset and produce a decision per PR."""
    results: list[EntryResult] = []
    runner = runner or _default_runner

    for entry in triage_payload.get("results") or []:
        if not isinstance(entry, dict):
            continue
        bucket = str(entry.get("bucket") or "")
        pr_number = int(entry.get("pr_number") or 0)
        title = str(entry.get("title") or "")
        reason = str(entry.get("reason") or "")

        if bucket != BUCKET_C:
            # Stage 3 only operates on Bucket C. Other buckets are
            # silently dropped here; callers can use --json to see
            # the full Stage 1 output if they want context.
            continue

        response = responses.get(pr_number)
        if response is None:
            results.append(
                EntryResult(
                    pr_number=pr_number,
                    title=title,
                    response=None,
                    status=STATUS_NO_RESPONSE,
                    reason=f"no response provided (bucket-C reason: {reason})",
                )
            )
            continue
        if response not in VALID_RESPONSES:
            results.append(
                EntryResult(
                    pr_number=pr_number,
                    title=title,
                    response=response,
                    status=STATUS_INVALID_RESPONSE,
                    reason=f"invalid response {response!r}",
                )
            )
            continue

        if pr_number in HELD_PR_NUMBERS:
            results.append(
                EntryResult(
                    pr_number=pr_number,
                    title=title,
                    response=response,
                    status=STATUS_HELD,
                    reason=f"#{pr_number} is on the policy hold list",
                )
            )
            continue

        live_snapshot: PrSnapshot | None = None
        if files_provider is None:
            try:
                live_snapshot = fetch_pr_snapshot(pr_number, runner=runner)
            except LivePrCheckError as exc:
                results.append(
                    EntryResult(
                        pr_number=pr_number,
                        title=title,
                        response=response,
                        status=STATUS_PROTECTED,
                        reason=f"could not verify protected paths ({exc})",
                    )
                )
                continue

        effective_files_provider = files_provider
        if effective_files_provider is None:
            assert live_snapshot is not None
            snapshot_for_files = live_snapshot

            def _files_from_snapshot(
                _n: int,
                snapshot: PrSnapshot = snapshot_for_files,
            ) -> list[str]:
                return list(snapshot.files)

            effective_files_provider = _files_from_snapshot

        tripwire = protected_path_tripwire(
            pr_number,
            runner=runner,
            files_provider=effective_files_provider,
        )
        if tripwire is not None:
            results.append(
                EntryResult(
                    pr_number=pr_number,
                    title=title,
                    response=response,
                    status=STATUS_PROTECTED,
                    reason=tripwire,
                )
            )
            continue

        if response == RESPONSE_DEFER:
            results.append(
                EntryResult(
                    pr_number=pr_number,
                    title=title,
                    response=response,
                    status=STATUS_DEFERRED,
                    reason="operator deferred (no action this cycle)",
                )
            )
            continue

        if response == RESPONSE_ADVANCE:
            commands: list[tuple[str, ...]] = []
            commands.append(("gh", "pr", "ready", str(pr_number), "--repo", GH_REPO))
            commands.append(
                ("gh", "pr", "comment", str(pr_number), "--repo", GH_REPO, "--body", "<advance>")
            )
            if not apply:
                results.append(
                    EntryResult(
                        pr_number=pr_number,
                        title=title,
                        response=response,
                        status=STATUS_WOULD_ADVANCE,
                        reason="bucket-C, y, tripwires clear (dry-run)",
                        gh_commands=tuple(commands),
                    )
                )
                continue
            if live_snapshot is None:
                try:
                    live_snapshot = fetch_pr_snapshot(pr_number, runner=runner)
                except LivePrCheckError as exc:
                    results.append(
                        EntryResult(
                            pr_number=pr_number,
                            title=title,
                            response=response,
                            status=STATUS_LIVE_CHECK_FAILED,
                            reason=f"could not verify live PR before mutation ({exc})",
                            gh_commands=tuple(commands),
                        )
                    )
                    continue
            live_guard = _verify_live_pr_before_mutation(
                pr_number,
                expected_head=live_snapshot.head_sha,
                runner=runner,
            )
            if live_guard is not None:
                results.append(
                    EntryResult(
                        pr_number=pr_number,
                        title=title,
                        response=response,
                        status=STATUS_LIVE_CHECK_FAILED,
                        reason=live_guard,
                        gh_commands=tuple(commands),
                    )
                )
                continue
            ready_proc = gh_pr_ready(pr_number, runner=runner)
            if ready_proc.returncode != 0:
                results.append(
                    EntryResult(
                        pr_number=pr_number,
                        title=title,
                        response=response,
                        status=STATUS_GH_FAILED,
                        reason=(
                            f"gh pr ready failed: "
                            f"{(ready_proc.stderr or ready_proc.stdout).strip()}"
                        ),
                        gh_commands=tuple(commands),
                    )
                )
                continue
            comment_proc = gh_pr_comment(pr_number, _advance_comment_body(), runner=runner)
            if comment_proc.returncode != 0:
                # The ready succeeded; the comment is best-effort.
                results.append(
                    EntryResult(
                        pr_number=pr_number,
                        title=title,
                        response=response,
                        status=STATUS_ADVANCED,
                        reason=(
                            "marked ready; comment failed (best-effort): "
                            f"{(comment_proc.stderr or comment_proc.stdout).strip()}"
                        ),
                        gh_commands=tuple(commands),
                    )
                )
                continue
            results.append(
                EntryResult(
                    pr_number=pr_number,
                    title=title,
                    response=response,
                    status=STATUS_ADVANCED,
                    reason="marked ready + commented",
                    gh_commands=tuple(commands),
                )
            )
            continue

        if response == RESPONSE_CLOSE:
            commands = [
                (
                    "gh",
                    "pr",
                    "close",
                    str(pr_number),
                    "--repo",
                    GH_REPO,
                    "--comment",
                    "<close>",
                )
            ]
            if not apply:
                results.append(
                    EntryResult(
                        pr_number=pr_number,
                        title=title,
                        response=response,
                        status=STATUS_WOULD_CLOSE,
                        reason="bucket-C, n, tripwires clear (dry-run)",
                        gh_commands=tuple(commands),
                    )
                )
                continue
            if live_snapshot is None:
                try:
                    live_snapshot = fetch_pr_snapshot(pr_number, runner=runner)
                except LivePrCheckError as exc:
                    results.append(
                        EntryResult(
                            pr_number=pr_number,
                            title=title,
                            response=response,
                            status=STATUS_LIVE_CHECK_FAILED,
                            reason=f"could not verify live PR before mutation ({exc})",
                            gh_commands=tuple(commands),
                        )
                    )
                    continue
            live_guard = _verify_live_pr_before_mutation(
                pr_number,
                expected_head=live_snapshot.head_sha,
                runner=runner,
            )
            if live_guard is not None:
                results.append(
                    EntryResult(
                        pr_number=pr_number,
                        title=title,
                        response=response,
                        status=STATUS_LIVE_CHECK_FAILED,
                        reason=live_guard,
                        gh_commands=tuple(commands),
                    )
                )
                continue
            close_proc = gh_pr_close(pr_number, _close_comment_body(), runner=runner)
            if close_proc.returncode != 0:
                results.append(
                    EntryResult(
                        pr_number=pr_number,
                        title=title,
                        response=response,
                        status=STATUS_GH_FAILED,
                        reason=(
                            f"gh pr close failed: "
                            f"{(close_proc.stderr or close_proc.stdout).strip()}"
                        ),
                        gh_commands=tuple(commands),
                    )
                )
                continue
            results.append(
                EntryResult(
                    pr_number=pr_number,
                    title=title,
                    response=response,
                    status=STATUS_CLOSED,
                    reason="closed with operator comment",
                    gh_commands=tuple(commands),
                )
            )
            continue

    return results


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


def _policy_version() -> str:
    if not POLICY_DOC.is_file():
        return "unknown"
    text = POLICY_DOC.read_text(encoding="utf-8")
    for marker in ("Version: ", "version: "):
        idx = text.find(marker)
        if idx == -1:
            continue
        end = text.find("\n", idx)
        return text[idx + len(marker) : end].strip()
    return "tracked-doc"


def render_receipt(
    results: list[EntryResult],
    *,
    apply: bool,
    now: datetime.datetime | None = None,
) -> str:
    now = now or datetime.datetime.now(datetime.timezone.utc)
    lines: list[str] = []
    lines.append("# Bucket C receipt")
    lines.append("")
    lines.append(f"- Generated: `{now.strftime('%Y-%m-%dT%H:%M:%SZ')}`")
    lines.append(f"- Mode: `{'apply' if apply else 'dry-run'}`")
    lines.append(f"- Policy version: `{_policy_version()}`")
    lines.append("")
    lines.append("## Decisions")
    lines.append("")
    if not results:
        lines.append("No Bucket C PRs available with responses; nothing to do.")
        lines.append("")
        return "\n".join(lines)
    lines.append("| PR | Response | Status | Reason |")
    lines.append("|---|---|---|---|")
    for r in results:
        resp = r.response or "—"
        lines.append(f"| #{r.pr_number} | `{resp}` | `{r.status}` | {r.reason} |")
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
    path = receipt_dir / f"BUCKET_C_RECEIPT_{stamp}.md"
    path.write_text(receipt_md, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def emit_json(
    results: list[EntryResult],
    *,
    apply: bool,
    receipt_path: Path | None,
) -> str:
    payload = {
        "mode": "apply" if apply else "dry-run",
        "policy_version": _policy_version(),
        "receipt_path": str(receipt_path) if receipt_path else None,
        "decisions": [dataclasses.asdict(r) for r in results],
    }
    return json.dumps(payload, indent=2, default=list)


def emit_table(results: list[EntryResult], *, apply: bool) -> str:
    lines = [f"triage_bucket_c — mode={'apply' if apply else 'dry-run'}"]
    if not results:
        lines.append("  (no Bucket C PRs with responses)")
        return "\n".join(lines)
    for r in results:
        resp = r.response or "—"
        lines.append(f"  #{r.pr_number:<5d} [{resp}] {r.status:24s} — {r.reason}")
        lines.append(f"      title: {r.title}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Stage 3 of operator-delegation rollout — Bucket C y/n/d batcher.")
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually mutate (advance/close). Without this, dry-run.",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=None,
        help=("Path to a JSON response file mapping PR number to one of y / n / d."),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for y/n/d per Bucket C PR via stdin.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable table.",
    )

    args = parser.parse_args(argv)

    if not args.responses and not args.interactive:
        print(
            "error: must pass --responses FILE or --interactive",
            file=sys.stderr,
        )
        return 2

    if args.responses and args.interactive:
        print(
            "error: --responses and --interactive are mutually exclusive",
            file=sys.stderr,
        )
        return 2

    try:
        triage_payload = run_triage()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    bucket_c_entries = [
        e
        for e in (triage_payload.get("results") or [])
        if isinstance(e, dict) and e.get("bucket") == BUCKET_C
    ]

    if args.responses:
        try:
            responses = load_responses_file(args.responses)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        responses = collect_responses_interactive(bucket_c_entries)

    results = decide(triage_payload, responses, apply=args.apply)

    receipt_path: Path | None = None
    if args.apply:
        receipt_md = render_receipt(results, apply=True)
        receipt_path = write_receipt(receipt_md)

    if args.json:
        print(emit_json(results, apply=args.apply, receipt_path=receipt_path))
    else:
        print(emit_table(results, apply=args.apply))
        if receipt_path is not None:
            print(f"receipt: {receipt_path}")

    # Exit non-zero if any tripwire blocked an attempted advance/close.
    blocked = any(
        r.status in {STATUS_HELD, STATUS_PROTECTED}
        and r.response in {RESPONSE_ADVANCE, RESPONSE_CLOSE}
        for r in results
    )
    failed = any(
        r.status in {STATUS_GH_FAILED, STATUS_INVALID_RESPONSE, STATUS_LIVE_CHECK_FAILED}
        for r in results
    )
    return 1 if (blocked or failed) else 0


if __name__ == "__main__":
    sys.exit(main())
