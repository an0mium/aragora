"""Deterministic non-LLM predicate oracle for Aragora Delegation Contract.

Factory's review of the original Capability Certificate proposal flagged
this as the single most important addition: **predicates must be evaluated
by deterministic code, not by the agent itself or by spot-check models.**

Without this, the Delegation Contract becomes "debugging-by-LLM all the
way down" — every acceptance criterion would be evaluable only by another
LLM, recreating the reward-hacking surface the contract is trying to
close.

This module is the trust kernel: its evaluators are small-surface,
separately-versioned, and never invoke an LLM. Each evaluator is a
direct shell-out to a deterministic tool (gh, pytest, git) or a stdlib
filesystem check.

Predicate string grammar (v0.1, intentionally narrow):

    <predicate>          ::= <name> "(" <arg> ("," <arg>)* ")"
    <name>               ::= identifier matching r"^[a-z][a-z_]*$"
    <arg>                ::= integer | quoted_string | bare_token

Supported predicates (v0.1):

  pr_merged(N)           pr_open(N)            pr_state(N, state)
  tests_pass(path)       file_exists(path)     dir_exists(path)
  branch_exists(name)    commit_landed(sha)    issue_closed(N)

Each evaluator returns a PredicateResult with explicit boolean +
evidence string + error if the evaluator failed. Failures (e.g. gh
unavailable) are NOT silently treated as `satisfied=False`; they
propagate as `error != None` so the caller can distinguish
"definitely-not-satisfied" from "couldn't-check."

Pure stdlib + shell-out. No new pip deps. No network calls except via
the already-installed `gh` and `git` binaries.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Callable

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredicateResult:
    """Outcome of evaluating a single predicate."""

    predicate: str  # original string, e.g. "pr_merged(7336)"
    satisfied: bool  # boolean outcome
    evidence: str  # what was checked (e.g. "merged_at=2026-05-19T18:13:28Z")
    evaluator: str  # which named evaluator ran it
    evaluated_at: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    error: str | None = None  # populated if the evaluator itself failed


# ---------------------------------------------------------------------------
# Predicate string parser
# ---------------------------------------------------------------------------


_PREDICATE_RE = re.compile(r"^([a-z][a-z_]*)\((.*)\)$")


class PredicateParseError(ValueError):
    """Raised when a predicate string is malformed."""


def parse_predicate(predicate: str) -> tuple[str, list[str]]:
    """Split a predicate string into (name, args).

    >>> parse_predicate("pr_merged(7336)")
    ('pr_merged', ['7336'])
    >>> parse_predicate('tests_pass("tests/scripts/test_foo.py")')
    ('tests_pass', ['tests/scripts/test_foo.py'])
    """
    text = predicate.strip()
    match = _PREDICATE_RE.match(text)
    if not match:
        raise PredicateParseError(f"malformed predicate: {predicate!r}")
    name = match.group(1)
    args_blob = match.group(2).strip()
    if not args_blob:
        return name, []
    args: list[str] = []
    for raw in _split_args(args_blob):
        raw = raw.strip()
        if (raw.startswith('"') and raw.endswith('"')) or (
            raw.startswith("'") and raw.endswith("'")
        ):
            args.append(raw[1:-1])
        else:
            args.append(raw)
    return name, args


def _split_args(text: str) -> list[str]:
    """Split comma-separated args, honoring single/double-quoted strings."""
    out: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    for ch in text:
        if quote is None:
            if ch in ('"', "'"):
                quote = ch
                buf.append(ch)
            elif ch == ",":
                out.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        else:
            buf.append(ch)
            if ch == quote:
                quote = None
    if buf:
        out.append("".join(buf))
    return out


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with capture + timeout. Never raises on non-zero."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def evaluate_pr_merged(predicate: str, args: list[str]) -> PredicateResult:
    if len(args) != 1:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="pr_merged",
            error=f"expected 1 arg, got {len(args)}",
        )
    try:
        pr_num = int(args[0])
    except ValueError:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="pr_merged",
            error=f"arg must be int, got {args[0]!r}",
        )
    try:
        result = _run(["gh", "pr", "view", str(pr_num), "--json", "mergedAt"], timeout=10.0)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="pr_merged",
            error=f"gh unavailable: {exc!r}",
        )
    if result.returncode != 0:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence=result.stderr.strip()[:200],
            evaluator="pr_merged",
            error=f"gh exit {result.returncode}",
        )
    import json

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence=result.stdout.strip()[:200],
            evaluator="pr_merged",
            error=f"gh json decode failed: {exc!r}",
        )
    merged_at = payload.get("mergedAt")
    return PredicateResult(
        predicate=predicate,
        satisfied=merged_at is not None and merged_at != "",
        evidence=f"merged_at={merged_at!r}",
        evaluator="pr_merged",
    )


def evaluate_pr_state(predicate: str, args: list[str]) -> PredicateResult:
    if len(args) != 2:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="pr_state",
            error=f"expected 2 args, got {len(args)}",
        )
    try:
        pr_num = int(args[0])
    except ValueError:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="pr_state",
            error=f"first arg must be int, got {args[0]!r}",
        )
    expected_state = args[1].upper()
    try:
        result = _run(["gh", "pr", "view", str(pr_num), "--json", "state"], timeout=10.0)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="pr_state",
            error=f"gh unavailable: {exc!r}",
        )
    if result.returncode != 0:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence=result.stderr.strip()[:200],
            evaluator="pr_state",
            error=f"gh exit {result.returncode}",
        )
    import json

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence=result.stdout.strip()[:200],
            evaluator="pr_state",
            error=f"gh json decode failed: {exc!r}",
        )
    actual = (payload.get("state") or "").upper()
    return PredicateResult(
        predicate=predicate,
        satisfied=actual == expected_state,
        evidence=f"state={actual!r} expected={expected_state!r}",
        evaluator="pr_state",
    )


def evaluate_pr_open(predicate: str, args: list[str]) -> PredicateResult:
    if len(args) != 1:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="pr_open",
            error=f"expected 1 arg, got {len(args)}",
        )
    inner = evaluate_pr_state(predicate, [args[0], "OPEN"])
    return PredicateResult(
        predicate=inner.predicate,
        satisfied=inner.satisfied,
        evidence=inner.evidence,
        evaluator="pr_open",
        evaluated_at=inner.evaluated_at,
        error=inner.error,
    )


def evaluate_tests_pass(predicate: str, args: list[str]) -> PredicateResult:
    if len(args) != 1:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="tests_pass",
            error=f"expected 1 arg, got {len(args)}",
        )
    path = args[0]
    if not os.path.exists(path):
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence=f"path_missing={path!r}",
            evaluator="tests_pass",
        )
    try:
        result = _run(["python3", "-m", "pytest", path, "-q", "--tb=no"], timeout=300.0)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="tests_pass",
            error=f"pytest unavailable or timed out: {exc!r}",
        )
    return PredicateResult(
        predicate=predicate,
        satisfied=result.returncode == 0,
        evidence=f"pytest exit={result.returncode}; last_line={result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ''!r}",
        evaluator="tests_pass",
    )


def evaluate_file_exists(predicate: str, args: list[str]) -> PredicateResult:
    if len(args) != 1:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="file_exists",
            error=f"expected 1 arg, got {len(args)}",
        )
    path = args[0]
    exists = os.path.isfile(path)
    return PredicateResult(
        predicate=predicate,
        satisfied=exists,
        evidence=f"isfile({path!r})={exists}",
        evaluator="file_exists",
    )


def evaluate_dir_exists(predicate: str, args: list[str]) -> PredicateResult:
    if len(args) != 1:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="dir_exists",
            error=f"expected 1 arg, got {len(args)}",
        )
    path = args[0]
    exists = os.path.isdir(path)
    return PredicateResult(
        predicate=predicate,
        satisfied=exists,
        evidence=f"isdir({path!r})={exists}",
        evaluator="dir_exists",
    )


def evaluate_branch_exists(predicate: str, args: list[str]) -> PredicateResult:
    if len(args) != 1:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="branch_exists",
            error=f"expected 1 arg, got {len(args)}",
        )
    name = args[0]
    try:
        local = _run(["git", "branch", "--list", name], timeout=5.0)
        remote = _run(["git", "ls-remote", "--heads", "origin", name], timeout=10.0)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="branch_exists",
            error=f"git unavailable: {exc!r}",
        )
    local_hit = bool(local.stdout.strip())
    remote_hit = bool(remote.stdout.strip())
    return PredicateResult(
        predicate=predicate,
        satisfied=local_hit or remote_hit,
        evidence=f"local={local_hit} remote={remote_hit}",
        evaluator="branch_exists",
    )


def evaluate_commit_landed(predicate: str, args: list[str]) -> PredicateResult:
    if len(args) != 1:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="commit_landed",
            error=f"expected 1 arg, got {len(args)}",
        )
    sha = args[0]
    try:
        result = _run(["git", "merge-base", "--is-ancestor", sha, "origin/main"], timeout=10.0)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="commit_landed",
            error=f"git unavailable: {exc!r}",
        )
    # merge-base --is-ancestor exits 0 if ancestor, 1 if not.
    return PredicateResult(
        predicate=predicate,
        satisfied=result.returncode == 0,
        evidence=f"is_ancestor_of_origin_main={result.returncode == 0}",
        evaluator="commit_landed",
    )


def evaluate_issue_closed(predicate: str, args: list[str]) -> PredicateResult:
    if len(args) != 1:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="issue_closed",
            error=f"expected 1 arg, got {len(args)}",
        )
    try:
        issue_num = int(args[0])
    except ValueError:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="issue_closed",
            error=f"arg must be int, got {args[0]!r}",
        )
    try:
        result = _run(["gh", "issue", "view", str(issue_num), "--json", "state"], timeout=10.0)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="issue_closed",
            error=f"gh unavailable: {exc!r}",
        )
    if result.returncode != 0:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence=result.stderr.strip()[:200],
            evaluator="issue_closed",
            error=f"gh exit {result.returncode}",
        )
    import json

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence=result.stdout.strip()[:200],
            evaluator="issue_closed",
            error=f"gh json decode failed: {exc!r}",
        )
    state = (payload.get("state") or "").upper()
    return PredicateResult(
        predicate=predicate,
        satisfied=state == "CLOSED",
        evidence=f"state={state!r}",
        evaluator="issue_closed",
    )


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------


EvaluatorFn = Callable[[str, list[str]], PredicateResult]

EVALUATORS: dict[str, EvaluatorFn] = {
    "pr_merged": evaluate_pr_merged,
    "pr_open": evaluate_pr_open,
    "pr_state": evaluate_pr_state,
    "tests_pass": evaluate_tests_pass,
    "file_exists": evaluate_file_exists,
    "dir_exists": evaluate_dir_exists,
    "branch_exists": evaluate_branch_exists,
    "commit_landed": evaluate_commit_landed,
    "issue_closed": evaluate_issue_closed,
}


def evaluate_predicate(
    predicate: str, *, evaluators: dict[str, EvaluatorFn] | None = None
) -> PredicateResult:
    """Parse + dispatch a predicate string to its registered evaluator."""
    registry = evaluators if evaluators is not None else EVALUATORS
    try:
        name, args = parse_predicate(predicate)
    except PredicateParseError as exc:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="(none)",
            error=str(exc),
        )
    fn = registry.get(name)
    if fn is None:
        return PredicateResult(
            predicate=predicate,
            satisfied=False,
            evidence="",
            evaluator="(none)",
            error=f"unknown predicate name: {name!r}",
        )
    return fn(predicate, args)


def evaluate_all(predicates: list[str]) -> list[PredicateResult]:
    """Evaluate a batch of predicate strings in declared order."""
    return [evaluate_predicate(p) for p in predicates]
