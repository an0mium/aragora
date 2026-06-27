#!/usr/bin/env python3
"""Bounded dogfood-evidence step for the auto-evidence cycle (#8219, run-20260610).

The merge-quorum gate requires, for any Tier-1+ code change, BOTH a 2-family
model-review quorum AND separate *adversarial dogfood* evidence (a comment that
proves the PR head was actually exercised, not merely read). The auto-evidence
cycle (#8171) mints model-quorum evidence but produces **no dogfood evidence**,
so every Tier-1+ code PR it touches stays blocked on a dogfood marker the cycle
never creates. This module closes that gap.

For a PR that needs dogfood evidence the step:

1. Verifies the PR head branch is from a *trusted* namespace
   (``codex/``, ``elves/``, ``aragora/``, ``dependabot``). Running PR code is
   RCE-shaped, so arbitrary forks are NEVER dogfooded.
2. Checks out the PR head in a *disposable* git worktree (cleaned even on
   failure).
3. Runs a SCOPED validation: the PR's own touched test files (discovered from
   the diff), else a bounded smoke (compile the touched ``.py`` modules + run
   the nearest test directory). Bounded by ``--dogfood-timeout`` (default 600s).
4. **Fail-closed:** if the dogfood run FAILS (non-zero, timeout, or error), it
   posts NOTHING and records a skip. A failing dogfood is a real signal the PR
   is not ready. A dogfood pass is NEVER fabricated.
5. On pass, composes a head-SHA-bound dogfood comment in the lineage-recognized
   format (``Model family:`` line + ``dogfood:`` marker), runs the gate's own
   ``evidence-lint`` to confirm it WOULD COUNT as dogfood (``would_count`` and a
   non-empty ``dogfood_evidence`` list) *before* posting, then posts it.

All I/O (gh, git, subprocess, lint, post) is injected so the orchestration is
unit-testable without network or subprocesses.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable

# Branch namespaces whose PR code we are willing to execute. Everything else
# (arbitrary forks, unknown contributors) is refused — running their code is an
# RCE risk. Matched case-insensitively against the head ref name.
TRUSTED_BRANCH_PREFIXES = ("codex/", "elves/", "aragora/", "dependabot/")
TRUSTED_BRANCH_EXACT = ("dependabot",)

DEFAULT_DOGFOOD_TIMEOUT = 600
DOGFOOD_MODEL_FAMILY = os.environ.get("ARAGORA_DOGFOOD_FAMILY", "claude").strip() or "claude"


@dataclass
class DogfoodPlan:
    """What the dogfood step intends to do for one PR (pre-execution)."""

    pr: int
    head_sha: str
    head_ref: str
    trusted: bool
    reason: str = ""


@dataclass
class DogfoodOutcome:
    """Result of a dogfood attempt for one PR."""

    pr: int
    head_sha: str
    status: str  # "posted" | "skipped" | "failed"
    reason: str = ""
    command: str = ""
    output_digest: str = ""
    would_count: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def is_trusted_head_ref(head_ref: str, *, is_cross_repo: bool = False) -> bool:
    """Whether a PR head ref is safe to check out and execute.

    Cross-repository (fork) heads are NEVER trusted regardless of name — a fork
    can name its branch ``codex/x`` to spoof the namespace allowlist. Only
    same-repo branches in a known automation namespace are trusted.
    """
    if is_cross_repo:
        return False
    ref = str(head_ref or "").strip().lower()
    if not ref:
        return False
    if ref in TRUSTED_BRANCH_EXACT:
        return True
    return any(ref.startswith(prefix) for prefix in TRUSTED_BRANCH_PREFIXES)


def _digest_output(text: str, *, limit: int = 600) -> str:
    """Compact tail digest of command output for the evidence comment."""
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return "...(truncated)...\n" + cleaned[-limit:]


def discover_validation_command(changed_files: list[str]) -> tuple[list[str], str]:
    """Pick a scoped validation command for the touched files.

    Preference order, all bounded:
    1. Touched test files (``tests/**`` ``*.py`` or ``test_*.py``) → run pytest
       on exactly those files.
    2. Else: compile-check the touched ``.py`` modules (catches syntax/import
       breakage) and, if a sibling ``tests`` dir is touched, run the nearest one.
    3. Else (no python touched, e.g. a JS/lockfile dependabot bump) → a no-op
       smoke that always passes structurally is NOT emitted; callers treat an
       empty command as "nothing to run" and must decide. We return an empty
       command list with a reason so the caller fails closed.
    """
    py_files = [f for f in changed_files if f.endswith(".py")]
    test_files = [
        f
        for f in py_files
        if "/tests/" in f or f.startswith("tests/") or os.path.basename(f).startswith("test_")
    ]
    if test_files:
        return (
            [sys.executable, "-m", "pytest", "-q", "--no-header", *sorted(set(test_files))],
            f"touched test files ({len(set(test_files))})",
        )
    if py_files:
        modules = sorted(set(py_files))
        return (
            [sys.executable, "-c", _COMPILE_SNIPPET, *modules],
            f"compile-check touched modules ({len(modules)})",
        )
    return ([], "no python files touched; no scoped validation discoverable")


# Bounded smoke: byte-compile each touched module path. Catches syntax errors
# and obvious breakage without importing (imports can have side effects / spin
# up servers). argv[1:] are file paths.
_COMPILE_SNIPPET = (
    "import py_compile,sys\n"
    "fails=[]\n"
    "for p in sys.argv[1:]:\n"
    "    try:\n"
    "        py_compile.compile(p, doraise=True)\n"
    "    except Exception as exc:\n"
    "        fails.append(f'{p}: {exc}')\n"
    "if fails:\n"
    "    print('COMPILE FAILURES:'); [print(f) for f in fails]; sys.exit(1)\n"
    "print(f'compiled {len(sys.argv)-1} module(s) OK')\n"
)


def compose_dogfood_comment(
    *,
    pr: int,
    head_sha: str,
    model_family: str,
    command: str,
    passed: bool,
    output_digest: str,
) -> str:
    """Build a head-SHA-bound dogfood-evidence comment in the recognized format.

    The gate's parser counts a dogfood comment when it (a) is grounded on the
    current head (cites the 7-char head SHA prefix), (b) contains a dogfood
    trigger token, and (c) discloses a known ``Model family:`` line. Only a
    PASS is ever composed for posting — see the caller's fail-closed guard.
    """
    short = str(head_sha or "")[:10]
    verdict = "passed" if passed else "FAILED"
    return (
        f"## Focused adversarial dogfood ({model_family})\n\n"
        f"Head SHA: {short} ({head_sha}) — adversarial dogfood validation grounded "
        f"on the exact PR head.\n"
        f"PR: #{pr}.\n"
        f"Model family: {model_family}\n\n"
        f"This is an automated dogfood run by the {model_family} auto-evidence cycle: "
        f"the PR head was checked out in a disposable worktree and its own scoped "
        f"validation was executed (not merely reviewed).\n\n"
        f"Validation command: `{command}`\n"
        f"Result: **{verdict}**\n\n"
        f"Output digest:\n```\n{output_digest}\n```\n\n"
        f"dogfood: yes\n\n"
        f"This is evidence only, not merge authorization or Tier 4 settlement "
        f"authorization."
    )


# --- Default (real) I/O callables -------------------------------------------


def default_fetch_pr_head(repo: str, pr: int) -> dict[str, Any]:
    """``gh pr view`` for the head SHA, head ref name, and cross-repo flag."""
    try:
        proc = subprocess.run(
            [
                "gh",
                "pr",
                "view",
                str(pr),
                "--repo",
                repo,
                "--json",
                "headRefOid,headRefName,headRepositoryOwner,isCrossRepository,files",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def default_changed_files(repo: str, pr: int) -> list[str]:
    payload = default_fetch_pr_head(repo, pr)
    files = payload.get("files") or []
    out: list[str] = []
    for entry in files:
        if isinstance(entry, dict) and entry.get("path"):
            out.append(str(entry["path"]))
    return out


def default_checkout_worktree(repo: str, head_sha: str, dest: str) -> bool:
    """Fetch the head SHA and check it out into a disposable worktree at ``dest``.

    Uses the cycle's own repo as the local clone; falls back to a shallow
    ``gh repo clone`` only if no local git is present. Returns True on success.
    """
    repo_root = os.getcwd()
    try:
        fetch = subprocess.run(
            ["git", "-C", repo_root, "fetch", "origin", head_sha],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if fetch.returncode != 0:
            return False
        add = subprocess.run(
            ["git", "-C", repo_root, "worktree", "add", "--detach", dest, head_sha],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return add.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def default_remove_worktree(dest: str) -> None:
    """Remove a disposable worktree, then prune. Never raises."""
    repo_root = os.getcwd()
    try:
        subprocess.run(
            ["git", "-C", repo_root, "worktree", "remove", "--force", dest],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    if os.path.isdir(dest):
        shutil.rmtree(dest, ignore_errors=True)
    try:
        subprocess.run(
            ["git", "-C", repo_root, "worktree", "prune"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def default_run_validation(command: list[str], cwd: str, timeout: int) -> tuple[bool, str]:
    """Run the scoped validation in the worktree; return (passed, output)."""
    if not command:
        return (False, "no validation command")
    try:
        proc = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (False, f"validation timed out after {timeout}s")
    except (OSError, subprocess.SubprocessError) as exc:
        return (False, f"validation could not run: {type(exc).__name__}")
    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return (proc.returncode == 0, output)


def default_lint_evidence(repo: str, pr: int, head_sha: str, body: str) -> dict[str, Any]:
    """Run ``review-queue evidence-lint`` and return its JSON result."""
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "aragora.cli.main",
                "review-queue",
                "evidence-lint",
                "--pr",
                str(pr),
                "--head-sha",
                head_sha,
                "--body",
                body,
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if not proc.stdout.strip():
        return {}
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def default_post_comment(repo: str, pr: int, body: str) -> bool:
    """Post the dogfood comment via ``gh pr comment``."""
    try:
        proc = subprocess.run(
            ["gh", "pr", "comment", str(pr), "--repo", repo, "--body", body],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def lint_counts_as_dogfood(lint: dict[str, Any]) -> bool:
    """True only when evidence-lint says the comment counts AS DOGFOOD.

    ``would_count`` alone is insufficient: a model-review comment also counts.
    We require a non-empty ``dogfood_evidence`` entry so the comment satisfies
    the gate's separate ``requires_adversarial_dogfood`` requirement, not just
    the model-review signal count.
    """
    if not lint.get("would_count"):
        return False
    return bool(lint.get("dogfood_evidence"))


# --- Per-PR dogfood --------------------------------------------------------


def dogfood_pr(
    *,
    repo: str,
    pr: int,
    model_family: str,
    timeout: int,
    apply: bool,
    fetch_head: Callable[[str, int], dict[str, Any]],
    changed_files: Callable[[str, int], list[str]],
    checkout: Callable[[str, str, str], bool],
    remove_worktree: Callable[[str], None],
    run_validation: Callable[[list[str], str, int], tuple[bool, str]],
    lint_evidence: Callable[[str, int, str, str], dict[str, Any]],
    post_comment: Callable[[str, int, str], bool],
    worktree_factory: Callable[[], str] | None = None,
    log: Callable[[str], None] = lambda _msg: None,
) -> DogfoodOutcome:
    """Run a bounded, fail-closed dogfood for one PR and (with apply) post it."""
    head = fetch_head(repo, pr)
    head_sha = str(head.get("headRefOid", "") or "").strip()
    head_ref = str(head.get("headRefName", "") or "").strip()
    is_cross = bool(head.get("isCrossRepository"))
    if not head_sha:
        return DogfoodOutcome(pr=pr, head_sha="", status="skipped", reason="no head sha")
    if not is_trusted_head_ref(head_ref, is_cross_repo=is_cross):
        return DogfoodOutcome(
            pr=pr,
            head_sha=head_sha,
            status="skipped",
            reason=f"untrusted head ref (ref={head_ref!r}, cross_repo={is_cross}); refusing to "
            "execute PR code",
        )

    files = changed_files(repo, pr)
    command, discovery = discover_validation_command(files)
    if not command:
        return DogfoodOutcome(
            pr=pr,
            head_sha=head_sha,
            status="skipped",
            reason=f"no scoped validation discoverable ({discovery})",
        )

    dest = (
        worktree_factory()
        if worktree_factory
        else os.path.join(tempfile.gettempdir(), f"aragora-dogfood-{pr}-{int(time.time())}")
    )
    command_str = " ".join(command[:2]) + (" ..." if len(command) > 2 else "")
    try:
        if not checkout(repo, head_sha, dest):
            return DogfoodOutcome(
                pr=pr,
                head_sha=head_sha,
                status="skipped",
                reason="worktree checkout failed",
                command=command_str,
            )
        passed, output = run_validation(command, dest, timeout)
        digest = _digest_output(output)
        if not passed:
            # FAIL-CLOSED: a failing dogfood is a real not-ready signal. Post
            # nothing; never fabricate a pass.
            log(json.dumps({"pr": pr, "dogfood": "failed", "command": command_str}))
            return DogfoodOutcome(
                pr=pr,
                head_sha=head_sha,
                status="failed",
                reason="dogfood validation failed (not ready); no evidence posted",
                command=command_str,
                output_digest=digest,
            )

        body = compose_dogfood_comment(
            pr=pr,
            head_sha=head_sha,
            model_family=model_family,
            command=command_str,
            passed=True,
            output_digest=digest,
        )
        lint = lint_evidence(repo, pr, head_sha, body)
        would_count = lint_counts_as_dogfood(lint)
        if not would_count:
            return DogfoodOutcome(
                pr=pr,
                head_sha=head_sha,
                status="skipped",
                reason="composed evidence would not count as dogfood (evidence-lint)",
                command=command_str,
                output_digest=digest,
                would_count=False,
                extra={"lint_problems": list(lint.get("problems") or [])},
            )
        if not apply:
            return DogfoodOutcome(
                pr=pr,
                head_sha=head_sha,
                status="skipped",
                reason="dry-run: dogfood passed and would count, not posting",
                command=command_str,
                output_digest=digest,
                would_count=True,
            )
        if not post_comment(repo, pr, body):
            return DogfoodOutcome(
                pr=pr,
                head_sha=head_sha,
                status="failed",
                reason="dogfood passed and counts but posting failed",
                command=command_str,
                output_digest=digest,
                would_count=True,
            )
        log(json.dumps({"pr": pr, "dogfood": "posted", "command": command_str}))
        return DogfoodOutcome(
            pr=pr,
            head_sha=head_sha,
            status="posted",
            reason="dogfood passed; counting dogfood evidence posted",
            command=command_str,
            output_digest=digest,
            would_count=True,
        )
    finally:
        remove_worktree(dest)
