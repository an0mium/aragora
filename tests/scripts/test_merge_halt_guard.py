"""Guard the merge halt against being bypassed by any merge path (#9216).

`.aragora/merge_executor.halt` is armed when main is red and needs a human to
re-arm. It was read by exactly one of the **seven** scripts that execute a merge,
so PRs #9115 and #9111 merged on 2026-07-11 while it was armed — the merging path
never opened the file. The halt SHA-256 was byte-identical before and after
because nothing consulted it, let alone cleared it.

Three kinds of test here:

* behavioural — the guard fails closed on every ambiguous input (armed halt,
  corrupt marker, corrupt/expired/mismatched waiver) and admits only an
  exact-head, unexpired, same-PR waiver;
* structural — `test_every_merge_capable_script_routes_through_the_guard` scans
  `scripts/` for real merge invocations and asserts each one consults the halt.
  That is the test that would have caught the original bug, and it is what keeps
  an eighth merge path from landing unguarded;
* end-to-end — `test_bucket_a_does_not_execute_gh_while_halted` arms a halt and
  calls a real merge function with a spy runner. The structural test only proves
  the guard is *imported*; an import that is never called would satisfy it and
  still merge, so one path is proven to actually stop.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.merge_halt_guard as guard  # noqa: E402

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

PR = 9115
HEAD = "f4a650dcd532596dcf93ceeed69e70b4bce90420"
NOW = dt.datetime(2026, 7, 11, 6, 0, tzinfo=dt.timezone.utc)


@pytest.fixture
def halt(tmp_path: Path) -> Path:
    path = tmp_path / "merge_executor.halt"
    path.write_text(
        json.dumps({"reason": "main_red", "written_at": "2026-07-10T08:30:12Z"}) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def waiver_path(tmp_path: Path) -> Path:
    return tmp_path / "merge_executor.waiver"


def _write_waiver(path: Path, **overrides) -> Path:
    payload = {
        "pr": PR,
        "head_sha": HEAD,
        "actor": "scarmani",
        "scope": "single-pr",
        "reason": "incident waiver",
        "granted_at": "2026-07-11T05:00:00+00:00",
        "expires_at": "2026-07-11T12:00:00+00:00",
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _evaluate(halt_file: Path, waiver_file: Path, *, pr: int = PR, head: str = HEAD):
    return guard.evaluate(pr, head, halt_file=halt_file, waiver_file=waiver_file, now=NOW)


def test_no_halt_marker_allows_merge(tmp_path: Path, waiver_path: Path) -> None:
    decision = _evaluate(tmp_path / "absent.halt", waiver_path)
    assert decision.allowed, decision.reason


def test_armed_halt_blocks_merge(halt: Path, waiver_path: Path) -> None:
    """The #9115 / #9111 case: armed halt, no waiver, merge must not proceed."""
    decision = _evaluate(halt, waiver_path)
    assert not decision.allowed
    assert decision.halt_reason == "main_red"


def test_exact_head_waiver_allows_merge(halt: Path, waiver_path: Path) -> None:
    _write_waiver(waiver_path)
    decision = _evaluate(halt, waiver_path)
    assert decision.allowed, decision.reason
    assert decision.waiver_actor == "scarmani"


def test_waiver_requires_single_pr_scope(halt: Path, waiver_path: Path) -> None:
    _write_waiver(waiver_path, scope="all")
    decision = _evaluate(halt, waiver_path)
    assert not decision.allowed
    assert "single-pr" in decision.reason


def test_waiver_requires_timezone_aware_expiry(halt: Path, waiver_path: Path) -> None:
    _write_waiver(waiver_path, expires_at="2026-07-11T12:00:00")
    decision = _evaluate(halt, waiver_path)
    assert not decision.allowed
    assert "timezone" in decision.reason


def test_linked_worktree_uses_primary_checkout_state(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    common = primary / ".git"
    git_dir = common / "worktrees" / "worker"
    (common / "objects").mkdir(parents=True)
    git_dir.mkdir(parents=True)
    (git_dir / "commondir").write_text("../..\n", encoding="utf-8")

    worker = tmp_path / "worker"
    worker.mkdir()
    (worker / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")

    assert guard._shared_checkout_root(worker) == primary


def test_malformed_linked_worktree_metadata_fails_closed(tmp_path: Path) -> None:
    worker = tmp_path / "worker"
    worker.mkdir()
    (worker / ".git").write_text("not-a-gitdir\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="could not resolve shared git state"):
        guard._shared_checkout_root(worker)


def test_waiver_for_a_different_pr_does_not_apply(halt: Path, waiver_path: Path) -> None:
    _write_waiver(waiver_path, pr=9111)
    assert not _evaluate(halt, waiver_path).allowed


def test_waiver_for_a_different_head_does_not_apply(halt: Path, waiver_path: Path) -> None:
    """Stale-head waivers must die on force-push, which is why the match is exact."""
    _write_waiver(waiver_path, head_sha="0" * 40)
    assert not _evaluate(halt, waiver_path).allowed


def test_waiver_head_prefix_is_not_enough(halt: Path, waiver_path: Path) -> None:
    """A prefix match would survive a force-push to a sibling commit."""
    _write_waiver(waiver_path, head_sha=HEAD[:12])
    assert not _evaluate(halt, waiver_path).allowed


def test_matching_prefix_on_both_sides_still_does_not_apply(halt: Path, waiver_path: Path) -> None:
    """Equality is not enough — both sides must be a full SHA.

    Found by the openai reviewer on #9677, verified by calling `_waiver_applies`
    directly: with waiver head and caller head both set to `HEAD[:12]`, the old
    code returned `(True, "scarmani")`. A caller passing an abbreviated SHA (e.g.
    from `git rev-parse --short`) would match a waiver covering every commit that
    shares the prefix, which is the stale-head hole the exact match exists to close.
    """
    _write_waiver(waiver_path, head_sha=HEAD[:12])
    decision = guard.evaluate(PR, HEAD[:12], halt_file=halt, waiver_file=waiver_path, now=NOW)
    assert not decision.allowed, decision.reason


def test_unreadable_halt_marker_is_not_treated_as_absent(tmp_path: Path, waiver_path: Path) -> None:
    """`Path.exists()` returns False on any OSError, which would read armed as absent.

    Found by the claude reviewer on #9677. Simulated with an unsearchable parent
    directory: the marker is present, but a stat failure must fail closed rather
    than report "no halt marker present".
    """
    import os as _os
    import stat as _stat

    parent = tmp_path / "locked"
    parent.mkdir()
    marker = parent / "merge_executor.halt"
    marker.write_text('{"reason": "main_red"}', encoding="utf-8")
    _os.chmod(parent, 0o000)
    try:
        if _os.access(parent, _os.X_OK):  # running as root; the stat would succeed
            pytest.skip("cannot revoke directory access as this user")
        decision = guard.evaluate(PR, HEAD, halt_file=marker, waiver_file=waiver_path, now=NOW)
        assert not decision.allowed, decision.reason
        assert "no halt marker present" not in decision.reason
    finally:
        _os.chmod(parent, _stat.S_IRWXU)


def test_expired_waiver_does_not_apply(halt: Path, waiver_path: Path) -> None:
    _write_waiver(waiver_path, expires_at="2026-07-11T05:59:00+00:00")
    assert not _evaluate(halt, waiver_path).allowed


def test_waiver_without_actor_does_not_apply(halt: Path, waiver_path: Path) -> None:
    _write_waiver(waiver_path, actor="")
    assert not _evaluate(halt, waiver_path).allowed


def test_corrupt_halt_marker_fails_closed(tmp_path: Path, waiver_path: Path) -> None:
    """An unreadable marker must not read as 'no halt'."""
    bad = tmp_path / "merge_executor.halt"
    bad.write_text("{not json", encoding="utf-8")
    decision = guard.evaluate(PR, HEAD, halt_file=bad, waiver_file=waiver_path, now=NOW)
    assert not decision.allowed
    assert "failing closed" in decision.reason


def test_corrupt_waiver_fails_closed(halt: Path, waiver_path: Path) -> None:
    waiver_path.write_text("{not json", encoding="utf-8")
    assert not _evaluate(halt, waiver_path).allowed


def test_waiver_without_halt_is_irrelevant(tmp_path: Path, waiver_path: Path) -> None:
    """A stray waiver must not be required, or block, when nothing is halted."""
    _write_waiver(waiver_path)
    assert _evaluate(tmp_path / "absent.halt", waiver_path).allowed


def test_assert_merge_allowed_raises_when_halted(halt: Path, waiver_path: Path) -> None:
    with pytest.raises(guard.MergeHalted) as excinfo:
        guard.assert_merge_allowed(PR, HEAD, halt_file=halt, waiver_file=waiver_path, now=NOW)
    assert str(PR) in str(excinfo.value)


def test_cli_exits_nonzero_while_halted(halt: Path, waiver_path: Path) -> None:
    code = guard.main(
        [
            "--pr",
            str(PR),
            "--head-sha",
            HEAD,
            "--halt-file",
            str(halt),
            "--waiver-file",
            str(waiver_path),
        ]
    )
    assert code == 3


# --------------------------------------------------------------------------
# Structural: the test that would have caught #9216
# --------------------------------------------------------------------------

# A merge that is *invoked*, not merely named. A bare `"merge",` match is far too
# loose — it also hits gh-subcommand allowlists, worktree `--strategy merge`, and
# running-process patterns, which is why these require the full argv sequence or an
# interpolated command string.
_MERGE_INVOCATIONS = (
    re.compile(r'\[[^\]]*"gh"\s*,\s*(?:\n\s*)?"pr"\s*,\s*(?:\n\s*)?"merge"', re.S),
    re.compile(r'"pr"\s*,\s*(?:\n\s*)?"merge"\s*,\s*(?:\n\s*)?str\(', re.S),
    re.compile(r'f?"gh pr merge [^"]*\{'),
    # Incrementally built argv: `cmd = ["gh", "pr"]` then `cmd += ["merge", ...]`.
    # Flagged as a residual gap by the claude reviewer on #9677.
    re.compile(r'\+=\s*\[\s*"merge"'),
    re.compile(r'\.append\(\s*"merge"\s*\)'),
)


def _scripts_that_merge() -> set[str]:
    found: set[str] = set()
    for path in sorted(SCRIPTS_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        # Strip docstrings so prose mentioning `gh pr merge` does not count.
        stripped = re.sub(r'"""(?:.|\n)*?"""', "", text)
        if any(rx.search(stripped) for rx in _MERGE_INVOCATIONS):
            found.add(path.name)
    return found


def test_merge_capable_script_list_is_current() -> None:
    """Keep the declared list honest as scripts/ changes.

    The first version of this scan matched a bare `"merge",` and produced 13 false
    positives (`generate_openapi.py`, worktree `--strategy merge`, gh-subcommand
    allowlists). A guard that cries wolf gets its list padded with junk, and the
    real entries stop being legible — so the pattern demands an actual invocation.
    """
    actual = _scripts_that_merge()
    declared = set(guard.MERGE_CAPABLE_SCRIPTS)
    unlisted = actual - declared - set(guard.NON_MERGE_MENTIONS)
    assert not unlisted, (
        f"script(s) invoke a merge but are absent from MERGE_CAPABLE_SCRIPTS: {sorted(unlisted)}. "
        "Add them and wire the guard, or the halt will not cover them (#9216)."
    )


# What makes each exclusion true, re-checked rather than taken on faith.
_EXCLUSION_EVIDENCE = {
    "fable_goal_cycle.py": "ACTIVE_PROCESS_COMMAND_PATTERNS",
    "settle_one_pr.py": 'suggested_commands"].append',
}

# The execution-shaped forms: an argv list handed to a runner. A script that only
# interpolates a command into a *string* may still be merely reporting it, which is
# the distinction between settle_one_pr.py (reports) and settle_tier4_pr.py (runs).
_EXECUTION_SHAPED = _MERGE_INVOCATIONS[:2]


@pytest.mark.parametrize("script", sorted(guard.NON_MERGE_MENTIONS))
def test_non_merge_exclusions_really_do_not_merge(script: str) -> None:
    """The exclusion list must not become a place to hide a real merge path."""
    text = (SCRIPTS_DIR / script).read_text(encoding="utf-8")
    stripped = re.sub(r'"""(?:.|\n)*?"""', "", text)

    executing = [rx.pattern[:30] for rx in _EXECUTION_SHAPED if rx.search(stripped)]
    assert not executing, (
        f"{script} is excluded as non-merging but now builds an executable merge argv "
        f"({executing}). Move it into MERGE_CAPABLE_SCRIPTS and wire the guard (#9216)."
    )

    marker = _EXCLUSION_EVIDENCE[script]
    assert marker in text, (
        f"{script} is excluded because of {marker!r}, which is no longer present. "
        "Re-verify whether it now merges."
    )


@pytest.mark.parametrize("script", guard.MERGE_CAPABLE_SCRIPTS)
def test_every_merge_capable_script_routes_through_the_guard(script: str) -> None:
    """Each merge path must consult the halt.

    `merge_executor.py` owns the halt file itself; the rest must import this guard.
    Before #9216 only `merge_executor.py` consulted it, and the four others merged
    freely while main was halted.
    """
    text = (SCRIPTS_DIR / script).read_text(encoding="utf-8")
    if script == "merge_executor.py":
        assert "merge_executor.halt" in text or "HALT_FILE" in text, (
            "merge_executor.py no longer reads its own halt marker"
        )
        return
    assert "merge_halt_guard" in text, (
        f"{script} can merge but never consults the halt. This is exactly how "
        "PRs #9115 and #9111 merged while the halt was armed (#9216)."
    )


# --------------------------------------------------------------------------
# End-to-end: a wired path must not merely import the guard, it must obey it
# --------------------------------------------------------------------------


def test_bucket_a_does_not_execute_gh_while_halted(
    halt: Path, waiver_path: Path, monkeypatch
) -> None:
    """Arm the halt, call the real merge function, assert no gh command ran.

    The structural tests above prove the guard is imported. This proves it is
    *obeyed*: an import that is never called would satisfy them and still merge.
    """
    import scripts.auto_merge_bucket_a as bucket_a

    monkeypatch.setattr(guard, "DEFAULT_HALT_FILE", halt)
    monkeypatch.setattr(guard, "DEFAULT_WAIVER_FILE", waiver_path)

    invoked: list[list[str]] = []

    def spy_runner(args, *rest, **kwargs):
        invoked.append(list(args))
        raise AssertionError(f"a merge command was executed while halted: {args}")

    with pytest.raises(guard.MergeHalted):
        bucket_a.gh_pr_merge_squash(PR, HEAD, runner=spy_runner)

    assert invoked == [], f"guard did not stop execution; ran {invoked}"


def test_bucket_a_executes_when_not_halted(tmp_path: Path, waiver_path: Path, monkeypatch) -> None:
    """The guard must not block the normal path — otherwise it is just an outage."""
    import subprocess as sp

    import scripts.auto_merge_bucket_a as bucket_a

    monkeypatch.setattr(guard, "DEFAULT_HALT_FILE", tmp_path / "absent.halt")
    monkeypatch.setattr(guard, "DEFAULT_WAIVER_FILE", waiver_path)

    invoked: list[list[str]] = []

    def spy_runner(args, *rest, **kwargs):
        invoked.append(list(args))
        return sp.CompletedProcess(args, 0, stdout="merged", stderr="")

    bucket_a.gh_pr_merge_squash(PR, HEAD, runner=spy_runner)
    assert invoked and invoked[0][:3] == ["gh", "pr", "merge"], invoked
