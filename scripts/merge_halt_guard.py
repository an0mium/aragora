#!/usr/bin/env python3
"""The single merge precondition every automated merge path must satisfy.

``.aragora/merge_executor.halt`` is armed when main is red and requires a human
to re-arm. It was only ever read by ``scripts/merge_executor.py``. **Seven** other
scripts can merge and none of them consulted it:

    auto_merge_quorum_green.py        gh pr merge --squash --admin  (daemon-driven)
    auto_merge_bucket_a.py            gh pr merge --squash          (daemon-driven)
    boss_drain_pass.py                subprocess.run                (:232)
    drain_codex_automation_value.py   runner(merge_cmd, ...)        (:706)
    merge_codex_automation_prs.py     gh pr merge                   (:235)
    settle_tier4_pr.py                _run_command(merge_command)   (:2034)

``settle_one_pr.py`` builds the same command string but only appends it to
``report["suggested_commands"]`` — it reports, it does not merge, which matches
the note in #9216 that the reporting lane never invoked ``gh pr merge``.

That is how PRs #9115 (`f4a650dc`) and #9111 (`f40cd7df`) merged on 2026-07-11
while the halt was armed and byte-identical before and after: nothing on the
merging path ever opened the file. See #9216.

This module is the shared guard. Every merge-capable entry point calls
``assert_merge_allowed`` immediately before invoking a merge.

One deliberate divergence: ``merge_executor.py`` keeps its own existence-based
halt check and does **not** honour waivers, because it also *writes* the marker
and re-locks branch protection. So a waived PR can merge through the six guarded
paths while ``merge_executor`` still refuses it. That asymmetry is strictly
fail-closed — the stricter path stays stricter — and is left rather than
rewiring the most sensitive merge path in the same change that introduces the
guard. It is recorded here so the divergence is intentional and visible rather
than discovered later.

Everything here fails CLOSED. An armed halt, a corrupt halt file, a waiver that
does not parse, a waiver for a different PR or a different head — all block. The
only path to a merge while halted is an exact-head waiver that is well-formed,
unexpired, and matches both the PR number and the full head SHA.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

_FULL_SHA = re.compile(r"[0-9a-f]{40}")


def _shared_checkout_root(repo_root: Path) -> Path:
    """Return the primary checkout root for a normal or linked worktree.

    Main-health automation arms one repository-wide halt. A linked worktree
    must therefore consult the primary checkout's marker rather than its own
    private ``.aragora`` directory. Malformed linked-worktree metadata raises
    at import time so merge entry points fail closed.
    """
    dot_git = repo_root / ".git"
    try:
        mode = os.stat(dot_git).st_mode
    except FileNotFoundError:
        return repo_root
    except OSError as exc:
        raise RuntimeError(f"could not inspect git metadata at {dot_git}: {exc}") from exc

    if stat.S_ISDIR(mode):
        return repo_root
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"git metadata at {dot_git} is neither a file nor directory")

    try:
        marker = dot_git.read_text(encoding="utf-8").strip()
        prefix = "gitdir:"
        if not marker.lower().startswith(prefix):
            raise ValueError("missing gitdir marker")
        git_dir = Path(marker[len(prefix) :].strip())
        if not git_dir.is_absolute():
            git_dir = (repo_root / git_dir).resolve()
        common_raw = (git_dir / "commondir").read_text(encoding="utf-8").strip()
        if not common_raw:
            raise ValueError("empty commondir")
        common_dir = Path(common_raw)
        if not common_dir.is_absolute():
            common_dir = (git_dir / common_dir).resolve()
        if not (common_dir / "objects").is_dir():
            raise ValueError(f"invalid common git directory {common_dir}")
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not resolve shared git state from {dot_git}: {exc}") from exc
    return common_dir.parent


SHARED_REPO_ROOT = _shared_checkout_root(_REPO_ROOT)
DEFAULT_HALT_FILE = SHARED_REPO_ROOT / ".aragora" / "merge_executor.halt"
DEFAULT_WAIVER_FILE = SHARED_REPO_ROOT / ".aragora" / "merge_executor.waiver"

# Merge-capable scripts that must route through this guard. The companion test
# asserts this list matches reality, so a new merge path cannot be added without
# either wiring the guard or consciously editing this constant.
MERGE_CAPABLE_SCRIPTS = (
    "auto_merge_bucket_a.py",
    "auto_merge_quorum_green.py",
    "boss_drain_pass.py",
    "drain_codex_automation_value.py",
    "merge_codex_automation_prs.py",
    "merge_executor.py",
    "settle_tier4_pr.py",
)

# Scripts that build or name a merge command but never execute one. Recorded
# explicitly so the scan can stay aggressive without the exclusions being invisible;
# the companion test re-derives each claim rather than trusting this dict.
NON_MERGE_MENTIONS = {
    "fable_goal_cycle.py": "ACTIVE_PROCESS_COMMAND_PATTERNS — matches running processes",
    "settle_one_pr.py": 'appends to report["suggested_commands"] — reports, never executes',
}


class MergeHalted(RuntimeError):
    """Raised when a merge is attempted while the halt is armed."""


@dataclass(frozen=True)
class HaltDecision:
    allowed: bool
    reason: str
    halt_reason: str | None = None
    waiver_actor: str | None = None


def _now(now: dt.datetime | None) -> dt.datetime:
    return now or dt.datetime.now(dt.timezone.utc)


def _read_json(path: Path) -> tuple[dict | None, str | None]:
    """Return (payload, error). A present-but-unreadable file is an error, not absence.

    Deliberately uses ``os.stat`` rather than ``Path.exists()``: ``exists()``
    swallows every ``OSError`` and returns False, so an unreadable parent
    directory or a permissions failure would present an *armed* halt marker as
    absent — allowing the merge. That is a fail-open path inside the guard whose
    only job is to fail closed. Only ``FileNotFoundError`` means "no halt".
    """
    try:
        os.stat(path)
    except FileNotFoundError:
        return None, None
    except OSError as exc:
        return None, f"{path.name} could not be stat'd ({exc})"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"{path.name} exists but could not be read ({exc})"
    if not isinstance(data, dict):
        return None, f"{path.name} is not a JSON object"
    return data, None


def _waiver_applies(waiver: dict, *, pr: int, head_sha: str, now: dt.datetime) -> tuple[bool, str]:
    """An exact-head, unexpired, same-PR waiver — or a reason it does not apply."""
    try:
        waiver_pr = int(waiver["pr"])
        waiver_head = str(waiver["head_sha"]).strip().lower()
        actor = str(waiver["actor"]).strip()
        scope = str(waiver["scope"]).strip()
        expires_raw = str(waiver["expires_at"]).strip()
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"waiver is missing or malformed required fields ({exc})"

    if not actor:
        return False, "waiver has an empty actor"
    if scope != "single-pr":
        return False, f"waiver scope must be 'single-pr', got {scope!r}"
    if waiver_pr != pr:
        return False, f"waiver is for PR #{waiver_pr}, not #{pr}"

    head = head_sha.strip().lower()
    # Exact head only. A prefix match would let a waiver survive a force-push,
    # which is precisely the "stale-head waiver" case #9216 asks to reject.
    #
    # Equality alone is not enough: if a caller passes an abbreviated SHA and the
    # waiver holds the same abbreviation, the two match and the waiver applies to
    # every commit sharing that prefix. Both sides must therefore be full 40-hex.
    for label, value in (("waiver", waiver_head), ("PR", head)):
        if not _FULL_SHA.fullmatch(value):
            return False, f"{label} head {value[:12] or '(empty)'!r} is not a full 40-char SHA"
    if waiver_head != head:
        return False, f"waiver head {waiver_head[:12]} != PR head {head[:12]}"

    try:
        expires = dt.datetime.fromisoformat(expires_raw)
    except ValueError:
        return False, f"waiver expires_at is not ISO-8601 ({expires_raw!r})"
    if expires.tzinfo is None:
        return False, "waiver expires_at must include an explicit timezone"
    if expires <= now:
        return False, f"waiver expired at {expires.isoformat()}"

    return True, actor


def evaluate(
    pr: int,
    head_sha: str,
    *,
    halt_file: Path | None = None,
    waiver_file: Path | None = None,
    now: dt.datetime | None = None,
) -> HaltDecision:
    """Decide whether this exact PR at this exact head may merge right now."""
    halt_path = halt_file or DEFAULT_HALT_FILE
    waiver_path = waiver_file or DEFAULT_WAIVER_FILE
    moment = _now(now)

    halt, halt_error = _read_json(halt_path)
    if halt_error:
        # Present but unreadable: assume armed. A corrupt marker must not read
        # as "no halt" — that is the fail-open shape this guard exists to remove.
        return HaltDecision(False, f"halt marker unreadable, failing closed: {halt_error}")
    if halt is None:
        return HaltDecision(True, "no halt marker present")

    halt_reason = str(halt.get("reason") or "unknown")

    waiver, waiver_error = _read_json(waiver_path)
    if waiver_error:
        return HaltDecision(
            False, f"halt armed ({halt_reason}); {waiver_error}", halt_reason=halt_reason
        )
    if waiver is None:
        return HaltDecision(
            False,
            f"halt armed ({halt_reason}) and no waiver present",
            halt_reason=halt_reason,
        )

    applies, detail = _waiver_applies(waiver, pr=pr, head_sha=head_sha, now=moment)
    if not applies:
        return HaltDecision(False, f"halt armed ({halt_reason}); {detail}", halt_reason=halt_reason)

    return HaltDecision(
        True,
        f"halt armed ({halt_reason}) but waived for PR #{pr} at {head_sha[:12]} by {detail}",
        halt_reason=halt_reason,
        waiver_actor=detail,
    )


def assert_merge_allowed(
    pr: int,
    head_sha: str,
    *,
    halt_file: Path | None = None,
    waiver_file: Path | None = None,
    now: dt.datetime | None = None,
) -> HaltDecision:
    """Raise ``MergeHalted`` unless this PR at this head may merge."""
    decision = evaluate(pr, head_sha, halt_file=halt_file, waiver_file=waiver_file, now=now)
    if not decision.allowed:
        raise MergeHalted(f"refusing to merge PR #{pr}: {decision.reason}")
    return decision


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--halt-file", type=Path, default=DEFAULT_HALT_FILE)
    parser.add_argument("--waiver-file", type=Path, default=DEFAULT_WAIVER_FILE)
    args = parser.parse_args(argv)

    decision = evaluate(
        args.pr, args.head_sha, halt_file=args.halt_file, waiver_file=args.waiver_file
    )
    print(("ALLOW: " if decision.allowed else "BLOCK: ") + decision.reason)
    return 0 if decision.allowed else 3


if __name__ == "__main__":
    sys.exit(main())
