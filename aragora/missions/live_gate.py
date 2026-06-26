"""Live FleetGate adapter for Aragora mission dispatch.

The adapter keeps the existing policy helpers as the source of read truth, then
uses one exact-head, non-admin GitHub CLI merge call when dispatch is explicitly
running in an auto-drain path.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .dispatch import GateVerdict
from .state import Feature

Runner = Callable[[list[str], Path], str]


class LiveBossLoopGate:
    """Thin live binding for :class:`aragora.missions.dispatch.FleetGate`."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        repo_slug: str = "synaptent/aragora",
        base: str = "origin/main",
        runner: Runner | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.repo_slug = repo_slug
        self.base = base
        self.runner = runner or _run
        self._metadata_by_branch: dict[str, dict[str, Any]] = {}

    def branch_for(self, feature: Feature) -> str:
        metadata = dict(feature.metadata)
        branch = str(metadata.get("branch") or f"mission/{feature.id}")
        self._metadata_by_branch[branch] = metadata
        return branch

    def already_merged(self, branch: str) -> bool:
        output = self.runner(
            ["git", "branch", "--merged", self.base, "--list", branch], self.repo_root
        )
        return bool(output.strip())

    def head_of(self, branch: str) -> str:
        return self.runner(["git", "rev-parse", branch], self.repo_root).strip()

    def foreign_commits(
        self, branch: str, base: str, allowed_prefixes: tuple[str, ...]
    ) -> list[str]:
        output = self.runner(
            ["git", "log", "--format=%H%x09%s", f"{base}..{branch}"],
            self.repo_root,
        )
        if not output.strip():
            return []
        subject_prefixes = _subject_prefixes(allowed_prefixes)
        foreign: list[str] = []
        for line in output.splitlines():
            commit, _, subject = line.partition("\t")
            if not subject.startswith(subject_prefixes):
                foreign.append(f"{commit} {subject}".strip())
        return foreign

    def tier_of(self, feature: Feature) -> int:
        try:
            return int(feature.metadata.get("tier", 0))
        except (TypeError, ValueError):
            return 3

    def collect_evidence(self, branch: str, head: str) -> GateVerdict:
        metadata = self._metadata_by_branch.get(branch, {})
        pr = metadata.get("pr")
        if pr is None:
            return GateVerdict(satisfied=False, tier=3, dissent=["feature metadata has no PR"])

        payload = self._run_json(
            [
                "python",
                "-m",
                "aragora.cli.main",
                "review-queue",
                "merge-packet",
                "--pr",
                str(pr),
                "--repo",
                self.repo_slug,
                "--json",
            ]
        )
        entry = _find_packet_entry(payload, int(pr), head)
        if entry is None:
            return GateVerdict(
                satisfied=False,
                tier=3,
                dissent=[f"merge-packet had no exact-head entry for PR {pr} at {head}"],
            )

        quorum = entry.get("model_review_quorum")
        if not isinstance(quorum, dict):
            quorum = entry
        tier = _int_or_default(quorum.get("tier"), _int_or_default(entry.get("tier"), 3))
        verdict = str(quorum.get("verdict") or entry.get("verdict") or "").lower()
        reasons = quorum.get("reasons") or entry.get("reasons") or []
        if isinstance(reasons, str):
            reasons = [reasons]
        satisfied = verdict in {"satisfied", "pass", "passed", "green"} or bool(
            entry.get("satisfied")
        )
        return GateVerdict(satisfied=satisfied, tier=tier, dissent=list(reasons))

    def merge_head_bound(self, branch: str, head: str) -> bool:
        metadata = self._metadata_by_branch.get(branch, {})
        pr = metadata.get("pr")
        if pr is None:
            return False
        self.runner(
            [
                "gh",
                "pr",
                "merge",
                str(pr),
                "--repo",
                self.repo_slug,
                "--squash",
                "--match-head-commit",
                head,
            ],
            self.repo_root,
        )
        return True

    def _run_json(self, cmd: list[str]) -> dict[str, Any]:
        output = self.runner(cmd, self.repo_root)
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            return {"error": "unparseable JSON", "raw": output}
        return payload if isinstance(payload, dict) else {"items": payload}


def _run(cmd: list[str], cwd: Path) -> str:
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            (proc.stderr or proc.stdout or f"{cmd[0]} exited {proc.returncode}").strip()
        )
    return proc.stdout


def _find_packet_entry(payload: dict[str, Any], pr: int, head: str) -> dict[str, Any] | None:
    candidates: list[Any] = []
    for key in ("ready", "not_ready", "items", "entries"):
        raw = payload.get(key)
        if isinstance(raw, list):
            candidates.extend(raw)
    if not candidates and any(k in payload for k in ("pr_number", "number", "head_sha")):
        candidates.append(payload)

    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        entry_pr = _int_or_default(entry.get("pr_number", entry.get("number")), -1)
        entry_head = entry.get("head_sha") or entry.get("headRefOid") or entry.get("head")
        if entry_pr == pr and entry_head == head:
            return entry
    return None


def _subject_prefixes(allowed_prefixes: tuple[str, ...]) -> tuple[str, ...]:
    prefixes: set[str] = set(allowed_prefixes)
    for prefix in allowed_prefixes:
        if prefix.endswith("/"):
            prefixes.add(f"{prefix[:-1]}:")
    return tuple(sorted(prefixes))


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
