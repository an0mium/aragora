"""Tests for ``scripts/docs_sync_drift_detector.py`` and its loop-control wiring.

The central assertions mirror the Loop Control Plane v1 proof style: driving the
detector with a faked subprocess seam records every command issued, so tests can
prove the mutation surface exactly - check mode issues no ``commit``/``push``/
``gh pr`` calls at all, apply mode pushes and opens at most one PR, and **no**
mode ever merges, approves, comments, or reruns anything.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import aragora.swarm.loop_control_io as io_mod  # noqa: E402
from aragora.swarm.loop_control import (  # noqa: E402
    LOOP_SPECS,
    HaltVerdict,
    LoopKind,
    audit_halt_readiness,
    classify_loop,
)


def _load_detector() -> Any:
    script_path = REPO_ROOT / "scripts" / "docs_sync_drift_detector.py"
    spec = importlib.util.spec_from_file_location("docs_sync_drift_under_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


detector = _load_detector()

# Arguments that must never appear as an exact arg in any command the detector
# issues, in any mode.
FORBIDDEN_ARGS = {
    "merge",
    "--admin",
    "approve",
    "--approve",
    "comment",
    "rerun",
    "--force-with-lease",
}

SHA = "1234567890abcdef1234567890abcdef12345678"
MIRROR = "docs-site/docs/contributing/b0-benchmark-truth-status.md"


class FakeRunner:
    """Scriptable stand-in for the detector's single subprocess seam."""

    def __init__(self, responses: list[tuple[Any, int, str]] | None = None) -> None:
        self.commands: list[list[str]] = []
        self._responses = responses or []

    def __call__(
        self, cmd: list[str], cwd: Path, timeout: float
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(list(cmd))
        for match, returncode, stdout in self._responses:
            if self._matches(cmd, match):
                return subprocess.CompletedProcess(cmd, returncode, stdout, "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    @staticmethod
    def _matches(cmd: list[str], match: Any) -> bool:
        if callable(match):
            return bool(match(cmd))
        prefix = list(match)
        return cmd[: len(prefix)] == prefix

    def issued(self, *prefix: str) -> list[list[str]]:
        return [c for c in self.commands if c[: len(prefix)] == list(prefix)]


def _base_responses(porcelain: str, pr_list: str = "[]") -> list[tuple[Any, int, str]]:
    return [
        (("git", "rev-parse"), 0, SHA + "\n"),
        (("git", "status", "--porcelain"), 0, porcelain),
        (("gh", "pr", "list"), 0, pr_list),
        (("gh", "pr", "create"), 0, "https://github.com/synaptent/aragora/pull/9999\n"),
    ]


def _run_main(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runner: FakeRunner,
    *args: str,
) -> tuple[int, dict[str, Any]]:
    monkeypatch.setattr(detector, "_run", runner)
    scratches: list[Path] = []

    def fake_mkdtemp(prefix: str = "tmp") -> str:
        scratch = tmp_path / f"{prefix}{len(scratches)}"
        scratch.mkdir()
        scratches.append(scratch)
        return str(scratch)

    monkeypatch.setattr(detector.tempfile, "mkdtemp", fake_mkdtemp)
    status_path = tmp_path / "status.json"
    code = detector.main(
        [
            "--repo-root",
            str(tmp_path),
            "--status-path",
            str(status_path),
            "--json",
            *args,
        ]
    )
    payload = json.loads(status_path.read_text()) if status_path.exists() else {}
    leaked = [str(s) for s in scratches if s.exists()]
    assert not leaked, f"scratch dirs leaked: {leaked}"
    return code, payload


def _assert_no_forbidden(runner: FakeRunner) -> None:
    for cmd in runner.commands:
        assert not FORBIDDEN_ARGS.intersection(cmd), f"forbidden arg in {cmd}"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_porcelain_handles_variants() -> None:
    text = (
        f" M {MIRROR}\x00"
        "?? docs-site/docs/new-file.md\x00"
        "R  docs/NEW.md\x00docs/OLD.md\x00"
        " M docs-site/docs/with space.md\x00"
        ' M docs-site/docs/"quoted" \u00fcnicode.md\x00'
        f" M {MIRROR}\x00"
        "x\x00"
    )
    paths = detector.parse_porcelain(text)
    assert paths == sorted(
        {
            MIRROR,
            "docs-site/docs/new-file.md",
            "docs/NEW.md",
            "docs-site/docs/with space.md",
            'docs-site/docs/"quoted" \u00fcnicode.md',
        }
    )
    assert "docs/OLD.md" not in paths


def test_partition_drift_splits_on_allowlist() -> None:
    mirrors, other = detector.partition_drift([MIRROR, "CLAUDE.md", "docs/METRICS.md"])
    assert mirrors == [MIRROR]
    assert other == ["CLAUDE.md", "docs/METRICS.md"]


def test_next_consecutive_errors_counts_faults_only() -> None:
    assert detector.next_consecutive_errors(2, detector.OUTCOME_ERROR) == 3
    assert detector.next_consecutive_errors(2, detector.OUTCOME_OUTSIDE_ALLOWLIST) == 3
    assert detector.next_consecutive_errors(2, detector.OUTCOME_CLEAN) == 0
    assert detector.next_consecutive_errors(2, detector.OUTCOME_DRIFT_PR_OPEN) == 0
    assert detector.next_consecutive_errors(2, detector.OUTCOME_DRIFT_DETECTED) == 0


def test_exit_codes_cover_every_outcome() -> None:
    outcomes = {
        detector.OUTCOME_CLEAN,
        detector.OUTCOME_DRIFT_DETECTED,
        detector.OUTCOME_DRIFT_PR_OPEN,
        detector.OUTCOME_DRIFT_PR_OPENED,
        detector.OUTCOME_OUTSIDE_ALLOWLIST,
        detector.OUTCOME_ERROR,
    }
    assert set(detector.EXIT_BY_OUTCOME) == outcomes
    assert detector.EXIT_BY_OUTCOME[detector.OUTCOME_DRIFT_DETECTED] == 1
    assert detector.EXIT_BY_OUTCOME[detector.OUTCOME_ERROR] == 2


def test_status_write_is_atomic_and_reloadable(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "status.json"
    detector.write_status_atomic(path, {"outcome": "clean", "consecutive_errors": 0})
    assert detector.load_previous_status(path)["outcome"] == "clean"
    assert [p.name for p in path.parent.iterdir()] == ["status.json"]


def test_load_previous_status_tolerates_garbage(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    path.write_text("not json")
    assert detector.load_previous_status(path) == {}
    assert detector.load_previous_status(tmp_path / "missing.json") == {}


# ---------------------------------------------------------------------------
# Iterations through the faked subprocess seam
# ---------------------------------------------------------------------------


def test_check_mode_reports_drift_without_mutating(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = FakeRunner(_base_responses(f" M {MIRROR}\x00"))
    code, payload = _run_main(monkeypatch, tmp_path, runner)
    assert code == 1
    assert payload["outcome"] == detector.OUTCOME_DRIFT_DETECTED
    assert payload["drifted_mirror_paths"] == [MIRROR]
    assert payload["base_sha"] == SHA
    for verb in (("git", "commit"), ("git", "push"), ("gh",)):
        assert runner.issued(*verb) == []
    _assert_no_forbidden(runner)


def test_check_mode_clean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runner = FakeRunner(_base_responses(""))
    code, payload = _run_main(monkeypatch, tmp_path, runner)
    assert code == 0
    assert payload["outcome"] == detector.OUTCOME_CLEAN
    assert payload["consecutive_errors"] == 0


def test_drift_outside_allowlist_fails_closed_even_with_apply(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = FakeRunner(_base_responses(f" M {MIRROR}\x00 M CLAUDE.md\x00"))
    code, payload = _run_main(monkeypatch, tmp_path, runner, "--apply")
    assert code == 2
    assert payload["outcome"] == detector.OUTCOME_OUTSIDE_ALLOWLIST
    assert payload["drifted_other_paths"] == ["CLAUDE.md"]
    assert "CLAUDE.md" in (payload["error"] or "")
    for verb in (("git", "commit"), ("git", "push"), ("gh",)):
        assert runner.issued(*verb) == []


def test_apply_is_idempotent_when_sync_pr_already_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pr_list = json.dumps(
        [
            {"url": "https://github.com/synaptent/aragora/pull/9001", "headRefName": "feature/x"},
            {
                "url": "https://github.com/synaptent/aragora/pull/9002",
                "headRefName": f"{detector.BRANCH_PREFIX}-20260601-000000",
            },
        ]
    )
    runner = FakeRunner(_base_responses(f" M {MIRROR}\x00", pr_list=pr_list))
    code, payload = _run_main(monkeypatch, tmp_path, runner, "--apply")
    assert code == 0
    assert payload["outcome"] == detector.OUTCOME_DRIFT_PR_OPEN
    assert payload["pr_url"].endswith("/9002")
    assert runner.issued("git", "push") == []
    assert runner.issued("gh", "pr", "create") == []


def test_apply_opens_single_pr_and_never_merges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = FakeRunner(_base_responses(f" M {MIRROR}\x00"))
    code, payload = _run_main(monkeypatch, tmp_path, runner, "--apply")
    assert code == 0
    assert payload["outcome"] == detector.OUTCOME_DRIFT_PR_OPENED
    assert payload["pr_url"] == "https://github.com/synaptent/aragora/pull/9999"
    pushes = runner.issued("git", "push")
    assert len(pushes) == 1
    assert pushes[0][:4] == ["git", "push", "-u", "origin"]
    assert pushes[0][4].startswith(detector.BRANCH_PREFIX)
    creates = runner.issued("gh", "pr", "create")
    assert len(creates) == 1
    adds = runner.issued("git", "add")
    assert adds and adds[0][3:] == [MIRROR]
    _assert_no_forbidden(runner)


def test_apply_pushes_to_remote_parsed_from_base_ref(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = FakeRunner(_base_responses(f" M {MIRROR}\x00"))
    code, payload = _run_main(
        monkeypatch, tmp_path, runner, "--apply", "--base-ref", "upstream/main"
    )
    assert code == 0
    assert payload["outcome"] == detector.OUTCOME_DRIFT_PR_OPENED
    fetches = runner.issued("git", "fetch")
    assert fetches and fetches[0][3:] == ["upstream", "main"]
    pushes = runner.issued("git", "push")
    assert len(pushes) == 1
    assert pushes[0][:4] == ["git", "push", "-u", "upstream"]
    _assert_no_forbidden(runner)


def test_regen_failure_is_fault_and_increments_streak(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status_path = tmp_path / "status.json"
    detector.write_status_atomic(status_path, {"outcome": "error", "consecutive_errors": 2})
    runner = FakeRunner([(("node",), 1, "boom")] + _base_responses(""))
    monkeypatch.setattr(detector, "_run", runner)
    code = detector.main(
        ["--repo-root", str(tmp_path), "--status-path", str(status_path), "--json"]
    )
    payload = json.loads(status_path.read_text())
    assert code == 2
    assert payload["outcome"] == detector.OUTCOME_ERROR
    assert "sync-docs" in payload["error"]
    assert payload["consecutive_errors"] == 3


def test_worktree_is_cleaned_up_even_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = FakeRunner([(("node",), 1, "boom")] + _base_responses(""))
    code, _payload = _run_main(monkeypatch, tmp_path, runner)
    assert code == 2
    assert runner.issued("git", "worktree", "remove") != []
    assert runner.issued("git", "worktree", "prune") != []


# ---------------------------------------------------------------------------
# Loop Control Plane wiring
# ---------------------------------------------------------------------------


def _write_status(root: Path, payload: dict[str, Any]) -> Path:
    path = root / ".aragora" / "docs_drift_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_spec_is_registered_with_honest_budget_gap() -> None:
    spec = LOOP_SPECS[LoopKind.DOCS_SYNC_DRIFT]
    readiness = audit_halt_readiness(spec.guards)
    assert readiness.verdict == HaltVerdict.INCOMPLETE.value
    assert readiness.gaps == ["no dollar/budget ceiling (bounded by time/iterations only)"]
    assert spec.durable_state_path == ".aragora/docs_drift_status.json"


def test_collector_missing_file_is_unavailable(tmp_path: Path) -> None:
    raw = io_mod.collect_docs_sync_drift(tmp_path)
    assert raw["source_status"] == "unavailable"
    record = classify_loop(LOOP_SPECS[LoopKind.DOCS_SYNC_DRIFT], raw)
    assert record.state == "unknown"
    assert record.next_action == "report_only"


def test_collector_waiting_on_open_pr_is_waiting_not_fault(tmp_path: Path) -> None:
    path = _write_status(
        tmp_path,
        {
            "outcome": "drift_pr_open",
            "consecutive_errors": 0,
            "generated_at": "2026-06-10T08:00:00Z",
        },
    )
    raw = io_mod.collect_docs_sync_drift(tmp_path, now=path.stat().st_mtime + 60.0)
    assert raw["waiting_only"] is True
    assert raw["operational_fault"] is False
    record = classify_loop(LOOP_SPECS[LoopKind.DOCS_SYNC_DRIFT], raw)
    assert record.state == "waiting"
    assert record.next_action == "wait"


def test_collector_fault_outcome_halts(tmp_path: Path) -> None:
    path = _write_status(
        tmp_path,
        {"outcome": "drift_outside_allowlist", "consecutive_errors": 1, "error": "CLAUDE.md"},
    )
    raw = io_mod.collect_docs_sync_drift(tmp_path, now=path.stat().st_mtime + 60.0)
    assert raw["operational_fault"] is True
    record = classify_loop(LOOP_SPECS[LoopKind.DOCS_SYNC_DRIFT], raw)
    assert record.state == "blocked"
    assert record.next_action == "halt"
    assert record.blocker == "CLAUDE.md"


def test_collector_stale_status_is_halted_report_only(tmp_path: Path) -> None:
    path = _write_status(tmp_path, {"outcome": "clean", "consecutive_errors": 0})
    stale_now = path.stat().st_mtime + io_mod._DOCS_DRIFT_STATE_FRESH_SECONDS + 1.0
    raw = io_mod.collect_docs_sync_drift(tmp_path, now=stale_now)
    assert raw["alive"] is False
    record = classify_loop(LOOP_SPECS[LoopKind.DOCS_SYNC_DRIFT], raw)
    assert record.state == "halted"
    assert record.next_action == "report_only"


def test_collector_fresh_clean_is_running(tmp_path: Path) -> None:
    path = _write_status(tmp_path, {"outcome": "clean", "consecutive_errors": 0})
    raw = io_mod.collect_docs_sync_drift(tmp_path, now=path.stat().st_mtime + 60.0)
    record = classify_loop(LOOP_SPECS[LoopKind.DOCS_SYNC_DRIFT], raw)
    assert record.state == "running"
    assert record.next_action == "continue"
    assert record.source_status == "ok"
