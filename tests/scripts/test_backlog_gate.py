"""Tests for ``scripts/backlog_gate.py`` (closure-first backpressure gate).

All boundaries (the single ``gh pr list`` call) are injected; no test touches
the network. Style mirrors ``tests/scripts/test_pr_ready_triage.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_module("backlog_gate.py")

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)


def _pr(
    number: int,
    *,
    head: str = "codex/feature",
    title: str = "",
    labels: list[str] | None = None,
    draft: bool = True,
) -> dict[str, Any]:
    return {
        "number": number,
        "headRefName": head,
        "title": title,
        "labels": [{"name": label} for label in (labels or [])],
        "isDraft": draft,
        "createdAt": "2026-06-11T12:00:00Z",
        "updatedAt": "2026-06-11T12:00:00Z",
    }


def _outbox(tmp_path: Path, json_files: int = 0, *, other_files: int = 0) -> Path:
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    for i in range(json_files):
        (outbox / f"item-{i}.json").write_text("{}", encoding="utf-8")
    for i in range(other_files):
        (outbox / f"noise-{i}.txt").write_text("x", encoding="utf-8")
    return outbox


def _run(
    tmp_path: Path,
    prs: list[dict[str, Any]] | Exception,
    *,
    outbox_dir: Path | None = None,
    max_open_prs: int = 60,
    max_outbox: int = 50,
    max_maintenance_ratio: float = 0.5,
    quiet: bool = False,
    branch_prefixes: tuple[str, ...] = ("codex/",),
) -> tuple[int, dict[str, Any] | None, list[str]]:
    """Run the gate with injected listing; return (exit, signal payload, stdout lines)."""
    signal_file = tmp_path / "signals" / "backpressure.json"
    lines: list[str] = []

    def list_prs() -> list[dict[str, Any]]:
        if isinstance(prs, Exception):
            raise prs
        return prs

    exit_code = gate.run_gate(
        list_prs=list_prs,
        branch_prefixes=branch_prefixes,
        outbox_dir=str(outbox_dir if outbox_dir is not None else tmp_path / "outbox"),
        max_open_prs=max_open_prs,
        max_outbox=max_outbox,
        max_maintenance_ratio=max_maintenance_ratio,
        signal_file=str(signal_file),
        quiet=quiet,
        now=NOW,
        log=lines.append,
    )
    payload = json.loads(signal_file.read_text(encoding="utf-8")) if signal_file.exists() else None
    return exit_code, payload, lines


# ---------------------------------------------------------------------------
# Decision + exit codes
# ---------------------------------------------------------------------------


def test_under_both_thresholds_generates_exit_zero(tmp_path: Path) -> None:
    outbox = _outbox(tmp_path, json_files=3)
    exit_code, payload, lines = _run(tmp_path, [_pr(1), _pr(2, draft=False)], outbox_dir=outbox)
    assert exit_code == 0
    assert payload is not None
    assert payload["mode"] == "generate"
    assert payload["reasons"] == []
    assert payload["open_prs"] == 2
    assert payload["drafts"] == 1
    assert payload["ready"] == 1
    assert payload["outbox_depth"] == 3
    assert payload["thresholds"] == {
        "max_open_prs": 60,
        "max_outbox": 50,
        "max_maintenance_ratio": 0.5,
    }
    assert payload["value_composition"]["total"] == 2
    assert payload["value_composition"]["maintenance_ratio"] == 0.0
    assert "admission" not in payload
    assert payload["generated_at"] == "2026-06-12T12:00:00Z"
    # stdout JSON matches the signal file
    assert json.loads(lines[-1]) == payload


def test_open_prs_at_threshold_is_shepherd_exit_three(tmp_path: Path) -> None:
    prs = [_pr(n) for n in range(5)]
    exit_code, payload, _ = _run(tmp_path, prs, max_open_prs=5)
    assert exit_code == 3
    assert payload is not None
    assert payload["mode"] == "shepherd"
    assert payload["reasons"] == ["open_prs:5>=max_open_prs:5"]
    assert payload["admission"] == {
        "withhold_classes": ["maintenance"],
        "source": "backlog_gate",
    }


def test_open_prs_just_under_threshold_generates(tmp_path: Path) -> None:
    prs = [_pr(n) for n in range(4)]
    exit_code, payload, _ = _run(tmp_path, prs, max_open_prs=5)
    assert exit_code == 0
    assert payload is not None
    assert payload["mode"] == "generate"


def test_outbox_at_threshold_is_shepherd(tmp_path: Path) -> None:
    outbox = _outbox(tmp_path, json_files=4)
    exit_code, payload, _ = _run(tmp_path, [_pr(1)], outbox_dir=outbox, max_outbox=4)
    assert exit_code == 3
    assert payload is not None
    assert payload["reasons"] == ["outbox_depth:4>=max_outbox:4"]


def test_outbox_just_under_threshold_generates(tmp_path: Path) -> None:
    outbox = _outbox(tmp_path, json_files=3)
    exit_code, payload, _ = _run(tmp_path, [_pr(1)], outbox_dir=outbox, max_outbox=4)
    assert exit_code == 0
    assert payload is not None
    assert payload["mode"] == "generate"


def test_both_over_threshold_lists_both_reasons(tmp_path: Path) -> None:
    outbox = _outbox(tmp_path, json_files=9)
    exit_code, payload, _ = _run(
        tmp_path, [_pr(n) for n in range(7)], outbox_dir=outbox, max_open_prs=2, max_outbox=2
    )
    assert exit_code == 3
    assert payload is not None
    assert payload["reasons"] == [
        "open_prs:7>=max_open_prs:2",
        "outbox_depth:9>=max_outbox:2",
    ]


def test_maintenance_ratio_breach_is_shepherd_with_admission(tmp_path: Path) -> None:
    prs = [
        _pr(1, title="drift repair", labels=["codex-automation"]),
        _pr(2, title="reconcile lane"),
        _pr(3, title="new ODR receipt endpoint"),
    ]
    exit_code, payload, _ = _run(tmp_path, prs, max_maintenance_ratio=0.5)
    assert exit_code == 3
    assert payload is not None
    assert payload["mode"] == "shepherd"
    assert "maintenance_ratio:0.6667>max_maintenance_ratio:0.5" in payload["reasons"]
    assert payload["value_composition"]["by_class"]["maintenance"] == 2
    assert payload["value_composition"]["by_class"]["product"] == 1
    assert payload["admission"] == {
        "withhold_classes": ["maintenance"],
        "source": "backlog_gate",
    }


def test_legacy_shepherd_shape_without_admission_is_not_emitted_for_generate(
    tmp_path: Path,
) -> None:
    exit_code, payload, _ = _run(tmp_path, [_pr(1, title="new ODR endpoint")])
    assert exit_code == 0
    assert payload is not None
    assert payload["mode"] == "generate"
    assert "admission" not in payload


# ---------------------------------------------------------------------------
# PR counting: prefixes, drafts/ready
# ---------------------------------------------------------------------------


def test_non_automation_branches_not_counted(tmp_path: Path) -> None:
    prs = [_pr(1), _pr(2, head="feature/manual"), _pr(3, head="main-fix")]
    exit_code, payload, _ = _run(tmp_path, prs, max_open_prs=2)
    assert exit_code == 0
    assert payload is not None
    assert payload["open_prs"] == 1


def test_repeatable_branch_prefixes(tmp_path: Path) -> None:
    prs = [_pr(1), _pr(2, head="elves/run-1"), _pr(3, head="feature/x")]
    _, payload, _ = _run(tmp_path, prs, branch_prefixes=("codex/", "elves/"))
    assert payload is not None
    assert payload["open_prs"] == 2


def test_draft_and_ready_split(tmp_path: Path) -> None:
    prs = [_pr(1, draft=True), _pr(2, draft=True), _pr(3, draft=False)]
    _, payload, _ = _run(tmp_path, prs)
    assert payload is not None
    assert payload["drafts"] == 2
    assert payload["ready"] == 1


def test_malformed_pr_entries_ignored(tmp_path: Path) -> None:
    prs: list[Any] = ["garbage", None, {"headRefName": None}, _pr(1)]
    _, payload, _ = _run(tmp_path, prs)
    assert payload is not None
    assert payload["open_prs"] == 1


# ---------------------------------------------------------------------------
# Outbox depth
# ---------------------------------------------------------------------------


def test_missing_outbox_dir_is_zero_with_annotation_not_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    exit_code, payload, _ = _run(tmp_path, [_pr(1)], outbox_dir=missing)
    assert exit_code == 0
    assert payload is not None
    assert payload["outbox_depth"] == 0
    assert payload["annotations"] == [f"outbox_dir_missing:{missing}"]


def test_only_direct_json_files_counted(tmp_path: Path) -> None:
    outbox = _outbox(tmp_path, json_files=2, other_files=3)
    nested = outbox / "nested"
    nested.mkdir()
    (nested / "deep.json").write_text("{}", encoding="utf-8")
    depth, missing = gate.count_outbox_depth(str(outbox))
    assert depth == 2
    assert missing is False


# ---------------------------------------------------------------------------
# Fail closed: gh failure writes a shepherd signal then exits 1
# ---------------------------------------------------------------------------


def test_gh_failure_writes_shepherd_signal_and_exits_one(tmp_path: Path) -> None:
    exit_code, payload, _ = _run(tmp_path, RuntimeError("gh pr list failed (exit 4): boom"))
    assert exit_code == 1
    assert payload is not None, "the fail-closed signal file must still be written"
    assert payload["mode"] == "shepherd"
    assert payload["reasons"] == ["gate_failure:gh pr list failed (exit 4): boom"]
    assert payload["open_prs"] is None
    assert payload["outbox_depth"] is None
    assert payload["value_composition"] is None
    assert "admission" not in payload


def test_gh_failure_with_unwritable_signal_still_exits_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    signal_file = tmp_path / "blocked" / "signal.json"
    (tmp_path / "blocked").write_text("a file, not a dir", encoding="utf-8")
    exit_code = gate.run_gate(
        list_prs=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        outbox_dir=str(tmp_path / "outbox"),
        signal_file=str(signal_file),
        now=NOW,
        log=lambda line: None,
    )
    assert exit_code == 1
    assert "signal_write_failed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Fail closed: truncated listing forces shepherd
# ---------------------------------------------------------------------------


def test_truncated_listing_forces_shepherd_even_with_small_counts(tmp_path: Path) -> None:
    # gh truncates BEFORE the prefix filter: a limit-sized payload with few
    # matching branches means in-scope PRs may have been silently dropped.
    prs = [_pr(n, head="other/branch") for n in range(gate.GH_LIST_LIMIT - 1)] + [_pr(999)]
    exit_code, payload, _ = _run(tmp_path, prs)
    assert exit_code == 3
    assert payload is not None
    assert payload["mode"] == "shepherd"
    assert payload["open_prs"] == 1, "visible in-scope count stays small"
    assert f"list_truncated:>={gate.GH_LIST_LIMIT}" in payload["reasons"]
    assert f"list_truncated:>={gate.GH_LIST_LIMIT}" in payload["annotations"]
    assert payload["admission"]["withhold_classes"] == ["maintenance"]


def test_listing_just_below_limit_is_unaffected(tmp_path: Path) -> None:
    prs = [_pr(n, head="other/branch") for n in range(gate.GH_LIST_LIMIT - 2)] + [_pr(999)]
    exit_code, payload, _ = _run(tmp_path, prs)
    assert exit_code == 0
    assert payload is not None
    assert payload["mode"] == "generate"
    assert not any("list_truncated" in r for r in payload["reasons"])
    assert not any("list_truncated" in a for a in payload["annotations"])


# ---------------------------------------------------------------------------
# Atomic signal writes
# ---------------------------------------------------------------------------


def test_atomic_write_no_partial_file_on_simulated_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outbox = _outbox(tmp_path, json_files=1)
    signal_dir = tmp_path / "signals"
    signal_file = signal_dir / "backpressure.json"

    def broken_replace(src: str, dst: str) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(gate.os, "replace", broken_replace)
    exit_code = gate.run_gate(
        list_prs=lambda: [_pr(1)],
        outbox_dir=str(outbox),
        signal_file=str(signal_file),
        now=NOW,
        log=lambda line: None,
    )
    assert exit_code == 1, "an unwritable signal file must never exit 0"
    assert not signal_file.exists(), "no partial signal file may be left behind"
    leftovers = [p.name for p in signal_dir.iterdir()]
    assert leftovers == [], f"temp files must be cleaned up: {leftovers}"


def test_atomic_write_replaces_existing_signal(tmp_path: Path) -> None:
    target = tmp_path / "sig.json"
    target.write_text('{"mode": "stale"}', encoding="utf-8")
    gate.atomic_write_json(str(target), {"mode": "generate"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"mode": "generate"}
    assert [p.name for p in tmp_path.iterdir()] == ["sig.json"]


# ---------------------------------------------------------------------------
# --quiet and main()
# ---------------------------------------------------------------------------


def test_quiet_suppresses_stdout_but_writes_signal(tmp_path: Path) -> None:
    exit_code, payload, lines = _run(tmp_path, [_pr(1)], quiet=True)
    assert exit_code == 0
    assert lines == []
    assert payload is not None
    assert payload["mode"] == "generate"


def test_quiet_suppresses_stdout_on_failure_too(tmp_path: Path) -> None:
    exit_code, payload, lines = _run(tmp_path, RuntimeError("boom"), quiet=True)
    assert exit_code == 1
    assert lines == []
    assert payload is not None
    assert payload["mode"] == "shepherd"


def test_main_generate_exit_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "default_list_open_prs", lambda repo: [_pr(1)])
    signal_file = tmp_path / "sig.json"
    assert (
        gate.main(
            [
                "--outbox-dir",
                str(tmp_path / "outbox"),
                "--signal-file",
                str(signal_file),
                "--quiet",
            ]
        )
        == 0
    )
    assert json.loads(signal_file.read_text(encoding="utf-8"))["mode"] == "generate"


def test_main_shepherd_exit_three(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "default_list_open_prs", lambda repo: [_pr(n) for n in range(3)])
    assert (
        gate.main(
            [
                "--max-open-prs",
                "3",
                "--outbox-dir",
                str(tmp_path / "outbox"),
                "--signal-file",
                str(tmp_path / "sig.json"),
                "--quiet",
            ]
        )
        == 3
    )


def test_main_gh_failure_exit_one_with_shepherd_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(repo: str) -> list[dict[str, Any]]:
        raise RuntimeError("gh exploded")

    monkeypatch.setattr(gate, "default_list_open_prs", boom)
    signal_file = tmp_path / "sig.json"
    assert (
        gate.main(
            [
                "--outbox-dir",
                str(tmp_path / "outbox"),
                "--signal-file",
                str(signal_file),
                "--quiet",
            ]
        )
        == 1
    )
    payload = json.loads(signal_file.read_text(encoding="utf-8"))
    assert payload["mode"] == "shepherd"
    assert payload["reasons"][0].startswith("gate_failure:")


def test_main_dry_by_default_invokes_no_subprocess_with_mocked_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_run(command: list[str], **kwargs: Any) -> Any:
        raise AssertionError(f"subprocess must not run with mocked listing: {command}")

    monkeypatch.setattr(gate, "default_list_open_prs", lambda repo: [])
    monkeypatch.setattr(gate.subprocess, "run", forbidden_run)
    assert (
        gate.main(
            [
                "--outbox-dir",
                str(tmp_path / "outbox"),
                "--signal-file",
                str(tmp_path / "sig.json"),
                "--quiet",
            ]
        )
        == 0
    )


@pytest.mark.parametrize("repo", ["not-a-repo", "owner/name/extra", "owner/", "owner/na me"])
def test_malformed_repo_rejected_at_parse_time(repo: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        gate.main(["--repo", repo, "--quiet"])
    assert excinfo.value.code == 2, "argparse must reject before any gh call"


def test_well_formed_repo_accepted_by_validator() -> None:
    assert gate.repo_arg("synaptent/aragora") == "synaptent/aragora"


def test_exit_code_constants_documented_contract() -> None:
    assert gate.EXIT_GENERATE == 0
    assert gate.EXIT_FAILURE == 1
    assert gate.EXIT_SHEPHERD == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
