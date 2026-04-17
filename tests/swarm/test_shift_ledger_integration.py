"""Integration-style shift ledger truth-surface tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aragora.swarm.shift_ledger import ShiftLedger


@pytest.fixture()
def ledger(tmp_path: Path) -> ShiftLedger:
    return ShiftLedger(path=tmp_path / "shift_ledger.jsonl")


def _record_at(
    monkeypatch: pytest.MonkeyPatch,
    ledger: ShiftLedger,
    timestamp: str,
    method_name: str,
    **payload: Any,
) -> None:
    monkeypatch.setattr(ledger, "_now_iso", lambda: timestamp)
    getattr(ledger, method_name)(**payload)


def test_get_status_summary_tracks_active_shift_recovery_story(
    ledger: ShiftLedger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _record_at(
        monkeypatch,
        ledger,
        "2999-01-01T00:00:00Z",
        "record_shift_start",
        shift_id="shift-1",
        max_hours=12.0,
        benchmark_mode="hybrid",
        queue_size=4,
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-01-01T00:30:00Z",
        "record_cycle_tick",
        queue_size=4,
        queue_removed=0,
        open_prs=3,
        boss_running=False,
        merge_running=True,
        benchmark_fresh=False,
        actions=["scan"],
        stop_reason="",
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-01-01T00:31:00Z",
        "record_service_restart",
        service="boss_loop",
        success=False,
        detail="launchctl kickstart timed out",
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-01-01T00:31:05Z",
        "record_failure",
        failure_type="service_failure",
        detail="BossRestartFailed: launchctl kickstart timed out",
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-01-01T01:00:00Z",
        "record_cycle_tick",
        queue_size=2,
        queue_removed=2,
        open_prs=2,
        boss_running=True,
        merge_running=True,
        benchmark_fresh=False,
        actions=["restart_boss_loop", "trim_queue"],
        stop_reason="",
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-01-01T01:35:00Z",
        "record_failure",
        failure_type="rate_limit",
        detail="HTTP 429: secondary rate limit exceeded",
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-01-01T01:35:30Z",
        "record_benchmark_run",
        run_id=24566214260,
        conclusion="failure",
        created_at="2999-01-01T01:35:00Z",
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-01-01T02:00:00Z",
        "record_cycle_tick",
        queue_size=1,
        queue_removed=0,
        open_prs=1,
        boss_running=True,
        merge_running=True,
        benchmark_fresh=True,
        actions=["trigger_benchmark:retry_after_failed_run"],
        stop_reason="",
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-01-01T02:05:00Z",
        "record_benchmark_run",
        run_id=24566214261,
        conclusion="success",
        created_at="2999-01-01T02:05:00Z",
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-01-01T02:10:00Z",
        "record_pr_merged",
        pr_number=6120,
        title="fix(proof-first): add shift status",
    )

    summary = ledger.get_status_summary(max_age_hours=48.0)

    assert summary["total_entries"] == 10
    assert summary["cycle_ticks"] == 3
    assert summary["benchmark_runs"] == 2
    assert summary["last_benchmark_conclusion"] == "success"
    assert summary["service_restarts"] == 1
    assert summary["restart_failures"] == 1
    assert summary["service_failures"] == 1
    assert summary["rate_limit_failures"] == 1
    assert summary["prs_merged"] == 1
    assert summary["pr_numbers_merged"] == [6120]
    assert summary["current_queue_size"] == 1
    assert summary["current_queue_removed"] == 0
    assert summary["current_open_prs"] == 1
    assert summary["current_boss_running"] is True
    assert summary["current_merge_running"] is True
    assert summary["current_benchmark_fresh"] is True
    assert summary["failure_policy"]["service_failure"]["count"] == 1
    assert summary["failure_policy"]["service_failure"]["will_stop"] is True
    assert summary["failure_policy"]["rate_limit"]["count"] == 1
    assert summary["failure_policy"]["rate_limit"]["will_stop"] is False
    assert summary["green_shift"]["queue_removed"] == 2
    assert summary["green_shift"]["queue_disciplined"] is False
    assert summary["green_shift"]["no_repeated_failures"] is False
    assert summary["green_shift"]["is_green"] is False


def test_get_status_summary_resets_failure_policy_for_new_shift(
    ledger: ShiftLedger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _record_at(
        monkeypatch,
        ledger,
        "2999-02-01T00:00:00Z",
        "record_shift_start",
        shift_id="shift-1",
        max_hours=12.0,
        benchmark_mode="hybrid",
        queue_size=1,
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-02-01T01:00:00Z",
        "record_failure",
        failure_type="auth_failure",
        detail="codex login required",
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-02-01T01:15:00Z",
        "record_failure",
        failure_type="auth_failure",
        detail="codex login required",
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-02-01T01:30:00Z",
        "record_cycle_tick",
        queue_size=1,
        queue_removed=0,
        open_prs=0,
        boss_running=False,
        merge_running=True,
        benchmark_fresh=False,
        actions=["retry_auth"],
        stop_reason="CodexAuthMissing: codex login required",
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-02-01T01:31:00Z",
        "record_shift_stop",
        shift_id="shift-1",
        reason="CodexAuthMissing: codex login required",
        cycles=2,
        duration_seconds=5460.0,
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-02-02T00:00:00Z",
        "record_shift_start",
        shift_id="shift-2",
        max_hours=12.0,
        benchmark_mode="hybrid",
        queue_size=0,
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-02-02T00:30:00Z",
        "record_cycle_tick",
        queue_size=0,
        queue_removed=0,
        open_prs=0,
        boss_running=False,
        merge_running=False,
        benchmark_fresh=True,
        actions=["steady_state"],
        stop_reason="",
    )
    _record_at(
        monkeypatch,
        ledger,
        "2999-02-02T00:35:00Z",
        "record_benchmark_run",
        run_id=24566214262,
        conclusion="success",
        created_at="2999-02-02T00:35:00Z",
    )

    summary = ledger.get_status_summary(max_age_hours=48.0)

    assert summary["shifts_started"] == 2
    assert summary["shifts_stopped"] == 1
    assert summary["auth_failures"] == 2
    assert summary["benchmark_runs"] == 1
    assert summary["last_benchmark_conclusion"] == "success"
    assert summary["current_queue_size"] == 0
    assert summary["current_open_prs"] == 0
    assert summary["current_boss_running"] is False
    assert summary["current_merge_running"] is False
    assert summary["current_benchmark_fresh"] is True
    assert summary["failure_policy"]["auth_failure"]["count"] == 0
    assert summary["failure_policy"]["auth_failure"]["will_stop"] is False
    assert summary["green_shift"]["last_stop_reason"] == ""
    assert summary["green_shift"]["queue_disciplined"] is True
    assert summary["green_shift"]["boss_service_healthy"] is True
    assert summary["green_shift"]["merge_service_healthy"] is True
    assert summary["green_shift"]["no_repeated_failures"] is True
    assert summary["green_shift"]["window_complete"] is False
    assert summary["green_shift"]["is_green"] is False
