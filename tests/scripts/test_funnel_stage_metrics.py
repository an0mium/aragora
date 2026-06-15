"""Tests for ``scripts/funnel_stage_metrics.py`` (funnel telemetry snapshot).

All boundaries (the two ``gh pr list`` calls) are injected; no test touches
the network. Style mirrors ``tests/scripts/test_backlog_gate.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
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


metrics = _load_module("funnel_stage_metrics.py")

NOW = datetime(2026, 6, 12, 12, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _open_pr(
    number: int,
    *,
    head: str = "codex/feature",
    draft: bool = True,
    age_hours: float = 1.0,
    created: str | None = None,
) -> dict[str, Any]:
    created_at = created if created is not None else _iso(NOW - timedelta(hours=age_hours))
    return {
        "number": number,
        "headRefName": head,
        "isDraft": draft,
        "createdAt": created_at,
        "updatedAt": created_at,
        "labels": [],
    }


def _merged_pr(
    number: int, *, head: str = "codex/done", merged_hours_ago: float = 2.0
) -> dict[str, Any]:
    return {
        "number": number,
        "headRefName": head,
        "mergedAt": _iso(NOW - timedelta(hours=merged_hours_ago)),
        "createdAt": _iso(NOW - timedelta(hours=merged_hours_ago + 24)),
    }


def _run(
    tmp_path: Path,
    open_prs: list[dict[str, Any]] | Exception,
    merged_prs: list[dict[str, Any]] | Exception | None = None,
    *,
    outbox_dir: Path | None = None,
    stale_days: int = 4,
    max_draft_age_hours: float = 72.0,
    max_stale_tail: int = 20,
    out_file: str | None = None,
    branch_prefixes: tuple[str, ...] = ("codex/",),
) -> tuple[int, dict[str, Any] | None, list[datetime]]:
    """Run a snapshot with injected runners; return (exit, payload, merged since args)."""
    lines: list[str] = []
    since_calls: list[datetime] = []

    def list_open() -> list[dict[str, Any]]:
        if isinstance(open_prs, Exception):
            raise open_prs
        return open_prs

    def list_merged(since: datetime) -> list[dict[str, Any]]:
        since_calls.append(since)
        if isinstance(merged_prs, Exception):
            raise merged_prs
        return merged_prs or []

    exit_code = metrics.run_snapshot(
        list_open_prs=list_open,
        list_merged_prs=list_merged,
        branch_prefixes=branch_prefixes,
        outbox_dir=str(outbox_dir if outbox_dir is not None else tmp_path / "outbox"),
        stale_days=stale_days,
        max_draft_age_hours=max_draft_age_hours,
        max_stale_tail=max_stale_tail,
        out_file=out_file,
        now=NOW,
        log=lines.append,
    )
    payload = json.loads(lines[-1]) if lines else None
    return exit_code, payload, since_calls


# ---------------------------------------------------------------------------
# Percentile math
# ---------------------------------------------------------------------------


def test_percentile_empty_is_none() -> None:
    assert metrics.percentile([], 50) is None


def test_percentile_single_value() -> None:
    assert metrics.percentile([7.0], 50) == 7.0
    assert metrics.percentile([7.0], 90) == 7.0


def test_percentile_linear_interpolation() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert metrics.percentile(values, 50) == pytest.approx(25.0)
    assert metrics.percentile(values, 90) == pytest.approx(37.0)
    assert metrics.percentile(values, 0) == 10.0
    assert metrics.percentile(values, 100) == 40.0


def test_percentile_order_independent() -> None:
    assert metrics.percentile([40.0, 10.0, 30.0, 20.0], 50) == pytest.approx(25.0)


def test_stage_age_percentiles_in_payload(tmp_path: Path) -> None:
    drafts = [_open_pr(n, age_hours=h) for n, h in enumerate([10, 20, 30, 40], start=1)]
    ready = [_open_pr(50, draft=False, age_hours=5)]
    exit_code, payload, _ = _run(tmp_path, drafts + ready)
    assert exit_code == 0
    assert payload is not None
    assert payload["stages"]["draft"]["count"] == 4
    assert payload["stages"]["draft"]["age_hours"] == {"p50": 25.0, "p90": 37.0, "max": 40.0}
    assert payload["stages"]["ready"]["count"] == 1
    assert payload["stages"]["ready"]["age_hours"] == {"p50": 5.0, "p90": 5.0, "max": 5.0}


def test_empty_stage_has_null_age_stats(tmp_path: Path) -> None:
    _, payload, _ = _run(tmp_path, [])
    assert payload is not None
    assert payload["stages"]["draft"] == {
        "count": 0,
        "age_hours": {"p50": None, "p90": None, "max": None},
    }
    assert payload["stages"]["ready"]["count"] == 0


def test_unparseable_created_at_counted_but_excluded_from_ages(tmp_path: Path) -> None:
    prs = [_open_pr(1, created="not-a-date"), _open_pr(2, age_hours=8)]
    _, payload, _ = _run(tmp_path, prs)
    assert payload is not None
    assert payload["stages"]["draft"]["count"] == 2
    assert payload["stages"]["draft"]["age_hours"]["max"] == 8.0


# ---------------------------------------------------------------------------
# Branch prefix filtering
# ---------------------------------------------------------------------------


def test_non_automation_branches_excluded(tmp_path: Path) -> None:
    prs = [_open_pr(1), _open_pr(2, head="feature/manual"), _open_pr(3, head="main")]
    _, payload, _ = _run(tmp_path, prs)
    assert payload is not None
    assert payload["stages"]["draft"]["count"] == 1


def test_repeatable_branch_prefixes(tmp_path: Path) -> None:
    prs = [_open_pr(1), _open_pr(2, head="elves/run"), _open_pr(3, head="feature/x")]
    _, payload, _ = _run(tmp_path, prs, branch_prefixes=("codex/", "elves/"))
    assert payload is not None
    assert payload["stages"]["draft"]["count"] == 2


# ---------------------------------------------------------------------------
# merged_24h with graceful degradation
# ---------------------------------------------------------------------------


def test_merged_24h_counts_prefix_matches_within_window(tmp_path: Path) -> None:
    merged = [
        _merged_pr(1, merged_hours_ago=2),
        _merged_pr(2, merged_hours_ago=23),
        _merged_pr(3, head="feature/manual", merged_hours_ago=2),  # wrong prefix
        _merged_pr(4, merged_hours_ago=30),  # outside the 24h window
    ]
    exit_code, payload, since_calls = _run(tmp_path, [], merged)
    assert exit_code == 0
    assert payload is not None
    assert payload["merged_24h"] == 2
    assert since_calls == [NOW - timedelta(hours=24)]


def test_merged_search_failure_degrades_to_null_with_annotation(tmp_path: Path) -> None:
    exit_code, payload, _ = _run(
        tmp_path,
        [_open_pr(1)],
        RuntimeError("gh pr list (merged search) failed (exit 1): bad search"),
    )
    assert exit_code == 0, "a merged-search failure must not fail the snapshot"
    assert payload is not None
    assert payload["merged_24h"] is None
    assert any(a.startswith("merged_search_failed:") for a in payload["annotations"])


# ---------------------------------------------------------------------------
# Truncated listings: counts are floors, not totals
# ---------------------------------------------------------------------------


def test_open_list_truncation_annotated_and_breach(tmp_path: Path) -> None:
    # gh truncates BEFORE the prefix filter, so a limit-sized payload makes
    # every derived count an unreliable floor — that itself is a breach.
    prs = [_open_pr(n, head="other/branch") for n in range(metrics.OPEN_LIST_LIMIT - 1)]
    exit_code, payload, _ = _run(tmp_path, prs + [_open_pr(999)])
    assert exit_code == 3
    assert payload is not None
    assert "open_list_truncated" in payload["thresholds_breached"]
    assert f"list_truncated_open:>={metrics.OPEN_LIST_LIMIT}" in payload["annotations"]


def test_open_list_just_below_limit_is_unaffected(tmp_path: Path) -> None:
    prs = [_open_pr(n, head="other/branch") for n in range(metrics.OPEN_LIST_LIMIT - 2)]
    exit_code, payload, _ = _run(tmp_path, prs + [_open_pr(999)])
    assert exit_code == 0
    assert payload is not None
    assert payload["thresholds_breached"] == []
    assert not any("list_truncated_open" in a for a in payload["annotations"])


def test_merged_list_truncation_nulls_merged_24h_with_annotation(tmp_path: Path) -> None:
    # A truncated merged list would yield a misleading prefix-filtered
    # undercount; degrade to null like the failed-search path instead.
    merged = [_merged_pr(n) for n in range(metrics.MERGED_LIST_LIMIT)]
    exit_code, payload, _ = _run(tmp_path, [], merged)
    assert exit_code == 0, "merged-list truncation must not breach or fail the snapshot"
    assert payload is not None
    assert payload["merged_24h"] is None
    assert f"list_truncated_merged:>={metrics.MERGED_LIST_LIMIT}" in payload["annotations"]


def test_merged_list_just_below_limit_counts_normally(tmp_path: Path) -> None:
    merged = [_merged_pr(n) for n in range(metrics.MERGED_LIST_LIMIT - 1)]
    exit_code, payload, _ = _run(tmp_path, [], merged)
    assert exit_code == 0
    assert payload is not None
    assert payload["merged_24h"] == metrics.MERGED_LIST_LIMIT - 1
    assert not any("list_truncated_merged" in a for a in payload["annotations"])


# ---------------------------------------------------------------------------
# Outbox depth
# ---------------------------------------------------------------------------


def test_outbox_depth_counts_direct_json_files(tmp_path: Path) -> None:
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    (outbox / "a.json").write_text("{}", encoding="utf-8")
    (outbox / "b.json").write_text("{}", encoding="utf-8")
    (outbox / "c.txt").write_text("x", encoding="utf-8")
    (outbox / "nested").mkdir()
    (outbox / "nested" / "d.json").write_text("{}", encoding="utf-8")
    _, payload, _ = _run(tmp_path, [], outbox_dir=outbox)
    assert payload is not None
    assert payload["outbox_depth"] == 2


def test_missing_outbox_dir_is_zero_with_annotation(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    exit_code, payload, _ = _run(tmp_path, [], outbox_dir=missing)
    assert exit_code == 0
    assert payload is not None
    assert payload["outbox_depth"] == 0
    assert f"outbox_dir_missing:{missing}" in payload["annotations"]


# ---------------------------------------------------------------------------
# Stale tail + threshold breaches (exit 3)
# ---------------------------------------------------------------------------


def test_stale_tail_counts_open_prs_older_than_stale_days(tmp_path: Path) -> None:
    prs = [
        _open_pr(1, age_hours=24 * 5),  # 5 days: stale
        _open_pr(2, draft=False, age_hours=24 * 4 + 1),  # just over 4 days: stale
        _open_pr(3, age_hours=24 * 3),  # 3 days: not stale
    ]
    _, payload, _ = _run(tmp_path, prs, stale_days=4)
    assert payload is not None
    assert payload["stale_tail"] == 2


def test_draft_p90_breach_exits_three(tmp_path: Path) -> None:
    prs = [_open_pr(n, age_hours=100) for n in range(5)]
    exit_code, payload, _ = _run(tmp_path, prs, max_draft_age_hours=72.0, max_stale_tail=99)
    assert exit_code == 3
    assert payload is not None
    assert payload["thresholds_breached"] == ["draft_age_p90:100.0>max_draft_age_hours:72.0"]


def test_draft_p90_at_threshold_is_not_breach(tmp_path: Path) -> None:
    prs = [_open_pr(n, age_hours=72) for n in range(3)]
    exit_code, payload, _ = _run(tmp_path, prs, max_draft_age_hours=72.0)
    assert exit_code == 0
    assert payload is not None
    assert payload["thresholds_breached"] == []


def test_stale_tail_breach_exits_three(tmp_path: Path) -> None:
    prs = [_open_pr(n, age_hours=24 * 10) for n in range(3)]
    exit_code, payload, _ = _run(tmp_path, prs, max_stale_tail=2, max_draft_age_hours=10_000.0)
    assert exit_code == 3
    assert payload is not None
    assert payload["thresholds_breached"] == ["stale_tail:3>max_stale_tail:2"]


def test_ready_age_does_not_trigger_draft_threshold(tmp_path: Path) -> None:
    prs = [_open_pr(n, draft=False, age_hours=100) for n in range(3)]
    exit_code, payload, _ = _run(
        tmp_path, prs, max_draft_age_hours=72.0, max_stale_tail=99, stale_days=30
    )
    assert exit_code == 0
    assert payload is not None
    assert payload["thresholds_breached"] == []


# ---------------------------------------------------------------------------
# Failure path (exit 1) and --out atomic writes
# ---------------------------------------------------------------------------


def test_open_listing_failure_exits_one(tmp_path: Path) -> None:
    exit_code, payload, _ = _run(tmp_path, RuntimeError("gh pr list failed"))
    assert exit_code == 1
    assert payload is None


def test_out_file_written_atomically(tmp_path: Path) -> None:
    out = tmp_path / "snapshots" / "funnel.json"
    exit_code, payload, _ = _run(tmp_path, [_open_pr(1)], out_file=str(out))
    assert exit_code == 0
    assert payload is not None
    assert json.loads(out.read_text(encoding="utf-8")) == payload
    assert [p.name for p in out.parent.iterdir()] == ["funnel.json"]


def test_out_write_failure_leaves_no_partial_file_and_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "snapshots"
    out = out_dir / "funnel.json"

    def broken_replace(src: str, dst: str) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(metrics.os, "replace", broken_replace)
    exit_code, _, _ = _run(tmp_path, [_open_pr(1)], out_file=str(out))
    assert exit_code == 1
    assert not out.exists(), "no partial out file may be left behind"
    assert [p.name for p in out_dir.iterdir()] == [], "temp files must be cleaned up"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_normal_exit_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # main() uses the real clock, so timestamps must be relative to real now.
    real_now = datetime.now(timezone.utc)
    open_pr = dict(_open_pr(1), createdAt=_iso(real_now - timedelta(hours=1)))
    merged_pr = dict(_merged_pr(2), mergedAt=_iso(real_now - timedelta(hours=2)))
    monkeypatch.setattr(metrics, "default_list_open_prs", lambda repo: [open_pr])
    monkeypatch.setattr(metrics, "default_list_merged_prs", lambda repo, since: [merged_pr])
    out = tmp_path / "funnel.json"
    assert metrics.main(["--outbox-dir", str(tmp_path / "outbox"), "--out", str(out)]) == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["merged_24h"] == 1
    assert payload["stages"]["draft"]["count"] == 1


def test_main_breach_exit_three(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_now = datetime.now(timezone.utc)
    prs = [dict(_open_pr(n), createdAt=_iso(real_now - timedelta(hours=200))) for n in range(4)]
    monkeypatch.setattr(metrics, "default_list_open_prs", lambda repo: prs)
    monkeypatch.setattr(metrics, "default_list_merged_prs", lambda repo, since: [])
    assert (
        metrics.main(["--outbox-dir", str(tmp_path / "outbox"), "--max-draft-age-hours", "72"]) == 3
    )


def test_main_listing_failure_exit_one(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(repo: str) -> list[dict[str, Any]]:
        raise RuntimeError("gh exploded")

    monkeypatch.setattr(metrics, "default_list_open_prs", boom)
    assert metrics.main([]) == 1


def test_main_invokes_no_subprocess_with_mocked_runners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_run(command: list[str], **kwargs: Any) -> Any:
        raise AssertionError(f"subprocess must not run with mocked runners: {command}")

    monkeypatch.setattr(metrics, "default_list_open_prs", lambda repo: [])
    monkeypatch.setattr(metrics, "default_list_merged_prs", lambda repo, since: [])
    monkeypatch.setattr(metrics.subprocess, "run", forbidden_run)
    assert metrics.main(["--outbox-dir", str(tmp_path / "outbox")]) == 0


@pytest.mark.parametrize("repo", ["not-a-repo", "owner/name/extra", "owner/", "owner/na me"])
def test_malformed_repo_rejected_at_parse_time(repo: str) -> None:
    with pytest.raises(SystemExit) as excinfo:
        metrics.main(["--repo", repo])
    assert excinfo.value.code == 2, "argparse must reject before any gh call"


def test_well_formed_repo_accepted_by_validator() -> None:
    assert metrics.repo_arg("synaptent/aragora") == "synaptent/aragora"


def test_exit_code_constants_documented_contract() -> None:
    assert metrics.EXIT_OK == 0
    assert metrics.EXIT_FAILURE == 1
    assert metrics.EXIT_BREACH == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
