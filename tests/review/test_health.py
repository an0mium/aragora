"""Tests for aragora.review.health — operational health surfaces."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from pathlib import Path

import pytest

from aragora.cli.commands.review_queue import _cmd_health
from aragora.review.health import (
    HealthReport,
    STATUS_AGING,
    STATUS_EMPTY,
    STATUS_FRESH,
    STATUS_MISSING,
    STATUS_STALE,
    SurfaceCheck,
    gather_health,
    render_text,
)

UTC = timezone.utc


def _touch_with_age(path: Path, hours_ago: float) -> None:
    """Create a file and set its mtime to N hours in the past."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")
    target = time.time() - hours_ago * 3600
    os.utime(path, (target, target))


def _write_status_doc(path: Path, last_updated: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# A status doc\n\nLast updated: {last_updated}\n\nsome content\n")


def _setup_proof_loop(tmp_path: Path) -> dict[str, Path]:
    """Create the canonical directory layout the health command inspects."""
    repo = tmp_path / "repo"
    repo.mkdir()
    rq = repo / ".aragora" / "review-queue"
    receipts = rq / "receipts"
    briefs = rq / "briefs"
    overnight = repo / ".aragora" / "overnight"
    auto = repo / ".aragora" / "automation-receipts"
    docs_status = repo / "docs" / "status"
    for d in (receipts, briefs, overnight, auto, docs_status):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "repo": repo,
        "review_queue_root": rq,
        "receipts": receipts,
        "briefs": briefs,
        "overnight": overnight,
        "auto": auto,
        "docs_status": docs_status,
    }


class TestSurfaceCheckSerialization:
    def test_to_dict_minimal(self) -> None:
        check = SurfaceCheck(name="x", status=STATUS_FRESH)
        d = check.to_dict()
        assert d["name"] == "x"
        assert d["status"] == STATUS_FRESH
        assert d["latest_mtime"] is None
        assert d["age_hours"] is None
        assert d["count"] is None

    def test_to_dict_full(self) -> None:
        mtime = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)
        check = SurfaceCheck(
            name="x",
            status=STATUS_AGING,
            count=5,
            latest_mtime=mtime,
            age_hours=25.5,
            path="/some/path",
            detail="newest: foo",
            extra={"row_count_total": 100},
        )
        d = check.to_dict()
        assert d["count"] == 5
        assert d["age_hours"] == 25.5
        assert d["latest_mtime"] == mtime.isoformat()
        assert d["path"] == "/some/path"
        assert d["extra"]["row_count_total"] == 100


class TestGatherHealthEmpty:
    """Empty repo: nearly everything missing or empty, overall is missing or empty."""

    def test_all_missing_in_empty_tmp_path(self, tmp_path: Path) -> None:
        # No .aragora structure at all.
        report = gather_health(
            repo_root=tmp_path,
            review_queue_root=tmp_path / ".aragora" / "review-queue",
            overnight_root=tmp_path / ".aragora" / "overnight",
            automation_receipts_root=tmp_path / ".aragora" / "automation-receipts",
        )
        assert isinstance(report, HealthReport)
        assert report.overall_status == STATUS_MISSING
        names = {s.name for s in report.surfaces}
        assert {
            "settlement_receipts",
            "briefs",
            "boss_metrics",
            "automation_receipts",
            "boss_loop_log",
            "watchdog_log",
            "b0_publication",
            "tw03_rescue",
        }.issubset(names)
        # All surfaces should be MISSING (no files at all).
        for surface in report.surfaces:
            assert surface.status == STATUS_MISSING, f"{surface.name} -> {surface.status}"

    def test_empty_dirs_distinguish_from_missing(self, tmp_path: Path) -> None:
        layout = _setup_proof_loop(tmp_path)
        report = gather_health(
            repo_root=layout["repo"],
            review_queue_root=layout["review_queue_root"],
            overnight_root=layout["overnight"],
            automation_receipts_root=layout["auto"],
        )
        # Directories exist but are empty.
        by_name = {s.name: s for s in report.surfaces}
        assert by_name["settlement_receipts"].status == STATUS_EMPTY
        # Briefs default to "fresh" when empty since they are founder-dogfood.
        assert by_name["briefs"].status == STATUS_FRESH
        # automation_receipts also expect_nonempty=False
        assert by_name["automation_receipts"].status == STATUS_FRESH
        # Files still missing.
        assert by_name["boss_metrics"].status == STATUS_MISSING
        assert by_name["b0_publication"].status == STATUS_MISSING

    def test_empty_expected_receipts_are_unhealthy(self, tmp_path: Path) -> None:
        layout = _setup_proof_loop(tmp_path)
        _touch_with_age(layout["overnight"] / "boss_metrics.jsonl", 1)
        _touch_with_age(layout["overnight"] / "boss-loop-launchd.log", 1)
        _touch_with_age(layout["overnight"] / "watchdog.log", 1)
        _touch_with_age(layout["auto"] / "x.json", 1)
        today = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_status_doc(layout["docs_status"] / "B0_BENCHMARK_TRUTH_STATUS.md", today)
        _write_status_doc(layout["docs_status"] / "TW03_RESCUE_PRODUCTIZATION_STATUS.md", today)

        report = gather_health(
            repo_root=layout["repo"],
            review_queue_root=layout["review_queue_root"],
            overnight_root=layout["overnight"],
            automation_receipts_root=layout["auto"],
        )

        by_name = {s.name: s for s in report.surfaces}
        assert by_name["settlement_receipts"].status == STATUS_EMPTY
        assert report.overall_status == STATUS_EMPTY

    def test_cli_exits_nonzero_on_empty_expected_receipts(self, tmp_path: Path) -> None:
        layout = _setup_proof_loop(tmp_path)
        _touch_with_age(layout["overnight"] / "boss_metrics.jsonl", 1)
        _touch_with_age(layout["overnight"] / "boss-loop-launchd.log", 1)
        _touch_with_age(layout["overnight"] / "watchdog.log", 1)
        _touch_with_age(layout["auto"] / "x.json", 1)
        today = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_status_doc(layout["docs_status"] / "B0_BENCHMARK_TRUTH_STATUS.md", today)
        _write_status_doc(layout["docs_status"] / "TW03_RESCUE_PRODUCTIZATION_STATUS.md", today)

        rc = _cmd_health(
            SimpleNamespace(
                repo_root=str(layout["repo"]),
                review_queue_root=str(layout["review_queue_root"]),
                overnight_root=str(layout["overnight"]),
                automation_receipts_root=str(layout["auto"]),
                json=True,
                json_output=True,
            )
        )

        assert rc == 1


class TestGatherHealthFresh:
    def test_all_fresh(self, tmp_path: Path) -> None:
        layout = _setup_proof_loop(tmp_path)
        # Settlement receipt 1h old
        _touch_with_age(layout["receipts"] / "pr-1-recorded-1-abc-admin_squash_merge.json", 1)
        # Boss metrics 2h old
        _touch_with_age(layout["overnight"] / "boss_metrics.jsonl", 2)
        # Automation receipt 0.5h old
        _touch_with_age(layout["auto"] / "x.json", 0.5)
        # Boss-loop log 1h old
        _touch_with_age(layout["overnight"] / "boss-loop-launchd.log", 1)
        # Watchdog log 5h old
        _touch_with_age(layout["overnight"] / "watchdog.log", 5)
        # Status docs fresh (today)
        today = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_status_doc(layout["docs_status"] / "B0_BENCHMARK_TRUTH_STATUS.md", today)
        _write_status_doc(layout["docs_status"] / "TW03_RESCUE_PRODUCTIZATION_STATUS.md", today)

        report = gather_health(
            repo_root=layout["repo"],
            review_queue_root=layout["review_queue_root"],
            overnight_root=layout["overnight"],
            automation_receipts_root=layout["auto"],
        )
        assert report.overall_status == STATUS_FRESH
        by_name = {s.name: s for s in report.surfaces}
        assert by_name["settlement_receipts"].count == 1
        assert by_name["settlement_receipts"].status == STATUS_FRESH
        assert by_name["boss_metrics"].status == STATUS_FRESH
        assert by_name["b0_publication"].status == STATUS_FRESH

    def test_worktree_uses_shared_aragora_state_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shared = _setup_proof_loop(tmp_path)
        worktree = tmp_path / "worktree"
        docs_status = worktree / "docs" / "status"
        docs_status.mkdir(parents=True)

        _touch_with_age(shared["receipts"] / "pr-1-recorded-1-abc-admin_squash_merge.json", 1)
        _touch_with_age(shared["overnight"] / "boss_metrics.jsonl", 1)
        _touch_with_age(shared["overnight"] / "boss-loop-launchd.log", 1)
        _touch_with_age(shared["overnight"] / "watchdog.log", 1)
        _touch_with_age(shared["auto"] / "x.json", 1)
        today = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        _write_status_doc(docs_status / "B0_BENCHMARK_TRUTH_STATUS.md", today)
        _write_status_doc(docs_status / "TW03_RESCUE_PRODUCTIZATION_STATUS.md", today)
        monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(shared["repo"] / ".aragora"))

        report = gather_health(repo_root=worktree)

        by_name = {s.name: s for s in report.surfaces}
        assert report.overall_status == STATUS_FRESH
        assert by_name["settlement_receipts"].path == str(shared["receipts"])
        assert by_name["boss_metrics"].path == str(shared["overnight"] / "boss_metrics.jsonl")
        assert by_name["automation_receipts"].path == str(shared["auto"])


class TestGatherHealthStaleness:
    def test_boss_metrics_stale_after_critical_window(self, tmp_path: Path) -> None:
        layout = _setup_proof_loop(tmp_path)
        # 20 days old: definitely past crit (48h * 4 = 192h).
        _touch_with_age(layout["overnight"] / "boss_metrics.jsonl", 20 * 24)
        report = gather_health(
            repo_root=layout["repo"],
            review_queue_root=layout["review_queue_root"],
            overnight_root=layout["overnight"],
            automation_receipts_root=layout["auto"],
        )
        by_name = {s.name: s for s in report.surfaces}
        assert by_name["boss_metrics"].status == STATUS_STALE
        # Overall must reflect the worst surface among present ones; with
        # B0 and TW03 still missing, MISSING wins (worse than STALE).
        assert report.overall_status == STATUS_MISSING

    def test_b0_publication_aging(self, tmp_path: Path) -> None:
        layout = _setup_proof_loop(tmp_path)
        # 10 days ago -> aging (warn=168h=7d, crit=168*4=28d)
        ten_days_ago = (datetime.now(UTC) - timedelta(days=10)).strftime("%Y-%m-%d")
        _write_status_doc(layout["docs_status"] / "B0_BENCHMARK_TRUTH_STATUS.md", ten_days_ago)
        report = gather_health(
            repo_root=layout["repo"],
            review_queue_root=layout["review_queue_root"],
            overnight_root=layout["overnight"],
            automation_receipts_root=layout["auto"],
        )
        by_name = {s.name: s for s in report.surfaces}
        assert by_name["b0_publication"].status == STATUS_AGING


class TestBossLoopLogCounter:
    def test_counts_crashes_and_exits(self, tmp_path: Path) -> None:
        layout = _setup_proof_loop(tmp_path)
        log = layout["overnight"] / "boss-loop-launchd.log"
        log.write_text(
            "Boss loop exited with status 0\n"
            "Boss loop exited with status 0\n"
            "ModuleNotFoundError: No module named 'httpx'\n"
            "ModuleNotFoundError: No module named 'defusedxml'\n"
            "Boss loop exited with status 1\n"
        )
        # Set mtime fresh.
        os.utime(log, (time.time() - 600, time.time() - 600))
        report = gather_health(
            repo_root=layout["repo"],
            review_queue_root=layout["review_queue_root"],
            overnight_root=layout["overnight"],
            automation_receipts_root=layout["auto"],
        )
        by_name = {s.name: s for s in report.surfaces}
        bl = by_name["boss_loop_log"]
        assert bl.status == STATUS_FRESH
        assert bl.extra["crashes_total"] == 2
        assert bl.extra["exits_ok_total"] == 2
        assert bl.extra["exits_fail_total"] == 1


class TestBossMetricsJsonlCounter:
    def test_counts_jsonl_rows(self, tmp_path: Path) -> None:
        layout = _setup_proof_loop(tmp_path)
        metrics = layout["overnight"] / "boss_metrics.jsonl"
        now_ts = time.time()
        rows = [
            {"recorded_at_ts": now_ts - 60, "worker_outcome": "completed"},
            {"recorded_at_ts": now_ts - 3 * 24 * 3600, "worker_outcome": "failed"},
            {"recorded_at_ts": now_ts - 14 * 24 * 3600, "worker_outcome": "completed"},
        ]
        metrics.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        os.utime(metrics, (now_ts - 600, now_ts - 600))
        report = gather_health(
            repo_root=layout["repo"],
            review_queue_root=layout["review_queue_root"],
            overnight_root=layout["overnight"],
            automation_receipts_root=layout["auto"],
        )
        by_name = {s.name: s for s in report.surfaces}
        bm = by_name["boss_metrics"]
        assert bm.extra["row_count_total"] == 3
        # Only first two rows in last 7 days.
        assert bm.extra["row_count_7d"] == 2


class TestRenderText:
    def test_render_text_includes_all_surfaces(self, tmp_path: Path) -> None:
        layout = _setup_proof_loop(tmp_path)
        report = gather_health(
            repo_root=layout["repo"],
            review_queue_root=layout["review_queue_root"],
            overnight_root=layout["overnight"],
            automation_receipts_root=layout["auto"],
        )
        text = render_text(report)
        for name in (
            "settlement_receipts",
            "briefs",
            "boss_metrics",
            "automation_receipts",
            "boss_loop_log",
            "watchdog_log",
            "b0_publication",
            "tw03_rescue",
        ):
            assert name in text
        assert "overall_status" in text


class TestStatusDocParsing:
    def test_parses_iso_with_z(self, tmp_path: Path) -> None:
        layout = _setup_proof_loop(tmp_path)
        _write_status_doc(
            layout["docs_status"] / "B0_BENCHMARK_TRUTH_STATUS.md",
            "2026-05-14T17:00:00Z",
        )
        report = gather_health(
            repo_root=layout["repo"],
            review_queue_root=layout["review_queue_root"],
            overnight_root=layout["overnight"],
            automation_receipts_root=layout["auto"],
        )
        by_name = {s.name: s for s in report.surfaces}
        b0 = by_name["b0_publication"]
        assert b0.latest_mtime is not None
        assert b0.latest_mtime.year == 2026
        assert b0.latest_mtime.tzinfo is not None

    def test_parses_bare_date(self, tmp_path: Path) -> None:
        layout = _setup_proof_loop(tmp_path)
        _write_status_doc(
            layout["docs_status"] / "B0_BENCHMARK_TRUTH_STATUS.md",
            "2026-05-14",
        )
        report = gather_health(
            repo_root=layout["repo"],
            review_queue_root=layout["review_queue_root"],
            overnight_root=layout["overnight"],
            automation_receipts_root=layout["auto"],
        )
        by_name = {s.name: s for s in report.surfaces}
        b0 = by_name["b0_publication"]
        assert b0.latest_mtime is not None
        assert b0.latest_mtime.year == 2026

    def test_unparseable_returns_aging(self, tmp_path: Path) -> None:
        layout = _setup_proof_loop(tmp_path)
        path = layout["docs_status"] / "B0_BENCHMARK_TRUTH_STATUS.md"
        path.write_text("# B0\n\nLast updated: lol-not-a-date\n")
        report = gather_health(
            repo_root=layout["repo"],
            review_queue_root=layout["review_queue_root"],
            overnight_root=layout["overnight"],
            automation_receipts_root=layout["auto"],
        )
        by_name = {s.name: s for s in report.surfaces}
        b0 = by_name["b0_publication"]
        assert b0.status == STATUS_AGING


class TestOverallSeverityAggregation:
    def test_overall_picks_worst_surface(self, tmp_path: Path) -> None:
        layout = _setup_proof_loop(tmp_path)
        # Make some fresh.
        _touch_with_age(layout["receipts"] / "pr-1-x.json", 1)
        _touch_with_age(layout["overnight"] / "boss_metrics.jsonl", 1)
        _touch_with_age(layout["overnight"] / "boss-loop-launchd.log", 1)
        _touch_with_age(layout["overnight"] / "watchdog.log", 1)
        _touch_with_age(layout["auto"] / "x.json", 1)
        # Status doc 60d old -> STALE.
        _write_status_doc(
            layout["docs_status"] / "B0_BENCHMARK_TRUTH_STATUS.md",
            (datetime.now(UTC) - timedelta(days=60)).strftime("%Y-%m-%d"),
        )
        # TW03 missing -> still missing.
        report = gather_health(
            repo_root=layout["repo"],
            review_queue_root=layout["review_queue_root"],
            overnight_root=layout["overnight"],
            automation_receipts_root=layout["auto"],
        )
        # TW03 missing should dominate.
        assert report.overall_status == STATUS_MISSING


class TestNoExceptionsOnHostileInputs:
    def test_corrupt_jsonl_doesnt_crash(self, tmp_path: Path) -> None:
        layout = _setup_proof_loop(tmp_path)
        metrics = layout["overnight"] / "boss_metrics.jsonl"
        metrics.write_text("not even close to json\nstill not json\n{maybe?\n")
        os.utime(metrics, (time.time() - 600, time.time() - 600))
        # Must not raise.
        report = gather_health(
            repo_root=layout["repo"],
            review_queue_root=layout["review_queue_root"],
            overnight_root=layout["overnight"],
            automation_receipts_root=layout["auto"],
        )
        by_name = {s.name: s for s in report.surfaces}
        bm = by_name["boss_metrics"]
        # The mtime check still works; the row counter just sees 3 lines with 0 dated.
        assert bm.latest_mtime is not None
        assert bm.extra["row_count_total"] == 3
