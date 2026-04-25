"""Unit tests for `aragora review-queue baseline` subcommand (#6375 step C).

The subcommand is the operator-facing surface for the empirical-
threshold framework that landed in #6602 (phase 1) and #6615 (phase 2
adapter). Tests cover argument validation, json/text output, default-
empty-stores behavior, the placeholder-vs-derived branch, and graceful
failure when the calibration store cannot be opened.
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from aragora.cli.commands.review_queue import (
    _cmd_baseline,
    _fmt_rate,
    _render_baseline_report,
    add_review_queue_parser,
    cmd_review_queue,
)
from aragora.review.invalidation import (
    BaselineMeasurement,
    DEFAULT_BASELINE_WINDOW_DAYS,
    DEFAULT_MIN_BASELINE_SAMPLES,
    DEFAULT_MINIMUM_MEANINGFUL_RATE,
    DEFAULT_SAFETY_MARGIN,
    ThresholdProposal,
)
from aragora.triage.auto_handle_calibration import (
    AUTO_HANDLE_PATH_FIRE_AND_FORGET,
    OUTCOME_REVERT,
    OUTCOME_SUCCESS,
    AutoHandleCalibrationStore,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def calibration_db(tmp_path: Path) -> str:
    """A file-backed calibration store path; the CLI cannot use ``:memory:``
    because it constructs the store itself."""
    return str(tmp_path / "auto_handle_calibration.db")


def _seed_calibration(db_path: str, *, success: int = 0, revert: int = 0) -> None:
    store = AutoHandleCalibrationStore(db_path=db_path)
    counter = 0
    for outcome, n in ((OUTCOME_SUCCESS, success), (OUTCOME_REVERT, revert)):
        for _ in range(n):
            counter += 1
            store.record_outcome(
                decision_id=f"d-{counter}-{outcome}",
                auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                decision_class="low_risk:scope=tests:size=S",
                outcome=outcome,
                pr_number=counter,
            )


def _write_receipt(receipts_dir: Path, *, pr_number: int) -> None:
    receipts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "pr_number": pr_number,
        "action": "approve",
        "reviewed_at": datetime.now(UTC).isoformat(),
        "session_id": f"sess-{pr_number}",
    }
    (receipts_dir / f"pr-{pr_number}-sess-{pr_number}-approve.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _baseline_args(**overrides: Any) -> argparse.Namespace:
    """Build a Namespace mirroring the parser defaults."""
    defaults: dict[str, Any] = {
        "review_queue_command": "baseline",
        "window_days": DEFAULT_BASELINE_WINDOW_DAYS,
        "min_samples": DEFAULT_MIN_BASELINE_SAMPLES,
        "safety_margin": DEFAULT_SAFETY_MARGIN,
        "minimum_meaningful_rate": DEFAULT_MINIMUM_MEANINGFUL_RATE,
        "placeholder_value": 0.05,
        "calibration_db": None,
        "review_queue_root": None,
        "json": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def test_parser_registers_baseline_subcommand() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    add_review_queue_parser(sub)
    args = parser.parse_args(["review-queue", "baseline"])
    assert args.review_queue_command == "baseline"
    assert args.window_days == DEFAULT_BASELINE_WINDOW_DAYS
    assert args.min_samples == DEFAULT_MIN_BASELINE_SAMPLES
    assert args.safety_margin == DEFAULT_SAFETY_MARGIN
    assert args.json is False


def test_parser_passes_through_overrides() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    add_review_queue_parser(sub)
    args = parser.parse_args(
        [
            "review-queue",
            "baseline",
            "--window-days",
            "60",
            "--min-samples",
            "100",
            "--safety-margin",
            "0.4",
            "--minimum-meaningful-rate",
            "0.005",
            "--placeholder-value",
            "0.07",
            "--calibration-db",
            "/tmp/x.db",
            "--review-queue-root",
            "/tmp/rq",
            "--json",
        ]
    )
    assert args.window_days == 60
    assert args.min_samples == 100
    assert args.safety_margin == pytest.approx(0.4)
    assert args.minimum_meaningful_rate == pytest.approx(0.005)
    assert args.placeholder_value == pytest.approx(0.07)
    assert args.calibration_db == "/tmp/x.db"
    assert args.review_queue_root == "/tmp/rq"
    assert args.json is True


def test_dispatcher_routes_baseline_to_handler(tmp_path: Path) -> None:
    # cmd_review_queue routes review_queue_command="baseline" to _cmd_baseline.
    args = _baseline_args(
        calibration_db=str(tmp_path / "x.db"),
        review_queue_root=tmp_path,
    )
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = cmd_review_queue(args)
    assert rc == 0
    assert "Empirical invalidation baseline" in out.getvalue()


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_cmd_baseline_rejects_zero_window_days() -> None:
    args = _baseline_args(window_days=0)
    err = io.StringIO()
    with redirect_stderr(err):
        rc = _cmd_baseline(args)
    assert rc == 2
    assert "--window-days must be positive" in err.getvalue()


def test_cmd_baseline_rejects_negative_min_samples() -> None:
    args = _baseline_args(min_samples=-1)
    err = io.StringIO()
    with redirect_stderr(err):
        rc = _cmd_baseline(args)
    assert rc == 2
    assert "--min-samples must be positive" in err.getvalue()


def test_cmd_baseline_rejects_safety_margin_above_one() -> None:
    args = _baseline_args(safety_margin=1.5)
    err = io.StringIO()
    with redirect_stderr(err):
        rc = _cmd_baseline(args)
    assert rc == 2
    assert "--safety-margin must be in (0, 1]" in err.getvalue()


def test_cmd_baseline_rejects_safety_margin_zero() -> None:
    args = _baseline_args(safety_margin=0.0)
    err = io.StringIO()
    with redirect_stderr(err):
        rc = _cmd_baseline(args)
    assert rc == 2


def test_cmd_baseline_rejects_zero_minimum_meaningful_rate() -> None:
    args = _baseline_args(minimum_meaningful_rate=0.0)
    err = io.StringIO()
    with redirect_stderr(err):
        rc = _cmd_baseline(args)
    assert rc == 2
    assert "--minimum-meaningful-rate must be positive" in err.getvalue()


def test_cmd_baseline_rejects_placeholder_outside_unit_interval() -> None:
    args = _baseline_args(placeholder_value=1.5)
    err = io.StringIO()
    with redirect_stderr(err):
        rc = _cmd_baseline(args)
    assert rc == 2
    assert "--placeholder-value must be in (0, 1)" in err.getvalue()


def test_cmd_baseline_rejects_placeholder_zero() -> None:
    args = _baseline_args(placeholder_value=0.0)
    err = io.StringIO()
    with redirect_stderr(err):
        rc = _cmd_baseline(args)
    assert rc == 2


# ---------------------------------------------------------------------------
# Empty-store path: no data anywhere
# ---------------------------------------------------------------------------


def test_cmd_baseline_no_data_emits_placeholder_text(tmp_path: Path, calibration_db: str) -> None:
    args = _baseline_args(
        calibration_db=calibration_db,
        review_queue_root=tmp_path,
    )
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = _cmd_baseline(args)
    assert rc == 0
    text = out.getvalue()
    assert "Empirical invalidation baseline" in text
    assert "human-settled:   0 invalidated / 0 total" in text
    assert "placeholder:   True" in text
    # Default placeholder is 5%
    assert "0.0500 (5.00%)" in text
    assert err.getvalue() == ""


def test_cmd_baseline_no_data_emits_placeholder_json(tmp_path: Path, calibration_db: str) -> None:
    args = _baseline_args(
        calibration_db=calibration_db,
        review_queue_root=tmp_path,
        json=True,
    )
    out = io.StringIO()
    with redirect_stdout(out):
        rc = _cmd_baseline(args)
    assert rc == 0
    payload = json.loads(out.getvalue())
    assert "measurement" in payload
    assert "proposal" in payload
    proposal = payload["proposal"]
    assert proposal["is_placeholder"] is True
    assert proposal["threshold"] == pytest.approx(0.05)
    assert proposal["sample_size"] == 0
    measurement = payload["measurement"]
    assert measurement["total_human_settled"] == 0
    assert measurement["total_auto_handled"] == 0


# ---------------------------------------------------------------------------
# With calibration data + receipts
# ---------------------------------------------------------------------------


def test_cmd_baseline_with_data_text_output(tmp_path: Path, calibration_db: str) -> None:
    _seed_calibration(calibration_db, success=50, revert=5)

    receipts_dir = tmp_path / ".aragora" / "review-queue" / "receipts"
    for i in range(60):
        _write_receipt(receipts_dir, pr_number=i)

    args = _baseline_args(
        calibration_db=calibration_db,
        review_queue_root=tmp_path / ".aragora" / "review-queue",
    )
    out = io.StringIO()
    with redirect_stdout(out):
        rc = _cmd_baseline(args)
    assert rc == 0
    text = out.getvalue()
    # Auto-handle side has 5/55
    assert "auto-handled:    5 invalidated / 55 total" in text
    # Human side denominator from receipts
    assert "human-settled:   0 invalidated / 60 total" in text
    # Sample-size acceptable since 60 >= 50 default
    assert "acceptable: True" in text
    # Threshold derived not placeholder (baseline = 0/60 = 0; floor kicks in)
    assert "placeholder:   False" in text
    # Note: human invalidation source gap surfaced
    assert "human_invalidations_source" in text


def test_cmd_baseline_with_data_json_output(tmp_path: Path, calibration_db: str) -> None:
    _seed_calibration(calibration_db, success=50, revert=5)
    receipts_dir = tmp_path / ".aragora" / "review-queue" / "receipts"
    for i in range(60):
        _write_receipt(receipts_dir, pr_number=i)

    args = _baseline_args(
        calibration_db=calibration_db,
        review_queue_root=tmp_path / ".aragora" / "review-queue",
        json=True,
    )
    out = io.StringIO()
    with redirect_stdout(out):
        rc = _cmd_baseline(args)
    assert rc == 0
    payload = json.loads(out.getvalue())
    measurement = payload["measurement"]
    proposal = payload["proposal"]
    assert measurement["total_human_settled"] == 60
    assert measurement["total_auto_handled"] == 55
    assert measurement["invalidated_auto_handled"] == 5
    assert measurement["invalidated_human_settled"] == 0
    assert proposal["is_placeholder"] is False
    assert proposal["threshold"] == pytest.approx(
        DEFAULT_MINIMUM_MEANINGFUL_RATE
    )  # baseline 0 * margin → floor


def test_cmd_baseline_below_min_samples_falls_back_to_placeholder(
    tmp_path: Path, calibration_db: str
) -> None:
    receipts_dir = tmp_path / ".aragora" / "review-queue" / "receipts"
    for i in range(10):  # below default min 50
        _write_receipt(receipts_dir, pr_number=i)
    args = _baseline_args(
        calibration_db=calibration_db,
        review_queue_root=tmp_path / ".aragora" / "review-queue",
        json=True,
    )
    out = io.StringIO()
    with redirect_stdout(out):
        rc = _cmd_baseline(args)
    assert rc == 0
    payload = json.loads(out.getvalue())
    assert payload["proposal"]["is_placeholder"] is True
    assert payload["measurement"]["sample_size_acceptable"] is False


def test_cmd_baseline_custom_safety_margin(tmp_path: Path, calibration_db: str) -> None:
    args = _baseline_args(
        calibration_db=calibration_db,
        review_queue_root=tmp_path,
        safety_margin=0.25,
        json=True,
    )
    out = io.StringIO()
    with redirect_stdout(out):
        rc = _cmd_baseline(args)
    assert rc == 0
    payload = json.loads(out.getvalue())
    assert payload["proposal"]["safety_margin"] == pytest.approx(0.25)


def test_cmd_baseline_custom_placeholder_value(tmp_path: Path, calibration_db: str) -> None:
    args = _baseline_args(
        calibration_db=calibration_db,
        review_queue_root=tmp_path,
        placeholder_value=0.07,
        json=True,
    )
    out = io.StringIO()
    with redirect_stdout(out):
        rc = _cmd_baseline(args)
    assert rc == 0
    payload = json.loads(out.getvalue())
    proposal = payload["proposal"]
    assert proposal["is_placeholder"] is True
    assert proposal["threshold"] == pytest.approx(0.07)


# ---------------------------------------------------------------------------
# Defensive paths
# ---------------------------------------------------------------------------


def test_cmd_baseline_handles_unreadable_calibration_db(tmp_path: Path) -> None:
    # Pass a path under a non-existent parent so SQLite cannot create the file
    bad = tmp_path / "no" / "such" / "dir" / "x.db"
    args = _baseline_args(calibration_db=str(bad), review_queue_root=tmp_path)
    err = io.StringIO()
    out = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = _cmd_baseline(args)
    # SQLite raises OperationalError for unable-to-open; the CLI should
    # return an error exit and emit a single-line message.
    assert rc == 1
    assert "calibration store" in err.getvalue() or "error" in err.getvalue()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_fmt_rate_handles_none() -> None:
    assert _fmt_rate(None) == "n/a"


def test_fmt_rate_formats_value() -> None:
    out = _fmt_rate(0.0123)
    assert "0.0123" in out
    assert "1.23%" in out


def test_render_baseline_report_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    """Sanity-check the renderer prints without error for a synthetic
    measurement + proposal."""
    now = datetime.now(UTC)
    measurement = BaselineMeasurement(
        window_start=now - timedelta(days=30),
        window_end=now,
        window_days=30,
        total_human_settled=60,
        invalidated_human_settled=2,
        total_auto_handled=55,
        invalidated_auto_handled=5,
        baseline_human_rate=2 / 60,
        baseline_human_rate_ci_low=0.005,
        baseline_human_rate_ci_high=0.10,
        auto_handle_rate=5 / 55,
        auto_handle_rate_ci_low=0.04,
        auto_handle_rate_ci_high=0.18,
        per_class_human={"low_risk": (2, 60)},
        per_class_auto={"low_risk": (5, 55)},
        min_samples_required=50,
        sample_size_acceptable=True,
        notes={"caveat": "synthetic"},
    )
    proposal = ThresholdProposal(
        threshold=0.0167,
        baseline=2 / 60,
        sample_size=60,
        safety_margin=0.5,
        minimum_meaningful_rate=0.01,
        is_placeholder=False,
        rationale="threshold derived from sample",
        measured_at=now,
        measurement_window_days=30,
    )
    _render_baseline_report(measurement=measurement, proposal=proposal)
    captured = capsys.readouterr()
    assert "Empirical invalidation baseline" in captured.out
    assert "per-class human breakdown" in captured.out
    assert "per-class auto-handle breakdown" in captured.out
    assert "caveat: synthetic" in captured.out
    assert "threshold derived from sample" in captured.out
    assert "advisory" in captured.out.lower() or "read-only" in captured.out.lower()
