"""Tests for B0 benchmark freshness detection in live_shift_status.

Closes #6798 Fix B: populate the previously-reserved-but-never-set
``current_benchmark_fresh`` payload key from the on-disk B0 truth artifact's
``generated_at`` timestamp, so a stale publication shows up as a degraded
operator signal instead of looking healthy.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from aragora.swarm.live_shift_status import (
    BENCHMARK_TRUTH_LATEST,
    _compose_freshness_warning,
    _detect_benchmark_freshness,
)


def _write_artifact(repo_root: Path, generated_at: str | None) -> Path:
    """Write a stub B0 truth artifact at the canonical relative path."""
    target = repo_root / BENCHMARK_TRUTH_LATEST
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"corpus_id": "tw-01-bounded-execution-v1", "revision": 3}
    if generated_at is not None:
        payload["generated_at"] = generated_at
    target.write_text(json.dumps(payload))
    return target


# ---------------------------------------------------------------------------
# _detect_benchmark_freshness
# ---------------------------------------------------------------------------


class TestMissingArtifact:
    def test_artifact_missing_yields_all_none(self, tmp_path: Path) -> None:
        out = _detect_benchmark_freshness(tmp_path)
        assert out == {
            "current_benchmark_fresh": None,
            "current_benchmark_age_hours": None,
            "current_benchmark_generated_at": None,
        }

    def test_repo_root_does_not_exist(self, tmp_path: Path) -> None:
        # A non-existent root is treated like a missing artifact, not an error.
        out = _detect_benchmark_freshness(tmp_path / "nope")
        assert out["current_benchmark_fresh"] is None


class TestUnparseableArtifact:
    def test_invalid_json(self, tmp_path: Path) -> None:
        target = tmp_path / BENCHMARK_TRUTH_LATEST
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("not json {")
        out = _detect_benchmark_freshness(tmp_path)
        assert out["current_benchmark_fresh"] is None

    def test_no_generated_at(self, tmp_path: Path) -> None:
        _write_artifact(tmp_path, None)
        out = _detect_benchmark_freshness(tmp_path)
        assert out["current_benchmark_fresh"] is None

    def test_unparseable_iso_timestamp(self, tmp_path: Path) -> None:
        _write_artifact(tmp_path, "yesterday afternoon")
        out = _detect_benchmark_freshness(tmp_path)
        assert out["current_benchmark_fresh"] is None


class TestFresh:
    def test_under_threshold_is_fresh(self, tmp_path: Path) -> None:
        recent = (
            (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
        )
        _write_artifact(tmp_path, recent)
        out = _detect_benchmark_freshness(tmp_path)
        assert out["current_benchmark_fresh"] is True
        assert out["current_benchmark_age_hours"] is not None
        assert 0 <= out["current_benchmark_age_hours"] <= 2.0
        assert out["current_benchmark_generated_at"] == recent

    def test_recorded_on_field_used_when_generated_at_missing(self, tmp_path: Path) -> None:
        recent = (
            (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
        )
        target = tmp_path / BENCHMARK_TRUTH_LATEST
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"recorded_on": recent}))
        out = _detect_benchmark_freshness(tmp_path)
        assert out["current_benchmark_fresh"] is True


class TestStale:
    def test_over_threshold_is_stale(self, tmp_path: Path) -> None:
        old = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat().replace("+00:00", "Z")
        _write_artifact(tmp_path, old)
        out = _detect_benchmark_freshness(tmp_path)
        assert out["current_benchmark_fresh"] is False
        assert out["current_benchmark_age_hours"] is not None
        assert out["current_benchmark_age_hours"] >= 24.0

    def test_threshold_override_changes_verdict(self, tmp_path: Path) -> None:
        # 5h old: fresh under default 24h, stale under 1h override.
        five_hours_ago = (
            (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat().replace("+00:00", "Z")
        )
        _write_artifact(tmp_path, five_hours_ago)
        assert _detect_benchmark_freshness(tmp_path)["current_benchmark_fresh"] is True
        assert (
            _detect_benchmark_freshness(tmp_path, max_age_hours=1.0)["current_benchmark_fresh"]
            is False
        )


class TestNaiveTimestamp:
    def test_naive_timestamp_treated_as_utc(self, tmp_path: Path) -> None:
        # Some publishers may forget the trailing Z. We treat naive as UTC.
        recent_naive = (
            (datetime.now(timezone.utc) - timedelta(hours=1)).replace(tzinfo=None).isoformat()
        )
        _write_artifact(tmp_path, recent_naive)
        out = _detect_benchmark_freshness(tmp_path)
        assert out["current_benchmark_fresh"] is True


# ---------------------------------------------------------------------------
# _compose_freshness_warning
# ---------------------------------------------------------------------------


class TestComposeWarning:
    def test_fresh_payload_no_warning_added(self) -> None:
        payload: dict[str, Any] = {"current_benchmark_fresh": True}
        out = _compose_freshness_warning(payload)
        assert "observer_warning" not in out

    def test_none_payload_no_warning_added(self) -> None:
        # Missing artifact => current_benchmark_fresh=None; we don't fabricate
        # a warning from absent data.
        payload: dict[str, Any] = {"current_benchmark_fresh": None}
        out = _compose_freshness_warning(payload)
        assert "observer_warning" not in out

    def test_stale_with_age_appends_to_existing_warning(self) -> None:
        payload: dict[str, Any] = {
            "current_benchmark_fresh": False,
            "current_benchmark_age_hours": 36.4,
            "observer_warning": "observer checkout is dirty",
        }
        out = _compose_freshness_warning(payload)
        assert "observer checkout is dirty" in out["observer_warning"]
        assert "benchmark truth stale" in out["observer_warning"]
        assert "36.4h" in out["observer_warning"]
        assert ";" in out["observer_warning"]

    def test_stale_with_no_existing_warning_creates_one(self) -> None:
        payload: dict[str, Any] = {
            "current_benchmark_fresh": False,
            "current_benchmark_age_hours": 8.9,
        }
        out = _compose_freshness_warning(payload)
        assert "observer_warning" in out
        assert "benchmark truth stale" in out["observer_warning"]
        assert "8.9h" in out["observer_warning"]

    def test_stale_with_no_age_uses_terse_fragment(self) -> None:
        payload: dict[str, Any] = {
            "current_benchmark_fresh": False,
            "current_benchmark_age_hours": None,
        }
        out = _compose_freshness_warning(payload)
        assert "observer_warning" in out
        assert "benchmark truth stale" in out["observer_warning"]
        # No age fragment when age is missing.
        assert "h old" not in out["observer_warning"]


# ---------------------------------------------------------------------------
# Schema invariant: the three new keys are always present in the payload
# even when the artifact is missing
# ---------------------------------------------------------------------------


class TestSchemaInvariant:
    @pytest.mark.parametrize(
        "scenario",
        ["missing", "invalid_json", "no_generated_at", "fresh", "stale"],
    )
    def test_three_keys_always_present(self, tmp_path: Path, scenario: str) -> None:
        if scenario == "missing":
            pass  # no file
        elif scenario == "invalid_json":
            target = tmp_path / BENCHMARK_TRUTH_LATEST
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("not json")
        elif scenario == "no_generated_at":
            _write_artifact(tmp_path, None)
        elif scenario == "fresh":
            _write_artifact(
                tmp_path,
                (datetime.now(timezone.utc) - timedelta(hours=1))
                .isoformat()
                .replace("+00:00", "Z"),
            )
        elif scenario == "stale":
            _write_artifact(
                tmp_path,
                (datetime.now(timezone.utc) - timedelta(hours=30))
                .isoformat()
                .replace("+00:00", "Z"),
            )
        out = _detect_benchmark_freshness(tmp_path)
        assert set(out.keys()) == {
            "current_benchmark_fresh",
            "current_benchmark_age_hours",
            "current_benchmark_generated_at",
        }
