"""Tests for :mod:`aragora.swarm.queue_autofill`.

These exercise the policy surface of ``maybe_autofill_queue`` without
touching the real scanner, classifier, or GitHub.  Every dependency is
injected via keyword arguments to the function under test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from aragora.swarm.queue_autofill import (
    ALLOWED_CATEGORIES,
    AutofillCandidate,
    AutofillResult,
    DEFAULT_EMPTY_TICK_THRESHOLD,
    QUEUE_AUTOFILL_FLAG_ENV,
    maybe_autofill_queue,
    queue_autofill_enabled,
)


# ---------------------------------------------------------------------------
# Minimal test doubles
# ---------------------------------------------------------------------------


@dataclass
class _FakeCandidate:
    category: str
    title: str
    description: str = ""
    file_scope: list[str] = field(default_factory=list)
    new_files: list[str] = field(default_factory=list)
    validation_command: str = "pytest -q"
    acceptance_criteria: list[str] = field(default_factory=list)
    fingerprint: str = "fp-test"


@dataclass
class _FakeDecision:
    allowed: bool
    lane: str = "test_coverage_autofill"


def _make_scan(
    results: list[_FakeCandidate] | None = None,
    *,
    raise_exc: Exception | None = None,
) -> Any:
    results = results or []

    def scan(repo_root: Path, *, categories: list[str]) -> list[_FakeCandidate]:
        if raise_exc is not None:
            raise raise_exc
        return [candidate for candidate in results if candidate.category in categories]

    return scan


def _make_classify(allowed: bool = True, lane: str = "test_coverage_autofill") -> Any:
    def classify(title: str, body: str, *, labels: Any = (), repo_root: Path | None = None) -> Any:
        return _FakeDecision(allowed=allowed, lane=lane)

    return classify


def _make_validate(ok: bool = True, reason: str = "") -> Any:
    def validate(body: str) -> tuple[bool, str]:
        return ok, reason

    return validate


FLAG_ENV = {QUEUE_AUTOFILL_FLAG_ENV: "1"}


# ---------------------------------------------------------------------------
# Flag gate
# ---------------------------------------------------------------------------


def test_queue_autofill_enabled_defaults_off() -> None:
    assert queue_autofill_enabled({}) is False
    assert queue_autofill_enabled({QUEUE_AUTOFILL_FLAG_ENV: "0"}) is False


def test_queue_autofill_enabled_truthy() -> None:
    assert queue_autofill_enabled({QUEUE_AUTOFILL_FLAG_ENV: "1"}) is True
    assert queue_autofill_enabled({QUEUE_AUTOFILL_FLAG_ENV: "YES"}) is True


def test_flag_disabled_returns_skipped_and_does_not_scan(tmp_path: Path) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> list[Any]:  # pragma: no cover
        raise AssertionError("scan must not run when flag is off")

    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=5,
        env={},  # flag off
        scan_fn=boom,
        classify_fn=_make_classify(),
        validate_body_fn=_make_validate(),
        sentinel_path=tmp_path / "sentinel.json",
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
    )

    assert isinstance(result, AutofillResult)
    assert result.attempted is False
    assert result.reason == "flag_disabled"


# ---------------------------------------------------------------------------
# Threshold gate
# ---------------------------------------------------------------------------


def test_below_threshold_skips(tmp_path: Path) -> None:
    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=DEFAULT_EMPTY_TICK_THRESHOLD - 1,
        env=FLAG_ENV,
        scan_fn=_make_scan([]),
        classify_fn=_make_classify(),
        validate_body_fn=_make_validate(),
        sentinel_path=tmp_path / "sentinel.json",
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
    )
    assert result.attempted is False
    assert result.reason == "below_threshold"


# ---------------------------------------------------------------------------
# Existing work still matches filter → skip
# ---------------------------------------------------------------------------


class _ExistingCandidate:
    def __init__(self, title: str, body: str = "", labels: tuple[str, ...] = ("boss-ready",)):
        self.title = title
        self.body = body
        self.labels = labels


def test_skips_when_existing_candidate_would_pass_filter(tmp_path: Path) -> None:
    existing = [_ExistingCandidate("[TW-02] refresh benchmark truth artifact")]

    def classify(title: str, body: str, *, labels: Any = (), repo_root: Path | None = None) -> Any:
        # Existing candidate passes the filter.
        return _FakeDecision(allowed=True, lane="benchmark_regression")

    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=5,
        existing_candidates=existing,
        env=FLAG_ENV,
        scan_fn=_make_scan([_FakeCandidate(category="test_coverage", title="Add tests")]),
        classify_fn=classify,
        validate_body_fn=_make_validate(),
        sentinel_path=tmp_path / "sentinel.json",
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
    )

    assert result.attempted is False
    assert result.reason == "existing_queue_has_work"
    assert result.duplicate_count == 1


# ---------------------------------------------------------------------------
# Category filtering
# ---------------------------------------------------------------------------


def test_ignores_candidates_outside_allowed_categories(tmp_path: Path) -> None:
    candidates = [
        _FakeCandidate(category="silent_exception", title="drop silent excepts"),
        _FakeCandidate(category="type_annotation", title="add return types"),
    ]

    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=5,
        env=FLAG_ENV,
        scan_fn=_make_scan(candidates),
        classify_fn=_make_classify(),
        validate_body_fn=_make_validate(),
        sentinel_path=tmp_path / "sentinel.json",
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
    )

    # scan_fn filters by the explicit categories list we pass, so nothing
    # should even surface in the scanner output, and no issues get created.
    assert result.attempted is True
    assert result.reason == "no_eligible_candidates"
    assert "silent_exception" not in ALLOWED_CATEGORIES


def test_disallows_categories_even_if_scan_returns_them(tmp_path: Path) -> None:
    """Defence-in-depth: reject non-allowed categories even if scan leaks them."""

    def leaky_scan(repo_root: Path, *, categories: list[str]) -> list[_FakeCandidate]:
        return [_FakeCandidate(category="silent_exception", title="sneaky")]

    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=5,
        env=FLAG_ENV,
        scan_fn=leaky_scan,
        classify_fn=_make_classify(),
        validate_body_fn=_make_validate(),
        sentinel_path=tmp_path / "sentinel.json",
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
    )

    assert result.attempted is True
    assert result.reason == "no_eligible_candidates"
    assert result.filtered_out == 1


# ---------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------


def test_rate_limited_within_interval(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text(json.dumps({"last_run_ts": 1000.0}), encoding="utf-8")

    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=5,
        env=FLAG_ENV,
        now=1500.0,  # 500s since last_run, default interval is 3600
        scan_fn=_make_scan([]),
        classify_fn=_make_classify(),
        validate_body_fn=_make_validate(),
        sentinel_path=sentinel,
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
    )
    assert result.attempted is False
    assert result.reason == "rate_limited"
    assert result.rate_limited is True
    assert result.seconds_since_last == pytest.approx(500.0)


def test_passes_rate_limit_when_interval_elapsed(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text(json.dumps({"last_run_ts": 1000.0}), encoding="utf-8")
    candidate = _FakeCandidate(
        category="test_coverage",
        title="Add unit tests for module X",
        description="Add unit tests.",
        file_scope=["aragora/foo.py"],
    )

    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=5,
        env=FLAG_ENV,
        now=5000.0,  # 4000s > 3600s
        scan_fn=_make_scan([candidate]),
        classify_fn=_make_classify(),
        validate_body_fn=_make_validate(),
        sentinel_path=sentinel,
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
    )
    assert result.attempted is True
    assert result.reason == "created_dry_run"
    assert result.created_count == 1
    # Sentinel bumped forward to now=5000.0
    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    assert payload == {"last_run_ts": 5000.0}


# ---------------------------------------------------------------------------
# Happy path + max_issues clamp
# ---------------------------------------------------------------------------


def test_surfaces_up_to_max_issues(tmp_path: Path) -> None:
    candidates = [
        _FakeCandidate(
            category="test_coverage",
            title=f"Add unit tests for module {i}",
            description="Description",
            file_scope=[f"aragora/module_{i}.py"],
        )
        for i in range(5)
    ]
    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=10,
        env=FLAG_ENV,
        max_issues=2,
        scan_fn=_make_scan(candidates),
        classify_fn=_make_classify(),
        validate_body_fn=_make_validate(),
        sentinel_path=tmp_path / "sentinel.json",
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
    )

    assert result.attempted is True
    assert result.reason == "created_dry_run"
    assert result.created_count == 2


def test_classifier_rejection_is_filtered_out(tmp_path: Path) -> None:
    candidate = _FakeCandidate(
        category="test_coverage",
        title="Add tests",
        description="Boring",
        file_scope=["aragora/foo.py"],
    )
    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=5,
        env=FLAG_ENV,
        scan_fn=_make_scan([candidate]),
        classify_fn=_make_classify(allowed=False),
        validate_body_fn=_make_validate(),
        sentinel_path=tmp_path / "sentinel.json",
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
    )

    assert result.attempted is True
    assert result.reason == "no_eligible_candidates"
    assert result.filtered_out == 1


def test_sanitation_rejection_is_filtered_out(tmp_path: Path) -> None:
    candidate = _FakeCandidate(category="test_coverage", title="Add tests")
    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=5,
        env=FLAG_ENV,
        scan_fn=_make_scan([candidate]),
        classify_fn=_make_classify(),
        validate_body_fn=_make_validate(ok=False, reason="too_short"),
        sentinel_path=tmp_path / "sentinel.json",
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
    )

    assert result.attempted is True
    assert result.reason == "no_eligible_candidates"
    assert result.filtered_out == 1


# ---------------------------------------------------------------------------
# Metrics emission
# ---------------------------------------------------------------------------


def test_metrics_row_records_event_queue_autofill(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=5,
        env=FLAG_ENV,
        scan_fn=_make_scan([]),
        classify_fn=_make_classify(),
        validate_body_fn=_make_validate(),
        sentinel_path=tmp_path / "sentinel.json",
        metrics_jsonl_path=metrics,
    )
    assert result.attempted is True

    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "queue_autofill"
    assert rows[0]["reason"] == "no_eligible_candidates"
    assert "timestamp" in rows[0]


def test_metrics_row_includes_created_titles(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    candidate = _FakeCandidate(
        category="broad_exception",
        title="Narrow broad except in foo.py",
        description="Some description",
        file_scope=["aragora/foo.py"],
    )

    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=5,
        env=FLAG_ENV,
        scan_fn=_make_scan([candidate]),
        classify_fn=_make_classify(),
        validate_body_fn=_make_validate(),
        sentinel_path=tmp_path / "sentinel.json",
        metrics_jsonl_path=metrics,
    )
    assert result.attempted is True
    assert result.reason == "created_dry_run"
    assert result.created_count == 1

    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]
    payload = rows[0]
    assert payload["event"] == "queue_autofill"
    assert payload["created_count"] == 1
    assert payload["created"][0]["title"] == "Narrow broad except in foo.py"
    assert payload["created"][0]["category"] == "broad_exception"


# ---------------------------------------------------------------------------
# Failure accounting
# ---------------------------------------------------------------------------


def test_scan_failure_reports_error(tmp_path: Path) -> None:
    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=5,
        env=FLAG_ENV,
        scan_fn=_make_scan([], raise_exc=RuntimeError("boom")),
        classify_fn=_make_classify(),
        validate_body_fn=_make_validate(),
        sentinel_path=tmp_path / "sentinel.json",
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
    )
    assert result.attempted is True
    assert result.reason == "scan_failed"
    assert result.errors and "boom" in result.errors[0]


def test_create_issue_callback_is_ignored(tmp_path: Path) -> None:
    candidate = _FakeCandidate(
        category="test_coverage",
        title="Add unit tests",
        file_scope=["aragora/foo.py"],
    )
    invoked: list[AutofillCandidate] = []

    def should_not_run(item: AutofillCandidate) -> bool:
        invoked.append(item)
        return True

    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=5,
        env=FLAG_ENV,
        scan_fn=_make_scan([candidate]),
        classify_fn=_make_classify(),
        validate_body_fn=_make_validate(),
        create_issue=should_not_run,
        sentinel_path=tmp_path / "sentinel.json",
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
    )

    assert result.attempted is True
    assert result.reason == "created_dry_run"
    assert result.created_count == 1
    assert invoked == []
    assert result.errors


# ---------------------------------------------------------------------------
# Dry-run (no create callback)
# ---------------------------------------------------------------------------


def test_missing_create_callback_yields_dry_run(tmp_path: Path) -> None:
    candidate = _FakeCandidate(
        category="test_coverage",
        title="Add unit tests for bar.py",
        description="Cover bar.py",
        file_scope=["aragora/bar.py"],
    )
    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=5,
        env=FLAG_ENV,
        scan_fn=_make_scan([candidate]),
        classify_fn=_make_classify(),
        validate_body_fn=_make_validate(),
        create_issue=None,
        sentinel_path=tmp_path / "sentinel.json",
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
    )

    assert result.attempted is True
    assert result.reason == "created_dry_run"
    assert result.created_count == 1
    assert result.created[0].title == "Add unit tests for bar.py"


def test_max_issues_zero_short_circuits(tmp_path: Path) -> None:
    result = maybe_autofill_queue(
        repo_root=tmp_path,
        consecutive_empty_ticks=5,
        env=FLAG_ENV,
        max_issues=0,
        scan_fn=_make_scan([_FakeCandidate(category="test_coverage", title="x")]),
        classify_fn=_make_classify(),
        validate_body_fn=_make_validate(),
        sentinel_path=tmp_path / "sentinel.json",
        metrics_jsonl_path=tmp_path / "metrics.jsonl",
    )
    assert result.attempted is False
    assert result.reason == "max_issues_zero"
