"""Tests for runs handler — RunLedger data access via get_runs / get_run_by_id."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.pipeline.backbone_contracts import RunLedger, RunStageEvent
from aragora.server.handlers.runs import (
    get_backbone_run,
    get_run_by_id,
    get_runs,
    _run_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(stage: str = "intake", status: str = "completed") -> RunStageEvent:
    return RunStageEvent(
        stage=stage,
        status=status,
    )


def _make_run(
    run_id: str = "run-1",
    status: str = "running",
    execution_id: str = "exec-1",
    receipt_id: str = "rcpt-1",
    safety_mode: str = "strict",
    events: list[RunStageEvent] | None = None,
) -> RunLedger:
    return RunLedger(
        run_id=run_id,
        entrypoint="test",
        status=status,
        execution_id=execution_id,
        receipt_id=receipt_id,
        stage_events=events or [],
        metadata={"safety_mode": safety_mode},
    )


def _mock_store(
    runs: list[RunLedger] | None = None,
    single: RunLedger | None = None,
) -> MagicMock:
    store = MagicMock()
    store.list_runs.return_value = runs or []
    store.get_run.return_value = single
    return store


# ---------------------------------------------------------------------------
# _run_summary
# ---------------------------------------------------------------------------


class TestRunSummary:
    def test_extracts_expected_fields(self) -> None:
        run = _make_run()
        summary = _run_summary(run)
        assert summary["run_id"] == "run-1"
        assert summary["status"] == "running"
        assert summary["execution_id"] == "exec-1"
        assert summary["receipt_id"] == "rcpt-1"
        assert summary["safety_mode"] == "strict"
        assert summary["stages"] == []

    def test_stages_serialised(self) -> None:
        evt = _make_event("deliberation", "in_progress")
        run = _make_run(events=[evt])
        summary = _run_summary(run)
        assert len(summary["stages"]) == 1
        assert summary["stages"][0]["stage"] == "deliberation"

    def test_safety_mode_defaults_to_standard(self) -> None:
        run = _make_run()
        run.metadata = {}
        summary = _run_summary(run)
        assert summary["safety_mode"] == "standard"


# ---------------------------------------------------------------------------
# get_backbone_run
# ---------------------------------------------------------------------------


class TestGetBackboneRun:
    def test_delegates_to_store_get_run(self) -> None:
        run = _make_run()
        store = _mock_store(single=run)
        result = get_backbone_run(store, "run-1")
        store.get_run.assert_called_once_with("run-1")
        assert result is run

    def test_returns_none_when_missing(self) -> None:
        store = _mock_store(single=None)
        assert get_backbone_run(store, "no-such") is None


# ---------------------------------------------------------------------------
# get_runs
# ---------------------------------------------------------------------------


class TestGetRuns:
    def test_returns_empty_list(self) -> None:
        store = _mock_store(runs=[])
        result = get_runs(store)
        assert result == {"runs": [], "total": 0}

    def test_returns_summaries(self) -> None:
        runs = [_make_run("r1"), _make_run("r2", status="completed")]
        store = _mock_store(runs=runs)
        result = get_runs(store)
        assert result["total"] == 2
        assert result["runs"][0]["run_id"] == "r1"
        assert result["runs"][1]["status"] == "completed"

    def test_passes_filters_to_store(self) -> None:
        store = _mock_store()
        get_runs(store, status="completed", limit=10, offset=5)
        store.list_runs.assert_called_once_with(
            status="completed",
            limit=10,
            offset=5,
        )


# ---------------------------------------------------------------------------
# get_run_by_id
# ---------------------------------------------------------------------------


class TestGetRunById:
    def test_returns_summary_for_existing_run(self) -> None:
        run = _make_run(safety_mode="permissive")
        store = _mock_store(single=run)
        result = get_run_by_id(store, "run-1")
        assert result is not None
        assert result["run_id"] == "run-1"
        assert result["safety_mode"] == "permissive"

    def test_returns_none_for_missing_run(self) -> None:
        store = _mock_store(single=None)
        assert get_run_by_id(store, "nope") is None

    def test_includes_stage_events(self) -> None:
        evt = _make_event("spec", "completed")
        run = _make_run(events=[evt])
        store = _mock_store(single=run)
        result = get_run_by_id(store, "run-1")
        assert result is not None
        assert len(result["stages"]) == 1
        assert result["stages"][0]["stage"] == "spec"
