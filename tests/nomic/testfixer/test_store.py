"""Tests for TestFixerAttemptStore."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from aragora.nomic.testfixer.store import TestFixerAttemptStore
from aragora.nomic.testfixer.runner import TestFailure
from aragora.nomic.testfixer.analyzer import FailureAnalysis, FailureCategory, FixTarget
from aragora.nomic.testfixer.proposer import PatchProposal, PatchStatus
from aragora.nomic.testfixer.orchestrator import FixAttempt, FixLoopResult, LoopStatus


def _make_failure() -> TestFailure:
    return TestFailure(
        test_name="test_example",
        test_file="tests/test_foo.py",
        error_type="AssertionError",
        error_message="expected 1 got 2",
        stack_trace="Traceback ...",
    )


def _make_analysis(failure: TestFailure | None = None) -> FailureAnalysis:
    return FailureAnalysis(
        failure=failure or _make_failure(),
        category=FailureCategory.TEST_ASSERTION,
        confidence=0.8,
        fix_target=FixTarget.TEST_FILE,
        root_cause="Wrong assertion value",
        root_cause_file="tests/test_foo.py",
    )


def _make_proposal(analysis: FailureAnalysis | None = None) -> PatchProposal:
    return PatchProposal(
        id="fix_1",
        analysis=analysis or _make_analysis(),
        status=PatchStatus.PROPOSED,
        description="Fix assertion",
        post_debate_confidence=0.75,
    )


def _make_attempt() -> FixAttempt:
    failure = _make_failure()
    analysis = _make_analysis(failure)
    proposal = _make_proposal(analysis)
    return FixAttempt(
        iteration=1,
        failure=failure,
        analysis=analysis,
        proposal=proposal,
        applied=True,
        test_result_after=None,
        success=True,
        notes=["Applied successfully"],
    )


def _make_run_result() -> FixLoopResult:
    attempt = _make_attempt()
    now = datetime.now()
    return FixLoopResult(
        status=LoopStatus.SUCCESS,
        started_at=now,
        finished_at=now,
        total_iterations=1,
        fixes_applied=1,
        fixes_successful=1,
        fixes_reverted=0,
        attempts=[attempt],
    )


class TestStoreCreatesDirectories:
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "store.jsonl"
        store = TestFixerAttemptStore(nested)
        assert store.path == nested
        assert nested.parent.exists()


class TestAppend:
    def test_writes_json_line_with_recorded_at(self, tmp_path: Path) -> None:
        path = tmp_path / "store.jsonl"
        store = TestFixerAttemptStore(path)
        store._append({"key": "value"})

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["key"] == "value"
        assert "recorded_at" in record
        # Verify recorded_at is a valid ISO timestamp
        datetime.fromisoformat(record["recorded_at"])


class TestRecordAttempt:
    def test_writes_attempt_type_record_with_all_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "store.jsonl"
        store = TestFixerAttemptStore(path)
        attempt = _make_attempt()

        store.record_attempt(attempt)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["type"] == "attempt"
        assert record["iteration"] == 1
        assert record["applied"] is True
        assert record["success"] is True
        assert record["notes"] == ["Applied successfully"]
        assert "recorded_at" in record

        # Check nested failure fields
        assert record["failure"]["test_name"] == "test_example"
        assert record["failure"]["test_file"] == "tests/test_foo.py"
        assert record["failure"]["error_type"] == "AssertionError"
        assert record["failure"]["error_message"] == "expected 1 got 2"

        # Check nested analysis fields
        assert record["analysis"]["category"] == "test_assertion"
        assert record["analysis"]["fix_target"] == "test_file"
        assert record["analysis"]["confidence"] == 0.8
        assert record["analysis"]["root_cause"] == "Wrong assertion value"
        assert record["analysis"]["root_cause_file"] == "tests/test_foo.py"

        # Check nested proposal fields
        assert record["proposal"]["id"] == "fix_1"
        assert record["proposal"]["description"] == "Fix assertion"
        assert record["proposal"]["confidence"] == 0.75


class TestRecordRun:
    def test_writes_run_type_record(self, tmp_path: Path) -> None:
        path = tmp_path / "store.jsonl"
        store = TestFixerAttemptStore(path)
        result = _make_run_result()

        store.record_run(result)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["type"] == "run"
        assert record["status"] == "success"
        assert record["total_iterations"] == 1
        assert record["fixes_applied"] == 1
        assert record["fixes_successful"] == 1
        assert record["fixes_reverted"] == 0
        assert "recorded_at" in record
        assert "started_at" in record
        assert "finished_at" in record

        # Check attempts summary
        assert len(record["attempts"]) == 1
        assert record["attempts"][0]["iteration"] == 1
        assert record["attempts"][0]["failure"] == "test_example"
        assert record["attempts"][0]["success"] is True


class TestMultipleRecords:
    def test_multiple_records_append_correctly(self, tmp_path: Path) -> None:
        path = tmp_path / "store.jsonl"
        store = TestFixerAttemptStore(path)

        attempt = _make_attempt()
        result = _make_run_result()

        store.record_attempt(attempt)
        store.record_attempt(attempt)
        store.record_run(result)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 3

        records = [json.loads(line) for line in lines]
        assert records[0]["type"] == "attempt"
        assert records[1]["type"] == "attempt"
        assert records[2]["type"] == "run"

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "store.jsonl"
        store = TestFixerAttemptStore(path)

        attempt = _make_attempt()
        result = _make_run_result()

        store.record_attempt(attempt)
        store.record_run(result)
        store.record_attempt(attempt)

        lines = path.read_text().strip().splitlines()
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
            assert "recorded_at" in parsed
