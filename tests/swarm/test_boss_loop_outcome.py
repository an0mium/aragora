"""Tests for the boss loop outcome and metrics extraction module."""

from __future__ import annotations

import json
from pathlib import Path

from aragora.swarm.boss_loop_outcome import (
    append_iteration_metrics,
    extract_iteration_metrics,
    freshness_is_fresh,
    freshness_to_dict,
)


class TestExtractIterationMetrics:
    def test_empty_result(self) -> None:
        assert extract_iteration_metrics({}) == (0, 0, 0)

    def test_no_run_dict(self) -> None:
        assert extract_iteration_metrics({"status": "completed"}) == (0, 0, 0)

    def test_run_with_no_work_orders(self) -> None:
        assert extract_iteration_metrics({"run": {}}) == (0, 0, 0)

    def test_changed_files_counted(self) -> None:
        result = {
            "run": {
                "work_orders": [
                    {"changed_paths": ["a.py", "b.py"]},
                    {"changed_paths": ["c.py"]},
                ]
            }
        }
        files, tests, passed = extract_iteration_metrics(result)
        assert files == 3

    def test_duplicate_files_deduped(self) -> None:
        result = {
            "run": {
                "work_orders": [
                    {"changed_paths": ["a.py", "b.py"]},
                    {"changed_paths": ["a.py"]},
                ]
            }
        }
        files, _, _ = extract_iteration_metrics(result)
        assert files == 2

    def test_tests_run_counted(self) -> None:
        result = {
            "run": {
                "work_orders": [
                    {"tests_run": ["pytest tests/a.py", "pytest tests/b.py"]},
                ]
            }
        }
        _, tests, _ = extract_iteration_metrics(result)
        assert tests == 2

    def test_verification_results_passed(self) -> None:
        result = {
            "run": {
                "work_orders": [
                    {
                        "verification_results": [
                            {"passed": True},
                            {"passed": False},
                            {"passed": True},
                        ]
                    },
                ]
            }
        }
        _, _, passed = extract_iteration_metrics(result)
        assert passed == 2

    def test_completed_without_verification_assumes_pass(self) -> None:
        result = {
            "status": "completed",
            "run": {
                "work_orders": [
                    {"tests_run": ["pytest a.py", "pytest b.py"]},
                ]
            },
        }
        _, tests, passed = extract_iteration_metrics(result)
        assert tests == 2
        assert passed == 2  # assumes pass when completed + no verification results


class TestFreshnessHelpers:
    def test_freshness_to_dict_from_dict(self) -> None:
        assert freshness_to_dict({"fresh": True, "details": {}}) == {"fresh": True, "details": {}}

    def test_freshness_to_dict_from_object(self) -> None:
        class Obj:
            def to_dict(self) -> dict:
                return {"fresh": True}

        assert freshness_to_dict(Obj()) == {"fresh": True}

    def test_freshness_to_dict_fallback(self) -> None:
        assert freshness_to_dict("not a dict") == {}

    def test_freshness_is_fresh_true(self) -> None:
        assert freshness_is_fresh(None, {"fresh": True}) is True

    def test_freshness_is_fresh_false(self) -> None:
        assert freshness_is_fresh(None, {"fresh": False}) is False

    def test_freshness_is_fresh_from_attr(self) -> None:
        class Obj:
            fresh = True

        assert freshness_is_fresh(Obj(), {}) is True

    def test_freshness_is_fresh_missing(self) -> None:
        assert freshness_is_fresh(None, {}) is False


class TestDispatchSkipReason:
    """Regression tests for the dispatch_skip_reason field.

    The empty-prompt no-op pathology in
    .aragora/overnight/boss_metrics.jsonl had 289/406 rows with
    prompt_chars=0, hiding 35 retry-loop rows on a single closed/merged
    issue. dispatch_skip_reason makes those rows immediately
    distinguishable from real attempts that genuinely had a 0-char
    prompt.
    """

    def _common_args(self, tmp_path: Path) -> dict:
        return {
            "metrics_jsonl_path": str(tmp_path / "boss_metrics.jsonl"),
            "outcome_learner_window": 50,
            "deferred_queue_depth": 0,
            "iteration": 1,
            "issue_number": 1,
            "elapsed_seconds": 0.5,
            "files_changed": 0,
            "tests_run": 0,
            "tests_passed": 0,
        }

    def _read_row(self, tmp_path: Path) -> dict:
        path = tmp_path / "boss_metrics.jsonl"
        assert path.exists(), "metrics file should be created"
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        assert len(lines) == 1
        return json.loads(lines[0])

    def test_needs_human_no_prompt_marks_skip_reason(self, tmp_path: Path) -> None:
        worker_result = {
            "status": "needs_human",
            "outcome": "blocked",
            "failure_reason": "Approval required for merge.",
            "run": {"work_orders": []},
        }
        append_iteration_metrics(
            **self._common_args(tmp_path),
            worker_result=worker_result,
        )
        row = self._read_row(tmp_path)
        assert row["prompt_chars"] == 0
        assert row["worker_status"] == "needs_human"
        assert row["dispatch_skip_reason"] == "needs_human_no_prompt"

    def test_dropped_no_prompt_marks_skip_reason(self, tmp_path: Path) -> None:
        worker_result = {
            "status": "dropped",
            "outcome": "no_dispatch",
            "run": {},
        }
        append_iteration_metrics(
            **self._common_args(tmp_path),
            worker_result=worker_result,
        )
        row = self._read_row(tmp_path)
        assert row["dispatch_skip_reason"] == "dispatch_dropped_no_prompt"

    def test_no_work_orders_marks_skip_reason(self, tmp_path: Path) -> None:
        worker_result = {
            "status": "completed",
            "run": {"work_orders": []},
        }
        append_iteration_metrics(
            **self._common_args(tmp_path),
            worker_result=worker_result,
        )
        row = self._read_row(tmp_path)
        assert row["dispatch_skip_reason"] == "no_work_orders"

    def test_real_run_has_no_skip_reason(self, tmp_path: Path) -> None:
        worker_result = {
            "status": "completed",
            "outcome": "merged",
            "run": {
                "work_orders": [
                    {"prompt_chars": 1234, "enriched_context_chars": 4321},
                ]
            },
        }
        append_iteration_metrics(
            **self._common_args(tmp_path),
            worker_result=worker_result,
        )
        row = self._read_row(tmp_path)
        assert row["prompt_chars"] == 1234
        assert row["enriched_context_chars"] == 4321
        assert row["dispatch_skip_reason"] is None
