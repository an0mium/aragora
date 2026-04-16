"""Tests for the boss loop outcome and metrics extraction module."""

from __future__ import annotations

from aragora.swarm.boss_loop_outcome import (
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
