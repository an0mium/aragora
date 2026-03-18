"""Tests for Ralph Prometheus metrics."""

from __future__ import annotations

from aragora.observability.metrics.ralph import (
    record_blocker_classified,
    record_budget_spent,
    record_campaign_status,
    record_campaign_step,
    record_pr_gate_disposition,
    record_repair_attempt,
    record_repair_outcome,
)


class TestRalphMetrics:
    def test_record_campaign_step_no_crash(self) -> None:
        record_campaign_step("camp-1", "campaign_iteration")

    def test_record_campaign_status_no_crash(self) -> None:
        record_campaign_status("camp-1", "running")

    def test_record_blocker_classified_no_crash(self) -> None:
        record_blocker_classified("scope_false_positive", is_deterministic=True)

    def test_record_repair_attempt_no_crash(self) -> None:
        record_repair_attempt("scope_false_positive")

    def test_record_repair_outcome_no_crash(self) -> None:
        record_repair_outcome("scope_false_positive", "success")

    def test_record_budget_spent_no_crash(self) -> None:
        record_budget_spent("camp-1", 5.50)

    def test_record_pr_gate_disposition_no_crash(self) -> None:
        record_pr_gate_disposition("merge_now")
