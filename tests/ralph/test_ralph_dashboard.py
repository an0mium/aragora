"""Tests for Ralph campaign dashboard data service."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from aragora.ralph.dashboard import RalphDashboard


def _write_state(state_dir: Path, filename: str, data: dict) -> Path:
    """Write a YAML state file to the test state directory."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / filename
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return path


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "ralph_states"


@pytest.fixture
def dashboard(state_dir: Path) -> RalphDashboard:
    return RalphDashboard(state_dir=state_dir)


@pytest.fixture
def sample_campaign(state_dir: Path) -> dict:
    data = {
        "supervisor_id": "sup-001",
        "campaign_id": "camp-alpha",
        "campaign_manifest_path": "/manifests/alpha.yaml",
        "status": "running",
        "current_step": 5,
        "last_campaign_result": {},
        "last_stop_reason": "campaign_blocked",
        "active_blocker": "scope_false_positive",
        "blocker_history": [
            {"step": 2, "kind": "scope_false_positive", "stop_reason": "campaign_blocked"},
            {"step": 4, "kind": "reviewer_missing_diff_context", "stop_reason": "campaign_blocked"},
        ],
        "repair_attempts": 1,
        "max_repair_attempts": 2,
        "active_merge_target": {
            "pr_url": "https://github.com/org/repo/pull/42",
            "branch": "fix/scope",
            "last_gate_snapshot": {
                "disposition": "wait_for_required_checks",
                "checks_passed": ["lint"],
                "checks_pending": ["test"],
                "checks_failed": [],
                "review_decision": "approved",
            },
        },
        "active_repair_pr": "https://github.com/org/repo/pull/42",
        "active_repair_branch": "fix/scope",
        "active_repair_task": None,
        "merge_commit_sha": None,
        "resume_attempts": 0,
        "resume_cursor": None,
        "budget_spent_usd": 3.50,
        "escalation_reason": None,
        "updated_at": "2026-03-18T12:00:00+00:00",
    }
    _write_state(state_dir, "camp-alpha.yaml", data)
    return data


@pytest.fixture
def completed_campaign(state_dir: Path) -> dict:
    data = {
        "supervisor_id": "sup-002",
        "campaign_id": "camp-beta",
        "campaign_manifest_path": "/manifests/beta.yaml",
        "status": "completed",
        "current_step": 8,
        "last_campaign_result": {},
        "last_stop_reason": "campaign_complete",
        "active_blocker": None,
        "blocker_history": [
            {
                "step": 3,
                "kind": "reviewer_auth_or_billing_failure",
                "stop_reason": "campaign_blocked",
            },
        ],
        "repair_attempts": 0,
        "max_repair_attempts": 2,
        "active_merge_target": None,
        "active_repair_pr": None,
        "active_repair_branch": None,
        "active_repair_task": None,
        "merge_commit_sha": "abc123",
        "resume_attempts": 0,
        "resume_cursor": None,
        "budget_spent_usd": 7.25,
        "escalation_reason": None,
        "updated_at": "2026-03-18T14:00:00+00:00",
    }
    _write_state(state_dir, "camp-beta.yaml", data)
    return data


class TestListCampaigns:
    def test_empty_dir(self, dashboard: RalphDashboard) -> None:
        assert dashboard.list_campaigns() == []

    def test_lists_campaigns(
        self, dashboard: RalphDashboard, sample_campaign: dict, completed_campaign: dict
    ) -> None:
        campaigns = dashboard.list_campaigns()
        assert len(campaigns) == 2
        ids = {c["campaign_id"] for c in campaigns}
        assert ids == {"camp-alpha", "camp-beta"}

    def test_includes_summary_fields(
        self, dashboard: RalphDashboard, sample_campaign: dict
    ) -> None:
        campaigns = dashboard.list_campaigns()
        c = campaigns[0]
        assert "status" in c
        assert "budget_spent_usd" in c
        assert "blocker_count" in c


class TestGetCampaignDetail:
    def test_found(self, dashboard: RalphDashboard, sample_campaign: dict) -> None:
        detail = dashboard.get_campaign_detail("camp-alpha")
        assert detail is not None
        assert detail["campaign_id"] == "camp-alpha"
        assert detail["status"] == "running"

    def test_not_found(self, dashboard: RalphDashboard) -> None:
        assert dashboard.get_campaign_detail("nonexistent") is None


class TestGetCampaignTimeline:
    def test_timeline_includes_blocker_history(
        self, dashboard: RalphDashboard, sample_campaign: dict
    ) -> None:
        timeline = dashboard.get_campaign_timeline("camp-alpha")
        assert len(timeline) >= 3  # 2 blocker entries + current state
        blocker_events = [e for e in timeline if e["event"] == "blocker_classified"]
        assert len(blocker_events) == 2
        assert blocker_events[0]["kind"] == "scope_false_positive"

    def test_timeline_ends_with_current_state(
        self, dashboard: RalphDashboard, sample_campaign: dict
    ) -> None:
        timeline = dashboard.get_campaign_timeline("camp-alpha")
        last = timeline[-1]
        assert last["event"] == "current_state"
        assert last["status"] == "running"

    def test_empty_for_missing(self, dashboard: RalphDashboard) -> None:
        assert dashboard.get_campaign_timeline("nonexistent") == []


class TestGetBlockerBreakdown:
    def test_single_campaign(self, dashboard: RalphDashboard, sample_campaign: dict) -> None:
        breakdown = dashboard.get_blocker_breakdown("camp-alpha")
        assert breakdown["total"] == 2
        assert breakdown["by_kind"]["scope_false_positive"] == 1
        assert breakdown["by_kind"]["reviewer_missing_diff_context"] == 1
        assert breakdown["deterministic_total"] == 2

    def test_global_aggregation(
        self, dashboard: RalphDashboard, sample_campaign: dict, completed_campaign: dict
    ) -> None:
        breakdown = dashboard.get_blocker_breakdown()
        assert breakdown["total"] == 3
        assert breakdown["escalation_total"] == 1  # auth/billing is escalation

    def test_empty(self, dashboard: RalphDashboard) -> None:
        breakdown = dashboard.get_blocker_breakdown()
        assert breakdown["total"] == 0


class TestGetRepairStats:
    def test_stats(
        self, dashboard: RalphDashboard, sample_campaign: dict, completed_campaign: dict
    ) -> None:
        stats = dashboard.get_repair_stats()
        assert stats["total_attempts"] == 1
        assert stats["campaigns_completed"] == 1
        assert stats["campaigns_total"] == 2


class TestGetBudgetSummary:
    def test_total(
        self, dashboard: RalphDashboard, sample_campaign: dict, completed_campaign: dict
    ) -> None:
        budget = dashboard.get_budget_summary()
        assert budget["total_spent_usd"] == 10.75
        assert len(budget["per_campaign"]) == 2

    def test_single_campaign(self, dashboard: RalphDashboard, sample_campaign: dict) -> None:
        budget = dashboard.get_budget_summary("camp-alpha")
        assert budget["total_spent_usd"] == 3.50


class TestGetPrGateStatus:
    def test_with_active_pr(self, dashboard: RalphDashboard, sample_campaign: dict) -> None:
        gate = dashboard.get_pr_gate_status("camp-alpha")
        assert gate is not None
        assert gate["has_active_pr"] is True
        assert gate["disposition"] == "wait_for_required_checks"
        assert "lint" in gate["checks_passed"]

    def test_without_active_pr(self, dashboard: RalphDashboard, completed_campaign: dict) -> None:
        gate = dashboard.get_pr_gate_status("camp-beta")
        assert gate is not None
        assert gate["has_active_pr"] is False


class TestGetOverview:
    def test_overview(
        self, dashboard: RalphDashboard, sample_campaign: dict, completed_campaign: dict
    ) -> None:
        overview = dashboard.get_overview()
        assert overview["total_campaigns"] == 2
        assert overview["by_status"]["running"] == 1
        assert overview["by_status"]["completed"] == 1
        assert overview["blockers"]["total"] == 3
        assert overview["budget"]["total_spent_usd"] == 10.75
