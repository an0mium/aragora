"""Tests for triage profile benchmark comparison logic."""

from __future__ import annotations

from pathlib import Path

from aragora.inbox.triage_profile_benchmark import (
    ProfileRunResult,
    compare_profile_runs,
    load_fixture_messages,
)


def _make_run(
    *,
    profile: str,
    decisions: list[dict[str, object]],
) -> ProfileRunResult:
    return ProfileRunResult(
        profile=profile,
        fixture_path="fixtures.json",
        message_count=len(decisions),
        total_duration_seconds=1.0,
        diagnostics_artifact_dir=f"/tmp/{profile}",
        meta={
            "blocked_count": sum(1 for item in decisions if item["blocked_by_policy"]),
            "suppressed_diagnostics_count": sum(
                int(item.get("suppressed_diagnostics_count", 0)) for item in decisions
            ),
            "fast_tier_count": sum(1 for item in decisions if item["execution_tier"] == "fast"),
            "escalated_count": sum(
                1 for item in decisions if item["execution_tier"] == "escalated"
            ),
            "artifact_dir": f"/tmp/{profile}",
        },
        decisions=decisions,
    )


def test_load_fixture_messages_reads_sample_fixture() -> None:
    fixture = (
        Path(__file__).resolve().parent.parent
        / "fixtures"
        / "inbox"
        / "triage_profile_eval_sample.json"
    )
    messages = load_fixture_messages(fixture)

    assert len(messages) == 4
    assert messages[0]["id"] == "msg-newsletter"
    assert messages[1]["from_address"] == "founder@startup.io"


def test_compare_profile_runs_accepts_clean_staged_result() -> None:
    baseline = _make_run(
        profile="baseline",
        decisions=[
            {
                "message_id": "m1",
                "subject": "One",
                "final_action": "archive",
                "blocked_by_policy": False,
                "execution_tier": "baseline",
                "latency_seconds": 10.0,
                "auto_approval_candidate": True,
                "suppressed_diagnostics_count": 0,
            },
            {
                "message_id": "m2",
                "subject": "Two",
                "final_action": "ignore",
                "blocked_by_policy": False,
                "execution_tier": "baseline",
                "latency_seconds": 8.0,
                "auto_approval_candidate": True,
                "suppressed_diagnostics_count": 0,
            },
        ],
    )
    staged = _make_run(
        profile="staged_v1",
        decisions=[
            {
                "message_id": "m1",
                "subject": "One",
                "final_action": "archive",
                "blocked_by_policy": False,
                "execution_tier": "fast",
                "latency_seconds": 4.0,
                "auto_approval_candidate": True,
                "suppressed_diagnostics_count": 0,
            },
            {
                "message_id": "m2",
                "subject": "Two",
                "final_action": "ignore",
                "blocked_by_policy": False,
                "execution_tier": "fast",
                "latency_seconds": 4.0,
                "auto_approval_candidate": True,
                "suppressed_diagnostics_count": 0,
            },
        ],
    )

    comparison = compare_profile_runs(baseline, staged)

    assert comparison["agreement_rate"] == 1.0
    assert comparison["latency_improvement_pct"] == 55.56
    assert comparison["blocked_rate_delta_pp"] == 0.0
    assert comparison["unsafe_auto_approval_ids"] == []
    assert comparison["passes_all_thresholds"] is True


def test_compare_profile_runs_flags_unsafe_and_regressed_staged_result() -> None:
    baseline = _make_run(
        profile="baseline",
        decisions=[
            {
                "message_id": "m1",
                "subject": "One",
                "final_action": "archive",
                "blocked_by_policy": True,
                "execution_tier": "baseline",
                "latency_seconds": 10.0,
                "auto_approval_candidate": False,
                "suppressed_diagnostics_count": 0,
            },
            {
                "message_id": "m2",
                "subject": "Two",
                "final_action": "ignore",
                "blocked_by_policy": False,
                "execution_tier": "baseline",
                "latency_seconds": 7.0,
                "auto_approval_candidate": True,
                "suppressed_diagnostics_count": 0,
            },
        ],
    )
    staged = _make_run(
        profile="staged_v1",
        decisions=[
            {
                "message_id": "m1",
                "subject": "One",
                "final_action": "archive",
                "blocked_by_policy": False,
                "execution_tier": "fast",
                "latency_seconds": 9.5,
                "auto_approval_candidate": True,
                "suppressed_diagnostics_count": 0,
            },
            {
                "message_id": "m2",
                "subject": "Two",
                "final_action": "star",
                "blocked_by_policy": False,
                "execution_tier": "fast",
                "latency_seconds": 6.5,
                "auto_approval_candidate": True,
                "suppressed_diagnostics_count": 0,
            },
        ],
    )

    comparison = compare_profile_runs(baseline, staged)

    assert comparison["agreement_count"] == 0
    assert comparison["unsafe_auto_approval_ids"] == ["m1"]
    assert comparison["blocked_rate_delta_pp"] == -50.0
    assert comparison["passes_all_thresholds"] is False
