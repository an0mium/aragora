from __future__ import annotations

import json
from pathlib import Path

from scripts.reconcile_b0_pr_truth import (
    IssueMetricsAggregate,
    IssueTruthRecord,
    LinkedPullRequest,
    TruthSummary,
    aggregate_b0_issues,
    extract_pr_numbers_from_issue,
    main,
    reconcile_issue_truth,
    render_table,
    report_to_json,
    summarize_truth,
)


class FakeGitHubTruthClient:
    def __init__(
        self,
        *,
        issues: dict[int, dict],
        prs: dict[int, dict],
        cross_refs: dict[int, list[int]] | None = None,
    ) -> None:
        self.issues = issues
        self.prs = prs
        self.cross_refs = cross_refs or {}

    def get_issue(self, repo: str, number: int) -> dict:
        return self.issues[number]

    def get_pr(self, repo: str, number: int) -> dict:
        return self.prs[number]

    def get_cross_referenced_pr_numbers(self, repo: str, number: int) -> list[int]:
        return list(self.cross_refs.get(number, []))


def _write_metrics(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "boss_metrics.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_aggregate_b0_issues_detects_cohort_tag_and_title() -> None:
    rows = [
        {
            "issue_number": 101,
            "cohort_tag": "B0-cohort",
            "issue_title": "Add tests for alpha.py",
            "terminal_class": "deliverable_pr_created",
            "publish_action": "pr_created",
        },
        {
            "issue_number": 101,
            "cohort_tag": "B0-cohort",
            "issue_title": "Add tests for alpha.py",
            "terminal_class": "blocked_not_dispatch_bounded",
        },
        {
            "issue_number": 102,
            "issue_title": "[B0-cohort] Add tests for beta.py",
            "terminal_class": "rescue_worker_crash",
        },
        {
            "issue_number": 103,
            "issue_title": "Non cohort issue",
            "terminal_class": "deliverable_pr_created",
        },
    ]

    aggregates = aggregate_b0_issues(rows)

    assert [item.issue_number for item in aggregates] == [101, 102]
    assert aggregates[0].proxy_pr_signal is True
    assert aggregates[0].row_count == 2
    assert aggregates[1].had_rescue is True


def test_extract_pr_numbers_from_issue_filters_repo() -> None:
    issue_payload = {
        "comments": [
            {
                "body": (
                    "PR: https://github.com/synaptent/aragora/pull/5107\n"
                    "Other repo: https://github.com/other/repo/pull/99"
                )
            }
        ]
    }

    result = extract_pr_numbers_from_issue("synaptent/aragora", issue_payload)

    assert result == [5107]


def test_reconcile_issue_truth_prefers_merged_pr() -> None:
    aggregate = IssueMetricsAggregate(
        issue_number=5102,
        title="[B0-cohort] Add tests for protocols.py",
        row_count=2,
        proxy_pr_signal=True,
        had_rescue=False,
    )
    client = FakeGitHubTruthClient(
        issues={
            5102: {
                "title": aggregate.title,
                "comments": [
                    {
                        "body": (
                            "PR: https://github.com/synaptent/aragora/pull/5107\n"
                            "PR: https://github.com/synaptent/aragora/pull/5108"
                        )
                    }
                ],
            }
        },
        prs={
            5107: {
                "number": 5107,
                "title": "draft pr",
                "url": "https://github.com/synaptent/aragora/pull/5107",
                "state": "OPEN",
                "mergeable": "UNKNOWN",
                "mergeStateStatus": "UNKNOWN",
                "mergedAt": None,
                "isDraft": True,
            },
            5108: {
                "number": 5108,
                "title": "merged pr",
                "url": "https://github.com/synaptent/aragora/pull/5108",
                "state": "MERGED",
                "mergeable": "MERGEABLE",
                "mergeStateStatus": "CLEAN",
                "mergedAt": "2026-04-12T20:40:00Z",
                "isDraft": False,
            },
        },
    )

    record = reconcile_issue_truth("synaptent/aragora", aggregate, client)

    assert record.truth_state == "merged_pr"
    assert record.truth_success is True
    assert record.no_rescue_truth_success is True
    assert [pr.number for pr in record.linked_prs] == [5107, 5108]


def test_reconcile_issue_truth_falls_back_to_cross_refs() -> None:
    aggregate = IssueMetricsAggregate(
        issue_number=5103,
        title="[B0-cohort] Add tests for connector.py",
        proxy_pr_signal=False,
        had_rescue=True,
    )
    client = FakeGitHubTruthClient(
        issues={5103: {"title": aggregate.title, "comments": []}},
        prs={
            5111: {
                "number": 5111,
                "title": "active pr",
                "url": "https://github.com/synaptent/aragora/pull/5111",
                "state": "OPEN",
                "mergeable": "MERGEABLE",
                "mergeStateStatus": "CLEAN",
                "mergedAt": None,
                "isDraft": False,
            }
        },
        cross_refs={5103: [5111]},
    )

    record = reconcile_issue_truth("synaptent/aragora", aggregate, client)

    assert record.truth_state == "mergeable_pr"
    assert record.truth_success is True
    assert record.no_rescue_truth_success is False


def test_reconcile_issue_truth_preserves_closed_unmerged_pr() -> None:
    aggregate = IssueMetricsAggregate(
        issue_number=5104,
        title="[B0-cohort] Closed PR attempt",
        proxy_pr_signal=False,
        had_rescue=False,
    )
    client = FakeGitHubTruthClient(
        issues={
            5104: {
                "title": aggregate.title,
                "comments": [{"body": "PR: https://github.com/synaptent/aragora/pull/5112"}],
            }
        },
        prs={
            5112: {
                "number": 5112,
                "title": "closed attempt",
                "url": "https://github.com/synaptent/aragora/pull/5112",
                "state": "CLOSED",
                "mergeable": "CONFLICTING",
                "mergeStateStatus": "DIRTY",
                "mergedAt": None,
                "isDraft": False,
            }
        },
    )

    record = reconcile_issue_truth("synaptent/aragora", aggregate, client)

    assert record.truth_state == "closed_unmerged_pr"
    assert record.truth_success is False
    assert record.no_rescue_truth_success is False
    assert [pr.truth_state for pr in record.linked_prs] == ["closed_unmerged_pr"]


def test_summary_and_renderers_include_proxy_vs_truth_language() -> None:
    records = [
        IssueTruthRecord(
            issue_number=5098,
            issue_title="[B0-cohort] alpha",
            proxy_pr_signal=True,
            had_rescue=False,
            truth_state="open_pr",
            truth_success=False,
            no_rescue_truth_success=False,
            linked_prs=[
                LinkedPullRequest(
                    number=5105,
                    title="alpha pr",
                    url="https://github.com/synaptent/aragora/pull/5105",
                    state="OPEN",
                    mergeable="UNKNOWN",
                    merge_state_status="UNKNOWN",
                    merged_at=None,
                    is_draft=True,
                )
            ],
        ),
        IssueTruthRecord(
            issue_number=5099,
            issue_title="[B0-cohort] beta",
            proxy_pr_signal=True,
            had_rescue=False,
            truth_state="merged_pr",
            truth_success=True,
            no_rescue_truth_success=True,
            linked_prs=[
                LinkedPullRequest(
                    number=5106,
                    title="beta pr",
                    url="https://github.com/synaptent/aragora/pull/5106",
                    state="MERGED",
                    mergeable="MERGEABLE",
                    merge_state_status="CLEAN",
                    merged_at="2026-04-12T20:41:00Z",
                    is_draft=False,
                )
            ],
        ),
    ]

    summary = summarize_truth(records)
    table = render_table("synaptent/aragora", Path("metrics.jsonl"), records, summary)
    payload = json.loads(
        report_to_json("synaptent/aragora", Path("metrics.jsonl"), records, summary)
    )

    assert summary.proxy_success_issue_count == 2
    assert summary.truth_success_issue_count == 1
    assert summary.truth_state_counts == {"merged_pr": 1, "open_pr": 1}
    assert "Proxy success (proxy): 2/2 (100.0%)" in table
    assert "Mergeable or merged issues (truth success): 1/2 (50.0%)" in table
    assert "#5105:open_pr:UNKNOWN" in table
    assert payload["summary"]["truth_success_issue_count"] == 1
    assert payload["issues"][0]["truth_state"] == "open_pr"


def test_main_json_output_with_mocked_github_client(tmp_path: Path, monkeypatch, capsys) -> None:
    metrics_file = _write_metrics(
        tmp_path,
        [
            {
                "issue_number": 5102,
                "cohort_tag": "B0-cohort",
                "issue_title": "[B0-cohort] Add tests for protocols.py",
                "terminal_class": "deliverable_pr_created",
                "publish_action": "pr_created",
            }
        ],
    )

    client = FakeGitHubTruthClient(
        issues={
            5102: {
                "title": "[B0-cohort] Add tests for protocols.py",
                "comments": [{"body": "PR: https://github.com/synaptent/aragora/pull/5107"}],
            }
        },
        prs={
            5107: {
                "number": 5107,
                "title": "draft pr",
                "url": "https://github.com/synaptent/aragora/pull/5107",
                "state": "OPEN",
                "mergeable": "UNKNOWN",
                "mergeStateStatus": "UNKNOWN",
                "mergedAt": None,
                "isDraft": True,
            }
        },
    )
    monkeypatch.setattr(
        "scripts.reconcile_b0_pr_truth.GitHubTruthClient",
        lambda: client,
    )

    assert main(["--metrics-file", str(metrics_file), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["summary"]["attempted_issue_count"] == 1
    assert payload["summary"]["proxy_success_issue_count"] == 1
    assert payload["summary"]["truth_success_issue_count"] == 0
    assert payload["issues"][0]["issue_number"] == 5102
    assert payload["issues"][0]["truth_state"] == "open_pr"
