from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.reconcile_b0_pr_truth import (
    CROSS_REFS_MAX_PAGES,
    CROSS_REFS_PER_PAGE,
    ISSUE_COMMENTS_MAX_PAGES,
    ISSUE_COMMENTS_PER_PAGE,
    GitHubTruthClient,
    IssueMetricsAggregate,
    IssueTruthRecord,
    LinkedPullRequest,
    TruthSummary,
    aggregate_b0_issues,
    classify_issue_truth_state,
    extract_pr_numbers_from_issue,
    main,
    reconcile_issue_truth,
    render_table,
    report_to_json,
    resolve_metrics_path,
    summarize_truth,
)


class FakeGitHubTruthClient:
    def __init__(
        self,
        *,
        issues: dict[int, dict],
        prs: dict[int, dict],
        cross_refs: dict[int, list[int] | Exception] | None = None,
    ) -> None:
        self.issues = issues
        self.prs = prs
        self.cross_refs = cross_refs or {}

    def get_issue(self, repo: str, number: int) -> dict:
        value = self.issues[number]
        if isinstance(value, Exception):
            raise value
        return value

    def get_pr(self, repo: str, number: int) -> dict:
        return self.prs[number]

    def get_cross_referenced_pr_numbers(self, repo: str, number: int) -> list[int]:
        value = self.cross_refs.get(number, [])
        if isinstance(value, Exception):
            raise value
        return list(value)


class PaginatedGitHubTruthClient(GitHubTruthClient):
    def __init__(
        self,
        *,
        comment_pages: list[list[dict]] | None = None,
        cross_ref_pages: list[dict] | None = None,
    ) -> None:
        self.comment_pages = comment_pages or []
        self.cross_ref_pages = cross_ref_pages or []
        self.comment_calls = 0
        self.cross_ref_calls = 0

    def _run_json_object(self, args: list[str]) -> dict:
        if args[:2] == ["api", "graphql"]:
            payload = self.cross_ref_pages[self.cross_ref_calls]
            self.cross_ref_calls += 1
            return payload
        raise AssertionError(f"unexpected object args: {args}")

    def _run_json_list(self, args: list[str]) -> list[dict]:
        if args and args[0] == "api" and "/comments?" in args[1]:
            payload = self.comment_pages[self.comment_calls]
            self.comment_calls += 1
            return payload
        raise AssertionError(f"unexpected list args: {args}")


class IssueViewGitHubTruthClient(GitHubTruthClient):
    def __init__(self, *, issue_payload: dict, comment_error: Exception) -> None:
        self.issue_payload = issue_payload
        self.comment_error = comment_error

    def _run_json_object(self, args: list[str]) -> dict:
        if args[:2] == ["issue", "view"]:
            return dict(self.issue_payload)
        raise AssertionError(f"unexpected object args: {args}")

    def _run_json_list(self, args: list[str]) -> list[dict]:
        if args and args[0] == "api" and "/comments?" in args[1]:
            raise self.comment_error
        raise AssertionError(f"unexpected list args: {args}")


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


def test_extract_pr_numbers_from_issue_includes_closed_by_pull_references() -> None:
    issue_payload = {
        "closedByPullRequestsReferences": [
            {
                "number": 5763,
                "repository": {
                    "name": "aragora",
                    "owner": {"login": "synaptent"},
                },
            },
            {
                "number": 99,
                "repository": {
                    "name": "repo",
                    "owner": {"login": "other"},
                },
            },
        ],
        "comments": [],
    }

    result = extract_pr_numbers_from_issue("synaptent/aragora", issue_payload)

    assert result == [5763]


def test_extract_pr_numbers_strict_excludes_forensic_reference_prs() -> None:
    # 2026-04-17 honesty audit: forensic-reference PRs (unrelated merged PRs
    # that merely cite the issue in comments) must not count as closure
    # evidence. Strict linkage returns PRs from closedByPullRequestsReferences
    # only — in this fixture, that edge is empty (the classic #873 pattern
    # where the issue was closed manually as stale).
    issue_payload = {
        "closedByPullRequestsReferences": [],
        "comments": [
            {"body": "Forensic case study: https://github.com/synaptent/aragora/pull/880"},
            {"body": "Also see https://github.com/synaptent/aragora/pull/881 and #882"},
        ],
    }

    strict = extract_pr_numbers_from_issue("synaptent/aragora", issue_payload, strict=True)
    lenient = extract_pr_numbers_from_issue("synaptent/aragora", issue_payload)

    assert strict == []
    # The lenient (default) path still returns the comment-derived PRs so the
    # B0 cohort reconciliation flow keeps its current behaviour.
    assert lenient == [880, 881]


def test_extract_pr_numbers_strict_keeps_closed_by_references() -> None:
    issue_payload = {
        "closedByPullRequestsReferences": [
            {
                "number": 5763,
                "repository": {"name": "aragora", "owner": {"login": "synaptent"}},
            }
        ],
        "comments": [
            {"body": "Forensic ref: https://github.com/synaptent/aragora/pull/880"},
        ],
    }

    result = extract_pr_numbers_from_issue("synaptent/aragora", issue_payload, strict=True)

    assert result == [5763]


@pytest.mark.parametrize(
    ("linked_prs", "expected"),
    [
        (
            [
                LinkedPullRequest(
                    number=5107,
                    title="merged pr",
                    url="https://github.com/synaptent/aragora/pull/5107",
                    state="MERGED",
                    mergeable="MERGEABLE",
                    merge_state_status="CLEAN",
                    merged_at="2026-04-12T20:40:00Z",
                    is_draft=False,
                ),
                LinkedPullRequest(
                    number=5108,
                    title="mergeable pr",
                    url="https://github.com/synaptent/aragora/pull/5108",
                    state="OPEN",
                    mergeable="MERGEABLE",
                    merge_state_status="CLEAN",
                    merged_at=None,
                    is_draft=False,
                ),
            ],
            "merged_pr",
        ),
        (
            [
                LinkedPullRequest(
                    number=5111,
                    title="mergeable pr",
                    url="https://github.com/synaptent/aragora/pull/5111",
                    state="OPEN",
                    mergeable="MERGEABLE",
                    merge_state_status="CLEAN",
                    merged_at=None,
                    is_draft=False,
                ),
                LinkedPullRequest(
                    number=5112,
                    title="open pr",
                    url="https://github.com/synaptent/aragora/pull/5112",
                    state="OPEN",
                    mergeable="UNKNOWN",
                    merge_state_status="UNKNOWN",
                    merged_at=None,
                    is_draft=True,
                ),
            ],
            "mergeable_pr",
        ),
        ([], "no_linked_pr"),
    ],
)
def test_classify_issue_truth_state_prioritizes_truth_states(
    linked_prs: list[LinkedPullRequest], expected: str
) -> None:
    assert classify_issue_truth_state(linked_prs) == expected


def test_reconcile_issue_truth_strict_linkage_ignores_forensic_references() -> None:
    # End-to-end: a CLOSED issue with no closedByPullRequestsReferences but
    # a comment forensically referencing a merged PR. Under strict linkage
    # (benchmark truth path), truth_state resolves to no_linked_pr even
    # though a non-strict pass would have falsely reported merged_pr.
    aggregate = IssueMetricsAggregate(
        issue_number=873,
        title="closed as stale",
        row_count=1,
        proxy_pr_signal=False,
        had_rescue=False,
    )
    client = FakeGitHubTruthClient(
        issues={
            873: {
                "title": aggregate.title,
                "state": "CLOSED",
                "stateReason": "COMPLETED",
                "closedAt": "2026-03-19T15:24:06Z",
                "closedByPullRequestsReferences": [],
                "comments": [
                    {"body": "Forensic case study: https://github.com/synaptent/aragora/pull/880"},
                ],
            }
        },
        prs={
            880: {
                "number": 880,
                "title": "unrelated swarm hardening",
                "url": "https://github.com/synaptent/aragora/pull/880",
                "state": "MERGED",
                "mergeable": "UNKNOWN",
                "mergeStateStatus": "UNKNOWN",
                "mergedAt": "2026-03-09T17:12:37Z",
                "isDraft": False,
            }
        },
    )

    record = reconcile_issue_truth("synaptent/aragora", aggregate, client, strict_linkage=True)

    assert record.truth_state == "no_linked_pr"
    assert record.truth_success is False
    assert record.linked_prs == []
    assert record.stale_corpus_issue is True
    assert record.stale_corpus_reason == "closed_without_linked_pr"


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


def test_issue_comment_pagination_collects_multiple_pages() -> None:
    client = PaginatedGitHubTruthClient(
        comment_pages=[
            [{"body": "page1"}] * ISSUE_COMMENTS_PER_PAGE,
            [{"body": "page2"}],
        ]
    )

    comments = client.get_issue_comments("synaptent/aragora", 5102)

    assert len(comments) == ISSUE_COMMENTS_PER_PAGE + 1
    assert client.comment_calls == 2


def test_issue_comment_pagination_raises_when_bound_exceeded() -> None:
    client = PaginatedGitHubTruthClient(
        comment_pages=[
            [{"body": f"page-{idx}"}] * ISSUE_COMMENTS_PER_PAGE
            for idx in range(ISSUE_COMMENTS_MAX_PAGES)
        ]
    )

    with pytest.raises(RuntimeError, match="issue comment pagination exceeded bound"):
        client.get_issue_comments("synaptent/aragora", 5102)


def test_cross_reference_pagination_collects_multiple_pages() -> None:
    client = PaginatedGitHubTruthClient(
        cross_ref_pages=[
            {
                "data": {
                    "repository": {
                        "issue": {
                            "timelineItems": {
                                "nodes": [
                                    {"source": {"__typename": "PullRequest", "number": 5107}}
                                ],
                                "pageInfo": {"hasNextPage": True, "endCursor": "cursor-1"},
                            }
                        }
                    }
                }
            },
            {
                "data": {
                    "repository": {
                        "issue": {
                            "timelineItems": {
                                "nodes": [
                                    {"source": {"__typename": "PullRequest", "number": 5108}}
                                ],
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                            }
                        }
                    }
                }
            },
        ]
    )

    pr_numbers = client.get_cross_referenced_pr_numbers("synaptent/aragora", 5102)

    assert pr_numbers == [5107, 5108]
    assert client.cross_ref_calls == 2


def test_cross_reference_pagination_raises_when_bound_exceeded() -> None:
    client = PaginatedGitHubTruthClient(
        cross_ref_pages=[
            {
                "data": {
                    "repository": {
                        "issue": {
                            "timelineItems": {
                                "nodes": [
                                    {"source": {"__typename": "PullRequest", "number": 6000 + idx}}
                                ],
                                "pageInfo": {"hasNextPage": True, "endCursor": f"cursor-{idx}"},
                            }
                        }
                    }
                }
            }
            for idx in range(CROSS_REFS_MAX_PAGES)
        ]
    )

    with pytest.raises(RuntimeError, match="cross-reference pagination exceeded bound"):
        client.get_cross_referenced_pr_numbers("synaptent/aragora", 5102)


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


def test_reconcile_issue_truth_uses_closed_by_pull_request_references() -> None:
    aggregate = IssueMetricsAggregate(
        issue_number=2712,
        title="Boolean parsing fix",
        proxy_pr_signal=False,
        had_rescue=False,
    )
    client = FakeGitHubTruthClient(
        issues={
            2712: {
                "title": aggregate.title,
                "state": "CLOSED",
                "stateReason": "COMPLETED",
                "closedAt": "2026-04-15T05:27:50Z",
                "closedByPullRequestsReferences": [
                    {
                        "number": 5763,
                        "repository": {
                            "name": "aragora",
                            "owner": {"login": "synaptent"},
                        },
                    }
                ],
                "comments": [],
            }
        },
        prs={
            5763: {
                "number": 5763,
                "title": "merged fix",
                "url": "https://github.com/synaptent/aragora/pull/5763",
                "state": "MERGED",
                "mergeable": "MERGEABLE",
                "mergeStateStatus": "CLEAN",
                "mergedAt": "2026-04-15T05:27:49Z",
                "isDraft": False,
            }
        },
        cross_refs={2712: RuntimeError("cross refs should not be used")},
    )

    record = reconcile_issue_truth("synaptent/aragora", aggregate, client)

    assert record.truth_state == "merged_pr"
    assert record.truth_success is True
    assert record.issue_state == "CLOSED"
    assert [pr.number for pr in record.linked_prs] == [5763]


def test_reconcile_issue_truth_marks_closed_issue_without_linked_pr_as_stale() -> None:
    aggregate = IssueMetricsAggregate(
        issue_number=1733,
        title="Detached worker cleanup",
        proxy_pr_signal=False,
        had_rescue=False,
    )
    client = FakeGitHubTruthClient(
        issues={
            1733: {
                "title": aggregate.title,
                "url": "https://github.com/synaptent/aragora/issues/1733",
                "state": "CLOSED",
                "stateReason": "COMPLETED",
                "closedAt": "2026-03-31T23:45:29Z",
                "closedByPullRequestsReferences": [],
                "comments": [],
            }
        },
        prs={},
        cross_refs={1733: []},
    )

    record = reconcile_issue_truth("synaptent/aragora", aggregate, client)

    assert record.truth_state == "no_linked_pr"
    assert record.issue_state == "CLOSED"
    assert record.stale_corpus_issue is True
    assert record.stale_corpus_reason == "closed_without_linked_pr"


def test_reconcile_issue_truth_skips_stale_flag_when_linkage_lookup_fails() -> None:
    aggregate = IssueMetricsAggregate(
        issue_number=1733,
        title="Detached worker cleanup",
        proxy_pr_signal=False,
        had_rescue=False,
    )
    client = FakeGitHubTruthClient(
        issues={
            1733: {
                "title": aggregate.title,
                "url": "https://github.com/synaptent/aragora/issues/1733",
                "state": "CLOSED",
                "stateReason": "COMPLETED",
                "closedAt": "2026-03-31T23:45:29Z",
                "closedByPullRequestsReferences": [],
                "comments": [],
            }
        },
        prs={},
        cross_refs={1733: RuntimeError("error connecting to api.github.com")},
    )

    record = reconcile_issue_truth("synaptent/aragora", aggregate, client)

    assert record.truth_state == "no_linked_pr"
    assert record.issue_state == "CLOSED"
    assert record.linkage_status == "cross_reference_lookup_failed"
    assert record.linkage_verification_incomplete is True
    assert record.linkage_error == "error connecting to api.github.com"
    assert record.stale_corpus_issue is False
    assert record.stale_corpus_reason is None


def test_get_issue_tolerates_issue_comment_lookup_failure() -> None:
    client = IssueViewGitHubTruthClient(
        issue_payload={
            "number": 873,
            "title": "ESLint bump",
            "url": "https://github.com/synaptent/aragora/issues/873",
            "state": "CLOSED",
            "stateReason": "COMPLETED",
            "closedAt": "2026-03-09T17:12:37Z",
            "closedByPullRequestsReferences": [],
        },
        comment_error=RuntimeError("error connecting to api.github.com"),
    )

    payload = client.get_issue("synaptent/aragora", 873)

    assert payload["comments"] == []
    assert payload["_comments_lookup_error"] == "error connecting to api.github.com"


def test_reconcile_issue_truth_marks_comment_lookup_failure_incomplete() -> None:
    aggregate = IssueMetricsAggregate(
        issue_number=873,
        title="ESLint bump",
        proxy_pr_signal=False,
        had_rescue=False,
    )
    client = FakeGitHubTruthClient(
        issues={
            873: {
                "title": aggregate.title,
                "url": "https://github.com/synaptent/aragora/issues/873",
                "state": "CLOSED",
                "stateReason": "COMPLETED",
                "closedAt": "2026-03-09T17:12:37Z",
                "closedByPullRequestsReferences": [],
                "comments": [],
                "_comments_lookup_error": "error connecting to api.github.com",
            }
        },
        prs={},
        cross_refs={873: []},
    )

    record = reconcile_issue_truth("synaptent/aragora", aggregate, client)

    assert record.truth_state == "no_linked_pr"
    assert record.linkage_status == "issue_comments_lookup_failed"
    assert record.linkage_verification_incomplete is True
    assert record.linkage_error == "error connecting to api.github.com"
    assert record.stale_corpus_issue is False


def test_reconcile_issue_truth_tolerates_issue_lookup_failure() -> None:
    aggregate = IssueMetricsAggregate(
        issue_number=873,
        title="ESLint bump",
        proxy_pr_signal=True,
        had_rescue=False,
    )
    client = FakeGitHubTruthClient(
        issues={873: RuntimeError("error connecting to api.github.com")},
        prs={},
    )

    record = reconcile_issue_truth("synaptent/aragora", aggregate, client)

    assert record.issue_number == 873
    assert record.issue_title == "ESLint bump"
    assert record.truth_state == "no_linked_pr"
    assert record.proxy_pr_signal is True
    assert record.linkage_status == "issue_lookup_failed"
    assert record.linkage_verification_incomplete is True
    assert record.linkage_error == "error connecting to api.github.com"
    assert record.stale_corpus_issue is False


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


def test_resolve_metrics_path_falls_back_to_git_common_root(tmp_path: Path, monkeypatch) -> None:
    shared_root = tmp_path / "shared-root"
    metrics_file = shared_root / ".aragora" / "overnight" / "boss_metrics.jsonl"
    metrics_file.parent.mkdir(parents=True)
    metrics_file.write_text("", encoding="utf-8")

    monkeypatch.setattr("scripts.reconcile_b0_pr_truth.REPO_ROOT", tmp_path / "worktree-root")
    monkeypatch.setattr(
        "scripts.reconcile_b0_pr_truth._git_common_repo_root",
        lambda: shared_root,
    )

    resolved = resolve_metrics_path(Path(".aragora/overnight/boss_metrics.jsonl"))

    assert resolved == metrics_file.resolve()


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
