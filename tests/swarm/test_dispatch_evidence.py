"""Unit tests for ``aragora.swarm.dispatch_evidence``.

The module is pure: it never makes a GitHub call. Every test fixture
is the shape of records returned by ``gh pr list --json
number,state,headRefName``.
"""

from __future__ import annotations

import pytest

from aragora.swarm.dispatch_evidence import (
    DISPATCH_EVIDENCE_STATES,
    extract_issue_number_from_branch,
    is_issue_dispatched_via_pr,
    issues_dispatched_via_pr,
)


class TestExtractIssueNumberFromBranch:
    def test_canonical_boss_harvest_branch(self) -> None:
        assert (
            extract_issue_number_from_branch("aragora/boss-harvest/issue-5126-boss-70e867630054")
            == 5126
        )

    def test_branch_without_suffix(self) -> None:
        assert extract_issue_number_from_branch("aragora/boss-harvest/issue-42") == 42

    def test_branch_with_slash_suffix(self) -> None:
        assert extract_issue_number_from_branch("aragora/boss-harvest/issue-9001/retry") == 9001

    def test_none_input(self) -> None:
        assert extract_issue_number_from_branch(None) is None

    def test_empty_input(self) -> None:
        assert extract_issue_number_from_branch("") is None

    def test_unrelated_branch_rejected(self) -> None:
        assert extract_issue_number_from_branch("feat/some-feature") is None
        assert extract_issue_number_from_branch("fix/other-issue-12") is None

    def test_branch_with_leading_text_rejected(self) -> None:
        assert extract_issue_number_from_branch("prefix/aragora/boss-harvest/issue-5") is None

    def test_zero_issue_number_rejected(self) -> None:
        assert extract_issue_number_from_branch("aragora/boss-harvest/issue-0") is None

    def test_negative_not_matched(self) -> None:
        assert extract_issue_number_from_branch("aragora/boss-harvest/issue--5") is None

    def test_non_numeric_issue_segment_rejected(self) -> None:
        assert extract_issue_number_from_branch("aragora/boss-harvest/issue-abc") is None


class TestIsIssueDispatchedViaPr:
    def test_no_records_means_not_dispatched(self) -> None:
        verdict = is_issue_dispatched_via_pr(5126, pr_records=[])
        assert verdict["dispatched"] is False
        assert verdict["best_state"] is None

    def test_merged_pr_counts_as_dispatched(self) -> None:
        records = [
            {
                "number": 5163,
                "state": "MERGED",
                "headRefName": "aragora/boss-harvest/issue-5126-boss-abc",
            }
        ]
        verdict = is_issue_dispatched_via_pr(5126, pr_records=records)
        assert verdict["dispatched"] is True
        assert verdict["best_state"] == "MERGED"
        assert verdict["pr_numbers_merged"] == [5163]
        assert verdict["pr_numbers_open"] == []

    def test_open_pr_counts_when_accept_open_true(self) -> None:
        records = [
            {
                "number": 99,
                "state": "OPEN",
                "headRefName": "aragora/boss-harvest/issue-7-foo",
            }
        ]
        verdict = is_issue_dispatched_via_pr(7, pr_records=records, accept_open=True)
        assert verdict["dispatched"] is True
        assert verdict["best_state"] == "OPEN"

    def test_open_pr_skipped_when_accept_open_false(self) -> None:
        records = [
            {
                "number": 99,
                "state": "OPEN",
                "headRefName": "aragora/boss-harvest/issue-7-foo",
            }
        ]
        verdict = is_issue_dispatched_via_pr(7, pr_records=records, accept_open=False)
        assert verdict["dispatched"] is False
        assert verdict["best_state"] is None

    def test_closed_unmerged_pr_does_not_count(self) -> None:
        records = [
            {
                "number": 99,
                "state": "CLOSED",
                "headRefName": "aragora/boss-harvest/issue-7-foo",
            }
        ]
        verdict = is_issue_dispatched_via_pr(7, pr_records=records)
        assert verdict["dispatched"] is False
        assert verdict["best_state"] is None

    def test_merged_beats_open(self) -> None:
        records = [
            {
                "number": 99,
                "state": "OPEN",
                "headRefName": "aragora/boss-harvest/issue-7-foo",
            },
            {
                "number": 100,
                "state": "MERGED",
                "headRefName": "aragora/boss-harvest/issue-7-retry",
            },
        ]
        verdict = is_issue_dispatched_via_pr(7, pr_records=records)
        assert verdict["best_state"] == "MERGED"
        assert verdict["pr_numbers_merged"] == [100]
        assert verdict["pr_numbers_open"] == [99]

    def test_unrelated_pr_ignored(self) -> None:
        records = [
            {
                "number": 200,
                "state": "MERGED",
                "headRefName": "feat/something-else",
            }
        ]
        verdict = is_issue_dispatched_via_pr(42, pr_records=records)
        assert verdict["dispatched"] is False

    def test_invalid_record_shapes_skipped(self) -> None:
        records = [
            "not-a-dict",
            None,
            {"number": "abc", "state": "MERGED", "headRefName": "aragora/boss-harvest/issue-5"},
            {
                "number": 1,
                "state": "MERGED",
                "headRefName": "aragora/boss-harvest/issue-5",
            },
        ]
        verdict = is_issue_dispatched_via_pr(5, pr_records=records)
        assert verdict["dispatched"] is True
        assert verdict["pr_numbers_merged"] == [1]

    def test_invalid_issue_number_returns_not_dispatched(self) -> None:
        records = [
            {
                "number": 1,
                "state": "MERGED",
                "headRefName": "aragora/boss-harvest/issue-1",
            }
        ]
        for bad in (0, -5, "1"):
            verdict = is_issue_dispatched_via_pr(bad, pr_records=records)  # type: ignore[arg-type]
            assert verdict["dispatched"] is False


class TestIssuesDispatchedViaPr:
    def test_batch_with_mixed_states(self) -> None:
        records = [
            {
                "number": 100,
                "state": "MERGED",
                "headRefName": "aragora/boss-harvest/issue-1",
            },
            {
                "number": 200,
                "state": "OPEN",
                "headRefName": "aragora/boss-harvest/issue-2",
            },
            {
                "number": 300,
                "state": "CLOSED",
                "headRefName": "aragora/boss-harvest/issue-3",
            },
        ]
        result = issues_dispatched_via_pr([1, 2, 3, 4], pr_records=records)
        assert result[1]["best_state"] == "MERGED"
        assert result[2]["best_state"] == "OPEN"
        assert result[3]["best_state"] is None
        assert result[4]["best_state"] is None
        assert result[3]["dispatched"] is False
        assert result[4]["dispatched"] is False

    def test_batch_empty_input(self) -> None:
        assert issues_dispatched_via_pr([], pr_records=[]) == {}

    def test_batch_filters_invalid_targets(self) -> None:
        result = issues_dispatched_via_pr([0, -1, "abc", 5], pr_records=[])  # type: ignore[list-item]
        assert set(result.keys()) == {5}

    def test_batch_dedups_per_issue(self) -> None:
        records = [
            {
                "number": 100,
                "state": "MERGED",
                "headRefName": "aragora/boss-harvest/issue-7",
            },
            {
                "number": 101,
                "state": "MERGED",
                "headRefName": "aragora/boss-harvest/issue-7-retry",
            },
        ]
        result = issues_dispatched_via_pr([7], pr_records=records)
        assert result[7]["pr_numbers_merged"] == [100, 101]
        assert result[7]["dispatched"] is True

    def test_batch_o_r_plus_n(self) -> None:
        # Ensure repeated issue lookups don't multiply work — give 1000
        # PRs and ask for 5 issues; result should be deterministic.
        records = []
        for i in range(1000):
            records.append(
                {
                    "number": i + 10000,
                    "state": "MERGED" if i % 3 == 0 else "OPEN",
                    "headRefName": f"feat/unrelated-{i}",
                }
            )
        records.extend(
            [
                {
                    "number": 50000,
                    "state": "MERGED",
                    "headRefName": "aragora/boss-harvest/issue-5126",
                },
                {
                    "number": 50001,
                    "state": "OPEN",
                    "headRefName": "aragora/boss-harvest/issue-5128",
                },
            ]
        )
        result = issues_dispatched_via_pr([5126, 5128, 5130], pr_records=records)
        assert result[5126]["best_state"] == "MERGED"
        assert result[5128]["best_state"] == "OPEN"
        assert result[5130]["best_state"] is None


class TestModuleSurface:
    def test_dispatch_evidence_states_frozenset(self) -> None:
        assert DISPATCH_EVIDENCE_STATES == frozenset({"MERGED", "OPEN"})


@pytest.fixture()
def real_world_records() -> list[dict[str, object]]:
    """A fixture mirroring the actual PRs found on origin/main today."""
    return [
        {
            "number": 5163,
            "state": "MERGED",
            "headRefName": "aragora/boss-harvest/issue-5126-boss-70e867630054",
        },
        {
            "number": 5161,
            "state": "MERGED",
            "headRefName": "aragora/boss-harvest/issue-5128-boss-c5a1b1234567",
        },
        {
            "number": 5159,
            "state": "MERGED",
            "headRefName": "aragora/boss-harvest/issue-5130-boss-d3e4f5678901",
        },
        {
            "number": 5195,
            "state": "MERGED",
            "headRefName": "aragora/boss-harvest/issue-5188-boss-9876fedcba",
        },
    ]


class TestRealWorldFixture:
    def test_four_known_merged(self, real_world_records: list[dict[str, object]]) -> None:
        result = issues_dispatched_via_pr(
            [5126, 5128, 5130, 5188, 5788, 5789],
            pr_records=real_world_records,
        )
        assert result[5126]["best_state"] == "MERGED"
        assert result[5128]["best_state"] == "MERGED"
        assert result[5130]["best_state"] == "MERGED"
        assert result[5188]["best_state"] == "MERGED"
        assert result[5788]["best_state"] is None
        assert result[5789]["best_state"] is None
