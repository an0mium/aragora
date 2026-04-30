"""Unit tests for ``aragora.swarm.unstick``.

Tests cover:

- The three actions: ``unstick`` (merged PR exists, issue still open),
  ``close`` (issue already closed/merged on GitHub), and ``hold``
  (truly stuck or only open in-flight PRs).
- Robustness against malformed records.
- The summary helper and the markdown renderer.
"""

from __future__ import annotations

import json
from pathlib import Path

from aragora.swarm.unstick import (
    STUCK_LABEL,
    UnstickRecommendation,
    plan_unstick,
    render_markdown,
    summarize_plan,
)


def _issue(
    number: int,
    *,
    state: str = "OPEN",
    labels: list[str] | None = None,
) -> dict:
    return {
        "number": number,
        "state": state,
        "labels": [{"name": n} for n in (labels or [STUCK_LABEL])],
    }


def _pr(number: int, *, state: str, branch: str) -> dict:
    return {"number": number, "state": state, "headRefName": branch}


class TestPlanUnstickActions:
    def test_merged_pr_open_issue_yields_unstick(self) -> None:
        issues = [_issue(5126)]
        prs = [_pr(5163, state="MERGED", branch="aragora/boss-harvest/issue-5126-boss-x")]
        plan = plan_unstick(stuck_issue_records=issues, pr_records=prs)
        assert len(plan) == 1
        assert plan[0].issue_number == 5126
        assert plan[0].action == "unstick"
        assert plan[0].evidence["merged_pr_numbers"] == [5163]

    def test_closed_issue_yields_close(self) -> None:
        issues = [_issue(7, state="CLOSED")]
        prs = []
        plan = plan_unstick(stuck_issue_records=issues, pr_records=prs)
        assert plan[0].action == "close"

    def test_merged_issue_state_yields_close(self) -> None:
        issues = [_issue(7, state="MERGED")]
        plan = plan_unstick(stuck_issue_records=issues, pr_records=[])
        assert plan[0].action == "close"

    def test_open_pr_only_yields_hold(self) -> None:
        issues = [_issue(8)]
        prs = [_pr(101, state="OPEN", branch="aragora/boss-harvest/issue-8-foo")]
        plan = plan_unstick(stuck_issue_records=issues, pr_records=prs)
        assert plan[0].action == "hold"
        assert plan[0].evidence["open_pr_numbers"] == [101]

    def test_no_pr_evidence_yields_hold(self) -> None:
        issues = [_issue(9)]
        plan = plan_unstick(stuck_issue_records=issues, pr_records=[])
        assert plan[0].action == "hold"
        assert plan[0].evidence["best_pr_state"] is None

    def test_closed_unmerged_pr_does_not_unstick(self) -> None:
        issues = [_issue(10)]
        prs = [_pr(102, state="CLOSED", branch="aragora/boss-harvest/issue-10-foo")]
        plan = plan_unstick(stuck_issue_records=issues, pr_records=prs)
        assert plan[0].action == "hold"


class TestPlanUnstickFiltering:
    def test_issue_without_stuck_label_excluded(self) -> None:
        issues = [_issue(11, labels=["enhancement"]), _issue(12)]
        plan = plan_unstick(stuck_issue_records=issues, pr_records=[])
        assert len(plan) == 1
        assert plan[0].issue_number == 12

    def test_duplicate_issue_records_deduped(self) -> None:
        issues = [_issue(13), _issue(13)]
        plan = plan_unstick(stuck_issue_records=issues, pr_records=[])
        assert len(plan) == 1

    def test_malformed_records_skipped(self) -> None:
        issues = [
            "not-a-dict",
            None,
            {"number": "abc", "state": "OPEN", "labels": [{"name": STUCK_LABEL}]},
            {"number": -5, "state": "OPEN", "labels": [{"name": STUCK_LABEL}]},
            _issue(14),
        ]
        plan = plan_unstick(stuck_issue_records=issues, pr_records=[])
        assert len(plan) == 1
        assert plan[0].issue_number == 14

    def test_string_labels_accepted(self) -> None:
        issues = [{"number": 15, "state": "OPEN", "labels": [STUCK_LABEL, "other"]}]
        plan = plan_unstick(stuck_issue_records=issues, pr_records=[])
        assert len(plan) == 1
        assert plan[0].issue_number == 15

    def test_results_sorted_by_issue_number(self) -> None:
        issues = [_issue(20), _issue(10), _issue(15)]
        plan = plan_unstick(stuck_issue_records=issues, pr_records=[])
        assert [r.issue_number for r in plan] == [10, 15, 20]


class TestSummarizePlan:
    def test_summary_counts(self) -> None:
        plan = [
            UnstickRecommendation(1, "unstick", "x", {}),
            UnstickRecommendation(2, "unstick", "x", {}),
            UnstickRecommendation(3, "close", "x", {}),
            UnstickRecommendation(4, "hold", "x", {}),
        ]
        s = summarize_plan(plan)
        assert s["total"] == 4
        assert s["by_action"] == {"unstick": 2, "close": 1, "hold": 1}
        assert s["by_action_issue_numbers"]["unstick"] == [1, 2]

    def test_summary_empty(self) -> None:
        s = summarize_plan([])
        assert s["total"] == 0

    def test_summary_skips_invalid(self) -> None:
        plan = [
            UnstickRecommendation(1, "unstick", "x", {}),
            UnstickRecommendation(2, "bogus", "x", {}),
        ]
        s = summarize_plan(plan)
        assert s["total"] == 1


class TestRenderMarkdown:
    def test_renders_summary_and_rows(self) -> None:
        plan = [
            UnstickRecommendation(
                1,
                "unstick",
                "merged PR #100",
                {
                    "issue_state": "OPEN",
                    "best_pr_state": "MERGED",
                    "merged_pr_numbers": [100],
                    "open_pr_numbers": [],
                },
            ),
            UnstickRecommendation(
                2,
                "hold",
                "no PR",
                {
                    "issue_state": "OPEN",
                    "best_pr_state": None,
                    "merged_pr_numbers": [],
                    "open_pr_numbers": [],
                },
            ),
        ]
        md = render_markdown(plan)
        assert "Boss-loop unstick plan (dry-run)" in md
        assert "Total stuck issues considered: **2**" in md
        assert "| #1 | `unstick` |" in md
        assert "| #2 | `hold` |" in md
        assert "_This plan is dry-run only" in md

    def test_renders_empty_plan(self) -> None:
        md = render_markdown([])
        assert "Total stuck issues considered: **0**" in md


class TestEndToEndCli:
    def test_cli_emits_expected_json(self, tmp_path: Path) -> None:
        # Just verify the script's main() returns 0 and produces parseable JSON.
        import sys

        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root / "scripts"))
        try:
            import boss_loop_unstick_plan as cli  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)

        issues_path = tmp_path / "issues.json"
        prs_path = tmp_path / "prs.json"
        out_path = tmp_path / "out.json"
        issues_path.write_text(json.dumps([_issue(99)]))
        prs_path.write_text(
            json.dumps([_pr(101, state="MERGED", branch="aragora/boss-harvest/issue-99-x")])
        )
        rc = cli.main(
            [
                "--issues",
                str(issues_path),
                "--pr-records",
                str(prs_path),
                "--format",
                "json",
                "--output",
                str(out_path),
            ]
        )
        assert rc == 0
        payload = json.loads(out_path.read_text())
        assert payload["summary"]["total"] == 1
        assert payload["recommendations"][0]["action"] == "unstick"
