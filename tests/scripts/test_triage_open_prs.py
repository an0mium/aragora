"""Tests for ``scripts/triage_open_prs.py``.

Fixture-driven; never calls real ``gh``. Each test constructs a
synthetic PR dict matching the ``gh pr list --json
number,title,isDraft,author,mergeable,mergeStateStatus,additions,
deletions,changedFiles,createdAt,updatedAt,headRefName,
headRefOid,statusCheckRollup,reviewDecision,labels,files`` shape and asserts
the classifier puts it in the expected bucket.

Coverage targets every Bucket A/B/C path + every tripwire from
``docs/governance/OPERATOR_DELEGATION_POLICY.md``.
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module() -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "triage_open_prs.py"
    spec = importlib.util.spec_from_file_location("triage_open_prs_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tri = _load_module()


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


NOW = datetime.datetime(2026, 5, 17, 18, 0, 0, tzinfo=datetime.timezone.utc)


def _days_ago(d: int) -> str:
    return (NOW - datetime.timedelta(days=d)).isoformat().replace("+00:00", "Z")


def make_merge_packet(
    number: int,
    head_sha: str,
    *,
    admin_squash_allowed: bool = True,
    not_ready: list[int] | None = None,
    unresolved_dissent: bool = False,
    tier: int = 2,
    requires_human_risk_settlement: bool = False,
) -> dict[str, Any]:
    return {
        "not_ready": [] if not_ready is None else not_ready,
        "entries": [
            {
                "pr_number": number,
                "head_sha": head_sha,
                "admin_squash_allowed": admin_squash_allowed,
                "unresolved_dissent": unresolved_dissent,
                "tier": tier,
                "requires_human_risk_settlement": requires_human_risk_settlement,
            }
        ],
    }


def make_pr(
    number: int = 9000,
    *,
    title: str = "test PR",
    author: str = "an0mium",
    files: list[dict[str, Any]] | None = None,
    additions: int = 100,
    deletions: int = 0,
    mergeable: str = "MERGEABLE",
    merge_state: str = "CLEAN",
    # Default to NOT draft so the happy-path Bucket A tests still
    # qualify after the policy tightening that excludes drafts from A.
    # Tests that want a draft pass is_draft=True explicitly.
    is_draft: bool = False,
    ci: list[dict[str, Any]] | None = None,
    review: str = "",
    created_days_ago: int = 0,
    updated_days_ago: int = 0,
    head_sha: str | None = None,
    merge_packet: dict[str, Any] | None | bool = True,
) -> dict[str, Any]:
    if head_sha is None:
        head_sha = f"{number:040x}"[-40:]
    if files is None:
        files = [
            {"path": "scripts/some_helper.py", "additions": additions, "deletions": 0},
            {"path": "tests/scripts/test_some_helper.py", "additions": 50, "deletions": 0},
        ]
    if ci is None:
        ci = [
            {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
            {"name": "tests", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ]
    pr = {
        "number": number,
        "title": title,
        "isDraft": is_draft,
        "author": {"login": author},
        "mergeable": mergeable,
        "mergeStateStatus": merge_state,
        "additions": additions,
        "deletions": deletions,
        "changedFiles": len(files),
        "createdAt": _days_ago(created_days_ago),
        "updatedAt": _days_ago(updated_days_ago),
        "headRefName": f"branch-{number}",
        "headRefOid": head_sha,
        "statusCheckRollup": ci,
        "reviewDecision": review,
        "labels": [],
        "files": files,
    }
    if merge_packet is True:
        pr["mergePacket"] = make_merge_packet(number, head_sha)
    elif isinstance(merge_packet, dict):
        pr["mergePacket"] = merge_packet
    return pr


def classify(pr: dict[str, Any], all_open: list[dict[str, Any]] | None = None):
    return tri.classify(pr, all_open or [pr], now=NOW)


# ---------------------------------------------------------------------------
# Bucket A — the default happy path
# ---------------------------------------------------------------------------


class TestBucketA:
    def test_clean_additive_with_tests_ready(self):
        # NOT-DRAFT + CLEAN + green CI + tests + trusted author +
        # exact-head merge-packet authorization = A.
        pr = make_pr(number=9001, is_draft=False, merge_state="CLEAN")
        r = classify(pr)
        assert r.bucket == tri.BUCKET_A
        assert r.recommended_action == "MERGE"
        assert "green CI" in r.reason
        assert "merge-packet authorized" in r.reason

    def test_draft_goes_to_c_not_a(self):
        # Policy: drafts must NEVER auto-merge. Same clean PR but
        # is_draft=True → Bucket C with "READY?" recommendation.
        pr = make_pr(number=9002, is_draft=True, merge_state="BLOCKED")
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert r.recommended_action == "READY?"
        assert "draft" in r.reason

    def test_blocked_merge_state_goes_to_c(self):
        # BLOCKED without review-required context is not the policy's
        # review-only/admin-squash exception.
        pr = make_pr(number=9003, is_draft=False, merge_state="BLOCKED")
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "merge state status: BLOCKED" in r.reason

    def test_review_required_blocked_with_admin_packet_can_be_a(self):
        # Policy: CLEAN is not the only A path. Review-only branch
        # protection can qualify when the exact-head merge packet
        # authorizes admin squash.
        pr = make_pr(
            number=9010,
            is_draft=False,
            merge_state="BLOCKED",
            review="REVIEW_REQUIRED",
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_A
        assert "merge-packet authorized" in r.reason

    def test_missing_merge_packet_goes_to_c(self):
        pr = make_pr(number=9004, is_draft=False, merge_state="CLEAN", merge_packet=None)
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "merge-packet not checked" in r.reason

    def test_merge_packet_not_ready_goes_to_c(self):
        head_sha = "1" * 40
        pr = make_pr(
            number=9005,
            head_sha=head_sha,
            merge_packet=make_merge_packet(9005, head_sha, not_ready=[9005]),
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "not_ready" in r.reason

    def test_merge_packet_admin_false_goes_to_c(self):
        head_sha = "2" * 40
        pr = make_pr(
            number=9006,
            head_sha=head_sha,
            merge_packet=make_merge_packet(9006, head_sha, admin_squash_allowed=False),
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "admin_squash_allowed" in r.reason

    def test_merge_packet_head_mismatch_goes_to_c(self):
        pr = make_pr(
            number=9007,
            head_sha="3" * 40,
            merge_packet=make_merge_packet(9007, "4" * 40),
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "head mismatch" in r.reason

    def test_tier3_without_settlement_goes_to_c(self):
        head_sha = "5" * 40
        pr = make_pr(
            number=9008,
            head_sha=head_sha,
            merge_packet=make_merge_packet(
                9008,
                head_sha,
                tier=3,
                requires_human_risk_settlement=True,
            ),
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "Tier 3/4" in r.reason

    def test_tier3_with_settlement_can_be_a(self):
        head_sha = "6" * 40
        pr = make_pr(
            number=9009,
            head_sha=head_sha,
            merge_packet=make_merge_packet(
                9009,
                head_sha,
                tier=3,
                requires_human_risk_settlement=False,
            ),
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_A


# ---------------------------------------------------------------------------
# Bucket C — tripwires (one test per criterion in the policy)
# ---------------------------------------------------------------------------


class TestBucketCTripwires:
    def test_held_pr(self):
        held_number = next(iter(tri.HELD_PR_NUMBERS))
        pr = make_pr(number=held_number)
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert r.recommended_action == "STAY HELD"
        assert "held" in r.reason

    def test_protected_file_edit_claude_md(self):
        pr = make_pr(
            number=9100,
            files=[{"path": "CLAUDE.md", "additions": 5, "deletions": 0}],
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "CLAUDE.md" in r.reason

    def test_protected_file_edit_automation_toml(self):
        pr = make_pr(
            number=9101,
            files=[{"path": "automation.toml", "additions": 1, "deletions": 0}],
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "automation.toml" in r.reason

    def test_protected_file_edit_aragora_init(self):
        pr = make_pr(
            number=9102,
            files=[
                {"path": "aragora/__init__.py", "additions": 1, "deletions": 0},
                {"path": "tests/test_init.py", "additions": 5, "deletions": 0},
            ],
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C

    def test_flag_flip_tripwire(self):
        pr = make_pr(number=9114)
        pr["flagFlip"] = True
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "flag" in r.reason

    def test_operator_only_label_tripwire(self):
        pr = make_pr(number=9115)
        pr["labels"] = [{"name": "boss-ready"}]
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "boss-ready" in r.reason

    def test_external_dependency_manifest_tripwire(self):
        pr = make_pr(
            number=9116,
            files=[
                {"path": "pyproject.toml", "additions": 5, "deletions": 0},
                {"path": "tests/test_deps.py", "additions": 5, "deletions": 0},
            ],
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "external dependency" in r.reason

    def test_network_call_tripwire(self):
        pr = make_pr(number=9117)
        pr["networkCall"] = True
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "network" in r.reason

    def test_secret_read_tripwire(self):
        pr = make_pr(number=9118)
        pr["secretRead"] = True
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "secret" in r.reason

    def test_large_diff_over_threshold(self):
        pr = make_pr(number=9103, additions=1600, deletions=0)
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "large diff" in r.reason

    def test_ci_red_recent(self):
        pr = make_pr(
            number=9104,
            updated_days_ago=2,  # too recent for Bucket B (7d threshold)
            ci=[
                {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "tests", "status": "COMPLETED", "conclusion": "FAILURE"},
            ],
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "CI red" in r.reason

    def test_ci_pending(self):
        pr = make_pr(
            number=9105,
            ci=[
                {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "tests", "status": "IN_PROGRESS", "conclusion": None},
            ],
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "CI pending" in r.reason
        assert r.recommended_action == "DEFER"

    def test_ci_cancelled_is_non_green(self):
        pr = make_pr(
            number=9113,
            ci=[
                {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "docs", "status": "COMPLETED", "conclusion": "CANCELLED"},
            ],
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "CI non-green" in r.reason

    def test_non_trusted_author(self):
        pr = make_pr(number=9106, author="random-contributor")
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "non-trusted author" in r.reason

    def test_not_mergeable_conflicting(self):
        pr = make_pr(number=9107, mergeable="CONFLICTING")
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "not mergeable" in r.reason

    def test_merge_state_dirty(self):
        pr = make_pr(number=9108, merge_state="DIRTY")
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "DIRTY" in r.reason

    def test_merge_state_behind(self):
        pr = make_pr(number=9109, merge_state="BEHIND")
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C

    def test_code_change_without_tests(self):
        pr = make_pr(
            number=9110,
            files=[
                {"path": "aragora/foo/bar.py", "additions": 50, "deletions": 0},
            ],
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "without test" in r.reason

    def test_pure_docs_does_not_trip_no_tests_rule(self):
        # Pure docs PR (no code files at all) is Bucket A — tests aren't
        # required when there's no code.
        pr = make_pr(
            number=9111,
            files=[
                {"path": "docs/foo.md", "additions": 50, "deletions": 0},
                {"path": "docs/bar.md", "additions": 30, "deletions": 0},
            ],
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_A

    def test_changes_requested_review(self):
        pr = make_pr(number=9112, review="CHANGES_REQUESTED")
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "CHANGES_REQUESTED" in r.reason

    def test_unresolved_review_comments(self):
        pr = make_pr(number=9119)
        pr["unresolvedReviewComments"] = 2
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "unresolved review" in r.reason

    def test_unresolved_review_threads(self):
        pr = make_pr(number=9120)
        pr["reviewThreads"] = [
            {"isResolved": True},
            {"isResolved": False},
        ]
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "unresolved review" in r.reason


# ---------------------------------------------------------------------------
# Bucket B — auto-close
# ---------------------------------------------------------------------------


class TestBucketB:
    def test_ci_red_seven_days(self):
        pr = make_pr(
            number=9200,
            updated_days_ago=8,  # > 7 days
            ci=[
                {"name": "tests", "status": "COMPLETED", "conclusion": "FAILURE"},
            ],
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_B
        assert "≥7d" in r.reason
        assert r.recommended_action == "CLOSE"

    def test_stale_draft_over_thresholds(self):
        pr = make_pr(
            number=9201,
            is_draft=True,
            created_days_ago=70,
            updated_days_ago=40,
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_B
        assert "stale draft" in r.reason

    def test_stale_but_recently_updated_not_in_B(self):
        # 70 days old but updated 5 days ago → does NOT trigger Bucket B
        # stale path. Under the tightened policy, drafts go to C with
        # the "READY?" recommendation rather than to A — the test
        # confirms the stale-B path specifically doesn't fire.
        pr = make_pr(
            number=9202,
            is_draft=True,
            created_days_ago=70,
            updated_days_ago=5,
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "draft" in r.reason
        assert r.recommended_action == "READY?"

    def test_ready_pr_not_marked_stale(self):
        # Non-draft PRs of any age don't trigger the stale-draft path.
        pr = make_pr(
            number=9203,
            is_draft=False,
            created_days_ago=200,
            updated_days_ago=200,
        )
        r = classify(pr)
        # Ready + old + no fresh CI failures → bucket A (assuming tests
        # present in default fixture).
        assert r.bucket == tri.BUCKET_A

    def test_supersede_by_newer_with_clean_ci(self):
        older = make_pr(
            number=9300,
            files=[
                {"path": "aragora/foo/a.py", "additions": 10, "deletions": 0},
                {"path": "aragora/foo/b.py", "additions": 10, "deletions": 0},
                {"path": "tests/test_a.py", "additions": 10, "deletions": 0},
            ],
        )
        newer = make_pr(
            number=9301,
            files=[
                {"path": "aragora/foo/a.py", "additions": 12, "deletions": 0},
                {"path": "aragora/foo/b.py", "additions": 8, "deletions": 0},
                {"path": "tests/test_a.py", "additions": 14, "deletions": 0},
            ],
        )
        r = classify(older, all_open=[older, newer])
        assert r.bucket == tri.BUCKET_B
        assert "superseded by #9301" in r.reason

    def test_no_supersede_when_overlap_too_low(self):
        older = make_pr(
            number=9302,
            files=[
                {"path": "a.py", "additions": 10, "deletions": 0},
                {"path": "b.py", "additions": 10, "deletions": 0},
                {"path": "tests/test_a.py", "additions": 10, "deletions": 0},
            ],
        )
        newer = make_pr(
            number=9303,
            files=[
                {"path": "a.py", "additions": 5, "deletions": 0},
                {"path": "completely_different.py", "additions": 5, "deletions": 0},
                {"path": "tests/test_x.py", "additions": 10, "deletions": 0},
            ],
        )
        r = classify(older, all_open=[older, newer])
        # 1/3 overlap, below 0.8 threshold → no supersede; stays in A.
        assert r.bucket == tri.BUCKET_A

    def test_no_supersede_by_newer_with_ci_failure(self):
        older = make_pr(
            number=9304,
            files=[
                {"path": "a.py", "additions": 10, "deletions": 0},
                {"path": "tests/test_a.py", "additions": 10, "deletions": 0},
            ],
        )
        newer = make_pr(
            number=9305,
            files=[
                {"path": "a.py", "additions": 12, "deletions": 0},
                {"path": "tests/test_a.py", "additions": 12, "deletions": 0},
            ],
            ci=[{"name": "tests", "status": "COMPLETED", "conclusion": "FAILURE"}],
        )
        r = classify(older, all_open=[older, newer])
        # Newer has CI failure → not a valid superseder; older stays in A.
        assert r.bucket == tri.BUCKET_A


# ---------------------------------------------------------------------------
# Bucket B supersede — tightened: superseder must be Bucket-A-eligible
# (closes Codex Gap #3 from the PR #7285 review)
# ---------------------------------------------------------------------------


class TestSupersedeRequiresBucketAEligibility:
    """The tightened policy requires the newer PR to itself be in Bucket A
    (or already merged) before it can supersede an older PR. These tests
    verify that draft / held / CI-pending / merge-packet-blocked / non-
    trusted / protected-file / large / dirty candidate superseders are
    REJECTED and the older PR stays in A.
    """

    @staticmethod
    def _overlap_files(label: str) -> list[dict[str, Any]]:
        # Two-file PR with code + tests so the candidate has the right
        # shape for Bucket-A eligibility once each disqualifying gate
        # is removed.
        return [
            {"path": f"{label}.py", "additions": 10, "deletions": 0},
            {"path": f"tests/test_{label}.py", "additions": 10, "deletions": 0},
        ]

    def _older(self) -> dict[str, Any]:
        return make_pr(number=9400, files=self._overlap_files("a"))

    def _newer(self, **overrides: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = dict(
            number=9401,
            files=self._overlap_files("a"),
        )
        kwargs.update(overrides)
        return make_pr(**kwargs)

    def test_draft_superseder_does_not_fire(self):
        r = classify(self._older(), all_open=[self._older(), self._newer(is_draft=True)])
        assert r.bucket == tri.BUCKET_A

    def test_held_superseder_does_not_fire(self):
        held = next(iter(tri.HELD_PR_NUMBERS))
        r = classify(self._older(), all_open=[self._older(), self._newer(number=held)])
        assert r.bucket == tri.BUCKET_A

    def test_pending_ci_superseder_does_not_fire(self):
        ci = [
            {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
            {"name": "tests", "status": "IN_PROGRESS", "conclusion": None},
        ]
        r = classify(self._older(), all_open=[self._older(), self._newer(ci=ci)])
        assert r.bucket == tri.BUCKET_A

    def test_missing_merge_packet_superseder_does_not_fire(self):
        r = classify(
            self._older(),
            all_open=[self._older(), self._newer(merge_packet=None)],
        )
        assert r.bucket == tri.BUCKET_A

    def test_non_trusted_superseder_does_not_fire(self):
        r = classify(
            self._older(),
            all_open=[self._older(), self._newer(author="drive-by-contributor")],
        )
        assert r.bucket == tri.BUCKET_A

    def test_protected_file_superseder_does_not_fire(self):
        protected_files = [
            {"path": "a.py", "additions": 10, "deletions": 0},
            {"path": "tests/test_a.py", "additions": 10, "deletions": 0},
            {"path": "CLAUDE.md", "additions": 1, "deletions": 0},
        ]
        r = classify(
            self._older(),
            all_open=[self._older(), self._newer(files=protected_files)],
        )
        assert r.bucket == tri.BUCKET_A

    def test_large_superseder_does_not_fire(self):
        r = classify(
            self._older(),
            all_open=[self._older(), self._newer(additions=1600)],
        )
        assert r.bucket == tri.BUCKET_A

    def test_dirty_superseder_does_not_fire(self):
        r = classify(
            self._older(),
            all_open=[self._older(), self._newer(merge_state="DIRTY")],
        )
        assert r.bucket == tri.BUCKET_A

    def test_blocked_superseder_does_not_fire(self):
        # BLOCKED without review-required context is not Bucket-A-eligible.
        r = classify(
            self._older(),
            all_open=[self._older(), self._newer(merge_state="BLOCKED")],
        )
        assert r.bucket == tri.BUCKET_A

    def test_review_required_blocked_superseder_can_fire(self):
        r = classify(
            self._older(),
            all_open=[
                self._older(),
                self._newer(merge_state="BLOCKED", review="REVIEW_REQUIRED"),
            ],
        )
        assert r.bucket == tri.BUCKET_B
        assert "superseded by #9401" in r.reason

    def test_changes_requested_superseder_does_not_fire(self):
        r = classify(
            self._older(),
            all_open=[self._older(), self._newer(review="CHANGES_REQUESTED")],
        )
        assert r.bucket == tri.BUCKET_A

    def test_fully_eligible_superseder_still_fires(self):
        # Regression: the happy path still triggers the supersede.
        r = classify(self._older(), all_open=[self._older(), self._newer()])
        assert r.bucket == tri.BUCKET_B
        assert "superseded by #9401" in r.reason
        assert "Bucket-A-eligible" in r.reason


# ---------------------------------------------------------------------------
# Bucket precedence — most-restrictive wins
# ---------------------------------------------------------------------------


class TestPrecedence:
    def test_held_beats_all_other_signals(self):
        held = next(iter(tri.HELD_PR_NUMBERS))
        pr = make_pr(
            number=held,
            files=[{"path": "CLAUDE.md", "additions": 5, "deletions": 0}],
            additions=2000,
            ci=[{"name": "tests", "status": "COMPLETED", "conclusion": "FAILURE"}],
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        # The reason should specifically cite the hold (precedence proof).
        assert "held" in r.reason

    def test_protected_beats_large_diff(self):
        pr = make_pr(
            number=9400,
            files=[{"path": "automation.toml", "additions": 5, "deletions": 0}],
            additions=1600,
        )
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "automation.toml" in r.reason

    def test_ci_red_seven_days_beats_supersede(self):
        # Even with a newer superseder, CI red ≥7d wins (auto-close
        # signal is stronger than supersede).
        older = make_pr(
            number=9401,
            updated_days_ago=10,
            files=[
                {"path": "a.py", "additions": 5, "deletions": 0},
                {"path": "tests/test_a.py", "additions": 5, "deletions": 0},
            ],
            ci=[{"name": "tests", "status": "COMPLETED", "conclusion": "FAILURE"}],
        )
        newer = make_pr(
            number=9402,
            files=[
                {"path": "a.py", "additions": 5, "deletions": 0},
                {"path": "tests/test_a.py", "additions": 5, "deletions": 0},
            ],
        )
        r = classify(older, all_open=[older, newer])
        assert r.bucket == tri.BUCKET_B
        assert "≥7d" in r.reason


# ---------------------------------------------------------------------------
# Output formatting + CLI
# ---------------------------------------------------------------------------


class TestCliOutput:
    def test_main_from_json_human(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        prs = [
            make_pr(number=9500),
            make_pr(
                number=9501,
                files=[{"path": "CLAUDE.md", "additions": 1, "deletions": 0}],
            ),
        ]
        p = tmp_path / "prs.json"
        p.write_text(json.dumps(prs))
        rc = tri.main(["--from-json", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "BUCKET A" in out
        assert "BUCKET C" in out
        assert "#9500" in out
        assert "#9501" in out
        assert "summary:" in out

    def test_main_from_json_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        prs = [make_pr(number=9510)]
        p = tmp_path / "prs.json"
        p.write_text(json.dumps(prs))
        rc = tri.main(["--from-json", str(p), "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        data = json.loads(out)
        assert data["policy_doc"] == "docs/governance/OPERATOR_DELEGATION_POLICY.md"
        assert "rollout_doc" in data
        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["pr_number"] == 9510

    def test_bucket_filter(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        prs = [
            make_pr(number=9520),
            make_pr(
                number=9521,
                files=[{"path": "CLAUDE.md", "additions": 1, "deletions": 0}],
            ),
        ]
        p = tmp_path / "prs.json"
        p.write_text(json.dumps(prs))
        rc = tri.main(["--from-json", str(p), "--bucket", "C", "--json"])
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert all(r["bucket"] == "C" for r in data["results"])
        assert any(r["pr_number"] == 9521 for r in data["results"])
        assert not any(r["pr_number"] == 9520 for r in data["results"])

    def test_main_missing_from_json_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = tri.main(["--from-json", str(tmp_path / "nope.json")])
        assert rc == 2
        assert "file not found" in capsys.readouterr().err

    def test_main_invalid_from_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        p = tmp_path / "bad.json"
        p.write_text("not json")
        rc = tri.main(["--from-json", str(p)])
        assert rc == 2
        assert "invalid JSON" in capsys.readouterr().err

    def test_main_non_array_root(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        p = tmp_path / "obj.json"
        p.write_text('{"not": "an array"}')
        rc = tri.main(["--from-json", str(p)])
        assert rc == 2
        assert "must be a JSON array" in capsys.readouterr().err

    def test_no_gh_on_path(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(tri.shutil, "which", lambda _exe: None)
        rc = tri.main([])
        assert rc == 2
        assert "gh CLI not found" in capsys.readouterr().err

    def test_deterministic_output_across_runs(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        prs = [make_pr(number=9530), make_pr(number=9531), make_pr(number=9532)]
        p = tmp_path / "prs.json"
        p.write_text(json.dumps(prs))
        tri.main(["--from-json", str(p), "--json"])
        first = capsys.readouterr().out
        tri.main(["--from-json", str(p), "--json"])
        second = capsys.readouterr().out
        assert first == second


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_pr_list(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        p = tmp_path / "empty.json"
        p.write_text("[]")
        rc = tri.main(["--from-json", str(p)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "total: 0" in out

    def test_pr_with_zero_files(self) -> None:
        pr = make_pr(number=9600, files=[])
        r = classify(pr)
        # No code files, no test files, no protected files → A
        # (the no-test rule only fires if there ARE code files).
        assert r.bucket == tri.BUCKET_A

    def test_pr_with_unknown_author_dict(self) -> None:
        pr = make_pr(number=9601)
        pr["author"] = {}  # no login key
        r = classify(pr)
        assert r.bucket == tri.BUCKET_C
        assert "non-trusted author" in r.reason

    def test_reason_capped_at_200_chars(self) -> None:
        very_long_title = "x" * 1000
        pr = make_pr(number=9602, title=very_long_title)
        r = classify(pr)
        assert len(r.reason) <= 200
        assert len(r.title) <= 80
