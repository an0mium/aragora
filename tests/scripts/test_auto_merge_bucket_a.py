"""Tests for ``scripts/auto_merge_bucket_a.py``.

Fixture-driven; never invokes real ``gh`` or merges any PR. Every
test constructs a synthetic Stage 1 classifier payload plus per-PR
metadata payloads and asserts that the decisions match the policy.

Coverage:
  - dry-run never mutates (no merge calls captured)
  - --apply merges only Bucket A; never B/C/D
  - defense-in-depth tripwire on protected paths aborts and exits
    non-zero overall
  - defense-in-depth tripwire on CI-pending aborts (independent of
    Stage 1)
  - settling window skips young PRs
  - --only-pr filters correctly
  - receipt is written on apply runs, with sha256 of classifier output
  - --json emits valid JSON with the same decisions
"""

from __future__ import annotations

import dataclasses
import datetime
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module() -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "auto_merge_bucket_a.py"
    spec = importlib.util.spec_from_file_location("auto_merge_bucket_a_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


amba = _load_module()


NOW = datetime.datetime(2026, 5, 18, 1, 0, 0, tzinfo=datetime.timezone.utc)


def _ago(*, minutes: int = 0, hours: int = 0) -> str:
    delta = datetime.timedelta(minutes=minutes, hours=hours)
    return (NOW - delta).isoformat().replace("+00:00", "Z")


def make_triage_payload(*entries: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": {"A": sum(1 for e in entries if e["bucket"] == "A")},
        "results": list(entries),
    }


def bucket_a_entry(pr_number: int, title: str = "feat: tiny safe PR") -> dict[str, Any]:
    return {
        "pr_number": pr_number,
        "title": title,
        "bucket": "A",
        "reason": "all gates clean",
        "recommended_action": "MERGE",
    }


def bucket_c_entry(pr_number: int, title: str = "docs: needs review") -> dict[str, Any]:
    return {
        "pr_number": pr_number,
        "title": title,
        "bucket": "C",
        "reason": "draft",
        "recommended_action": "READY?",
    }


def make_metadata(
    pr_number: int,
    *,
    last_commit_minutes_ago: int = 999,
    is_draft: bool = False,
    mergeable: str = "MERGEABLE",
    merge_state: str = "CLEAN",
    files: list[dict[str, Any]] | None = None,
    ci: list[dict[str, Any]] | None = None,
    labels: list[dict[str, Any]] | None = None,
    author: str = "an0mium",
    head_sha: str | None = None,
    title: str = "feat: tiny safe PR",
) -> dict[str, Any]:
    if head_sha is None:
        head_sha = f"{pr_number:040x}"[-40:]
    if files is None:
        files = [
            {"path": "scripts/some_helper.py", "additions": 10, "deletions": 0},
            {
                "path": "tests/scripts/test_some_helper.py",
                "additions": 10,
                "deletions": 0,
            },
        ]
    if ci is None:
        ci = [
            {
                "name": "lint",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
            },
            {
                "name": "tests",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
            },
        ]
    return {
        "number": pr_number,
        "title": title,
        "author": {"login": author},
        "isDraft": is_draft,
        "mergeable": mergeable,
        "mergeStateStatus": merge_state,
        "labels": labels or [],
        "files": files,
        "commits": [
            {"committedDate": _ago(minutes=last_commit_minutes_ago)},
        ],
        "statusCheckRollup": ci,
        "headRefOid": head_sha,
    }


class _MetadataRegistry:
    """Map pr_number → metadata dict, with a callable interface."""

    def __init__(self, entries: dict[int, dict[str, Any]]):
        self._entries = entries

    def __call__(self, pr_number: int) -> dict[str, Any]:
        return self._entries[pr_number]


class _MergeRecorder:
    """Captures merge attempts. Raises if expected_failures[n] is set."""

    def __init__(self, expected_failures: set[int] | None = None):
        self.calls: list[int] = []
        self.head_shas: list[str] = []
        self.delete_branch_flags: list[bool] = []
        self.admin_squash_flags: list[bool] = []
        self._failures = expected_failures or set()

    def __call__(
        self,
        pr_number: int,
        head_sha: str,
        delete_branch_on_merge: bool,
        admin_squash: bool,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(pr_number)
        self.head_shas.append(head_sha)
        self.delete_branch_flags.append(delete_branch_on_merge)
        self.admin_squash_flags.append(admin_squash)
        if pr_number in self._failures:
            raise RuntimeError(f"simulated merge failure on #{pr_number}")
        return subprocess.CompletedProcess(
            args=["gh", "pr", "merge", str(pr_number)], returncode=0, stdout="", stderr=""
        )


def make_merge_packet(
    pr_number: int,
    *,
    head_sha: str,
    admin_squash_allowed: bool = True,
    not_ready: list[int] | None = None,
) -> dict[str, Any]:
    return {
        "entries": [
            {
                "pr_number": pr_number,
                "head_sha": head_sha,
                "admin_squash_allowed": admin_squash_allowed,
            }
        ],
        "not_ready": [] if not_ready is None else not_ready,
    }


class _MergePacketRegistry:
    def __init__(self, entries: dict[int, dict[str, Any]]):
        self._entries = entries
        self.calls: list[int] = []

    def __call__(self, pr_number: int) -> dict[str, Any]:
        self.calls.append(pr_number)
        return self._entries[pr_number]


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------


class TestGhMerge:
    def test_merge_uses_exact_head_and_preserves_branch_by_default(self):
        calls: list[list[str]] = []

        def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
            calls.append(args)
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        head_sha = "a" * 40
        amba.gh_pr_merge_squash(9001, head_sha, runner=runner)

        assert calls == [
            [
                "gh",
                "pr",
                "merge",
                "9001",
                "--squash",
                "--match-head-commit",
                head_sha,
            ]
        ]

    def test_delete_branch_is_opt_in(self):
        calls: list[list[str]] = []

        def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
            calls.append(args)
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        amba.gh_pr_merge_squash(9001, "b" * 40, delete_branch_on_merge=True, runner=runner)

        assert "--delete-branch" in calls[0]
        assert "--admin" not in calls[0]

    def test_admin_squash_is_opt_in(self):
        calls: list[list[str]] = []

        def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
            calls.append(args)
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        amba.gh_pr_merge_squash(9001, "c" * 40, admin_squash=True, runner=runner)

        assert calls == [
            [
                "gh",
                "pr",
                "merge",
                "9001",
                "--squash",
                "--match-head-commit",
                "c" * 40,
                "--admin",
            ]
        ]


class TestDecide:
    def test_dry_run_never_mutates(self):
        payload = make_triage_payload(bucket_a_entry(9001), bucket_c_entry(9002))
        meta = _MetadataRegistry({9001: make_metadata(9001)})
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=False,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )
        assert recorder.calls == []
        assert exit_code == 0
        merge_d = [d for d in decisions if d.decision == "merge"]
        skip_d = [d for d in decisions if d.decision == "skip-non-bucket-a"]
        assert [d.pr_number for d in merge_d] == [9001]
        assert all(d.applied is False for d in merge_d)
        assert [d.pr_number for d in skip_d] == [9002]

    def test_apply_skips_b_c_d_buckets(self):
        payload = make_triage_payload(
            bucket_a_entry(9001),
            bucket_c_entry(9002),
            {"pr_number": 9003, "title": "stale", "bucket": "B", "reason": "stale"},
            {"pr_number": 9004, "title": "strategic", "bucket": "D", "reason": "future"},
        )
        meta = _MetadataRegistry({9001: make_metadata(9001)})
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )
        assert recorder.calls == [9001]
        assert exit_code == 0
        merged = [d for d in decisions if d.decision == "merge" and d.applied]
        assert [d.pr_number for d in merged] == [9001]
        assert recorder.head_shas == [make_metadata(9001)["headRefOid"]]
        assert recorder.delete_branch_flags == [False]
        assert recorder.admin_squash_flags == [False]
        skipped_non_a = sorted(d.pr_number for d in decisions if d.decision == "skip-non-bucket-a")
        assert skipped_non_a == [9002, 9003, 9004]

    def test_apply_can_opt_in_to_branch_deletion(self):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry({9001: make_metadata(9001)})
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            delete_branch_on_merge=True,
            now=NOW,
        )
        assert [d.pr_number for d in decisions] == [9001]
        assert recorder.calls == [9001]
        assert recorder.delete_branch_flags == [True]
        assert recorder.admin_squash_flags == [False]
        assert exit_code == 0

    def test_settling_window_skips_young_pr(self):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry({9001: make_metadata(9001, last_commit_minutes_ago=5)})
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )
        assert recorder.calls == []
        assert exit_code == 0
        assert len(decisions) == 1
        assert decisions[0].decision == "skip-settling"
        assert "settling window" in decisions[0].reason

    def test_only_pr_filter(self):
        payload = make_triage_payload(bucket_a_entry(9001), bucket_a_entry(9002))
        meta = _MetadataRegistry({9001: make_metadata(9001), 9002: make_metadata(9002)})
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            only_pr=9002,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )
        assert recorder.calls == [9002]
        assert exit_code == 0
        assert len(decisions) == 1
        assert decisions[0].pr_number == 9002

    def test_metadata_fetch_failure_is_loud_nonzero(self):
        def failing_metadata_provider(pr_number: int) -> dict[str, Any]:
            raise RuntimeError(f"simulated metadata failure for #{pr_number}")

        payload = make_triage_payload(bucket_a_entry(9001))
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=failing_metadata_provider,
            merger=recorder,
            now=NOW,
        )

        assert recorder.calls == []
        assert exit_code == 1
        assert decisions[0].decision == "metadata-fetch-failed"
        assert "simulated metadata failure" in decisions[0].reason

    def test_metadata_number_mismatch_is_tripwire(self):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry({9001: make_metadata(9002, head_sha="c" * 40)})
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )

        assert recorder.calls == []
        assert exit_code == 1
        assert decisions[0].decision == "skip-tripwire"
        assert "PR number mismatch" in decisions[0].reason


# ---------------------------------------------------------------------------
# Defense-in-depth tripwires
# ---------------------------------------------------------------------------


class TestDefenseInDepth:
    @pytest.mark.parametrize(
        "merge_state",
        ["UNSTABLE", "HAS_HOOKS", "BEHIND", "DIRTY", "DRAFT", "UNKNOWN", ""],
    )
    def test_unsafe_merge_states_are_tripwires(self, merge_state: str):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry({9001: make_metadata(9001, merge_state=merge_state)})
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )

        assert recorder.calls == []
        assert exit_code == 1
        assert decisions[0].decision == "skip-tripwire"
        assert "merge state not clean" in decisions[0].reason

    def test_blocked_merge_state_requires_merge_packet(self):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry({9001: make_metadata(9001, merge_state="BLOCKED")})
        recorder = _MergeRecorder()

        def missing_packet(_pr_number: int) -> dict[str, Any]:
            raise RuntimeError("simulated packet failure")

        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merge_packet_provider=missing_packet,
            merger=recorder,
            now=NOW,
        )

        assert recorder.calls == []
        assert exit_code == 1
        assert decisions[0].decision == "skip-tripwire"
        assert "merge-packet fetch failed" in decisions[0].reason

    def test_blocked_merge_state_allowed_with_exact_head_authorized_merge_packet(self):
        payload = make_triage_payload(bucket_a_entry(9001))
        metadata = make_metadata(9001, merge_state="BLOCKED")
        meta = _MetadataRegistry({9001: metadata})
        packets = _MergePacketRegistry(
            {9001: make_merge_packet(9001, head_sha=metadata["headRefOid"])}
        )
        recorder = _MergeRecorder()

        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merge_packet_provider=packets,
            merger=recorder,
            now=NOW,
        )

        assert packets.calls == [9001]
        assert recorder.calls == [9001]
        assert recorder.admin_squash_flags == [True]
        assert exit_code == 0
        assert decisions[0].decision == "merge"

    @pytest.mark.parametrize(
        ("packet", "reason_fragment"),
        [
            (
                make_merge_packet(9001, head_sha="b" * 40),
                "head does not match",
            ),
            (
                make_merge_packet(9001, head_sha="a" * 40, admin_squash_allowed=False),
                "admin squash is not authorized",
            ),
            (
                make_merge_packet(9001, head_sha="a" * 40, not_ready=[9001]),
                "not_ready is non-empty",
            ),
            (
                make_merge_packet(9002, head_sha="a" * 40),
                "absent from merge-packet",
            ),
        ],
    )
    def test_blocked_merge_state_rejects_non_authorized_merge_packet(
        self,
        packet: dict[str, Any],
        reason_fragment: str,
    ):
        payload = make_triage_payload(bucket_a_entry(9001))
        metadata = make_metadata(9001, merge_state="BLOCKED", head_sha="a" * 40)
        meta = _MetadataRegistry({9001: metadata})
        packets = _MergePacketRegistry({9001: packet})
        recorder = _MergeRecorder()

        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merge_packet_provider=packets,
            merger=recorder,
            now=NOW,
        )

        assert recorder.calls == []
        assert exit_code == 1
        assert decisions[0].decision == "skip-tripwire"
        assert reason_fragment in decisions[0].reason

    @pytest.mark.parametrize(
        "protected_path",
        [
            "scripts/nomic_loop.py",
            "scripts/auto_merge_bucket_a.py",
            "scripts/triage_open_prs.py",
        ],
    )
    def test_protected_path_tripwire_forces_nonzero_exit(self, protected_path: str):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry(
            {
                9001: make_metadata(
                    9001,
                    files=[
                        {"path": protected_path, "additions": 1},
                        {"path": "tests/x.py", "additions": 1},
                    ],
                )
            }
        )
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )
        assert recorder.calls == []
        assert exit_code == 1
        assert decisions[0].decision == "skip-tripwire"
        assert "protected path" in decisions[0].reason

    def test_workflow_path_tripwire(self):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry(
            {
                9001: make_metadata(
                    9001,
                    files=[{"path": ".github/workflows/ci.yml", "additions": 1}],
                )
            }
        )
        recorder = _MergeRecorder()
        _, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )
        assert exit_code == 1
        assert recorder.calls == []

    def test_ci_pending_caught_by_defense_in_depth(self):
        # Classifier said A but metadata shows a pending check —
        # defense in depth must catch the race.
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry(
            {
                9001: make_metadata(
                    9001,
                    ci=[
                        {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
                        {"name": "slow", "status": "IN_PROGRESS", "conclusion": None},
                    ],
                )
            }
        )
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )
        assert recorder.calls == []
        assert exit_code == 1
        assert decisions[0].decision == "skip-tripwire"
        assert "CI pending" in decisions[0].reason

    @pytest.mark.parametrize(
        ("rollup", "reason_fragment"),
        [
            (None, "CI rollup unavailable"),
            ("not-a-list", "CI rollup unavailable"),
            ([], "CI rollup empty"),
            (["not-a-dict"], "CI rollup malformed"),
            ([{"name": "empty"}], "CI rollup malformed"),
        ],
    )
    def test_missing_or_malformed_ci_rollup_fails_closed(
        self,
        rollup: Any,
        reason_fragment: str,
    ):
        payload = make_triage_payload(bucket_a_entry(9001))
        metadata = make_metadata(9001)
        metadata["statusCheckRollup"] = rollup
        meta = _MetadataRegistry({9001: metadata})
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )

        assert recorder.calls == []
        assert exit_code == 1
        assert decisions[0].decision == "skip-tripwire"
        assert reason_fragment in decisions[0].reason

    @pytest.mark.parametrize(
        "conclusion",
        ["ACTION_REQUIRED", "CANCELLED", "FAILURE", "STARTUP_FAILURE", "TIMED_OUT"],
    )
    def test_completed_non_green_ci_caught_by_defense_in_depth(self, conclusion: str):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry(
            {
                9001: make_metadata(
                    9001,
                    ci=[
                        {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
                        {"name": "late", "status": "COMPLETED", "conclusion": conclusion},
                    ],
                )
            }
        )
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )

        assert recorder.calls == []
        assert exit_code == 1
        assert decisions[0].decision == "skip-tripwire"
        if conclusion == "FAILURE":
            assert decisions[0].reason == "CI red (1 failures)"
        else:
            assert decisions[0].reason == f"CI non-green ({conclusion})"

    @pytest.mark.parametrize("status", ["IN_PROGRESS", "PENDING", "QUEUED", "REQUESTED", "WAITING"])
    def test_non_completed_check_status_caught_by_defense_in_depth(self, status: str):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry(
            {
                9001: make_metadata(
                    9001,
                    ci=[
                        {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
                        {"name": "late", "status": status, "conclusion": None},
                    ],
                )
            }
        )
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )

        assert recorder.calls == []
        assert exit_code == 1
        assert decisions[0].decision == "skip-tripwire"
        assert decisions[0].reason == "CI pending (1 in-flight)"

    @pytest.mark.parametrize(
        ("state", "reason"),
        [
            ("ERROR", "CI red (1 failures)"),
            ("FAILURE", "CI red (1 failures)"),
            ("EXPECTED", "CI pending (1 in-flight)"),
            ("PENDING", "CI pending (1 in-flight)"),
        ],
    )
    def test_status_context_state_caught_by_defense_in_depth(self, state: str, reason: str):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry(
            {
                9001: make_metadata(
                    9001,
                    ci=[
                        {"__typename": "StatusContext", "context": "lint", "state": "SUCCESS"},
                        {"__typename": "StatusContext", "context": "late", "state": state},
                    ],
                )
            }
        )
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )

        assert recorder.calls == []
        assert exit_code == 1
        assert decisions[0].decision == "skip-tripwire"
        assert decisions[0].reason == reason

    def test_unknown_status_context_state_fails_closed(self):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry(
            {
                9001: make_metadata(
                    9001,
                    ci=[
                        {"__typename": "StatusContext", "context": "lint", "state": "SUCCESS"},
                        {
                            "__typename": "StatusContext",
                            "context": "future-state",
                            "state": "SOMETHING_NEW",
                        },
                    ],
                )
            }
        )
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )

        assert recorder.calls == []
        assert exit_code == 1
        assert decisions[0].decision == "skip-tripwire"
        assert decisions[0].reason == "CI non-green (STATUS_CONTEXT_SOMETHING_NEW)"

    @pytest.mark.parametrize(
        "label",
        [
            "autonomous",
            "boss-ready",
            "do not merge",
            "do-not-merge",
            "hold",
            "manual_review",
        ],
    )
    def test_blocking_label_caught_by_defense_in_depth(self, label: str):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry({9001: make_metadata(9001, labels=[{"name": label}])})
        recorder = _MergeRecorder()
        decisions, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )

        assert recorder.calls == []
        assert exit_code == 1
        assert decisions[0].decision == "skip-tripwire"
        assert "operator label tripwire" in decisions[0].reason

    def test_draft_state_caught(self):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry({9001: make_metadata(9001, is_draft=True)})
        recorder = _MergeRecorder()
        _, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )
        assert exit_code == 1
        assert recorder.calls == []

    def test_non_trusted_author_caught(self):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry({9001: make_metadata(9001, author="some-other-author")})
        recorder = _MergeRecorder()
        _, exit_code = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )
        assert exit_code == 1
        assert recorder.calls == []


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------


class TestReceipt:
    def test_receipt_is_written_on_apply(self, tmp_path: Path):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry({9001: make_metadata(9001)})
        recorder = _MergeRecorder()
        decisions, _ = amba.decide(
            payload,
            apply=True,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )
        receipt_md = amba.render_receipt(
            decisions,
            triage_payload=payload,
            apply=True,
            settling_minutes=30,
            now=NOW,
        )
        path = amba.write_receipt(receipt_md, now=NOW, receipt_dir=tmp_path)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "#9001" in content
        assert "merge" in content
        assert "Classifier output sha256" in content

    def test_classifier_sha256_changes_when_results_change(self):
        a = make_triage_payload(bucket_a_entry(9001))
        b = make_triage_payload(bucket_a_entry(9001), bucket_c_entry(9002))
        assert amba._classifier_sha256(a) != amba._classifier_sha256(b)

    def test_policy_version_uses_anchored_version_line(self, tmp_path: Path, monkeypatch):
        policy_doc = tmp_path / "policy.md"
        policy_doc.write_text(
            "This prose mentions Version: stale but is not the version line.\n"
            "\n"
            "Version: operator-delegation-2026-05-18\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(amba, "POLICY_DOC", policy_doc)

        assert amba._policy_version() == "operator-delegation-2026-05-18"

    def test_json_output_round_trip(self):
        payload = make_triage_payload(bucket_a_entry(9001))
        meta = _MetadataRegistry({9001: make_metadata(9001)})
        recorder = _MergeRecorder()
        decisions, _ = amba.decide(
            payload,
            apply=False,
            settling_minutes=30,
            metadata_provider=meta,
            merger=recorder,
            now=NOW,
        )
        out = amba.emit_json(
            decisions,
            triage_payload=payload,
            apply=False,
            settling_minutes=30,
            receipt_path=None,
        )
        parsed = json.loads(out)
        assert parsed["mode"] == "dry-run"
        assert parsed["settling_minutes"] == 30
        assert parsed["receipt_path"] is None
        assert len(parsed["decisions"]) == 1
        assert parsed["decisions"][0]["pr_number"] == 9001
