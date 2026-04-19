"""Tests for aragora review-queue (Phase 2a, read-only)."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from typing import Any

import pytest

from aragora.cli.commands.review_queue import (
    ADVISORY_NOTE,
    HIGH_RISK_PATHS,
    LARGE_DIFF_THRESHOLD,
    PARKED_LABELS,
    QueueItem,
    ReviewPacket,
    _build_packet,
    _build_queue,
    _classify_pr,
    _filter_lanes,
    _GhError,
    _is_high_risk_path,
    _parse_pr_number,
    _subsystem_for,
    _summarize_checks,
    add_review_queue_parser,
    cmd_review_queue,
)


# --- Synthetic PR payload builder ------------------------------------------


def _make_pr(
    *,
    number: int = 1,
    title: str = "test PR",
    is_draft: bool = False,
    mergeable: str = "MERGEABLE",
    review_decision: str = "",
    labels: list[str] | None = None,
    additions: int = 10,
    deletions: int = 5,
    changed_files: int = 2,
    checks: list[dict[str, Any]] | None = None,
    files: list[str] | None = None,
    author: str = "an0mium",
) -> dict[str, Any]:
    """Build a synthetic gh-pr-list-style payload."""
    return {
        "number": number,
        "title": title,
        "url": f"https://github.com/synaptent/aragora/pull/{number}",
        "headRefName": f"branch-{number}",
        "headRefOid": f"sha{number:08d}",
        "isDraft": is_draft,
        "mergeable": mergeable,
        "reviewDecision": review_decision,
        "labels": [{"name": lab} for lab in (labels or [])],
        "author": {"login": author},
        "additions": additions,
        "deletions": deletions,
        "changedFiles": changed_files,
        "statusCheckRollup": checks
        or [{"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"}],
        "files": [{"path": p} for p in (files or [])],
    }


# --- _summarize_checks -----------------------------------------------------


class TestSummarizeChecks:
    def test_all_green(self) -> None:
        checks = [
            {"status": "COMPLETED", "conclusion": "SUCCESS"},
            {"status": "COMPLETED", "conclusion": "SUCCESS"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert "2/2 green" in summary
        assert not has_fail
        assert not has_pending

    def test_one_failing(self) -> None:
        checks = [
            {"status": "COMPLETED", "conclusion": "SUCCESS"},
            {"status": "COMPLETED", "conclusion": "FAILURE"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert has_fail
        assert "1 failing" in summary

    def test_pending(self) -> None:
        checks = [
            {"status": "IN_PROGRESS", "conclusion": ""},
            {"status": "COMPLETED", "conclusion": "SUCCESS"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert has_pending
        assert not has_fail
        assert "1 pending" in summary

    def test_state_based_green_rollups_are_counted_as_green(self) -> None:
        checks = [
            {"context": "lint", "state": "SUCCESS"},
            {"context": "ci/unit", "state": "SUCCESS"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert summary == "2/2 green"
        assert not has_fail
        assert not has_pending

    def test_state_based_pending_and_failure_rollups_are_preserved(self) -> None:
        checks = [
            {"context": "lint", "state": "SUCCESS"},
            {"context": "ci/unit", "state": "PENDING"},
            {"context": "security", "state": "FAILURE"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert summary == "1 failing / 3 total"
        assert has_fail
        assert has_pending

    def test_skipped_excluded_from_meaningful_total(self) -> None:
        checks = [
            {"status": "COMPLETED", "conclusion": "SUCCESS"},
            {"status": "COMPLETED", "conclusion": "SKIPPED"},
            {"status": "COMPLETED", "conclusion": "CANCELLED"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert not has_fail
        assert not has_pending
        assert "1/1 green" in summary

    def test_no_checks(self) -> None:
        summary, has_fail, has_pending = _summarize_checks([])
        assert summary == "no checks"
        assert not has_fail
        assert not has_pending

    def test_malformed_check_ignored(self) -> None:
        checks: list[Any] = ["not a dict", None, {"status": "COMPLETED", "conclusion": "SUCCESS"}]
        summary, _, _ = _summarize_checks(checks)
        assert "1/1 green" in summary


# --- _classify_pr lane logic -----------------------------------------------


class TestClassifyPR:
    def test_ready_now_when_all_green_small_diff(self) -> None:
        pr = _make_pr()
        item = _classify_pr(pr)
        assert item.lane == "ready_now"

    def test_state_based_green_rollups_are_ready_now(self) -> None:
        pr = _make_pr(
            checks=[
                {"context": "lint", "state": "SUCCESS"},
                {"context": "ci/unit", "state": "SUCCESS"},
            ]
        )
        item = _classify_pr(pr)
        assert item.lane == "ready_now"

    def test_draft_is_parked(self) -> None:
        pr = _make_pr(is_draft=True)
        item = _classify_pr(pr)
        assert item.lane == "parked"
        assert "draft" in item.lane_reason.lower()

    def test_parked_label_parks_pr(self) -> None:
        for label in PARKED_LABELS:
            pr = _make_pr(labels=[label])
            item = _classify_pr(pr)
            assert item.lane == "parked", f"label={label} should park PR"

    def test_merge_conflict_is_parked(self) -> None:
        pr = _make_pr(mergeable="CONFLICTING")
        item = _classify_pr(pr)
        assert item.lane == "parked"
        assert "conflict" in item.lane_reason.lower()

    def test_failing_check_is_repairable(self) -> None:
        pr = _make_pr(
            checks=[
                {"status": "COMPLETED", "conclusion": "SUCCESS"},
                {"status": "COMPLETED", "conclusion": "FAILURE"},
            ]
        )
        item = _classify_pr(pr)
        assert item.lane == "repairable"

    def test_pending_check_needs_attention(self) -> None:
        pr = _make_pr(
            checks=[
                {"status": "IN_PROGRESS", "conclusion": ""},
            ]
        )
        item = _classify_pr(pr)
        assert item.lane == "needs_attention"

    def test_state_based_pending_check_needs_attention(self) -> None:
        pr = _make_pr(
            checks=[
                {"context": "ci/unit", "state": "PENDING"},
            ]
        )
        item = _classify_pr(pr)
        assert item.lane == "needs_attention"

    def test_state_based_failure_is_repairable(self) -> None:
        pr = _make_pr(
            checks=[
                {"context": "ci/unit", "state": "FAILURE"},
            ]
        )
        item = _classify_pr(pr)
        assert item.lane == "repairable"

    def test_large_diff_needs_attention(self) -> None:
        pr = _make_pr(additions=LARGE_DIFF_THRESHOLD + 100, deletions=10)
        item = _classify_pr(pr)
        assert item.lane == "needs_attention"
        assert "large diff" in item.lane_reason.lower()

    def test_priority_order_draft_beats_failing(self) -> None:
        # A draft PR with failing checks should still be parked, not repairable.
        pr = _make_pr(
            is_draft=True,
            checks=[{"status": "COMPLETED", "conclusion": "FAILURE"}],
        )
        item = _classify_pr(pr)
        assert item.lane == "parked"

    def test_priority_order_conflict_beats_failing(self) -> None:
        # Conflict parks the PR even when checks also fail.
        pr = _make_pr(
            mergeable="CONFLICTING",
            checks=[{"status": "COMPLETED", "conclusion": "FAILURE"}],
        )
        item = _classify_pr(pr)
        assert item.lane == "parked"


# --- _filter_lanes ---------------------------------------------------------


class TestFilterLanes:
    def _items(self) -> list[QueueItem]:
        # Build 4 items, one per lane, smallest synthetic representation.
        out: list[QueueItem] = []
        for lane in ("ready_now", "needs_attention", "repairable", "parked"):
            out.append(
                _classify_pr(
                    _make_pr(
                        number=hash(lane) & 0xFFFF,
                        is_draft=(lane == "parked"),
                        checks=[{"status": "COMPLETED", "conclusion": "FAILURE"}]
                        if lane == "repairable"
                        else (
                            [{"status": "IN_PROGRESS", "conclusion": ""}]
                            if lane == "needs_attention"
                            else [{"status": "COMPLETED", "conclusion": "SUCCESS"}]
                        ),
                    )
                )
            )
        return out

    def test_ready_only(self) -> None:
        items = self._items()
        result = _filter_lanes(items, ready_only=True, include_parked=False)
        assert {it.lane for it in result} == {"ready_now"}

    def test_default_excludes_parked(self) -> None:
        items = self._items()
        result = _filter_lanes(items, ready_only=False, include_parked=False)
        assert "parked" not in {it.lane for it in result}

    def test_include_parked_keeps_all(self) -> None:
        items = self._items()
        result = _filter_lanes(items, ready_only=False, include_parked=True)
        assert "parked" in {it.lane for it in result}


# --- _subsystem_for + _is_high_risk_path -----------------------------------


class TestSubsystemAndRisk:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("aragora/cli/commands/review_pr.py", "aragora/cli"),
            ("tests/cli/commands/test_review_queue.py", "tests/cli"),
            ("docs/CI_LANES.md", "docs"),
            ("scripts/automation_pr_preflight.sh", "scripts"),
            ("benchmarks/bench_readiness/README.md", "benchmarks"),
            (".github/workflows/test.yml", ".github"),
            ("README.md", "README.md"),
        ],
    )
    def test_subsystem_mapping(self, path: str, expected: str) -> None:
        assert _subsystem_for(path) == expected

    def test_high_risk_exact_paths(self) -> None:
        for path in HIGH_RISK_PATHS:
            assert _is_high_risk_path(path), f"{path} should be flagged"

    @pytest.mark.parametrize(
        "path",
        [
            "aragora/security/encryption.py",
            "aragora/auth/oidc.py",
            "aragora/blockchain/wallet.py",
            "aragora/rbac/checker.py",
            "scripts/auto_revert_main_required_failures.py",
            ".github/workflows/release.yml",
        ],
    )
    def test_high_risk_prefixes(self, path: str) -> None:
        assert _is_high_risk_path(path)

    def test_non_high_risk(self) -> None:
        assert not _is_high_risk_path("aragora/cli/commands/review_queue.py")
        assert not _is_high_risk_path("docs/CI_LANES.md")
        assert not _is_high_risk_path("tests/cli/commands/test_review_queue.py")


# --- _parse_pr_number ------------------------------------------------------


class TestParsePRNumber:
    @pytest.mark.parametrize(
        ("ref", "expected"),
        [
            ("6280", 6280),
            ("#6280", 6280),
            ("https://github.com/synaptent/aragora/pull/6280", 6280),
            ("https://github.com/synaptent/aragora/pull/6280/", 6280),
        ],
    )
    def test_parses(self, ref: str, expected: int) -> None:
        assert _parse_pr_number(ref) == expected

    def test_rejects_invalid(self) -> None:
        with pytest.raises(_GhError):
            _parse_pr_number("not a number")


# --- _build_queue + _build_packet (with mocked gh) -------------------------


class TestBuildQueueAndPacket:
    def test_build_queue_classifies_and_sorts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        prs = [
            _make_pr(number=10, is_draft=True),  # parked
            _make_pr(number=20),  # ready_now
            _make_pr(
                number=30,
                checks=[{"status": "COMPLETED", "conclusion": "FAILURE"}],
            ),  # repairable
            _make_pr(
                number=40,
                checks=[{"status": "IN_PROGRESS", "conclusion": ""}],
            ),  # needs_attention
        ]
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: prs,
        )
        items = _build_queue(limit=100)
        assert [it.number for it in items] == [20, 40, 30, 10]
        assert [it.lane for it in items] == [
            "ready_now",
            "needs_attention",
            "repairable",
            "parked",
        ]

    def test_build_packet_sets_recommendation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pr_payload = _make_pr(
            number=6280,
            files=["aragora/cli/commands/review_pr.py", "tests/cli/commands/test_review_pr.py"],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("6280", repo_override=None)
        assert packet.pr_number == 6280
        assert packet.advisory_only is True
        assert packet.settlement_note == ADVISORY_NOTE
        assert packet.machine_recommendation == "approve_candidate"
        assert "aragora/cli" in packet.touched_subsystems
        assert "tests/cli" in packet.touched_subsystems
        assert packet.high_risk_paths_touched == []

    def test_build_packet_accepts_state_based_green_rollups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=6280,
            checks=[
                {"context": "lint", "state": "SUCCESS"},
                {"context": "ci/unit", "state": "SUCCESS"},
            ],
            files=["aragora/cli/commands/review_pr.py"],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("6280", repo_override=None)
        assert packet.machine_recommendation == "approve_candidate"
        assert packet.checks_summary == "2/2 green"

    def test_build_packet_flags_high_risk_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pr_payload = _make_pr(
            number=42,
            files=["aragora/security/encryption.py", "aragora/cli/commands/review_pr.py"],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("42", repo_override=None)
        assert packet.machine_recommendation == "needs_human_attention"
        assert "aragora/security/encryption.py" in packet.high_risk_paths_touched

    def test_build_packet_failures_recommend_repair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pr_payload = _make_pr(
            number=99,
            checks=[{"status": "COMPLETED", "conclusion": "FAILURE"}],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("99", repo_override=None)
        assert packet.machine_recommendation == "repair_first"

    def test_build_packet_draft_needs_attention(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pr_payload = _make_pr(number=101, is_draft=True)
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("101", repo_override=None)
        assert packet.machine_recommendation == "needs_human_attention"
        assert "draft" in packet.machine_recommendation_reason.lower()
        assert "draft PR" in packet.risk_flags

    def test_build_packet_parked_label_needs_attention(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=102, labels=["blocked"])
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("102", repo_override=None)
        assert packet.machine_recommendation == "needs_human_attention"
        assert "parked label" in packet.machine_recommendation_reason.lower()
        assert "parked label (blocked)" in packet.risk_flags

    def test_build_packet_pending_checks_need_attention(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=100,
            checks=[{"status": "IN_PROGRESS", "conclusion": ""}],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("100", repo_override=None)
        assert packet.machine_recommendation == "needs_human_attention"
        assert "checks still pending" in packet.machine_recommendation_reason

    def test_build_packet_state_based_pending_checks_need_attention(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=100,
            checks=[{"context": "ci/unit", "state": "PENDING"}],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("100", repo_override=None)
        assert packet.machine_recommendation == "needs_human_attention"
        assert "checks still pending" in packet.machine_recommendation_reason

    def test_build_packet_raises_when_pr_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: None,
        )
        with pytest.raises(_GhError, match="not found"):
            _build_packet("9999", repo_override=None)


# --- JSON output schema ----------------------------------------------------


class TestJsonOutput:
    def test_queue_item_to_dict_keys(self) -> None:
        item = _classify_pr(_make_pr())
        d = item.to_dict()
        for key in (
            "number",
            "title",
            "url",
            "head_sha",
            "author",
            "is_draft",
            "mergeable",
            "labels",
            "additions",
            "deletions",
            "changed_files",
            "checks_summary",
            "lane",
            "lane_reason",
        ):
            assert key in d, f"QueueItem dict missing key: {key}"

    def test_packet_to_dict_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: _make_pr(number=1, files=["aragora/cli/main.py"]),
        )
        packet = _build_packet("1", repo_override=None)
        d = packet.to_dict()
        for key in (
            "pr_number",
            "title",
            "url",
            "head_sha",
            "author",
            "is_draft",
            "additions",
            "deletions",
            "changed_files",
            "touched_subsystems",
            "high_risk_paths_touched",
            "checks_summary",
            "risk_flags",
            "machine_recommendation",
            "machine_recommendation_reason",
            "generated_at",
            "advisory_only",
            "settlement_note",
        ):
            assert key in d, f"ReviewPacket dict missing key: {key}"
        # ReviewPacket.advisory_only must always be True (signature property).
        assert d["advisory_only"] is True

    def test_packet_json_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: _make_pr(number=1, files=["aragora/cli/main.py"]),
        )
        packet = _build_packet("1", repo_override=None)
        roundtrip = json.loads(json.dumps(packet.to_dict()))
        assert roundtrip["pr_number"] == 1
        assert roundtrip["advisory_only"] is True


# --- cmd_review_queue dispatch + parser ------------------------------------


class TestCommandDispatch:
    def test_parser_registers_build_and_packet(self) -> None:
        root = argparse.ArgumentParser()
        sub = root.add_subparsers()
        add_review_queue_parser(sub)
        # build invocation parses
        ns_build = root.parse_args(["review-queue", "build", "--limit", "5", "--json"])
        assert ns_build.review_queue_command == "build"
        assert ns_build.limit == 5
        assert ns_build.json is True
        # packet invocation parses
        ns_packet = root.parse_args(["review-queue", "packet", "6280", "--json"])
        assert ns_packet.review_queue_command == "packet"
        assert ns_packet.pr == "6280"

    def test_cmd_review_queue_with_no_subcommand_returns_2(self) -> None:
        ns = argparse.Namespace(review_queue_command=None)
        rc = cmd_review_queue(ns)
        assert rc == 2

    def test_build_command_renders_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: [_make_pr(number=1)],
        )
        ns = argparse.Namespace(
            review_queue_command="build",
            limit=10,
            ready_only=False,
            include_parked=False,
            json=False,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_review_queue(ns)
        assert rc == 0
        out = buf.getvalue()
        assert "Review queue" in out
        assert "advisory only" in out

    def test_build_command_json_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: [_make_pr(number=1), _make_pr(number=2)],
        )
        ns = argparse.Namespace(
            review_queue_command="build",
            limit=10,
            ready_only=False,
            include_parked=False,
            json=True,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_review_queue(ns)
        assert rc == 0
        payload = json.loads(buf.getvalue())
        assert isinstance(payload, list)
        assert {item["number"] for item in payload} == {1, 2}

    def test_packet_command_json_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: _make_pr(number=42, files=["aragora/cli/main.py"]),
        )
        ns = argparse.Namespace(
            review_queue_command="packet",
            pr="42",
            repo=None,
            json=True,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_review_queue(ns)
        assert rc == 0
        payload = json.loads(buf.getvalue())
        assert payload["pr_number"] == 42
        assert payload["advisory_only"] is True
        assert payload["settlement_note"] == ADVISORY_NOTE
