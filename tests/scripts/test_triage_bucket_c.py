"""Tests for ``scripts/triage_bucket_c.py``.

Fixture-driven; never invokes real ``gh``. Every test constructs a
synthetic Stage 1 classifier payload plus a response map and
asserts that the decisions match the policy.

Coverage:
  - dry-run never mutates (no gh calls captured beyond view-files)
  - --apply enacts y → ready + comment
  - --apply enacts n → close
  - --apply d is no-op
  - held PR is hard-skipped with operator override
  - protected-path tripwire blocks advance/close even on Bucket C
  - non-Bucket-C entries are silently dropped from the result set
  - PRs without a response get NO_RESPONSE_SKIPPED
  - JSON response file round-trips
  - receipt is written on --apply
  - tripwire hit forces non-zero exit
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_module() -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / "triage_bucket_c.py"
    spec = importlib.util.spec_from_file_location("triage_bucket_c_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tbc = _load_module()


def make_triage_payload(*entries: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": {
            "A": sum(1 for e in entries if e.get("bucket") == "A"),
            "B": sum(1 for e in entries if e.get("bucket") == "B"),
            "C": sum(1 for e in entries if e.get("bucket") == "C"),
            "D": sum(1 for e in entries if e.get("bucket") == "D"),
        },
        "results": list(entries),
    }


def bucket_c(
    pr_number: int,
    *,
    title: str | None = None,
    reason: str = "draft",
) -> dict[str, Any]:
    return {
        "pr_number": pr_number,
        "title": title or f"PR #{pr_number}",
        "bucket": "C",
        "reason": reason,
        "recommended_action": "READY?",
    }


def bucket_a(pr_number: int) -> dict[str, Any]:
    return {
        "pr_number": pr_number,
        "title": f"PR #{pr_number}",
        "bucket": "A",
        "reason": "all gates clean",
        "recommended_action": "MERGE",
    }


def live_view_key(pr_number: int) -> tuple[str, ...]:
    return (
        "gh",
        "pr",
        "view",
        str(pr_number),
        "--repo",
        tbc.GH_REPO,
        "--json",
        "state,isDraft,headRefOid,files",
    )


def live_view_stdout(
    *,
    state: str = "OPEN",
    head: str = "head-1",
    files: list[str] | None = None,
) -> str:
    return json.dumps(
        {
            "state": state,
            "isDraft": True,
            "headRefOid": head,
            "files": [{"path": path} for path in (files or [])],
        }
    )


def triage_key() -> tuple[str, ...]:
    return ("python3", str(tbc.TRIAGE_SCRIPT), "--json")


def triage_stdout(*entries: dict[str, Any]) -> str:
    return json.dumps(make_triage_payload(*entries))


class _RunnerRecorder:
    """Captures every subprocess call. Returns success by default."""

    def __init__(
        self,
        *,
        failure_for: dict[tuple[str, ...], int] | None = None,
        stdout_for: dict[tuple[str, ...], str] | None = None,
    ):
        self.calls: list[tuple[str, ...]] = []
        self._failure_for = failure_for or {}
        self._stdout_for = stdout_for or {}

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        key = tuple(args)
        self.calls.append(key)
        rc = self._failure_for.get(key, 0)
        stdout = self._stdout_for.get(key, "")
        return subprocess.CompletedProcess(args=args, returncode=rc, stdout=stdout, stderr="")


class TestDecideDryRun:
    def test_dry_run_advance_emits_would_advance(self):
        payload = make_triage_payload(bucket_c(9001))
        results = tbc.decide(
            payload,
            {9001: "y"},
            apply=False,
            files_provider=lambda n: [],
        )
        assert len(results) == 1
        assert results[0].pr_number == 9001
        assert results[0].status == tbc.STATUS_WOULD_ADVANCE
        # No real gh mutation happened.
        assert len(results[0].gh_commands) >= 1

    def test_dry_run_close_emits_would_close(self):
        payload = make_triage_payload(bucket_c(9001, reason="non-trusted author"))
        results = tbc.decide(
            payload,
            {9001: "n"},
            apply=False,
            files_provider=lambda n: [],
        )
        assert results[0].status == tbc.STATUS_WOULD_CLOSE

    def test_dry_run_defer_emits_deferred(self):
        payload = make_triage_payload(bucket_c(9001))
        results = tbc.decide(
            payload,
            {9001: "d"},
            apply=False,
            files_provider=lambda n: [],
        )
        assert results[0].status == tbc.STATUS_DEFERRED


class TestDecideApply:
    def test_apply_y_calls_gh_ready_and_comment(self):
        payload = make_triage_payload(bucket_c(9001))
        recorder = _RunnerRecorder(
            stdout_for={
                live_view_key(9001): live_view_stdout(files=["tests/example.py"]),
                triage_key(): triage_stdout(bucket_c(9001)),
            }
        )
        results = tbc.decide(
            payload,
            {9001: "y"},
            apply=True,
            runner=recorder,
            files_provider=lambda n: [],
        )
        # Must have called both gh pr ready and gh pr comment.
        ready_calls = [c for c in recorder.calls if c[:3] == ("gh", "pr", "ready")]
        comment_calls = [c for c in recorder.calls if c[:3] == ("gh", "pr", "comment")]
        assert ready_calls == [("gh", "pr", "ready", "9001", "--repo", tbc.GH_REPO)]
        assert len(comment_calls) == 1
        assert comment_calls[0][:6] == (
            "gh",
            "pr",
            "comment",
            "9001",
            "--repo",
            tbc.GH_REPO,
        )
        assert results[0].status == tbc.STATUS_ADVANCED

    def test_apply_n_calls_gh_close(self):
        payload = make_triage_payload(bucket_c(9001, reason="CI red"))
        recorder = _RunnerRecorder(
            stdout_for={
                live_view_key(9001): live_view_stdout(files=["tests/example.py"]),
                triage_key(): triage_stdout(bucket_c(9001)),
            }
        )
        results = tbc.decide(
            payload,
            {9001: "n"},
            apply=True,
            runner=recorder,
            files_provider=lambda n: [],
        )
        close_calls = [c for c in recorder.calls if c[:3] == ("gh", "pr", "close")]
        assert len(close_calls) == 1
        assert close_calls[0][:6] == (
            "gh",
            "pr",
            "close",
            "9001",
            "--repo",
            tbc.GH_REPO,
        )
        assert results[0].status == tbc.STATUS_CLOSED

    def test_apply_d_makes_no_gh_calls(self):
        payload = make_triage_payload(bucket_c(9001))
        recorder = _RunnerRecorder()
        results = tbc.decide(
            payload,
            {9001: "d"},
            apply=True,
            runner=recorder,
            files_provider=lambda n: [],
        )
        # No mutation calls. (files_provider injected — no view call.)
        gh_mutation_calls = [
            c
            for c in recorder.calls
            if c[:3] in (("gh", "pr", "ready"), ("gh", "pr", "close"), ("gh", "pr", "comment"))
        ]
        assert gh_mutation_calls == []
        assert results[0].status == tbc.STATUS_DEFERRED


class TestTripwires:
    def test_held_pr_is_skipped_regardless_of_response(self):
        # 7252 is on the policy hold list.
        payload = make_triage_payload(bucket_c(7252))
        recorder = _RunnerRecorder()
        results = tbc.decide(
            payload,
            {7252: "y"},
            apply=True,
            runner=recorder,
            files_provider=lambda n: [],
        )
        assert results[0].status == tbc.STATUS_HELD
        # No mutating gh calls.
        mutation_calls = [
            c for c in recorder.calls if c[:3] in (("gh", "pr", "ready"), ("gh", "pr", "close"))
        ]
        assert mutation_calls == []

    def test_protected_path_blocks_advance(self):
        payload = make_triage_payload(bucket_c(9001))
        recorder = _RunnerRecorder()
        results = tbc.decide(
            payload,
            {9001: "y"},
            apply=True,
            runner=recorder,
            files_provider=lambda n: ["scripts/nomic_loop.py"],
        )
        assert results[0].status == tbc.STATUS_PROTECTED
        mutation_calls = [
            c for c in recorder.calls if c[:3] in (("gh", "pr", "ready"), ("gh", "pr", "close"))
        ]
        assert mutation_calls == []

    def test_workflow_path_blocks_close(self):
        payload = make_triage_payload(bucket_c(9001))
        recorder = _RunnerRecorder()
        results = tbc.decide(
            payload,
            {9001: "n"},
            apply=True,
            runner=recorder,
            files_provider=lambda n: [".github/workflows/ci.yml"],
        )
        assert results[0].status == tbc.STATUS_PROTECTED

    def test_file_fetch_failure_fails_closed_before_mutation(self):
        payload = make_triage_payload(bucket_c(9001))
        recorder = _RunnerRecorder(failure_for={live_view_key(9001): 1})

        results = tbc.decide(payload, {9001: "y"}, apply=True, runner=recorder)

        assert results[0].status == tbc.STATUS_PROTECTED
        assert "could not verify protected paths" in results[0].reason
        mutation_calls = [
            c
            for c in recorder.calls
            if c[:3] in (("gh", "pr", "ready"), ("gh", "pr", "close"), ("gh", "pr", "comment"))
        ]
        assert mutation_calls == []

    def test_file_fetch_malformed_json_fails_closed_before_mutation(self):
        payload = make_triage_payload(bucket_c(9001))
        recorder = _RunnerRecorder(stdout_for={live_view_key(9001): "not-json"})

        results = tbc.decide(payload, {9001: "n"}, apply=True, runner=recorder)

        assert results[0].status == tbc.STATUS_PROTECTED
        assert "non-JSON" in results[0].reason
        mutation_calls = [
            c
            for c in recorder.calls
            if c[:3] in (("gh", "pr", "ready"), ("gh", "pr", "close"), ("gh", "pr", "comment"))
        ]
        assert mutation_calls == []

    def test_file_fetch_malformed_files_payload_fails_closed_before_mutation(self):
        payload = make_triage_payload(bucket_c(9001))
        recorder = _RunnerRecorder(
            stdout_for={
                live_view_key(9001): json.dumps(
                    {"state": "OPEN", "isDraft": True, "headRefOid": "head-1", "files": "bad"}
                )
            }
        )

        results = tbc.decide(payload, {9001: "y"}, apply=True, runner=recorder)

        assert results[0].status == tbc.STATUS_PROTECTED
        assert "missing files list" in results[0].reason
        mutation_calls = [
            c
            for c in recorder.calls
            if c[:3] in (("gh", "pr", "ready"), ("gh", "pr", "close"), ("gh", "pr", "comment"))
        ]
        assert mutation_calls == []

    def test_live_head_change_fails_closed_before_mutation(self):
        payload = make_triage_payload(bucket_c(9001))
        calls = 0

        def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
            nonlocal calls
            key = tuple(args)
            if key == live_view_key(9001):
                calls += 1
                head = "head-1" if calls == 1 else "head-2"
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout=live_view_stdout(head=head, files=["tests/example.py"]),
                    stderr="",
                )
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        results = tbc.decide(payload, {9001: "y"}, apply=True, runner=runner)

        assert results[0].status == tbc.STATUS_LIVE_CHECK_FAILED
        assert "live PR head changed" in results[0].reason

    def test_live_tripwire_change_fails_closed_before_mutation(self):
        payload = make_triage_payload(bucket_c(9001))
        calls = 0
        seen: list[tuple[str, ...]] = []

        def runner(args: list[str]) -> subprocess.CompletedProcess[str]:
            nonlocal calls
            key = tuple(args)
            seen.append(key)
            if key == live_view_key(9001):
                calls += 1
                files = ["tests/example.py"] if calls == 1 else ["scripts/nomic_loop.py"]
                return subprocess.CompletedProcess(
                    args=args,
                    returncode=0,
                    stdout=live_view_stdout(head="head-1", files=files),
                    stderr="",
                )
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        results = tbc.decide(payload, {9001: "n"}, apply=True, runner=runner)

        assert results[0].status == tbc.STATUS_LIVE_CHECK_FAILED
        assert "live PR tripwire changed" in results[0].reason
        mutation_calls = [
            c
            for c in seen
            if c[:3] in (("gh", "pr", "ready"), ("gh", "pr", "close"), ("gh", "pr", "comment"))
        ]
        assert mutation_calls == []

    def test_live_bucket_change_fails_closed_before_mutation(self):
        payload = make_triage_payload(bucket_c(9001))
        recorder = _RunnerRecorder(
            stdout_for={
                live_view_key(9001): live_view_stdout(files=["tests/example.py"]),
                triage_key(): triage_stdout(bucket_a(9001)),
            }
        )

        results = tbc.decide(payload, {9001: "y"}, apply=True, runner=recorder)

        assert results[0].status == tbc.STATUS_LIVE_CHECK_FAILED
        assert "live PR bucket" in results[0].reason
        mutation_calls = [
            c
            for c in recorder.calls
            if c[:3] in (("gh", "pr", "ready"), ("gh", "pr", "close"), ("gh", "pr", "comment"))
        ]
        assert mutation_calls == []


class TestFiltering:
    def test_non_bucket_c_entries_are_dropped(self):
        payload = make_triage_payload(bucket_a(9001), bucket_c(9002))
        results = tbc.decide(payload, {9002: "y"}, apply=False, files_provider=lambda n: [])
        # Only the Bucket C entry shows up in results.
        assert [r.pr_number for r in results] == [9002]

    def test_no_response_yields_no_response_skipped(self):
        payload = make_triage_payload(bucket_c(9001), bucket_c(9002))
        results = tbc.decide(payload, {9001: "y"}, apply=False, files_provider=lambda n: [])
        statuses = {r.pr_number: r.status for r in results}
        assert statuses[9001] == tbc.STATUS_WOULD_ADVANCE
        assert statuses[9002] == tbc.STATUS_NO_RESPONSE


class TestResponseFile:
    def test_json_response_round_trip(self, tmp_path: Path):
        response_file = tmp_path / "responses.json"
        response_file.write_text(json.dumps({"9001": "y", "9002": "n", "9003": "d"}))
        responses = tbc.load_responses_file(response_file)
        assert responses == {9001: "y", 9002: "n", 9003: "d"}

    def test_invalid_response_value_raises(self, tmp_path: Path):
        response_file = tmp_path / "responses.json"
        response_file.write_text(json.dumps({"9001": "maybe"}))
        with pytest.raises(ValueError):
            tbc.load_responses_file(response_file)

    def test_response_file_with_hash_prefix_keys(self, tmp_path: Path):
        # Accepts "#9001" as well as "9001".
        response_file = tmp_path / "responses.json"
        response_file.write_text(json.dumps({"#9001": "y"}))
        responses = tbc.load_responses_file(response_file)
        assert responses == {9001: "y"}


class TestReceipt:
    def test_receipt_is_written(self, tmp_path: Path):
        results = [
            tbc.EntryResult(
                pr_number=9001,
                title="example",
                response="y",
                status=tbc.STATUS_ADVANCED,
                reason="marked ready + commented",
            )
        ]
        receipt_md = tbc.render_receipt(results, apply=True)
        path = tbc.write_receipt(receipt_md, receipt_dir=tmp_path)
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "#9001" in text
        assert tbc.STATUS_ADVANCED in text
        assert "Bucket C receipt" in text


class TestInteractive:
    def test_interactive_reads_y_n_d(self):
        bucket_c_entries = [bucket_c(9001), bucket_c(9002), bucket_c(9003)]
        stdin = io.StringIO("y\nn\nd\n")
        stdout = io.StringIO()
        responses = tbc.collect_responses_interactive(bucket_c_entries, stdin=stdin, stdout=stdout)
        assert responses == {9001: "y", 9002: "n", 9003: "d"}

    def test_interactive_treats_empty_as_defer(self):
        bucket_c_entries = [bucket_c(9001)]
        stdin = io.StringIO("\n")
        stdout = io.StringIO()
        responses = tbc.collect_responses_interactive(bucket_c_entries, stdin=stdin, stdout=stdout)
        assert responses == {9001: "d"}

    def test_interactive_treats_garbage_as_defer(self):
        bucket_c_entries = [bucket_c(9001)]
        stdin = io.StringIO("zzz\n")
        stdout = io.StringIO()
        responses = tbc.collect_responses_interactive(bucket_c_entries, stdin=stdin, stdout=stdout)
        assert responses == {9001: "d"}
