"""Tests for :mod:`aragora.pdb.input_loader`."""

from __future__ import annotations

import json
import subprocess
from typing import Any
from unittest.mock import patch

import pytest

from aragora.pdb import input_loader as il
from aragora.pdb.input_loader import (
    DEFAULT_DIFF_EXCERPT_CHARS,
    InputLoaderError,
    InputLoaderErrorReason,
    load_execution_input,
)
from aragora.pdb.protocol import PDBExecutionInput


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _fake_pr_payload(
    *,
    number: int = 4242,
    head_sha: str = "deadbeefcafe0011223344556677889900aabbcc",
    base_sha: str = "feedface0011",
    labels: list[str] | None = None,
    files: list[str] | None = None,
    additions: int = 20,
    deletions: int = 3,
    is_draft: bool = False,
    url: str | None = None,
) -> dict[str, Any]:
    return {
        "number": number,
        "title": "Tighten rate limiter tests",
        "body": "Adds ring-buffer invariants.",
        "url": url or f"https://github.com/synaptent/aragora/pull/{number}",
        "headRefOid": head_sha,
        "baseRefOid": base_sha,
        "isDraft": is_draft,
        "mergeable": "MERGEABLE",
        "reviewDecision": "REVIEW_REQUIRED",
        "labels": [{"name": lab} for lab in (labels or ["backend"])],
        "author": {"login": "armand"},
        "additions": additions,
        "deletions": deletions,
        "changedFiles": len(files or ["aragora/server/rate_limit.py"]),
        "statusCheckRollup": [{"conclusion": "SUCCESS", "status": "COMPLETED"}],
        "files": [{"path": p} for p in (files or ["aragora/server/rate_limit.py"])],
    }


def _run_gh_view_success(*, payload: dict[str, Any]) -> tuple[int, str, str]:
    return (0, json.dumps(payload), "")


class _RunGHStub:
    """Queue-based replacement for ``_run_gh`` inside the loader."""

    def __init__(self, responses: list[tuple[int, str, str]]) -> None:
        self._responses = list(responses)
        self.calls: list[list[str]] = []

    def __call__(self, args, *, capture: bool = True) -> tuple[int, str, str]:  # noqa: D401
        self.calls.append(list(args))
        if not self._responses:
            raise AssertionError(f"unexpected gh call: {args}")
        return self._responses.pop(0)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestLoadExecutionInputHappyPath:
    def test_builds_pdb_execution_input(self):
        payload = _fake_pr_payload(
            head_sha="aaaabbbbccccddddeeeeffff0011223344556677",
            base_sha="11112222333344445555",
            labels=["backend", "needs-review"],
            files=["aragora/server/rate_limit.py", "tests/server/test_rate_limit.py"],
            additions=40,
            deletions=5,
        )
        diff_text = "diff --git a/foo b/foo\n+pass\n"
        stub = _RunGHStub(
            [
                _run_gh_view_success(payload=payload),
                (0, diff_text, ""),
            ]
        )
        with patch.object(il, "_run_gh", stub):
            result = load_execution_input(pr_number=4242)

        assert isinstance(result.input, PDBExecutionInput)
        assert result.head_sha == "aaaabbbbccccddddeeeeffff0011223344556677"
        assert result.base_sha == "11112222333344445555"
        assert result.repo == "synaptent/aragora"
        assert result.input.binding.pr_number == 4242
        assert result.input.binding.head_sha == result.head_sha
        assert result.input.binding.base_sha == result.base_sha
        assert result.input.pr_title == "Tighten rate limiter tests"
        assert result.input.pr_body == "Adds ring-buffer invariants."
        assert result.input.labels == ("backend", "needs-review")
        assert "aragora/server/rate_limit.py" in result.input.changed_files
        assert result.input.diff_excerpt == diff_text
        assert result.input.panel_id == "protocol_b_default"
        assert result.input.validation_summary["additions"] == 40
        assert result.input.validation_summary["labels"] == ["backend", "needs-review"]
        # Panel models derived from the heuristic packet's resolved slots
        assert isinstance(result.panel_models, tuple)
        # One view call + one diff call
        assert len(stub.calls) == 2
        assert stub.calls[0][:3] == ["pr", "view", "4242"]
        assert stub.calls[1][:3] == ["pr", "diff", "4242"]

    def test_repo_override_is_forwarded(self):
        stub = _RunGHStub(
            [
                _run_gh_view_success(payload=_fake_pr_payload()),
                (0, "", ""),
            ]
        )
        with patch.object(il, "_run_gh", stub):
            result = load_execution_input(pr_number=10, repo="synaptent/aragora")

        assert "--repo" in stub.calls[0]
        assert "--repo" in stub.calls[1]
        assert result.repo == "synaptent/aragora"

    def test_diff_excerpt_truncated_at_char_limit(self):
        huge_diff = "diff --git a/x b/x\n" + ("+" + "y" * 200 + "\n") * 5000
        stub = _RunGHStub(
            [
                _run_gh_view_success(payload=_fake_pr_payload()),
                (0, huge_diff, ""),
            ]
        )
        with patch.object(il, "_run_gh", stub):
            result = load_execution_input(pr_number=7, diff_excerpt_char_limit=5000)

        assert len(result.input.diff_excerpt) <= 5000 + len("\n[diff truncated]\n") + 1
        assert result.input.diff_excerpt.endswith("[diff truncated]\n")

    def test_zero_diff_limit_skips_diff_call(self):
        stub = _RunGHStub(
            [
                _run_gh_view_success(payload=_fake_pr_payload()),
            ]
        )
        with patch.object(il, "_run_gh", stub):
            result = load_execution_input(pr_number=7, diff_excerpt_char_limit=0)

        assert result.input.diff_excerpt == ""
        # Only the view call should be made when diff is disabled
        assert len(stub.calls) == 1

    def test_diff_failure_is_nonfatal(self, caplog):
        stub = _RunGHStub(
            [
                _run_gh_view_success(payload=_fake_pr_payload()),
                (1, "", "gh pr diff failed"),
            ]
        )
        with patch.object(il, "_run_gh", stub), caplog.at_level("WARNING"):
            result = load_execution_input(pr_number=7)

        assert result.input.diff_excerpt == ""
        assert any("gh pr diff" in rec.message for rec in caplog.records)

    def test_panel_id_is_passed_through(self):
        stub = _RunGHStub(
            [
                _run_gh_view_success(payload=_fake_pr_payload()),
                (0, "diff", ""),
            ]
        )
        with patch.object(il, "_run_gh", stub):
            result = load_execution_input(pr_number=1, panel_id="custom_panel")

        assert result.input.panel_id == "custom_panel"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestLoadExecutionInputErrors:
    def test_invalid_pr_number_rejected(self):
        with pytest.raises(ValueError):
            load_execution_input(pr_number=0)
        with pytest.raises(ValueError):
            load_execution_input(pr_number=-5)

    def test_gh_missing_raises(self):
        def _raise_missing(args, *, capture: bool = True):
            raise FileNotFoundError("no gh")

        with patch.object(il.subprocess, "run", side_effect=FileNotFoundError("no gh")):
            with pytest.raises(InputLoaderError) as excinfo:
                load_execution_input(pr_number=1)
        assert excinfo.value.reason is InputLoaderErrorReason.GH_MISSING

    def test_gh_timeout_raises(self):
        def _raise_timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=5)

        with patch.object(il.subprocess, "run", side_effect=_raise_timeout):
            with pytest.raises(InputLoaderError) as excinfo:
                load_execution_input(pr_number=1)
        assert excinfo.value.reason is InputLoaderErrorReason.TIMEOUT

    def test_pr_not_found_classified(self):
        stub = _RunGHStub([(1, "", "could not find pull request 99")])
        with patch.object(il, "_run_gh", stub):
            with pytest.raises(InputLoaderError) as excinfo:
                load_execution_input(pr_number=99)
        assert excinfo.value.reason is InputLoaderErrorReason.PR_NOT_FOUND

    def test_authentication_error_classified(self):
        stub = _RunGHStub([(1, "", "authentication required; run gh auth login")])
        with patch.object(il, "_run_gh", stub):
            with pytest.raises(InputLoaderError) as excinfo:
                load_execution_input(pr_number=1)
        assert excinfo.value.reason is InputLoaderErrorReason.GH_AUTHENTICATION

    def test_generic_gh_error_classified(self):
        stub = _RunGHStub([(1, "", "some other failure")])
        with patch.object(il, "_run_gh", stub):
            with pytest.raises(InputLoaderError) as excinfo:
                load_execution_input(pr_number=1)
        assert excinfo.value.reason is InputLoaderErrorReason.GH_ERROR

    def test_malformed_json_raises(self):
        stub = _RunGHStub([(0, "not json at all", "")])
        with patch.object(il, "_run_gh", stub):
            with pytest.raises(InputLoaderError) as excinfo:
                load_execution_input(pr_number=1)
        assert excinfo.value.reason is InputLoaderErrorReason.MALFORMED_RESPONSE

    def test_non_object_json_raises(self):
        stub = _RunGHStub([(0, json.dumps([]), "")])
        with patch.object(il, "_run_gh", stub):
            with pytest.raises(InputLoaderError) as excinfo:
                load_execution_input(pr_number=1)
        assert excinfo.value.reason is InputLoaderErrorReason.MALFORMED_RESPONSE

    def test_empty_head_sha_raises(self):
        payload = _fake_pr_payload()
        payload["headRefOid"] = ""
        stub = _RunGHStub([_run_gh_view_success(payload=payload)])
        with patch.object(il, "_run_gh", stub):
            with pytest.raises(InputLoaderError) as excinfo:
                load_execution_input(pr_number=1)
        assert excinfo.value.reason is InputLoaderErrorReason.EMPTY_HEAD_SHA

    def test_missing_repo_raises(self):
        payload = _fake_pr_payload()
        payload["url"] = ""  # bypass the fallback in _fake_pr_payload
        stub = _RunGHStub([_run_gh_view_success(payload=payload)])
        with patch.object(il, "_run_gh", stub):
            with pytest.raises(InputLoaderError) as excinfo:
                load_execution_input(pr_number=1)
        assert excinfo.value.reason is InputLoaderErrorReason.MALFORMED_RESPONSE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestRepoParser:
    def test_standard_url(self):
        assert (
            il._repo_from_url("https://github.com/synaptent/aragora/pull/42") == "synaptent/aragora"
        )

    def test_http_url(self):
        assert il._repo_from_url("http://github.com/a/b/pull/1") == "a/b"

    def test_malformed_returns_empty(self):
        assert il._repo_from_url("not-a-url") == ""
        assert il._repo_from_url("") == ""


class TestChecksSummary:
    def test_counts_success_failure_pending(self):
        payload = [
            {"conclusion": "SUCCESS", "status": "COMPLETED"},
            {"conclusion": "FAILURE", "status": "COMPLETED"},
            {"status": "IN_PROGRESS"},
            {"conclusion": "SKIPPED"},
        ]
        out = il._summarize_checks(payload)
        assert out["success"] == 1
        assert out["failure"] == 1
        assert out["pending"] == 1
        assert out["total"] == 3


class TestTruncateDiff:
    def test_short_diff_passthrough(self):
        assert il._truncate_diff("abc\n", limit=10) == "abc\n"

    def test_long_diff_truncated(self):
        text = "aa\nbb\ncc\ndd\nee\nff\n"
        out = il._truncate_diff(text, limit=8)
        assert "[diff truncated]" in out
        assert len(out) < len(text) + 20
