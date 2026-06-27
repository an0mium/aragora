"""Tests for ``aragora.policy.predicate_oracle``.

Covers:
- predicate-string parsing
- each evaluator's happy-path and error paths (with subprocess mocked)
- dispatcher behavior for unknown predicates and parse errors
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from aragora.policy import (
    EVALUATORS,
    PredicateParseError,
    PredicateResult,
    evaluate_all,
    evaluate_predicate,
    parse_predicate,
)
from aragora.policy import predicate_oracle as oracle_module


# ---------- parse_predicate ----------


def test_parse_simple_int_arg() -> None:
    assert parse_predicate("pr_merged(7336)") == ("pr_merged", ["7336"])


def test_parse_quoted_string_arg() -> None:
    assert parse_predicate('tests_pass("tests/foo.py")') == (
        "tests_pass",
        ["tests/foo.py"],
    )


def test_parse_two_args() -> None:
    assert parse_predicate("pr_state(7336, OPEN)") == ("pr_state", ["7336", "OPEN"])


def test_parse_zero_args() -> None:
    assert parse_predicate("status()") == ("status", [])


def test_parse_invalid_raises() -> None:
    with pytest.raises(PredicateParseError):
        parse_predicate("not a predicate")
    with pytest.raises(PredicateParseError):
        parse_predicate("missing_close_paren(")


# ---------- evaluate_file_exists / dir_exists (pure stdlib, no mock needed) ----------


def test_file_exists_true(tmp_path: Path) -> None:
    target = tmp_path / "real.txt"
    target.write_text("x")
    result = evaluate_predicate(f"file_exists({target})")
    assert result.satisfied is True
    assert result.error is None
    assert "isfile" in result.evidence


def test_file_exists_false(tmp_path: Path) -> None:
    result = evaluate_predicate(f"file_exists({tmp_path / 'missing.txt'})")
    assert result.satisfied is False
    assert result.error is None


def test_file_exists_wrong_arg_count() -> None:
    result = evaluate_predicate("file_exists()")
    assert result.satisfied is False
    assert result.error and "expected 1 arg" in result.error


def test_dir_exists_true(tmp_path: Path) -> None:
    result = evaluate_predicate(f"dir_exists({tmp_path})")
    assert result.satisfied is True
    assert result.error is None


def test_dir_exists_false_for_file(tmp_path: Path) -> None:
    target = tmp_path / "afile"
    target.write_text("x")
    result = evaluate_predicate(f"dir_exists({target})")
    assert result.satisfied is False  # isfile, not isdir


# ---------- evaluate_pr_merged (mock subprocess) ----------


def _fake_proc(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_pr_merged_when_gh_reports_mergedAt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        oracle_module,
        "_run",
        lambda cmd, timeout=15.0: _fake_proc(
            0, stdout=json.dumps({"mergedAt": "2026-05-19T18:13:28Z"})
        ),
    )
    result = evaluate_predicate("pr_merged(7336)")
    assert result.satisfied is True
    assert result.error is None
    assert "merged_at" in result.evidence


def test_pr_merged_when_mergedAt_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        oracle_module,
        "_run",
        lambda cmd, timeout=15.0: _fake_proc(0, stdout=json.dumps({"mergedAt": None})),
    )
    result = evaluate_predicate("pr_merged(7336)")
    assert result.satisfied is False
    assert result.error is None


def test_pr_merged_when_gh_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, timeout=15.0):
        raise FileNotFoundError("gh missing")

    monkeypatch.setattr(oracle_module, "_run", fake_run)
    result = evaluate_predicate("pr_merged(7336)")
    assert result.satisfied is False
    assert result.error and "gh unavailable" in result.error


def test_pr_merged_when_gh_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        oracle_module, "_run", lambda cmd, timeout=15.0: _fake_proc(1, stderr="not found")
    )
    result = evaluate_predicate("pr_merged(99999)")
    assert result.satisfied is False
    assert result.error and "gh exit" in result.error


def test_pr_merged_non_int_arg(monkeypatch: pytest.MonkeyPatch) -> None:
    result = evaluate_predicate("pr_merged(foo)")
    assert result.satisfied is False
    assert result.error and "must be int" in result.error


# ---------- evaluate_pr_open / pr_state ----------


def test_pr_open_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        oracle_module,
        "_run",
        lambda cmd, timeout=15.0: _fake_proc(0, stdout=json.dumps({"state": "OPEN"})),
    )
    result = evaluate_predicate("pr_open(7327)")
    assert result.satisfied is True
    assert result.evaluator == "pr_open"


def test_pr_state_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        oracle_module,
        "_run",
        lambda cmd, timeout=15.0: _fake_proc(0, stdout=json.dumps({"state": "MERGED"})),
    )
    result = evaluate_predicate("pr_state(7327, MERGED)")
    assert result.satisfied is True


def test_pr_state_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        oracle_module,
        "_run",
        lambda cmd, timeout=15.0: _fake_proc(0, stdout=json.dumps({"state": "OPEN"})),
    )
    result = evaluate_predicate("pr_state(7327, MERGED)")
    assert result.satisfied is False


# ---------- evaluate_branch_exists ----------


def test_branch_exists_local_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(cmd, timeout=15.0):
        calls.append(list(cmd))
        if "branch" in cmd:
            return _fake_proc(0, stdout="  main\n")
        return _fake_proc(0, stdout="")

    monkeypatch.setattr(oracle_module, "_run", fake_run)
    result = evaluate_predicate("branch_exists(main)")
    assert result.satisfied is True


def test_branch_exists_remote_hit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, timeout=15.0):
        if cmd[1] == "branch":
            return _fake_proc(0, stdout="")
        return _fake_proc(0, stdout="abc123\trefs/heads/foo\n")

    monkeypatch.setattr(oracle_module, "_run", fake_run)
    result = evaluate_predicate("branch_exists(foo)")
    assert result.satisfied is True


def test_branch_exists_neither(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oracle_module, "_run", lambda cmd, timeout=15.0: _fake_proc(0, stdout=""))
    result = evaluate_predicate("branch_exists(missing)")
    assert result.satisfied is False


# ---------- evaluate_commit_landed ----------


def test_commit_landed_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oracle_module, "_run", lambda cmd, timeout=15.0: _fake_proc(0))
    result = evaluate_predicate("commit_landed(abc123)")
    assert result.satisfied is True


def test_commit_landed_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oracle_module, "_run", lambda cmd, timeout=15.0: _fake_proc(1))
    result = evaluate_predicate("commit_landed(deadbeef)")
    assert result.satisfied is False


# ---------- evaluate_issue_closed ----------


def test_issue_closed_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        oracle_module,
        "_run",
        lambda cmd, timeout=15.0: _fake_proc(0, stdout=json.dumps({"state": "CLOSED"})),
    )
    result = evaluate_predicate("issue_closed(123)")
    assert result.satisfied is True


def test_issue_closed_when_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        oracle_module,
        "_run",
        lambda cmd, timeout=15.0: _fake_proc(0, stdout=json.dumps({"state": "OPEN"})),
    )
    result = evaluate_predicate("issue_closed(123)")
    assert result.satisfied is False


# ---------- evaluate_tests_pass ----------


def test_tests_pass_path_missing(tmp_path: Path) -> None:
    result = evaluate_predicate(f"tests_pass({tmp_path / 'nope.py'})")
    assert result.satisfied is False
    assert result.error is None  # not an oracle error, just unsatisfied
    assert "path_missing" in result.evidence


def test_tests_pass_pytest_exit_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "test_x.py"
    target.write_text("def test_x():\n    assert True\n")
    monkeypatch.setattr(
        oracle_module, "_run", lambda cmd, timeout=300.0: _fake_proc(0, stdout="1 passed")
    )
    result = evaluate_predicate(f"tests_pass({target})")
    assert result.satisfied is True


def test_tests_pass_pytest_exit_nonzero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "test_x.py"
    target.write_text("def test_x():\n    assert False\n")
    monkeypatch.setattr(
        oracle_module, "_run", lambda cmd, timeout=300.0: _fake_proc(1, stdout="1 failed")
    )
    result = evaluate_predicate(f"tests_pass({target})")
    assert result.satisfied is False


# ---------- dispatcher ----------


def test_unknown_predicate_returns_error() -> None:
    result = evaluate_predicate("does_not_exist(x)")
    assert result.satisfied is False
    assert result.error and "unknown predicate" in result.error


def test_malformed_predicate_returns_error() -> None:
    result = evaluate_predicate("not a predicate")
    assert result.satisfied is False
    assert result.error is not None


def test_evaluate_all_preserves_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f1 = tmp_path / "a.txt"
    f1.write_text("x")
    results = evaluate_all(
        [
            f"file_exists({f1})",
            f"file_exists({tmp_path / 'missing'})",
        ]
    )
    assert len(results) == 2
    assert results[0].satisfied is True
    assert results[1].satisfied is False


def test_registry_lists_all_evaluators() -> None:
    expected = {
        "pr_merged",
        "pr_open",
        "pr_state",
        "tests_pass",
        "file_exists",
        "dir_exists",
        "branch_exists",
        "commit_landed",
        "issue_closed",
    }
    assert expected.issubset(EVALUATORS.keys())


def test_predicate_result_is_frozen() -> None:
    r = PredicateResult(
        predicate="x",
        satisfied=True,
        evidence="",
        evaluator="t",
    )
    with pytest.raises(Exception):  # frozen dataclasses raise FrozenInstanceError
        r.satisfied = False  # type: ignore[misc]
