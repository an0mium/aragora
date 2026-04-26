from __future__ import annotations

import pytest

from aragora.swarm.supervisor_probes import (
    _derive_blocker_evidence,
    _summarize_verification_failure,
    _verification_blocker_evidence,
)
from aragora.swarm.worker_launcher import WorkerProcess


def _worker_result(*, stdout: str = "", stderr: str = "") -> WorkerProcess:
    return WorkerProcess(
        work_order_id="wo-1",
        agent="codex",
        worktree_path="/tmp/worktree",
        branch="codex/test",
        pid=123,
        session_id="session-1",
        lease_id="lease-1",
        completed_at="2026-04-17T00:00:00Z",
        exit_code=1,
        stdout=stdout,
        stderr=stderr,
        diff="",
        initial_head="abc123",
        head_sha="def456",
        commit_shas=[],
        changed_paths=[],
        tests_run=[],
        expected_tests=[],
        prompt_chars=0,
        enriched_context_chars=0,
    )


def test_summarize_verification_failure_returns_first_failed_entry() -> None:
    summary = _summarize_verification_failure(
        [
            "ignore-me",
            {"passed": True, "command": "python -m pytest tests/pass.py -q"},
            {
                "passed": False,
                "command": "python -m pytest tests/fail.py -q",
                "exit_code": 2,
                "stdout": "a" * 900,
                "stderr": "b" * 900,
            },
            {"passed": False, "command": "python -m pytest tests/later.py -q"},
        ]
    )

    assert summary["command"] == "python -m pytest tests/fail.py -q"
    assert summary["exit_code"] == 2
    assert summary["stdout_tail"] == "a" * 800
    assert summary["stderr_tail"] == "b" * 800


def test_verification_blocker_evidence_formats_command_and_exit_code() -> None:
    evidence = _verification_blocker_evidence(
        {
            "command": "python -m pytest tests/fail.py -q",
            "exit_code": 5,
            "stderr_tail": "  verification\n failed  ",
            "stdout_tail": "ignored stdout",
        }
    )

    assert evidence == "python -m pytest tests/fail.py -q (exit 5): verification failed"


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ({"blocker_evidence": "  direct\n evidence  "}, "direct evidence"),
        (
            {"metadata": {"blocker_evidence": "  metadata\n evidence  "}},
            "metadata evidence",
        ),
    ],
)
def test_derive_blocker_evidence_prefers_existing_compacted_evidence(
    item: dict[str, object],
    expected: str,
) -> None:
    assert _derive_blocker_evidence(item) == expected


def test_derive_blocker_evidence_prefers_verification_failure_over_other_sources() -> None:
    evidence = _derive_blocker_evidence(
        {
            "verification_results": [
                {
                    "passed": False,
                    "command": "python -m pytest tests/fail.py -q",
                    "exit_code": 1,
                    "stderr": "assert 1 == 2",
                }
            ],
            "dispatch_error": "dispatch failed",
        },
        result=_worker_result(stderr="worker stderr"),
        merge_gate={"blocked_reasons": ["merge gate blocked"]},
        reason="manual fallback",
    )

    assert evidence == "python -m pytest tests/fail.py -q (exit 1): assert 1 == 2"


def test_derive_blocker_evidence_uses_merge_gate_before_worker_output() -> None:
    evidence = _derive_blocker_evidence(
        {},
        result=_worker_result(stderr="worker stderr"),
        merge_gate={"blocked_reasons": ["  review gate still blocked  "]},
    )

    assert evidence == "review gate still blocked"


def test_derive_blocker_evidence_uses_worker_stderr_before_later_fallbacks() -> None:
    evidence = _derive_blocker_evidence(
        {
            "stdout_tail": "item stdout",
            "dispatch_error": "dispatch failed",
            "resource_error": "resource failed",
            "failure_reason": "final fallback",
        },
        result=_worker_result(stderr="worker stderr", stdout="worker stdout"),
        reason="manual fallback",
    )

    assert evidence == "worker stderr"


@pytest.mark.parametrize(
    ("item", "reason", "expected"),
    [
        ({"dispatch_error": "dispatch failed"}, None, "dispatch failed"),
        ({"resource_error": "resource failed"}, None, "resource failed"),
        ({}, "manual fallback", "manual fallback"),
        ({"failure_reason": "final fallback"}, None, "final fallback"),
        ({}, None, "needs_human"),
    ],
)
def test_derive_blocker_evidence_falls_back_through_remaining_sources(
    item: dict[str, object],
    reason: str | None,
    expected: str,
) -> None:
    assert _derive_blocker_evidence(item, reason=reason) == expected
