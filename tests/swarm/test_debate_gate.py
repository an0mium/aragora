"""Unit tests for the publish-time debate gate."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from aragora.swarm.debate_gate import DebateGate, DebateGateConfig, DebateGateRequest


def _request(**overrides: object) -> DebateGateRequest:
    payload = {
        "issue_number": 101,
        "issue_title": "Tighten publish metadata handling",
        "issue_body": "Acceptance Criteria:\n- keep publish metadata truthful\n",
        "source_branch": "codex/issue-101",
        "target_branch": "main",
        "commit_shas": ["abc123"],
        "changed_files": ["aragora/swarm/boss_loop.py"],
        "tests_run": ["python3 -m pytest tests/swarm/test_boss_loop.py -q"],
        "verification_results": [{"name": "pytest", "passed": True}],
        "receipt_id": "lane-101",
    }
    payload.update(overrides)
    return DebateGateRequest(**payload)


def _payload(*, changed_files: list[str] | None = None) -> dict[str, object]:
    return {
        "changed_files": changed_files or ["aragora/swarm/boss_loop.py"],
        "diff_stat": " aragora/swarm/boss_loop.py | 12 ++++++++++--",
        "diff_excerpt": "@@ -1,2 +1,4 @@\n+guard publish metadata\n",
        "verification_summary": {
            "tests_run": ["python3 -m pytest tests/swarm/test_boss_loop.py -q"],
            "checks_observed": 1,
            "checks_passed": 1,
            "checks_failed": 0,
            "check_names": ["pytest"],
        },
    }


def test_debate_gate_skips_when_disabled() -> None:
    llm_caller = MagicMock()
    diff_loader = MagicMock()
    gate = DebateGate(
        repo_root=Path.cwd(),
        config=DebateGateConfig(enabled=False),
        llm_caller=llm_caller,
        diff_loader=diff_loader,
    )

    result = gate.evaluate(_request())

    assert result.verdict == "skipped_disabled"
    assert result.publication_allowed is True
    assert result.ran is False
    llm_caller.assert_not_called()
    diff_loader.assert_not_called()


def test_debate_gate_passes_structured_response() -> None:
    gate = DebateGate(
        repo_root=Path.cwd(),
        config=DebateGateConfig(enabled=True),
        llm_caller=lambda prompt, agent_type, timeout_seconds: (
            '{"passed": true, "confidence": 0.82, "reason": "Diff matches the verified task.", '
            '"concerns": []}'
        ),
        diff_loader=lambda repo_root, request, config: _payload(),
    )

    result = gate.evaluate(_request())

    assert result.verdict == "passed"
    assert result.publication_allowed is True
    assert result.passed is True
    assert result.confidence == 0.82
    assert result.reason == "Diff matches the verified task."
    assert result.changed_files == ["aragora/swarm/boss_loop.py"]


def test_debate_gate_blocks_on_embedded_json_response() -> None:
    gate = DebateGate(
        repo_root=Path.cwd(),
        config=DebateGateConfig(enabled=True),
        llm_caller=lambda prompt, agent_type, timeout_seconds: (
            "Review complete.\n"
            '{"passed": false, "confidence": 0.34, "reason": "Observed a publish-time truth regression.", '
            '"concerns": ["publish metadata omits blocker context"]}'
        ),
        diff_loader=lambda repo_root, request, config: _payload(
            changed_files=["aragora/swarm/boss_loop.py", "tests/swarm/test_boss_loop.py"]
        ),
    )

    result = gate.evaluate(_request())

    assert result.verdict == "blocked"
    assert result.publication_allowed is False
    assert result.passed is False
    assert result.concerns == ["publish metadata omits blocker context"]
    assert result.reason == "Observed a publish-time truth regression."


def test_debate_gate_runtime_failure_fails_open_by_default() -> None:
    gate = DebateGate(
        repo_root=Path.cwd(),
        config=DebateGateConfig(enabled=True),
        llm_caller=lambda prompt, agent_type, timeout_seconds: (_ for _ in ()).throw(
            RuntimeError("model unavailable")
        ),
        diff_loader=lambda repo_root, request, config: _payload(),
    )

    result = gate.evaluate(_request())

    assert result.verdict == "fail_open"
    assert result.publication_allowed is True
    assert result.fail_open_used is True
    assert "model unavailable" in result.reason


def test_debate_gate_runtime_failure_can_fail_closed() -> None:
    gate = DebateGate(
        repo_root=Path.cwd(),
        config=DebateGateConfig(enabled=True, fail_closed=True),
        llm_caller=lambda prompt, agent_type, timeout_seconds: (_ for _ in ()).throw(
            RuntimeError("network unreachable")
        ),
        diff_loader=lambda repo_root, request, config: _payload(),
    )

    result = gate.evaluate(_request())

    assert result.verdict == "blocked"
    assert result.publication_allowed is False
    assert result.fail_open_used is False
    assert "network unreachable" in result.reason
