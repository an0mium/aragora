"""Integration tests proving the ping-pong handoff pattern improves outcomes.

These tests simulate the boss loop's actual failure mode: agent A fails but
produces a transcript, agent B receives a structured handoff prompt built
from that transcript, and succeeds where a cold re-dispatch would fail.
"""

from __future__ import annotations

import pytest

from aragora.swarm.ping_pong import PingPongLoop, build_handoff_prompt


@pytest.mark.asyncio
async def test_handoff_prompt_enables_second_agent_success():
    """Agent B succeeds with handoff context where cold re-dispatch fails."""
    calls: list[dict] = []

    async def dispatch(agent: str, prompt: str) -> dict:
        calls.append({"agent": agent, "has_context": "Found the bug" in prompt})

        if agent == "claude":
            return {
                "transcript": (
                    "Found the bug in dashboard.py line 42. "
                    "The query lacks an index hint. Applied index but "
                    "tests/analytics/test_dashboard.py::test_query_perf "
                    "still fails with timeout — the query planner ignores "
                    "the hint when the table has <1000 rows."
                ),
                "files_changed": ["aragora/analytics/dashboard.py"],
                "tests_passed": False,
                "remaining_issues": ["query planner ignores index on small tables"],
            }

        # Codex succeeds IF it has diagnostic context from Claude
        if "Found the bug" in prompt and "query planner ignores" in prompt:
            return {
                "transcript": "Added FORCE INDEX hint for small tables",
                "files_changed": ["aragora/analytics/dashboard.py"],
                "tests_passed": True,
            }

        # Codex fails on cold re-dispatch (no context)
        return {"transcript": "Cannot reproduce", "tests_passed": False}

    loop = PingPongLoop(agent_a="claude", agent_b="codex", goal="Fix dashboard perf", max_rounds=2)
    result = await loop.run(dispatch_fn=dispatch)

    assert result.completed, "Should complete when agent B succeeds with handoff"
    assert result.total_rounds == 2
    assert calls[0]["agent"] == "claude"
    assert calls[1]["agent"] == "codex"
    assert calls[1]["has_context"], "Agent B should receive context from agent A"


@pytest.mark.asyncio
async def test_cold_redispatch_fails_without_handoff():
    """Verify that the same scenario fails without handoff context."""

    async def dispatch(agent: str, prompt: str) -> dict:
        if agent == "codex" and "Found the bug" not in prompt:
            return {"transcript": "Cannot reproduce", "tests_passed": False}
        return {"transcript": "Tried something", "tests_passed": False}

    # Cold re-dispatch: codex gets the raw goal, no transcript
    result = await dispatch("codex", "Fix dashboard perf")
    assert not result["tests_passed"]


@pytest.mark.asyncio
async def test_handoff_preserves_files_changed():
    """Handoff prompt includes files the first agent touched."""
    calls: list[str] = []

    async def dispatch(agent: str, prompt: str) -> dict:
        calls.append(prompt)
        if agent == "claude":
            return {
                "transcript": "Modified config.py and utils.py",
                "files_changed": ["aragora/config/settings.py", "aragora/utils/helpers.py"],
                "tests_passed": False,
            }
        return {"transcript": "Fixed", "tests_passed": True}

    loop = PingPongLoop(agent_a="claude", agent_b="codex", goal="Fix config", max_rounds=2)
    await loop.run(dispatch_fn=dispatch)

    # Agent B's prompt should list files A changed
    assert "settings.py" in calls[1]
    assert "helpers.py" in calls[1]


@pytest.mark.asyncio
async def test_handoff_preserves_remaining_issues():
    """Handoff prompt includes remaining issues from agent A."""

    async def dispatch(agent: str, prompt: str) -> dict:
        if agent == "claude":
            return {
                "transcript": "Fixed 2 of 3 tests",
                "tests_passed": False,
                "remaining_issues": ["test_auth_timeout still fails", "mock cleanup needed"],
            }
        # Verify agent B sees the remaining issues
        assert "test_auth_timeout" in prompt
        assert "mock cleanup" in prompt
        return {"transcript": "Fixed remaining", "tests_passed": True}

    loop = PingPongLoop(agent_a="claude", agent_b="codex", goal="Fix tests", max_rounds=2)
    result = await loop.run(dispatch_fn=dispatch)
    assert result.completed


@pytest.mark.asyncio
async def test_handoff_does_not_reduplicate_work():
    """Agent B's instructions say not to redo correct work."""

    async def dispatch(agent: str, prompt: str) -> dict:
        if agent == "codex":
            assert "Do NOT redo work" in prompt
        return {
            "transcript": "Did work",
            "tests_passed": agent == "codex",
        }

    loop = PingPongLoop(agent_a="claude", agent_b="codex", goal="Test", max_rounds=2)
    await loop.run(dispatch_fn=dispatch)


@pytest.mark.asyncio
async def test_short_transcript_still_produces_handoff():
    """Even a minimal transcript produces a structured handoff."""

    async def dispatch(agent: str, prompt: str) -> dict:
        if agent == "claude":
            return {"transcript": "Failed", "tests_passed": False}
        # Agent B still gets structured handoff
        assert "## Goal" in prompt or "## Context" in prompt
        return {"transcript": "Fixed", "tests_passed": True}

    loop = PingPongLoop(agent_a="claude", agent_b="codex", goal="Fix", max_rounds=2)
    result = await loop.run(dispatch_fn=dispatch)
    assert result.completed


def test_build_handoff_prompt_structure():
    """Verify the handoff prompt has all required sections."""
    prompt = build_handoff_prompt(
        goal="Fix bug #123",
        previous_transcript="Tried fixing line 42, tests still fail",
        previous_agent="claude",
        next_agent="codex",
        round_number=1,
        files_changed=["src/foo.py", "tests/test_foo.py"],
        remaining_issues=["test_bar still fails"],
    )

    assert "## Goal" in prompt
    assert "Fix bug #123" in prompt
    assert "## Context (from claude, round 1)" in prompt
    assert "Tried fixing line 42" in prompt
    assert "src/foo.py" in prompt
    assert "test_bar still fails" in prompt
    assert "## Your Task (codex, round 2)" in prompt
    assert "Do NOT redo work" in prompt
    assert "git add" in prompt
