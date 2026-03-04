"""Tests for debate context engineering helpers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from aragora.debate.context_engineering import (
    ContextEngineeringConfig,
    build_debate_context_engineering,
)


class _FakeBuilder:
    """Minimal async builder stub for deterministic map tests."""

    def __init__(self, root_path, include_tests=True, full_corpus=False):  # noqa: ANN001
        self.root_path = root_path
        self.include_tests = include_tests
        self.full_corpus = full_corpus

    async def build_index(self):
        return SimpleNamespace(total_files=42, total_lines=1337, total_tokens_estimate=4096)

    async def build_debate_context(self):
        return "## fake-map\n" + ("module: x\n" * 1200)


class _FakeAgent:
    """Simple async agent stub."""

    def __init__(self, name: str):
        self.name = name

    async def generate(self, prompt: str, context=None):  # noqa: ANN001
        return f"{self.name} analyzed prompt length={len(prompt)}"


class _SlowAgent:
    """Agent stub that blocks long enough to trigger timeout enforcement."""

    async def generate(self, prompt: str, context=None):  # noqa: ANN001
        await asyncio.sleep(5.0)
        return "late output"


@pytest.mark.asyncio
async def test_build_context_engineering_base_only(monkeypatch, tmp_path):
    """Builds deterministic context and anchor map without harnesses."""
    anchor = tmp_path / "aragora" / "debate"
    anchor.mkdir(parents=True)
    (anchor / "orchestrator.py").write_text("class Arena: ...\n", encoding="utf-8")

    monkeypatch.setattr("aragora.debate.context_engineering.CodebaseContextBuilder", _FakeBuilder)

    cfg = ContextEngineeringConfig(
        task="Improve self-improvement debate quality",
        repo_path=tmp_path,
        include_harness_exploration=False,
        max_output_chars=12000,
    )
    result = await build_debate_context_engineering(cfg)

    assert "Codebase Reality Check (Auto-Generated)" in result.context
    assert "Canonical Existing Anchors" in result.context
    assert "aragora/debate/orchestrator.py" in result.context
    assert result.metadata["base"]["indexed_files"] == 42
    assert result.metadata["harnesses"]["enabled"] is False


@pytest.mark.asyncio
async def test_build_context_engineering_with_harnesses(monkeypatch, tmp_path):
    """Adds explorer synthesis when harness mode is enabled."""
    monkeypatch.setattr("aragora.debate.context_engineering.CodebaseContextBuilder", _FakeBuilder)

    def _fake_create_agent(model_type, name=None, role="analyst", timeout=None):  # noqa: ANN001
        return _FakeAgent(name or str(model_type))

    class _FakeKilo(_FakeAgent):
        def __init__(
            self,
            name: str,
            provider_id: str = "",
            model: str | None = None,
            role: str = "analyst",
            timeout: int = 180,
            mode: str = "architect",
        ):
            super().__init__(name=name)
            self.provider_id = provider_id
            self.model = model
            self.role = role
            self.timeout = timeout
            self.mode = mode

    monkeypatch.setattr("aragora.debate.context_engineering.create_agent", _fake_create_agent)
    monkeypatch.setattr("aragora.debate.context_engineering.KiloCodeAgent", _FakeKilo)
    monkeypatch.setattr(
        "aragora.debate.context_engineering.shutil.which", lambda _: "/usr/bin/kilo"
    )

    cfg = ContextEngineeringConfig(
        task="Ground plans in existing implementations",
        repo_path=tmp_path,
        include_harness_exploration=True,
        include_kilocode=True,
        max_output_chars=20000,
    )
    result = await build_debate_context_engineering(cfg)

    assert "Harness Explorer Synthesis" in result.context
    assert "### claude" in result.context
    assert "### codex" in result.context
    assert "### kilocode-gemini" in result.context
    assert result.metadata["harnesses"]["enabled"] is True
    assert result.metadata["harnesses"]["successes"] >= 2


@pytest.mark.asyncio
async def test_build_context_engineering_missing_repo(tmp_path):
    """Returns empty context and metadata error for missing repo path."""
    missing = tmp_path / "missing-repo"
    cfg = ContextEngineeringConfig(
        task="test",
        repo_path=missing,
    )
    result = await build_debate_context_engineering(cfg)
    assert result.context == ""
    assert "does not exist" in str(result.metadata.get("error", ""))


@pytest.mark.asyncio
async def test_explorer_timeout_triggers_cleanup_and_error(monkeypatch, caplog):
    """Timed-out explorer calls should force CLI cleanup and emit explicit logs."""
    from aragora.debate import context_engineering as ce

    cleanup_calls: list[float] = []

    def _fake_cleanup(grace_seconds: float = 0.2) -> dict[str, int]:
        cleanup_calls.append(grace_seconds)
        return {"tracked": 1, "terminated": 1, "killed": 0, "remaining": 0}

    monkeypatch.setattr("aragora.agents.cli_agents.terminate_tracked_cli_processes", _fake_cleanup)
    caplog.set_level("WARNING")

    result = await ce._run_single_explorer(  # type: ignore[attr-defined]
        name="slow-agent",
        agent=_SlowAgent(),
        prompt="audit",
        timeout_seconds=1,
        max_chars=500,
    )

    assert result.success is False
    assert "timeout after 1s" in str(result.error)
    assert "cleanup:" in str(result.error)
    assert cleanup_calls, "expected timeout path to invoke CLI cleanup"
    assert any("context_engineering_explorer_timeout" in rec.message for rec in caplog.records)
