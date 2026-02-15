"""
Public API contract tests for the Gauntlet.

These tests intentionally lock down the minimal callable surface that other
modules (and external users) rely on. They should be small, fast, and stable.
"""

from __future__ import annotations

import inspect

import pytest

from aragora.gauntlet.orchestrator import GauntletOrchestrator, run_gauntlet
from aragora.gauntlet.types import InputType


def test_gauntlet_orchestrator_init_has_no_required_args() -> None:
    sig = inspect.signature(GauntletOrchestrator.__init__)
    required = [
        p
        for p in sig.parameters.values()
        if p.name != "self"
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect._empty
    ]
    assert required == []


def test_run_gauntlet_signature_is_stable() -> None:
    sig = inspect.signature(run_gauntlet)
    params = sig.parameters

    assert list(params.keys())[0] == "input_content"
    assert "agents" in params
    assert params["agents"].default is None

    assert "input_type" in params
    assert params["input_type"].default == InputType.SPEC

    assert "template" in params
    assert params["template"].kind == inspect.Parameter.KEYWORD_ONLY

    assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


@pytest.mark.asyncio
async def test_run_gauntlet_legacy_mode_works_without_agents() -> None:
    # Explicitly call the stress-test mode (legacy), but keep it fast by
    # disabling all phases.
    result = await run_gauntlet(
        "hello",
        agents=[],
        enable_redteam=False,
        enable_probing=False,
        enable_deep_audit=False,
        enable_verification=False,
        enable_risk_assessment=False,
        max_duration_seconds=1,
    )

    # Legacy mode returns the orchestrator module's GauntletResult.
    assert result is not None
    assert result.input_summary
    assert hasattr(result, "verdict")
