"""Tests for aragora.nomic.handlers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aragora.nomic.events import Event, EventType
from aragora.nomic.handlers import create_verify_handler, verify_handler
from aragora.nomic.states import NomicState, StateContext


@pytest.mark.asyncio
async def test_verify_handler_sica_disabled_smoke(monkeypatch, tmp_path):
    """When SICA is disabled, verify_handler should recover without invoking SICA."""
    monkeypatch.delenv("NOMIC_SICA_ENABLED", raising=False)

    verify_phase = AsyncMock()
    verify_phase.execute = AsyncMock(
        return_value={
            "tests_passed": False,
            "syntax_valid": False,
            "test_output": "failed",
            "data": {
                "checks": [
                    {"name": "pytest", "passed": False},
                    {"name": "ruff", "passed": True},
                ]
            },
        }
    )

    with patch("aragora.nomic.handlers._run_sica_cycle", new=AsyncMock()) as mock_sica:
        next_state, data = await verify_handler(
            StateContext(),
            Event(event_type=EventType.VERIFY_COMPLETE),
            verify_phase=verify_phase,
            repo_path=tmp_path,
            sica_agents=[],
        )

    assert next_state == NomicState.RECOVERY
    assert data["error"] == "verification_failed"
    assert data["phase"] == "verify"
    assert data["sica"] is None
    assert len(data["failed_checks"]) == 1
    mock_sica.assert_not_called()


@pytest.mark.asyncio
async def test_create_verify_handler_sica_disabled_smoke(monkeypatch, tmp_path):
    """Bound verify handler should preserve disabled-SICA behavior."""
    monkeypatch.delenv("NOMIC_SICA_ENABLED", raising=False)

    verify_phase = AsyncMock()
    verify_phase.execute = AsyncMock(
        return_value={
            "tests_passed": False,
            "syntax_valid": False,
            "test_output": "failed",
            "data": {"checks": [{"name": "pytest", "passed": False}]},
        }
    )

    handler = create_verify_handler(
        verify_phase,
        repo_path=tmp_path,
        sica_agents=[],
    )

    with patch("aragora.nomic.handlers._run_sica_cycle", new=AsyncMock()) as mock_sica:
        next_state, data = await handler(
            StateContext(),
            Event(event_type=EventType.VERIFY_COMPLETE),
        )

    assert next_state == NomicState.RECOVERY
    assert data["error"] == "verification_failed"
    assert data["sica"] is None
    mock_sica.assert_not_called()
