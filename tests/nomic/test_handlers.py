"""Tests for aragora.nomic.handlers."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aragora.nomic.events import Event, EventType
from aragora.nomic.handlers import (
    create_recovery_handler,
    create_verify_handler,
    recovery_state_handler,
    verify_handler,
)
from aragora.nomic.recovery import RecoveryManager
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


@pytest.mark.asyncio
async def test_recovery_state_handler_sica_disabled_delegates(monkeypatch, tmp_path):
    """Recovery should use core strategy handler when SICA is disabled."""
    monkeypatch.setenv("NOMIC_SICA_ENABLED", "0")
    context = StateContext(previous_state=NomicState.VERIFY)
    event = Event(
        event_type=EventType.ERROR,
        error_message="verification_failed",
        error_type="RuntimeError",
    )
    recovery_manager = RecoveryManager()

    with (
        patch("aragora.nomic.handlers._run_sica_cycle", new=AsyncMock()) as mock_sica,
        patch(
            "aragora.nomic.handlers.core_recovery_handler",
            new=AsyncMock(
                return_value=(
                    NomicState.IMPLEMENT,
                    {"decision": {"strategy": "RETRY"}, "recovered_from": "VERIFY"},
                )
            ),
        ) as mock_core,
    ):
        next_state, data = await recovery_state_handler(
            context,
            event,
            recovery_manager=recovery_manager,
            repo_path=tmp_path,
            sica_agents=[],
        )

    assert next_state == NomicState.IMPLEMENT
    assert data["decision"]["strategy"] == "RETRY"
    mock_sica.assert_not_called()
    mock_core.assert_called_once()


@pytest.mark.asyncio
async def test_recovery_state_handler_sica_success_short_circuits(monkeypatch, tmp_path):
    """Successful SICA recovery should send flow back to verification."""
    monkeypatch.setenv("NOMIC_SICA_ENABLED", "1")
    context = StateContext(previous_state=NomicState.VERIFY)
    event = Event(
        event_type=EventType.ERROR,
        error_message="verification_failed",
        error_type="RuntimeError",
    )
    recovery_manager = RecoveryManager()
    sica_payload = {"status": "success", "summary": "patched"}

    with (
        patch(
            "aragora.nomic.handlers._run_sica_cycle",
            new=AsyncMock(return_value=sica_payload),
        ) as mock_sica,
        patch("aragora.nomic.handlers.core_recovery_handler", new=AsyncMock()) as mock_core,
    ):
        next_state, data = await recovery_state_handler(
            context,
            event,
            recovery_manager=recovery_manager,
            repo_path=tmp_path,
            sica_agents=[],
        )

    assert next_state == NomicState.VERIFY
    assert data["decision"]["strategy"] == "SICA_REPAIR"
    assert data["sica"] == sica_payload
    mock_sica.assert_called_once()
    mock_core.assert_not_called()


@pytest.mark.asyncio
async def test_create_recovery_handler_binds_dependencies(monkeypatch, tmp_path):
    """Factory should bind recovery manager and route through recovery handler."""
    monkeypatch.setenv("NOMIC_SICA_ENABLED", "0")
    recovery_manager = RecoveryManager()
    handler = create_recovery_handler(
        recovery_manager,
        repo_path=tmp_path,
        sica_agents=[],
    )

    context = StateContext(previous_state=NomicState.DESIGN)
    event = Event(
        event_type=EventType.ERROR,
        error_message="design_error",
        error_type="ValueError",
    )

    with patch(
        "aragora.nomic.handlers.core_recovery_handler",
        new=AsyncMock(
            return_value=(NomicState.IMPLEMENT, {"decision": {"strategy": "SKIP"}})
        ),
    ):
        next_state, data = await handler(context, event)

    assert next_state == NomicState.IMPLEMENT
    assert data["decision"]["strategy"] == "SKIP"
