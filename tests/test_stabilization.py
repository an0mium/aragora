"""
Tests for stabilization fixes: null byte sanitization, timeouts, loop_id routing.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from aragora.monitoring.simple_observer import SimpleObserver


@pytest.mark.asyncio
async def test_observer_basic():
    """Test basic observer functionality."""
    observer = SimpleObserver(":memory:")
    attempt_id = observer.record_agent_attempt("test_agent", 30.0)
    observer.record_agent_completion(attempt_id, "valid output")
    assert observer.get_failure_rate() == 0.0


@pytest.mark.asyncio
async def test_observer_null_byte_detection():
    """Test null byte detection."""
    observer = SimpleObserver(":memory:")
    attempt_id = observer.record_agent_attempt("test_agent", 30.0)
    observer.record_agent_completion(attempt_id, "output\x00with nulls")
    metrics = observer.metrics[attempt_id]
    assert metrics["has_null_bytes"] == True


@pytest.mark.asyncio
async def test_observer_timeout():
    """Test timeout recording."""
    observer = SimpleObserver(":memory:")
    attempt_id = observer.record_agent_attempt("test_agent", 1.0)
    # Simulate timeout by recording with error
    observer.record_agent_completion(attempt_id, None, asyncio.TimeoutError("Timeout"))
    assert observer.get_failure_rate() == 1.0


def test_diagnostic_script():
    """Test diagnostic script imports and runs."""
    from scripts.verify_system_health import diagnose_system, prioritize_fixes

    diag = diagnose_system()
    assert "agent_health" in diag
    assert "loop_id_routing" in diag
    priorities = prioritize_fixes(diag)
    assert isinstance(priorities, list)
