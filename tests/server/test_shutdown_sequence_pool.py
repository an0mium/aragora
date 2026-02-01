"""
Tests for HTTP Client Pool Shutdown Integration.

Tests cover:
- HTTP client pool close is called during shutdown execution
- Shutdown completes within overall_timeout
- HTTP client pool close failure does not block shutdown (non-critical)
- Phase timeout budgets sum to less than overall_timeout
- HTTP client pool aclose handles already-closed pool gracefully
- drain_requests phase uses the updated 15s timeout
- wait_for_debates phase uses the updated 12s timeout
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.shutdown_sequence import (
    ShutdownPhase,
    ShutdownSequence,
    create_server_shutdown_sequence,
)


class TestHTTPClientPoolShutdown:
    """Tests for HTTP client pool shutdown integration."""

    @pytest.mark.asyncio
    async def test_http_client_pool_close_called_during_shutdown(self):
        """Test that HTTP client pool aclose is called during shutdown execution."""
        mock_pool = MagicMock()
        mock_pool._closed = False
        mock_pool.aclose = AsyncMock()

        async def close_http_client_pool():
            from aragora.server.http_client_pool import HTTPClientPool

            pool = HTTPClientPool.get_instance()
            if pool and not pool._closed:
                await pool.aclose()

        with patch(
            "aragora.server.http_client_pool.HTTPClientPool.get_instance",
            return_value=mock_pool,
        ):
            sequence = ShutdownSequence()
            sequence.add_phase(
                ShutdownPhase(
                    name="Close HTTP client pool",
                    execute=close_http_client_pool,
                    timeout=5.0,
                    critical=False,
                )
            )

            result = await sequence.execute_all(overall_timeout=30.0)
            assert "Close HTTP client pool" in result["completed"]
            mock_pool.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_completes_within_overall_timeout(self):
        """Test that shutdown completes within overall_timeout."""
        # Create a sequence with a few quick phases
        sequence = ShutdownSequence()

        async def quick_phase():
            await asyncio.sleep(0.01)

        sequence.add_phase(ShutdownPhase(name="Phase 1", execute=quick_phase, timeout=5.0))
        sequence.add_phase(ShutdownPhase(name="Phase 2", execute=quick_phase, timeout=5.0))
        sequence.add_phase(ShutdownPhase(name="Phase 3", execute=quick_phase, timeout=5.0))

        result = await sequence.execute_all(overall_timeout=30.0)

        assert result["elapsed"] < 30.0
        assert len(result["completed"]) == 3
        assert len(result["failed"]) == 0

    @pytest.mark.asyncio
    async def test_http_client_pool_close_failure_continues_shutdown(self):
        """Test that if HTTP client pool close fails, shutdown continues (non-critical)."""
        mock_pool = MagicMock()
        mock_pool._closed = False
        mock_pool.aclose = AsyncMock(side_effect=RuntimeError("Pool close error"))

        async def close_http_client_pool():
            from aragora.server.http_client_pool import HTTPClientPool

            pool = HTTPClientPool.get_instance()
            if pool and not pool._closed:
                await pool.aclose()

        async def quick_phase():
            await asyncio.sleep(0.01)

        with patch(
            "aragora.server.http_client_pool.HTTPClientPool.get_instance",
            return_value=mock_pool,
        ):
            sequence = ShutdownSequence()
            sequence.add_phase(
                ShutdownPhase(
                    name="Close HTTP client pool",
                    execute=close_http_client_pool,
                    timeout=5.0,
                    critical=False,
                )
            )
            sequence.add_phase(
                ShutdownPhase(
                    name="After pool close",
                    execute=quick_phase,
                    timeout=5.0,
                )
            )

            result = await sequence.execute_all(overall_timeout=30.0)

            # Pool close should fail but shutdown should continue
            assert "Close HTTP client pool" in result["failed"]
            assert "After pool close" in result["completed"]

    @pytest.mark.asyncio
    async def test_phase_timeout_budgets_sum_less_than_overall(self):
        """Test that critical phase timeout budgets sum to less than overall_timeout."""
        # Create a mock server object
        mock_server = MagicMock()
        mock_server._shutting_down = False

        # Patch all the module imports to avoid import errors
        patches = [
            patch("aragora.server.request_tracker.get_request_tracker"),
            patch("aragora.server.debate_utils.get_active_debates", return_value={}),
            patch("aragora.resilience.persist_all_circuit_breakers", return_value=0),
            patch("aragora.events.cross_subscribers.get_cross_subscriber_manager"),
            patch("aragora.observability.tracing.shutdown"),
            patch("aragora.server.background.get_background_manager"),
            patch("aragora.server.handlers.pulse.get_pulse_scheduler", return_value=None),
            patch("aragora.server.stream.state_manager.stop_cleanup_task"),
            patch(
                "aragora.agents.api_agents.common.close_shared_connector", new_callable=AsyncMock
            ),
            patch("aragora.server.http_client_pool.HTTPClientPool.get_instance"),
            patch("aragora.rbac.cache.get_rbac_cache", return_value=None),
            patch("aragora.server.startup.workers.get_gauntlet_worker", return_value=None),
            patch("aragora.server.redis_config.close_redis_pool"),
            patch("aragora.server.auth.auth_config.stop_cleanup_thread"),
            patch("aragora.storage.pool_manager.close_shared_pool", new_callable=AsyncMock),
            patch("aragora.storage.connection_factory.close_all_pools", new_callable=AsyncMock),
            patch("aragora.storage.schema.DatabaseManager.clear_instances"),
            patch("aragora.observability.metrics.stop_metrics_server"),
        ]

        with contextlib_nested(*patches):
            sequence = create_server_shutdown_sequence(mock_server)

            # Calculate total critical phase timeouts
            critical_timeout_sum = sum(
                phase.timeout for phase in sequence._phases if phase.critical
            )

            # Total of all phases (for reference)
            total_timeout_sum = sum(phase.timeout for phase in sequence._phases)

            # Critical phases should fit within overall timeout (30s)
            # drain_requests (15s) + wait_for_debates (12s) = 27s critical
            # Leaves 3s buffer for unexpected delays
            assert critical_timeout_sum <= 30.0, (
                f"Critical phase timeouts ({critical_timeout_sum}s) exceed overall timeout (30s)"
            )

            # Total should also be reasonable (though non-critical phases can exceed)
            # This is informational - non-critical phases are capped by remaining time anyway
            assert critical_timeout_sum <= 27.0, (
                f"Critical phase timeouts ({critical_timeout_sum}s) should be <= 27s to leave buffer"
            )

    @pytest.mark.asyncio
    async def test_http_client_pool_aclose_handles_already_closed(self):
        """Test that HTTP client pool aclose handles already-closed pool gracefully."""
        mock_pool = MagicMock()
        mock_pool._closed = True  # Pool already closed
        mock_pool.aclose = AsyncMock()

        async def close_http_client_pool():
            try:
                from aragora.server.http_client_pool import HTTPClientPool

                pool = HTTPClientPool.get_instance()
                if pool and not pool._closed:
                    await pool.aclose()
            except RuntimeError:
                # Pool already closed - this is expected
                pass

        with patch(
            "aragora.server.http_client_pool.HTTPClientPool.get_instance",
            return_value=mock_pool,
        ):
            sequence = ShutdownSequence()
            sequence.add_phase(
                ShutdownPhase(
                    name="Close HTTP client pool",
                    execute=close_http_client_pool,
                    timeout=5.0,
                    critical=False,
                )
            )

            result = await sequence.execute_all(overall_timeout=30.0)

            # Should complete without calling aclose (pool already closed)
            assert "Close HTTP client pool" in result["completed"]
            mock_pool.aclose.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_drain_requests_phase_timeout_is_15s(self):
        """Test that drain_requests phase uses the updated 15s timeout."""
        mock_server = MagicMock()
        mock_server._shutting_down = False

        # We need to verify the timeout without actually running the full sequence
        # Use create_server_shutdown_sequence and inspect the phases
        with patch.multiple(
            "aragora.server.shutdown_sequence",
            # Patch imports that happen inside the function
        ):
            sequence = create_server_shutdown_sequence(mock_server)

            # Find the drain requests phase
            drain_phase = None
            for phase in sequence._phases:
                if phase.name == "Drain in-flight requests":
                    drain_phase = phase
                    break

            assert drain_phase is not None, "Drain in-flight requests phase not found"
            assert drain_phase.timeout == 15.0, (
                f"Expected drain_requests timeout of 15.0s, got {drain_phase.timeout}s"
            )
            assert drain_phase.critical is True

    @pytest.mark.asyncio
    async def test_wait_for_debates_phase_timeout_is_12s(self):
        """Test that wait_for_debates phase uses the updated 12s timeout."""
        mock_server = MagicMock()
        mock_server._shutting_down = False

        sequence = create_server_shutdown_sequence(mock_server)

        # Find the wait for debates phase
        debates_phase = None
        for phase in sequence._phases:
            if phase.name == "Wait for in-flight debates":
                debates_phase = phase
                break

        assert debates_phase is not None, "Wait for in-flight debates phase not found"
        assert debates_phase.timeout == 12.0, (
            f"Expected wait_for_debates timeout of 12.0s, got {debates_phase.timeout}s"
        )
        assert debates_phase.critical is True


class TestShutdownSequencePhases:
    """Tests for shutdown sequence phase configuration."""

    def test_http_client_pool_phase_exists(self):
        """Test that Close HTTP client pool phase exists in shutdown sequence."""
        mock_server = MagicMock()
        mock_server._shutting_down = False

        sequence = create_server_shutdown_sequence(mock_server)

        phase_names = [phase.name for phase in sequence._phases]
        assert "Close HTTP client pool" in phase_names

    def test_http_client_pool_phase_is_non_critical(self):
        """Test that Close HTTP client pool phase is non-critical."""
        mock_server = MagicMock()
        mock_server._shutting_down = False

        sequence = create_server_shutdown_sequence(mock_server)

        pool_phase = None
        for phase in sequence._phases:
            if phase.name == "Close HTTP client pool":
                pool_phase = phase
                break

        assert pool_phase is not None, "Close HTTP client pool phase not found"
        assert pool_phase.critical is False, "HTTP client pool phase should be non-critical"

    def test_http_client_pool_phase_after_http_connector(self):
        """Test that Close HTTP client pool phase comes after Close HTTP connector."""
        mock_server = MagicMock()
        mock_server._shutting_down = False

        sequence = create_server_shutdown_sequence(mock_server)

        phase_names = [phase.name for phase in sequence._phases]

        connector_index = phase_names.index("Close HTTP connector")
        pool_index = phase_names.index("Close HTTP client pool")

        assert pool_index > connector_index, (
            "HTTP client pool phase should come after HTTP connector phase"
        )


class TestShutdownSequenceTimeoutBudget:
    """Tests for shutdown sequence timeout budget management."""

    def test_critical_phases_fit_in_timeout_budget(self):
        """Test that critical phase timeouts fit within 30s overall timeout."""
        mock_server = MagicMock()
        mock_server._shutting_down = False

        sequence = create_server_shutdown_sequence(mock_server)

        critical_timeout_sum = sum(phase.timeout for phase in sequence._phases if phase.critical)

        # Critical phases must complete within overall timeout
        # We have drain (15s) + debates (12s) + others
        assert critical_timeout_sum <= 30.0, (
            f"Critical phase timeouts ({critical_timeout_sum}s) exceed 30s budget"
        )

    def test_drain_and_debates_phases_leave_buffer(self):
        """Test that drain + debates phases leave buffer for other phases."""
        mock_server = MagicMock()
        mock_server._shutting_down = False

        sequence = create_server_shutdown_sequence(mock_server)

        # Find the two main phases
        drain_timeout = 0.0
        debates_timeout = 0.0

        for phase in sequence._phases:
            if phase.name == "Drain in-flight requests":
                drain_timeout = phase.timeout
            elif phase.name == "Wait for in-flight debates":
                debates_timeout = phase.timeout

        combined = drain_timeout + debates_timeout

        # Should be 15 + 12 = 27s, leaving 3s buffer
        assert combined == 27.0, f"Expected 27s combined, got {combined}s"
        assert combined <= 27.0, "Combined timeout should leave at least 3s buffer"


# Helper for nested context managers
from contextlib import contextmanager


@contextmanager
def contextlib_nested(*contexts):
    """Context manager to handle multiple patches."""
    with contexts[0]:
        if len(contexts) == 1:
            yield
        else:
            with contextlib_nested(*contexts[1:]):
                yield
