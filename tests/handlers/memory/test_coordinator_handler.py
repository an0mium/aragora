"""Tests for Memory Coordinator API handler."""

import json
import pytest
from unittest.mock import MagicMock, patch


class TestCoordinatorHandler:
    """Tests for CoordinatorHandler class."""

    @pytest.fixture
    def handler(self):
        """Create a CoordinatorHandler instance."""
        from aragora.server.handlers.memory.coordinator import CoordinatorHandler

        return CoordinatorHandler(server_context={})

    def test_can_handle_metrics_endpoint(self, handler):
        """Test handler recognizes metrics endpoint."""
        assert handler.can_handle("/api/v1/memory/coordinator/metrics") is True

    def test_can_handle_config_endpoint(self, handler):
        """Test handler recognizes config endpoint."""
        assert handler.can_handle("/api/v1/memory/coordinator/config") is True

    def test_cannot_handle_unknown_endpoint(self, handler):
        """Test handler rejects unknown endpoints."""
        assert handler.can_handle("/api/v1/memory/coordinator/unknown") is False
        assert handler.can_handle("/api/v1/memory/other") is False

    @pytest.mark.asyncio
    async def test_metrics_without_coordinator(self, handler):
        """Test metrics returns default when no coordinator configured."""
        result = await handler.handle("/api/v1/memory/coordinator/metrics", {})

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["configured"] is False
        assert body["metrics"]["total_transactions"] == 0
        assert body["metrics"]["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_config_without_coordinator(self, handler):
        """Test config returns defaults when no coordinator configured."""
        result = await handler.handle("/api/v1/memory/coordinator/config", {})

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["configured"] is False

    @pytest.mark.asyncio
    async def test_metrics_with_coordinator(self):
        """Test metrics returns real data when coordinator exists."""
        from aragora.server.handlers.memory.coordinator import CoordinatorHandler

        mock_coordinator = MagicMock()
        mock_coordinator.get_metrics.return_value = {
            "total_transactions": 100,
            "successful_transactions": 95,
            "partial_failures": 5,
            "rollbacks_performed": 3,
            "success_rate": 0.95,
        }
        mock_coordinator.continuum_memory = MagicMock()
        mock_coordinator.consensus_memory = MagicMock()
        mock_coordinator.critique_store = None
        mock_coordinator.knowledge_mound = None
        mock_coordinator._rollback_handlers = {"continuum": lambda x: x, "consensus": lambda x: x}

        handler = CoordinatorHandler(server_context={"memory_coordinator": mock_coordinator})
        result = await handler.handle("/api/v1/memory/coordinator/metrics", {})

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["configured"] is True
        assert body["metrics"]["total_transactions"] == 100
        assert body["metrics"]["success_rate"] == 0.95
        assert body["memory_systems"]["continuum"] is True
        assert body["memory_systems"]["consensus"] is True
        assert body["memory_systems"]["critique"] is False
        assert body["memory_systems"]["mound"] is False
        assert "continuum" in body["rollback_handlers"]
        assert "consensus" in body["rollback_handlers"]

    @pytest.mark.asyncio
    async def test_config_with_coordinator(self):
        """Test config returns real options when coordinator exists."""
        from aragora.server.handlers.memory.coordinator import CoordinatorHandler
        from aragora.memory.coordinator import CoordinatorOptions

        mock_coordinator = MagicMock()
        mock_coordinator.options = CoordinatorOptions(
            write_continuum=True,
            write_consensus=True,
            write_critique=False,
            write_mound=False,
            rollback_on_failure=True,
            parallel_writes=True,
            min_confidence_for_mound=0.8,
        )

        handler = CoordinatorHandler(server_context={"memory_coordinator": mock_coordinator})
        result = await handler.handle("/api/v1/memory/coordinator/config", {})

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["configured"] is True
        assert body["options"]["write_continuum"] is True
        assert body["options"]["write_consensus"] is True
        assert body["options"]["write_critique"] is False
        assert body["options"]["write_mound"] is False
        assert body["options"]["rollback_on_failure"] is True
        assert body["options"]["parallel_writes"] is True
        assert body["options"]["min_confidence_for_mound"] == 0.8


class TestCoordinatorHandlerRateLimiting:
    """Tests for rate limiting on coordinator endpoints."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        """Test rate limiting returns 429 when exceeded."""
        from aragora.server.handlers.memory.coordinator import (
            CoordinatorHandler,
            _coordinator_limiter,
        )

        handler = CoordinatorHandler(server_context={})

        # Exhaust rate limit
        with patch.object(_coordinator_limiter, "is_allowed", return_value=False):
            result = await handler.handle("/api/v1/memory/coordinator/metrics", {})

            assert result is not None
            assert result.status_code == 429
            body = json.loads(result.body)
            assert "Rate limit" in body.get("error", "")
