"""Tests for autonomous triggers handler."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.autonomous import triggers


# =============================================================================
# Mock Classes
# =============================================================================


class MockTrigger:
    """Mock trigger for testing."""

    def __init__(
        self,
        id: str = "trigger-001",
        name: str = "Test Trigger",
        interval_seconds: int = 3600,
        cron_expression: str = None,
        enabled: bool = True,
        last_run=None,
        next_run=None,
        run_count: int = 0,
        max_runs: int = None,
        metadata: dict = None,
    ):
        self.id = id
        self.name = name
        self.interval_seconds = interval_seconds
        self.cron_expression = cron_expression
        self.enabled = enabled
        self.last_run = last_run
        self.next_run = next_run or datetime.now()
        self.run_count = run_count
        self.max_runs = max_runs
        self.metadata = metadata or {}


class MockScheduledTrigger:
    """Mock ScheduledTrigger for testing."""

    def __init__(self):
        self._triggers = {}

    def list_triggers(self):
        return list(self._triggers.values())

    def get_trigger(self, trigger_id):
        return self._triggers.get(trigger_id)

    def add_trigger(self, **kwargs):
        trigger = MockTrigger(**kwargs)
        self._triggers[trigger.id] = trigger
        return trigger

    def update_trigger(self, trigger_id, **kwargs):
        trigger = self._triggers.get(trigger_id)
        if trigger:
            for key, value in kwargs.items():
                if hasattr(trigger, key):
                    setattr(trigger, key, value)
            return trigger
        return None

    def delete_trigger(self, trigger_id):
        if trigger_id in self._triggers:
            del self._triggers[trigger_id]
            return True
        return False

    def enable_trigger(self, trigger_id):
        trigger = self._triggers.get(trigger_id)
        if trigger:
            trigger.enabled = True
            return True
        return False

    def disable_trigger(self, trigger_id):
        trigger = self._triggers.get(trigger_id)
        if trigger:
            trigger.enabled = False
            return True
        return False


class MockAuthContext:
    """Mock authorization context."""

    def __init__(self, user_id="test-user"):
        self.user_id = user_id


class MockPermissionDecision:
    """Mock permission decision."""

    def __init__(self, allowed=True, reason=None):
        self.allowed = allowed
        self.reason = reason or ""


class MockPermissionChecker:
    """Mock permission checker."""

    def check_permission(self, ctx, permission):
        return MockPermissionDecision(allowed=True)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_trigger():
    """Create mock scheduled trigger."""
    return MockScheduledTrigger()


@pytest.fixture
def mock_auth_context():
    """Create mock auth context."""
    return MockAuthContext()


@pytest.fixture
def mock_permission_checker():
    """Create mock permission checker."""
    return MockPermissionChecker()


# =============================================================================
# Test TriggerHandler.list_triggers
# =============================================================================


class TestTriggerHandlerListTriggers:
    """Tests for GET /api/autonomous/triggers endpoint."""

    @pytest.mark.asyncio
    async def test_list_triggers_empty(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should return empty list when no triggers."""
        with (
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(
                triggers,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                triggers,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.list_triggers(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["triggers"] == []
            assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_list_triggers_with_data(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should return triggers list."""
        mock_trigger._triggers = {
            "t1": MockTrigger(id="t1", name="Trigger 1"),
            "t2": MockTrigger(id="t2", name="Trigger 2"),
        }

        with (
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(
                triggers,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                triggers,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.list_triggers(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["count"] == 2

    @pytest.mark.asyncio
    async def test_list_triggers_unauthorized(self, mock_trigger):
        """Should return 401 when unauthorized."""
        with (
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(
                triggers,
                "get_auth_context",
                AsyncMock(side_effect=triggers.UnauthorizedError("Not authenticated")),
            ),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.list_triggers(request)

            assert response.status == 401

    @pytest.mark.asyncio
    async def test_list_triggers_forbidden(self, mock_trigger, mock_auth_context):
        """Should return 403 when permission denied."""
        mock_checker = MockPermissionChecker()
        mock_checker.check_permission = MagicMock(
            return_value=MockPermissionDecision(allowed=False, reason="No permission")
        )

        with (
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(
                triggers,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                triggers,
                "get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.list_triggers(request)

            assert response.status == 403


# =============================================================================
# Test Route Registration
# =============================================================================


class TestTriggerHandlerRoutes:
    """Tests for route registration."""

    def test_register_routes(self):
        """Should register all trigger routes."""
        app = web.Application()
        triggers.TriggerHandler.register_routes(app)

        routes = [r.resource.canonical for r in app.router.routes()]
        assert "/api/v1/autonomous/triggers" in routes
        assert "/api/v1/autonomous/triggers/{trigger_id}" in routes
        assert "/api/v1/autonomous/triggers/{trigger_id}/enable" in routes
        assert "/api/v1/autonomous/triggers/{trigger_id}/disable" in routes
        assert "/api/v1/autonomous/triggers/start" in routes
        assert "/api/v1/autonomous/triggers/stop" in routes


# =============================================================================
# Test Global Functions
# =============================================================================


class TestScheduledTriggerSingleton:
    """Tests for scheduled trigger singleton functions."""

    def test_get_scheduled_trigger_creates_singleton(self):
        """get_scheduled_trigger should return same instance."""
        triggers._scheduled_trigger = None

        trigger1 = triggers.get_scheduled_trigger()
        trigger2 = triggers.get_scheduled_trigger()

        assert trigger1 is trigger2

        # Clean up
        triggers._scheduled_trigger = None

    def test_set_scheduled_trigger(self):
        """set_scheduled_trigger should update the global instance."""
        mock = MockScheduledTrigger()
        triggers.set_scheduled_trigger(mock)

        assert triggers.get_scheduled_trigger() is mock

        # Clean up
        triggers._scheduled_trigger = None


# =============================================================================
# Additional Tests - Circuit Breaker
# =============================================================================


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""

    def __init__(self, can_exec: bool = True):
        self._can_execute = can_exec

    def can_execute(self) -> bool:
        return self._can_execute


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker protection."""

    @pytest.mark.asyncio
    async def test_list_triggers_circuit_breaker_open(self):
        """Should return 503 when circuit breaker is open."""
        cb = MockCircuitBreaker(can_exec=False)
        with patch.object(triggers, "_get_circuit_breaker", return_value=cb):
            request = MagicMock()
            response = await triggers.TriggerHandler.list_triggers(request)

            assert response.status == 503
            body = json.loads(response.body)
            assert "unavailable" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_add_trigger_circuit_breaker_open(self):
        """Should return 503 when circuit breaker is open."""
        cb = MockCircuitBreaker(can_exec=False)
        with patch.object(triggers, "_get_circuit_breaker", return_value=cb):
            request = MagicMock()
            response = await triggers.TriggerHandler.add_trigger(request)

            assert response.status == 503

    @pytest.mark.asyncio
    async def test_remove_trigger_circuit_breaker_open(self):
        """Should return 503 when circuit breaker is open."""
        cb = MockCircuitBreaker(can_exec=False)
        with patch.object(triggers, "_get_circuit_breaker", return_value=cb):
            request = MagicMock()
            request.match_info.get.return_value = "trigger-1"
            response = await triggers.TriggerHandler.remove_trigger(request)

            assert response.status == 503

    @pytest.mark.asyncio
    async def test_enable_trigger_circuit_breaker_open(self):
        """Should return 503 when circuit breaker is open."""
        cb = MockCircuitBreaker(can_exec=False)
        with patch.object(triggers, "_get_circuit_breaker", return_value=cb):
            request = MagicMock()
            request.match_info.get.return_value = "trigger-1"
            response = await triggers.TriggerHandler.enable_trigger(request)

            assert response.status == 503

    @pytest.mark.asyncio
    async def test_disable_trigger_circuit_breaker_open(self):
        """Should return 503 when circuit breaker is open."""
        cb = MockCircuitBreaker(can_exec=False)
        with patch.object(triggers, "_get_circuit_breaker", return_value=cb):
            request = MagicMock()
            request.match_info.get.return_value = "trigger-1"
            response = await triggers.TriggerHandler.disable_trigger(request)

            assert response.status == 503

    def test_circuit_breaker_singleton(self):
        """Should return same circuit breaker instance."""
        triggers._trigger_circuit_breaker = None

        with patch.object(triggers, "get_circuit_breaker") as mock_get_cb:
            mock_cb = MockCircuitBreaker()
            mock_get_cb.return_value = mock_cb

            cb1 = triggers._get_circuit_breaker()
            cb2 = triggers._get_circuit_breaker()

            assert cb1 is cb2

        triggers._trigger_circuit_breaker = None


# =============================================================================
# Additional Tests - Add Trigger
# =============================================================================


class TestTriggerHandlerAddTrigger:
    """Tests for POST /api/autonomous/triggers endpoint."""

    @pytest.mark.asyncio
    async def test_add_trigger_success(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should add trigger successfully."""
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
            patch.object(
                triggers,
                "parse_json_body",
                AsyncMock(
                    return_value=(
                        {
                            "trigger_id": "new-trigger",
                            "name": "New Trigger",
                            "interval_seconds": 3600,
                        },
                        None,
                    )
                ),
            ),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.add_trigger(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["trigger"]["id"] == "new-trigger"

    @pytest.mark.asyncio
    async def test_add_trigger_missing_id(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should return 400 when trigger_id is missing."""
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
            patch.object(
                triggers,
                "parse_json_body",
                AsyncMock(return_value=({"name": "Test"}, None)),
            ),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.add_trigger(request)

            assert response.status == 400
            body = json.loads(response.body)
            assert "required" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_add_trigger_missing_name(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should return 400 when name is missing."""
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
            patch.object(
                triggers,
                "parse_json_body",
                AsyncMock(return_value=({"trigger_id": "test"}, None)),
            ),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.add_trigger(request)

            assert response.status == 400

    @pytest.mark.asyncio
    async def test_add_trigger_with_cron(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should add trigger with cron expression."""
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
            patch.object(
                triggers,
                "parse_json_body",
                AsyncMock(
                    return_value=(
                        {
                            "trigger_id": "cron-trigger",
                            "name": "Cron Trigger",
                            "cron_expression": "0 * * * *",
                        },
                        None,
                    )
                ),
            ),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.add_trigger(request)

            assert response.status == 200

    @pytest.mark.asyncio
    async def test_add_trigger_with_metadata(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should add trigger with metadata."""
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
            patch.object(
                triggers,
                "parse_json_body",
                AsyncMock(
                    return_value=(
                        {
                            "trigger_id": "meta-trigger",
                            "name": "Meta Trigger",
                            "interval_seconds": 1800,
                            "metadata": {"topic": "AI safety", "agents": ["claude"]},
                        },
                        None,
                    )
                ),
            ),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.add_trigger(request)

            assert response.status == 200

    @pytest.mark.asyncio
    async def test_add_trigger_unauthorized(self, mock_trigger):
        """Should return 401 when unauthorized."""
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(
                triggers,
                "get_auth_context",
                AsyncMock(side_effect=triggers.UnauthorizedError("Not authenticated")),
            ),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.add_trigger(request)

            assert response.status == 401


# =============================================================================
# Additional Tests - Remove Trigger
# =============================================================================


class TestTriggerHandlerRemoveTrigger:
    """Tests for DELETE /api/autonomous/triggers/{trigger_id} endpoint."""

    @pytest.mark.asyncio
    async def test_remove_trigger_success(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should remove trigger successfully."""
        mock_trigger._triggers = {"trigger-1": MockTrigger(id="trigger-1")}
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "trigger-1"

            response = await triggers.TriggerHandler.remove_trigger(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["removed"] is True

    @pytest.mark.asyncio
    async def test_remove_trigger_not_found(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should return 404 when trigger not found."""
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "non-existent"

            response = await triggers.TriggerHandler.remove_trigger(request)

            assert response.status == 404


# =============================================================================
# Additional Tests - Enable/Disable Trigger
# =============================================================================


class TestTriggerHandlerEnableDisable:
    """Tests for enable/disable trigger endpoints."""

    @pytest.mark.asyncio
    async def test_enable_trigger_success(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should enable trigger successfully."""
        mock_trigger._triggers = {"trigger-1": MockTrigger(id="trigger-1", enabled=False)}
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "trigger-1"

            response = await triggers.TriggerHandler.enable_trigger(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["enabled"] is True

    @pytest.mark.asyncio
    async def test_enable_trigger_not_found(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should return 404 when trigger not found."""
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "non-existent"

            response = await triggers.TriggerHandler.enable_trigger(request)

            assert response.status == 404

    @pytest.mark.asyncio
    async def test_disable_trigger_success(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should disable trigger successfully."""
        mock_trigger._triggers = {"trigger-1": MockTrigger(id="trigger-1", enabled=True)}
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "trigger-1"

            response = await triggers.TriggerHandler.disable_trigger(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["enabled"] is False

    @pytest.mark.asyncio
    async def test_disable_trigger_not_found(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should return 404 when trigger not found."""
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "non-existent"

            response = await triggers.TriggerHandler.disable_trigger(request)

            assert response.status == 404


# =============================================================================
# Additional Tests - Start/Stop Scheduler
# =============================================================================


class TestTriggerHandlerScheduler:
    """Tests for scheduler start/stop endpoints."""

    @pytest.mark.asyncio
    async def test_start_scheduler_success(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should start scheduler successfully."""
        mock_trigger.start = AsyncMock()

        with (
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.start_scheduler(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["scheduler_running"] is True

    @pytest.mark.asyncio
    async def test_start_scheduler_unauthorized(self, mock_trigger):
        """Should return 401 when unauthorized."""
        with (
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(
                triggers,
                "get_auth_context",
                AsyncMock(side_effect=triggers.UnauthorizedError("Not authenticated")),
            ),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.start_scheduler(request)

            assert response.status == 401

    @pytest.mark.asyncio
    async def test_start_scheduler_forbidden(self, mock_trigger, mock_auth_context):
        """Should return 403 when permission denied."""
        mock_checker = MockPermissionChecker()
        mock_checker.check_permission = MagicMock(
            return_value=MockPermissionDecision(allowed=False, reason="No permission")
        )

        with (
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_checker),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.start_scheduler(request)

            assert response.status == 403

    @pytest.mark.asyncio
    async def test_stop_scheduler_success(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should stop scheduler successfully."""
        mock_trigger.stop = AsyncMock()

        with (
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.stop_scheduler(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["scheduler_running"] is False

    @pytest.mark.asyncio
    async def test_stop_scheduler_unauthorized(self, mock_trigger):
        """Should return 401 when unauthorized."""
        with (
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(
                triggers,
                "get_auth_context",
                AsyncMock(side_effect=triggers.UnauthorizedError("Not authenticated")),
            ),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.stop_scheduler(request)

            assert response.status == 401

    @pytest.mark.asyncio
    async def test_stop_scheduler_forbidden(self, mock_trigger, mock_auth_context):
        """Should return 403 when permission denied."""
        mock_checker = MockPermissionChecker()
        mock_checker.check_permission = MagicMock(
            return_value=MockPermissionDecision(allowed=False, reason="No permission")
        )

        with (
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_checker),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.stop_scheduler(request)

            assert response.status == 403


# =============================================================================
# Additional Tests - Error Handling
# =============================================================================


class TestTriggerHandlerErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_list_triggers_internal_error(self, mock_auth_context, mock_permission_checker):
        """Should return 500 on internal error."""
        mock_trigger = MockScheduledTrigger()
        mock_trigger.list_triggers = MagicMock(side_effect=ValueError("Internal error"))
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.list_triggers(request)

            assert response.status == 500
            body = json.loads(response.body)
            assert body["success"] is False

    @pytest.mark.asyncio
    async def test_add_trigger_json_parse_error(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should handle JSON parse errors."""
        cb = MockCircuitBreaker(can_exec=True)

        error_response = MagicMock()
        error_response.status = 400

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
            patch.object(
                triggers,
                "parse_json_body",
                AsyncMock(return_value=(None, error_response)),
            ),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.add_trigger(request)

            assert response.status == 400

    @pytest.mark.asyncio
    async def test_enable_trigger_internal_error(self, mock_auth_context, mock_permission_checker):
        """Should return 500 on internal error."""
        mock_trigger = MockScheduledTrigger()
        mock_trigger.enable_trigger = MagicMock(side_effect=ValueError("Database error"))
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "trigger-1"
            response = await triggers.TriggerHandler.enable_trigger(request)

            assert response.status == 500


# =============================================================================
# Additional Tests - Handler Initialization
# =============================================================================


class TestTriggerHandlerInitialization:
    """Tests for handler initialization."""

    def test_handler_init_with_context(self):
        """Should initialize with provided context."""
        ctx = {"key": "value"}
        handler = triggers.TriggerHandler(ctx)
        assert handler.ctx == ctx

    def test_handler_init_without_context(self):
        """Should initialize with empty context if not provided."""
        handler = triggers.TriggerHandler()
        assert handler.ctx == {}

    def test_handler_init_with_none_context(self):
        """Should initialize with empty context if None provided."""
        handler = triggers.TriggerHandler(None)
        assert handler.ctx == {}


# =============================================================================
# Additional Tests - Route Registration with Custom Prefix
# =============================================================================


class TestTriggerRouteRegistrationCustomPrefix:
    """Tests for route registration with custom prefix."""

    def test_register_routes_custom_prefix(self):
        """Should register routes with custom prefix."""
        app = web.Application()
        triggers.TriggerHandler.register_routes(app, prefix="/api/v2/custom")

        routes = [r.resource.canonical for r in app.router.routes()]
        assert "/api/v2/custom/triggers" in routes
        assert "/api/v2/custom/triggers/{trigger_id}" in routes
        assert "/api/v2/custom/triggers/{trigger_id}/enable" in routes
        assert "/api/v2/custom/triggers/{trigger_id}/disable" in routes
        assert "/api/v2/custom/triggers/start" in routes
        assert "/api/v2/custom/triggers/stop" in routes


# =============================================================================
# Additional Tests - Trigger Data
# =============================================================================


class TestTriggerData:
    """Tests for trigger data handling."""

    @pytest.mark.asyncio
    async def test_list_triggers_returns_all_fields(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should return all trigger fields."""
        test_trigger = MockTrigger(
            id="full-trigger",
            name="Full Trigger",
            interval_seconds=7200,
            cron_expression="0 0 * * *",
            enabled=True,
            run_count=5,
            max_runs=10,
            metadata={"topic": "AI"},
        )
        mock_trigger._triggers = {"full-trigger": test_trigger}
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.list_triggers(request)

            assert response.status == 200
            body = json.loads(response.body)
            trigger = body["triggers"][0]
            assert trigger["id"] == "full-trigger"
            assert trigger["name"] == "Full Trigger"
            assert trigger["interval_seconds"] == 7200
            assert trigger["cron_expression"] == "0 0 * * *"
            assert trigger["enabled"] is True
            assert trigger["run_count"] == 5
            assert trigger["max_runs"] == 10
            assert trigger["metadata"] == {"topic": "AI"}

    @pytest.mark.asyncio
    async def test_list_triggers_with_last_run(
        self, mock_trigger, mock_auth_context, mock_permission_checker
    ):
        """Should include last_run if set."""
        last_run = datetime.now()
        test_trigger = MockTrigger(id="ran-trigger", last_run=last_run)
        mock_trigger._triggers = {"ran-trigger": test_trigger}
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_permission_checker),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.list_triggers(request)

            body = json.loads(response.body)
            trigger = body["triggers"][0]
            assert trigger["last_run"] == last_run.isoformat()


# =============================================================================
# Additional Tests - Permission Checks
# =============================================================================


class TestTriggerPermissions:
    """Tests for permission handling."""

    @pytest.mark.asyncio
    async def test_list_triggers_requires_read_permission(self, mock_trigger, mock_auth_context):
        """Should check read permission."""
        mock_checker = MockPermissionChecker()
        mock_checker.check_permission = MagicMock(
            return_value=MockPermissionDecision(allowed=False, reason="No read access")
        )
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_checker),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.list_triggers(request)

            assert response.status == 403
            mock_checker.check_permission.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_trigger_requires_write_permission(self, mock_trigger, mock_auth_context):
        """Should check write permission."""
        mock_checker = MockPermissionChecker()
        mock_checker.check_permission = MagicMock(
            return_value=MockPermissionDecision(allowed=False, reason="No write access")
        )
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(triggers, "_get_circuit_breaker", return_value=cb),
            patch.object(triggers, "get_scheduled_trigger", return_value=mock_trigger),
            patch.object(triggers, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(triggers, "get_permission_checker", return_value=mock_checker),
        ):
            request = MagicMock()
            response = await triggers.TriggerHandler.add_trigger(request)

            assert response.status == 403
