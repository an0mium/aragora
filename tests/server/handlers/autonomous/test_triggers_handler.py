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
