"""Comprehensive tests for TriggerHandler.

Tests cover:
- list_triggers (GET /api/v1/autonomous/triggers)
- add_trigger (POST /api/v1/autonomous/triggers)
- remove_trigger (DELETE /api/v1/autonomous/triggers/{trigger_id})
- enable_trigger (POST /api/v1/autonomous/triggers/{trigger_id}/enable)
- disable_trigger (POST /api/v1/autonomous/triggers/{trigger_id}/disable)
- start_scheduler (POST /api/v1/autonomous/triggers/start)
- stop_scheduler (POST /api/v1/autonomous/triggers/stop)
- Auth / permission checks
- Circuit breaker behaviour
- Error-handling paths
- Global accessors (get/set_scheduled_trigger)
- register_routes
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.autonomous.triggers import (
    TriggerHandler,
    get_scheduled_trigger,
    set_scheduled_trigger,
    _get_circuit_breaker,
    AUTONOMOUS_READ_PERMISSION,
    AUTONOMOUS_WRITE_PERMISSION,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _parse(response: web.Response) -> dict:
    """Extract JSON dict from an aiohttp json_response."""
    return json.loads(response.body)


def _make_request(
    method: str = "GET",
    query: dict | None = None,
    match_info: dict | None = None,
    body: dict | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics an aiohttp web.Request."""
    req = MagicMock()
    req.method = method
    req.query = query or {}

    mi_data = match_info or {}
    mi_mock = MagicMock()
    mi_mock.get = MagicMock(side_effect=lambda k, default=None: mi_data.get(k, default))
    req.match_info = mi_mock

    if body is not None:
        req.json = AsyncMock(return_value=body)
        raw = json.dumps(body).encode()
        req.read = AsyncMock(return_value=raw)
        req.text = AsyncMock(return_value=json.dumps(body))
        req.content_type = "application/json"
        req.content_length = len(raw)
        req.can_read_body = True
    else:
        req.json = AsyncMock(return_value={})
        req.read = AsyncMock(return_value=b"{}")
        req.text = AsyncMock(return_value="{}")
        req.content_type = "application/json"
        req.content_length = 2
        req.can_read_body = True

    req.remote = "127.0.0.1"
    req.transport = MagicMock()
    req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)

    return req


def _make_trigger_config(
    trigger_id: str = "trig-1",
    name: str = "Daily review",
    interval_seconds: int | None = 3600,
    cron_expression: str | None = None,
    enabled: bool = True,
    last_run: datetime | None = None,
    next_run: datetime | None = None,
    run_count: int = 0,
    max_runs: int | None = None,
    metadata: dict | None = None,
) -> MagicMock:
    """Build a mock ScheduledTriggerConfig."""
    obj = MagicMock()
    obj.id = trigger_id
    obj.name = name
    obj.interval_seconds = interval_seconds
    obj.cron_expression = cron_expression
    obj.enabled = enabled
    obj.last_run = last_run
    obj.next_run = next_run or datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    obj.run_count = run_count
    obj.max_runs = max_runs
    obj.metadata = metadata or {}
    return obj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_trigger_globals():
    """Reset the global scheduled trigger and circuit breaker between tests."""
    import aragora.server.handlers.autonomous.triggers as mod

    old_trigger = mod._scheduled_trigger
    old_cb = mod._trigger_circuit_breaker
    mod._scheduled_trigger = None
    mod._trigger_circuit_breaker = None
    yield
    mod._scheduled_trigger = old_trigger
    mod._trigger_circuit_breaker = old_cb


@pytest.fixture
def mock_trigger():
    """Create a mock ScheduledTrigger instance."""
    trigger = MagicMock()
    trigger.list_triggers = MagicMock(return_value=[])
    trigger.add_trigger = MagicMock(return_value=_make_trigger_config())
    trigger.remove_trigger = MagicMock(return_value=True)
    trigger.enable_trigger = MagicMock(return_value=True)
    trigger.disable_trigger = MagicMock(return_value=True)
    trigger.start = AsyncMock()
    trigger.stop = AsyncMock()
    return trigger


@pytest.fixture
def install_trigger(mock_trigger):
    """Set mock trigger as the global singleton."""
    set_scheduled_trigger(mock_trigger)
    return mock_trigger


@pytest.fixture
def mock_cb():
    """Create a mock circuit breaker that allows execution."""
    cb = MagicMock()
    cb.can_execute.return_value = True
    return cb


@pytest.fixture
def install_cb(mock_cb):
    """Patch _get_circuit_breaker to return our mock."""
    with patch(
        "aragora.server.handlers.autonomous.triggers._get_circuit_breaker",
        return_value=mock_cb,
    ):
        yield mock_cb


# ---------------------------------------------------------------------------
# list_triggers endpoint
# ---------------------------------------------------------------------------


class TestListTriggers:
    @pytest.mark.asyncio
    async def test_list_empty(self, install_trigger, install_cb):
        install_trigger.list_triggers.return_value = []
        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["triggers"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_list_single_trigger(self, install_trigger, install_cb):
        config = _make_trigger_config()
        install_trigger.list_triggers.return_value = [config]

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["count"] == 1
        assert data["triggers"][0]["id"] == "trig-1"
        assert data["triggers"][0]["name"] == "Daily review"
        assert data["triggers"][0]["interval_seconds"] == 3600
        assert data["triggers"][0]["enabled"] is True

    @pytest.mark.asyncio
    async def test_list_multiple_triggers(self, install_trigger, install_cb):
        configs = [
            _make_trigger_config(trigger_id="t-1", name="First"),
            _make_trigger_config(trigger_id="t-2", name="Second", enabled=False),
            _make_trigger_config(trigger_id="t-3", name="Third", run_count=5),
        ]
        install_trigger.list_triggers.return_value = configs

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["count"] == 3
        assert data["triggers"][1]["enabled"] is False
        assert data["triggers"][2]["run_count"] == 5

    @pytest.mark.asyncio
    async def test_list_trigger_with_last_run(self, install_trigger, install_cb):
        last = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        config = _make_trigger_config(last_run=last)
        install_trigger.list_triggers.return_value = [config]

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        data = await _parse(resp)
        assert data["triggers"][0]["last_run"] == last.isoformat()

    @pytest.mark.asyncio
    async def test_list_trigger_with_null_last_run(self, install_trigger, install_cb):
        config = _make_trigger_config(last_run=None)
        install_trigger.list_triggers.return_value = [config]

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        data = await _parse(resp)
        assert data["triggers"][0]["last_run"] is None

    @pytest.mark.asyncio
    async def test_list_trigger_with_null_next_run(self, install_trigger, install_cb):
        config = _make_trigger_config(next_run=None)
        # Override mock to actually return None for next_run
        config.next_run = None
        install_trigger.list_triggers.return_value = [config]

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        data = await _parse(resp)
        assert data["triggers"][0]["next_run"] is None

    @pytest.mark.asyncio
    async def test_list_trigger_metadata(self, install_trigger, install_cb):
        config = _make_trigger_config(metadata={"topic": "security review", "agents": 5})
        install_trigger.list_triggers.return_value = [config]

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        data = await _parse(resp)
        assert data["triggers"][0]["metadata"]["topic"] == "security review"
        assert data["triggers"][0]["metadata"]["agents"] == 5

    @pytest.mark.asyncio
    async def test_list_trigger_max_runs(self, install_trigger, install_cb):
        config = _make_trigger_config(max_runs=10, run_count=3)
        install_trigger.list_triggers.return_value = [config]

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        data = await _parse(resp)
        assert data["triggers"][0]["max_runs"] == 10
        assert data["triggers"][0]["run_count"] == 3

    @pytest.mark.asyncio
    async def test_list_trigger_cron_expression(self, install_trigger, install_cb):
        config = _make_trigger_config(cron_expression="0 */6 * * *")
        install_trigger.list_triggers.return_value = [config]

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        data = await _parse(resp)
        assert data["triggers"][0]["cron_expression"] == "0 */6 * * *"

    @pytest.mark.asyncio
    async def test_list_circuit_breaker_open(self):
        cb = MagicMock()
        cb.can_execute.return_value = False

        with patch(
            "aragora.server.handlers.autonomous.triggers._get_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request()
            resp = await TriggerHandler.list_triggers(req)

        assert resp.status == 503
        data = await _parse(resp)
        assert data["success"] is False
        assert "unavailable" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_list_internal_error(self, install_trigger, install_cb):
        install_trigger.list_triggers.side_effect = RuntimeError("db down")

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert data["success"] is False
        assert "Failed to list triggers" in data["error"]

    @pytest.mark.asyncio
    async def test_list_key_error(self, install_trigger, install_cb):
        install_trigger.list_triggers.side_effect = KeyError("missing")

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_list_value_error(self, install_trigger, install_cb):
        install_trigger.list_triggers.side_effect = ValueError("invalid")

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_list_type_error(self, install_trigger, install_cb):
        install_trigger.list_triggers.side_effect = TypeError("bad type")

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_list_attribute_error(self, install_trigger, install_cb):
        install_trigger.list_triggers.side_effect = AttributeError("missing attr")

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_list_unauthorized(self, install_trigger, install_cb):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.triggers.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request()
            resp = await TriggerHandler.list_triggers(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_list_forbidden(self, install_trigger, install_cb):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.triggers.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.triggers.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request()
            resp = await TriggerHandler.list_triggers(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]

    @pytest.mark.asyncio
    async def test_list_response_keys(self, install_trigger, install_cb):
        config = _make_trigger_config()
        install_trigger.list_triggers.return_value = [config]

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        data = await _parse(resp)
        trigger = data["triggers"][0]
        expected_keys = {
            "id",
            "name",
            "interval_seconds",
            "cron_expression",
            "enabled",
            "last_run",
            "next_run",
            "run_count",
            "max_runs",
            "metadata",
        }
        assert expected_keys == set(trigger.keys())


# ---------------------------------------------------------------------------
# add_trigger endpoint
# ---------------------------------------------------------------------------


class TestAddTrigger:
    @pytest.mark.asyncio
    async def test_add_success(self, install_trigger, install_cb):
        config = _make_trigger_config(trigger_id="new-1", name="New trigger")
        install_trigger.add_trigger.return_value = config

        req = _make_request(
            method="POST",
            body={
                "trigger_id": "new-1",
                "name": "New trigger",
                "interval_seconds": 3600,
                "enabled": True,
            },
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["trigger"]["id"] == "new-1"
        assert data["trigger"]["name"] == "New trigger"

    @pytest.mark.asyncio
    async def test_add_with_cron(self, install_trigger, install_cb):
        config = _make_trigger_config()
        install_trigger.add_trigger.return_value = config

        req = _make_request(
            method="POST",
            body={
                "trigger_id": "cron-1",
                "name": "Cron trigger",
                "cron_expression": "0 */6 * * *",
            },
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True

    @pytest.mark.asyncio
    async def test_add_with_metadata(self, install_trigger, install_cb):
        config = _make_trigger_config()
        install_trigger.add_trigger.return_value = config

        req = _make_request(
            method="POST",
            body={
                "trigger_id": "meta-1",
                "name": "Meta trigger",
                "metadata": {"topic": "AI safety", "agents": 3, "rounds": 5},
            },
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 200
        install_trigger.add_trigger.assert_called_once()
        call_kwargs = install_trigger.add_trigger.call_args[1]
        assert call_kwargs["metadata"] == {"topic": "AI safety", "agents": 3, "rounds": 5}

    @pytest.mark.asyncio
    async def test_add_with_max_runs(self, install_trigger, install_cb):
        config = _make_trigger_config()
        install_trigger.add_trigger.return_value = config

        req = _make_request(
            method="POST",
            body={
                "trigger_id": "max-1",
                "name": "Limited trigger",
                "max_runs": 10,
                "interval_seconds": 60,
            },
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 200
        call_kwargs = install_trigger.add_trigger.call_args[1]
        assert call_kwargs["max_runs"] == 10

    @pytest.mark.asyncio
    async def test_add_disabled(self, install_trigger, install_cb):
        config = _make_trigger_config(enabled=False)
        install_trigger.add_trigger.return_value = config

        req = _make_request(
            method="POST",
            body={
                "trigger_id": "dis-1",
                "name": "Disabled trigger",
                "enabled": False,
            },
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 200
        call_kwargs = install_trigger.add_trigger.call_args[1]
        assert call_kwargs["enabled"] is False

    @pytest.mark.asyncio
    async def test_add_defaults_enabled_true(self, install_trigger, install_cb):
        config = _make_trigger_config()
        install_trigger.add_trigger.return_value = config

        req = _make_request(
            method="POST",
            body={"trigger_id": "def-1", "name": "Default trigger"},
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 200
        call_kwargs = install_trigger.add_trigger.call_args[1]
        assert call_kwargs["enabled"] is True

    @pytest.mark.asyncio
    async def test_add_missing_trigger_id(self, install_trigger, install_cb):
        req = _make_request(
            method="POST",
            body={"name": "No ID trigger"},
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False
        assert "trigger_id" in data["error"]

    @pytest.mark.asyncio
    async def test_add_missing_name(self, install_trigger, install_cb):
        req = _make_request(
            method="POST",
            body={"trigger_id": "no-name"},
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False
        assert "name" in data["error"]

    @pytest.mark.asyncio
    async def test_add_missing_both(self, install_trigger, install_cb):
        req = _make_request(
            method="POST",
            body={},
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_add_empty_trigger_id(self, install_trigger, install_cb):
        req = _make_request(
            method="POST",
            body={"trigger_id": "", "name": "Empty ID"},
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_add_empty_name(self, install_trigger, install_cb):
        req = _make_request(
            method="POST",
            body={"trigger_id": "valid-id", "name": ""},
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_add_null_trigger_id(self, install_trigger, install_cb):
        req = _make_request(
            method="POST",
            body={"trigger_id": None, "name": "Null ID"},
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_add_response_fields(self, install_trigger, install_cb):
        config = _make_trigger_config()
        install_trigger.add_trigger.return_value = config

        req = _make_request(
            method="POST",
            body={"trigger_id": "f-1", "name": "Fields test"},
        )
        resp = await TriggerHandler.add_trigger(req)

        data = await _parse(resp)
        trigger = data["trigger"]
        expected_keys = {"id", "name", "interval_seconds", "enabled", "next_run"}
        assert expected_keys == set(trigger.keys())

    @pytest.mark.asyncio
    async def test_add_circuit_breaker_open(self):
        cb = MagicMock()
        cb.can_execute.return_value = False

        with patch(
            "aragora.server.handlers.autonomous.triggers._get_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(
                method="POST",
                body={"trigger_id": "t-1", "name": "Test"},
            )
            resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 503

    @pytest.mark.asyncio
    async def test_add_internal_error(self, install_trigger, install_cb):
        install_trigger.add_trigger.side_effect = RuntimeError("boom")

        req = _make_request(
            method="POST",
            body={"trigger_id": "err-1", "name": "Error trigger"},
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert "Failed to create trigger" in data["error"]

    @pytest.mark.asyncio
    async def test_add_value_error(self, install_trigger, install_cb):
        install_trigger.add_trigger.side_effect = ValueError("bad value")

        req = _make_request(
            method="POST",
            body={"trigger_id": "err-2", "name": "Error trigger"},
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_add_unauthorized(self, install_trigger, install_cb):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.triggers.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(
                method="POST",
                body={"trigger_id": "t-1", "name": "Test"},
            )
            resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_add_forbidden(self, install_trigger, install_cb):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.triggers.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.triggers.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(
                method="POST",
                body={"trigger_id": "t-1", "name": "Test"},
            )
            resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_add_passes_all_params(self, install_trigger, install_cb):
        config = _make_trigger_config()
        install_trigger.add_trigger.return_value = config

        req = _make_request(
            method="POST",
            body={
                "trigger_id": "full-1",
                "name": "Full params",
                "interval_seconds": 7200,
                "cron_expression": "0 0 * * *",
                "enabled": False,
                "max_runs": 5,
                "metadata": {"key": "value"},
            },
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 200
        call_kwargs = install_trigger.add_trigger.call_args[1]
        assert call_kwargs["id"] == "full-1"
        assert call_kwargs["name"] == "Full params"
        assert call_kwargs["interval_seconds"] == 7200
        assert call_kwargs["cron_expression"] == "0 0 * * *"
        assert call_kwargs["enabled"] is False
        assert call_kwargs["max_runs"] == 5
        assert call_kwargs["metadata"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_add_next_run_null_in_response(self, install_trigger, install_cb):
        config = _make_trigger_config()
        config.next_run = None
        install_trigger.add_trigger.return_value = config

        req = _make_request(
            method="POST",
            body={"trigger_id": "null-nr", "name": "No next run"},
        )
        resp = await TriggerHandler.add_trigger(req)

        data = await _parse(resp)
        assert data["trigger"]["next_run"] is None


# ---------------------------------------------------------------------------
# remove_trigger endpoint
# ---------------------------------------------------------------------------


class TestRemoveTrigger:
    @pytest.mark.asyncio
    async def test_remove_success(self, install_trigger, install_cb):
        install_trigger.remove_trigger.return_value = True

        req = _make_request(method="DELETE", match_info={"trigger_id": "trig-1"})
        resp = await TriggerHandler.remove_trigger(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["trigger_id"] == "trig-1"
        assert data["removed"] is True

    @pytest.mark.asyncio
    async def test_remove_not_found(self, install_trigger, install_cb):
        install_trigger.remove_trigger.return_value = False

        req = _make_request(method="DELETE", match_info={"trigger_id": "nonexistent"})
        resp = await TriggerHandler.remove_trigger(req)

        assert resp.status == 404
        data = await _parse(resp)
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_remove_uses_hasattr_path(self, install_cb):
        """When trigger doesn't have remove_trigger method, fallback to _triggers."""
        trigger = MagicMock(spec=[])
        trigger._triggers = {"trig-1": _make_trigger_config()}
        set_scheduled_trigger(trigger)

        req = _make_request(method="DELETE", match_info={"trigger_id": "trig-1"})
        resp = await TriggerHandler.remove_trigger(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["removed"] is True

    @pytest.mark.asyncio
    async def test_remove_fallback_no_triggers_dict(self, install_cb):
        """When trigger lacks both remove_trigger and _triggers, returns not found."""
        trigger = MagicMock(spec=[])
        set_scheduled_trigger(trigger)

        req = _make_request(method="DELETE", match_info={"trigger_id": "trig-1"})
        resp = await TriggerHandler.remove_trigger(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_remove_fallback_trigger_not_in_dict(self, install_cb):
        """Fallback _triggers dict path returns not found when id not present."""
        trigger = MagicMock(spec=[])
        trigger._triggers = {}
        set_scheduled_trigger(trigger)

        req = _make_request(method="DELETE", match_info={"trigger_id": "nope"})
        resp = await TriggerHandler.remove_trigger(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_remove_circuit_breaker_open(self):
        cb = MagicMock()
        cb.can_execute.return_value = False

        with patch(
            "aragora.server.handlers.autonomous.triggers._get_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(method="DELETE", match_info={"trigger_id": "t-1"})
            resp = await TriggerHandler.remove_trigger(req)

        assert resp.status == 503

    @pytest.mark.asyncio
    async def test_remove_internal_error(self, install_trigger, install_cb):
        install_trigger.remove_trigger.side_effect = RuntimeError("boom")

        req = _make_request(method="DELETE", match_info={"trigger_id": "t-1"})
        resp = await TriggerHandler.remove_trigger(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert "Failed to remove trigger" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_remove_unauthorized(self, install_trigger, install_cb):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.triggers.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(method="DELETE", match_info={"trigger_id": "t-1"})
            resp = await TriggerHandler.remove_trigger(req)

        assert resp.status == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_remove_forbidden(self, install_trigger, install_cb):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.triggers.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.triggers.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(method="DELETE", match_info={"trigger_id": "t-1"})
            resp = await TriggerHandler.remove_trigger(req)

        assert resp.status == 403


# ---------------------------------------------------------------------------
# enable_trigger endpoint
# ---------------------------------------------------------------------------


class TestEnableTrigger:
    @pytest.mark.asyncio
    async def test_enable_success(self, install_trigger, install_cb):
        install_trigger.enable_trigger.return_value = True

        req = _make_request(method="POST", match_info={"trigger_id": "trig-1"})
        resp = await TriggerHandler.enable_trigger(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["trigger_id"] == "trig-1"
        assert data["enabled"] is True

    @pytest.mark.asyncio
    async def test_enable_not_found(self, install_trigger, install_cb):
        install_trigger.enable_trigger.return_value = False

        req = _make_request(method="POST", match_info={"trigger_id": "missing"})
        resp = await TriggerHandler.enable_trigger(req)

        assert resp.status == 404
        data = await _parse(resp)
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_enable_circuit_breaker_open(self):
        cb = MagicMock()
        cb.can_execute.return_value = False

        with patch(
            "aragora.server.handlers.autonomous.triggers._get_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(method="POST", match_info={"trigger_id": "t-1"})
            resp = await TriggerHandler.enable_trigger(req)

        assert resp.status == 503

    @pytest.mark.asyncio
    async def test_enable_internal_error(self, install_trigger, install_cb):
        install_trigger.enable_trigger.side_effect = RuntimeError("boom")

        req = _make_request(method="POST", match_info={"trigger_id": "t-1"})
        resp = await TriggerHandler.enable_trigger(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert "Failed to enable trigger" in data["error"]

    @pytest.mark.asyncio
    async def test_enable_value_error(self, install_trigger, install_cb):
        install_trigger.enable_trigger.side_effect = ValueError("bad")

        req = _make_request(method="POST", match_info={"trigger_id": "t-1"})
        resp = await TriggerHandler.enable_trigger(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_enable_unauthorized(self, install_trigger, install_cb):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.triggers.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(method="POST", match_info={"trigger_id": "t-1"})
            resp = await TriggerHandler.enable_trigger(req)

        assert resp.status == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_enable_forbidden(self, install_trigger, install_cb):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.triggers.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.triggers.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(method="POST", match_info={"trigger_id": "t-1"})
            resp = await TriggerHandler.enable_trigger(req)

        assert resp.status == 403


# ---------------------------------------------------------------------------
# disable_trigger endpoint
# ---------------------------------------------------------------------------


class TestDisableTrigger:
    @pytest.mark.asyncio
    async def test_disable_success(self, install_trigger, install_cb):
        install_trigger.disable_trigger.return_value = True

        req = _make_request(method="POST", match_info={"trigger_id": "trig-1"})
        resp = await TriggerHandler.disable_trigger(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["trigger_id"] == "trig-1"
        assert data["enabled"] is False

    @pytest.mark.asyncio
    async def test_disable_not_found(self, install_trigger, install_cb):
        install_trigger.disable_trigger.return_value = False

        req = _make_request(method="POST", match_info={"trigger_id": "missing"})
        resp = await TriggerHandler.disable_trigger(req)

        assert resp.status == 404
        data = await _parse(resp)
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_disable_circuit_breaker_open(self):
        cb = MagicMock()
        cb.can_execute.return_value = False

        with patch(
            "aragora.server.handlers.autonomous.triggers._get_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(method="POST", match_info={"trigger_id": "t-1"})
            resp = await TriggerHandler.disable_trigger(req)

        assert resp.status == 503

    @pytest.mark.asyncio
    async def test_disable_internal_error(self, install_trigger, install_cb):
        install_trigger.disable_trigger.side_effect = RuntimeError("boom")

        req = _make_request(method="POST", match_info={"trigger_id": "t-1"})
        resp = await TriggerHandler.disable_trigger(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert "Failed to disable trigger" in data["error"]

    @pytest.mark.asyncio
    async def test_disable_attribute_error(self, install_trigger, install_cb):
        install_trigger.disable_trigger.side_effect = AttributeError("missing")

        req = _make_request(method="POST", match_info={"trigger_id": "t-1"})
        resp = await TriggerHandler.disable_trigger(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_disable_unauthorized(self, install_trigger, install_cb):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.triggers.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(method="POST", match_info={"trigger_id": "t-1"})
            resp = await TriggerHandler.disable_trigger(req)

        assert resp.status == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_disable_forbidden(self, install_trigger, install_cb):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.triggers.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.triggers.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(method="POST", match_info={"trigger_id": "t-1"})
            resp = await TriggerHandler.disable_trigger(req)

        assert resp.status == 403


# ---------------------------------------------------------------------------
# start_scheduler endpoint
# ---------------------------------------------------------------------------


class TestStartScheduler:
    @pytest.mark.asyncio
    async def test_start_success(self, install_trigger):
        req = _make_request(method="POST")
        resp = await TriggerHandler.start_scheduler(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["scheduler_running"] is True
        install_trigger.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_internal_error(self, install_trigger):
        install_trigger.start.side_effect = RuntimeError("failed to start")

        req = _make_request(method="POST")
        resp = await TriggerHandler.start_scheduler(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert "Failed to start scheduler" in data["error"]

    @pytest.mark.asyncio
    async def test_start_os_error(self, install_trigger):
        install_trigger.start.side_effect = OSError("resource error")

        req = _make_request(method="POST")
        resp = await TriggerHandler.start_scheduler(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_start_value_error(self, install_trigger):
        install_trigger.start.side_effect = ValueError("invalid config")

        req = _make_request(method="POST")
        resp = await TriggerHandler.start_scheduler(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_start_key_error(self, install_trigger):
        install_trigger.start.side_effect = KeyError("missing key")

        req = _make_request(method="POST")
        resp = await TriggerHandler.start_scheduler(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_start_unauthorized(self, install_trigger):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.triggers.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(method="POST")
            resp = await TriggerHandler.start_scheduler(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_start_forbidden(self, install_trigger):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.triggers.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.triggers.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(method="POST")
            resp = await TriggerHandler.start_scheduler(req)

        assert resp.status == 403


# ---------------------------------------------------------------------------
# stop_scheduler endpoint
# ---------------------------------------------------------------------------


class TestStopScheduler:
    @pytest.mark.asyncio
    async def test_stop_success(self, install_trigger):
        req = _make_request(method="POST")
        resp = await TriggerHandler.stop_scheduler(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["scheduler_running"] is False
        install_trigger.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_internal_error(self, install_trigger):
        install_trigger.stop.side_effect = RuntimeError("failed to stop")

        req = _make_request(method="POST")
        resp = await TriggerHandler.stop_scheduler(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert "Failed to stop scheduler" in data["error"]

    @pytest.mark.asyncio
    async def test_stop_os_error(self, install_trigger):
        install_trigger.stop.side_effect = OSError("resource error")

        req = _make_request(method="POST")
        resp = await TriggerHandler.stop_scheduler(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_stop_type_error(self, install_trigger):
        install_trigger.stop.side_effect = TypeError("bad type")

        req = _make_request(method="POST")
        resp = await TriggerHandler.stop_scheduler(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_stop_unauthorized(self, install_trigger):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.triggers.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(method="POST")
            resp = await TriggerHandler.stop_scheduler(req)

        assert resp.status == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_stop_forbidden(self, install_trigger):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.triggers.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.triggers.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(method="POST")
            resp = await TriggerHandler.stop_scheduler(req)

        assert resp.status == 403


# ---------------------------------------------------------------------------
# register_routes and handler meta
# ---------------------------------------------------------------------------


class TestRegisterRoutes:
    def test_register_routes_default_prefix(self):
        app = web.Application()
        TriggerHandler.register_routes(app)

        route_paths = [
            r.resource.canonical
            for r in app.router.routes()
            if hasattr(r, "resource") and r.resource
        ]

        assert any("/api/v1/autonomous/triggers" == p for p in route_paths)
        assert any("trigger_id" in p and "/enable" in p for p in route_paths)
        assert any("trigger_id" in p and "/disable" in p for p in route_paths)
        assert any("/start" in p for p in route_paths)
        assert any("/stop" in p for p in route_paths)

    def test_register_routes_custom_prefix(self):
        app = web.Application()
        TriggerHandler.register_routes(app, prefix="/custom/api")

        route_paths = [
            r.resource.canonical
            for r in app.router.routes()
            if hasattr(r, "resource") and r.resource
        ]

        assert any("/custom/api/triggers" in p for p in route_paths)
        assert any("/custom/api/triggers/start" in p for p in route_paths)
        assert any("/custom/api/triggers/stop" in p for p in route_paths)

    def test_register_routes_count(self):
        """There should be 8 routes: 7 explicit + 1 implicit HEAD for GET."""
        app = web.Application()
        TriggerHandler.register_routes(app)

        route_count = sum(1 for r in app.router.routes() if hasattr(r, "resource") and r.resource)
        # 7 registered + aiohttp auto-adds HEAD for GET routes = 8
        assert route_count == 8

    def test_register_routes_methods(self):
        """Verify correct HTTP methods for each route."""
        app = web.Application()
        TriggerHandler.register_routes(app)

        method_map = {}
        for r in app.router.routes():
            if hasattr(r, "resource") and r.resource:
                method_map[r.resource.canonical] = r.method

        triggers_base = "/api/v1/autonomous/triggers"
        assert method_map.get(triggers_base) in ("GET", "POST")
        assert method_map.get(f"{triggers_base}/start") == "POST"
        assert method_map.get(f"{triggers_base}/stop") == "POST"

    def test_handler_init_default(self):
        handler = TriggerHandler()
        assert handler.ctx == {}

    def test_handler_init_custom_ctx(self):
        handler = TriggerHandler(ctx={"env": "test"})
        assert handler.ctx == {"env": "test"}


# ---------------------------------------------------------------------------
# Global accessors
# ---------------------------------------------------------------------------


class TestGlobalAccessors:
    def test_get_scheduled_trigger_creates_instance(self):
        trigger = get_scheduled_trigger()
        assert trigger is not None

    def test_get_scheduled_trigger_singleton(self):
        t1 = get_scheduled_trigger()
        t2 = get_scheduled_trigger()
        assert t1 is t2

    def test_set_and_get_scheduled_trigger(self):
        custom = MagicMock()
        set_scheduled_trigger(custom)
        assert get_scheduled_trigger() is custom

    def test_set_scheduled_trigger_replaces(self):
        first = MagicMock()
        second = MagicMock()
        set_scheduled_trigger(first)
        set_scheduled_trigger(second)
        assert get_scheduled_trigger() is second

    def test_get_circuit_breaker_creates_instance(self):
        cb = _get_circuit_breaker()
        assert cb is not None

    def test_get_circuit_breaker_singleton(self):
        cb1 = _get_circuit_breaker()
        cb2 = _get_circuit_breaker()
        assert cb1 is cb2

    def test_permission_constants(self):
        assert AUTONOMOUS_READ_PERMISSION == "autonomous:read"
        assert AUTONOMOUS_WRITE_PERMISSION == "autonomous:write"


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_add_trigger_invalid_json(self, install_trigger, install_cb):
        """Malformed JSON body returns 400 via parse_json_body."""
        req = MagicMock()
        req.method = "POST"
        req.json = AsyncMock(side_effect=ValueError("bad json"))
        req.read = AsyncMock(return_value=b"not json")
        req.text = AsyncMock(return_value="not json")
        req.content_type = "application/json"
        req.content_length = 8
        req.can_read_body = True
        req.remote = "127.0.0.1"
        req.transport = MagicMock()
        req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)
        mi = MagicMock()
        mi.get = MagicMock(return_value=None)
        req.match_info = mi

        resp = await TriggerHandler.add_trigger(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_enable_trigger_with_none_id(self, install_trigger, install_cb):
        """When match_info returns None for trigger_id."""
        install_trigger.enable_trigger.return_value = False

        req = _make_request(method="POST", match_info={})
        resp = await TriggerHandler.enable_trigger(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_disable_trigger_with_none_id(self, install_trigger, install_cb):
        """When match_info returns None for trigger_id."""
        install_trigger.disable_trigger.return_value = False

        req = _make_request(method="POST", match_info={})
        resp = await TriggerHandler.disable_trigger(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_remove_trigger_with_none_id(self, install_trigger, install_cb):
        """When match_info returns None for trigger_id."""
        install_trigger.remove_trigger.return_value = False

        req = _make_request(method="DELETE", match_info={})
        resp = await TriggerHandler.remove_trigger(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_list_triggers_calls_get_scheduled_trigger(self, install_cb):
        """Verify list_triggers goes through get_scheduled_trigger()."""
        mock_trigger = MagicMock()
        mock_trigger.list_triggers.return_value = []
        set_scheduled_trigger(mock_trigger)

        req = _make_request()
        resp = await TriggerHandler.list_triggers(req)

        assert resp.status == 200
        mock_trigger.list_triggers.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_trigger_calls_get_scheduled_trigger(self, install_cb):
        """Verify add_trigger goes through get_scheduled_trigger()."""
        mock_trigger = MagicMock()
        config = _make_trigger_config()
        mock_trigger.add_trigger.return_value = config
        set_scheduled_trigger(mock_trigger)

        req = _make_request(
            method="POST",
            body={"trigger_id": "x-1", "name": "Test"},
        )
        resp = await TriggerHandler.add_trigger(req)

        assert resp.status == 200
        mock_trigger.add_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_scheduler_no_circuit_breaker_check(self, install_trigger):
        """start_scheduler does NOT check circuit breaker (by design)."""
        req = _make_request(method="POST")
        resp = await TriggerHandler.start_scheduler(req)

        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_stop_scheduler_no_circuit_breaker_check(self, install_trigger):
        """stop_scheduler does NOT check circuit breaker (by design)."""
        req = _make_request(method="POST")
        resp = await TriggerHandler.stop_scheduler(req)

        assert resp.status == 200
