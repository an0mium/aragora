"""Tests for OpenClaw gateway handler endpoints.

Covers all 7 handler functions:
- handle_openclaw_execute
- handle_openclaw_status
- handle_openclaw_device_register
- handle_openclaw_device_unregister
- handle_openclaw_plugin_install
- handle_openclaw_plugin_uninstall
- handle_openclaw_config
- get_openclaw_handlers (registration helper)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# We import the handler module and patch its singleton so tests do not
# create real adapters or hit the network.
# ---------------------------------------------------------------------------
MODULE = "aragora.server.handlers.gateway.openclaw"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _body(result) -> dict[str, Any]:
    """Decode a HandlerResult body to dict."""
    if hasattr(result, "body"):
        return json.loads(result.body)
    return result


def _make_gateway_result(
    success: bool = True,
    request_id: str = "req-001",
    response: Any = None,
    error: str | None = None,
    blocked_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MagicMock:
    """Build a mock GatewayResult."""
    r = MagicMock()
    r.success = success
    r.request_id = request_id
    r.response = response
    r.error = error
    r.blocked_reason = blocked_reason
    r.metadata = metadata or {}
    return r


def _make_response(status: str = "completed", result: Any = "done") -> MagicMock:
    """Build a mock AragoraResponse."""
    resp = MagicMock()
    resp.status = status
    resp.result = result
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure the module-level adapter singleton is cleared between tests."""
    import aragora.server.handlers.gateway.openclaw as mod

    original = mod._gateway_adapter
    mod._gateway_adapter = None
    yield
    mod._gateway_adapter = original


@pytest.fixture()
def mock_adapter():
    """Provide a fully-mocked OpenClawGatewayAdapter and patch _get_gateway_adapter."""
    adapter = MagicMock()
    adapter.openclaw_endpoint = "http://localhost:8081"
    adapter.sandbox_config = MagicMock()
    adapter.sandbox_config.max_memory_mb = 512
    adapter.sandbox_config.max_cpu_percent = 50
    adapter.sandbox_config.max_execution_seconds = 300
    adapter.sandbox_config.allow_external_network = False
    adapter.sandbox_config.plugin_allowlist_mode = False
    adapter.sandbox_config.allowed_plugins = []
    adapter.capability_filter = MagicMock()
    adapter.capability_filter.blocked_override = set()
    adapter.capability_filter.tenant_enabled = set()

    adapter.execute_task = AsyncMock()
    adapter.register_device = AsyncMock()
    adapter.unregister_device = AsyncMock()
    adapter.install_plugin = AsyncMock()
    adapter.uninstall_plugin = AsyncMock()

    with patch(f"{MODULE}._get_gateway_adapter", return_value=adapter):
        yield adapter


# =========================================================================
# handle_openclaw_execute
# =========================================================================


class TestHandleOpenclawExecute:
    """Tests for POST /api/v1/gateway/openclaw/execute."""

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        resp_mock = _make_response("completed", "42")
        mock_adapter.execute_task.return_value = _make_gateway_result(
            success=True,
            request_id="req-123",
            response=resp_mock,
            metadata={"execution_time_ms": 150},
        )

        result = await handle_openclaw_execute(
            {"content": "What is 6*7?"}, user_id="u1"
        )
        body = _body(result)
        assert result.status_code == 200
        assert body["success"] is True
        assert body["data"]["request_id"] == "req-123"
        assert body["data"]["status"] == "completed"
        assert body["data"]["result"] == "42"
        assert body["data"]["execution_time_ms"] == 150

    @pytest.mark.asyncio
    async def test_execute_missing_content(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        result = await handle_openclaw_execute({}, user_id="u1")
        body = _body(result)
        assert result.status_code == 400
        assert "content" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_empty_content(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        result = await handle_openclaw_execute({"content": ""}, user_id="u1")
        body = _body(result)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_execute_failure_blocked(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        mock_adapter.execute_task.return_value = _make_gateway_result(
            success=False,
            error="Capabilities blocked",
            blocked_reason="capability_blocked",
        )

        result = await handle_openclaw_execute(
            {"content": "test"}, user_id="u1"
        )
        body = _body(result)
        assert result.status_code == 400
        assert "blocked" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_failure_server_error(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        mock_adapter.execute_task.return_value = _make_gateway_result(
            success=False,
            error="Sandbox crashed",
            blocked_reason=None,
        )

        result = await handle_openclaw_execute(
            {"content": "test"}, user_id="u1"
        )
        body = _body(result)
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_execute_failure_no_error_message(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        mock_adapter.execute_task.return_value = _make_gateway_result(
            success=False,
            error=None,
            blocked_reason=None,
        )

        result = await handle_openclaw_execute(
            {"content": "test"}, user_id="u1"
        )
        body = _body(result)
        assert result.status_code == 500
        assert "failed" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_connection_error(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        mock_adapter.execute_task.side_effect = ConnectionError("unreachable")

        result = await handle_openclaw_execute(
            {"content": "test"}, user_id="u1"
        )
        body = _body(result)
        assert result.status_code == 500
        assert "failed" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_execute_timeout_error(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        mock_adapter.execute_task.side_effect = TimeoutError("timed out")

        result = await handle_openclaw_execute(
            {"content": "test"}, user_id="u1"
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_execute_value_error(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        mock_adapter.execute_task.side_effect = ValueError("bad data")

        result = await handle_openclaw_execute(
            {"content": "test"}, user_id="u1"
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_execute_os_error(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        mock_adapter.execute_task.side_effect = OSError("disk fail")

        result = await handle_openclaw_execute(
            {"content": "test"}, user_id="u1"
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_execute_type_error(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        mock_adapter.execute_task.side_effect = TypeError("bad type")

        result = await handle_openclaw_execute(
            {"content": "test"}, user_id="u1"
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_execute_key_error(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        mock_adapter.execute_task.side_effect = KeyError("missing")

        result = await handle_openclaw_execute(
            {"content": "test"}, user_id="u1"
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_execute_with_optional_fields(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        resp_mock = _make_response("completed", {"answer": "yes"})
        mock_adapter.execute_task.return_value = _make_gateway_result(
            success=True,
            request_id="req-456",
            response=resp_mock,
            metadata={"execution_time_ms": 75},
        )

        data = {
            "content": "Is the sky blue?",
            "request_type": "query",
            "capabilities": ["vision"],
            "plugins": ["image-gen"],
            "priority": "high",
            "timeout_seconds": 60,
            "context": {"source": "test"},
            "metadata": {"trace": "t-1"},
        }
        result = await handle_openclaw_execute(data, user_id="u2")
        body = _body(result)
        assert result.status_code == 200
        assert body["data"]["request_id"] == "req-456"

        # Verify request constructed with our fields
        call_args = mock_adapter.execute_task.call_args
        request_obj = call_args.kwargs.get("request") or call_args[1].get("request") or call_args[0][0]
        assert request_obj.content == "Is the sky blue?"
        assert request_obj.request_type == "query"
        assert request_obj.priority == "high"

    @pytest.mark.asyncio
    async def test_execute_timeout_clamped_to_max(self, mock_adapter):
        """timeout_seconds exceeding sandbox max should be clamped."""
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        mock_adapter.sandbox_config.max_execution_seconds = 100

        resp_mock = _make_response()
        mock_adapter.execute_task.return_value = _make_gateway_result(
            success=True, response=resp_mock, metadata={"execution_time_ms": 0}
        )

        await handle_openclaw_execute(
            {"content": "test", "timeout_seconds": 999}, user_id="u1"
        )

        call_args = mock_adapter.execute_task.call_args
        request_obj = call_args.kwargs.get("request") or call_args[1].get("request") or call_args[0][0]
        assert request_obj.timeout_seconds <= 100

    @pytest.mark.asyncio
    async def test_execute_timeout_invalid_type_defaults(self, mock_adapter):
        """Non-integer timeout falls back to default 300."""
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        resp_mock = _make_response()
        mock_adapter.execute_task.return_value = _make_gateway_result(
            success=True, response=resp_mock, metadata={"execution_time_ms": 0}
        )

        await handle_openclaw_execute(
            {"content": "test", "timeout_seconds": "not-a-number"}, user_id="u1"
        )

        call_args = mock_adapter.execute_task.call_args
        request_obj = call_args.kwargs.get("request") or call_args[1].get("request") or call_args[0][0]
        assert request_obj.timeout_seconds == 300

    @pytest.mark.asyncio
    async def test_execute_timeout_negative_defaults(self, mock_adapter):
        """Negative timeout falls back to default 300."""
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        resp_mock = _make_response()
        mock_adapter.execute_task.return_value = _make_gateway_result(
            success=True, response=resp_mock, metadata={"execution_time_ms": 0}
        )

        await handle_openclaw_execute(
            {"content": "test", "timeout_seconds": -5}, user_id="u1"
        )

        call_args = mock_adapter.execute_task.call_args
        request_obj = call_args.kwargs.get("request") or call_args[1].get("request") or call_args[0][0]
        assert request_obj.timeout_seconds == 300

    @pytest.mark.asyncio
    async def test_execute_timeout_zero_defaults(self, mock_adapter):
        """Zero timeout falls back to default 300."""
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        resp_mock = _make_response()
        mock_adapter.execute_task.return_value = _make_gateway_result(
            success=True, response=resp_mock, metadata={"execution_time_ms": 0}
        )

        await handle_openclaw_execute(
            {"content": "test", "timeout_seconds": 0}, user_id="u1"
        )

        call_args = mock_adapter.execute_task.call_args
        request_obj = call_args.kwargs.get("request") or call_args[1].get("request") or call_args[0][0]
        assert request_obj.timeout_seconds == 300

    @pytest.mark.asyncio
    async def test_execute_uses_default_user_id(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        resp_mock = _make_response()
        mock_adapter.execute_task.return_value = _make_gateway_result(
            success=True, response=resp_mock, metadata={"execution_time_ms": 0}
        )

        await handle_openclaw_execute({"content": "test"})

        call_args = mock_adapter.execute_task.call_args
        auth_ctx = call_args.kwargs.get("auth_context") or call_args[1].get("auth_context") or call_args[0][1]
        assert auth_ctx.actor_id == "default"

    @pytest.mark.asyncio
    async def test_execute_with_tenant_context(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        resp_mock = _make_response()
        mock_adapter.execute_task.return_value = _make_gateway_result(
            success=True, response=resp_mock, metadata={"execution_time_ms": 0}
        )

        data = {
            "content": "test",
            "tenant_id": "t-100",
            "organization_id": "org-50",
            "workspace_id": "ws-25",
        }
        await handle_openclaw_execute(data, user_id="u1")

        call_args = mock_adapter.execute_task.call_args
        tenant_ctx = call_args.kwargs.get("tenant_context") or call_args[1].get("tenant_context") or call_args[0][2]
        assert tenant_ctx.tenant_id == "t-100"
        assert tenant_ctx.organization_id == "org-50"

    @pytest.mark.asyncio
    async def test_execute_no_tenant_context(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        resp_mock = _make_response()
        mock_adapter.execute_task.return_value = _make_gateway_result(
            success=True, response=resp_mock, metadata={"execution_time_ms": 0}
        )

        await handle_openclaw_execute({"content": "test"}, user_id="u1")

        call_args = mock_adapter.execute_task.call_args
        # tenant_context may be passed as kwarg or positional; when None it
        # won't survive an `or` chain so we check kwargs directly.
        if "tenant_context" in call_args.kwargs:
            tenant_ctx = call_args.kwargs["tenant_context"]
        elif len(call_args[0]) > 2:
            tenant_ctx = call_args[0][2]
        else:
            tenant_ctx = None
        assert tenant_ctx is None

    @pytest.mark.asyncio
    async def test_execute_response_is_none(self, mock_adapter):
        """When result.response is None, status/result fields default."""
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_execute

        mock_adapter.execute_task.return_value = _make_gateway_result(
            success=True,
            request_id="req-nil",
            response=None,
            metadata={"execution_time_ms": 0},
        )

        result = await handle_openclaw_execute({"content": "test"}, user_id="u1")
        body = _body(result)
        assert result.status_code == 200
        assert body["data"]["status"] == "completed"
        assert body["data"]["result"] is None


# =========================================================================
# handle_openclaw_status
# =========================================================================


class TestHandleOpenclawStatus:
    """Tests for GET /api/v1/gateway/openclaw/status/:task_id."""

    @pytest.mark.asyncio
    async def test_status_success(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_status

        result = await handle_openclaw_status(
            {"task_id": "task-abc"}, user_id="u1"
        )
        body = _body(result)
        assert result.status_code == 200
        assert body["data"]["task_id"] == "task-abc"
        assert body["data"]["status"] == "unknown"

    @pytest.mark.asyncio
    async def test_status_missing_task_id(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_status

        result = await handle_openclaw_status({}, user_id="u1")
        body = _body(result)
        assert result.status_code == 400
        assert "task_id" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_status_empty_task_id(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_status

        result = await handle_openclaw_status({"task_id": ""}, user_id="u1")
        body = _body(result)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_status_uses_default_user_id(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_status

        result = await handle_openclaw_status({"task_id": "t1"})
        assert result.status_code == 200


# =========================================================================
# handle_openclaw_device_register
# =========================================================================


class TestHandleOpenclawDeviceRegister:
    """Tests for POST /api/v1/gateway/openclaw/devices."""

    @pytest.mark.asyncio
    async def test_register_success(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_register

        mock_adapter.register_device.return_value = _make_gateway_result(
            success=True,
            request_id="dev-1",
            metadata={
                "device_handle": {
                    "registration_id": "reg-001",
                    "registered_at": "2026-01-01T00:00:00Z",
                }
            },
        )

        data = {
            "device_id": "dev-1",
            "device_name": "My Desktop",
            "device_type": "desktop",
            "capabilities": ["compute"],
        }
        result = await handle_openclaw_device_register(data, user_id="u1")
        body = _body(result)
        assert result.status_code == 200
        assert body["data"]["device_id"] == "dev-1"
        assert body["data"]["registration_id"] == "reg-001"
        assert body["data"]["registered_at"] == "2026-01-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_register_missing_device_id(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_register

        data = {"device_name": "name", "device_type": "desktop"}
        result = await handle_openclaw_device_register(data, user_id="u1")
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_register_missing_device_name(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_register

        data = {"device_id": "d1", "device_type": "desktop"}
        result = await handle_openclaw_device_register(data, user_id="u1")
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_register_missing_device_type(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_register

        data = {"device_id": "d1", "device_name": "name"}
        result = await handle_openclaw_device_register(data, user_id="u1")
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_register_missing_all_fields(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_register

        result = await handle_openclaw_device_register({}, user_id="u1")
        body = _body(result)
        assert result.status_code == 400
        assert "device_id" in body["error"]

    @pytest.mark.asyncio
    async def test_register_adapter_failure(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_register

        mock_adapter.register_device.return_value = _make_gateway_result(
            success=False, error="Permission denied"
        )

        data = {"device_id": "d1", "device_name": "name", "device_type": "desktop"}
        result = await handle_openclaw_device_register(data, user_id="u1")
        body = _body(result)
        assert result.status_code == 400
        assert "denied" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_register_adapter_failure_no_message(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_register

        mock_adapter.register_device.return_value = _make_gateway_result(
            success=False, error=None
        )

        data = {"device_id": "d1", "device_name": "name", "device_type": "desktop"}
        result = await handle_openclaw_device_register(data, user_id="u1")
        body = _body(result)
        assert result.status_code == 400
        assert "registration failed" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_register_connection_error(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_register

        mock_adapter.register_device.side_effect = ConnectionError("down")

        data = {"device_id": "d1", "device_name": "name", "device_type": "desktop"}
        result = await handle_openclaw_device_register(data, user_id="u1")
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_register_with_metadata(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_register

        mock_adapter.register_device.return_value = _make_gateway_result(
            success=True,
            metadata={"device_handle": {"registration_id": "r1", "registered_at": "now"}},
        )

        data = {
            "device_id": "d1",
            "device_name": "name",
            "device_type": "mobile",
            "metadata": {"os": "ios"},
        }
        result = await handle_openclaw_device_register(data, user_id="u1")
        assert result.status_code == 200


# =========================================================================
# handle_openclaw_device_unregister
# =========================================================================


class TestHandleOpenclawDeviceUnregister:
    """Tests for DELETE /api/v1/gateway/openclaw/devices/:device_id."""

    @pytest.mark.asyncio
    async def test_unregister_success(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_unregister

        mock_adapter.unregister_device.return_value = _make_gateway_result(success=True)

        result = await handle_openclaw_device_unregister(
            {"device_id": "dev-1"}, user_id="u1"
        )
        body = _body(result)
        assert result.status_code == 200
        assert body["data"]["device_id"] == "dev-1"
        assert body["data"]["unregistered"] is True

    @pytest.mark.asyncio
    async def test_unregister_missing_device_id(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_unregister

        result = await handle_openclaw_device_unregister({}, user_id="u1")
        body = _body(result)
        assert result.status_code == 400
        assert "device_id" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_unregister_empty_device_id(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_unregister

        result = await handle_openclaw_device_unregister({"device_id": ""}, user_id="u1")
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_unregister_adapter_failure(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_unregister

        mock_adapter.unregister_device.return_value = _make_gateway_result(
            success=False, error="Not found"
        )

        result = await handle_openclaw_device_unregister(
            {"device_id": "dev-1"}, user_id="u1"
        )
        body = _body(result)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_unregister_adapter_failure_no_message(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_unregister

        mock_adapter.unregister_device.return_value = _make_gateway_result(
            success=False, error=None
        )

        result = await handle_openclaw_device_unregister(
            {"device_id": "dev-1"}, user_id="u1"
        )
        body = _body(result)
        assert result.status_code == 400
        assert "unregistration failed" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_unregister_connection_error(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_unregister

        mock_adapter.unregister_device.side_effect = ConnectionError("down")

        result = await handle_openclaw_device_unregister(
            {"device_id": "dev-1"}, user_id="u1"
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_unregister_timeout_error(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_unregister

        mock_adapter.unregister_device.side_effect = TimeoutError("timeout")

        result = await handle_openclaw_device_unregister(
            {"device_id": "dev-1"}, user_id="u1"
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_unregister_uses_default_user_id(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_device_unregister

        mock_adapter.unregister_device.return_value = _make_gateway_result(success=True)
        result = await handle_openclaw_device_unregister({"device_id": "d1"})
        assert result.status_code == 200


# =========================================================================
# handle_openclaw_plugin_install
# =========================================================================


class TestHandleOpenclawPluginInstall:
    """Tests for POST /api/v1/gateway/openclaw/plugins."""

    @pytest.mark.asyncio
    async def test_install_success(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_install

        mock_adapter.install_plugin.return_value = _make_gateway_result(
            success=True,
            metadata={"installed_at": "2026-01-15T12:00:00Z"},
        )

        data = {
            "plugin_id": "plug-1",
            "plugin_name": "Image Gen",
            "version": "2.0.0",
            "source": "marketplace",
        }
        result = await handle_openclaw_plugin_install(data, user_id="u1")
        body = _body(result)
        assert result.status_code == 200
        assert body["data"]["plugin_id"] == "plug-1"
        assert body["data"]["installed"] is True
        assert body["data"]["installed_at"] == "2026-01-15T12:00:00Z"

    @pytest.mark.asyncio
    async def test_install_missing_plugin_id(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_install

        data = {"plugin_name": "name", "version": "1.0"}
        result = await handle_openclaw_plugin_install(data, user_id="u1")
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_install_missing_plugin_name(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_install

        data = {"plugin_id": "p1", "version": "1.0"}
        result = await handle_openclaw_plugin_install(data, user_id="u1")
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_install_missing_version(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_install

        data = {"plugin_id": "p1", "plugin_name": "name"}
        result = await handle_openclaw_plugin_install(data, user_id="u1")
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_install_missing_all_fields(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_install

        result = await handle_openclaw_plugin_install({}, user_id="u1")
        body = _body(result)
        assert result.status_code == 400
        assert "plugin_id" in body["error"]

    @pytest.mark.asyncio
    async def test_install_default_source(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_install

        mock_adapter.install_plugin.return_value = _make_gateway_result(
            success=True, metadata={"installed_at": "now"}
        )

        data = {"plugin_id": "p1", "plugin_name": "n", "version": "1.0"}
        await handle_openclaw_plugin_install(data, user_id="u1")

        call_args = mock_adapter.install_plugin.call_args
        plugin_obj = call_args[0][0] if call_args[0] else call_args.kwargs.get("plugin")
        assert plugin_obj.source == "marketplace"

    @pytest.mark.asyncio
    async def test_install_custom_source(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_install

        mock_adapter.install_plugin.return_value = _make_gateway_result(
            success=True, metadata={"installed_at": "now"}
        )

        data = {"plugin_id": "p1", "plugin_name": "n", "version": "1.0", "source": "local"}
        await handle_openclaw_plugin_install(data, user_id="u1")

        call_args = mock_adapter.install_plugin.call_args
        plugin_obj = call_args[0][0] if call_args[0] else call_args.kwargs.get("plugin")
        assert plugin_obj.source == "local"

    @pytest.mark.asyncio
    async def test_install_adapter_failure(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_install

        mock_adapter.install_plugin.return_value = _make_gateway_result(
            success=False, error="Not in allowlist"
        )

        data = {"plugin_id": "p1", "plugin_name": "n", "version": "1.0"}
        result = await handle_openclaw_plugin_install(data, user_id="u1")
        body = _body(result)
        assert result.status_code == 400
        assert "allowlist" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_install_adapter_failure_no_message(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_install

        mock_adapter.install_plugin.return_value = _make_gateway_result(
            success=False, error=None
        )

        data = {"plugin_id": "p1", "plugin_name": "n", "version": "1.0"}
        result = await handle_openclaw_plugin_install(data, user_id="u1")
        body = _body(result)
        assert result.status_code == 400
        assert "installation failed" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_install_connection_error(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_install

        mock_adapter.install_plugin.side_effect = ConnectionError("down")

        data = {"plugin_id": "p1", "plugin_name": "n", "version": "1.0"}
        result = await handle_openclaw_plugin_install(data, user_id="u1")
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_install_with_tenant_context(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_install

        mock_adapter.install_plugin.return_value = _make_gateway_result(
            success=True, metadata={"installed_at": "now"}
        )

        data = {
            "plugin_id": "p1",
            "plugin_name": "n",
            "version": "1.0",
            "tenant_id": "t-200",
            "enabled_capabilities": ["cap1"],
        }
        await handle_openclaw_plugin_install(data, user_id="u1")

        call_args = mock_adapter.install_plugin.call_args
        tenant_ctx = call_args[0][2] if len(call_args[0]) > 2 else call_args.kwargs.get("tenant_context")
        assert tenant_ctx is not None
        assert tenant_ctx.tenant_id == "t-200"

    @pytest.mark.asyncio
    async def test_install_with_metadata(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_install

        mock_adapter.install_plugin.return_value = _make_gateway_result(
            success=True, metadata={"installed_at": "now"}
        )

        data = {
            "plugin_id": "p1",
            "plugin_name": "n",
            "version": "1.0",
            "metadata": {"author": "aragora"},
        }
        await handle_openclaw_plugin_install(data, user_id="u1")

        call_args = mock_adapter.install_plugin.call_args
        plugin_obj = call_args[0][0] if call_args[0] else call_args.kwargs.get("plugin")
        assert plugin_obj.metadata == {"author": "aragora"}


# =========================================================================
# handle_openclaw_plugin_uninstall
# =========================================================================


class TestHandleOpenclawPluginUninstall:
    """Tests for DELETE /api/v1/gateway/openclaw/plugins/:plugin_id."""

    @pytest.mark.asyncio
    async def test_uninstall_success(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_uninstall

        mock_adapter.uninstall_plugin.return_value = _make_gateway_result(success=True)

        result = await handle_openclaw_plugin_uninstall(
            {"plugin_id": "plug-1"}, user_id="u1"
        )
        body = _body(result)
        assert result.status_code == 200
        assert body["data"]["plugin_id"] == "plug-1"
        assert body["data"]["uninstalled"] is True

    @pytest.mark.asyncio
    async def test_uninstall_missing_plugin_id(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_uninstall

        result = await handle_openclaw_plugin_uninstall({}, user_id="u1")
        body = _body(result)
        assert result.status_code == 400
        assert "plugin_id" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_uninstall_empty_plugin_id(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_uninstall

        result = await handle_openclaw_plugin_uninstall({"plugin_id": ""}, user_id="u1")
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_uninstall_adapter_failure(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_uninstall

        mock_adapter.uninstall_plugin.return_value = _make_gateway_result(
            success=False, error="Not found"
        )

        result = await handle_openclaw_plugin_uninstall(
            {"plugin_id": "p1"}, user_id="u1"
        )
        body = _body(result)
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_uninstall_adapter_failure_no_message(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_uninstall

        mock_adapter.uninstall_plugin.return_value = _make_gateway_result(
            success=False, error=None
        )

        result = await handle_openclaw_plugin_uninstall(
            {"plugin_id": "p1"}, user_id="u1"
        )
        body = _body(result)
        assert result.status_code == 400
        assert "uninstallation failed" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_uninstall_connection_error(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_uninstall

        mock_adapter.uninstall_plugin.side_effect = ConnectionError("down")

        result = await handle_openclaw_plugin_uninstall(
            {"plugin_id": "p1"}, user_id="u1"
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_uninstall_timeout_error(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_uninstall

        mock_adapter.uninstall_plugin.side_effect = TimeoutError("timeout")

        result = await handle_openclaw_plugin_uninstall(
            {"plugin_id": "p1"}, user_id="u1"
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_uninstall_value_error(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_uninstall

        mock_adapter.uninstall_plugin.side_effect = ValueError("bad")

        result = await handle_openclaw_plugin_uninstall(
            {"plugin_id": "p1"}, user_id="u1"
        )
        assert result.status_code == 500

    @pytest.mark.asyncio
    async def test_uninstall_uses_default_user(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_plugin_uninstall

        mock_adapter.uninstall_plugin.return_value = _make_gateway_result(success=True)
        result = await handle_openclaw_plugin_uninstall({"plugin_id": "p1"})
        assert result.status_code == 200


# =========================================================================
# handle_openclaw_config
# =========================================================================


class TestHandleOpenclawConfig:
    """Tests for GET /api/v1/gateway/openclaw/config."""

    @pytest.mark.asyncio
    async def test_config_success(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_config

        result = await handle_openclaw_config({}, user_id="u1")
        body = _body(result)
        assert result.status_code == 200
        assert body["success"] is True

        config = body["data"]
        assert config["endpoint"] == "http://localhost:8081"
        assert config["sandbox"]["max_memory_mb"] == 512
        assert config["sandbox"]["max_cpu_percent"] == 50
        assert config["sandbox"]["max_execution_seconds"] == 300
        assert config["sandbox"]["allow_external_network"] is False
        assert config["sandbox"]["plugin_allowlist_mode"] is False
        assert config["sandbox"]["allowed_plugins_count"] == 0
        assert config["capabilities"]["blocked_override_count"] == 0
        assert config["capabilities"]["tenant_enabled_count"] == 0

    @pytest.mark.asyncio
    async def test_config_with_plugins(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_config

        mock_adapter.sandbox_config.plugin_allowlist_mode = True
        mock_adapter.sandbox_config.allowed_plugins = ["p1", "p2", "p3"]

        result = await handle_openclaw_config({}, user_id="u1")
        body = _body(result)
        assert body["data"]["sandbox"]["plugin_allowlist_mode"] is True
        assert body["data"]["sandbox"]["allowed_plugins_count"] == 3

    @pytest.mark.asyncio
    async def test_config_with_blocked_overrides(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_config

        mock_adapter.capability_filter.blocked_override = {"cap1", "cap2"}
        mock_adapter.capability_filter.tenant_enabled = {"cap3"}

        result = await handle_openclaw_config({}, user_id="u1")
        body = _body(result)
        assert body["data"]["capabilities"]["blocked_override_count"] == 2
        assert body["data"]["capabilities"]["tenant_enabled_count"] == 1

    @pytest.mark.asyncio
    async def test_config_uses_default_user_id(self, mock_adapter):
        from aragora.server.handlers.gateway.openclaw import handle_openclaw_config

        result = await handle_openclaw_config({})
        assert result.status_code == 200


# =========================================================================
# _build_auth_context
# =========================================================================


class TestBuildAuthContext:
    """Tests for _build_auth_context helper."""

    def test_default_actor_type(self):
        from aragora.server.handlers.gateway.openclaw import _build_auth_context

        ctx = _build_auth_context("user-1", {})
        assert ctx.actor_id == "user-1"
        assert ctx.actor_type == "user"
        assert ctx.permissions == set()
        assert ctx.roles == []
        assert ctx.session_id is None

    def test_service_actor_type(self):
        from aragora.server.handlers.gateway.openclaw import _build_auth_context

        ctx = _build_auth_context("svc-1", {"actor_type": "service"})
        assert ctx.actor_type == "service"

    def test_agent_actor_type(self):
        from aragora.server.handlers.gateway.openclaw import _build_auth_context

        ctx = _build_auth_context("agent-1", {"actor_type": "agent"})
        assert ctx.actor_type == "agent"

    def test_invalid_actor_type_defaults_to_user(self):
        from aragora.server.handlers.gateway.openclaw import _build_auth_context

        ctx = _build_auth_context("user-1", {"actor_type": "hacker"})
        assert ctx.actor_type == "user"

    def test_permissions_ignored_from_data(self):
        """Permissions should NOT come from request data (security)."""
        from aragora.server.handlers.gateway.openclaw import _build_auth_context

        ctx = _build_auth_context(
            "user-1", {"permissions": ["admin"], "roles": ["superuser"]}
        )
        assert ctx.permissions == set()
        assert ctx.roles == []

    def test_session_id_from_data(self):
        from aragora.server.handlers.gateway.openclaw import _build_auth_context

        ctx = _build_auth_context("user-1", {"session_id": "sess-abc"})
        assert ctx.session_id == "sess-abc"


# =========================================================================
# _build_tenant_context
# =========================================================================


class TestBuildTenantContext:
    """Tests for _build_tenant_context helper."""

    def test_returns_none_when_no_tenant_id(self):
        from aragora.server.handlers.gateway.openclaw import _build_tenant_context

        assert _build_tenant_context({}) is None

    def test_returns_none_when_empty_tenant_id(self):
        from aragora.server.handlers.gateway.openclaw import _build_tenant_context

        assert _build_tenant_context({"tenant_id": ""}) is None

    def test_basic_tenant_context(self):
        from aragora.server.handlers.gateway.openclaw import _build_tenant_context

        ctx = _build_tenant_context({"tenant_id": "t-1"})
        assert ctx is not None
        assert ctx.tenant_id == "t-1"
        assert ctx.organization_id is None
        assert ctx.workspace_id is None

    def test_full_tenant_context(self):
        from aragora.server.handlers.gateway.openclaw import _build_tenant_context

        ctx = _build_tenant_context({
            "tenant_id": "t-1",
            "organization_id": "org-1",
            "workspace_id": "ws-1",
            "user_id": "u-1",
            "enabled_capabilities": ["cap1", "cap2"],
            "enabled_plugins": ["plug1"],
        })
        assert ctx.tenant_id == "t-1"
        assert ctx.organization_id == "org-1"
        assert ctx.workspace_id == "ws-1"
        assert ctx.user_id == "u-1"
        assert ctx.enabled_capabilities == {"cap1", "cap2"}
        assert ctx.enabled_plugins == {"plug1"}


# =========================================================================
# get_openclaw_handlers
# =========================================================================


class TestGetOpenclawHandlers:
    """Tests for handler registration helper."""

    def test_returns_all_handlers(self):
        from aragora.server.handlers.gateway.openclaw import get_openclaw_handlers

        handlers = get_openclaw_handlers()
        assert isinstance(handlers, dict)
        expected_keys = {
            "openclaw_execute",
            "openclaw_status",
            "openclaw_device_register",
            "openclaw_device_unregister",
            "openclaw_plugin_install",
            "openclaw_plugin_uninstall",
            "openclaw_config",
        }
        assert set(handlers.keys()) == expected_keys

    def test_handlers_are_callable(self):
        from aragora.server.handlers.gateway.openclaw import get_openclaw_handlers

        handlers = get_openclaw_handlers()
        for name, handler in handlers.items():
            assert callable(handler), f"{name} is not callable"

    def test_handler_count(self):
        from aragora.server.handlers.gateway.openclaw import get_openclaw_handlers

        handlers = get_openclaw_handlers()
        assert len(handlers) == 7


# =========================================================================
# _get_gateway_adapter (singleton creation)
# =========================================================================


class TestGetGatewayAdapter:
    """Tests for the singleton adapter factory."""

    def test_creates_adapter_with_defaults(self):
        import aragora.server.handlers.gateway.openclaw as mod

        mod._gateway_adapter = None

        with patch.dict("os.environ", {}, clear=True):
            with patch(f"{MODULE}.OpenClawGatewayAdapter") as MockAdapter, \
                 patch(f"{MODULE}.SandboxConfig") as MockSandbox, \
                 patch(f"{MODULE}.CapabilityFilter") as MockFilter, \
                 patch(f"{MODULE}.OpenClawProtocolTranslator") as MockTranslator:
                adapter = mod._get_gateway_adapter()
                MockAdapter.assert_called_once()
                call_kwargs = MockAdapter.call_args.kwargs
                assert call_kwargs["openclaw_endpoint"] == "http://localhost:8081"

    def test_creates_adapter_with_env_vars(self):
        import aragora.server.handlers.gateway.openclaw as mod

        mod._gateway_adapter = None

        env = {
            "OPENCLAW_ENDPOINT": "http://custom:9090",
            "OPENCLAW_MAX_MEMORY_MB": "1024",
            "OPENCLAW_MAX_EXECUTION_SECONDS": "600",
            "OPENCLAW_PLUGIN_ALLOWLIST": "p1,p2,p3",
        }
        with patch.dict("os.environ", env, clear=True):
            with patch(f"{MODULE}.OpenClawGatewayAdapter") as MockAdapter, \
                 patch(f"{MODULE}.SandboxConfig") as MockSandbox, \
                 patch(f"{MODULE}.CapabilityFilter"), \
                 patch(f"{MODULE}.OpenClawProtocolTranslator"):
                mod._get_gateway_adapter()
                sandbox_call = MockSandbox.call_args.kwargs
                assert sandbox_call["max_memory_mb"] == 1024
                assert sandbox_call["max_execution_seconds"] == 600
                assert sandbox_call["allowed_plugins"] == ["p1", "p2", "p3"]
                assert sandbox_call["plugin_allowlist_mode"] is True

                adapter_call = MockAdapter.call_args.kwargs
                assert adapter_call["openclaw_endpoint"] == "http://custom:9090"

    def test_memory_clamped_low(self):
        import aragora.server.handlers.gateway.openclaw as mod

        mod._gateway_adapter = None

        env = {"OPENCLAW_MAX_MEMORY_MB": "10"}
        with patch.dict("os.environ", env, clear=True):
            with patch(f"{MODULE}.OpenClawGatewayAdapter"), \
                 patch(f"{MODULE}.SandboxConfig") as MockSandbox, \
                 patch(f"{MODULE}.CapabilityFilter"), \
                 patch(f"{MODULE}.OpenClawProtocolTranslator"):
                mod._get_gateway_adapter()
                assert MockSandbox.call_args.kwargs["max_memory_mb"] == 64

    def test_memory_clamped_high(self):
        import aragora.server.handlers.gateway.openclaw as mod

        mod._gateway_adapter = None

        env = {"OPENCLAW_MAX_MEMORY_MB": "99999"}
        with patch.dict("os.environ", env, clear=True):
            with patch(f"{MODULE}.OpenClawGatewayAdapter"), \
                 patch(f"{MODULE}.SandboxConfig") as MockSandbox, \
                 patch(f"{MODULE}.CapabilityFilter"), \
                 patch(f"{MODULE}.OpenClawProtocolTranslator"):
                mod._get_gateway_adapter()
                assert MockSandbox.call_args.kwargs["max_memory_mb"] == 16384

    def test_execution_clamped_low(self):
        import aragora.server.handlers.gateway.openclaw as mod

        mod._gateway_adapter = None

        env = {"OPENCLAW_MAX_EXECUTION_SECONDS": "0"}
        with patch.dict("os.environ", env, clear=True):
            with patch(f"{MODULE}.OpenClawGatewayAdapter"), \
                 patch(f"{MODULE}.SandboxConfig") as MockSandbox, \
                 patch(f"{MODULE}.CapabilityFilter"), \
                 patch(f"{MODULE}.OpenClawProtocolTranslator"):
                mod._get_gateway_adapter()
                assert MockSandbox.call_args.kwargs["max_execution_seconds"] == 1

    def test_execution_clamped_high(self):
        import aragora.server.handlers.gateway.openclaw as mod

        mod._gateway_adapter = None

        env = {"OPENCLAW_MAX_EXECUTION_SECONDS": "99999"}
        with patch.dict("os.environ", env, clear=True):
            with patch(f"{MODULE}.OpenClawGatewayAdapter"), \
                 patch(f"{MODULE}.SandboxConfig") as MockSandbox, \
                 patch(f"{MODULE}.CapabilityFilter"), \
                 patch(f"{MODULE}.OpenClawProtocolTranslator"):
                mod._get_gateway_adapter()
                assert MockSandbox.call_args.kwargs["max_execution_seconds"] == 3600

    def test_empty_allowlist_disables_allowlist_mode(self):
        import aragora.server.handlers.gateway.openclaw as mod

        mod._gateway_adapter = None

        env = {"OPENCLAW_PLUGIN_ALLOWLIST": ""}
        with patch.dict("os.environ", env, clear=True):
            with patch(f"{MODULE}.OpenClawGatewayAdapter"), \
                 patch(f"{MODULE}.SandboxConfig") as MockSandbox, \
                 patch(f"{MODULE}.CapabilityFilter"), \
                 patch(f"{MODULE}.OpenClawProtocolTranslator"):
                mod._get_gateway_adapter()
                assert MockSandbox.call_args.kwargs["plugin_allowlist_mode"] is False
                assert MockSandbox.call_args.kwargs["allowed_plugins"] == []

    def test_singleton_returns_same_adapter(self):
        import aragora.server.handlers.gateway.openclaw as mod

        mod._gateway_adapter = None

        with patch(f"{MODULE}.OpenClawGatewayAdapter") as MockAdapter, \
             patch(f"{MODULE}.SandboxConfig"), \
             patch(f"{MODULE}.CapabilityFilter"), \
             patch(f"{MODULE}.OpenClawProtocolTranslator"):
            a1 = mod._get_gateway_adapter()
            a2 = mod._get_gateway_adapter()
            assert a1 is a2
            assert MockAdapter.call_count == 1

    def test_allowlist_strips_whitespace(self):
        import aragora.server.handlers.gateway.openclaw as mod

        mod._gateway_adapter = None

        env = {"OPENCLAW_PLUGIN_ALLOWLIST": " p1 , p2 , , p3 "}
        with patch.dict("os.environ", env, clear=True):
            with patch(f"{MODULE}.OpenClawGatewayAdapter"), \
                 patch(f"{MODULE}.SandboxConfig") as MockSandbox, \
                 patch(f"{MODULE}.CapabilityFilter"), \
                 patch(f"{MODULE}.OpenClawProtocolTranslator"):
                mod._get_gateway_adapter()
                assert MockSandbox.call_args.kwargs["allowed_plugins"] == ["p1", "p2", "p3"]


# =========================================================================
# Module __all__ exports
# =========================================================================


class TestModuleExports:
    """Verify the module exports expected symbols."""

    def test_all_exports(self):
        from aragora.server.handlers.gateway.openclaw import __all__

        expected = {
            "handle_openclaw_execute",
            "handle_openclaw_status",
            "handle_openclaw_device_register",
            "handle_openclaw_device_unregister",
            "handle_openclaw_plugin_install",
            "handle_openclaw_plugin_uninstall",
            "handle_openclaw_config",
            "get_openclaw_handlers",
        }
        assert set(__all__) == expected

    def test_valid_actor_types_constant(self):
        from aragora.server.handlers.gateway.openclaw import VALID_ACTOR_TYPES

        assert VALID_ACTOR_TYPES == {"user", "service", "agent"}
