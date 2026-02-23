"""Tests for openclaw_gateway.py backwards-compat shim and the full OpenClaw handler.

Covers:
- Module re-exports (all names in __all__ are importable)
- OpenClawGatewayHandler.can_handle() routing
- Path normalisation (_normalize_path)
- GET routes: sessions list, session by ID, action by ID, credentials list,
  health, metrics, audit, policy rules, approvals, stats
- POST routes: create session, execute action, cancel action, end session,
  add policy rule, approve action, deny action, store credential, rotate credential
- DELETE routes: close session, remove policy rule, delete credential
- Error paths: 404, 400, 403
- Circuit breaker helpers
- Handler factory
- Unmatched routes return None
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.openclaw_gateway import (
    CREDENTIAL_ROTATION_WINDOW_SECONDS,
    MAX_ACTION_INPUT_SIZE,
    MAX_ACTION_TYPE_LENGTH,
    MAX_CREDENTIAL_NAME_LENGTH,
    MAX_CREDENTIAL_ROTATIONS_PER_HOUR,
    MAX_CREDENTIAL_SECRET_LENGTH,
    MAX_SESSION_CONFIG_DEPTH,
    MAX_SESSION_CONFIG_KEYS,
    MAX_SESSION_CONFIG_SIZE,
    MIN_CREDENTIAL_SECRET_LENGTH,
    Action,
    ActionStatus,
    AuditEntry,
    Credential,
    CredentialRotationRateLimiter,
    CredentialType,
    OpenClawGatewayHandler,
    OpenClawGatewayStore,
    Session,
    SessionStatus,
    _get_credential_rotation_limiter,
    _get_store,
    get_openclaw_circuit_breaker,
    get_openclaw_circuit_breaker_status,
    get_openclaw_gateway_handler,
    sanitize_action_parameters,
    validate_action_input,
    validate_action_type,
    validate_credential_name,
    validate_credential_secret,
    validate_metadata,
    validate_session_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract the body dict from a HandlerResult."""
    if hasattr(result, "to_dict"):
        d = result.to_dict()
        return d.get("body", d)
    if isinstance(result, dict):
        return result.get("body", result)
    try:
        body, status, _ = result
        return body if isinstance(body, dict) else {}
    except (TypeError, ValueError):
        return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", result.get("status", 200))
    try:
        _, status, _ = result
        return status
    except (TypeError, ValueError):
        return 200


class MockHTTPHandler:
    """Mock HTTP handler used by BaseHandler.read_json_body."""

    def __init__(self, body: dict | None = None):
        self.rfile = MagicMock()
        self._body = body
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {
                "Content-Length": "2",
                "Content-Type": "application/json",
            }
        self.client_address = ("127.0.0.1", 54321)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store():
    """Create a fresh in-memory store for each test."""
    return OpenClawGatewayStore()


@pytest.fixture()
def handler(store):
    """Create an OpenClawGatewayHandler backed by the test store."""
    h = OpenClawGatewayHandler({})
    return h


@pytest.fixture(autouse=True)
def _patch_store(store):
    """Patch _get_store in both the store module and all mixin modules to return our test store."""
    with (
        patch("aragora.server.handlers.openclaw.store._get_store", return_value=store),
        patch(
            "aragora.server.handlers.openclaw.orchestrator._get_store",
            return_value=store,
        ),
        patch(
            "aragora.server.handlers.openclaw.credentials._get_store",
            return_value=store,
        ),
        patch(
            "aragora.server.handlers.openclaw.policies._get_store",
            return_value=store,
        ),
    ):
        yield


@pytest.fixture(autouse=True)
def _patch_mixin_base(monkeypatch):
    """Patch get_current_user on OpenClawMixinBase so it doesn't shadow the
    conftest-patched BaseHandler.get_current_user.

    OpenClawMixinBase defines get_current_user as a stub that raises
    NotImplementedError.  Because it sits before BaseHandler in the MRO,
    it masks the conftest's monkeypatch on BaseHandler.  We delete it
    from OpenClawMixinBase so that BaseHandler's (patched) version is used.
    """
    from aragora.server.handlers.openclaw._base import OpenClawMixinBase

    # Remove the stub methods so MRO falls through to BaseHandler
    for name in ("get_current_user", "_get_user_id", "_get_tenant_id"):
        if name in OpenClawMixinBase.__dict__:
            monkeypatch.delattr(OpenClawMixinBase, name)


@pytest.fixture(autouse=True)
def _patch_rate_limit():
    """Disable rate limiters for handler tests."""
    with (
        patch(
            "aragora.server.handlers.openclaw.orchestrator.rate_limit",
            lambda **kw: (lambda fn: fn),
        ),
        patch(
            "aragora.server.handlers.openclaw.orchestrator.auth_rate_limit",
            lambda **kw: (lambda fn: fn),
        ),
        patch(
            "aragora.server.handlers.openclaw.credentials.rate_limit",
            lambda **kw: (lambda fn: fn),
        ),
        patch(
            "aragora.server.handlers.openclaw.credentials.auth_rate_limit",
            lambda **kw: (lambda fn: fn),
        ),
        patch(
            "aragora.server.handlers.openclaw.policies.rate_limit",
            lambda **kw: (lambda fn: fn),
        ),
    ):
        yield


def _mock_handler(body: dict | None = None) -> MockHTTPHandler:
    return MockHTTPHandler(body=body)


# ===========================================================================
# Module Re-exports
# ===========================================================================


class TestModuleReexports:
    """Verify backwards-compat shim re-exports every public symbol."""

    def test_handler_class(self):
        assert OpenClawGatewayHandler is not None

    def test_factory_function(self):
        assert callable(get_openclaw_gateway_handler)

    def test_circuit_breaker_helpers(self):
        assert callable(get_openclaw_circuit_breaker)
        assert callable(get_openclaw_circuit_breaker_status)

    def test_model_classes(self):
        assert Session is not None
        assert Action is not None
        assert Credential is not None
        assert AuditEntry is not None

    def test_enums(self):
        assert SessionStatus.ACTIVE.value == "active"
        assert ActionStatus.PENDING.value == "pending"
        assert CredentialType.API_KEY.value == "api_key"

    def test_store_class(self):
        assert OpenClawGatewayStore is not None

    def test_validation_constants(self):
        assert MAX_CREDENTIAL_NAME_LENGTH == 128
        assert MAX_CREDENTIAL_SECRET_LENGTH == 8192
        assert MIN_CREDENTIAL_SECRET_LENGTH == 8
        assert MAX_SESSION_CONFIG_KEYS == 50
        assert MAX_SESSION_CONFIG_DEPTH == 5
        assert MAX_SESSION_CONFIG_SIZE == 8192
        assert MAX_ACTION_TYPE_LENGTH == 64
        assert MAX_ACTION_INPUT_SIZE == 65536
        assert MAX_CREDENTIAL_ROTATIONS_PER_HOUR == 10
        assert CREDENTIAL_ROTATION_WINDOW_SECONDS == 3600

    def test_validation_functions(self):
        assert callable(validate_credential_name)
        assert callable(validate_credential_secret)
        assert callable(validate_session_config)
        assert callable(validate_action_type)
        assert callable(validate_action_input)
        assert callable(validate_metadata)
        assert callable(sanitize_action_parameters)

    def test_rate_limiter_class(self):
        assert callable(CredentialRotationRateLimiter)
        assert callable(_get_credential_rotation_limiter)


# ===========================================================================
# can_handle and path normalisation
# ===========================================================================


class TestCanHandle:
    """Routing: can_handle returns True for OpenClaw prefixes."""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/gateway/openclaw/sessions",
            "/api/v1/gateway/openclaw/actions",
            "/api/v1/openclaw/credentials",
            "/api/openclaw/health",
        ],
    )
    def test_can_handle_true(self, handler, path):
        assert handler.can_handle(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/debates",
            "/api/gateway/other",
            "/api/v2/openclaw/sessions",
            "/health",
        ],
    )
    def test_can_handle_false(self, handler, path):
        assert handler.can_handle(path) is False


class TestNormalizePath:
    """Path normalisation collapses all variants to /api/gateway/openclaw/..."""

    def test_v1_gateway(self, handler):
        assert handler._normalize_path("/api/v1/gateway/openclaw/sessions") == "/api/gateway/openclaw/sessions"

    def test_v1_shorthand(self, handler):
        assert handler._normalize_path("/api/v1/openclaw/health") == "/api/gateway/openclaw/health"

    def test_no_version(self, handler):
        assert handler._normalize_path("/api/openclaw/metrics") == "/api/gateway/openclaw/metrics"

    def test_already_normalised(self, handler):
        assert handler._normalize_path("/api/gateway/openclaw/sessions") == "/api/gateway/openclaw/sessions"


# ===========================================================================
# Circuit breaker helpers
# ===========================================================================


class TestCircuitBreaker:
    def test_get_circuit_breaker_returns_instance(self):
        cb = get_openclaw_circuit_breaker()
        assert cb is not None
        assert cb.name == "openclaw_gateway_handler"

    def test_get_circuit_breaker_status_is_dict(self):
        status = get_openclaw_circuit_breaker_status()
        assert isinstance(status, dict)


# ===========================================================================
# Handler factory
# ===========================================================================


class TestHandlerFactory:
    def test_get_handler(self):
        h = get_openclaw_gateway_handler({})
        assert isinstance(h, OpenClawGatewayHandler)


# ===========================================================================
# GET routes
# ===========================================================================


class TestGetSessions:
    def test_list_sessions_empty(self, handler, store):
        result = handler.handle("/api/v1/openclaw/sessions", {}, _mock_handler())
        assert _status(result) == 200
        body = _body(result)
        assert body["sessions"] == []
        assert body["total"] == 0

    def test_list_sessions_with_data(self, handler, store):
        store.create_session(user_id="test-user-001", tenant_id="test-org-001")
        result = handler.handle("/api/v1/openclaw/sessions", {}, _mock_handler())
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        assert len(body["sessions"]) == 1

    def test_get_session_by_id(self, handler, store):
        session = store.create_session(user_id="test-user-001", tenant_id="test-org-001")
        result = handler.handle(
            f"/api/v1/openclaw/sessions/{session.id}", {}, _mock_handler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == session.id

    def test_get_session_not_found(self, handler, store):
        result = handler.handle(
            "/api/v1/openclaw/sessions/nonexistent", {}, _mock_handler()
        )
        assert _status(result) == 404


class TestGetActions:
    def test_get_action_by_id(self, handler, store):
        session = store.create_session(user_id="test-user-001", tenant_id="test-org-001")
        action = store.create_action(
            session_id=session.id,
            action_type="test.action",
            input_data={"key": "value"},
        )
        result = handler.handle(
            f"/api/v1/openclaw/actions/{action.id}", {}, _mock_handler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == action.id

    def test_get_action_not_found(self, handler, store):
        result = handler.handle(
            "/api/v1/openclaw/actions/nonexistent", {}, _mock_handler()
        )
        assert _status(result) == 404


class TestGetCredentials:
    def test_list_credentials_empty(self, handler, store):
        result = handler.handle("/api/v1/openclaw/credentials", {}, _mock_handler())
        assert _status(result) == 200
        body = _body(result)
        assert body["credentials"] == []
        assert body["total"] == 0


class TestGetHealth:
    def test_health_healthy(self, handler, store):
        result = handler.handle("/api/v1/openclaw/health", {}, _mock_handler())
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "healthy"
        assert body["healthy"] is True


class TestGetMetrics:
    def test_metrics_response(self, handler, store):
        result = handler.handle("/api/v1/openclaw/metrics", {}, _mock_handler())
        assert _status(result) == 200
        body = _body(result)
        assert "sessions" in body
        assert "actions" in body
        assert "credentials" in body


class TestGetAudit:
    def test_audit_empty(self, handler, store):
        result = handler.handle("/api/v1/openclaw/audit", {}, _mock_handler())
        assert _status(result) == 200
        body = _body(result)
        assert body["entries"] == []
        assert body["total"] == 0

    def test_audit_with_entries(self, handler, store):
        store.add_audit_entry(
            action="test", actor_id="user1", resource_type="session"
        )
        result = handler.handle("/api/v1/openclaw/audit", {}, _mock_handler())
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1


class TestGetPolicyRules:
    def test_get_policy_rules(self, handler, store):
        result = handler.handle("/api/v1/openclaw/policy/rules", {}, _mock_handler())
        assert _status(result) == 200
        body = _body(result)
        assert body["rules"] == []


class TestGetApprovals:
    def test_list_approvals_empty(self, handler, store):
        result = handler.handle("/api/v1/openclaw/approvals", {}, _mock_handler())
        assert _status(result) == 200
        body = _body(result)
        assert body["approvals"] == []
        assert body["total"] == 0


class TestGetStats:
    def test_stats_response(self, handler, store):
        result = handler.handle("/api/v1/openclaw/stats", {}, _mock_handler())
        assert _status(result) == 200
        body = _body(result)
        assert "active_sessions" in body
        assert "timestamp" in body


class TestGetUnmatchedRoute:
    def test_unmatched_get_returns_none(self, handler):
        result = handler.handle("/api/v1/openclaw/unknown", {}, _mock_handler())
        assert result is None


# ===========================================================================
# POST routes
# ===========================================================================


class TestPostCreateSession:
    def test_create_session(self, handler, store):
        http = _mock_handler({"config": {"timeout": 30}})
        result = handler.handle_post("/api/v1/openclaw/sessions", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["status"] == "active"
        assert body["user_id"] == "test-user-001"

    def test_create_session_invalid_config_depth(self, handler, store):
        deep = {"a": {"b": {"c": {"d": {"e": {"f": "too deep"}}}}}}
        http = _mock_handler({"config": deep})
        result = handler.handle_post("/api/v1/openclaw/sessions", {}, http)
        assert _status(result) == 400


class TestPostExecuteAction:
    def test_execute_action_success(self, handler, store):
        session = store.create_session(user_id="test-user-001", tenant_id="test-org-001")
        http = _mock_handler({
            "session_id": session.id,
            "action_type": "run.test",
            "input": {"cmd": "echo hello"},
        })
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 202
        body = _body(result)
        assert body["session_id"] == session.id

    def test_execute_action_missing_session_id(self, handler, store):
        http = _mock_handler({"action_type": "run.test"})
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 400

    def test_execute_action_missing_action_type(self, handler, store):
        session = store.create_session(user_id="test-user-001", tenant_id="test-org-001")
        http = _mock_handler({"session_id": session.id})
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 400

    def test_execute_action_session_not_found(self, handler, store):
        http = _mock_handler({
            "session_id": "nonexistent",
            "action_type": "run.test",
        })
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 404

    def test_execute_action_session_not_active(self, handler, store):
        session = store.create_session(user_id="test-user-001", tenant_id="test-org-001")
        store.update_session_status(session.id, SessionStatus.CLOSED)
        http = _mock_handler({
            "session_id": session.id,
            "action_type": "run.test",
        })
        result = handler.handle_post("/api/v1/openclaw/actions", {}, http)
        assert _status(result) == 400


class TestPostCancelAction:
    def test_cancel_action_success(self, handler, store):
        session = store.create_session(user_id="test-user-001", tenant_id="test-org-001")
        action = store.create_action(
            session_id=session.id, action_type="run.test", input_data={}
        )
        store.update_action(action.id, status=ActionStatus.RUNNING)
        result = handler.handle_post(
            f"/api/v1/openclaw/actions/{action.id}/cancel", {}, _mock_handler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["cancelled"] is True

    def test_cancel_action_not_found(self, handler, store):
        result = handler.handle_post(
            "/api/v1/openclaw/actions/nonexistent/cancel", {}, _mock_handler()
        )
        assert _status(result) == 404

    def test_cancel_completed_action(self, handler, store):
        session = store.create_session(user_id="test-user-001", tenant_id="test-org-001")
        action = store.create_action(
            session_id=session.id, action_type="run.test", input_data={}
        )
        store.update_action(action.id, status=ActionStatus.COMPLETED)
        result = handler.handle_post(
            f"/api/v1/openclaw/actions/{action.id}/cancel", {}, _mock_handler()
        )
        assert _status(result) == 400


class TestPostEndSession:
    def test_end_session(self, handler, store):
        session = store.create_session(user_id="test-user-001", tenant_id="test-org-001")
        result = handler.handle_post(
            f"/api/v1/openclaw/sessions/{session.id}/end", {}, _mock_handler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True

    def test_end_session_not_found(self, handler, store):
        result = handler.handle_post(
            "/api/v1/openclaw/sessions/nonexistent/end", {}, _mock_handler()
        )
        assert _status(result) == 404


class TestPostPolicyRule:
    def test_add_policy_rule(self, handler, store):
        http = _mock_handler({
            "name": "block_dangerous",
            "action_types": ["shell.exec"],
            "decision": "deny",
        })
        result = handler.handle_post("/api/v1/openclaw/policy/rules", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["name"] == "block_dangerous"

    def test_add_policy_rule_missing_name(self, handler, store):
        http = _mock_handler({"decision": "deny"})
        result = handler.handle_post("/api/v1/openclaw/policy/rules", {}, http)
        assert _status(result) == 400


class TestPostApprovalActions:
    def test_approve_action(self, handler, store):
        http = _mock_handler({"reason": "looks good"})
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/abc123/approve", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["approval_id"] == "abc123"

    def test_deny_action(self, handler, store):
        http = _mock_handler({"reason": "too risky"})
        result = handler.handle_post(
            "/api/v1/openclaw/approvals/abc123/deny", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True


class TestPostStoreCredential:
    def test_store_credential(self, handler, store):
        http = _mock_handler({
            "name": "MyApiKey",
            "type": "api_key",
            "secret": "sk-test-very-long-secret-key-1234",
        })
        result = handler.handle_post("/api/v1/openclaw/credentials", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["name"] == "MyApiKey"
        assert body["credential_type"] == "api_key"

    def test_store_credential_missing_name(self, handler, store):
        http = _mock_handler({"type": "api_key", "secret": "sk-1234567890"})
        result = handler.handle_post("/api/v1/openclaw/credentials", {}, http)
        assert _status(result) == 400

    def test_store_credential_missing_type(self, handler, store):
        http = _mock_handler({"name": "MyKey", "secret": "sk-1234567890"})
        result = handler.handle_post("/api/v1/openclaw/credentials", {}, http)
        assert _status(result) == 400

    def test_store_credential_invalid_type(self, handler, store):
        http = _mock_handler({
            "name": "MyKey",
            "type": "invalid_type",
            "secret": "sk-1234567890",
        })
        result = handler.handle_post("/api/v1/openclaw/credentials", {}, http)
        assert _status(result) == 400

    def test_store_credential_secret_too_short(self, handler, store):
        http = _mock_handler({
            "name": "MyKey",
            "type": "api_key",
            "secret": "abc",
        })
        result = handler.handle_post("/api/v1/openclaw/credentials", {}, http)
        assert _status(result) == 400


class TestPostRotateCredential:
    def test_rotate_credential(self, handler, store):
        cred = store.store_credential(
            name="OldKey",
            credential_type=CredentialType.API_KEY,
            secret_value="original-secret-value-12345678",
            user_id="test-user-001",
        )
        http = _mock_handler({"secret": "new-secret-value-12345678"})
        result = handler.handle_post(
            f"/api/v1/openclaw/credentials/{cred.id}/rotate", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["rotated"] is True

    def test_rotate_credential_not_found(self, handler, store):
        http = _mock_handler({"secret": "new-secret-value-12345678"})
        result = handler.handle_post(
            "/api/v1/openclaw/credentials/nonexistent/rotate", {}, http
        )
        assert _status(result) == 404

    def test_rotate_credential_invalid_secret(self, handler, store):
        cred = store.store_credential(
            name="OldKey",
            credential_type=CredentialType.API_KEY,
            secret_value="original-secret-value-12345678",
            user_id="test-user-001",
        )
        http = _mock_handler({"secret": "ab"})
        result = handler.handle_post(
            f"/api/v1/openclaw/credentials/{cred.id}/rotate", {}, http
        )
        assert _status(result) == 400


class TestPostUnmatchedRoute:
    def test_unmatched_post_returns_none(self, handler):
        result = handler.handle_post("/api/v1/openclaw/unknown", {}, _mock_handler())
        assert result is None


# ===========================================================================
# DELETE routes
# ===========================================================================


class TestDeleteSession:
    def test_close_session(self, handler, store):
        session = store.create_session(user_id="test-user-001", tenant_id="test-org-001")
        result = handler.handle_delete(
            f"/api/v1/openclaw/sessions/{session.id}", {}, _mock_handler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["closed"] is True

    def test_close_session_not_found(self, handler, store):
        result = handler.handle_delete(
            "/api/v1/openclaw/sessions/nonexistent", {}, _mock_handler()
        )
        assert _status(result) == 404


class TestDeletePolicyRule:
    def test_remove_policy_rule(self, handler, store):
        result = handler.handle_delete(
            "/api/v1/openclaw/policy/rules/block_dangerous", {}, _mock_handler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "block_dangerous"


class TestDeleteCredential:
    def test_delete_credential(self, handler, store):
        cred = store.store_credential(
            name="TempKey",
            credential_type=CredentialType.API_KEY,
            secret_value="secret-value-123456789",
            user_id="test-user-001",
        )
        result = handler.handle_delete(
            f"/api/v1/openclaw/credentials/{cred.id}", {}, _mock_handler()
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["deleted"] is True

    def test_delete_credential_not_found(self, handler, store):
        result = handler.handle_delete(
            "/api/v1/openclaw/credentials/nonexistent", {}, _mock_handler()
        )
        assert _status(result) == 404


class TestDeleteUnmatchedRoute:
    def test_unmatched_delete_returns_none(self, handler):
        result = handler.handle_delete("/api/v1/openclaw/unknown", {}, _mock_handler())
        assert result is None


# ===========================================================================
# Validation functions (tested through the shim imports)
# ===========================================================================


class TestValidateCredentialName:
    def test_valid_name(self):
        ok, err = validate_credential_name("MyApiKey")
        assert ok is True
        assert err is None

    def test_empty_name(self):
        ok, err = validate_credential_name("")
        assert ok is False
        assert "required" in err

    def test_name_too_long(self):
        ok, err = validate_credential_name("a" * 200)
        assert ok is False

    def test_name_with_special_chars(self):
        ok, err = validate_credential_name("bad name!")
        assert ok is False


class TestValidateCredentialSecret:
    def test_valid_secret(self):
        ok, err = validate_credential_secret("a-long-enough-secret")
        assert ok is True

    def test_empty_secret(self):
        ok, err = validate_credential_secret("")
        assert ok is False

    def test_secret_too_short(self):
        ok, err = validate_credential_secret("abc")
        assert ok is False

    def test_null_byte_in_secret(self):
        ok, err = validate_credential_secret("secret\x00value-long-enough")
        assert ok is False


class TestValidateSessionConfig:
    def test_none_config(self):
        ok, err = validate_session_config(None)
        assert ok is True

    def test_valid_config(self):
        ok, err = validate_session_config({"timeout": 30})
        assert ok is True

    def test_non_dict_config(self):
        ok, err = validate_session_config("not a dict")
        assert ok is False


class TestValidateActionType:
    def test_valid_type(self):
        ok, err = validate_action_type("run.test")
        assert ok is True

    def test_empty_type(self):
        ok, err = validate_action_type("")
        assert ok is False

    def test_type_starts_with_number(self):
        ok, err = validate_action_type("1badtype")
        assert ok is False


class TestSanitizeActionParameters:
    def test_escapes_shell_metacharacters(self):
        result = sanitize_action_parameters({"cmd": "echo; rm -rf /"})
        # Semicolons are escaped to \; not removed
        assert result["cmd"] == r"echo\; rm -rf /"

    def test_empty_params(self):
        assert sanitize_action_parameters(None) == {}
        assert sanitize_action_parameters({}) == {}

    def test_non_dict_params(self):
        assert sanitize_action_parameters("not a dict") == {}

    def test_nested_sanitization(self):
        result = sanitize_action_parameters({"nested": {"cmd": "echo | cat"}})
        # The pipe is escaped to \|, not removed
        assert result["nested"]["cmd"] == r"echo \| cat"


# ===========================================================================
# Credential Rotation Rate Limiter
# ===========================================================================


class TestCredentialRotationRateLimiter:
    def test_allows_within_limit(self):
        limiter = CredentialRotationRateLimiter(max_rotations=3, window_seconds=3600)
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True
        assert limiter.is_allowed("user1") is True

    def test_blocks_over_limit(self):
        limiter = CredentialRotationRateLimiter(max_rotations=2, window_seconds=3600)
        limiter.is_allowed("user1")
        limiter.is_allowed("user1")
        assert limiter.is_allowed("user1") is False

    def test_get_remaining(self):
        limiter = CredentialRotationRateLimiter(max_rotations=5, window_seconds=3600)
        limiter.is_allowed("user1")
        assert limiter.get_remaining("user1") == 4

    def test_get_retry_after_no_limit(self):
        limiter = CredentialRotationRateLimiter(max_rotations=5, window_seconds=3600)
        assert limiter.get_retry_after("user1") == 0


# ===========================================================================
# Legacy gateway paths
# ===========================================================================


class TestLegacyPaths:
    def test_legacy_sessions_path(self, handler, store):
        result = handler.handle("/api/gateway/openclaw/sessions", {}, _mock_handler())
        assert _status(result) == 200

    def test_legacy_health_path(self, handler, store):
        result = handler.handle("/api/gateway/openclaw/health", {}, _mock_handler())
        assert _status(result) == 200

    def test_legacy_metrics_path(self, handler, store):
        result = handler.handle("/api/gateway/openclaw/metrics", {}, _mock_handler())
        assert _status(result) == 200

    def test_legacy_audit_path(self, handler, store):
        result = handler.handle("/api/gateway/openclaw/audit", {}, _mock_handler())
        assert _status(result) == 200
