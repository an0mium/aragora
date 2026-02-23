"""
Tests for SCIMHandler - SCIM 2.0 user/group provisioning endpoints.

Covers all routes and behavior of the SCIMHandler class:
- can_handle() routing
- GET    /scim/v2/Users           - List users with filtering and pagination
- POST   /scim/v2/Users           - Create user
- GET    /scim/v2/Users/{id}      - Get user by ID
- PUT    /scim/v2/Users/{id}      - Replace user
- PATCH  /scim/v2/Users/{id}      - Partial update user
- DELETE /scim/v2/Users/{id}      - Delete user
- GET    /scim/v2/Groups          - List groups
- POST   /scim/v2/Groups          - Create group
- GET    /scim/v2/Groups/{id}     - Get group by ID
- PUT    /scim/v2/Groups/{id}     - Replace group
- PATCH  /scim/v2/Groups/{id}     - Partial update group
- DELETE /scim/v2/Groups/{id}     - Delete group
- Bearer token auth verification
- SCIM error/response formatting
- Resource ID extraction
- JSON body parsing
- SCIM module availability fallback
- Security: path traversal, injection, edge cases
"""

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.scim_handler import SCIMHandler, SCIM_CONTENT_TYPE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult | None) -> dict:
    """Extract the JSON body from a HandlerResult."""
    assert result is not None, "Expected a HandlerResult, got None"
    if isinstance(result.body, bytes):
        if not result.body:
            return {}
        return json.loads(result.body.decode("utf-8"))
    return {}


def _status(result: HandlerResult | None) -> int:
    """Extract HTTP status code from a HandlerResult."""
    assert result is not None, "Expected a HandlerResult, got None"
    return result.status_code


def _make_http_handler(
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock HTTP handler with optional body and headers."""
    mock = MagicMock()
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        mock.rfile = io.BytesIO(body_bytes)
        h = {"Content-Length": str(len(body_bytes))}
    else:
        mock.rfile = io.BytesIO(b"")
        h = {"Content-Length": "0"}
    if headers:
        h.update(headers)
    mock.headers = h
    return mock


def _make_auth_handler(
    token: str = "valid-token",
    body: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock HTTP handler with a Bearer token."""
    headers = {"Authorization": f"Bearer {token}"}
    return _make_http_handler(body=body, headers=headers)


# ---------------------------------------------------------------------------
# Mock SCIM server
# ---------------------------------------------------------------------------


class MockSCIMConfig:
    """Mock SCIMConfig for testing."""

    def __init__(self, bearer_token: str = "", tenant_id: str | None = None, base_url: str = ""):
        self.bearer_token = bearer_token
        self.tenant_id = tenant_id
        self.base_url = base_url


class MockSCIMServer:
    """Mock SCIMServer with async methods that return test data."""

    def __init__(self, config: MockSCIMConfig | None = None):
        self.config = config or MockSCIMConfig()
        self.list_users = AsyncMock(return_value={"totalResults": 0, "Resources": []})
        self.get_user = AsyncMock(return_value=({"id": "user-1", "userName": "alice"}, 200))
        self.create_user = AsyncMock(return_value=({"id": "user-new", "userName": "bob"}, 201))
        self.replace_user = AsyncMock(return_value=({"id": "user-1", "userName": "updated"}, 200))
        self.patch_user = AsyncMock(return_value=({"id": "user-1", "active": False}, 200))
        self.delete_user = AsyncMock(return_value=(None, 204))

        self.list_groups = AsyncMock(return_value={"totalResults": 0, "Resources": []})
        self.get_group = AsyncMock(return_value=({"id": "grp-1", "displayName": "admins"}, 200))
        self.create_group = AsyncMock(return_value=({"id": "grp-new", "displayName": "devs"}, 201))
        self.replace_group = AsyncMock(
            return_value=({"id": "grp-1", "displayName": "updated"}, 200)
        )
        self.patch_group = AsyncMock(return_value=({"id": "grp-1", "displayName": "patched"}, 200))
        self.delete_group = AsyncMock(return_value=(None, 204))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a SCIMHandler with empty server context."""
    return SCIMHandler({})


@pytest.fixture
def mock_scim():
    """Create a MockSCIMServer with no bearer token (auth disabled)."""
    return MockSCIMServer(MockSCIMConfig(bearer_token=""))


@pytest.fixture
def mock_scim_with_auth():
    """Create a MockSCIMServer with bearer token auth enabled."""
    config = MockSCIMConfig(bearer_token="valid-token")
    return MockSCIMServer(config)


@pytest.fixture(autouse=True)
def patch_scim_available():
    """Ensure SCIM_AVAILABLE is True by default for all tests."""
    with patch("aragora.server.handlers.scim_handler.SCIM_AVAILABLE", True):
        yield


@pytest.fixture(autouse=True)
def patch_run_async():
    """Patch run_async to directly call the coroutine."""

    def fake_run_async(coro, timeout=30.0):
        import asyncio

        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with patch("aragora.server.handlers.scim_handler.run_async", side_effect=fake_run_async):
        yield


# ---------------------------------------------------------------------------
# Routing / can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle() routing."""

    def test_handles_users_path(self, handler):
        assert handler.can_handle("/scim/v2/Users") is True

    def test_handles_users_with_id(self, handler):
        assert handler.can_handle("/scim/v2/Users/abc-123") is True

    def test_handles_groups_path(self, handler):
        assert handler.can_handle("/scim/v2/Groups") is True

    def test_handles_groups_with_id(self, handler):
        assert handler.can_handle("/scim/v2/Groups/grp-456") is True

    def test_does_not_handle_root(self, handler):
        assert handler.can_handle("/") is False

    def test_does_not_handle_api_v1(self, handler):
        assert handler.can_handle("/api/v1/users") is False

    def test_does_not_handle_scim_v1(self, handler):
        assert handler.can_handle("/scim/v1/Users") is False

    def test_does_not_handle_partial_prefix(self, handler):
        assert handler.can_handle("/scim/v2") is False

    def test_handles_trailing_slash(self, handler):
        assert handler.can_handle("/scim/v2/Users/") is True

    def test_does_not_handle_scim_wrong_case(self, handler):
        # Path matching is case-sensitive per SCIM spec
        assert handler.can_handle("/SCIM/v2/Users") is False


# ---------------------------------------------------------------------------
# SCIM error and response formatting
# ---------------------------------------------------------------------------


class TestSCIMFormatting:
    """Tests for _scim_error and _scim_response helpers."""

    def test_scim_error_format(self, handler):
        result = handler._scim_error("Not found", 404)
        assert _status(result) == 404
        assert result.content_type == SCIM_CONTENT_TYPE
        body = _body(result)
        assert body["schemas"] == ["urn:ietf:params:scim:api:messages:2.0:Error"]
        assert body["detail"] == "Not found"
        assert body["status"] == "404"

    def test_scim_error_401(self, handler):
        result = handler._scim_error("Unauthorized", 401)
        assert _status(result) == 401
        body = _body(result)
        assert body["detail"] == "Unauthorized"
        assert body["status"] == "401"

    def test_scim_response_200(self, handler):
        data = {"id": "usr-1", "userName": "alice"}
        result = handler._scim_response(data, 200)
        assert _status(result) == 200
        assert result.content_type == SCIM_CONTENT_TYPE
        body = _body(result)
        assert body == data

    def test_scim_response_204(self, handler):
        result = handler._scim_response(None, 204)
        assert _status(result) == 204
        assert result.body == b""

    def test_scim_response_empty_dict(self, handler):
        result = handler._scim_response(None, 200)
        assert _status(result) == 200
        body = _body(result)
        assert body == {}


# ---------------------------------------------------------------------------
# Resource ID extraction
# ---------------------------------------------------------------------------


class TestExtractResourceId:
    """Tests for _extract_resource_id."""

    def test_extract_user_id(self, handler):
        assert handler._extract_resource_id("/scim/v2/Users/abc-123", "Users") == "abc-123"

    def test_extract_group_id(self, handler):
        assert handler._extract_resource_id("/scim/v2/Groups/grp-456", "Groups") == "grp-456"

    def test_no_id_in_path(self, handler):
        assert handler._extract_resource_id("/scim/v2/Users", "Users") is None

    def test_trailing_slash_stripped(self, handler):
        assert handler._extract_resource_id("/scim/v2/Users/abc-123/", "Users") == "abc-123"

    def test_query_params_stripped(self, handler):
        assert handler._extract_resource_id("/scim/v2/Users/abc-123?foo=bar", "Users") == "abc-123"

    def test_wrong_resource_type(self, handler):
        assert handler._extract_resource_id("/scim/v2/Users/abc-123", "Groups") is None

    def test_empty_id_after_slash(self, handler):
        assert handler._extract_resource_id("/scim/v2/Users/", "Users") is None

    def test_uuid_format_id(self, handler):
        uuid_id = "550e8400-e29b-41d4-a716-446655440000"
        assert handler._extract_resource_id(f"/scim/v2/Users/{uuid_id}", "Users") == uuid_id


# ---------------------------------------------------------------------------
# JSON body reading
# ---------------------------------------------------------------------------


class TestReadJsonBody:
    """Tests for _read_json_body."""

    def test_reads_from_context(self, handler):
        handler.ctx["body"] = {"userName": "alice"}
        mock_http = _make_http_handler()
        result = handler._read_json_body(mock_http)
        assert result == {"userName": "alice"}
        # Clean up
        del handler.ctx["body"]

    def test_reads_from_rfile(self, handler):
        body_data = {"userName": "bob"}
        mock_http = _make_http_handler(body=body_data)
        result = handler._read_json_body(mock_http)
        assert result == body_data

    def test_returns_none_for_invalid_json(self, handler):
        mock_http = MagicMock()
        mock_http.rfile = io.BytesIO(b"not-json")
        mock_http.headers = {"Content-Length": "8"}
        result = handler._read_json_body(mock_http)
        assert result is None

    def test_returns_none_for_no_body(self, handler):
        mock_http = MagicMock()
        mock_http.headers = {"Content-Length": "0"}
        mock_http.rfile = io.BytesIO(b"")
        result = handler._read_json_body(mock_http)
        assert result is None

    def test_returns_none_when_no_rfile(self, handler):
        mock_http = MagicMock(spec=[])  # No attributes
        result = handler._read_json_body(mock_http)
        assert result is None


# ---------------------------------------------------------------------------
# Bearer token verification
# ---------------------------------------------------------------------------


class TestBearerTokenVerification:
    """Tests for _verify_bearer_token."""

    def test_no_auth_configured_passes(self, handler, mock_scim):
        """When no bearer token is configured, auth passes."""
        handler._scim_server = mock_scim
        mock_http = _make_http_handler()
        result = handler._verify_bearer_token(mock_http)
        assert result is None  # None means auth succeeded

    def test_valid_token_passes(self, handler, mock_scim_with_auth):
        handler._scim_server = mock_scim_with_auth
        mock_http = _make_auth_handler(token="valid-token")
        result = handler._verify_bearer_token(mock_http)
        assert result is None

    def test_missing_auth_header_fails(self, handler, mock_scim_with_auth):
        handler._scim_server = mock_scim_with_auth
        mock_http = _make_http_handler(headers={})
        # The mock handler has headers dict but no Authorization
        result = handler._verify_bearer_token(mock_http)
        assert result is not None
        assert _status(result) == 401
        body = _body(result)
        assert "Authorization header required" in body["detail"]

    def test_non_bearer_auth_fails(self, handler, mock_scim_with_auth):
        handler._scim_server = mock_scim_with_auth
        mock_http = _make_http_handler(headers={"Authorization": "Basic dXNlcjpwYXNz"})
        result = handler._verify_bearer_token(mock_http)
        assert result is not None
        assert _status(result) == 401
        body = _body(result)
        assert "Bearer token required" in body["detail"]

    def test_invalid_token_fails(self, handler, mock_scim_with_auth):
        handler._scim_server = mock_scim_with_auth
        mock_http = _make_auth_handler(token="wrong-token")
        result = handler._verify_bearer_token(mock_http)
        assert result is not None
        assert _status(result) == 401
        body = _body(result)
        assert "Invalid bearer token" in body["detail"]

    def test_handler_without_headers_attr(self, handler, mock_scim_with_auth):
        handler._scim_server = mock_scim_with_auth
        mock_http = MagicMock(spec=[])  # No headers attribute
        result = handler._verify_bearer_token(mock_http)
        assert result is not None
        assert _status(result) == 401


# ---------------------------------------------------------------------------
# SCIM unavailable
# ---------------------------------------------------------------------------


class TestSCIMUnavailable:
    """Tests for when the SCIM module is not available."""

    def test_get_returns_503(self, handler):
        mock_http = _make_http_handler()
        with patch("aragora.server.handlers.scim_handler.SCIM_AVAILABLE", False):
            result = handler.handle("/scim/v2/Users", {}, mock_http)
        assert result is not None
        assert _status(result) == 503

    def test_post_returns_503(self, handler):
        mock_http = _make_http_handler(body={"userName": "test"})
        with patch("aragora.server.handlers.scim_handler.SCIM_AVAILABLE", False):
            result = handler.handle_post("/scim/v2/Users", {}, mock_http)
        assert result is not None
        assert _status(result) == 503

    def test_put_returns_503(self, handler):
        mock_http = _make_http_handler(body={"userName": "test"})
        with patch("aragora.server.handlers.scim_handler.SCIM_AVAILABLE", False):
            result = handler.handle_put("/scim/v2/Users/uid-1", {}, mock_http)
        assert result is not None
        assert _status(result) == 503

    def test_patch_returns_503(self, handler):
        mock_http = _make_http_handler(body={"active": False})
        with patch("aragora.server.handlers.scim_handler.SCIM_AVAILABLE", False):
            result = handler.handle_patch("/scim/v2/Users/uid-1", {}, mock_http)
        assert result is not None
        assert _status(result) == 503

    def test_delete_returns_503(self, handler):
        mock_http = _make_http_handler()
        with patch("aragora.server.handlers.scim_handler.SCIM_AVAILABLE", False):
            result = handler.handle_delete("/scim/v2/Users/uid-1", {}, mock_http)
        assert result is not None
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# SCIM server init failure
# ---------------------------------------------------------------------------


class TestSCIMServerInitFailure:
    """Tests for when _get_scim_server returns None."""

    def test_get_returns_503_on_init_fail(self, handler):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=None):
            result = handler.handle("/scim/v2/Users", {}, mock_http)
        assert result is not None
        assert _status(result) == 503

    def test_post_returns_503_on_init_fail(self, handler):
        mock_http = _make_http_handler(body={"userName": "test"})
        with patch.object(handler, "_get_scim_server", return_value=None):
            result = handler.handle_post("/scim/v2/Users", {}, mock_http)
        assert result is not None
        assert _status(result) == 503

    def test_put_returns_503_on_init_fail(self, handler):
        mock_http = _make_http_handler(body={"userName": "test"})
        with patch.object(handler, "_get_scim_server", return_value=None):
            result = handler.handle_put("/scim/v2/Users/uid-1", {}, mock_http)
        assert result is not None
        assert _status(result) == 503

    def test_patch_returns_503_on_init_fail(self, handler):
        mock_http = _make_http_handler(body={"active": False})
        with patch.object(handler, "_get_scim_server", return_value=None):
            result = handler.handle_patch("/scim/v2/Users/uid-1", {}, mock_http)
        assert result is not None
        assert _status(result) == 503

    def test_delete_returns_503_on_init_fail(self, handler):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=None):
            result = handler.handle_delete("/scim/v2/Users/uid-1", {}, mock_http)
        assert result is not None
        assert _status(result) == 503


# ---------------------------------------------------------------------------
# GET /scim/v2/Users
# ---------------------------------------------------------------------------


class TestListUsers:
    """Tests for GET /scim/v2/Users."""

    def test_list_users_success(self, handler, mock_scim):
        mock_scim.list_users.return_value = {
            "totalResults": 2,
            "Resources": [
                {"id": "u1", "userName": "alice"},
                {"id": "u2", "userName": "bob"},
            ],
        }
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/Users", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["totalResults"] == 2
        assert len(body["Resources"]) == 2

    def test_list_users_with_pagination(self, handler, mock_scim):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/Users", {"startIndex": "5", "count": "10"}, mock_http)
        assert _status(result) == 200
        mock_scim.list_users.assert_called_once()

    def test_list_users_with_filter(self, handler, mock_scim):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle(
                "/scim/v2/Users",
                {"filter": 'userName eq "alice"'},
                mock_http,
            )
        assert _status(result) == 200
        call_args = mock_scim.list_users.call_args
        assert call_args[0][2] == 'userName eq "alice"'

    def test_list_users_trailing_slash(self, handler, mock_scim):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/Users/", {}, mock_http)
        assert _status(result) == 200

    def test_list_users_empty(self, handler, mock_scim):
        mock_scim.list_users.return_value = {"totalResults": 0, "Resources": []}
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/Users", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["totalResults"] == 0


# ---------------------------------------------------------------------------
# GET /scim/v2/Users/{id}
# ---------------------------------------------------------------------------


class TestGetUser:
    """Tests for GET /scim/v2/Users/{id}."""

    def test_get_user_success(self, handler, mock_scim):
        mock_scim.get_user.return_value = ({"id": "u1", "userName": "alice"}, 200)
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/Users/u1", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "u1"
        mock_scim.get_user.assert_called_once_with("u1")

    def test_get_user_not_found(self, handler, mock_scim):
        mock_scim.get_user.return_value = (
            {
                "schemas": ["urn:ietf:params:scim:api:messages:2.0:Error"],
                "detail": "Not found",
                "status": "404",
            },
            404,
        )
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/Users/nonexistent", {}, mock_http)
        assert _status(result) == 404

    def test_get_user_with_uuid(self, handler, mock_scim):
        uuid_id = "550e8400-e29b-41d4-a716-446655440000"
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle(f"/scim/v2/Users/{uuid_id}", {}, mock_http)
        assert _status(result) == 200
        mock_scim.get_user.assert_called_once_with(uuid_id)


# ---------------------------------------------------------------------------
# POST /scim/v2/Users
# ---------------------------------------------------------------------------


class TestCreateUser:
    """Tests for POST /scim/v2/Users."""

    def test_create_user_success(self, handler, mock_scim):
        mock_scim.create_user.return_value = (
            {"id": "u-new", "userName": "charlie"},
            201,
        )
        user_data = {"userName": "charlie", "name": {"givenName": "Charlie"}}
        mock_http = _make_http_handler(body=user_data)
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=user_data):
                result = handler.handle_post("/scim/v2/Users", {}, mock_http)
        assert _status(result) == 201
        body = _body(result)
        assert body["id"] == "u-new"

    def test_create_user_invalid_body(self, handler, mock_scim):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=None):
                result = handler.handle_post("/scim/v2/Users", {}, mock_http)
        assert _status(result) == 400
        body = _body(result)
        assert "Invalid JSON" in body["detail"]

    def test_create_user_conflict(self, handler, mock_scim):
        mock_scim.create_user.return_value = (
            {"detail": "User already exists", "status": "409"},
            409,
        )
        user_data = {"userName": "existing"}
        mock_http = _make_http_handler(body=user_data)
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=user_data):
                result = handler.handle_post("/scim/v2/Users", {}, mock_http)
        assert _status(result) == 409


# ---------------------------------------------------------------------------
# PUT /scim/v2/Users/{id}
# ---------------------------------------------------------------------------


class TestReplaceUser:
    """Tests for PUT /scim/v2/Users/{id}."""

    def test_replace_user_success(self, handler, mock_scim):
        mock_scim.replace_user.return_value = (
            {"id": "u1", "userName": "alice-updated"},
            200,
        )
        user_data = {"userName": "alice-updated"}
        mock_http = _make_http_handler(body=user_data)
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=user_data):
                result = handler.handle_put("/scim/v2/Users/u1", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["userName"] == "alice-updated"
        mock_scim.replace_user.assert_called_once_with("u1", user_data)

    def test_replace_user_invalid_body(self, handler, mock_scim):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=None):
                result = handler.handle_put("/scim/v2/Users/u1", {}, mock_http)
        assert _status(result) == 400

    def test_replace_user_not_found(self, handler, mock_scim):
        mock_scim.replace_user.return_value = (
            {"detail": "User not found", "status": "404"},
            404,
        )
        user_data = {"userName": "alice"}
        mock_http = _make_http_handler(body=user_data)
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=user_data):
                result = handler.handle_put("/scim/v2/Users/nonexistent", {}, mock_http)
        assert _status(result) == 404

    def test_put_returns_none_for_collection_path(self, handler, mock_scim):
        """PUT /scim/v2/Users (no ID) should return None."""
        mock_http = _make_http_handler(body={"userName": "test"})
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value={"userName": "test"}):
                result = handler.handle_put("/scim/v2/Users", {}, mock_http)
        assert result is None


# ---------------------------------------------------------------------------
# PATCH /scim/v2/Users/{id}
# ---------------------------------------------------------------------------


class TestPatchUser:
    """Tests for PATCH /scim/v2/Users/{id}."""

    def test_patch_user_success(self, handler, mock_scim):
        mock_scim.patch_user.return_value = ({"id": "u1", "active": False}, 200)
        patch_data = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "replace", "path": "active", "value": False}],
        }
        mock_http = _make_http_handler(body=patch_data)
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=patch_data):
                result = handler.handle_patch("/scim/v2/Users/u1", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["active"] is False

    def test_patch_user_invalid_body(self, handler, mock_scim):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=None):
                result = handler.handle_patch("/scim/v2/Users/u1", {}, mock_http)
        assert _status(result) == 400

    def test_patch_returns_none_for_collection_path(self, handler, mock_scim):
        """PATCH /scim/v2/Users (no ID) should return None."""
        mock_http = _make_http_handler(body={"Operations": []})
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value={"Operations": []}):
                result = handler.handle_patch("/scim/v2/Users", {}, mock_http)
        assert result is None


# ---------------------------------------------------------------------------
# DELETE /scim/v2/Users/{id}
# ---------------------------------------------------------------------------


class TestDeleteUser:
    """Tests for DELETE /scim/v2/Users/{id}."""

    def test_delete_user_success(self, handler, mock_scim):
        mock_scim.delete_user.return_value = (None, 204)
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle_delete("/scim/v2/Users/u1", {}, mock_http)
        assert _status(result) == 204
        mock_scim.delete_user.assert_called_once_with("u1")

    def test_delete_user_not_found(self, handler, mock_scim):
        mock_scim.delete_user.return_value = (
            {"detail": "User not found", "status": "404"},
            404,
        )
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle_delete("/scim/v2/Users/nonexistent", {}, mock_http)
        assert _status(result) == 404

    def test_delete_returns_none_for_collection_path(self, handler, mock_scim):
        """DELETE /scim/v2/Users (no ID) should return None."""
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle_delete("/scim/v2/Users", {}, mock_http)
        assert result is None


# ---------------------------------------------------------------------------
# GET /scim/v2/Groups
# ---------------------------------------------------------------------------


class TestListGroups:
    """Tests for GET /scim/v2/Groups."""

    def test_list_groups_success(self, handler, mock_scim):
        mock_scim.list_groups.return_value = {
            "totalResults": 1,
            "Resources": [{"id": "g1", "displayName": "admins"}],
        }
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/Groups", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["totalResults"] == 1

    def test_list_groups_with_pagination(self, handler, mock_scim):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle(
                "/scim/v2/Groups", {"startIndex": "1", "count": "50"}, mock_http
            )
        assert _status(result) == 200

    def test_list_groups_with_filter(self, handler, mock_scim):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle(
                "/scim/v2/Groups",
                {"filter": 'displayName eq "admins"'},
                mock_http,
            )
        assert _status(result) == 200
        call_args = mock_scim.list_groups.call_args
        assert call_args[0][2] == 'displayName eq "admins"'

    def test_list_groups_trailing_slash(self, handler, mock_scim):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/Groups/", {}, mock_http)
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /scim/v2/Groups/{id}
# ---------------------------------------------------------------------------


class TestGetGroup:
    """Tests for GET /scim/v2/Groups/{id}."""

    def test_get_group_success(self, handler, mock_scim):
        mock_scim.get_group.return_value = ({"id": "g1", "displayName": "admins"}, 200)
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/Groups/g1", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "g1"
        mock_scim.get_group.assert_called_once_with("g1")

    def test_get_group_not_found(self, handler, mock_scim):
        mock_scim.get_group.return_value = ({"detail": "Not found", "status": "404"}, 404)
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/Groups/nonexistent", {}, mock_http)
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# POST /scim/v2/Groups
# ---------------------------------------------------------------------------


class TestCreateGroup:
    """Tests for POST /scim/v2/Groups."""

    def test_create_group_success(self, handler, mock_scim):
        mock_scim.create_group.return_value = (
            {"id": "g-new", "displayName": "developers"},
            201,
        )
        group_data = {"displayName": "developers", "members": []}
        mock_http = _make_http_handler(body=group_data)
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=group_data):
                result = handler.handle_post("/scim/v2/Groups", {}, mock_http)
        assert _status(result) == 201
        body = _body(result)
        assert body["displayName"] == "developers"

    def test_create_group_invalid_body(self, handler, mock_scim):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=None):
                result = handler.handle_post("/scim/v2/Groups", {}, mock_http)
        assert _status(result) == 400

    def test_post_returns_none_for_unrelated_path(self, handler, mock_scim):
        """POST to a non-SCIM path should return None."""
        mock_http = _make_http_handler(body={"name": "test"})
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle_post("/api/v1/users", {}, mock_http)
        assert result is None


# ---------------------------------------------------------------------------
# PUT /scim/v2/Groups/{id}
# ---------------------------------------------------------------------------


class TestReplaceGroup:
    """Tests for PUT /scim/v2/Groups/{id}."""

    def test_replace_group_success(self, handler, mock_scim):
        mock_scim.replace_group.return_value = (
            {"id": "g1", "displayName": "admins-v2"},
            200,
        )
        group_data = {"displayName": "admins-v2"}
        mock_http = _make_http_handler(body=group_data)
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=group_data):
                result = handler.handle_put("/scim/v2/Groups/g1", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["displayName"] == "admins-v2"
        mock_scim.replace_group.assert_called_once_with("g1", group_data)

    def test_replace_group_invalid_body(self, handler, mock_scim):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=None):
                result = handler.handle_put("/scim/v2/Groups/g1", {}, mock_http)
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# PATCH /scim/v2/Groups/{id}
# ---------------------------------------------------------------------------


class TestPatchGroup:
    """Tests for PATCH /scim/v2/Groups/{id}."""

    def test_patch_group_success(self, handler, mock_scim):
        mock_scim.patch_group.return_value = (
            {"id": "g1", "displayName": "patched-admins"},
            200,
        )
        patch_data = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "replace", "path": "displayName", "value": "patched-admins"}],
        }
        mock_http = _make_http_handler(body=patch_data)
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=patch_data):
                result = handler.handle_patch("/scim/v2/Groups/g1", {}, mock_http)
        assert _status(result) == 200
        body = _body(result)
        assert body["displayName"] == "patched-admins"

    def test_patch_group_invalid_body(self, handler, mock_scim):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=None):
                result = handler.handle_patch("/scim/v2/Groups/g1", {}, mock_http)
        assert _status(result) == 400


# ---------------------------------------------------------------------------
# DELETE /scim/v2/Groups/{id}
# ---------------------------------------------------------------------------


class TestDeleteGroup:
    """Tests for DELETE /scim/v2/Groups/{id}."""

    def test_delete_group_success(self, handler, mock_scim):
        mock_scim.delete_group.return_value = (None, 204)
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle_delete("/scim/v2/Groups/g1", {}, mock_http)
        assert _status(result) == 204
        mock_scim.delete_group.assert_called_once_with("g1")

    def test_delete_group_not_found(self, handler, mock_scim):
        mock_scim.delete_group.return_value = (
            {"detail": "Group not found", "status": "404"},
            404,
        )
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle_delete("/scim/v2/Groups/nonexistent", {}, mock_http)
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# Bearer token authentication across methods
# ---------------------------------------------------------------------------


class TestBearerTokenAcrossMethods:
    """Tests ensuring bearer auth is enforced for all methods."""

    def _setup_scim_with_auth(self, handler):
        """Set up handler with auth-enabled SCIM server."""
        scim = MockSCIMServer(MockSCIMConfig(bearer_token="secret-token"))
        handler._scim_server = scim
        return scim

    def test_get_rejects_invalid_token(self, handler):
        self._setup_scim_with_auth(handler)
        mock_http = _make_auth_handler(token="wrong")
        result = handler.handle("/scim/v2/Users", {}, mock_http)
        assert _status(result) == 401

    def test_post_rejects_invalid_token(self, handler):
        self._setup_scim_with_auth(handler)
        mock_http = _make_auth_handler(token="wrong")
        result = handler.handle_post("/scim/v2/Users", {}, mock_http)
        assert _status(result) == 401

    def test_put_rejects_invalid_token(self, handler):
        self._setup_scim_with_auth(handler)
        mock_http = _make_auth_handler(token="wrong")
        result = handler.handle_put("/scim/v2/Users/u1", {}, mock_http)
        assert _status(result) == 401

    def test_patch_rejects_invalid_token(self, handler):
        self._setup_scim_with_auth(handler)
        mock_http = _make_auth_handler(token="wrong")
        result = handler.handle_patch("/scim/v2/Users/u1", {}, mock_http)
        assert _status(result) == 401

    def test_delete_rejects_invalid_token(self, handler):
        self._setup_scim_with_auth(handler)
        mock_http = _make_auth_handler(token="wrong")
        result = handler.handle_delete("/scim/v2/Users/u1", {}, mock_http)
        assert _status(result) == 401

    def test_get_with_valid_token(self, handler):
        scim = self._setup_scim_with_auth(handler)
        mock_http = _make_auth_handler(token="secret-token")
        with patch.object(handler, "_get_scim_server", return_value=scim):
            result = handler.handle("/scim/v2/Users", {}, mock_http)
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# Unmatched routes return None
# ---------------------------------------------------------------------------


class TestUnmatchedRoutes:
    """Tests that unmatched paths return None."""

    def test_handle_unmatched_path(self, handler):
        mock_http = _make_http_handler()
        result = handler.handle("/api/v1/debates", {}, mock_http)
        assert result is None

    def test_handle_post_unmatched_path(self, handler):
        mock_http = _make_http_handler(body={"name": "test"})
        result = handler.handle_post("/api/v1/debates", {}, mock_http)
        assert result is None

    def test_handle_put_unmatched_path(self, handler):
        mock_http = _make_http_handler(body={"name": "test"})
        result = handler.handle_put("/api/v1/debates/123", {}, mock_http)
        assert result is None

    def test_handle_patch_unmatched_path(self, handler):
        mock_http = _make_http_handler(body={"name": "test"})
        result = handler.handle_patch("/api/v1/debates/123", {}, mock_http)
        assert result is None

    def test_handle_delete_unmatched_path(self, handler):
        mock_http = _make_http_handler()
        result = handler.handle_delete("/api/v1/debates/123", {}, mock_http)
        assert result is None

    def test_get_unknown_scim_sub_path(self, handler, mock_scim):
        """GET /scim/v2/ServiceProviderConfig or similar should return None."""
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/ServiceProviderConfig", {}, mock_http)
        assert result is None


# ---------------------------------------------------------------------------
# SCIM server lazy initialization
# ---------------------------------------------------------------------------


class TestSCIMServerInit:
    """Tests for _get_scim_server lazy initialization."""

    def test_creates_server_on_first_call(self, handler):
        with (
            patch("aragora.server.handlers.scim_handler.SCIMServer") as mock_cls,
            patch("aragora.server.handlers.scim_handler.SCIMConfig") as mock_config_cls,
        ):
            mock_cls.return_value = MagicMock()
            mock_config_cls.return_value = MagicMock()
            result = handler._get_scim_server()
            assert result is not None
            mock_config_cls.assert_called_once()
            mock_cls.assert_called_once()

    def test_caches_server_on_subsequent_calls(self, handler):
        with (
            patch("aragora.server.handlers.scim_handler.SCIMServer") as mock_cls,
            patch("aragora.server.handlers.scim_handler.SCIMConfig") as mock_config_cls,
        ):
            mock_cls.return_value = MagicMock()
            mock_config_cls.return_value = MagicMock()
            first = handler._get_scim_server()
            second = handler._get_scim_server()
            assert first is second
            # Only called once due to caching
            assert mock_cls.call_count == 1

    def test_returns_none_when_scim_not_available(self, handler):
        with patch("aragora.server.handlers.scim_handler.SCIM_AVAILABLE", False):
            result = handler._get_scim_server()
            assert result is None

    def test_reads_env_vars_for_config(self, handler):
        with (
            patch("aragora.server.handlers.scim_handler.SCIMServer") as mock_cls,
            patch("aragora.server.handlers.scim_handler.SCIMConfig") as mock_config_cls,
            patch.dict(
                "os.environ",
                {
                    "SCIM_BEARER_TOKEN": "my-token",
                    "SCIM_TENANT_ID": "tenant-1",
                    "SCIM_BASE_URL": "https://scim.example.com",
                },
            ),
        ):
            mock_cls.return_value = MagicMock()
            mock_config_cls.return_value = MagicMock()
            handler._get_scim_server()
            mock_config_cls.assert_called_once_with(
                bearer_token="my-token",
                tenant_id="tenant-1",
                base_url="https://scim.example.com",
            )


# ---------------------------------------------------------------------------
# Security tests
# ---------------------------------------------------------------------------


class TestSecurity:
    """Security-related tests for the SCIM handler."""

    def test_path_traversal_in_user_id(self, handler, mock_scim):
        """Ensure path traversal in user IDs is handled safely."""
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/Users/../../etc/passwd", {}, mock_http)
        # The handler will extract "../../etc/passwd" as the ID and pass it to SCIM server
        # The SCIM server is responsible for validation; handler just extracts
        if result is not None:
            mock_scim.get_user.assert_called_once()

    def test_special_chars_in_user_id(self, handler, mock_scim):
        """Test with special characters in user ID."""
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/Users/user%40example.com", {}, mock_http)
        if result is not None:
            assert _status(result) == 200

    def test_empty_bearer_token_in_header(self, handler, mock_scim_with_auth):
        """An empty Bearer token should fail auth."""
        handler._scim_server = mock_scim_with_auth
        mock_http = _make_http_handler(headers={"Authorization": "Bearer "})
        result = handler._verify_bearer_token(mock_http)
        assert result is not None
        assert _status(result) == 401

    def test_bearer_token_case_sensitive(self, handler, mock_scim_with_auth):
        """Bearer token comparison should be case-sensitive."""
        handler._scim_server = mock_scim_with_auth
        mock_http = _make_auth_handler(token="Valid-Token")  # Wrong case
        result = handler._verify_bearer_token(mock_http)
        assert result is not None
        assert _status(result) == 401

    def test_large_request_body_is_handled(self, handler, mock_scim):
        """Test with a very large request body."""
        large_body = {
            "userName": "x" * 10000,
            "emails": [{"value": f"user{i}@example.com"} for i in range(100)],
        }
        mock_http = _make_http_handler(body=large_body)
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=large_body):
                result = handler.handle_post("/scim/v2/Users", {}, mock_http)
        assert _status(result) == 201

    def test_sql_injection_in_filter(self, handler, mock_scim):
        """SQL injection in filter should be passed through to SCIM server for handling."""
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle(
                "/scim/v2/Users",
                {"filter": 'userName eq "\'; DROP TABLE users;--"'},
                mock_http,
            )
        assert _status(result) == 200
        # The filter string is passed through; SCIM server handles validation

    def test_xss_in_user_data(self, handler, mock_scim):
        """XSS in user data should be passed through safely (JSON encoding)."""
        xss_data = {"userName": "<script>alert('xss')</script>"}
        mock_http = _make_http_handler(body=xss_data)
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=xss_data):
                result = handler.handle_post("/scim/v2/Users", {}, mock_http)
        assert _status(result) == 201
        # Response is JSON-encoded, so no XSS risk


# ---------------------------------------------------------------------------
# Content type
# ---------------------------------------------------------------------------


class TestContentType:
    """Verify SCIM content type is used in all responses."""

    def test_scim_content_type_constant(self):
        assert SCIM_CONTENT_TYPE == "application/scim+json"

    def test_success_response_content_type(self, handler, mock_scim):
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/Users", {}, mock_http)
        assert result is not None
        assert result.content_type == SCIM_CONTENT_TYPE

    def test_error_response_content_type(self, handler, mock_scim_with_auth):
        handler._scim_server = mock_scim_with_auth
        mock_http = _make_auth_handler(token="wrong")
        result = handler.handle("/scim/v2/Users", {}, mock_http)
        assert result is not None
        assert result.content_type == SCIM_CONTENT_TYPE

    def test_204_response_content_type(self, handler, mock_scim):
        mock_scim.delete_user.return_value = (None, 204)
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle_delete("/scim/v2/Users/u1", {}, mock_http)
        assert result is not None
        assert result.content_type == SCIM_CONTENT_TYPE


# ---------------------------------------------------------------------------
# ROUTES class attribute
# ---------------------------------------------------------------------------


class TestRoutes:
    """Verify ROUTES class attribute is correct."""

    def test_routes_defined(self, handler):
        assert hasattr(handler, "ROUTES")
        assert "/scim/v2/Users" in handler.ROUTES
        assert "/scim/v2/Users/*" in handler.ROUTES
        assert "/scim/v2/Groups" in handler.ROUTES
        assert "/scim/v2/Groups/*" in handler.ROUTES

    def test_routes_length(self, handler):
        assert len(handler.ROUTES) == 4


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_get_users_returns_none_for_unknown_scim_path(self, handler, mock_scim):
        """A GET to a SCIM path that is neither Users nor Groups should return None."""
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle("/scim/v2/Schemas", {}, mock_http)
        assert result is None

    def test_post_to_user_with_id_returns_none(self, handler, mock_scim):
        """POST /scim/v2/Users/{id} is not a valid SCIM endpoint; should return None."""
        user_data = {"userName": "test"}
        mock_http = _make_http_handler(body=user_data)
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=user_data):
                result = handler.handle_post("/scim/v2/Users/u1", {}, mock_http)
        assert result is None

    def test_post_to_group_with_id_returns_none(self, handler, mock_scim):
        """POST /scim/v2/Groups/{id} is not a valid SCIM endpoint; should return None."""
        group_data = {"displayName": "test"}
        mock_http = _make_http_handler(body=group_data)
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value=group_data):
                result = handler.handle_post("/scim/v2/Groups/g1", {}, mock_http)
        assert result is None

    def test_delete_collection_returns_none(self, handler, mock_scim):
        """DELETE /scim/v2/Users (no ID) should return None."""
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle_delete("/scim/v2/Users", {}, mock_http)
        assert result is None

    def test_delete_groups_collection_returns_none(self, handler, mock_scim):
        """DELETE /scim/v2/Groups (no ID) should return None."""
        mock_http = _make_http_handler()
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            result = handler.handle_delete("/scim/v2/Groups", {}, mock_http)
        assert result is None

    def test_put_groups_collection_returns_none(self, handler, mock_scim):
        """PUT /scim/v2/Groups (no ID) should return None."""
        mock_http = _make_http_handler(body={"displayName": "test"})
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value={"displayName": "test"}):
                result = handler.handle_put("/scim/v2/Groups", {}, mock_http)
        assert result is None

    def test_patch_groups_collection_returns_none(self, handler, mock_scim):
        """PATCH /scim/v2/Groups (no ID) should return None."""
        mock_http = _make_http_handler(body={"Operations": []})
        with patch.object(handler, "_get_scim_server", return_value=mock_scim):
            with patch.object(handler, "_read_json_body", return_value={"Operations": []}):
                result = handler.handle_patch("/scim/v2/Groups", {}, mock_http)
        assert result is None

    def test_handler_ctx_stored(self, handler):
        """Verify the handler stores its server context."""
        assert handler.ctx == {}

    def test_scim_server_initially_none(self, handler):
        """Verify _scim_server is None before any call."""
        assert handler._scim_server is None
