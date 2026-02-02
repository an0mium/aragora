"""
Tests for SCIMHandler - SCIM 2.0 User/Group Provisioning.

Covers:
- SCIM 2.0 user provisioning (RFC 7643/7644)
- Group management
- Bearer token authentication
- Error responses per SCIM spec
- Route matching (can_handle)
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.scim_handler import SCIMHandler, SCIM_CONTENT_TYPE


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def server_context():
    """Create a mock server context."""
    return {"config": {"debug": True}}


@pytest.fixture
def scim_handler(server_context):
    """Create a SCIMHandler instance."""
    return SCIMHandler(server_context)


@pytest.fixture
def mock_handler():
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.headers = {
        "Authorization": "Bearer test-scim-token",
        "Content-Type": "application/scim+json",
    }
    return handler


@pytest.fixture
def mock_handler_no_auth():
    """Create a mock HTTP request handler without auth."""
    handler = MagicMock()
    handler.headers = {"Content-Type": "application/scim+json"}
    return handler


@pytest.fixture
def mock_scim_server():
    """Create a mock SCIM server."""
    server = MagicMock()
    server.config = MagicMock()
    server.config.bearer_token = "test-scim-token"

    # Mock user operations
    server.list_users = AsyncMock(
        return_value={
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
            "totalResults": 2,
            "Resources": [
                {"id": "user-1", "userName": "alice@example.com"},
                {"id": "user-2", "userName": "bob@example.com"},
            ],
        }
    )
    server.get_user = AsyncMock(
        return_value=(
            {"id": "user-1", "userName": "alice@example.com"},
            200,
        )
    )
    server.create_user = AsyncMock(
        return_value=(
            {"id": "user-new", "userName": "new@example.com"},
            201,
        )
    )
    server.replace_user = AsyncMock(
        return_value=(
            {"id": "user-1", "userName": "updated@example.com"},
            200,
        )
    )
    server.patch_user = AsyncMock(
        return_value=(
            {"id": "user-1", "userName": "patched@example.com"},
            200,
        )
    )
    server.delete_user = AsyncMock(return_value=(None, 204))

    # Mock group operations
    server.list_groups = AsyncMock(
        return_value={
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
            "totalResults": 1,
            "Resources": [{"id": "group-1", "displayName": "Admins"}],
        }
    )
    server.get_group = AsyncMock(
        return_value=(
            {"id": "group-1", "displayName": "Admins"},
            200,
        )
    )
    server.create_group = AsyncMock(
        return_value=(
            {"id": "group-new", "displayName": "New Group"},
            201,
        )
    )
    server.replace_group = AsyncMock(
        return_value=(
            {"id": "group-1", "displayName": "Updated Group"},
            200,
        )
    )
    server.patch_group = AsyncMock(
        return_value=(
            {"id": "group-1", "displayName": "Patched Group"},
            200,
        )
    )
    server.delete_group = AsyncMock(return_value=(None, 204))

    return server


# -----------------------------------------------------------------------------
# Route Matching Tests (can_handle)
# -----------------------------------------------------------------------------


class TestSCIMHandlerRouteMatching:
    """Tests for SCIMHandler.can_handle() method."""

    def test_can_handle_users_list(self, scim_handler):
        """Handler matches /scim/v2/Users."""
        assert scim_handler.can_handle("/scim/v2/Users") is True

    def test_can_handle_users_with_id(self, scim_handler):
        """Handler matches /scim/v2/Users/{id}."""
        assert scim_handler.can_handle("/scim/v2/Users/user-123") is True

    def test_can_handle_groups_list(self, scim_handler):
        """Handler matches /scim/v2/Groups."""
        assert scim_handler.can_handle("/scim/v2/Groups") is True

    def test_can_handle_groups_with_id(self, scim_handler):
        """Handler matches /scim/v2/Groups/{id}."""
        assert scim_handler.can_handle("/scim/v2/Groups/group-456") is True

    def test_can_handle_rejects_non_scim_path(self, scim_handler):
        """Handler rejects non-SCIM paths."""
        assert scim_handler.can_handle("/api/users") is False
        assert scim_handler.can_handle("/api/v2/users") is False

    def test_can_handle_rejects_partial_match(self, scim_handler):
        """Handler rejects paths that only partially match."""
        assert scim_handler.can_handle("/scim") is False
        assert scim_handler.can_handle("/scim/v1/Users") is False


# -----------------------------------------------------------------------------
# Bearer Token Authentication Tests
# -----------------------------------------------------------------------------


class TestSCIMHandlerAuthentication:
    """Tests for SCIMHandler bearer token authentication."""

    def test_verify_bearer_token_success(self, scim_handler, mock_handler, mock_scim_server):
        """Valid bearer token passes verification."""
        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            result = scim_handler._verify_bearer_token(mock_handler)
            assert result is None  # None means success

    def test_verify_bearer_token_missing_header(
        self, scim_handler, mock_handler_no_auth, mock_scim_server
    ):
        """Missing Authorization header fails verification."""
        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            result = scim_handler._verify_bearer_token(mock_handler_no_auth)
            assert result is not None
            body, status, _ = result
            assert status == 401
            assert "Authorization header required" in body

    def test_verify_bearer_token_invalid_format(self, scim_handler, mock_scim_server):
        """Invalid Authorization format fails verification."""
        handler = MagicMock()
        handler.headers = {"Authorization": "Basic dXNlcjpwYXNz"}

        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            result = scim_handler._verify_bearer_token(handler)
            assert result is not None
            body, status, _ = result
            assert status == 401
            assert "Bearer token required" in body

    def test_verify_bearer_token_wrong_token(self, scim_handler, mock_scim_server):
        """Wrong bearer token fails verification."""
        handler = MagicMock()
        handler.headers = {"Authorization": "Bearer wrong-token"}

        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            result = scim_handler._verify_bearer_token(handler)
            assert result is not None
            body, status, _ = result
            assert status == 401
            assert "Invalid bearer token" in body

    def test_verify_bearer_token_no_config(self, scim_handler, mock_handler):
        """No token configured allows any request."""
        mock_server = MagicMock()
        mock_server.config = MagicMock()
        mock_server.config.bearer_token = ""  # No token configured

        with patch.object(scim_handler, "_get_scim_server", return_value=mock_server):
            result = scim_handler._verify_bearer_token(mock_handler)
            assert result is None  # No auth configured = all allowed


# -----------------------------------------------------------------------------
# User CRUD Tests
# -----------------------------------------------------------------------------


class TestSCIMHandlerUserOperations:
    """Tests for SCIM user CRUD operations."""

    def test_list_users(self, scim_handler, mock_handler, mock_scim_server):
        """GET /scim/v2/Users returns user list."""
        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                result = scim_handler.handle("/scim/v2/Users", {}, mock_handler)

                assert result is not None
                body, status, content_type = result
                assert status == 200
                assert content_type == SCIM_CONTENT_TYPE

                data = json.loads(body)
                assert data["totalResults"] == 2
                assert len(data["Resources"]) == 2

    def test_list_users_with_pagination(self, scim_handler, mock_handler, mock_scim_server):
        """GET /scim/v2/Users supports pagination params."""
        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                query_params = {"startIndex": "1", "count": "10"}
                scim_handler.handle("/scim/v2/Users", query_params, mock_handler)

                mock_scim_server.list_users.assert_called()

    def test_list_users_with_filter(self, scim_handler, mock_handler, mock_scim_server):
        """GET /scim/v2/Users supports filter param."""
        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                query_params = {"filter": 'userName eq "alice@example.com"'}
                scim_handler.handle("/scim/v2/Users", query_params, mock_handler)

                mock_scim_server.list_users.assert_called()

    def test_get_user_by_id(self, scim_handler, mock_handler, mock_scim_server):
        """GET /scim/v2/Users/{id} returns specific user."""
        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                result = scim_handler.handle("/scim/v2/Users/user-1", {}, mock_handler)

                assert result is not None
                body, status, content_type = result
                assert status == 200
                assert content_type == SCIM_CONTENT_TYPE

                data = json.loads(body)
                assert data["id"] == "user-1"

    def test_create_user(self, scim_handler, mock_handler, mock_scim_server):
        """POST /scim/v2/Users creates a new user."""
        user_data = {"userName": "new@example.com", "emails": [{"value": "new@example.com"}]}
        scim_handler.ctx = {"body": user_data}

        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                result = scim_handler.handle_post("/scim/v2/Users", {}, mock_handler)

                assert result is not None
                body, status, _ = result
                assert status == 201

                data = json.loads(body)
                assert data["id"] == "user-new"

    def test_replace_user(self, scim_handler, mock_handler, mock_scim_server):
        """PUT /scim/v2/Users/{id} replaces a user."""
        user_data = {"userName": "updated@example.com"}
        scim_handler.ctx = {"body": user_data}

        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                result = scim_handler.handle_put("/scim/v2/Users/user-1", {}, mock_handler)

                assert result is not None
                body, status, _ = result
                assert status == 200

                data = json.loads(body)
                assert data["userName"] == "updated@example.com"

    def test_patch_user(self, scim_handler, mock_handler, mock_scim_server):
        """PATCH /scim/v2/Users/{id} partially updates a user."""
        patch_data = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "replace", "path": "userName", "value": "patched@example.com"}],
        }
        scim_handler.ctx = {"body": patch_data}

        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                result = scim_handler.handle_patch("/scim/v2/Users/user-1", {}, mock_handler)

                assert result is not None
                body, status, _ = result
                assert status == 200

    def test_delete_user(self, scim_handler, mock_handler, mock_scim_server):
        """DELETE /scim/v2/Users/{id} removes a user."""
        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                result = scim_handler.handle_delete("/scim/v2/Users/user-1", {}, mock_handler)

                assert result is not None
                _, status, _ = result
                assert status == 204


# -----------------------------------------------------------------------------
# Group CRUD Tests
# -----------------------------------------------------------------------------


class TestSCIMHandlerGroupOperations:
    """Tests for SCIM group CRUD operations."""

    def test_list_groups(self, scim_handler, mock_handler, mock_scim_server):
        """GET /scim/v2/Groups returns group list."""
        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                result = scim_handler.handle("/scim/v2/Groups", {}, mock_handler)

                assert result is not None
                body, status, content_type = result
                assert status == 200
                assert content_type == SCIM_CONTENT_TYPE

                data = json.loads(body)
                assert data["totalResults"] == 1

    def test_get_group_by_id(self, scim_handler, mock_handler, mock_scim_server):
        """GET /scim/v2/Groups/{id} returns specific group."""
        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                result = scim_handler.handle("/scim/v2/Groups/group-1", {}, mock_handler)

                assert result is not None
                body, status, _ = result
                assert status == 200

                data = json.loads(body)
                assert data["id"] == "group-1"

    def test_create_group(self, scim_handler, mock_handler, mock_scim_server):
        """POST /scim/v2/Groups creates a new group."""
        group_data = {"displayName": "New Group"}
        scim_handler.ctx = {"body": group_data}

        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                result = scim_handler.handle_post("/scim/v2/Groups", {}, mock_handler)

                assert result is not None
                body, status, _ = result
                assert status == 201

    def test_replace_group(self, scim_handler, mock_handler, mock_scim_server):
        """PUT /scim/v2/Groups/{id} replaces a group."""
        group_data = {"displayName": "Updated Group"}
        scim_handler.ctx = {"body": group_data}

        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                result = scim_handler.handle_put("/scim/v2/Groups/group-1", {}, mock_handler)

                assert result is not None
                body, status, _ = result
                assert status == 200

    def test_patch_group(self, scim_handler, mock_handler, mock_scim_server):
        """PATCH /scim/v2/Groups/{id} partially updates a group."""
        patch_data = {
            "schemas": ["urn:ietf:params:scim:api:messages:2.0:PatchOp"],
            "Operations": [{"op": "add", "path": "members", "value": [{"value": "user-1"}]}],
        }
        scim_handler.ctx = {"body": patch_data}

        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                result = scim_handler.handle_patch("/scim/v2/Groups/group-1", {}, mock_handler)

                assert result is not None
                _, status, _ = result
                assert status == 200

    def test_delete_group(self, scim_handler, mock_handler, mock_scim_server):
        """DELETE /scim/v2/Groups/{id} removes a group."""
        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                result = scim_handler.handle_delete("/scim/v2/Groups/group-1", {}, mock_handler)

                assert result is not None
                _, status, _ = result
                assert status == 204


# -----------------------------------------------------------------------------
# Error Handling Tests
# -----------------------------------------------------------------------------


class TestSCIMHandlerErrorHandling:
    """Tests for SCIM error handling."""

    def test_scim_module_not_available(self, scim_handler, mock_handler):
        """Returns 503 when SCIM module is not available."""
        with patch("aragora.server.handlers.scim_handler.SCIM_AVAILABLE", False):
            result = scim_handler.handle("/scim/v2/Users", {}, mock_handler)

            assert result is not None
            assert result.status_code == 503
            assert b"not available" in result.body

    def test_scim_server_init_failed(self, scim_handler, mock_handler):
        """Returns 503 when SCIM server initialization fails."""
        with patch.object(scim_handler, "_get_scim_server", return_value=None):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                result = scim_handler.handle("/scim/v2/Users", {}, mock_handler)

                assert result is not None
                assert result.status_code == 503

    def test_invalid_json_body(self, scim_handler, mock_handler, mock_scim_server):
        """Returns 400 for invalid JSON body."""
        scim_handler.ctx = {"body": None}

        with patch.object(scim_handler, "_get_scim_server", return_value=mock_scim_server):
            with patch.object(scim_handler, "_verify_bearer_token", return_value=None):
                with patch.object(scim_handler, "_read_json_body", return_value=None):
                    result = scim_handler.handle_post("/scim/v2/Users", {}, mock_handler)

                    assert result is not None
                    body, status, _ = result
                    assert status == 400
                    assert "Invalid JSON" in body

    def test_scim_error_format(self, scim_handler):
        """SCIM errors follow RFC 7644 format."""
        result = scim_handler._scim_error("Test error message", 400)

        body, status, content_type = result
        assert status == 400
        assert content_type == SCIM_CONTENT_TYPE

        data = json.loads(body)
        assert "schemas" in data
        assert "urn:ietf:params:scim:api:messages:2.0:Error" in data["schemas"]
        assert data["detail"] == "Test error message"
        assert data["status"] == "400"


# -----------------------------------------------------------------------------
# Resource ID Extraction Tests
# -----------------------------------------------------------------------------


class TestSCIMHandlerResourceIdExtraction:
    """Tests for resource ID extraction from paths."""

    def test_extract_user_id(self, scim_handler):
        """Extracts user ID from path."""
        user_id = scim_handler._extract_resource_id("/scim/v2/Users/user-123", "Users")
        assert user_id == "user-123"

    def test_extract_group_id(self, scim_handler):
        """Extracts group ID from path."""
        group_id = scim_handler._extract_resource_id("/scim/v2/Groups/group-456", "Groups")
        assert group_id == "group-456"

    def test_extract_id_with_trailing_slash(self, scim_handler):
        """Handles trailing slash in path."""
        user_id = scim_handler._extract_resource_id("/scim/v2/Users/user-123/", "Users")
        assert user_id == "user-123"

    def test_extract_id_with_query_params(self, scim_handler):
        """Handles query params in path."""
        user_id = scim_handler._extract_resource_id("/scim/v2/Users/user-123?foo=bar", "Users")
        assert user_id == "user-123"

    def test_extract_id_no_id(self, scim_handler):
        """Returns None when no ID in path."""
        user_id = scim_handler._extract_resource_id("/scim/v2/Users", "Users")
        assert user_id is None

    def test_extract_id_empty_id(self, scim_handler):
        """Returns None for empty ID."""
        user_id = scim_handler._extract_resource_id("/scim/v2/Users/", "Users")
        assert user_id is None


# -----------------------------------------------------------------------------
# SCIM Response Format Tests
# -----------------------------------------------------------------------------


class TestSCIMHandlerResponseFormat:
    """Tests for SCIM response formatting."""

    def test_scim_response_json(self, scim_handler):
        """SCIM response returns proper JSON."""
        result = scim_handler._scim_response({"id": "test", "userName": "test@example.com"}, 200)

        body, status, content_type = result
        assert status == 200
        assert content_type == SCIM_CONTENT_TYPE

        data = json.loads(body)
        assert data["id"] == "test"
        assert data["userName"] == "test@example.com"

    def test_scim_response_no_content(self, scim_handler):
        """SCIM response handles 204 No Content."""
        result = scim_handler._scim_response(None, 204)

        body, status, content_type = result
        assert status == 204
        assert body == ""
        assert content_type == SCIM_CONTENT_TYPE

    def test_scim_response_empty_body(self, scim_handler):
        """SCIM response handles empty body."""
        result = scim_handler._scim_response({}, 200)

        body, status, content_type = result
        assert status == 200
        assert content_type == SCIM_CONTENT_TYPE

        data = json.loads(body)
        assert data == {}
