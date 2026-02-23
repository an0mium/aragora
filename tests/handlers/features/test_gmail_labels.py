"""Tests for Gmail labels and filters handler.

Tests the GmailLabelsHandler covering:
- POST /api/v1/gmail/labels - Create label
- GET /api/v1/gmail/labels - List labels
- PATCH /api/v1/gmail/labels/:id - Update label
- DELETE /api/v1/gmail/labels/:id - Delete label
- POST /api/v1/gmail/messages/:id/labels - Modify message labels
- POST /api/v1/gmail/messages/:id/read - Mark as read/unread
- POST /api/v1/gmail/messages/:id/star - Star/unstar message
- POST /api/v1/gmail/messages/:id/archive - Archive message
- POST /api/v1/gmail/messages/:id/trash - Trash/untrash message
- POST /api/v1/gmail/filters - Create filter
- GET /api/v1/gmail/filters - List filters
- DELETE /api/v1/gmail/filters/:id - Delete filter
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.gmail_labels import GmailLabelsHandler
from aragora.storage.gmail_token_store import GmailUserState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _status(result) -> int:
    """Extract status code from HandlerResult."""
    return result.status_code


def _body(result) -> dict[str, Any]:
    """Extract parsed JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


@dataclass
class MockHTTPHandler:
    """Mock HTTP handler for tests."""

    path: str = "/"
    method: str = "GET"
    body: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    command: str = "GET"

    def __post_init__(self):
        if self.headers is None:
            self.headers = {"Content-Length": "0", "Content-Type": "application/json"}
        self.client_address = ("127.0.0.1", 12345)
        self.rfile = MagicMock()
        if self.body:
            body_bytes = json.dumps(self.body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a GmailLabelsHandler with minimal context."""
    return GmailLabelsHandler(server_context={})


@pytest.fixture
def mock_http():
    """Create a basic mock HTTP handler (no body)."""
    return MockHTTPHandler()


@pytest.fixture
def mock_http_with_body():
    """Factory for mock HTTP handler with body."""
    def _create(body: dict[str, Any]) -> MockHTTPHandler:
        return MockHTTPHandler(body=body)
    return _create


@pytest.fixture
def gmail_state():
    """Create a GmailUserState with valid tokens."""
    return GmailUserState(
        user_id="default",
        access_token="test-access-token",
        refresh_token="test-refresh-token",
        token_expiry=datetime(2099, 1, 1, tzinfo=timezone.utc),
        email_address="user@example.com",
    )


@pytest.fixture
def gmail_state_no_refresh():
    """Create a GmailUserState without a refresh token."""
    return GmailUserState(
        user_id="default",
        access_token="test-access-token",
        refresh_token="",
    )


# Patch target for get_user_state
_GET_USER_STATE = "aragora.server.handlers.features.gmail_labels.get_user_state"


# ---------------------------------------------------------------------------
# can_handle tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for can_handle route matching."""

    def test_labels_list(self, handler):
        assert handler.can_handle("/api/v1/gmail/labels") is True

    def test_filters_list(self, handler):
        assert handler.can_handle("/api/v1/gmail/filters") is True

    def test_label_by_id(self, handler):
        assert handler.can_handle("/api/v1/gmail/labels/Label_1") is True

    def test_filter_by_id(self, handler):
        assert handler.can_handle("/api/v1/gmail/filters/filter_abc") is True

    def test_message_labels(self, handler):
        assert handler.can_handle("/api/v1/gmail/messages/msg1/labels") is True

    def test_message_read(self, handler):
        assert handler.can_handle("/api/v1/gmail/messages/msg1/read") is True

    def test_message_star(self, handler):
        assert handler.can_handle("/api/v1/gmail/messages/msg1/star") is True

    def test_message_archive(self, handler):
        assert handler.can_handle("/api/v1/gmail/messages/msg1/archive") is True

    def test_message_trash(self, handler):
        assert handler.can_handle("/api/v1/gmail/messages/msg1/trash") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_partial_labels(self, handler):
        assert handler.can_handle("/api/v1/gmail/label") is False

    def test_partial_filters(self, handler):
        assert handler.can_handle("/api/v1/gmail/filter") is False

    def test_gmail_root(self, handler):
        assert handler.can_handle("/api/v1/gmail") is False

    def test_different_api(self, handler):
        assert handler.can_handle("/api/v2/gmail/labels") is False

    def test_can_handle_with_method(self, handler):
        assert handler.can_handle("/api/v1/gmail/labels", "POST") is True
        assert handler.can_handle("/api/v1/gmail/labels", "DELETE") is True


# ---------------------------------------------------------------------------
# GET /api/v1/gmail/labels - List labels
# ---------------------------------------------------------------------------


class TestListLabels:
    """Tests for GET /api/v1/gmail/labels."""

    @pytest.mark.asyncio
    async def test_list_labels_success(self, handler, mock_http, gmail_state):
        mock_label = MagicMock()
        mock_label.id = "Label_1"
        mock_label.name = "Work"
        mock_label.type = "user"
        mock_label.message_list_visibility = "show"
        mock_label.label_list_visibility = "labelShow"

        mock_connector = AsyncMock()
        mock_connector.list_labels.return_value = [mock_label]

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch(
                "aragora.server.handlers.features.gmail_labels.GmailConnector",
                return_value=mock_connector,
            ) as mock_cls:
                # The handler creates connector via cast(type, GmailConnector)()
                # We need to patch at the import location
                with patch.object(
                    handler,
                    "_list_labels",
                    new_callable=AsyncMock,
                ) as mock_list:
                    mock_list.return_value = MagicMock(
                        status_code=200,
                        body=json.dumps({
                            "labels": [
                                {
                                    "id": "Label_1",
                                    "name": "Work",
                                    "type": "user",
                                    "message_list_visibility": "show",
                                    "label_list_visibility": "labelShow",
                                }
                            ],
                            "count": 1,
                        }).encode(),
                    )
                    result = await handler.handle(
                        "/api/v1/gmail/labels", {}, mock_http
                    )

        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 1
        assert body["labels"][0]["id"] == "Label_1"
        assert body["labels"][0]["name"] == "Work"

    @pytest.mark.asyncio
    async def test_list_labels_multiple(self, handler, mock_http, gmail_state):
        mock_labels = []
        for i, name in enumerate(["INBOX", "SENT", "Work", "Personal"]):
            lbl = MagicMock()
            lbl.id = f"Label_{i}"
            lbl.name = name
            lbl.type = "system" if i < 2 else "user"
            lbl.message_list_visibility = "show"
            lbl.label_list_visibility = "labelShow"
            mock_labels.append(lbl)

        mock_connector = MagicMock()
        mock_connector.list_labels = AsyncMock(return_value=mock_labels)

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch(
                "aragora.connectors.enterprise.communication.gmail.GmailConnector",
                return_value=mock_connector,
            ):
                result = await handler.handle(
                    "/api/v1/gmail/labels", {}, mock_http
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 4

    @pytest.mark.asyncio
    async def test_list_labels_empty(self, handler, mock_http, gmail_state):
        mock_connector = MagicMock()
        mock_connector.list_labels = AsyncMock(return_value=[])

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch(
                "aragora.connectors.enterprise.communication.gmail.GmailConnector",
                return_value=mock_connector,
            ):
                result = await handler.handle(
                    "/api/v1/gmail/labels", {}, mock_http
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 0
        assert body["labels"] == []

    @pytest.mark.asyncio
    async def test_list_labels_api_error(self, handler, mock_http, gmail_state):
        mock_connector = MagicMock()
        mock_connector.list_labels = AsyncMock(side_effect=ConnectionError("API down"))

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch(
                "aragora.connectors.enterprise.communication.gmail.GmailConnector",
                return_value=mock_connector,
            ):
                result = await handler.handle(
                    "/api/v1/gmail/labels", {}, mock_http
                )

        assert _status(result) == 500
        assert "Failed to list labels" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_list_labels_timeout(self, handler, mock_http, gmail_state):
        mock_connector = MagicMock()
        mock_connector.list_labels = AsyncMock(side_effect=TimeoutError("timeout"))

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch(
                "aragora.connectors.enterprise.communication.gmail.GmailConnector",
                return_value=mock_connector,
            ):
                result = await handler.handle(
                    "/api/v1/gmail/labels", {}, mock_http
                )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_list_labels_value_error(self, handler, mock_http, gmail_state):
        mock_connector = MagicMock()
        mock_connector.list_labels = AsyncMock(side_effect=ValueError("bad data"))

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch(
                "aragora.connectors.enterprise.communication.gmail.GmailConnector",
                return_value=mock_connector,
            ):
                result = await handler.handle(
                    "/api/v1/gmail/labels", {}, mock_http
                )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_list_labels_no_state(self, handler, mock_http):
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle(
                "/api/v1/gmail/labels", {}, mock_http
            )

        assert _status(result) == 401
        assert "authenticate" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_list_labels_no_refresh_token(self, handler, mock_http, gmail_state_no_refresh):
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state_no_refresh,
        ):
            result = await handler.handle(
                "/api/v1/gmail/labels", {}, mock_http
            )

        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_list_labels_with_user_id(self, handler, mock_http, gmail_state):
        """Passing user_id query param should route to correct user state."""
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ) as mock_get:
            with patch.object(
                handler,
                "_list_labels",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    status_code=200,
                    body=json.dumps({"labels": [], "count": 0}).encode(),
                ),
            ):
                await handler.handle(
                    "/api/v1/gmail/labels", {"user_id": "user42"}, mock_http
                )
            mock_get.assert_called_once_with("user42")

    @pytest.mark.asyncio
    async def test_list_labels_default_user_id(self, handler, mock_http, gmail_state):
        """Missing user_id should default to 'default'."""
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ) as mock_get:
            with patch.object(
                handler,
                "_list_labels",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    status_code=200,
                    body=json.dumps({"labels": [], "count": 0}).encode(),
                ),
            ):
                await handler.handle(
                    "/api/v1/gmail/labels", {}, mock_http
                )
            mock_get.assert_called_once_with("default")


# ---------------------------------------------------------------------------
# POST /api/v1/gmail/labels - Create label
# ---------------------------------------------------------------------------


class TestCreateLabel:
    """Tests for POST /api/v1/gmail/labels."""

    @pytest.mark.asyncio
    async def test_create_label_success(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"name": "Projects"})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_create_label",
                new_callable=AsyncMock,
                return_value={"id": "Label_99", "name": "Projects"},
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/labels", {}, http
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["label"]["id"] == "Label_99"
        assert body["label"]["name"] == "Projects"

    @pytest.mark.asyncio
    async def test_create_label_missing_name(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/labels", {}, http
            )

        assert _status(result) == 400
        assert "name" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_label_empty_name(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"name": ""})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/labels", {}, http
            )

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_label_api_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"name": "Test"})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_create_label",
                new_callable=AsyncMock,
                side_effect=ConnectionError("API fail"),
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/labels", {}, http
                )

        assert _status(result) == 500
        assert "creation failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_label_timeout(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"name": "Test"})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_create_label",
                new_callable=AsyncMock,
                side_effect=TimeoutError("timeout"),
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/labels", {}, http
                )

        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_create_label_no_state(self, handler, mock_http_with_body):
        http = mock_http_with_body({"name": "Test"})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/labels", {}, http
            )

        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_create_label_invalid_json(self, handler, mock_http):
        """Handler with no body should get invalid JSON error."""
        # MockHTTPHandler with no body has Content-Length: 0 -> read_json_body returns {}
        # but we need to simulate None from read_json_body
        with patch.object(handler, "read_json_body", return_value=None):
            result = await handler.handle_post(
                "/api/v1/gmail/labels", {}, mock_http
            )

        assert _status(result) == 400
        assert "json" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_label_with_colors(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({
            "name": "Important",
            "background_color": "#ff0000",
            "text_color": "#ffffff",
        })

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_create_label",
                new_callable=AsyncMock,
                return_value={"id": "Label_c", "name": "Important"},
            ) as mock_api:
                result = await handler.handle_post(
                    "/api/v1/gmail/labels", {}, http
                )
                # Verify the body was passed through
                call_args = mock_api.call_args
                options = call_args[0][2]
                assert options["background_color"] == "#ff0000"
                assert options["text_color"] == "#ffffff"

        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_create_label_user_id_from_body(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"name": "Test", "user_id": "custom_user"})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ) as mock_get:
            with patch.object(
                handler,
                "_api_create_label",
                new_callable=AsyncMock,
                return_value={"id": "L1", "name": "Test"},
            ):
                await handler.handle_post(
                    "/api/v1/gmail/labels", {}, http
                )
            mock_get.assert_called_once_with("custom_user")


# ---------------------------------------------------------------------------
# PATCH /api/v1/gmail/labels/:id - Update label
# ---------------------------------------------------------------------------


class TestUpdateLabel:
    """Tests for PATCH /api/v1/gmail/labels/:id."""

    @pytest.mark.asyncio
    async def test_update_label_success(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"name": "Renamed"})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_update_label",
                new_callable=AsyncMock,
                return_value={"id": "Label_1", "name": "Renamed"},
            ):
                result = await handler.handle_patch(
                    "/api/v1/gmail/labels/Label_1", {}, http
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["label"]["name"] == "Renamed"

    @pytest.mark.asyncio
    async def test_update_label_api_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"name": "Renamed"})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_update_label",
                new_callable=AsyncMock,
                side_effect=ConnectionError("API fail"),
            ):
                result = await handler.handle_patch(
                    "/api/v1/gmail/labels/Label_1", {}, http
                )

        assert _status(result) == 500
        assert "update failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_update_label_no_state(self, handler, mock_http_with_body):
        http = mock_http_with_body({"name": "Renamed"})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle_patch(
                "/api/v1/gmail/labels/Label_1", {}, http
            )

        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_update_label_invalid_json(self, handler, mock_http):
        with patch.object(handler, "read_json_body", return_value=None):
            result = await handler.handle_patch(
                "/api/v1/gmail/labels/Label_1", {}, mock_http
            )

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_update_label_not_found_path(self, handler, mock_http_with_body, gmail_state):
        """PATCH on a path that doesn't match labels returns 404."""
        http = mock_http_with_body({"name": "Renamed"})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            result = await handler.handle_patch(
                "/api/v1/gmail/unknown/something", {}, http
            )

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_update_label_extracts_id(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"name": "Updated"})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_update_label",
                new_callable=AsyncMock,
                return_value={"id": "MyLabel123", "name": "Updated"},
            ) as mock_api:
                await handler.handle_patch(
                    "/api/v1/gmail/labels/MyLabel123", {}, http
                )
                call_args = mock_api.call_args
                assert call_args[0][1] == "MyLabel123"

    @pytest.mark.asyncio
    async def test_update_label_no_refresh_token(self, handler, mock_http_with_body, gmail_state_no_refresh):
        http = mock_http_with_body({"name": "Renamed"})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state_no_refresh,
        ):
            result = await handler.handle_patch(
                "/api/v1/gmail/labels/Label_1", {}, http
            )

        assert _status(result) == 401


# ---------------------------------------------------------------------------
# DELETE /api/v1/gmail/labels/:id - Delete label
# ---------------------------------------------------------------------------


class TestDeleteLabel:
    """Tests for DELETE /api/v1/gmail/labels/:id."""

    @pytest.mark.asyncio
    async def test_delete_label_success(self, handler, mock_http, gmail_state):
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_delete_label",
                new_callable=AsyncMock,
            ):
                result = await handler.handle_delete(
                    "/api/v1/gmail/labels/Label_1", {}, mock_http
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["deleted"] == "Label_1"

    @pytest.mark.asyncio
    async def test_delete_label_api_error(self, handler, mock_http, gmail_state):
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_delete_label",
                new_callable=AsyncMock,
                side_effect=ConnectionError("API fail"),
            ):
                result = await handler.handle_delete(
                    "/api/v1/gmail/labels/Label_1", {}, mock_http
                )

        assert _status(result) == 500
        assert "deletion failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_delete_label_no_state(self, handler, mock_http):
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle_delete(
                "/api/v1/gmail/labels/Label_1", {}, mock_http
            )

        assert _status(result) == 401

    @pytest.mark.asyncio
    async def test_delete_label_user_id_param(self, handler, mock_http, gmail_state):
        """DELETE picks user_id from query_params."""
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ) as mock_get:
            with patch.object(
                handler,
                "_api_delete_label",
                new_callable=AsyncMock,
            ):
                await handler.handle_delete(
                    "/api/v1/gmail/labels/Label_1", {"user_id": "u99"}, mock_http
                )
            mock_get.assert_called_once_with("u99")

    @pytest.mark.asyncio
    async def test_delete_label_not_found_path(self, handler, mock_http, gmail_state):
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            result = await handler.handle_delete(
                "/api/v1/gmail/unknown/something", {}, mock_http
            )

        assert _status(result) == 404


# ---------------------------------------------------------------------------
# POST /api/v1/gmail/messages/:id/labels - Modify message labels
# ---------------------------------------------------------------------------


class TestModifyMessageLabels:
    """Tests for POST /api/v1/gmail/messages/:id/labels."""

    @pytest.mark.asyncio
    async def test_modify_labels_add(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"add": ["STARRED"], "remove": []})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_modify_labels",
                new_callable=AsyncMock,
                return_value={"labelIds": ["STARRED", "INBOX"]},
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/labels", {}, http
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["message_id"] == "msg1"
        assert "STARRED" in body["labels"]

    @pytest.mark.asyncio
    async def test_modify_labels_remove(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"add": [], "remove": ["INBOX"]})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_modify_labels",
                new_callable=AsyncMock,
                return_value={"labelIds": ["SENT"]},
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/labels", {}, http
                )

        assert _status(result) == 200
        body = _body(result)
        assert "INBOX" not in body["labels"]

    @pytest.mark.asyncio
    async def test_modify_labels_empty(self, handler, mock_http_with_body, gmail_state):
        """Must specify at least one label to add or remove."""
        http = mock_http_with_body({"add": [], "remove": []})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/messages/msg1/labels", {}, http
            )

        assert _status(result) == 400
        assert "add or remove" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_modify_labels_no_fields(self, handler, mock_http_with_body, gmail_state):
        """Body without add/remove fields should default to empty."""
        http = mock_http_with_body({})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/messages/msg1/labels", {}, http
            )

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_modify_labels_api_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"add": ["STARRED"]})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_modify_labels",
                new_callable=AsyncMock,
                side_effect=ConnectionError("API fail"),
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/labels", {}, http
                )

        assert _status(result) == 500
        assert "modification failed" in _body(result)["error"].lower()


# ---------------------------------------------------------------------------
# POST /api/v1/gmail/messages/:id/read - Mark as read/unread
# ---------------------------------------------------------------------------


class TestMarkRead:
    """Tests for POST /api/v1/gmail/messages/:id/read."""

    @pytest.mark.asyncio
    async def test_mark_read(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"read": True})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_modify_labels",
                new_callable=AsyncMock,
                return_value={},
            ) as mock_api:
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/read", {}, http
                )
                # Mark as read removes UNREAD
                call_args = mock_api.call_args
                assert call_args[0][2] == []  # add_labels
                assert call_args[0][3] == ["UNREAD"]  # remove_labels

        assert _status(result) == 200
        body = _body(result)
        assert body["is_read"] is True
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_mark_unread(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"read": False})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_modify_labels",
                new_callable=AsyncMock,
                return_value={},
            ) as mock_api:
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/read", {}, http
                )
                call_args = mock_api.call_args
                assert call_args[0][2] == ["UNREAD"]
                assert call_args[0][3] == []

        assert _status(result) == 200
        body = _body(result)
        assert body["is_read"] is False

    @pytest.mark.asyncio
    async def test_mark_read_default_true(self, handler, mock_http_with_body, gmail_state):
        """read defaults to True when not specified."""
        http = mock_http_with_body({})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_modify_labels",
                new_callable=AsyncMock,
                return_value={},
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/read", {}, http
                )

        assert _status(result) == 200
        assert _body(result)["is_read"] is True

    @pytest.mark.asyncio
    async def test_mark_read_api_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"read": True})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_modify_labels",
                new_callable=AsyncMock,
                side_effect=ValueError("bad"),
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/read", {}, http
                )

        assert _status(result) == 500


# ---------------------------------------------------------------------------
# POST /api/v1/gmail/messages/:id/star - Star/unstar
# ---------------------------------------------------------------------------


class TestStarMessage:
    """Tests for POST /api/v1/gmail/messages/:id/star."""

    @pytest.mark.asyncio
    async def test_star_message(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"starred": True})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_modify_labels",
                new_callable=AsyncMock,
                return_value={},
            ) as mock_api:
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/star", {}, http
                )
                call_args = mock_api.call_args
                assert call_args[0][2] == ["STARRED"]
                assert call_args[0][3] == []

        assert _status(result) == 200
        body = _body(result)
        assert body["is_starred"] is True
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_unstar_message(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"starred": False})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_modify_labels",
                new_callable=AsyncMock,
                return_value={},
            ) as mock_api:
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/star", {}, http
                )
                call_args = mock_api.call_args
                assert call_args[0][2] == []
                assert call_args[0][3] == ["STARRED"]

        assert _status(result) == 200
        body = _body(result)
        assert body["is_starred"] is False

    @pytest.mark.asyncio
    async def test_star_default_true(self, handler, mock_http_with_body, gmail_state):
        """starred defaults to True."""
        http = mock_http_with_body({})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_modify_labels",
                new_callable=AsyncMock,
                return_value={},
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/star", {}, http
                )

        assert _body(result)["is_starred"] is True

    @pytest.mark.asyncio
    async def test_star_api_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"starred": True})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_modify_labels",
                new_callable=AsyncMock,
                side_effect=OSError("network"),
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/star", {}, http
                )

        assert _status(result) == 500


# ---------------------------------------------------------------------------
# POST /api/v1/gmail/messages/:id/archive - Archive
# ---------------------------------------------------------------------------


class TestArchiveMessage:
    """Tests for POST /api/v1/gmail/messages/:id/archive."""

    @pytest.mark.asyncio
    async def test_archive_message(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_modify_labels",
                new_callable=AsyncMock,
                return_value={},
            ) as mock_api:
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/archive", {}, http
                )
                # Archive removes INBOX label
                call_args = mock_api.call_args
                assert call_args[0][2] == []
                assert call_args[0][3] == ["INBOX"]

        assert _status(result) == 200
        body = _body(result)
        assert body["archived"] is True
        assert body["success"] is True
        assert body["message_id"] == "msg1"

    @pytest.mark.asyncio
    async def test_archive_api_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_modify_labels",
                new_callable=AsyncMock,
                side_effect=ConnectionError("fail"),
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/archive", {}, http
                )

        assert _status(result) == 500
        assert "archive" in _body(result)["error"].lower()


# ---------------------------------------------------------------------------
# POST /api/v1/gmail/messages/:id/trash - Trash/untrash
# ---------------------------------------------------------------------------


class TestTrashMessage:
    """Tests for POST /api/v1/gmail/messages/:id/trash."""

    @pytest.mark.asyncio
    async def test_trash_message(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"trash": True})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_trash_message",
                new_callable=AsyncMock,
            ) as mock_trash:
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/trash", {}, http
                )
                mock_trash.assert_called_once()

        assert _status(result) == 200
        body = _body(result)
        assert body["trashed"] is True
        assert body["success"] is True

    @pytest.mark.asyncio
    async def test_untrash_message(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"trash": False})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_untrash_message",
                new_callable=AsyncMock,
            ) as mock_untrash:
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/trash", {}, http
                )
                mock_untrash.assert_called_once()

        assert _status(result) == 200
        body = _body(result)
        assert body["trashed"] is False

    @pytest.mark.asyncio
    async def test_trash_default_true(self, handler, mock_http_with_body, gmail_state):
        """trash defaults to True."""
        http = mock_http_with_body({})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_trash_message",
                new_callable=AsyncMock,
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/trash", {}, http
                )

        assert _body(result)["trashed"] is True

    @pytest.mark.asyncio
    async def test_trash_api_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"trash": True})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_trash_message",
                new_callable=AsyncMock,
                side_effect=ConnectionError("fail"),
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/trash", {}, http
                )

        assert _status(result) == 500
        assert "trash" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_untrash_api_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({"trash": False})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_untrash_message",
                new_callable=AsyncMock,
                side_effect=TimeoutError("timeout"),
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/messages/msg1/trash", {}, http
                )

        assert _status(result) == 500


# ---------------------------------------------------------------------------
# POST /api/v1/gmail/filters - Create filter
# ---------------------------------------------------------------------------


class TestCreateFilter:
    """Tests for POST /api/v1/gmail/filters."""

    @pytest.mark.asyncio
    async def test_create_filter_success(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({
            "criteria": {"from": "noreply@example.com"},
            "action": {"add_labels": ["Label_1"]},
        })

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_create_filter",
                new_callable=AsyncMock,
                return_value={"id": "filter_1", "criteria": {}, "action": {}},
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/filters", {}, http
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["filter"]["id"] == "filter_1"

    @pytest.mark.asyncio
    async def test_create_filter_no_criteria(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({
            "criteria": {},
            "action": {"add_labels": ["INBOX"]},
        })

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/filters", {}, http
            )

        assert _status(result) == 400
        assert "criteria" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_filter_no_action(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({
            "criteria": {"from": "test@test.com"},
            "action": {},
        })

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/filters", {}, http
            )

        assert _status(result) == 400
        assert "action" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_filter_missing_both(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/filters", {}, http
            )

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_create_filter_api_error(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({
            "criteria": {"from": "x@x.com"},
            "action": {"star": True},
        })

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_create_filter",
                new_callable=AsyncMock,
                side_effect=ValueError("bad filter"),
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/filters", {}, http
                )

        assert _status(result) == 500
        assert "creation failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_create_filter_no_state(self, handler, mock_http_with_body):
        http = mock_http_with_body({
            "criteria": {"from": "a@b.com"},
            "action": {"star": True},
        })

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/filters", {}, http
            )

        assert _status(result) == 401


# ---------------------------------------------------------------------------
# GET /api/v1/gmail/filters - List filters
# ---------------------------------------------------------------------------


class TestListFilters:
    """Tests for GET /api/v1/gmail/filters."""

    @pytest.mark.asyncio
    async def test_list_filters_success(self, handler, mock_http, gmail_state):
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_list_filters",
                new_callable=AsyncMock,
                return_value=[
                    {"id": "f1", "criteria": {"from": "a@b.com"}, "action": {}},
                    {"id": "f2", "criteria": {"subject": "test"}, "action": {}},
                ],
            ):
                result = await handler.handle(
                    "/api/v1/gmail/filters", {}, mock_http
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 2
        assert len(body["filters"]) == 2

    @pytest.mark.asyncio
    async def test_list_filters_empty(self, handler, mock_http, gmail_state):
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_list_filters",
                new_callable=AsyncMock,
                return_value=[],
            ):
                result = await handler.handle(
                    "/api/v1/gmail/filters", {}, mock_http
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 0
        assert body["filters"] == []

    @pytest.mark.asyncio
    async def test_list_filters_api_error(self, handler, mock_http, gmail_state):
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_list_filters",
                new_callable=AsyncMock,
                side_effect=ConnectionError("API down"),
            ):
                result = await handler.handle(
                    "/api/v1/gmail/filters", {}, mock_http
                )

        assert _status(result) == 500
        assert "Failed to list filters" in _body(result)["error"]

    @pytest.mark.asyncio
    async def test_list_filters_no_state(self, handler, mock_http):
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle(
                "/api/v1/gmail/filters", {}, mock_http
            )

        assert _status(result) == 401


# ---------------------------------------------------------------------------
# DELETE /api/v1/gmail/filters/:id - Delete filter
# ---------------------------------------------------------------------------


class TestDeleteFilter:
    """Tests for DELETE /api/v1/gmail/filters/:id."""

    @pytest.mark.asyncio
    async def test_delete_filter_success(self, handler, mock_http, gmail_state):
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_delete_filter",
                new_callable=AsyncMock,
            ):
                result = await handler.handle_delete(
                    "/api/v1/gmail/filters/filter_1", {}, mock_http
                )

        assert _status(result) == 200
        body = _body(result)
        assert body["success"] is True
        assert body["deleted"] == "filter_1"

    @pytest.mark.asyncio
    async def test_delete_filter_api_error(self, handler, mock_http, gmail_state):
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            with patch.object(
                handler,
                "_api_delete_filter",
                new_callable=AsyncMock,
                side_effect=OSError("fail"),
            ):
                result = await handler.handle_delete(
                    "/api/v1/gmail/filters/filter_1", {}, mock_http
                )

        assert _status(result) == 500
        assert "deletion failed" in _body(result)["error"].lower()

    @pytest.mark.asyncio
    async def test_delete_filter_no_state(self, handler, mock_http):
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await handler.handle_delete(
                "/api/v1/gmail/filters/filter_1", {}, mock_http
            )

        assert _status(result) == 401


# ---------------------------------------------------------------------------
# POST routing - Not found paths
# ---------------------------------------------------------------------------


class TestPostRouting:
    """Tests for POST routing edge cases."""

    @pytest.mark.asyncio
    async def test_post_unknown_path(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/unknown", {}, http
            )

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_post_message_unknown_action(self, handler, mock_http_with_body, gmail_state):
        http = mock_http_with_body({})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/messages/msg1/unknown_action", {}, http
            )

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_post_message_too_short_path(self, handler, mock_http_with_body, gmail_state):
        """Path without enough segments should fall through to 404."""
        http = mock_http_with_body({})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/messages/", {}, http
            )

        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_post_invalid_json_body(self, handler, mock_http):
        with patch.object(handler, "read_json_body", return_value=None):
            result = await handler.handle_post(
                "/api/v1/gmail/labels", {}, mock_http
            )

        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_post_no_refresh_token(self, handler, mock_http_with_body, gmail_state_no_refresh):
        http = mock_http_with_body({"name": "Test"})

        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state_no_refresh,
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/labels", {}, http
            )

        assert _status(result) == 401


# ---------------------------------------------------------------------------
# GET routing - Not found
# ---------------------------------------------------------------------------


class TestGetRouting:
    """Tests for GET routing edge cases."""

    @pytest.mark.asyncio
    async def test_get_unknown_path(self, handler, mock_http, gmail_state):
        with patch(
            _GET_USER_STATE,
            new_callable=AsyncMock,
            return_value=gmail_state,
        ):
            result = await handler.handle(
                "/api/v1/gmail/something", {}, mock_http
            )

        assert _status(result) == 404


# ---------------------------------------------------------------------------
# Auth/RBAC tests
# ---------------------------------------------------------------------------


class TestAuth:
    """Tests for authentication and permission handling."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_unauthorized(self, mock_http):
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        handler = GmailLabelsHandler(server_context={})

        with patch.object(
            SecureHandler,
            "get_auth_context",
            new_callable=AsyncMock,
            side_effect=UnauthorizedError("not auth"),
        ):
            result = await handler.handle(
                "/api/v1/gmail/labels", {}, mock_http
            )

        assert _status(result) == 401
        assert "Authentication required" in _body(result)["error"]

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_forbidden(self, mock_http):
        from aragora.server.handlers.secure import ForbiddenError, SecureHandler
        from aragora.rbac.models import AuthorizationContext

        handler = GmailLabelsHandler(server_context={})
        mock_ctx = AuthorizationContext(
            user_id="user1",
            user_email="u@e.com",
            roles={"viewer"},
            permissions=set(),
        )

        with patch.object(
            SecureHandler,
            "get_auth_context",
            new_callable=AsyncMock,
            return_value=mock_ctx,
        ):
            with patch.object(
                SecureHandler,
                "check_permission",
                side_effect=ForbiddenError("no gmail:read"),
            ):
                result = await handler.handle(
                    "/api/v1/gmail/labels", {}, mock_http
                )

        assert _status(result) == 403
        assert "Permission denied" in _body(result)["error"]

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_post_unauthorized(self, mock_http_with_body):
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        handler = GmailLabelsHandler(server_context={})
        http = mock_http_with_body({"name": "Test"})

        with patch.object(
            SecureHandler,
            "get_auth_context",
            new_callable=AsyncMock,
            side_effect=UnauthorizedError("not auth"),
        ):
            result = await handler.handle_post(
                "/api/v1/gmail/labels", {}, http
            )

        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_post_forbidden(self, mock_http_with_body):
        from aragora.server.handlers.secure import ForbiddenError, SecureHandler
        from aragora.rbac.models import AuthorizationContext

        handler = GmailLabelsHandler(server_context={})
        http = mock_http_with_body({"name": "Test"})
        mock_ctx = AuthorizationContext(
            user_id="user1",
            user_email="u@e.com",
            roles={"viewer"},
            permissions=set(),
        )

        with patch.object(
            SecureHandler,
            "get_auth_context",
            new_callable=AsyncMock,
            return_value=mock_ctx,
        ):
            with patch.object(
                SecureHandler,
                "check_permission",
                side_effect=ForbiddenError("no gmail:write"),
            ):
                result = await handler.handle_post(
                    "/api/v1/gmail/labels", {}, http
                )

        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_patch_unauthorized(self, mock_http_with_body):
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        handler = GmailLabelsHandler(server_context={})
        http = mock_http_with_body({"name": "Update"})

        with patch.object(
            SecureHandler,
            "get_auth_context",
            new_callable=AsyncMock,
            side_effect=UnauthorizedError("not auth"),
        ):
            result = await handler.handle_patch(
                "/api/v1/gmail/labels/Label_1", {}, http
            )

        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_patch_forbidden(self, mock_http_with_body):
        from aragora.server.handlers.secure import ForbiddenError, SecureHandler
        from aragora.rbac.models import AuthorizationContext

        handler = GmailLabelsHandler(server_context={})
        http = mock_http_with_body({"name": "Update"})
        mock_ctx = AuthorizationContext(
            user_id="user1",
            user_email="u@e.com",
            roles={"viewer"},
            permissions=set(),
        )

        with patch.object(
            SecureHandler,
            "get_auth_context",
            new_callable=AsyncMock,
            return_value=mock_ctx,
        ):
            with patch.object(
                SecureHandler,
                "check_permission",
                side_effect=ForbiddenError("no gmail:write"),
            ):
                result = await handler.handle_patch(
                    "/api/v1/gmail/labels/Label_1", {}, http
                )

        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_delete_unauthorized(self, mock_http):
        from aragora.server.handlers.secure import SecureHandler, UnauthorizedError

        handler = GmailLabelsHandler(server_context={})

        with patch.object(
            SecureHandler,
            "get_auth_context",
            new_callable=AsyncMock,
            side_effect=UnauthorizedError("not auth"),
        ):
            result = await handler.handle_delete(
                "/api/v1/gmail/labels/Label_1", {}, mock_http
            )

        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_handle_delete_forbidden(self, mock_http):
        from aragora.server.handlers.secure import ForbiddenError, SecureHandler
        from aragora.rbac.models import AuthorizationContext

        handler = GmailLabelsHandler(server_context={})
        mock_ctx = AuthorizationContext(
            user_id="user1",
            user_email="u@e.com",
            roles={"viewer"},
            permissions=set(),
        )

        with patch.object(
            SecureHandler,
            "get_auth_context",
            new_callable=AsyncMock,
            return_value=mock_ctx,
        ):
            with patch.object(
                SecureHandler,
                "check_permission",
                side_effect=ForbiddenError("no gmail:write"),
            ):
                result = await handler.handle_delete(
                    "/api/v1/gmail/labels/Label_1", {}, mock_http
                )

        assert _status(result) == 403


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for handler initialization."""

    def test_init_with_server_context(self):
        h = GmailLabelsHandler(server_context={"key": "val"})
        assert h.ctx == {"key": "val"}

    def test_init_with_ctx(self):
        h = GmailLabelsHandler(ctx={"other": 1})
        assert h.ctx == {"other": 1}

    def test_init_empty(self):
        h = GmailLabelsHandler()
        assert h.ctx == {}

    def test_init_server_context_overrides_ctx(self):
        h = GmailLabelsHandler(ctx={"a": 1}, server_context={"b": 2})
        assert h.ctx == {"b": 2}

    def test_routes(self):
        h = GmailLabelsHandler(server_context={})
        assert "/api/v1/gmail/labels" in h.ROUTES
        assert "/api/v1/gmail/filters" in h.ROUTES

    def test_route_prefixes(self):
        h = GmailLabelsHandler(server_context={})
        assert "/api/v1/gmail/labels/" in h.ROUTE_PREFIXES
        assert "/api/v1/gmail/messages/" in h.ROUTE_PREFIXES
        assert "/api/v1/gmail/filters/" in h.ROUTE_PREFIXES


# ---------------------------------------------------------------------------
# _api_create_filter criteria/action mapping
# ---------------------------------------------------------------------------


class TestApiCreateFilterMapping:
    """Tests verifying _api_create_filter correctly maps criteria and action fields."""

    @pytest.mark.asyncio
    async def test_filter_criteria_all_fields(self, handler, gmail_state):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "f1"}
        mock_response.raise_for_status = MagicMock()
        mock_session.post = AsyncMock(return_value=mock_response)

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        ))

        criteria = {
            "from": "a@b.com",
            "to": "c@d.com",
            "subject": "test",
            "query": "is:important",
            "has_attachment": True,
            "exclude_chats": True,
            "size": 1000,
            "size_comparison": "smaller",
        }
        action = {
            "add_labels": ["L1"],
            "remove_labels": ["L2"],
            "star": True,
            "important": True,
            "archive": True,
            "delete": True,
            "mark_read": True,
            "forward": "fwd@test.com",
        }

        with patch(
            "aragora.server.handlers.features.gmail_labels.get_http_pool",
            return_value=mock_pool,
        ):
            result = await handler._api_create_filter(gmail_state, criteria, action)

        assert result == {"id": "f1"}

        # Verify the JSON body posted
        call_args = mock_session.post.call_args
        posted_json = call_args[1]["json"]

        # Criteria mapping
        assert posted_json["criteria"]["from"] == "a@b.com"
        assert posted_json["criteria"]["to"] == "c@d.com"
        assert posted_json["criteria"]["subject"] == "test"
        assert posted_json["criteria"]["query"] == "is:important"
        assert posted_json["criteria"]["hasAttachment"] is True
        assert posted_json["criteria"]["excludeChats"] is True
        assert posted_json["criteria"]["size"] == 1000
        assert posted_json["criteria"]["sizeComparison"] == "smaller"

        # Action mapping
        action_json = posted_json["action"]
        assert "L1" in action_json["addLabelIds"]
        assert "STARRED" in action_json["addLabelIds"]
        assert "IMPORTANT" in action_json["addLabelIds"]
        assert "TRASH" in action_json["addLabelIds"]
        assert "L2" in action_json["removeLabelIds"]
        assert "INBOX" in action_json["removeLabelIds"]
        assert "UNREAD" in action_json["removeLabelIds"]
        assert action_json["forward"] == "fwd@test.com"

    @pytest.mark.asyncio
    async def test_filter_size_default_comparison(self, handler, gmail_state):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "f2"}
        mock_response.raise_for_status = MagicMock()
        mock_session.post = AsyncMock(return_value=mock_response)

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        ))

        criteria = {"size": 500}  # No size_comparison -> defaults to "larger"
        action = {"star": True}

        with patch(
            "aragora.server.handlers.features.gmail_labels.get_http_pool",
            return_value=mock_pool,
        ):
            await handler._api_create_filter(gmail_state, criteria, action)

        posted_json = mock_session.post.call_args[1]["json"]
        assert posted_json["criteria"]["sizeComparison"] == "larger"


# ---------------------------------------------------------------------------
# _api_create_label / _api_update_label mapping
# ---------------------------------------------------------------------------


class TestApiLabelMapping:
    """Tests for label API mapping details."""

    @pytest.mark.asyncio
    async def test_create_label_with_color(self, handler, gmail_state):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "L1", "name": "Colored"}
        mock_response.raise_for_status = MagicMock()
        mock_session.post = AsyncMock(return_value=mock_response)

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        ))

        options = {
            "name": "Colored",
            "background_color": "#ff0000",
            "text_color": "#00ff00",
            "label_list_visibility": "labelHide",
            "message_list_visibility": "hide",
        }

        with patch(
            "aragora.server.handlers.features.gmail_labels.get_http_pool",
            return_value=mock_pool,
        ):
            result = await handler._api_create_label(gmail_state, "Colored", options)

        posted_json = mock_session.post.call_args[1]["json"]
        assert posted_json["name"] == "Colored"
        assert posted_json["labelListVisibility"] == "labelHide"
        assert posted_json["messageListVisibility"] == "hide"
        assert posted_json["color"]["backgroundColor"] == "#ff0000"
        assert posted_json["color"]["textColor"] == "#00ff00"

    @pytest.mark.asyncio
    async def test_create_label_default_visibility(self, handler, gmail_state):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "L2"}
        mock_response.raise_for_status = MagicMock()
        mock_session.post = AsyncMock(return_value=mock_response)

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch(
            "aragora.server.handlers.features.gmail_labels.get_http_pool",
            return_value=mock_pool,
        ):
            await handler._api_create_label(gmail_state, "Simple", {})

        posted_json = mock_session.post.call_args[1]["json"]
        assert posted_json["labelListVisibility"] == "labelShow"
        assert posted_json["messageListVisibility"] == "show"
        assert "color" not in posted_json

    @pytest.mark.asyncio
    async def test_update_label_partial(self, handler, gmail_state):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "L1", "name": "New"}
        mock_response.raise_for_status = MagicMock()
        mock_session.patch = AsyncMock(return_value=mock_response)

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch(
            "aragora.server.handlers.features.gmail_labels.get_http_pool",
            return_value=mock_pool,
        ):
            result = await handler._api_update_label(
                gmail_state, "L1", {"name": "New"}
            )

        posted_json = mock_session.patch.call_args[1]["json"]
        assert posted_json == {"name": "New"}

    @pytest.mark.asyncio
    async def test_update_label_with_color(self, handler, gmail_state):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "L1"}
        mock_response.raise_for_status = MagicMock()
        mock_session.patch = AsyncMock(return_value=mock_response)

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch(
            "aragora.server.handlers.features.gmail_labels.get_http_pool",
            return_value=mock_pool,
        ):
            await handler._api_update_label(
                gmail_state, "L1", {"background_color": "#aabbcc"}
            )

        posted_json = mock_session.patch.call_args[1]["json"]
        assert posted_json["color"]["backgroundColor"] == "#aabbcc"
        assert posted_json["color"]["textColor"] == "#ffffff"  # default


# ---------------------------------------------------------------------------
# _api_delete_label / _api_delete_filter
# ---------------------------------------------------------------------------


class TestApiDelete:
    """Tests for delete API calls."""

    @pytest.mark.asyncio
    async def test_delete_label_api_url(self, handler, gmail_state):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_session.delete = AsyncMock(return_value=mock_response)

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch(
            "aragora.server.handlers.features.gmail_labels.get_http_pool",
            return_value=mock_pool,
        ):
            await handler._api_delete_label(gmail_state, "Label_42")

        url = mock_session.delete.call_args[0][0]
        assert "Label_42" in url
        assert "labels/Label_42" in url

    @pytest.mark.asyncio
    async def test_delete_filter_api_url(self, handler, gmail_state):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_session.delete = AsyncMock(return_value=mock_response)

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch(
            "aragora.server.handlers.features.gmail_labels.get_http_pool",
            return_value=mock_pool,
        ):
            await handler._api_delete_filter(gmail_state, "filter_99")

        url = mock_session.delete.call_args[0][0]
        assert "filter_99" in url
        assert "filters/filter_99" in url


# ---------------------------------------------------------------------------
# _api_list_filters
# ---------------------------------------------------------------------------


class TestApiListFilters:
    """Tests for _api_list_filters response parsing."""

    @pytest.mark.asyncio
    async def test_list_filters_parses_filter_key(self, handler, gmail_state):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "filter": [{"id": "f1"}, {"id": "f2"}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch(
            "aragora.server.handlers.features.gmail_labels.get_http_pool",
            return_value=mock_pool,
        ):
            result = await handler._api_list_filters(gmail_state)

        assert len(result) == 2
        assert result[0]["id"] == "f1"

    @pytest.mark.asyncio
    async def test_list_filters_empty_response(self, handler, gmail_state):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {}  # No "filter" key
        mock_response.raise_for_status = MagicMock()
        mock_session.get = AsyncMock(return_value=mock_response)

        mock_pool = MagicMock()
        mock_pool.get_session = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_session),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch(
            "aragora.server.handlers.features.gmail_labels.get_http_pool",
            return_value=mock_pool,
        ):
            result = await handler._api_list_filters(gmail_state)

        assert result == []
