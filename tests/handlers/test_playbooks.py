"""
Tests for PlaybookHandler — REST endpoints for decision playbooks.

Tests cover:
- GET /api/v1/playbooks - List playbooks
- GET /api/v1/playbooks/{id} - Get playbook details
- POST /api/v1/playbooks/{id}/run - Run a playbook
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.playbooks import PlaybookHandler
from aragora.playbooks.models import Playbook, PlaybookStep, ApprovalGate
from aragora.playbooks.registry import PlaybookRegistry


@pytest.fixture
def handler():
    return PlaybookHandler()


@pytest.fixture
def mock_registry():
    """Create a mock registry with test playbooks."""
    registry = PlaybookRegistry()
    registry._loaded_builtins = True  # Skip auto-loading
    registry.register(Playbook(
        id="test_pb",
        name="Test Playbook",
        description="A test playbook",
        category="general",
        steps=[PlaybookStep(name="s1", action="debate")],
        tags=["test"],
    ))
    registry.register(Playbook(
        id="finance_pb",
        name="Finance Playbook",
        description="Finance workflow",
        category="finance",
        steps=[PlaybookStep(name="s1", action="review")],
        tags=["finance"],
    ))
    return registry


def _make_handler_with_body(body: dict) -> MagicMock:
    handler = MagicMock()
    handler.headers = {"Content-Length": str(len(json.dumps(body)))}
    handler.rfile.read.return_value = json.dumps(body).encode()
    return handler


def _make_handler_with_query(query: str = "") -> MagicMock:
    handler = MagicMock()
    parsed_url = MagicMock()
    parsed_url.query = query
    handler.parsed_url = parsed_url
    return handler


class TestCanHandle:

    def test_get_list(self, handler):
        assert handler.can_handle("GET", "/api/v1/playbooks") is True

    def test_get_single(self, handler):
        assert handler.can_handle("GET", "/api/v1/playbooks/test") is True

    def test_post_run(self, handler):
        assert handler.can_handle("POST", "/api/v1/playbooks/test/run") is True

    def test_delete_not_handled(self, handler):
        assert handler.can_handle("DELETE", "/api/v1/playbooks/test") is False


class TestListPlaybooks:

    @patch("aragora.playbooks.registry.get_playbook_registry")
    def test_list_all(self, mock_get_registry, handler, mock_registry):
        mock_get_registry.return_value = mock_registry
        mock_handler = _make_handler_with_query("")

        result = handler._handle_list_playbooks(mock_handler)
        body = json.loads(result["body"])
        assert body["count"] == 2

    @patch("aragora.playbooks.registry.get_playbook_registry")
    def test_list_by_category(self, mock_get_registry, handler, mock_registry):
        mock_get_registry.return_value = mock_registry
        mock_handler = _make_handler_with_query("category=finance")

        result = handler._handle_list_playbooks(mock_handler)
        body = json.loads(result["body"])
        assert body["count"] == 1
        assert body["playbooks"][0]["id"] == "finance_pb"


class TestGetPlaybook:

    @patch("aragora.playbooks.registry.get_playbook_registry")
    def test_get_existing(self, mock_get_registry, handler, mock_registry):
        mock_get_registry.return_value = mock_registry
        mock_handler = MagicMock()

        result = handler._handle_get_playbook("/api/v1/playbooks/test_pb", mock_handler)
        body = json.loads(result["body"])
        assert body["id"] == "test_pb"
        assert body["name"] == "Test Playbook"

    @patch("aragora.playbooks.registry.get_playbook_registry")
    def test_get_not_found(self, mock_get_registry, handler, mock_registry):
        mock_get_registry.return_value = mock_registry
        mock_handler = MagicMock()

        result = handler._handle_get_playbook("/api/v1/playbooks/nonexistent", mock_handler)
        assert result["status"] == 404


class TestRunPlaybook:

    @patch("aragora.playbooks.registry.get_playbook_registry")
    def test_run_success(self, mock_get_registry, handler, mock_registry):
        mock_get_registry.return_value = mock_registry
        mock_handler = _make_handler_with_body({"input": "Evaluate vendor X"})

        result = handler._handle_run_playbook("/api/v1/playbooks/test_pb/run", mock_handler)
        assert result["status"] == 202
        body = json.loads(result["body"])
        assert body["playbook_id"] == "test_pb"
        assert body["status"] == "queued"
        assert "run_id" in body

    @patch("aragora.playbooks.registry.get_playbook_registry")
    def test_run_not_found(self, mock_get_registry, handler, mock_registry):
        mock_get_registry.return_value = mock_registry
        mock_handler = _make_handler_with_body({"input": "test"})

        result = handler._handle_run_playbook("/api/v1/playbooks/nonexistent/run", mock_handler)
        assert result["status"] == 404

    @patch("aragora.playbooks.registry.get_playbook_registry")
    def test_run_missing_input(self, mock_get_registry, handler, mock_registry):
        mock_get_registry.return_value = mock_registry
        mock_handler = _make_handler_with_body({})

        result = handler._handle_run_playbook("/api/v1/playbooks/test_pb/run", mock_handler)
        assert result["status"] == 400
