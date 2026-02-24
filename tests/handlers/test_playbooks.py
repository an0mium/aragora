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
    registry.register(
        Playbook(
            id="test_pb",
            name="Test Playbook",
            description="A test playbook",
            category="general",
            steps=[PlaybookStep(name="s1", action="debate")],
            tags=["test"],
        )
    )
    registry.register(
        Playbook(
            id="finance_pb",
            name="Finance Playbook",
            description="Finance workflow",
            category="finance",
            steps=[PlaybookStep(name="s1", action="review")],
            tags=["finance"],
        )
    )
    return registry


class TestCanHandle:
    def test_get_list(self, handler):
        assert handler.can_handle("/api/v1/playbooks") is True

    def test_get_single(self, handler):
        assert handler.can_handle("/api/v1/playbooks/test") is True

    def test_post_run(self, handler):
        assert handler.can_handle("/api/v1/playbooks/test/run") is True

    def test_delete_not_handled(self, handler):
        assert handler.can_handle("/api/v1/other") is False


class TestListPlaybooks:
    @patch("aragora.playbooks.registry.get_playbook_registry")
    def test_list_all(self, mock_get_registry, handler, mock_registry):
        mock_get_registry.return_value = mock_registry

        result = handler._list_playbooks({})
        body = json.loads(result["body"])
        assert body["count"] == 2

    @patch("aragora.playbooks.registry.get_playbook_registry")
    def test_list_by_category(self, mock_get_registry, handler, mock_registry):
        mock_get_registry.return_value = mock_registry

        result = handler._list_playbooks({"category": "finance"})
        body = json.loads(result["body"])
        assert body["count"] == 1
        assert body["playbooks"][0]["id"] == "finance_pb"


class TestGetPlaybook:
    @patch("aragora.playbooks.registry.get_playbook_registry")
    def test_get_existing(self, mock_get_registry, handler, mock_registry):
        mock_get_registry.return_value = mock_registry

        result = handler._get_playbook("/api/playbooks/test_pb")
        body = json.loads(result["body"])
        assert body["id"] == "test_pb"
        assert body["name"] == "Test Playbook"

    @patch("aragora.playbooks.registry.get_playbook_registry")
    def test_get_not_found(self, mock_get_registry, handler, mock_registry):
        mock_get_registry.return_value = mock_registry

        result = handler._get_playbook("/api/playbooks/nonexistent")
        assert result["status"] == 404


class TestRunPlaybook:
    @patch("aragora.playbooks.registry.get_playbook_registry")
    def test_run_success(self, mock_get_registry, handler, mock_registry):
        mock_get_registry.return_value = mock_registry

        result = handler._run_playbook("/api/playbooks/test_pb/run", {"input": "Evaluate vendor X"})
        assert result["status"] == 202
        body = json.loads(result["body"])
        assert body["playbook_id"] == "test_pb"
        assert body["status"] == "queued"
        assert "run_id" in body

    @patch("aragora.playbooks.registry.get_playbook_registry")
    def test_run_not_found(self, mock_get_registry, handler, mock_registry):
        mock_get_registry.return_value = mock_registry

        result = handler._run_playbook("/api/playbooks/nonexistent/run", {"input": "test"})
        assert result["status"] == 404

    @patch("aragora.playbooks.registry.get_playbook_registry")
    def test_run_missing_input(self, mock_get_registry, handler, mock_registry):
        mock_get_registry.return_value = mock_registry

        result = handler._run_playbook("/api/playbooks/test_pb/run", {})
        assert result["status"] == 400
