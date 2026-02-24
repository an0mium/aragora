"""Tests for consensus detection endpoints.

Validates:
- POST /api/v1/consensus/detect - Analyze proposals for consensus
- GET /api/v1/consensus/status/{debate_id} - Get consensus status for an existing debate
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.consensus import ConsensusHandler, _consensus_limiter


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset and bypass rate limiter to prevent xdist cross-test interference."""
    if hasattr(_consensus_limiter, "_buckets"):
        _consensus_limiter._buckets.clear()
    with patch(
        "aragora.server.handlers.consensus._consensus_limiter.is_allowed",
        return_value=True,
    ):
        yield


@pytest.fixture
def consensus_handler():
    """Create a consensus handler with mocked dependencies."""
    ctx = {"storage": MagicMock(), "elo_system": None, "nomic_dir": None}
    handler = ConsensusHandler(ctx)
    return handler


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with auth context."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
    }
    handler.command = "POST"
    return handler


@pytest.fixture
def mock_get_handler():
    """Create a mock HTTP handler for GET requests."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {}
    handler.command = "GET"
    return handler


class TestConsensusDetectionCanHandle:
    """Test that can_handle recognizes the new routes."""

    def test_can_handle_detect(self, consensus_handler):
        assert consensus_handler.can_handle("/api/v1/consensus/detect")

    def test_can_handle_detect_legacy(self, consensus_handler):
        assert consensus_handler.can_handle("/api/consensus/detect")

    def test_can_handle_status(self, consensus_handler):
        assert consensus_handler.can_handle("/api/v1/consensus/status/test-id-123")

    def test_can_handle_status_legacy(self, consensus_handler):
        assert consensus_handler.can_handle("/api/consensus/status/test-id-123")


class TestConsensusDetect:
    """Test POST /api/v1/consensus/detect endpoint."""

    def _make_body(self, task="Choose a database", proposals=None, threshold=0.7):
        if proposals is None:
            proposals = [
                {"agent": "claude", "content": "Use PostgreSQL for reliability"},
                {"agent": "gpt-4", "content": "Use PostgreSQL for scalability and reliability"},
            ]
        return json.dumps({"task": task, "proposals": proposals, "threshold": threshold}).encode()

    def _setup_handler(self, mock_http_handler, body_bytes):
        mock_http_handler.headers["Content-Length"] = str(len(body_bytes))
        mock_http_handler.rfile = MagicMock()
        mock_http_handler.rfile.read.return_value = body_bytes
        return mock_http_handler

    @patch("aragora.server.handlers.consensus.ConsensusHandler.require_auth_or_error")
    @patch("aragora.server.handlers.consensus.ConsensusHandler.require_permission_or_error")
    def test_detect_consensus_success(
        self, mock_perm, mock_auth, consensus_handler, mock_http_handler
    ):
        """Test successful consensus detection."""
        mock_auth.return_value = (MagicMock(), None)
        mock_perm.return_value = (MagicMock(), None)

        body_bytes = self._make_body()
        self._setup_handler(mock_http_handler, body_bytes)

        result = consensus_handler.handle_post("/api/v1/consensus/detect", {}, mock_http_handler)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert "data" in data
        inner = data["data"]
        assert "consensus_reached" in inner
        assert "confidence" in inner
        assert "proof" in inner
        assert "checksum" in inner
        assert inner["debate_id"].startswith("detect-")

    @patch("aragora.server.handlers.consensus.ConsensusHandler.require_auth_or_error")
    @patch("aragora.server.handlers.consensus.ConsensusHandler.require_permission_or_error")
    def test_detect_missing_task(self, mock_perm, mock_auth, consensus_handler, mock_http_handler):
        """Test detection with missing task returns 400."""
        mock_auth.return_value = (MagicMock(), None)
        mock_perm.return_value = (MagicMock(), None)

        body_bytes = self._make_body(task="")
        self._setup_handler(mock_http_handler, body_bytes)

        result = consensus_handler.handle_post("/api/v1/consensus/detect", {}, mock_http_handler)

        assert result is not None
        assert result.status == 400

    @patch("aragora.server.handlers.consensus.ConsensusHandler.require_auth_or_error")
    @patch("aragora.server.handlers.consensus.ConsensusHandler.require_permission_or_error")
    def test_detect_empty_proposals(
        self, mock_perm, mock_auth, consensus_handler, mock_http_handler
    ):
        """Test detection with empty proposals returns 400."""
        mock_auth.return_value = (MagicMock(), None)
        mock_perm.return_value = (MagicMock(), None)

        body_bytes = self._make_body(proposals=[])
        self._setup_handler(mock_http_handler, body_bytes)

        result = consensus_handler.handle_post("/api/v1/consensus/detect", {}, mock_http_handler)

        assert result is not None
        assert result.status == 400

    @patch("aragora.server.handlers.consensus.ConsensusHandler.require_auth_or_error")
    @patch("aragora.server.handlers.consensus.ConsensusHandler.require_permission_or_error")
    def test_detect_invalid_threshold(
        self, mock_perm, mock_auth, consensus_handler, mock_http_handler
    ):
        """Test detection with invalid threshold returns 400."""
        mock_auth.return_value = (MagicMock(), None)
        mock_perm.return_value = (MagicMock(), None)

        body_bytes = self._make_body(threshold=2.0)
        self._setup_handler(mock_http_handler, body_bytes)

        result = consensus_handler.handle_post("/api/v1/consensus/detect", {}, mock_http_handler)

        assert result is not None
        assert result.status == 400

    @patch("aragora.server.handlers.consensus.ConsensusHandler.require_auth_or_error")
    @patch("aragora.server.handlers.consensus.ConsensusHandler.require_permission_or_error")
    def test_detect_high_agreement(
        self, mock_perm, mock_auth, consensus_handler, mock_http_handler
    ):
        """Test detection with very similar proposals reaches consensus."""
        mock_auth.return_value = (MagicMock(), None)
        mock_perm.return_value = (MagicMock(), None)

        proposals = [
            {"agent": "a1", "content": "We should use PostgreSQL database for the project"},
            {"agent": "a2", "content": "PostgreSQL database should be used for this project"},
            {
                "agent": "a3",
                "content": "Use PostgreSQL database is the right choice for the project",
            },
        ]
        body_bytes = self._make_body(proposals=proposals, threshold=0.3)
        self._setup_handler(mock_http_handler, body_bytes)

        result = consensus_handler.handle_post("/api/v1/consensus/detect", {}, mock_http_handler)

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)["data"]
        assert data["consensus_reached"] is True
        assert data["confidence"] > 0.3

    @patch("aragora.server.handlers.consensus.ConsensusHandler.require_auth_or_error")
    def test_detect_requires_auth(self, mock_auth, consensus_handler, mock_http_handler):
        """Test detection requires authentication."""
        from aragora.server.handlers.base import error_response

        mock_auth.return_value = (None, error_response("Authentication required", 401))

        body_bytes = self._make_body()
        self._setup_handler(mock_http_handler, body_bytes)

        result = consensus_handler.handle_post("/api/v1/consensus/detect", {}, mock_http_handler)

        assert result is not None
        assert result.status == 401


class TestConsensusStatus:
    """Test GET /api/v1/consensus/status/{debate_id} endpoint."""

    def test_status_debate_not_found(self, consensus_handler, mock_get_handler):
        """Test status for non-existent debate returns 404."""
        consensus_handler.ctx["storage"].get_debate.return_value = None

        result = consensus_handler.handle(
            "/api/v1/consensus/status/nonexistent-123", {}, mock_get_handler
        )

        assert result is not None
        assert result.status == 404

    def test_status_no_storage(self, mock_get_handler):
        """Test status without storage returns 503."""
        handler = ConsensusHandler(ctx={"storage": None})

        result = handler.handle("/api/v1/consensus/status/test-123", {}, mock_get_handler)

        assert result is not None
        assert result.status == 503

    @patch("aragora.server.handlers.consensus.ConsensusHandler.require_auth_or_error")
    def test_status_with_debate_result(self, mock_auth, consensus_handler, mock_get_handler):
        """Test status with a valid debate result."""
        # Create a mock debate result
        mock_result = MagicMock()
        mock_result.id = "test-debate-123"
        mock_result.task = "Choose a database"
        mock_result.final_answer = "Use PostgreSQL"
        mock_result.confidence = 0.85
        mock_result.consensus_reached = True
        mock_result.rounds_completed = 3
        mock_result.messages = []
        mock_result.critiques = []
        mock_result.participants = ["claude", "gpt-4"]
        mock_result.dissenting_views = []
        mock_result.debate_cruxes = []

        consensus_handler.ctx["storage"].get_debate.return_value = mock_result

        result = consensus_handler.handle(
            "/api/v1/consensus/status/test-debate-123", {}, mock_get_handler
        )

        assert result is not None
        assert result.status == 200
        data = json.loads(result.body)
        assert "data" in data
        inner = data["data"]
        assert inner["debate_id"] == "test-debate-123"
        assert inner["consensus_reached"] is True
        assert "proof" in inner
        assert "partial_consensus" in inner
        assert "checksum" in inner
