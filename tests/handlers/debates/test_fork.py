"""Tests for fork and follow-up debate operations handler.

Tests the fork mixin endpoints including:
- POST /api/debates/{id}/fork - Create counterfactual fork
- POST /api/debates/{id}/verify-outcome - Verify debate outcome
- GET /api/debates/{id}/followup-suggestions - Get follow-up suggestions
- POST /api/debates/{id}/followup - Create follow-up debate
- GET /api/debates/{id}/forks - List debate forks
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest


class MockHandler:
    """Mock HTTP handler for tests."""

    def __init__(self, json_body: Optional[Dict[str, Any]] = None):
        self._json_body = json_body

    def get_json_body(self) -> Optional[Dict[str, Any]]:
        return self._json_body


class MockForkHandler:
    """Mock handler that includes ForkOperationsMixin methods."""

    def __init__(self, ctx: Dict[str, Any] = None):
        from aragora.server.handlers.debates.fork import ForkOperationsMixin

        self.ctx = ctx or {}
        self._storage = None
        self._nomic_dir = None

        # Inject mixin methods
        for method_name in dir(ForkOperationsMixin):
            if method_name.startswith("_") and not method_name.startswith("__"):
                method = getattr(ForkOperationsMixin, method_name)
                if callable(method):
                    # Bind the method to self
                    setattr(self, method_name, method.__get__(self, type(self)))

    def read_json_body(self, handler: Any, max_size: int = None) -> Optional[Dict]:
        if hasattr(handler, "_json_body"):
            return handler._json_body
        return None

    def get_storage(self):
        return self._storage

    def get_nomic_dir(self) -> Optional[Path]:
        return self._nomic_dir


@pytest.fixture
def fork_handler():
    """Create fork handler with mock context."""
    return MockForkHandler()


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state before each test."""
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass
    yield
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass


# =============================================================================
# Fork Debate Tests
# =============================================================================


class TestForkDebate:
    """Tests for debate forking endpoint."""

    def test_returns_400_without_body(self, fork_handler):
        """Returns 400 when request body is missing."""
        mock_handler = MockHandler(json_body=None)
        fork_handler._storage = MagicMock()

        # Get the underlying method
        original = fork_handler._fork_debate
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        result = original(fork_handler, mock_handler, "test-123")
        assert result.status_code == 400

    def test_returns_404_debate_not_found(self, fork_handler):
        """Returns 404 when debate doesn't exist."""
        mock_handler = MockHandler(json_body={"branch_point": 0})
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = None
        fork_handler._storage = mock_storage

        original = fork_handler._fork_debate
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        result = original(fork_handler, mock_handler, "nonexistent")
        assert result.status_code == 404

    def test_returns_400_invalid_branch_point(self, fork_handler):
        """Returns 400 when branch point exceeds message count."""
        mock_handler = MockHandler(json_body={"branch_point": 100})
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {
            "id": "test-123",
            "messages": [{"content": "msg1"}, {"content": "msg2"}],
        }
        fork_handler._storage = mock_storage

        original = fork_handler._fork_debate
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        result = original(fork_handler, mock_handler, "test-123")
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "branch point" in data.get("error", "").lower()

    def test_creates_fork_successfully(self, fork_handler):
        """Creates fork successfully with valid inputs."""
        mock_handler = MockHandler(
            json_body={"branch_point": 1, "modified_context": "New assumption"}
        )
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {
            "id": "test-123",
            "messages": [{"content": "msg1"}, {"content": "msg2"}, {"content": "msg3"}],
        }
        fork_handler._storage = mock_storage

        with tempfile.TemporaryDirectory() as tmpdir:
            fork_handler._nomic_dir = Path(tmpdir)

            original = fork_handler._fork_debate
            while hasattr(original, "__wrapped__"):
                original = original.__wrapped__

            result = original(fork_handler, mock_handler, "test-123")
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["success"] is True
            assert "branch_id" in data
            assert data["parent_debate_id"] == "test-123"
            assert data["branch_point"] == 1
            assert data["messages_inherited"] == 1


# =============================================================================
# Verify Outcome Tests
# =============================================================================


class TestVerifyOutcome:
    """Tests for outcome verification endpoint."""

    def test_returns_400_without_body(self, fork_handler):
        """Returns 400 when request body is missing."""
        mock_handler = MockHandler(json_body=None)

        original = fork_handler._verify_outcome
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        result = original(fork_handler, mock_handler, "test-123")
        assert result.status_code == 400

    def test_returns_503_without_position_tracker(self, fork_handler):
        """Returns 503 when position tracking not configured."""
        mock_handler = MockHandler(json_body={"correct": True})
        fork_handler._nomic_dir = None
        # Ensure no tracker in context
        fork_handler.ctx.pop("position_tracker", None)

        original = fork_handler._verify_outcome
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        with patch(
            "aragora.agents.truth_grounding.PositionTracker",
            side_effect=ImportError("not available"),
        ):
            # Module import fails
            result = original(fork_handler, mock_handler, "test-123")
            assert result.status_code in [503, 500]

    def test_verifies_with_context_tracker(self, fork_handler):
        """Uses position tracker from context when available."""
        mock_handler = MockHandler(json_body={"correct": True, "source": "manual"})
        mock_tracker = MagicMock()
        fork_handler.ctx["position_tracker"] = mock_tracker

        original = fork_handler._verify_outcome
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        result = original(fork_handler, mock_handler, "test-123")
        assert result.status_code == 200
        mock_tracker.record_verification.assert_called_once_with("test-123", True, "manual")


# =============================================================================
# Followup Suggestions Tests
# =============================================================================


class TestFollowupSuggestions:
    """Tests for followup suggestions endpoint."""

    def test_returns_404_debate_not_found(self, fork_handler):
        """Returns 404 when debate doesn't exist."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = None
        fork_handler._storage = mock_storage

        original = fork_handler._get_followup_suggestions
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        result = original(fork_handler, "nonexistent")
        assert result.status_code == 404

    def test_returns_empty_when_no_cruxes(self, fork_handler):
        """Returns empty suggestions when no cruxes found."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {
            "id": "test-123",
            "messages": [],
            "votes": [],
            "proposals": {},
            "uncertainty_metrics": {},
        }
        fork_handler._storage = mock_storage

        original = fork_handler._get_followup_suggestions
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        result = original(fork_handler, "test-123")
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["suggestions"] == []


# =============================================================================
# Create Followup Debate Tests
# =============================================================================


class TestCreateFollowupDebate:
    """Tests for create followup debate endpoint."""

    def test_returns_400_without_body(self, fork_handler):
        """Returns 400 when request body is missing."""
        mock_handler = MockHandler(json_body=None)
        fork_handler._storage = MagicMock()

        original = fork_handler._create_followup_debate
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        result = original(fork_handler, mock_handler, "test-123")
        assert result.status_code == 400

    def test_returns_400_without_crux_or_task(self, fork_handler):
        """Returns 400 when neither crux_id nor task provided."""
        mock_handler = MockHandler(json_body={})
        fork_handler._storage = MagicMock()

        original = fork_handler._create_followup_debate
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        result = original(fork_handler, mock_handler, "test-123")
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "crux_id" in data.get("error", "").lower() or "task" in data.get("error", "").lower()

    def test_returns_404_parent_not_found(self, fork_handler):
        """Returns 404 when parent debate doesn't exist."""
        mock_handler = MockHandler(json_body={"task": "Test task"})
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = None
        fork_handler._storage = mock_storage

        original = fork_handler._create_followup_debate
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        result = original(fork_handler, mock_handler, "nonexistent")
        assert result.status_code == 404

    def test_creates_followup_with_custom_task(self, fork_handler):
        """Creates followup debate with custom task."""
        mock_handler = MockHandler(
            json_body={"task": "Investigate the API design", "agents": ["claude", "grok"]}
        )
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {
            "id": "test-123",
            "agents": ["claude", "gemini"],
        }
        fork_handler._storage = mock_storage

        with tempfile.TemporaryDirectory() as tmpdir:
            fork_handler._nomic_dir = Path(tmpdir)

            original = fork_handler._create_followup_debate
            while hasattr(original, "__wrapped__"):
                original = original.__wrapped__

            result = original(fork_handler, mock_handler, "test-123")
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["success"] is True
            assert "followup_id" in data
            assert data["task"] == "Investigate the API design"
            assert data["agents"] == ["claude", "grok"]


# =============================================================================
# List Forks Tests
# =============================================================================


class TestListForks:
    """Tests for list forks endpoint."""

    def test_returns_empty_without_nomic_dir(self, fork_handler):
        """Returns empty list when nomic dir not configured."""
        fork_handler._storage = MagicMock()
        fork_handler._nomic_dir = None

        original = fork_handler._list_debate_forks
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        result = original(fork_handler, "test-123")
        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["forks"] == []
        assert data["total"] == 0

    def test_returns_empty_without_branches_dir(self, fork_handler):
        """Returns empty list when branches directory doesn't exist."""
        fork_handler._storage = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            fork_handler._nomic_dir = Path(tmpdir)
            # Don't create branches dir

            original = fork_handler._list_debate_forks
            while hasattr(original, "__wrapped__"):
                original = original.__wrapped__

            result = original(fork_handler, "test-123")
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["forks"] == []

    def test_lists_existing_forks(self, fork_handler):
        """Lists existing forks for a debate."""
        fork_handler._storage = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_path = Path(tmpdir)
            branches_dir = nomic_path / "branches"
            branches_dir.mkdir()

            # Create test fork files
            fork1_data = {
                "branch_id": "fork-test-123-r1-abc",
                "parent_debate_id": "test-123",
                "branch_point": 1,
                "status": "created",
            }
            fork2_data = {
                "branch_id": "fork-test-123-r2-def",
                "parent_debate_id": "test-123",
                "branch_point": 2,
                "status": "created",
            }

            (branches_dir / "fork-test-123-r1-abc.json").write_text(json.dumps(fork1_data))
            (branches_dir / "fork-test-123-r2-def.json").write_text(json.dumps(fork2_data))

            fork_handler._nomic_dir = nomic_path

            original = fork_handler._list_debate_forks
            while hasattr(original, "__wrapped__"):
                original = original.__wrapped__

            result = original(fork_handler, "test-123")
            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["total"] == 2
            assert len(data["forks"]) == 2


# =============================================================================
# Build Fork Tree Tests
# =============================================================================


class TestBuildForkTree:
    """Tests for fork tree building helper."""

    def test_builds_tree_from_flat_list(self):
        """Builds hierarchical tree from flat fork list."""
        from aragora.server.handlers.debates.fork import _build_fork_tree

        forks = [
            {"branch_id": "fork-1", "parent_debate_id": "root", "branch_point": 1},
            {"branch_id": "fork-2", "parent_debate_id": "root", "branch_point": 2},
            {"branch_id": "fork-1-1", "parent_debate_id": "fork-1", "branch_point": 1},
        ]

        tree = _build_fork_tree("root", forks)

        assert tree["id"] == "root"
        assert tree["type"] == "root"
        assert len(tree["children"]) == 2  # fork-1 and fork-2

        # Find fork-1 in children
        fork1 = next((c for c in tree["children"] if c["id"] == "fork-1"), None)
        assert fork1 is not None
        assert len(fork1["children"]) == 1  # fork-1-1

    def test_calculates_tree_stats(self):
        """Calculates total nodes and max depth."""
        from aragora.server.handlers.debates.fork import _build_fork_tree

        forks = [
            {"branch_id": "fork-1", "parent_debate_id": "root", "branch_point": 1},
            {"branch_id": "fork-1-1", "parent_debate_id": "fork-1", "branch_point": 1},
            {"branch_id": "fork-1-1-1", "parent_debate_id": "fork-1-1", "branch_point": 1},
        ]

        tree = _build_fork_tree("root", forks)

        assert tree["total_nodes"] == 4  # root + 3 forks
        assert tree["max_depth"] == 4  # root -> fork-1 -> fork-1-1 -> fork-1-1-1

    def test_handles_empty_fork_list(self):
        """Handles empty fork list gracefully."""
        from aragora.server.handlers.debates.fork import _build_fork_tree

        tree = _build_fork_tree("root", [])

        assert tree["id"] == "root"
        assert tree["type"] == "root"
        assert tree["children"] == []
        assert tree["total_nodes"] == 1
        assert tree["max_depth"] == 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in fork operations."""

    def test_fork_handles_storage_error(self, fork_handler):
        """Fork handles storage errors gracefully."""
        mock_handler = MockHandler(json_body={"branch_point": 1})
        mock_storage = MagicMock()
        mock_storage.get_debate.side_effect = RuntimeError("Database error")
        fork_handler._storage = mock_storage

        original = fork_handler._fork_debate
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        result = original(fork_handler, mock_handler, "test-123")
        assert result.status_code == 500

    def test_followup_handles_import_error(self, fork_handler):
        """Followup suggestions handles import errors."""
        mock_storage = MagicMock()
        mock_storage.get_debate.return_value = {"id": "test-123"}
        fork_handler._storage = mock_storage

        original = fork_handler._get_followup_suggestions
        while hasattr(original, "__wrapped__"):
            original = original.__wrapped__

        with patch(
            "aragora.uncertainty.estimator.DisagreementAnalyzer",
            side_effect=ImportError("Not available"),
        ):
            result = original(fork_handler, "test-123")
            # Should return 500 or handle gracefully
            assert result.status_code in [200, 500]
