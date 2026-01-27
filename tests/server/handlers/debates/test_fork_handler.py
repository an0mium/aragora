"""Tests for Fork Operations handler mixin."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.debates.fork import (
    ForkOperationsMixin,
    _build_fork_tree,
)


def parse_result(result):
    """Parse HandlerResult into (body_dict, status_code) for easier testing."""
    body = json.loads(result.body) if result.body else {}
    return body, result.status_code


# =============================================================================
# Fixtures
# =============================================================================


class MockDebatesHandler(ForkOperationsMixin):
    """Mock handler that implements the protocol for testing."""

    def __init__(self, ctx=None, storage=None, nomic_dir=None):
        self.ctx = ctx or {}
        self._storage = storage
        self._nomic_dir = nomic_dir

    def read_json_body(self, handler, max_size=None):
        """Read JSON body from mock handler."""
        return getattr(handler, "_body", None)

    def get_storage(self):
        """Get storage instance."""
        return self._storage

    def get_nomic_dir(self):
        """Get nomic directory."""
        return self._nomic_dir


@pytest.fixture
def mock_storage():
    """Create a mock storage instance."""
    storage = MagicMock()
    storage.get_debate.return_value = {
        "id": "test-debate-123",
        "task": "Test debate task",
        "agents": ["claude", "gpt4"],
        "messages": [
            {"agent": "claude", "content": "Round 1", "round": 1},
            {"agent": "gpt4", "content": "Round 1", "round": 1},
            {"agent": "claude", "content": "Round 2", "round": 2},
        ],
        "status": "complete",
    }
    return storage


@pytest.fixture
def temp_nomic_dir():
    """Create a temporary nomic directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def handler(mock_storage, temp_nomic_dir):
    """Create a handler instance with mocks."""
    return MockDebatesHandler(
        ctx={},
        storage=mock_storage,
        nomic_dir=temp_nomic_dir,
    )


@pytest.fixture
def mock_handler_request():
    """Create a mock HTTP handler with request body."""
    handler = MagicMock()
    handler._body = {"branch_point": 2, "modified_context": "What if we assumed X?"}
    return handler


# =============================================================================
# Test _build_fork_tree
# =============================================================================


class TestBuildForkTree:
    """Tests for fork tree building."""

    def test_build_tree_no_forks(self):
        """Should return root node with no children."""
        tree = _build_fork_tree("root-123", [])

        assert tree["id"] == "root-123"
        assert tree["type"] == "root"
        assert tree["children"] == []
        assert tree["total_nodes"] == 1
        assert tree["max_depth"] == 1

    def test_build_tree_single_fork(self):
        """Should build tree with one fork."""
        forks = [
            {
                "branch_id": "fork-1",
                "parent_debate_id": "root-123",
                "branch_point": 2,
                "pivot_claim": "Test claim",
            }
        ]
        tree = _build_fork_tree("root-123", forks)

        assert tree["id"] == "root-123"
        assert len(tree["children"]) == 1
        assert tree["children"][0]["id"] == "fork-1"
        assert tree["children"][0]["branch_point"] == 2
        assert tree["total_nodes"] == 2
        assert tree["max_depth"] == 2

    def test_build_tree_nested_forks(self):
        """Should build tree with nested forks."""
        forks = [
            {
                "branch_id": "fork-1",
                "parent_debate_id": "root-123",
                "branch_point": 2,
            },
            {
                "branch_id": "fork-2",
                "parent_debate_id": "fork-1",
                "branch_point": 3,
            },
        ]
        tree = _build_fork_tree("root-123", forks)

        assert tree["total_nodes"] == 3
        assert tree["max_depth"] == 3
        assert tree["children"][0]["children"][0]["id"] == "fork-2"

    def test_build_tree_multiple_branches(self):
        """Should build tree with multiple branches at same level."""
        forks = [
            {
                "branch_id": "fork-1",
                "parent_debate_id": "root-123",
                "branch_point": 2,
            },
            {
                "branch_id": "fork-2",
                "parent_debate_id": "root-123",
                "branch_point": 3,
            },
        ]
        tree = _build_fork_tree("root-123", forks)

        assert len(tree["children"]) == 2
        assert tree["total_nodes"] == 3
        assert tree["max_depth"] == 2


# =============================================================================
# Test Fork Debate
# =============================================================================


class TestForkDebate:
    """Tests for debate forking."""

    def test_fork_debate_no_body(self, handler):
        """Should return error when no body provided."""
        mock_request = MagicMock()
        mock_request._body = None

        # Call the unwrapped method directly (bypasses RBAC decorator)
        result = ForkOperationsMixin._fork_debate.__wrapped__(handler, mock_request, "test-123")
        body, status = parse_result(result)

        assert status == 400
        assert "Invalid" in body.get("error", "")

    def test_fork_debate_branch_point_exceeds_messages(self, handler, mock_storage):
        """Should return error when branch point exceeds message count."""
        mock_request = MagicMock()
        mock_request._body = {"branch_point": 100}

        # Mock validation - must patch where it's imported, not where it's defined
        with patch("aragora.server.validation.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)

            result = ForkOperationsMixin._fork_debate.__wrapped__(
                handler, mock_request, "test-debate-123"
            )
            body, status = parse_result(result)

        assert status == 400
        assert "exceeds" in body.get("error", "")

    def test_fork_debate_not_found(self, handler, mock_storage):
        """Should return 404 when debate not found."""
        mock_storage.get_debate.return_value = None
        mock_request = MagicMock()
        mock_request._body = {"branch_point": 1}

        with patch("aragora.server.validation.validate_against_schema") as mock_validate:
            mock_validate.return_value = MagicMock(is_valid=True)

            result = ForkOperationsMixin._fork_debate.__wrapped__(
                handler, mock_request, "nonexistent"
            )
            body, status = parse_result(result)

        assert status == 404


# =============================================================================
# Test Verify Outcome
# =============================================================================


class TestVerifyOutcome:
    """Tests for outcome verification."""

    def test_verify_outcome_no_body(self, handler):
        """Should return error when no body provided."""
        mock_request = MagicMock()
        mock_request._body = None

        result = ForkOperationsMixin._verify_outcome.__wrapped__(handler, mock_request, "test-123")
        body, status = parse_result(result)

        assert status == 400

    def test_verify_outcome_with_tracker(self, handler):
        """Should verify outcome when position tracker available."""
        mock_tracker = MagicMock()
        handler.ctx["position_tracker"] = mock_tracker

        mock_request = MagicMock()
        mock_request._body = {"correct": True, "source": "ground_truth"}

        result = ForkOperationsMixin._verify_outcome.__wrapped__(handler, mock_request, "test-123")
        body, status = parse_result(result)

        assert status == 200
        assert body["status"] == "verified"
        mock_tracker.record_verification.assert_called_once_with("test-123", True, "ground_truth")


# =============================================================================
# Test List Forks
# =============================================================================


class TestListForks:
    """Tests for listing debate forks."""

    def test_list_forks_no_nomic_dir(self, mock_storage):
        """Should return empty when no nomic dir."""
        handler = MockDebatesHandler(storage=mock_storage, nomic_dir=None)

        result = ForkOperationsMixin._list_debate_forks.__wrapped__(handler, "test-123")
        body, status = parse_result(result)

        assert status == 200
        assert body["forks"] == []
        assert body["total"] == 0

    def test_list_forks_no_branches_dir(self, handler, temp_nomic_dir):
        """Should return empty when no branches directory."""
        result = ForkOperationsMixin._list_debate_forks.__wrapped__(handler, "test-123")
        body, status = parse_result(result)

        assert status == 200
        assert body["forks"] == []

    def test_list_forks_with_data(self, handler, temp_nomic_dir):
        """Should list forks from branches directory."""
        # Create branches directory and add a fork file
        branches_dir = temp_nomic_dir / "branches"
        branches_dir.mkdir()

        fork_data = {
            "branch_id": "fork-test-123-r2-abc12345",
            "parent_debate_id": "test-123",
            "branch_point": 2,
            "status": "created",
        }
        fork_file = branches_dir / "fork-test-123-r2-abc12345.json"
        with open(fork_file, "w") as f:
            json.dump(fork_data, f)

        result = ForkOperationsMixin._list_debate_forks.__wrapped__(handler, "test-123")
        body, status = parse_result(result)

        assert status == 200
        assert body["total"] == 1
        assert len(body["forks"]) == 1
        assert body["forks"][0]["branch_id"] == "fork-test-123-r2-abc12345"


# =============================================================================
# Test Get Followup Suggestions
# =============================================================================


class TestGetFollowupSuggestions:
    """Tests for followup suggestions."""

    def test_followup_suggestions_not_found(self, handler, mock_storage):
        """Should return 404 when debate not found."""
        mock_storage.get_debate.return_value = None

        result = ForkOperationsMixin._get_followup_suggestions.__wrapped__(handler, "nonexistent")
        body, status = parse_result(result)

        assert status == 404

    def test_followup_suggestions_no_cruxes(self, handler, mock_storage):
        """Should return empty suggestions when no cruxes."""
        mock_storage.get_debate.return_value = {
            "id": "test-123",
            "messages": [],
            "votes": [],
            "proposals": {},
        }

        with patch("aragora.uncertainty.estimator.DisagreementAnalyzer") as MockAnalyzer:
            mock_analyzer = MockAnalyzer.return_value
            mock_analyzer.analyze_disagreement.return_value = MagicMock(cruxes=[])

            result = ForkOperationsMixin._get_followup_suggestions.__wrapped__(handler, "test-123")
            body, status = parse_result(result)

        assert status == 200
        assert body["suggestions"] == []


# =============================================================================
# Test Create Followup Debate
# =============================================================================


class TestCreateFollowupDebate:
    """Tests for creating followup debates."""

    def test_create_followup_no_body(self, handler):
        """Should return error when no body provided."""
        mock_request = MagicMock()
        mock_request._body = None

        result = ForkOperationsMixin._create_followup_debate.__wrapped__(
            handler, mock_request, "test-123"
        )
        body, status = parse_result(result)

        assert status == 400

    def test_create_followup_no_crux_or_task(self, handler):
        """Should return error when neither crux_id nor task provided."""
        mock_request = MagicMock()
        mock_request._body = {}

        result = ForkOperationsMixin._create_followup_debate.__wrapped__(
            handler, mock_request, "test-123"
        )
        body, status = parse_result(result)

        assert status == 400
        assert "Either crux_id or task" in body.get("error", "")

    def test_create_followup_parent_not_found(self, handler, mock_storage):
        """Should return 404 when parent debate not found."""
        mock_storage.get_debate.return_value = None
        mock_request = MagicMock()
        mock_request._body = {"task": "Follow-up task"}

        result = ForkOperationsMixin._create_followup_debate.__wrapped__(
            handler, mock_request, "nonexistent"
        )
        body, status = parse_result(result)

        assert status == 404

    def test_create_followup_with_custom_task(self, handler, mock_storage, temp_nomic_dir):
        """Should create followup with custom task."""
        mock_request = MagicMock()
        mock_request._body = {
            "task": "Custom followup task",
            "agents": ["claude", "gemini"],
        }

        result = ForkOperationsMixin._create_followup_debate.__wrapped__(
            handler, mock_request, "test-debate-123"
        )
        body, status = parse_result(result)

        assert status == 200
        assert body["success"] is True
        assert body["task"] == "Custom followup task"
        assert body["agents"] == ["claude", "gemini"]
        assert "followup_id" in body

        # Verify file was created
        followups_dir = temp_nomic_dir / "followups"
        assert followups_dir.exists()
        followup_files = list(followups_dir.glob("followup-*.json"))
        assert len(followup_files) == 1
