"""
Tests for verification history API endpoints.

Tests:
- GET /api/verify/history - List verification history
- GET /api/verify/history/{id} - Get specific entry
- GET /api/verify/history/{id}/tree - Get proof tree
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import time

from aragora.server.handlers.verification.formal_verification import (
    FormalVerificationHandler,
    _verification_history,
    _add_to_history,
    _build_proof_tree,
    _cleanup_old_history,
    VerificationHistoryEntry,
    HISTORY_TTL_SECONDS,
)


def parse_response(result):
    """Parse HandlerResult body as JSON."""
    return json.loads(result.body.decode("utf-8"))


@pytest.fixture
def handler():
    """Create a handler instance for testing."""
    return FormalVerificationHandler({})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.path = "/api/verify/history"
    return handler


@pytest.fixture(autouse=True)
def clear_history():
    """Clear history before each test."""
    _verification_history.clear()
    yield
    _verification_history.clear()


class TestVerificationHistoryStorage:
    """Tests for the history storage functions."""

    def test_add_to_history(self):
        """Test adding entries to history."""
        entry_id = _add_to_history(
            claim="1 + 1 = 2",
            claim_type="MATHEMATICAL",
            context="arithmetic",
            result={"status": "proof_found", "is_verified": True},
        )

        assert entry_id is not None
        assert len(entry_id) == 16
        assert entry_id in _verification_history

        entry = _verification_history[entry_id]
        assert entry.claim == "1 + 1 = 2"
        assert entry.claim_type == "MATHEMATICAL"
        assert entry.result["is_verified"] is True

    def test_add_to_history_with_proof_tree(self):
        """Test adding entries with proof tree."""
        proof_tree = [
            {"id": "root", "type": "claim", "content": "test", "children": []},
        ]

        entry_id = _add_to_history(
            claim="test claim",
            claim_type=None,
            context="",
            result={"status": "proof_found"},
            proof_tree=proof_tree,
        )

        entry = _verification_history[entry_id]
        assert entry.proof_tree == proof_tree

    def test_history_eviction(self):
        """Test that old entries are evicted when limit is reached."""
        from aragora.server.handlers.verification.formal_verification import MAX_HISTORY_SIZE

        # Add entries up to the limit
        for i in range(MAX_HISTORY_SIZE + 10):
            _add_to_history(
                claim=f"claim {i}",
                claim_type=None,
                context="",
                result={"status": "proof_found"},
            )

        assert len(_verification_history) == MAX_HISTORY_SIZE

    def test_cleanup_old_history(self):
        """Test cleanup of old entries."""
        # Add an old entry
        old_entry = VerificationHistoryEntry(
            id="old_entry",
            claim="old claim",
            claim_type=None,
            context="",
            result={},
            timestamp=time.time() - HISTORY_TTL_SECONDS - 100,
        )
        _verification_history["old_entry"] = old_entry

        # Add a recent entry
        _add_to_history(
            claim="recent claim",
            claim_type=None,
            context="",
            result={},
        )

        _cleanup_old_history()

        assert "old_entry" not in _verification_history
        assert len(_verification_history) == 1


class TestBuildProofTree:
    """Tests for proof tree construction."""

    def test_build_proof_tree_verified(self):
        """Test building proof tree for verified claim."""
        result = {
            "claim": "1 + 1 = 2",
            "is_verified": True,
            "formal_statement": "theorem t : 1 + 1 = 2 := rfl",
            "language": "lean4",
            "status": "proof_found",
            "proof_hash": "abc123",
        }

        tree = _build_proof_tree(result)

        assert tree is not None
        assert len(tree) >= 3

        # Check root node
        root = next(n for n in tree if n["id"] == "root")
        assert root["type"] == "claim"
        assert "1 + 1 = 2" in root["content"]

        # Check translation node
        translation = next(n for n in tree if n["id"] == "translation")
        assert translation["type"] == "translation"
        assert translation["language"] == "lean4"

        # Check verification node
        verification = next(n for n in tree if n["id"] == "verification")
        assert verification["is_verified"] is True

    def test_build_proof_tree_not_verified(self):
        """Test building proof tree for unverified claim."""
        result = {
            "claim": "false claim",
            "is_verified": False,
        }

        tree = _build_proof_tree(result)
        assert tree is None

    def test_build_proof_tree_with_steps(self):
        """Test building proof tree with proof steps."""
        result = {
            "claim": "complex claim",
            "is_verified": True,
            "formal_statement": "theorem t : P := by simp",
            "language": "lean4",
            "status": "proof_found",
            "proof_steps": ["step 1", "step 2", "step 3"],
        }

        tree = _build_proof_tree(result)

        # Should have nodes for each step
        step_nodes = [n for n in tree if n["type"] == "proof_step"]
        assert len(step_nodes) == 3
        assert step_nodes[0]["step_number"] == 1


class TestHistoryEndpoints:
    """Tests for the history API endpoints."""

    @pytest.mark.asyncio
    async def test_get_history_empty(self, handler, mock_http_handler):
        """Test getting empty history."""
        result = await handler.handle_async(
            mock_http_handler,
            "GET",
            "/api/verify/history",
            query_params={},
        )

        assert result.status_code == 200
        data = parse_response(result)
        assert data["entries"] == []
        assert data["total"] == 0

    @pytest.mark.asyncio
    async def test_get_history_with_entries(self, handler, mock_http_handler):
        """Test getting history with entries."""
        # Add some entries
        _add_to_history("claim 1", None, "", {"status": "proof_found"})
        _add_to_history("claim 2", None, "", {"status": "translation_failed"})

        result = await handler.handle_async(
            mock_http_handler,
            "GET",
            "/api/verify/history",
            query_params={},
        )

        assert result.status_code == 200
        data = parse_response(result)
        assert len(data["entries"]) == 2
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_get_history_pagination(self, handler, mock_http_handler):
        """Test history pagination."""
        # Add 25 entries
        for i in range(25):
            _add_to_history(f"claim {i}", None, "", {"status": "proof_found"})

        # Get first page
        result = await handler.handle_async(
            mock_http_handler,
            "GET",
            "/api/verify/history",
            query_params={"limit": ["10"], "offset": ["0"]},
        )

        assert result.status_code == 200
        data = parse_response(result)
        assert len(data["entries"]) == 10
        assert data["total"] == 25
        assert data["limit"] == 10
        assert data["offset"] == 0

    @pytest.mark.asyncio
    async def test_get_history_status_filter(self, handler, mock_http_handler):
        """Test filtering history by status."""
        _add_to_history("claim 1", None, "", {"status": "proof_found"})
        _add_to_history("claim 2", None, "", {"status": "translation_failed"})
        _add_to_history("claim 3", None, "", {"status": "proof_found"})

        result = await handler.handle_async(
            mock_http_handler,
            "GET",
            "/api/verify/history",
            query_params={"status": ["proof_found"]},
        )

        assert result.status_code == 200
        data = parse_response(result)
        assert len(data["entries"]) == 2
        assert all(e["result"]["status"] == "proof_found" for e in data["entries"])

    @pytest.mark.asyncio
    async def test_get_history_entry(self, handler, mock_http_handler):
        """Test getting a specific history entry."""
        entry_id = _add_to_history(
            "test claim",
            "MATHEMATICAL",
            "test context",
            {"status": "proof_found", "is_verified": True},
        )

        result = await handler.handle_async(
            mock_http_handler,
            "GET",
            f"/api/verify/history/{entry_id}",
        )

        assert result.status_code == 200
        data = parse_response(result)
        assert data["id"] == entry_id
        assert data["claim"] == "test claim"
        assert data["claim_type"] == "MATHEMATICAL"

    @pytest.mark.asyncio
    async def test_get_history_entry_not_found(self, handler, mock_http_handler):
        """Test getting a non-existent history entry."""
        result = await handler.handle_async(
            mock_http_handler,
            "GET",
            "/api/verify/history/nonexistent123",
        )

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_get_proof_tree(self, handler, mock_http_handler):
        """Test getting proof tree for an entry."""
        proof_tree = [
            {"id": "root", "type": "claim", "content": "test", "children": ["t"]},
            {"id": "t", "type": "translation", "content": "formal", "children": []},
        ]

        entry_id = _add_to_history(
            "test claim",
            None,
            "",
            {"status": "proof_found", "is_verified": True},
            proof_tree=proof_tree,
        )

        result = await handler.handle_async(
            mock_http_handler,
            "GET",
            f"/api/verify/history/{entry_id}/tree",
        )

        assert result.status_code == 200
        data = parse_response(result)
        assert data["nodes"] == proof_tree

    @pytest.mark.asyncio
    async def test_get_proof_tree_builds_from_result(self, handler, mock_http_handler):
        """Test that proof tree is built from result if not stored."""
        entry_id = _add_to_history(
            "test claim",
            None,
            "",
            {
                "status": "proof_found",
                "is_verified": True,
                "formal_statement": "theorem t : P := by simp",
                "language": "lean4",
            },
            proof_tree=None,  # No stored tree
        )

        result = await handler.handle_async(
            mock_http_handler,
            "GET",
            f"/api/verify/history/{entry_id}/tree",
        )

        assert result.status_code == 200
        data = parse_response(result)
        assert len(data["nodes"]) > 0
        assert any(n["type"] == "claim" for n in data["nodes"])


class TestHistoryEntryToDict:
    """Tests for VerificationHistoryEntry.to_dict()."""

    def test_to_dict(self):
        """Test converting entry to dictionary."""
        entry = VerificationHistoryEntry(
            id="test123",
            claim="test claim",
            claim_type="LOGICAL",
            context="test context",
            result={"status": "proof_found"},
            timestamp=1704067200.0,  # 2024-01-01 00:00:00 UTC
            proof_tree=[{"id": "root"}],
        )

        result = entry.to_dict()

        assert result["id"] == "test123"
        assert result["claim"] == "test claim"
        assert result["claim_type"] == "LOGICAL"
        assert result["has_proof_tree"] is True
        assert "timestamp_iso" in result
