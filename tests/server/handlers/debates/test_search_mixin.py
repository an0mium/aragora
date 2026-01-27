"""Tests for search operations handler mixin."""

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.debates.search import SearchOperationsMixin


# =============================================================================
# Test Fixtures
# =============================================================================


class MockDebatesHandler(SearchOperationsMixin):
    """Mock debates handler with search mixin."""

    def __init__(self, storage=None):
        self._storage = storage
        self.ctx = {}

    def get_storage(self):
        return self._storage


# =============================================================================
# Test Search Debates
# =============================================================================


class TestSearchDebates:
    """Tests for debate search endpoint."""

    def test_search_no_storage(self):
        """Should return error when storage not configured."""
        handler = MockDebatesHandler(storage=None)

        # Bypass RBAC for testing
        with patch(
            "aragora.server.handlers.debates.search.require_permission",
            lambda p: lambda f: f,
        ):
            with patch(
                "aragora.server.handlers.debates.search.rate_limit",
                lambda **k: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.debates.search.require_storage",
                    lambda f: f,
                ):
                    with patch(
                        "aragora.server.handlers.debates.search.ttl_cache",
                        lambda **k: lambda f: f,
                    ):
                        result = handler._search_debates("test", 10, 0)

        # Should handle gracefully
        assert result is not None

    def test_search_empty_query(self):
        """Should list recent debates when query is empty."""
        mock_storage = MagicMock()
        mock_storage.list_recent.return_value = [
            {"id": "debate-1", "task": "Test 1", "status": "active"},
            {"id": "debate-2", "task": "Test 2", "status": "concluded"},
        ]

        handler = MockDebatesHandler(storage=mock_storage)

        with patch(
            "aragora.server.handlers.debates.search.require_permission",
            lambda p: lambda f: f,
        ):
            with patch(
                "aragora.server.handlers.debates.search.rate_limit",
                lambda **k: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.debates.search.require_storage",
                    lambda f: f,
                ):
                    with patch(
                        "aragora.server.handlers.debates.search.ttl_cache",
                        lambda **k: lambda f: f,
                    ):
                        result = handler._search_debates("", 10, 0)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "results" in body
        assert len(body["results"]) == 2
        mock_storage.list_recent.assert_called_once()

    def test_search_with_query(self):
        """Should search debates when query provided."""
        mock_storage = MagicMock()
        mock_storage.search.return_value = (
            [{"id": "debate-1", "task": "Microservices debate", "status": "active"}],
            1,
        )

        handler = MockDebatesHandler(storage=mock_storage)

        with patch(
            "aragora.server.handlers.debates.search.require_permission",
            lambda p: lambda f: f,
        ):
            with patch(
                "aragora.server.handlers.debates.search.rate_limit",
                lambda **k: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.debates.search.require_storage",
                    lambda f: f,
                ):
                    with patch(
                        "aragora.server.handlers.debates.search.ttl_cache",
                        lambda **k: lambda f: f,
                    ):
                        result = handler._search_debates("microservices", 10, 0)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["query"] == "microservices"
        assert body["total"] == 1
        mock_storage.search.assert_called_once_with(
            query="microservices", limit=10, offset=0, org_id=None
        )

    def test_search_with_pagination(self):
        """Should handle pagination correctly."""
        mock_storage = MagicMock()
        mock_storage.search.return_value = (
            [{"id": "debate-3", "task": "Test 3"}],
            25,  # Total of 25 results
        )

        handler = MockDebatesHandler(storage=mock_storage)

        with patch(
            "aragora.server.handlers.debates.search.require_permission",
            lambda p: lambda f: f,
        ):
            with patch(
                "aragora.server.handlers.debates.search.rate_limit",
                lambda **k: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.debates.search.require_storage",
                    lambda f: f,
                ):
                    with patch(
                        "aragora.server.handlers.debates.search.ttl_cache",
                        lambda **k: lambda f: f,
                    ):
                        result = handler._search_debates("test", 10, 20)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["offset"] == 20
        assert body["limit"] == 10
        assert body["total"] == 25
        assert body["has_more"] is True  # 20 + 1 < 25

    def test_search_with_org_filter(self):
        """Should filter by organization when org_id provided."""
        mock_storage = MagicMock()
        mock_storage.search.return_value = (
            [{"id": "debate-1", "task": "Test", "org_id": "org-123"}],
            1,
        )

        handler = MockDebatesHandler(storage=mock_storage)

        with patch(
            "aragora.server.handlers.debates.search.require_permission",
            lambda p: lambda f: f,
        ):
            with patch(
                "aragora.server.handlers.debates.search.rate_limit",
                lambda **k: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.debates.search.require_storage",
                    lambda f: f,
                ):
                    with patch(
                        "aragora.server.handlers.debates.search.ttl_cache",
                        lambda **k: lambda f: f,
                    ):
                        result = handler._search_debates("test", 10, 0, org_id="org-123")

        assert result.status_code == 200
        mock_storage.search.assert_called_once_with(
            query="test", limit=10, offset=0, org_id="org-123"
        )

    def test_search_database_error(self):
        """Should handle database errors gracefully."""
        from aragora.exceptions import DatabaseError

        mock_storage = MagicMock()
        mock_storage.search.side_effect = DatabaseError("Connection failed")

        handler = MockDebatesHandler(storage=mock_storage)

        with patch(
            "aragora.server.handlers.debates.search.require_permission",
            lambda p: lambda f: f,
        ):
            with patch(
                "aragora.server.handlers.debates.search.rate_limit",
                lambda **k: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.debates.search.require_storage",
                    lambda f: f,
                ):
                    with patch(
                        "aragora.server.handlers.debates.search.ttl_cache",
                        lambda **k: lambda f: f,
                    ):
                        result = handler._search_debates("test", 10, 0)

        assert result.status_code == 500
        body = json.loads(result.body)
        assert "error" in body


# =============================================================================
# Test Query Validation
# =============================================================================


class TestQueryValidation:
    """Tests for search query validation."""

    def test_search_invalid_query(self):
        """Should reject invalid/malicious search queries."""
        mock_storage = MagicMock()
        handler = MockDebatesHandler(storage=mock_storage)

        # Test with a very long query that could cause ReDoS
        very_long_query = "a" * 10000

        with patch(
            "aragora.server.handlers.debates.search.require_permission",
            lambda p: lambda f: f,
        ):
            with patch(
                "aragora.server.handlers.debates.search.rate_limit",
                lambda **k: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.debates.search.require_storage",
                    lambda f: f,
                ):
                    with patch(
                        "aragora.server.handlers.debates.search.ttl_cache",
                        lambda **k: lambda f: f,
                    ):
                        result = handler._search_debates(very_long_query, 10, 0)

        # Should return 400 for invalid query
        assert result.status_code == 400


# =============================================================================
# Test Response Normalization
# =============================================================================


class TestResponseNormalization:
    """Tests for debate response normalization."""

    def test_search_normalizes_debate_objects(self):
        """Should normalize debate objects with __dict__."""

        class MockDebate:
            def __init__(self):
                self.id = "debate-1"
                self.task = "Test task"
                self.status = "active"

        mock_storage = MagicMock()
        mock_storage.search.return_value = ([MockDebate()], 1)

        handler = MockDebatesHandler(storage=mock_storage)

        with patch(
            "aragora.server.handlers.debates.search.require_permission",
            lambda p: lambda f: f,
        ):
            with patch(
                "aragora.server.handlers.debates.search.rate_limit",
                lambda **k: lambda f: f,
            ):
                with patch(
                    "aragora.server.handlers.debates.search.require_storage",
                    lambda f: f,
                ):
                    with patch(
                        "aragora.server.handlers.debates.search.ttl_cache",
                        lambda **k: lambda f: f,
                    ):
                        result = handler._search_debates("test", 10, 0)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["results"]) == 1
        # Status should be normalized
        assert body["results"][0]["status"] == "running"
