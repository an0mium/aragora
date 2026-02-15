"""Tests for the Marketplace Template Browsing API handler."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from aragora.marketplace.registry import TemplateRegistry as MarketplaceRegistry
from aragora.server.handlers.marketplace_browse import MarketplaceBrowseHandler


@pytest.fixture
def mp_registry(tmp_path):
    """Create a marketplace registry with a temp database."""
    db_path = tmp_path / "test_marketplace.db"
    return MarketplaceRegistry(db_path=db_path)


@pytest.fixture
def handler(mp_registry):
    """Create a MarketplaceBrowseHandler with mock server context."""
    ctx = {"storage": MagicMock()}
    h = MarketplaceBrowseHandler(ctx)
    h._registry = mp_registry
    return h


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 12345)
    h.headers = {"Content-Length": "0"}
    return h


@pytest.fixture
def authed_handler():
    """Create a mock HTTP handler with auth."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 12345)
    h.headers = {
        "Content-Length": "50",
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
    }
    return h


class TestMarketplaceBrowse:
    """Tests for GET /api/v1/marketplace/templates."""

    def test_browse_returns_templates(self, handler, mock_handler):
        """Browse endpoint returns marketplace templates."""
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        assert "templates" in data
        assert "count" in data
        assert data["count"] >= 3

    def test_browse_with_search(self, handler, mock_handler):
        """Browse with search query filters results."""
        result = handler.handle("/api/v1/marketplace/templates", {"search": "Devil"}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        assert data["count"] >= 1
        names = [t["metadata"]["name"] for t in data["templates"]]
        assert "Devil's Advocate" in names

    def test_browse_with_category(self, handler, mock_handler):
        """Browse with category filter."""
        result = handler.handle(
            "/api/v1/marketplace/templates", {"category": "debate"}, mock_handler
        )
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        for t in data["templates"]:
            assert t["metadata"]["category"] == "debate"


class TestMarketplaceFeatured:
    """Tests for GET /api/v1/marketplace/featured."""

    def test_featured_returns_templates(self, handler, mock_handler):
        """Featured endpoint returns featured templates."""
        result = handler.handle("/api/v1/marketplace/featured", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        assert "featured" in data
        assert "count" in data
        assert data["count"] == 6

    def test_featured_max_six(self, handler, mock_handler):
        """Featured returns at most 6 templates."""
        result = handler.handle("/api/v1/marketplace/featured", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        assert data["count"] <= 6


class TestMarketplacePopular:
    """Tests for GET /api/v1/marketplace/popular."""

    def test_popular_returns_templates(self, handler, mock_handler):
        """Popular endpoint returns templates sorted by downloads."""
        result = handler.handle("/api/v1/marketplace/popular", {}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        assert "templates" in data
        assert data["count"] >= 1

    def test_popular_respects_limit(self, handler, mock_handler):
        """Popular endpoint respects limit parameter."""
        result = handler.handle("/api/v1/marketplace/popular", {"limit": "2"}, mock_handler)
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        assert data["count"] <= 2


class TestMarketplaceRate:
    """Tests for POST /api/v1/marketplace/templates/{id}/rate."""

    @pytest.mark.no_auto_auth
    def test_rate_requires_auth(self, handler, mock_handler):
        """Rating without auth returns 401."""
        # Without auto-auth, get_current_user returns None for unauthenticated
        result = handler.handle_post(
            "/api/v1/marketplace/templates/devil-advocate/rate", {}, mock_handler
        )
        assert result is not None
        assert result.status_code == 401

    def test_rate_valid_score(self, handler, authed_handler, mp_registry):
        """Rating with valid score succeeds."""
        mock_user = MagicMock()
        mock_user.user_id = "test-user"
        handler.get_current_user = MagicMock(return_value=mock_user)

        body_data = json.dumps({"score": 4, "review": "Great template!"}).encode()
        authed_handler.headers = {
            "Content-Length": str(len(body_data)),
            "Content-Type": "application/json",
            "Authorization": "Bearer test-token",
        }
        authed_handler.rfile = MagicMock()
        authed_handler.rfile.read.return_value = body_data

        result = handler.handle_post(
            "/api/v1/marketplace/templates/devil-advocate/rate", {}, authed_handler
        )
        assert result is not None
        assert result.status_code == 200
        data = result[0]
        assert data["status"] == "ok"
        assert data["score"] == 4

    def test_rate_invalid_score(self, handler, authed_handler):
        """Rating with invalid score returns 400."""
        mock_user = MagicMock()
        mock_user.user_id = "test-user"
        handler.get_current_user = MagicMock(return_value=mock_user)

        body_data = json.dumps({"score": 10}).encode()
        authed_handler.headers = {
            "Content-Length": str(len(body_data)),
            "Content-Type": "application/json",
        }
        authed_handler.rfile = MagicMock()
        authed_handler.rfile.read.return_value = body_data

        result = handler.handle_post(
            "/api/v1/marketplace/templates/devil-advocate/rate", {}, authed_handler
        )
        assert result is not None
        assert result.status_code == 400
