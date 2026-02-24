"""Tests for the Marketplace Pilot API handler.

Covers:
- GET  /api/v1/marketplace/listings          (browse with filters)
- GET  /api/v1/marketplace/listings/featured (featured items)
- GET  /api/v1/marketplace/listings/stats    (marketplace statistics)
- GET  /api/v1/marketplace/listings/{id}     (detail)
- POST /api/v1/marketplace/listings/{id}/install (install)
- POST /api/v1/marketplace/listings/{id}/rate    (rate)
"""

from __future__ import annotations

import json
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from aragora.marketplace.catalog import MarketplaceCatalog
from aragora.marketplace.service import MarketplaceService
from aragora.server.handlers.marketplace_pilot import MarketplacePilotHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Create a MarketplaceService with seed data."""
    return MarketplaceService(catalog=MarketplaceCatalog(seed=True))


@pytest.fixture
def handler(service):
    """Create a MarketplacePilotHandler with injected service."""
    ctx: dict = {"storage": MagicMock()}
    h = MarketplacePilotHandler(ctx)
    h._service = service
    return h


@pytest.fixture
def mock_http():
    """Create a bare mock HTTP handler (unauthenticated, no body)."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 54321)
    h.headers = {"Content-Length": "0"}
    return h


@pytest.fixture
def authed_http():
    """Create a mock HTTP handler that passes require_auth_or_error."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 54321)
    h.headers = {
        "Content-Length": "0",
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
    }
    return h


def _make_body_handler(body: dict) -> MagicMock:
    """Create an authed handler with a JSON body ready for read_json_body."""
    raw = json.dumps(body).encode()
    h = MagicMock()
    h.client_address = ("127.0.0.1", 54321)
    h.headers = {
        "Content-Length": str(len(raw)),
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
    }
    h.rfile = BytesIO(raw)
    return h


def _parse_data(result):
    """Unwrap the ``{"data": ...}`` envelope from a HandlerResult."""
    assert result is not None
    # HandlerResult supports index access: result[0] returns parsed JSON body
    body = result[0]
    if isinstance(body, dict):
        return body.get("data", body)
    return body


# ---------------------------------------------------------------------------
# GET /api/v1/marketplace/listings
# ---------------------------------------------------------------------------


class TestListListings:
    """Tests for the listing browse endpoint."""

    def test_list_returns_items(self, handler, mock_http):
        """Browse returns seed items."""
        result = handler.handle("/api/v1/marketplace/listings", {}, mock_http)
        data = _parse_data(result)
        assert data is not None
        assert "items" in data
        assert data["total"] >= 10  # seed has 15 items

    def test_list_filter_by_type(self, handler, mock_http):
        """Filter by type returns only matching items."""
        result = handler.handle(
            "/api/v1/marketplace/listings",
            {"type": "template"},
            mock_http,
        )
        data = _parse_data(result)
        for item in data["items"]:
            assert item["type"] == "template"

    def test_list_filter_by_tag(self, handler, mock_http):
        """Filter by tag returns items containing that tag."""
        result = handler.handle(
            "/api/v1/marketplace/listings",
            {"tag": "code"},
            mock_http,
        )
        data = _parse_data(result)
        assert data["total"] >= 1
        for item in data["items"]:
            assert any("code" in t.lower() for t in item["tags"])

    def test_list_search(self, handler, mock_http):
        """Free-text search over name and description."""
        result = handler.handle(
            "/api/v1/marketplace/listings",
            {"search": "compliance"},
            mock_http,
        )
        data = _parse_data(result)
        assert data["total"] >= 1
        names = [i["name"] for i in data["items"]]
        assert any("Compliance" in n for n in names)

    def test_list_pagination(self, handler, mock_http):
        """Pagination via limit and offset."""
        result = handler.handle(
            "/api/v1/marketplace/listings",
            {"limit": "3", "offset": "0"},
            mock_http,
        )
        data = _parse_data(result)
        assert len(data["items"]) <= 3
        assert data["limit"] == 3
        assert data["offset"] == 0

    def test_list_search_too_long(self, handler, mock_http):
        """Search query exceeding max length returns 400."""
        result = handler.handle(
            "/api/v1/marketplace/listings",
            {"search": "x" * 600},
            mock_http,
        )
        assert result is not None
        status = result.status_code if hasattr(result, "status_code") else result[1]
        assert status == 400


# ---------------------------------------------------------------------------
# GET /api/v1/marketplace/listings/featured
# ---------------------------------------------------------------------------


class TestFeatured:
    """Tests for the featured listings endpoint."""

    def test_featured_returns_items(self, handler, mock_http):
        """Featured endpoint returns featured items."""
        result = handler.handle(
            "/api/v1/marketplace/listings/featured", {}, mock_http
        )
        data = _parse_data(result)
        assert "items" in data
        for item in data["items"]:
            assert item["featured"] is True

    def test_featured_respects_limit(self, handler, mock_http):
        """Featured endpoint respects limit param."""
        result = handler.handle(
            "/api/v1/marketplace/listings/featured",
            {"limit": "2"},
            mock_http,
        )
        data = _parse_data(result)
        assert len(data["items"]) <= 2


# ---------------------------------------------------------------------------
# GET /api/v1/marketplace/listings/stats
# ---------------------------------------------------------------------------


class TestStats:
    """Tests for the statistics endpoint."""

    def test_stats_returns_counts(self, handler, mock_http):
        """Stats endpoint returns type breakdown."""
        result = handler.handle(
            "/api/v1/marketplace/listings/stats", {}, mock_http
        )
        data = _parse_data(result)
        assert "total_items" in data
        assert data["total_items"] >= 10
        assert "types" in data
        assert "template" in data["types"]


# ---------------------------------------------------------------------------
# GET /api/v1/marketplace/listings/{id}
# ---------------------------------------------------------------------------


class TestGetDetail:
    """Tests for the listing detail endpoint."""

    def test_get_existing_item(self, handler, mock_http):
        """Detail endpoint returns enriched item dict."""
        result = handler.handle(
            "/api/v1/marketplace/listings/tpl-code-review", {}, mock_http
        )
        data = _parse_data(result)
        assert data["id"] == "tpl-code-review"
        assert data["name"] == "Code Review Pipeline"
        assert "average_rating" in data
        assert "total_ratings" in data

    def test_get_nonexistent_item(self, handler, mock_http):
        """Detail endpoint returns 404 for unknown ID."""
        result = handler.handle(
            "/api/v1/marketplace/listings/nonexistent-id", {}, mock_http
        )
        assert result is not None
        status = result.status_code if hasattr(result, "status_code") else result[1]
        assert status == 404

    def test_get_invalid_id(self, handler, mock_http):
        """Detail endpoint returns 400 for invalid ID pattern."""
        result = handler.handle(
            "/api/v1/marketplace/listings/!!!invalid!!!", {}, mock_http
        )
        assert result is not None
        status = result.status_code if hasattr(result, "status_code") else result[1]
        assert status == 400


# ---------------------------------------------------------------------------
# POST /api/v1/marketplace/listings/{id}/install
# ---------------------------------------------------------------------------


class TestInstall:
    """Tests for the install endpoint."""

    def test_install_success(self, handler, service):
        """Install increments download count and tracks user."""
        http_handler = _make_body_handler({})

        # Patch auth to return a mock user
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_user = MagicMock()
            mock_user.user_id = "user-42"
            mock_auth.return_value = (mock_user, None)

            result = handler.handle_post(
                "/api/v1/marketplace/listings/tpl-code-review/install",
                {},
                http_handler,
            )

        data = _parse_data(result)
        assert data["success"] is True
        assert data["item_id"] == "tpl-code-review"

        # Verify user install tracked
        assert "tpl-code-review" in service.get_user_installs("user-42")

    def test_install_nonexistent(self, handler):
        """Install returns 404 for unknown item."""
        http_handler = _make_body_handler({})

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_user = MagicMock()
            mock_user.user_id = "user-42"
            mock_auth.return_value = (mock_user, None)

            result = handler.handle_post(
                "/api/v1/marketplace/listings/nonexistent-item/install",
                {},
                http_handler,
            )

        assert result is not None
        status = result.status_code if hasattr(result, "status_code") else result[1]
        assert status == 404

    def test_install_requires_auth(self, handler, mock_http):
        """Install without auth returns 401."""
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            from aragora.server.handlers.base import error_response

            mock_auth.return_value = (None, error_response("Authentication required", 401))

            result = handler.handle_post(
                "/api/v1/marketplace/listings/tpl-code-review/install",
                {},
                mock_http,
            )

        assert result is not None
        status = result.status_code if hasattr(result, "status_code") else result[1]
        assert status == 401


# ---------------------------------------------------------------------------
# POST /api/v1/marketplace/listings/{id}/rate
# ---------------------------------------------------------------------------


class TestRate:
    """Tests for the rating endpoint."""

    def test_rate_success(self, handler, service):
        """Rating a listing stores the rating and returns average."""
        http_handler = _make_body_handler({"score": 5, "review": "Excellent!"})

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_user = MagicMock()
            mock_user.user_id = "user-42"
            mock_auth.return_value = (mock_user, None)

            result = handler.handle_post(
                "/api/v1/marketplace/listings/tpl-code-review/rate",
                {},
                http_handler,
            )

        data = _parse_data(result)
        assert data["success"] is True
        assert data["average_rating"] == 5.0
        assert data["total_ratings"] == 1

    def test_rate_updates_existing(self, handler, service):
        """Re-rating replaces the previous rating from the same user."""
        # Rate once with score 3
        http_handler = _make_body_handler({"score": 3})
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_user = MagicMock()
            mock_user.user_id = "user-42"
            mock_auth.return_value = (mock_user, None)
            handler.handle_post(
                "/api/v1/marketplace/listings/tpl-code-review/rate",
                {},
                http_handler,
            )

        # Rate again with score 5
        http_handler = _make_body_handler({"score": 5})
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_user = MagicMock()
            mock_user.user_id = "user-42"
            mock_auth.return_value = (mock_user, None)
            result = handler.handle_post(
                "/api/v1/marketplace/listings/tpl-code-review/rate",
                {},
                http_handler,
            )

        data = _parse_data(result)
        assert data["average_rating"] == 5.0
        assert data["total_ratings"] == 1  # Still 1, not 2

    def test_rate_invalid_score(self, handler):
        """Rating with invalid score returns 400."""
        http_handler = _make_body_handler({"score": 10})

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_user = MagicMock()
            mock_user.user_id = "user-42"
            mock_auth.return_value = (mock_user, None)

            result = handler.handle_post(
                "/api/v1/marketplace/listings/tpl-code-review/rate",
                {},
                http_handler,
            )

        assert result is not None
        status = result.status_code if hasattr(result, "status_code") else result[1]
        assert status == 400

    def test_rate_nonexistent_item(self, handler):
        """Rating a nonexistent item returns 404."""
        http_handler = _make_body_handler({"score": 4})

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_user = MagicMock()
            mock_user.user_id = "user-42"
            mock_auth.return_value = (mock_user, None)

            result = handler.handle_post(
                "/api/v1/marketplace/listings/nonexistent-item/rate",
                {},
                http_handler,
            )

        assert result is not None
        status = result.status_code if hasattr(result, "status_code") else result[1]
        assert status == 404

    def test_rate_no_body(self, handler):
        """Rating without a body returns 400."""
        http_handler = MagicMock()
        http_handler.client_address = ("127.0.0.1", 54321)
        http_handler.headers = {
            "Content-Length": "0",
            "Content-Type": "application/json",
            "Authorization": "Bearer test-token",
        }

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_user = MagicMock()
            mock_user.user_id = "user-42"
            mock_auth.return_value = (mock_user, None)

            # read_json_body returns {} for zero-length body, then score is None -> 400
            result = handler.handle_post(
                "/api/v1/marketplace/listings/tpl-code-review/rate",
                {},
                http_handler,
            )

        assert result is not None
        status = result.status_code if hasattr(result, "status_code") else result[1]
        assert status == 400

    def test_rate_review_too_long(self, handler):
        """Rating with review exceeding max length returns 400."""
        http_handler = _make_body_handler({"score": 4, "review": "x" * 2100})

        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_user = MagicMock()
            mock_user.user_id = "user-42"
            mock_auth.return_value = (mock_user, None)

            result = handler.handle_post(
                "/api/v1/marketplace/listings/tpl-code-review/rate",
                {},
                http_handler,
            )

        assert result is not None
        status = result.status_code if hasattr(result, "status_code") else result[1]
        assert status == 400

    def test_rate_requires_auth(self, handler, mock_http):
        """Rating without auth returns 401."""
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            from aragora.server.handlers.base import error_response

            mock_auth.return_value = (None, error_response("Authentication required", 401))

            result = handler.handle_post(
                "/api/v1/marketplace/listings/tpl-code-review/rate",
                {},
                mock_http,
            )

        assert result is not None
        status = result.status_code if hasattr(result, "status_code") else result[1]
        assert status == 401


# ---------------------------------------------------------------------------
# MarketplaceService unit tests
# ---------------------------------------------------------------------------


class TestMarketplaceService:
    """Unit tests for the MarketplaceService itself."""

    def test_list_all(self, service):
        """Service lists all seed items."""
        result = service.list_listings()
        assert result["total"] >= 10

    def test_list_by_type(self, service):
        """Service filters by item type."""
        result = service.list_listings(item_type="agent_pack")
        for item in result["items"]:
            assert item["type"] == "agent_pack"
        assert result["total"] >= 3

    def test_get_listing_found(self, service):
        """Service returns item dict for valid ID."""
        item = service.get_listing("pack-speed")
        assert item is not None
        assert item["id"] == "pack-speed"

    def test_get_listing_not_found(self, service):
        """Service returns None for unknown ID."""
        assert service.get_listing("nope") is None

    def test_install_and_track(self, service):
        """Install increments downloads and tracks per user."""
        result = service.install_listing("tpl-brainstorm", user_id="u1")
        assert result.success
        assert "tpl-brainstorm" in service.get_user_installs("u1")

    def test_rate_and_average(self, service):
        """Rating computes correct average."""
        service.rate_listing("tpl-code-review", user_id="u1", score=4)
        service.rate_listing("tpl-code-review", user_id="u2", score=2)
        assert service.get_average_rating("tpl-code-review") == 3.0

    def test_rate_upsert(self, service):
        """Re-rating by same user replaces previous."""
        service.rate_listing("tpl-code-review", user_id="u1", score=1)
        service.rate_listing("tpl-code-review", user_id="u1", score=5)
        assert service.get_average_rating("tpl-code-review") == 5.0
        assert len(service.get_ratings("tpl-code-review")) == 1

    def test_rate_invalid_score(self, service):
        """Rating with invalid score raises ValueError."""
        with pytest.raises(ValueError):
            service.rate_listing("tpl-code-review", user_id="u1", score=0)

    def test_rate_nonexistent(self, service):
        """Rating nonexistent item raises KeyError."""
        with pytest.raises(KeyError):
            service.rate_listing("nope", user_id="u1", score=3)

    def test_stats(self, service):
        """Stats returns aggregate counts."""
        stats = service.get_stats()
        assert stats["total_items"] >= 10
        assert "template" in stats["types"]

    def test_enriched_fields(self, service):
        """Listing includes average_rating and total_ratings."""
        service.rate_listing("tpl-code-review", user_id="u1", score=4)
        item = service.get_listing("tpl-code-review")
        assert item is not None
        assert item["average_rating"] == 4.0
        assert item["total_ratings"] == 1


# ---------------------------------------------------------------------------
# Edge cases and routing
# ---------------------------------------------------------------------------


class TestRouting:
    """Tests for handler routing / can_handle logic."""

    def test_can_handle_listings_path(self, handler):
        """Handler accepts marketplace/listings paths."""
        assert handler.can_handle("/api/v1/marketplace/listings") is True
        assert handler.can_handle("/api/v1/marketplace/listings/foo") is True

    def test_cannot_handle_other_paths(self, handler):
        """Handler rejects non-marketplace paths."""
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/marketplace/templates") is False

    def test_invalid_item_id_in_post(self, handler, mock_http):
        """POST with invalid item ID returns 400."""
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_user = MagicMock()
            mock_user.user_id = "user-1"
            mock_auth.return_value = (mock_user, None)

            result = handler.handle_post(
                "/api/v1/marketplace/listings/!!!bad!!!/install",
                {},
                mock_http,
            )

        assert result is not None
        status = result.status_code if hasattr(result, "status_code") else result[1]
        assert status == 400

    def test_unknown_action(self, handler, mock_http):
        """POST with unknown action returns None (not handled)."""
        with patch.object(handler, "require_auth_or_error") as mock_auth:
            mock_user = MagicMock()
            mock_user.user_id = "user-1"
            mock_auth.return_value = (mock_user, None)

            result = handler.handle_post(
                "/api/v1/marketplace/listings/tpl-code-review/delete",
                {},
                mock_http,
            )

        assert result is None
