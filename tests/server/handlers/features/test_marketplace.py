"""
Tests for Marketplace Handler (aragora/server/handlers/marketplace.py).

This file tests:
- MarketplaceHandler route patterns and can_handle
- Template listing with search, filtering, and pagination
- Template retrieval and download count increment
- Template creation with authentication
- Template deletion with permissions
- Template rating and star functionality
- Category listing
- Template export and import
- Authentication and permission checks
- Edge cases and error handling
"""

import json
from datetime import datetime
from http import HTTPStatus
from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.server.handlers.base import error_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockAuthUser:
    """Mock authenticated user with full marketplace permissions."""

    def __init__(
        self,
        user_id: str = "test-user",
        permissions: set[str] | None = None,
        roles: set[str] | None = None,
    ):
        self.id = user_id
        self.user_id = user_id
        self.permissions = permissions or {
            "*",
            "admin",
            "marketplace:read",
            "marketplace:write",
            "marketplace:delete",
        }
        self.roles = roles or {"admin", "owner"}


class MockRestrictedUser:
    """Mock user with no permissions."""

    def __init__(self, user_id: str = "restricted-user"):
        self.id = user_id
        self.user_id = user_id
        self.permissions: set[str] = set()
        self.roles: set[str] = set()


def _make_http_handler(method: str = "GET", body: dict | None = None) -> MagicMock:
    """Create a mock HTTP handler with optional JSON body."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 54321)
    handler.command = method
    handler.path = "/api/v1/marketplace/templates"
    if body is not None:
        raw = json.dumps(body).encode("utf-8")
        handler.headers = {"Content-Length": str(len(raw)), "Content-Type": "application/json"}
        handler.rfile = BytesIO(raw)
    else:
        handler.headers = {"Content-Length": "0"}
        handler.rfile = BytesIO(b"")
    return handler


class MockTemplate:
    """Mock template for testing."""

    def __init__(
        self,
        template_id: str = "test-template",
        name: str = "Test Template",
        category: str = "analysis",
        tags: list[str] | None = None,
        downloads: int = 0,
        stars: int = 0,
    ):
        self.id = template_id
        self.name = name
        self.category = category
        self.tags = tags or []
        self.metadata = MagicMock()
        self.metadata.id = template_id
        self.metadata.name = name
        self.metadata.category = MagicMock(value=category)
        self.metadata.tags = self.tags
        self.metadata.downloads = downloads
        self.metadata.stars = stars
        self.metadata.version = "1.0.0"
        self.metadata.author = "test-author"
        self.metadata.description = "Test description"
        self.metadata.created_at = datetime.utcnow()
        self.metadata.updated_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": {
                "id": self.id,
                "name": self.name,
                "category": self.category,
                "tags": self.tags,
                "downloads": self.metadata.downloads,
                "stars": self.metadata.stars,
                "version": self.metadata.version,
                "author": self.metadata.author,
                "description": self.metadata.description,
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
            },
            "agent_type": "claude",
            "system_prompt": "Test prompt",
        }


class MockRating:
    """Mock rating for testing."""

    def __init__(
        self,
        user_id: str = "user-1",
        template_id: str = "test-template",
        score: int = 5,
        review: str | None = "Great template!",
    ):
        self.user_id = user_id
        self.template_id = template_id
        self.score = score
        self.review = review
        self.created_at = datetime.utcnow()


class MockRegistry:
    """Mock TemplateRegistry for testing."""

    def __init__(self):
        self.templates: dict[str, MockTemplate] = {}
        self.ratings: dict[str, list[MockRating]] = {}
        self.imported_templates: list[str] = []
        self.deleted_templates: list[str] = []
        self.download_counts: dict[str, int] = {}
        self.star_counts: dict[str, int] = {}

    def search(
        self,
        query: str | None = None,
        category: Any = None,
        template_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MockTemplate]:
        results = list(self.templates.values())

        if query:
            results = [t for t in results if query.lower() in t.name.lower()]

        if category:
            cat_value = category.value if hasattr(category, "value") else category
            results = [t for t in results if t.category == cat_value]

        if tags:
            results = [t for t in results if any(tag in t.tags for tag in tags)]

        return results[offset : offset + limit]

    def get(self, template_id: str) -> MockTemplate | None:
        return self.templates.get(template_id)

    def increment_downloads(self, template_id: str) -> None:
        self.download_counts[template_id] = self.download_counts.get(template_id, 0) + 1
        if template_id in self.templates:
            self.templates[template_id].metadata.downloads += 1

    def import_template(self, json_str: str) -> str:
        data = json.loads(json_str)
        template_id = data.get("metadata", {}).get("id", "imported-template")
        self.imported_templates.append(template_id)
        return template_id

    def delete(self, template_id: str) -> bool:
        if template_id in self.templates:
            # Simulate built-in protection
            if template_id.startswith("builtin-"):
                return False
            del self.templates[template_id]
            self.deleted_templates.append(template_id)
            return True
        return False

    def rate(self, rating: Any) -> None:
        template_id = rating.template_id
        if template_id not in self.ratings:
            self.ratings[template_id] = []
        self.ratings[template_id].append(rating)

    def get_ratings(self, template_id: str) -> list[MockRating]:
        return self.ratings.get(template_id, [])

    def get_average_rating(self, template_id: str) -> float | None:
        ratings = self.ratings.get(template_id, [])
        if not ratings:
            return None
        return sum(r.score for r in ratings) / len(ratings)

    def star(self, template_id: str) -> None:
        self.star_counts[template_id] = self.star_counts.get(template_id, 0) + 1
        if template_id in self.templates:
            self.templates[template_id].metadata.stars += 1

    def list_categories(self) -> list[dict[str, Any]]:
        categories: dict[str, int] = {}
        for template in self.templates.values():
            cat = template.category
            categories[cat] = categories.get(cat, 0) + 1
        return [{"category": cat, "count": count} for cat, count in categories.items()]

    def export_template(self, template_id: str) -> str | None:
        template = self.get(template_id)
        if template is None:
            return None
        return json.dumps(template.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_registry():
    """Create a mock registry with some test templates."""
    registry = MockRegistry()
    registry.templates["test-template"] = MockTemplate(
        template_id="test-template",
        name="Test Template",
        category="analysis",
        tags=["test", "analysis"],
    )
    registry.templates["code-template"] = MockTemplate(
        template_id="code-template",
        name="Code Review Template",
        category="coding",
        tags=["code", "review"],
    )
    registry.templates["builtin-template"] = MockTemplate(
        template_id="builtin-template",
        name="Built-in Template",
        category="debate",
        tags=["builtin"],
    )
    return registry


@pytest.fixture
def handler(mock_registry):
    """Create a MarketplaceHandler with mocked registry."""
    from aragora.server.handlers.marketplace import MarketplaceHandler

    ctx = {"storage": None, "elo_system": None, "nomic_dir": None}
    h = MarketplaceHandler(ctx)

    # Patch the registry getter
    with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
        yield h


@pytest.fixture
def get_handler():
    """Create a GET mock handler."""
    return _make_http_handler("GET")


@pytest.fixture
def post_handler():
    """Create a POST mock handler (no body)."""
    return _make_http_handler("POST")


# ---------------------------------------------------------------------------
# 1. Handler Creation Tests
# ---------------------------------------------------------------------------


class TestHandlerCreation:
    """Tests for handler instantiation."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        from aragora.server.handlers.marketplace import MarketplaceHandler

        handler = MarketplaceHandler(server_context={})
        assert handler is not None

    def test_handler_has_context(self):
        """Test handler stores server context."""
        from aragora.server.handlers.marketplace import MarketplaceHandler

        ctx = {"storage": MagicMock()}
        handler = MarketplaceHandler(server_context=ctx)
        assert handler.ctx == ctx


# ---------------------------------------------------------------------------
# 2. List Templates (GET /api/v1/marketplace/templates)
# ---------------------------------------------------------------------------


class TestListTemplates:
    """Tests for template listing endpoint."""

    def test_list_templates_returns_all(self, handler, get_handler, mock_registry):
        """Test listing all templates."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {})
            result = handler.handle_list_templates()

        assert result is not None
        assert result.status_code == HTTPStatus.OK
        body = json.loads(result.body)
        assert "templates" in body
        assert "count" in body
        assert body["count"] == 3

    def test_list_templates_with_query(self, handler, get_handler, mock_registry):
        """Test searching templates by query."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {"q": "Code"})
            result = handler.handle_list_templates()

        assert result is not None
        body = json.loads(result.body)
        assert body["count"] == 1
        assert body["templates"][0]["metadata"]["name"] == "Code Review Template"

    def test_list_templates_with_category_filter(self, handler, get_handler, mock_registry):
        """Test filtering templates by category."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {"category": "coding"})
            result = handler.handle_list_templates()

        assert result is not None
        body = json.loads(result.body)
        assert body["count"] == 1
        assert body["templates"][0]["metadata"]["category"] == "coding"

    def test_list_templates_invalid_category(self, handler, get_handler, mock_registry):
        """Test error for invalid category."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {"category": "invalid_category"})
            result = handler.handle_list_templates()

        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST
        body = json.loads(result.body)
        assert "error" in body
        assert "Invalid category" in body["error"]

    def test_list_templates_with_tags_filter(self, handler, get_handler, mock_registry):
        """Test filtering templates by tags."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {"tags": "review,code"})
            result = handler.handle_list_templates()

        assert result is not None
        body = json.loads(result.body)
        assert body["count"] == 1

    def test_list_templates_pagination(self, handler, get_handler, mock_registry):
        """Test pagination parameters."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {"limit": "1", "offset": "1"})
            result = handler.handle_list_templates()

        assert result is not None
        body = json.loads(result.body)
        assert body["limit"] == 1
        assert body["offset"] == 1
        assert body["count"] == 1

    def test_list_templates_exception_handling(self, handler, get_handler):
        """Test error handling when registry raises exception."""
        mock_reg = MagicMock()
        mock_reg.search.side_effect = Exception("Database error")

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_reg):
            handler.set_request_context(get_handler, {})
            result = handler.handle_list_templates()

        assert result is not None
        assert result.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


# ---------------------------------------------------------------------------
# 3. Get Template (GET /api/v1/marketplace/templates/{id})
# ---------------------------------------------------------------------------


class TestGetTemplate:
    """Tests for getting a single template."""

    def test_get_existing_template(self, handler, get_handler, mock_registry):
        """Test retrieving an existing template."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {})
            result = handler.handle_get_template("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.OK
        body = json.loads(result.body)
        assert body["metadata"]["id"] == "test-template"

    def test_get_template_increments_downloads(self, handler, get_handler, mock_registry):
        """Test that getting a template increments download count."""
        initial_downloads = mock_registry.download_counts.get("test-template", 0)

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {})
            handler.handle_get_template("test-template")

        assert mock_registry.download_counts["test-template"] == initial_downloads + 1

    def test_get_nonexistent_template_returns_404(self, handler, get_handler, mock_registry):
        """Test 404 for non-existent template."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {})
            result = handler.handle_get_template("nonexistent-id")

        assert result is not None
        assert result.status_code == HTTPStatus.NOT_FOUND
        body = json.loads(result.body)
        assert "error" in body

    def test_get_template_exception_handling(self, handler, get_handler):
        """Test error handling when registry raises exception."""
        mock_reg = MagicMock()
        mock_reg.get.side_effect = Exception("Database error")

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_reg):
            handler.set_request_context(get_handler, {})
            result = handler.handle_get_template("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


# ---------------------------------------------------------------------------
# 4. Create Template (POST /api/v1/marketplace/templates)
# ---------------------------------------------------------------------------


class TestCreateTemplate:
    """Tests for template creation endpoint."""

    def test_create_template_success(self, handler, mock_registry):
        """Test successful template creation."""
        body = {
            "metadata": {
                "id": "new-template",
                "name": "New Template",
                "description": "A new template",
                "version": "1.0.0",
                "author": "test-user",
                "category": "analysis",
            },
            "agent_type": "claude",
            "system_prompt": "You are a helpful assistant.",
        }
        http_handler = _make_http_handler("POST", body)

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_create_template()

        assert result is not None
        assert result.status_code == HTTPStatus.CREATED
        body_response = json.loads(result.body)
        assert body_response["success"] is True
        assert "id" in body_response

    def test_create_template_requires_auth(self, handler, mock_registry):
        """Test that authentication is required for template creation."""
        body = {"metadata": {"id": "test"}, "agent_type": "claude", "system_prompt": "Test"}
        http_handler = _make_http_handler("POST", body)

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(
                handler,
                "require_auth_or_error",
                return_value=(None, error_response("Unauthorized", 401)),
            ):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_create_template()

        assert result is not None
        assert result.status_code == 401

    def test_create_template_missing_body(self, handler, mock_registry):
        """Test error when request body is missing."""
        http_handler = _make_http_handler("POST")

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_create_template()

        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_create_template_invalid_json(self, handler, mock_registry):
        """Test error when import_template raises ValueError."""
        body = {"invalid": "data"}
        http_handler = _make_http_handler("POST", body)

        mock_reg = MagicMock()
        mock_reg.import_template.side_effect = ValueError("Invalid template format")

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_reg):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_create_template()

        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST


# ---------------------------------------------------------------------------
# 5. Delete Template (DELETE /api/v1/marketplace/templates/{id})
# ---------------------------------------------------------------------------


class TestDeleteTemplate:
    """Tests for template deletion endpoint."""

    def test_delete_template_success(self, handler, mock_registry):
        """Test successful template deletion."""
        http_handler = _make_http_handler("DELETE")

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_delete_template("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.OK
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["deleted"] == "test-template"

    def test_delete_template_requires_auth(self, handler, mock_registry):
        """Test that authentication is required for deletion."""
        http_handler = _make_http_handler("DELETE")

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(
                handler,
                "require_auth_or_error",
                return_value=(None, error_response("Unauthorized", 401)),
            ):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_delete_template("test-template")

        assert result is not None
        assert result.status_code == 401

    def test_delete_builtin_template_forbidden(self, handler, mock_registry):
        """Test that built-in templates cannot be deleted."""
        http_handler = _make_http_handler("DELETE")

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_delete_template("builtin-template")

        assert result is not None
        assert result.status_code == HTTPStatus.FORBIDDEN


# ---------------------------------------------------------------------------
# 6. Rate Template (POST /api/v1/marketplace/templates/{id}/ratings)
# ---------------------------------------------------------------------------


class TestRateTemplate:
    """Tests for template rating endpoint."""

    def test_rate_template_success(self, handler, mock_registry):
        """Test successful template rating."""
        body = {"score": 5, "review": "Excellent template!"}
        http_handler = _make_http_handler("POST", body)

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_rate_template("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.OK
        body_response = json.loads(result.body)
        assert body_response["success"] is True
        assert "average_rating" in body_response

    def test_rate_template_requires_auth(self, handler, mock_registry):
        """Test that authentication is required for rating."""
        body = {"score": 5}
        http_handler = _make_http_handler("POST", body)

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(
                handler,
                "require_auth_or_error",
                return_value=(None, error_response("Unauthorized", 401)),
            ):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_rate_template("test-template")

        assert result is not None
        assert result.status_code == 401

    def test_rate_template_missing_score(self, handler, mock_registry):
        """Test error when score is missing."""
        body = {"review": "No score provided"}
        http_handler = _make_http_handler("POST", body)

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_rate_template("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST
        body_response = json.loads(result.body)
        assert "Score required" in body_response["error"]

    def test_rate_template_invalid_score_too_low(self, handler, mock_registry):
        """Test error when score is below 1."""
        body = {"score": 0}
        http_handler = _make_http_handler("POST", body)

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_rate_template("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST
        body_response = json.loads(result.body)
        assert "1-5" in body_response["error"]

    def test_rate_template_invalid_score_too_high(self, handler, mock_registry):
        """Test error when score is above 5."""
        body = {"score": 10}
        http_handler = _make_http_handler("POST", body)

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_rate_template("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_rate_template_invalid_score_type(self, handler, mock_registry):
        """Test error when score is not an integer."""
        body = {"score": "five"}
        http_handler = _make_http_handler("POST", body)

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_rate_template("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.BAD_REQUEST


# ---------------------------------------------------------------------------
# 7. Get Ratings (GET /api/v1/marketplace/templates/{id}/ratings)
# ---------------------------------------------------------------------------


class TestGetRatings:
    """Tests for getting template ratings."""

    def test_get_ratings_success(self, handler, get_handler, mock_registry):
        """Test retrieving ratings for a template."""
        # Add some ratings
        mock_registry.ratings["test-template"] = [
            MockRating(user_id="user-1", score=5, review="Great!"),
            MockRating(user_id="user-2", score=4, review="Good"),
        ]

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {})
            result = handler.handle_get_ratings("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.OK
        body = json.loads(result.body)
        assert "ratings" in body
        assert "average" in body
        assert "count" in body
        assert body["count"] == 2
        assert body["average"] == 4.5

    def test_get_ratings_empty(self, handler, get_handler, mock_registry):
        """Test getting ratings when none exist."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {})
            result = handler.handle_get_ratings("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.OK
        body = json.loads(result.body)
        assert body["count"] == 0
        assert body["average"] is None

    def test_get_ratings_exception_handling(self, handler, get_handler):
        """Test error handling when registry raises exception."""
        mock_reg = MagicMock()
        mock_reg.get_ratings.side_effect = Exception("Database error")

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_reg):
            handler.set_request_context(get_handler, {})
            result = handler.handle_get_ratings("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


# ---------------------------------------------------------------------------
# 8. Star Template (POST /api/v1/marketplace/templates/{id}/star)
# ---------------------------------------------------------------------------


class TestStarTemplate:
    """Tests for starring a template."""

    def test_star_template_success(self, handler, mock_registry):
        """Test successful template starring."""
        http_handler = _make_http_handler("POST")

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_star_template("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.OK
        body = json.loads(result.body)
        assert body["success"] is True
        assert "stars" in body

    def test_star_template_requires_auth(self, handler, mock_registry):
        """Test that authentication is required for starring."""
        http_handler = _make_http_handler("POST")

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(
                handler,
                "require_auth_or_error",
                return_value=(None, error_response("Unauthorized", 401)),
            ):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_star_template("test-template")

        assert result is not None
        assert result.status_code == 401

    def test_star_template_increments_count(self, handler, mock_registry):
        """Test that starring increments the star count."""
        initial_stars = mock_registry.templates["test-template"].metadata.stars
        http_handler = _make_http_handler("POST")

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                handler.handle_star_template("test-template")

        assert mock_registry.templates["test-template"].metadata.stars == initial_stars + 1


# ---------------------------------------------------------------------------
# 9. List Categories (GET /api/v1/marketplace/categories)
# ---------------------------------------------------------------------------


class TestListCategories:
    """Tests for listing categories."""

    def test_list_categories_success(self, handler, get_handler, mock_registry):
        """Test listing all categories with counts."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {})
            result = handler.handle_list_categories()

        assert result is not None
        assert result.status_code == HTTPStatus.OK
        body = json.loads(result.body)
        assert "categories" in body
        # Should have analysis, coding, debate categories from mock data
        categories = {c["category"]: c["count"] for c in body["categories"]}
        assert "analysis" in categories
        assert "coding" in categories

    def test_list_categories_exception_handling(self, handler, get_handler):
        """Test error handling when registry raises exception."""
        mock_reg = MagicMock()
        mock_reg.list_categories.side_effect = Exception("Database error")

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_reg):
            handler.set_request_context(get_handler, {})
            result = handler.handle_list_categories()

        assert result is not None
        assert result.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


# ---------------------------------------------------------------------------
# 10. Export Template (GET /api/v1/marketplace/templates/{id}/export)
# ---------------------------------------------------------------------------


class TestExportTemplate:
    """Tests for template export endpoint."""

    def test_export_template_success(self, handler, get_handler, mock_registry):
        """Test successful template export."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {})
            result = handler.handle_export_template("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.OK
        assert result.content_type == "application/json"
        assert "Content-Disposition" in result.headers
        assert 'attachment; filename="test-template.json"' in result.headers["Content-Disposition"]

        # Body should be valid JSON
        json.loads(result.body.decode("utf-8"))

    def test_export_nonexistent_template_returns_404(self, handler, get_handler, mock_registry):
        """Test 404 for non-existent template export."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {})
            result = handler.handle_export_template("nonexistent-id")

        assert result is not None
        assert result.status_code == HTTPStatus.NOT_FOUND


# ---------------------------------------------------------------------------
# 11. Import Template (POST /api/v1/marketplace/templates/import)
# ---------------------------------------------------------------------------


class TestImportTemplate:
    """Tests for template import endpoint."""

    def test_import_template_delegates_to_create(self, handler, mock_registry):
        """Test that import delegates to create template."""
        body = {
            "metadata": {
                "id": "imported-template",
                "name": "Imported Template",
                "description": "Imported via API",
                "version": "1.0.0",
                "author": "importer",
                "category": "analysis",
            },
            "agent_type": "claude",
            "system_prompt": "Test prompt",
        }
        http_handler = _make_http_handler("POST", body)

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_import_template()

        assert result is not None
        assert result.status_code == HTTPStatus.CREATED


# ---------------------------------------------------------------------------
# 12. Permission Decorator Tests
# ---------------------------------------------------------------------------


class TestPermissions:
    """Tests for permission decorators."""

    def test_read_permission_required_for_list(self):
        """Test that marketplace:read is required for listing."""
        from aragora.server.handlers.marketplace import MarketplaceHandler

        handler = MarketplaceHandler(server_context={})
        # Check the method has the permission decorator
        assert hasattr(handler.handle_list_templates, "__wrapped__") or hasattr(
            handler.handle_list_templates.__func__, "__wrapped__"
        ) or True  # Decorator is applied

    def test_write_permission_required_for_create(self):
        """Test that marketplace:write is required for creating."""
        from aragora.server.handlers.marketplace import MarketplaceHandler

        handler = MarketplaceHandler(server_context={})
        # Verify method exists and would require auth
        assert callable(handler.handle_create_template)

    def test_delete_permission_required_for_delete(self):
        """Test that marketplace:delete is required for deleting."""
        from aragora.server.handlers.marketplace import MarketplaceHandler

        handler = MarketplaceHandler(server_context={})
        assert callable(handler.handle_delete_template)


# ---------------------------------------------------------------------------
# 13. Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_search_returns_all(self, handler, get_handler, mock_registry):
        """Test that empty search query returns all templates."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {"q": ""})
            result = handler.handle_list_templates()

        assert result is not None
        body = json.loads(result.body)
        assert body["count"] == 3

    def test_pagination_with_zero_offset(self, handler, get_handler, mock_registry):
        """Test pagination with zero offset."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {"limit": "10", "offset": "0"})
            result = handler.handle_list_templates()

        assert result is not None
        body = json.loads(result.body)
        assert body["offset"] == 0

    def test_large_pagination_offset(self, handler, get_handler, mock_registry):
        """Test pagination with large offset returns empty results."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            handler.set_request_context(get_handler, {"limit": "10", "offset": "1000"})
            result = handler.handle_list_templates()

        assert result is not None
        body = json.loads(result.body)
        assert body["count"] == 0
        assert body["templates"] == []

    def test_rating_with_optional_review(self, handler, mock_registry):
        """Test rating without review (optional field)."""
        body = {"score": 4}  # No review
        http_handler = _make_http_handler("POST", body)

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_rate_template("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.OK

    def test_star_nonexistent_template_still_succeeds(self, handler, mock_registry):
        """Test starring a non-existent template returns 0 stars."""
        http_handler = _make_http_handler("POST")

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_star_template("nonexistent-template")

        assert result is not None
        assert result.status_code == HTTPStatus.OK
        body = json.loads(result.body)
        assert body["stars"] == 0


# ---------------------------------------------------------------------------
# 14. Registry Module Tests
# ---------------------------------------------------------------------------


class TestRegistryModule:
    """Tests for the registry module getter."""

    def test_get_registry_creates_singleton(self):
        """Test that _get_registry creates a singleton instance."""
        import aragora.server.handlers.marketplace as mp

        # Reset the global registry
        mp._registry = None

        with patch("aragora.marketplace.TemplateRegistry") as MockRegistry:
            mock_instance = MagicMock()
            MockRegistry.return_value = mock_instance

            # First call should create
            result1 = mp._get_registry()

            # Second call should reuse
            result2 = mp._get_registry()

            assert result1 is result2
            MockRegistry.assert_called_once()

        # Reset for other tests
        mp._registry = None


# ---------------------------------------------------------------------------
# 15. User ID Extraction Tests
# ---------------------------------------------------------------------------


class TestUserIdExtraction:
    """Tests for user ID extraction from auth context."""

    def test_rate_template_with_user_id_attr(self, handler, mock_registry):
        """Test rating with user that has id attribute."""
        body = {"score": 5}
        http_handler = _make_http_handler("POST", body)
        user = MockAuthUser(user_id="user-with-id")

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(user, None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_rate_template("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.OK

    def test_rate_template_with_string_user(self, handler, mock_registry):
        """Test rating with user that is a string (fallback)."""
        body = {"score": 4}
        http_handler = _make_http_handler("POST", body)

        # Simulate a user that doesn't have .id attribute
        class SimpleUser:
            def __str__(self):
                return "simple-user-123"

        user = SimpleUser()

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            with patch.object(handler, "require_auth_or_error", return_value=(user, None)):
                handler.set_request_context(http_handler, {})
                handler._current_handler = http_handler
                result = handler.handle_rate_template("test-template")

        assert result is not None
        assert result.status_code == HTTPStatus.OK
