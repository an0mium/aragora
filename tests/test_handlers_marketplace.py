"""
Tests for TemplateMarketplaceHandler - community template marketplace endpoints.

Tests cover:
- GET    /api/marketplace/templates           - Browse templates
- GET    /api/marketplace/templates/:id       - Get template details
- POST   /api/marketplace/templates           - Publish template
- POST   /api/marketplace/templates/:id/rate  - Rate template
- POST   /api/marketplace/templates/:id/review - Review template
- POST   /api/marketplace/templates/:id/import - Import template
- GET    /api/marketplace/featured            - Get featured templates
- GET    /api/marketplace/trending            - Get trending templates
- GET    /api/marketplace/categories          - Get categories
"""

import json
import pytest
from unittest.mock import Mock, MagicMock

from aragora.server.handlers.template_marketplace import (
    TemplateMarketplaceHandler,
    MarketplaceTemplate,
    TemplateReview,
)
from aragora.server.handlers.base import HandlerResult


def parse_handler_result(result: HandlerResult) -> tuple[dict, int]:
    """Helper to parse HandlerResult into (body_dict, status_code)."""
    body_str = result.body.decode("utf-8") if isinstance(result.body, bytes) else result.body
    try:
        body_dict = json.loads(body_str)
    except (json.JSONDecodeError, TypeError):
        body_dict = {"raw": body_str}
    return body_dict, result.status_code


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def handler():
    """Create a fresh handler instance for each test."""
    return TemplateMarketplaceHandler(server_context={})


@pytest.fixture
def mock_get_request():
    """Create a mock GET request handler."""
    handler = Mock()
    handler.headers = {"Content-Type": "application/json", "Content-Length": "0"}
    handler.command = "GET"
    handler.client_address = ("127.0.0.1", 12345)
    return handler


@pytest.fixture
def mock_post_request():
    """Create a mock POST request handler."""
    handler = Mock()
    handler.headers = {"Content-Type": "application/json", "Content-Length": "100"}
    handler.command = "POST"
    handler.client_address = ("127.0.0.1", 12345)
    return handler


# ============================================================================
# Test can_handle
# ============================================================================


class TestCanHandle:
    """Test can_handle method routing."""

    def test_browse_templates(self, handler):
        assert handler.can_handle("/api/marketplace/templates") is True

    def test_get_template(self, handler):
        assert handler.can_handle("/api/marketplace/templates/tpl-123") is True

    def test_rate_template(self, handler):
        assert handler.can_handle("/api/marketplace/templates/tpl-123/rate") is True

    def test_review_template(self, handler):
        assert handler.can_handle("/api/marketplace/templates/tpl-123/review") is True

    def test_import_template(self, handler):
        assert handler.can_handle("/api/marketplace/templates/tpl-123/import") is True

    def test_featured(self, handler):
        assert handler.can_handle("/api/marketplace/featured") is True

    def test_trending(self, handler):
        assert handler.can_handle("/api/marketplace/trending") is True

    def test_categories(self, handler):
        assert handler.can_handle("/api/marketplace/categories") is True

    def test_invalid_path(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_non_marketplace_path(self, handler):
        assert handler.can_handle("/api/workflow/templates") is False


# ============================================================================
# Test Browse Templates
# ============================================================================


class TestBrowseTemplates:
    """Test GET /api/marketplace/templates endpoint."""

    def test_browse_returns_templates(self, handler, mock_get_request):
        result = handler.handle("/api/marketplace/templates", {}, mock_get_request)
        body, status = parse_handler_result(result)

        assert status == 200
        assert "templates" in body
        assert "total" in body
        assert isinstance(body["templates"], list)

    def test_browse_with_category_filter(self, handler, mock_get_request):
        result = handler.handle("/api/marketplace/templates", {"category": "security"}, mock_get_request)
        body, status = parse_handler_result(result)

        assert status == 200
        for template in body["templates"]:
            assert template["category"] == "security"

    def test_browse_with_search(self, handler, mock_get_request):
        result = handler.handle("/api/marketplace/templates", {"search": "code"}, mock_get_request)
        body, status = parse_handler_result(result)

        assert status == 200
        # Search should return templates matching "code" in name/description/tags

    def test_browse_with_pagination(self, handler, mock_get_request):
        result = handler.handle("/api/marketplace/templates", {"limit": "2", "offset": "0"}, mock_get_request)
        body, status = parse_handler_result(result)

        assert status == 200
        assert len(body["templates"]) <= 2

    def test_browse_sort_by_rating(self, handler, mock_get_request):
        result = handler.handle("/api/marketplace/templates", {"sort_by": "rating"}, mock_get_request)
        body, status = parse_handler_result(result)

        assert status == 200
        templates = body["templates"]
        if len(templates) > 1:
            # Verify sorted by rating descending
            for i in range(len(templates) - 1):
                assert templates[i]["rating"] >= templates[i + 1]["rating"]


# ============================================================================
# Test Get Template
# ============================================================================


class TestGetTemplate:
    """Test GET /api/marketplace/templates/:id endpoint."""

    def test_get_existing_template(self, handler, mock_get_request):
        # First browse to get a template ID
        browse_result = handler.handle("/api/marketplace/templates", {}, mock_get_request)
        browse_body, _ = parse_handler_result(browse_result)

        if browse_body["templates"]:
            template_id = browse_body["templates"][0]["id"]
            result = handler.handle(f"/api/marketplace/templates/{template_id}", {}, mock_get_request)
            body, status = parse_handler_result(result)

            assert status == 200
            assert body["id"] == template_id
            assert "name" in body
            assert "description" in body

    def test_get_nonexistent_template(self, handler, mock_get_request):
        result = handler.handle("/api/marketplace/templates/nonexistent-id", {}, mock_get_request)
        body, status = parse_handler_result(result)

        assert status == 404
        assert "error" in body


# ============================================================================
# Test Publish Template
# ============================================================================


class TestPublishTemplate:
    """Test POST /api/marketplace/templates endpoint."""

    def test_publish_valid_template(self, handler, mock_post_request):
        body = {
            "name": "My Test Template",
            "description": "A test template for unit tests",
            "category": "testing",
            "pattern": "review_cycle",
            "workflow_definition": {"nodes": [], "edges": []},
            "tags": ["test", "unit-test"],
        }
        mock_post_request.rfile = Mock()
        mock_post_request.rfile.read = Mock(return_value=json.dumps(body).encode())
        mock_post_request.headers["Content-Length"] = len(json.dumps(body))

        result = handler.handle("/api/marketplace/templates", {}, mock_post_request)
        response_body, status = parse_handler_result(result)

        assert status == 201
        assert "id" in response_body

    def test_publish_missing_required_fields(self, handler, mock_post_request):
        body = {"name": "Incomplete Template"}  # Missing description, category, pattern, workflow_definition
        mock_post_request.rfile = Mock()
        mock_post_request.rfile.read = Mock(return_value=json.dumps(body).encode())
        mock_post_request.headers["Content-Length"] = len(json.dumps(body))

        result = handler.handle("/api/marketplace/templates", {}, mock_post_request)
        response_body, status = parse_handler_result(result)

        assert status == 400
        assert "error" in response_body


# ============================================================================
# Test Rate Template
# ============================================================================


class TestRateTemplate:
    """Test POST /api/marketplace/templates/:id/rate endpoint."""

    def test_rate_valid(self, handler, mock_get_request, mock_post_request):
        # Get existing template ID
        browse_result = handler.handle("/api/marketplace/templates", {}, mock_get_request)
        browse_body, _ = parse_handler_result(browse_result)

        if browse_body["templates"]:
            template_id = browse_body["templates"][0]["id"]

            body = {"rating": 4}
            mock_post_request.rfile = Mock()
            mock_post_request.rfile.read = Mock(return_value=json.dumps(body).encode())
            mock_post_request.headers["Content-Length"] = len(json.dumps(body))

            result = handler.handle(f"/api/marketplace/templates/{template_id}/rate", {}, mock_post_request)
            response_body, status = parse_handler_result(result)

            assert status == 200
            assert response_body["status"] == "rated"
            assert "average_rating" in response_body

    def test_rate_invalid_rating(self, handler, mock_get_request, mock_post_request):
        # Get existing template ID first
        browse_result = handler.handle("/api/marketplace/templates", {}, mock_get_request)
        browse_body, _ = parse_handler_result(browse_result)

        if browse_body["templates"]:
            template_id = browse_body["templates"][0]["id"]
            body = {"rating": 10}  # Invalid: must be 1-5
            mock_post_request.rfile = Mock()
            mock_post_request.rfile.read = Mock(return_value=json.dumps(body).encode())
            mock_post_request.headers["Content-Length"] = len(json.dumps(body))

            result = handler.handle(f"/api/marketplace/templates/{template_id}/rate", {}, mock_post_request)
            response_body, status = parse_handler_result(result)

            assert status == 400
            assert "error" in response_body


# ============================================================================
# Test Review Template
# ============================================================================


class TestReviewTemplate:
    """Test POST /api/marketplace/templates/:id/review endpoint."""

    def test_review_valid(self, handler, mock_get_request, mock_post_request):
        # Get existing template ID
        browse_result = handler.handle("/api/marketplace/templates", {}, mock_get_request)
        browse_body, _ = parse_handler_result(browse_result)

        if browse_body["templates"]:
            template_id = browse_body["templates"][0]["id"]

            body = {
                "rating": 5,
                "title": "Great template!",
                "content": "This template saved me hours of work.",
            }
            mock_post_request.rfile = Mock()
            mock_post_request.rfile.read = Mock(return_value=json.dumps(body).encode())
            mock_post_request.headers["Content-Length"] = len(json.dumps(body))

            result = handler.handle(f"/api/marketplace/templates/{template_id}/review", {}, mock_post_request)
            response_body, status = parse_handler_result(result)

            assert status == 201
            assert response_body["success"] is True
            assert "review_id" in response_body


# ============================================================================
# Test Import Template
# ============================================================================


class TestImportTemplate:
    """Test POST /api/marketplace/templates/:id/import endpoint."""

    def test_import_valid(self, handler, mock_get_request, mock_post_request):
        # Get existing template ID
        browse_result = handler.handle("/api/marketplace/templates", {}, mock_get_request)
        browse_body, _ = parse_handler_result(browse_result)

        if browse_body["templates"]:
            template_id = browse_body["templates"][0]["id"]

            body = {}
            mock_post_request.rfile = Mock()
            mock_post_request.rfile.read = Mock(return_value=json.dumps(body).encode())
            mock_post_request.headers["Content-Length"] = len(json.dumps(body))

            result = handler.handle(f"/api/marketplace/templates/{template_id}/import", {}, mock_post_request)
            response_body, status = parse_handler_result(result)

            assert status == 200
            assert response_body["success"] is True
            assert "imported_id" in response_body

    def test_import_nonexistent_template(self, handler, mock_post_request):
        body = {}
        mock_post_request.rfile = Mock()
        mock_post_request.rfile.read = Mock(return_value=json.dumps(body).encode())
        mock_post_request.headers["Content-Length"] = len(json.dumps(body))

        result = handler.handle("/api/marketplace/templates/nonexistent-id/import", {}, mock_post_request)
        response_body, status = parse_handler_result(result)

        assert status == 404
        assert "error" in response_body


# ============================================================================
# Test Featured Templates
# ============================================================================


class TestFeaturedTemplates:
    """Test GET /api/marketplace/featured endpoint."""

    def test_get_featured(self, handler, mock_get_request):
        result = handler.handle("/api/marketplace/featured", {}, mock_get_request)
        body, status = parse_handler_result(result)

        assert status == 200
        assert "templates" in body
        assert isinstance(body["templates"], list)


# ============================================================================
# Test Trending Templates
# ============================================================================


class TestTrendingTemplates:
    """Test GET /api/marketplace/trending endpoint."""

    def test_get_trending(self, handler, mock_get_request):
        result = handler.handle("/api/marketplace/trending", {}, mock_get_request)
        body, status = parse_handler_result(result)

        assert status == 200
        assert "templates" in body
        assert isinstance(body["templates"], list)


# ============================================================================
# Test Categories
# ============================================================================


class TestCategories:
    """Test GET /api/marketplace/categories endpoint."""

    def test_get_categories(self, handler, mock_get_request):
        result = handler.handle("/api/marketplace/categories", {}, mock_get_request)
        body, status = parse_handler_result(result)

        assert status == 200
        assert "categories" in body
        assert isinstance(body["categories"], list)
        for category in body["categories"]:
            assert "id" in category
            assert "name" in category
            assert "count" in category


# ============================================================================
# Test MarketplaceTemplate Model
# ============================================================================


class TestMarketplaceTemplateModel:
    """Test MarketplaceTemplate dataclass."""

    def test_to_dict(self):
        template = MarketplaceTemplate(
            id="tpl-123",
            name="Test Template",
            description="A test template",
            author_id="test-user",
            author_name="Test User",
            category="testing",
            pattern="review_cycle",
            tags=["test"],
            download_count=10,
            rating=4.5,
            rating_count=2,
            created_at=1234567890.0,
            updated_at=1234567890.0,
        )

        data = template.to_dict()

        assert data["id"] == "tpl-123"
        assert data["name"] == "Test Template"
        assert data["rating"] == 4.5
        assert data["download_count"] == 10

    def test_to_summary(self):
        template = MarketplaceTemplate(
            id="tpl-123",
            name="Test Template",
            description="A test template",
            author_id="test-user",
            author_name="Test User",
            category="testing",
            pattern="review_cycle",
            tags=["test"],
            download_count=10,
            rating=4.5,
            rating_count=2,
            created_at=1234567890.0,
            updated_at=1234567890.0,
        )

        summary = template.to_summary()

        assert "id" in summary
        assert "name" in summary
        # Summary should be a subset of full dict
        assert len(summary) < len(template.to_dict())


# ============================================================================
# Test TemplateReview Model
# ============================================================================


class TestTemplateReviewModel:
    """Test TemplateReview dataclass."""

    def test_to_dict(self):
        review = TemplateReview(
            id="rev-123",
            template_id="tpl-456",
            user_id="user-789",
            user_name="Test User",
            rating=5,
            title="Great!",
            content="Really helpful template.",
            created_at=1234567890.0,
            helpful_count=3,
        )

        data = review.to_dict()

        assert data["id"] == "rev-123"
        assert data["rating"] == 5
        assert data["title"] == "Great!"
        assert data["helpful_count"] == 3
