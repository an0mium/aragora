"""
Tests for template marketplace handler endpoints.

Tests the template marketplace API handlers for:
- List/search templates
- Get template details
- Publish new templates
- Rate and review templates
- Import templates
- Featured and trending templates
- Categories
"""

import time
import uuid
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def sample_template():
    """Create a sample template for testing."""
    return {
        "id": "tpl-123",
        "name": "Test Template",
        "description": "A test template for unit testing",
        "category": "automation",
        "pattern": "sequential",
        "author_id": "user-1",
        "author_name": "Test Author",
        "version": "1.0.0",
        "tags": ["test", "automation"],
        "workflow_definition": {"steps": []},
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "documentation": "Test documentation",
        "examples": [],
        "rating": 4.5,
        "rating_count": 10,
        "download_count": 100,
        "is_featured": False,
        "is_verified": True,
        "created_at": time.time(),
        "updated_at": time.time(),
    }


@pytest.fixture
def sample_review():
    """Create a sample review for testing."""
    return {
        "id": "rev-123",
        "template_id": "tpl-123",
        "user_id": "user-2",
        "user_name": "Reviewer",
        "rating": 5,
        "title": "Great template!",
        "content": "This template works perfectly for my use case.",
        "helpful_count": 5,
        "created_at": time.time(),
    }


@pytest.fixture
def mock_marketplace_state(sample_template, sample_review):
    """Set up mock marketplace state."""
    from aragora.server.handlers.template_marketplace import (
        MarketplaceTemplate,
        TemplateReview,
        _marketplace_templates,
        _template_reviews,
        _user_ratings,
    )

    # Clear and populate mock data
    _marketplace_templates.clear()
    _template_reviews.clear()
    _user_ratings.clear()

    # Create template object
    template = MarketplaceTemplate(
        id=sample_template["id"],
        name=sample_template["name"],
        description=sample_template["description"],
        category=sample_template["category"],
        pattern=sample_template["pattern"],
        author_id=sample_template["author_id"],
        author_name=sample_template["author_name"],
        version=sample_template["version"],
        tags=sample_template["tags"],
        workflow_definition=sample_template["workflow_definition"],
        input_schema=sample_template["input_schema"],
        output_schema=sample_template["output_schema"],
        documentation=sample_template["documentation"],
        examples=sample_template["examples"],
        rating=sample_template["rating"],
        rating_count=sample_template["rating_count"],
        download_count=sample_template["download_count"],
        is_featured=sample_template["is_featured"],
        is_verified=sample_template["is_verified"],
    )
    _marketplace_templates[template.id] = template

    # Create review object
    review = TemplateReview(
        id=sample_review["id"],
        template_id=sample_review["template_id"],
        user_id=sample_review["user_id"],
        user_name=sample_review["user_name"],
        rating=sample_review["rating"],
        title=sample_review["title"],
        content=sample_review["content"],
        helpful_count=sample_review["helpful_count"],
        created_at=sample_review["created_at"],
    )
    _template_reviews[template.id] = [review]

    yield

    # Cleanup
    _marketplace_templates.clear()
    _template_reviews.clear()
    _user_ratings.clear()


class TestListTemplates:
    """Tests for listing templates."""

    def test_list_templates_returns_all(self, mock_marketplace_state):
        """Test listing all templates."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._list_templates({})

        assert result["success"] is True
        assert "templates" in result["data"]
        assert "pagination" in result["data"]
        assert len(result["data"]["templates"]) >= 1

    def test_list_templates_with_category_filter(self, mock_marketplace_state):
        """Test listing templates filtered by category."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._list_templates({"category": "automation"})

        assert result["success"] is True
        for template in result["data"]["templates"]:
            assert template["category"] == "automation"

    def test_list_templates_with_pattern_filter(self, mock_marketplace_state):
        """Test listing templates filtered by pattern."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._list_templates({"pattern": "sequential"})

        assert result["success"] is True

    def test_list_templates_with_search(self, mock_marketplace_state):
        """Test listing templates with search query."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._list_templates({"search": "test"})

        assert result["success"] is True

    def test_list_templates_with_tags_filter(self, mock_marketplace_state):
        """Test listing templates filtered by tags."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._list_templates({"tags": "test,automation"})

        assert result["success"] is True

    def test_list_templates_verified_only(self, mock_marketplace_state):
        """Test listing only verified templates."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._list_templates({"verified_only": "true"})

        assert result["success"] is True
        for template in result["data"]["templates"]:
            assert template.get("is_verified", False) is True

    def test_list_templates_sorting(self, mock_marketplace_state):
        """Test listing templates with different sort options."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()

        for sort_by in ["rating", "downloads", "newest", "name"]:
            result = handler._list_templates({"sort_by": sort_by})
            assert result["success"] is True, f"Failed for sort_by={sort_by}"

    def test_list_templates_pagination(self, mock_marketplace_state):
        """Test listing templates with pagination."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._list_templates({"limit": "10", "offset": "0"})

        assert result["success"] is True
        pagination = result["data"]["pagination"]
        assert pagination["limit"] == 10
        assert pagination["offset"] == 0

    def test_list_templates_empty(self):
        """Test listing templates when none exist."""
        from aragora.server.handlers.template_marketplace import (
            TemplateMarketplaceHandler,
            _marketplace_templates,
        )

        _marketplace_templates.clear()
        handler = TemplateMarketplaceHandler()
        result = handler._list_templates({})

        assert result["success"] is True
        # May have seeded templates, but should not error


class TestGetTemplate:
    """Tests for getting template details."""

    def test_get_template_found(self, mock_marketplace_state):
        """Test getting existing template."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._get_template("tpl-123")

        assert result["success"] is True
        assert result["data"]["id"] == "tpl-123"
        assert result["data"]["name"] == "Test Template"

    def test_get_template_not_found(self, mock_marketplace_state):
        """Test getting non-existent template returns 404."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._get_template("nonexistent")

        assert result["success"] is False
        assert result["status_code"] == 404


class TestPublishTemplate:
    """Tests for publishing templates."""

    def test_publish_template_success(self, mock_marketplace_state):
        """Test publishing a new template."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()
        mock_handler.get_json_body.return_value = {
            "name": "New Template",
            "description": "A brand new template",
            "category": "analytics",
            "pattern": "parallel",
            "workflow_definition": {"steps": []},
        }

        with patch.object(handler, "_check_rate_limit", return_value=True):
            result = handler._publish_template(mock_handler, "127.0.0.1")

        assert result["success"] is True
        assert "id" in result["data"]

    def test_publish_template_missing_required_fields(self, mock_marketplace_state):
        """Test publishing template with missing fields returns 400."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()
        mock_handler.get_json_body.return_value = {
            "name": "Incomplete Template",
            # Missing required fields
        }

        with patch.object(handler, "_check_rate_limit", return_value=True):
            result = handler._publish_template(mock_handler, "127.0.0.1")

        assert result["success"] is False
        assert result["status_code"] == 400

    def test_publish_template_rate_limited(self, mock_marketplace_state):
        """Test publishing template when rate limited."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()
        mock_handler.get_json_body.return_value = {
            "name": "Rate Limited Template",
            "description": "Should be rate limited",
            "category": "test",
            "pattern": "sequential",
            "workflow_definition": {},
        }

        with patch.object(handler, "_check_rate_limit", return_value=False):
            result = handler._publish_template(mock_handler, "127.0.0.1")

        assert result["success"] is False
        assert result["status_code"] == 429


class TestRateTemplate:
    """Tests for rating templates."""

    def test_rate_template_success(self, mock_marketplace_state):
        """Test rating a template."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()
        mock_handler.get_json_body.return_value = {
            "rating": 5,
            "user_id": "user-3",
        }

        with patch.object(handler, "_check_rate_limit", return_value=True):
            result = handler._rate_template("tpl-123", mock_handler, "127.0.0.1")

        assert result["success"] is True

    def test_rate_template_invalid_rating(self, mock_marketplace_state):
        """Test rating with invalid value returns 400."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()
        mock_handler.get_json_body.return_value = {
            "rating": 10,  # Invalid: should be 1-5
        }

        with patch.object(handler, "_check_rate_limit", return_value=True):
            result = handler._rate_template("tpl-123", mock_handler, "127.0.0.1")

        assert result["success"] is False
        assert result["status_code"] == 400

    def test_rate_template_not_found(self, mock_marketplace_state):
        """Test rating non-existent template returns 404."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()
        mock_handler.get_json_body.return_value = {"rating": 4}

        with patch.object(handler, "_check_rate_limit", return_value=True):
            result = handler._rate_template("nonexistent", mock_handler, "127.0.0.1")

        assert result["success"] is False
        assert result["status_code"] == 404

    def test_rate_template_updates_existing_rating(self, mock_marketplace_state):
        """Test updating an existing rating."""
        from aragora.server.handlers.template_marketplace import (
            TemplateMarketplaceHandler,
            _user_ratings,
        )

        # Set up existing rating
        _user_ratings["user-3"] = {"tpl-123": 3}

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()
        mock_handler.get_json_body.return_value = {
            "rating": 5,
            "user_id": "user-3",
        }

        with patch.object(handler, "_check_rate_limit", return_value=True):
            result = handler._rate_template("tpl-123", mock_handler, "127.0.0.1")

        assert result["success"] is True


class TestReviews:
    """Tests for template reviews."""

    def test_get_reviews(self, mock_marketplace_state):
        """Test getting reviews for a template."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._get_reviews("tpl-123", {})

        assert result["success"] is True
        assert "reviews" in result["data"]
        assert len(result["data"]["reviews"]) >= 1

    def test_get_reviews_pagination(self, mock_marketplace_state):
        """Test getting reviews with pagination."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._get_reviews("tpl-123", {"limit": "5", "offset": "0"})

        assert result["success"] is True
        assert "pagination" in result["data"]

    def test_get_reviews_not_found(self, mock_marketplace_state):
        """Test getting reviews for non-existent template."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._get_reviews("nonexistent", {})

        # Should return empty list, not error
        assert result["success"] is True
        assert result["data"]["reviews"] == []

    def test_submit_review_success(self, mock_marketplace_state):
        """Test submitting a review."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()
        mock_handler.get_json_body.return_value = {
            "rating": 4,
            "content": "Good template, works well.",
            "title": "Solid choice",
            "user_id": "user-4",
            "user_name": "New Reviewer",
        }

        with patch.object(handler, "_check_rate_limit", return_value=True):
            result = handler._submit_review("tpl-123", mock_handler, "127.0.0.1")

        assert result["success"] is True
        assert "id" in result["data"]

    def test_submit_review_missing_content(self, mock_marketplace_state):
        """Test submitting review without content returns 400."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()
        mock_handler.get_json_body.return_value = {
            "rating": 4,
            # Missing content
        }

        with patch.object(handler, "_check_rate_limit", return_value=True):
            result = handler._submit_review("tpl-123", mock_handler, "127.0.0.1")

        assert result["success"] is False
        assert result["status_code"] == 400


class TestImportTemplate:
    """Tests for importing templates."""

    def test_import_template_success(self, mock_marketplace_state):
        """Test importing a template."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()
        mock_handler.get_json_body.return_value = {
            "workspace_id": "ws-123",
        }

        result = handler._import_template("tpl-123", mock_handler)

        assert result["success"] is True
        assert "workflow_definition" in result["data"]

    def test_import_template_not_found(self, mock_marketplace_state):
        """Test importing non-existent template returns 404."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()
        mock_handler.get_json_body.return_value = {}

        result = handler._import_template("nonexistent", mock_handler)

        assert result["success"] is False
        assert result["status_code"] == 404

    def test_import_increments_download_count(self, mock_marketplace_state):
        """Test importing increments download count."""
        from aragora.server.handlers.template_marketplace import (
            TemplateMarketplaceHandler,
            _marketplace_templates,
        )

        initial_count = _marketplace_templates["tpl-123"].download_count

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()
        mock_handler.get_json_body.return_value = {}

        handler._import_template("tpl-123", mock_handler)

        assert _marketplace_templates["tpl-123"].download_count == initial_count + 1


class TestFeaturedAndTrending:
    """Tests for featured and trending templates."""

    def test_get_featured_templates(self, mock_marketplace_state):
        """Test getting featured templates."""
        from aragora.server.handlers.template_marketplace import (
            TemplateMarketplaceHandler,
            _marketplace_templates,
        )

        # Mark template as featured
        _marketplace_templates["tpl-123"].is_featured = True

        handler = TemplateMarketplaceHandler()
        result = handler._get_featured()

        assert result["success"] is True
        assert "templates" in result["data"]

    def test_get_trending_templates(self, mock_marketplace_state):
        """Test getting trending templates."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._get_trending({"period": "week"})

        assert result["success"] is True
        assert "templates" in result["data"]

    def test_get_trending_different_periods(self, mock_marketplace_state):
        """Test trending with different time periods."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()

        for period in ["week", "month"]:
            result = handler._get_trending({"period": period})
            assert result["success"] is True, f"Failed for period={period}"


class TestCategories:
    """Tests for template categories."""

    def test_get_categories(self, mock_marketplace_state):
        """Test getting all categories."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._get_categories()

        assert result["success"] is True
        assert "categories" in result["data"]

    def test_categories_include_counts(self, mock_marketplace_state):
        """Test categories include template counts."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        result = handler._get_categories()

        assert result["success"] is True
        for category in result["data"]["categories"]:
            assert "name" in category
            assert "count" in category


class TestHandlerRouting:
    """Tests for handler routing."""

    def test_can_handle_marketplace_paths(self):
        """Test handler recognizes marketplace paths."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()

        assert handler.can_handle("/api/v1/marketplace/templates")
        assert handler.can_handle("/api/v1/marketplace/templates/tpl-123")
        assert handler.can_handle("/api/v1/marketplace/featured")
        assert handler.can_handle("/api/v1/marketplace/trending")
        assert handler.can_handle("/api/v1/marketplace/categories")

    def test_cannot_handle_other_paths(self):
        """Test handler rejects non-marketplace paths."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()

        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/v2/marketplace")  # Wrong version
        assert not handler.can_handle("/health")

    def test_handle_routes_correctly(self, mock_marketplace_state):
        """Test handle method routes to correct handler."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/marketplace/templates"
        mock_handler.command = "GET"

        result = handler.handle("/api/v1/marketplace/templates", {}, mock_handler)

        assert result["success"] is True


class TestDataValidation:
    """Tests for data validation."""

    def test_rating_bounds(self, mock_marketplace_state):
        """Test rating must be between 1 and 5."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()

        # Test rating too low
        mock_handler.get_json_body.return_value = {"rating": 0}
        with patch.object(handler, "_check_rate_limit", return_value=True):
            result = handler._rate_template("tpl-123", mock_handler, "127.0.0.1")
        assert result["success"] is False

        # Test rating too high
        mock_handler.get_json_body.return_value = {"rating": 6}
        with patch.object(handler, "_check_rate_limit", return_value=True):
            result = handler._rate_template("tpl-123", mock_handler, "127.0.0.1")
        assert result["success"] is False

        # Test valid ratings
        for rating in [1, 2, 3, 4, 5]:
            mock_handler.get_json_body.return_value = {"rating": rating}
            with patch.object(handler, "_check_rate_limit", return_value=True):
                result = handler._rate_template("tpl-123", mock_handler, "127.0.0.1")
            assert result["success"] is True, f"Failed for rating={rating}"

    def test_limit_bounds(self, mock_marketplace_state):
        """Test limit parameter bounds."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()

        # Max limit should be 50
        result = handler._list_templates({"limit": "100"})
        assert result["success"] is True
        assert result["data"]["pagination"]["limit"] <= 50

    def test_template_id_generation(self, mock_marketplace_state):
        """Test template IDs are valid UUIDs."""
        from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

        handler = TemplateMarketplaceHandler()
        mock_handler = MagicMock()
        mock_handler.get_json_body.return_value = {
            "name": "UUID Test Template",
            "description": "Testing UUID generation",
            "category": "test",
            "pattern": "sequential",
            "workflow_definition": {},
        }

        with patch.object(handler, "_check_rate_limit", return_value=True):
            result = handler._publish_template(mock_handler, "127.0.0.1")

        assert result["success"] is True
        # ID should be a valid format (tpl-<uuid>)
        assert result["data"]["id"].startswith("tpl-")
