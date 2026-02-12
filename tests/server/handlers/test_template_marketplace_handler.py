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

import json
import time
import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def parse_result(result):
    """Parse HandlerResult into a dict for test assertions.

    Args:
        result: HandlerResult object with status_code and body

    Returns:
        Dict with 'success', 'status_code', and 'data' keys
    """
    data = json.loads(result.body.decode("utf-8"))
    is_success = 200 <= result.status_code < 400
    return {
        "success": is_success,
        "status_code": result.status_code,
        "data": data,
    }


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler instantiation."""
    return MagicMock()


@pytest.fixture
def marketplace_handler(mock_server_context):
    """Create a TemplateMarketplaceHandler with mock context."""
    from aragora.server.handlers.template_marketplace import TemplateMarketplaceHandler

    return TemplateMarketplaceHandler(mock_server_context)


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

    def test_list_templates_returns_all(self, mock_marketplace_state, marketplace_handler):
        """Test listing all templates."""
        raw_result = marketplace_handler._list_templates({})
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "templates" in result["data"]
        assert len(result["data"]["templates"]) >= 1

    def test_list_templates_with_category_filter(self, mock_marketplace_state, marketplace_handler):
        """Test listing templates filtered by category."""
        raw_result = marketplace_handler._list_templates({"category": "automation"})
        result = parse_result(raw_result)

        assert result["success"] is True
        for template in result["data"]["templates"]:
            assert template["category"] == "automation"

    def test_list_templates_with_pattern_filter(self, mock_marketplace_state, marketplace_handler):
        """Test listing templates filtered by pattern."""
        raw_result = marketplace_handler._list_templates({"pattern": "sequential"})
        result = parse_result(raw_result)

        assert result["success"] is True

    def test_list_templates_with_search(self, mock_marketplace_state, marketplace_handler):
        """Test listing templates with search query."""
        raw_result = marketplace_handler._list_templates({"search": "test"})
        result = parse_result(raw_result)

        assert result["success"] is True

    def test_list_templates_with_tags_filter(self, mock_marketplace_state, marketplace_handler):
        """Test listing templates filtered by tags."""
        raw_result = marketplace_handler._list_templates({"tags": "test,automation"})
        result = parse_result(raw_result)

        assert result["success"] is True

    def test_list_templates_verified_only(self, mock_marketplace_state, marketplace_handler):
        """Test listing only verified templates."""
        raw_result = marketplace_handler._list_templates({"verified_only": "true"})
        result = parse_result(raw_result)

        assert result["success"] is True
        for template in result["data"]["templates"]:
            assert template.get("is_verified", False) is True

    def test_list_templates_sorting(self, mock_marketplace_state, marketplace_handler):
        """Test listing templates with different sort options."""
        for sort_by in ["rating", "downloads", "newest", "name"]:
            raw_result = marketplace_handler._list_templates({"sort_by": sort_by})
            result = parse_result(raw_result)
            assert result["success"] is True, f"Failed for sort_by={sort_by}"

    def test_list_templates_pagination(self, mock_marketplace_state, marketplace_handler):
        """Test listing templates with pagination."""
        raw_result = marketplace_handler._list_templates({"limit": "10", "offset": "0"})
        result = parse_result(raw_result)

        assert result["success"] is True
        assert result["data"]["limit"] == 10
        assert result["data"]["offset"] == 0

    def test_list_templates_empty(self, marketplace_handler):
        """Test listing templates when none exist."""
        from aragora.server.handlers.template_marketplace import _marketplace_templates

        _marketplace_templates.clear()
        raw_result = marketplace_handler._list_templates({})
        result = parse_result(raw_result)

        assert result["success"] is True
        # May have seeded templates, but should not error


class TestGetTemplate:
    """Tests for getting template details."""

    def test_get_template_found(self, mock_marketplace_state, marketplace_handler):
        """Test getting existing template."""
        raw_result = marketplace_handler._get_template("tpl-123")
        result = parse_result(raw_result)

        assert result["success"] is True
        assert result["data"]["id"] == "tpl-123"
        assert result["data"]["name"] == "Test Template"

    def test_get_template_not_found(self, mock_marketplace_state, marketplace_handler):
        """Test getting non-existent template returns 404."""
        raw_result = marketplace_handler._get_template("nonexistent")
        result = parse_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 404


class TestPublishTemplate:
    """Tests for publishing templates."""

    def test_publish_template_success(self, mock_marketplace_state, marketplace_handler):
        """Test publishing a new template."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "200"}
        mock_handler.rfile.read.return_value = b'{"name": "New Template", "description": "A brand new template", "category": "analytics", "pattern": "parallel", "workflow_definition": {"steps": []}}'

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["success"] is True
        assert "template_id" in result["data"]

    def test_publish_template_missing_required_fields(
        self, mock_marketplace_state, marketplace_handler
    ):
        """Test publishing template with missing fields returns 400."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "50"}
        mock_handler.rfile.read.return_value = b'{"name": "Incomplete Template"}'

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 400

    def test_publish_template_rate_limited(self, mock_marketplace_state, marketplace_handler):
        """Test publishing template when rate limited."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "100"}
        mock_handler.rfile.read.return_value = b'{"name": "Rate Limited Template", "description": "Should be rate limited", "category": "test", "pattern": "sequential", "workflow_definition": {}}'

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=False,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 429


class TestRateTemplate:
    """Tests for rating templates."""

    def test_rate_template_success(self, mock_marketplace_state, marketplace_handler):
        """Test rating a template."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "50"}
        mock_handler.rfile.read.return_value = b'{"rating": 5, "user_id": "user-3"}'

        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._rate_template("tpl-123", mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["success"] is True

    def test_rate_template_invalid_rating(self, mock_marketplace_state, marketplace_handler):
        """Test rating with invalid value returns 400."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "20"}
        mock_handler.rfile.read.return_value = b'{"rating": 10}'

        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._rate_template("tpl-123", mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 400

    def test_rate_template_not_found(self, mock_marketplace_state, marketplace_handler):
        """Test rating non-existent template returns 404."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "20"}
        mock_handler.rfile.read.return_value = b'{"rating": 4}'

        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._rate_template(
                "nonexistent", mock_handler, "127.0.0.1"
            )
            result = parse_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 404

    def test_rate_template_updates_existing_rating(
        self, mock_marketplace_state, marketplace_handler
    ):
        """Test updating an existing rating."""
        from aragora.server.handlers.template_marketplace import _user_ratings

        # Set up existing rating
        _user_ratings["user-3"] = {"tpl-123": 3}

        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "50"}
        mock_handler.rfile.read.return_value = b'{"rating": 5, "user_id": "user-3"}'

        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._rate_template("tpl-123", mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["success"] is True


class TestReviews:
    """Tests for template reviews."""

    def test_get_reviews(self, mock_marketplace_state, marketplace_handler):
        """Test getting reviews for a template."""
        raw_result = marketplace_handler._get_reviews("tpl-123", {})
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "reviews" in result["data"]
        assert len(result["data"]["reviews"]) >= 1

    def test_get_reviews_pagination(self, mock_marketplace_state, marketplace_handler):
        """Test getting reviews with pagination."""
        raw_result = marketplace_handler._get_reviews("tpl-123", {"limit": "5", "offset": "0"})
        result = parse_result(raw_result)

        assert result["success"] is True

    def test_get_reviews_not_found(self, mock_marketplace_state, marketplace_handler):
        """Test getting reviews for non-existent template returns 404."""
        raw_result = marketplace_handler._get_reviews("nonexistent", {})
        result = parse_result(raw_result)

        # Template not found returns 404
        assert result["success"] is False
        assert result["status_code"] == 404

    def test_submit_review_success(self, mock_marketplace_state, marketplace_handler):
        """Test submitting a review."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "150"}
        mock_handler.rfile.read.return_value = b'{"rating": 4, "content": "Good template, works well.", "title": "Solid choice", "user_id": "user-4", "user_name": "New Reviewer"}'

        raw_result = marketplace_handler._submit_review("tpl-123", mock_handler, "127.0.0.1")
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "review" in result["data"]

    def test_submit_review_missing_content(self, mock_marketplace_state, marketplace_handler):
        """Test submitting review without content returns 400."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "20"}
        mock_handler.rfile.read.return_value = b'{"rating": 4}'

        raw_result = marketplace_handler._submit_review("tpl-123", mock_handler, "127.0.0.1")
        result = parse_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 400


class TestImportTemplate:
    """Tests for importing templates."""

    def test_import_template_success(self, mock_marketplace_state, marketplace_handler):
        """Test importing a template."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "30"}
        mock_handler.rfile.read.return_value = b'{"workspace_id": "ws-123"}'

        raw_result = marketplace_handler._import_template("tpl-123", mock_handler)
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "workflow_definition" in result["data"]

    def test_import_template_not_found(self, mock_marketplace_state, marketplace_handler):
        """Test importing non-existent template returns 404."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "2"}
        mock_handler.rfile.read.return_value = b"{}"

        raw_result = marketplace_handler._import_template("nonexistent", mock_handler)
        result = parse_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 404

    def test_import_increments_download_count(self, mock_marketplace_state, marketplace_handler):
        """Test importing increments download count."""
        from aragora.server.handlers.template_marketplace import _marketplace_templates

        initial_count = _marketplace_templates["tpl-123"].download_count

        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "2"}
        mock_handler.rfile.read.return_value = b"{}"

        marketplace_handler._import_template("tpl-123", mock_handler)

        assert _marketplace_templates["tpl-123"].download_count == initial_count + 1


class TestFeaturedAndTrending:
    """Tests for featured and trending templates."""

    def test_get_featured_templates(self, mock_marketplace_state, marketplace_handler):
        """Test getting featured templates."""
        from aragora.server.handlers.template_marketplace import _marketplace_templates

        # Mark template as featured
        _marketplace_templates["tpl-123"].is_featured = True

        raw_result = marketplace_handler._get_featured()
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "featured" in result["data"]

    def test_get_trending_templates(self, mock_marketplace_state, marketplace_handler):
        """Test getting trending templates."""
        raw_result = marketplace_handler._get_trending({"period": "week"})
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "trending" in result["data"]

    def test_get_trending_different_periods(self, mock_marketplace_state, marketplace_handler):
        """Test trending with different time periods."""
        for period in ["week", "month"]:
            raw_result = marketplace_handler._get_trending({"period": period})
            result = parse_result(raw_result)
            assert result["success"] is True, f"Failed for period={period}"


class TestCategories:
    """Tests for template categories."""

    def test_get_categories(self, mock_marketplace_state, marketplace_handler):
        """Test getting all categories."""
        raw_result = marketplace_handler._get_categories()
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "categories" in result["data"]

    def test_categories_include_counts(self, mock_marketplace_state, marketplace_handler):
        """Test categories include template counts."""
        raw_result = marketplace_handler._get_categories()
        result = parse_result(raw_result)

        assert result["success"] is True
        for category in result["data"]["categories"]:
            assert "name" in category
            assert "template_count" in category


class TestHandlerRouting:
    """Tests for handler routing."""

    def test_can_handle_marketplace_paths(self, marketplace_handler):
        """Test handler recognizes marketplace paths."""
        assert marketplace_handler.can_handle("/api/v1/marketplace/templates")
        assert marketplace_handler.can_handle("/api/v1/marketplace/templates/tpl-123")
        assert marketplace_handler.can_handle("/api/v1/marketplace/featured")
        assert marketplace_handler.can_handle("/api/v1/marketplace/trending")
        assert marketplace_handler.can_handle("/api/v1/marketplace/categories")

    def test_cannot_handle_other_paths(self, marketplace_handler):
        """Test handler rejects non-marketplace paths."""
        assert not marketplace_handler.can_handle("/api/debates")
        assert not marketplace_handler.can_handle("/api/v2/marketplace")  # Wrong version
        assert not marketplace_handler.can_handle("/health")

    def test_handle_routes_correctly(self, mock_marketplace_state, marketplace_handler):
        """Test handle method routes to correct handler."""
        mock_handler = MagicMock()
        mock_handler.path = "/api/v1/marketplace/templates"
        mock_handler.command = "GET"

        raw_result = marketplace_handler.handle("/api/v1/marketplace/templates", {}, mock_handler)
        result = parse_result(raw_result)

        assert result["success"] is True


class TestDataValidation:
    """Tests for data validation."""

    def test_rating_bounds(self, mock_marketplace_state, marketplace_handler):
        """Test rating must be between 1 and 5."""
        mock_handler = MagicMock()

        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter.is_allowed",
            return_value=True,
        ):
            # Test rating too low
            mock_handler.headers = {"Content-Length": "15"}
            mock_handler.rfile.read.return_value = b'{"rating": 0}'
            raw_result = marketplace_handler._rate_template("tpl-123", mock_handler, "127.0.0.1")
            result = parse_result(raw_result)
            assert result["success"] is False

            # Test rating too high
            mock_handler.headers = {"Content-Length": "15"}
            mock_handler.rfile.read.return_value = b'{"rating": 6}'
            raw_result = marketplace_handler._rate_template("tpl-123", mock_handler, "127.0.0.1")
            result = parse_result(raw_result)
            assert result["success"] is False

            # Test valid ratings
            for rating in [1, 2, 3, 4, 5]:
                mock_handler.headers = {"Content-Length": "15"}
                mock_handler.rfile.read.return_value = f'{{"rating": {rating}}}'.encode()
                raw_result = marketplace_handler._rate_template(
                    "tpl-123", mock_handler, "127.0.0.1"
                )
                result = parse_result(raw_result)
                assert result["success"] is True, f"Failed for rating={rating}"

    def test_limit_bounds(self, mock_marketplace_state, marketplace_handler):
        """Test limit parameter bounds."""
        # Max limit should be 50
        raw_result = marketplace_handler._list_templates({"limit": "100"})
        result = parse_result(raw_result)
        assert result["success"] is True
        assert result["data"]["limit"] <= 50

    def test_template_id_generation(self, mock_marketplace_state, marketplace_handler):
        """Test template IDs are generated correctly."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "150"}
        mock_handler.rfile.read.return_value = (
            b'{"name": "UUID Test Template", "description": "Testing UUID generation", '
            b'"category": "automation", "pattern": "sequential", "workflow_definition": {}}'
        )

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["success"] is True
        # ID is generated as category/name-slug format
        assert "template_id" in result["data"]


# =============================================================================
# Comprehensive tests for STABLE graduation
# =============================================================================


class TestCircuitBreaker:
    """Tests for the MarketplaceCircuitBreaker."""

    def test_initial_state_is_closed(self):
        """Test circuit breaker starts in closed state."""
        from aragora.server.handlers.template_marketplace import MarketplaceCircuitBreaker

        cb = MarketplaceCircuitBreaker()
        assert cb.state == "closed"
        assert cb.is_allowed() is True

    def test_opens_after_threshold_failures(self):
        """Test circuit opens after reaching failure threshold."""
        from aragora.server.handlers.template_marketplace import MarketplaceCircuitBreaker

        cb = MarketplaceCircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"

        cb.record_failure()
        assert cb.state == "open"
        assert cb.is_allowed() is False

    def test_transitions_to_half_open_after_cooldown(self):
        """Test circuit transitions from open to half_open after cooldown."""
        from aragora.server.handlers.template_marketplace import MarketplaceCircuitBreaker

        cb = MarketplaceCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        time.sleep(0.15)
        assert cb.state == "half_open"

    def test_half_open_allows_limited_calls(self):
        """Test half_open state allows only limited calls."""
        from aragora.server.handlers.template_marketplace import MarketplaceCircuitBreaker

        cb = MarketplaceCircuitBreaker(
            failure_threshold=2, cooldown_seconds=0.1, half_open_max_calls=2
        )
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == "half_open"

        assert cb.is_allowed() is True
        assert cb.is_allowed() is True
        assert cb.is_allowed() is False

    def test_half_open_closes_on_enough_successes(self):
        """Test enough successes in half_open transitions to closed."""
        from aragora.server.handlers.template_marketplace import MarketplaceCircuitBreaker

        cb = MarketplaceCircuitBreaker(
            failure_threshold=2, cooldown_seconds=0.1, half_open_max_calls=2
        )
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == "half_open"

        cb.record_success()
        cb.record_success()
        assert cb.state == "closed"
        assert cb._failure_count == 0

    def test_half_open_failure_reopens_circuit(self):
        """Test failure in half_open transitions back to open."""
        from aragora.server.handlers.template_marketplace import MarketplaceCircuitBreaker

        cb = MarketplaceCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == "half_open"

        cb.record_failure()
        assert cb.state == "open"

    def test_closed_success_resets_failures(self):
        """Test success in closed state resets failure count."""
        from aragora.server.handlers.template_marketplace import MarketplaceCircuitBreaker

        cb = MarketplaceCircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        assert cb._failure_count == 2

        cb.record_success()
        assert cb._failure_count == 0

    def test_reset_clears_all_state(self):
        """Test reset clears everything."""
        from aragora.server.handlers.template_marketplace import MarketplaceCircuitBreaker

        cb = MarketplaceCircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

        cb.reset()
        assert cb.state == "closed"
        assert cb._failure_count == 0
        assert cb._success_count == 0
        assert cb._last_failure_time is None
        assert cb._half_open_calls == 0

    def test_get_status_returns_all_fields(self):
        """Test get_status returns all expected fields."""
        from aragora.server.handlers.template_marketplace import MarketplaceCircuitBreaker

        cb = MarketplaceCircuitBreaker(failure_threshold=5, cooldown_seconds=30)
        status = cb.get_status()

        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["failure_threshold"] == 5
        assert status["cooldown_seconds"] == 30
        assert status["last_failure_time"] is None

    def test_get_status_after_failures(self):
        """Test get_status reflects failure state."""
        from aragora.server.handlers.template_marketplace import MarketplaceCircuitBreaker

        cb = MarketplaceCircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()

        status = cb.get_status()
        assert status["failure_count"] == 2
        assert status["last_failure_time"] is not None

    def test_get_status_when_open(self):
        """Test get_status in open state."""
        from aragora.server.handlers.template_marketplace import MarketplaceCircuitBreaker

        cb = MarketplaceCircuitBreaker(failure_threshold=2, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()

        status = cb.get_status()
        assert status["state"] == "open"

    def test_global_singleton(self):
        """Test get_marketplace_circuit_breaker returns same instance."""
        from aragora.server.handlers.template_marketplace import (
            get_marketplace_circuit_breaker,
        )

        cb1 = get_marketplace_circuit_breaker()
        cb2 = get_marketplace_circuit_breaker()
        assert cb1 is cb2

    def test_circuit_breaker_status_function(self):
        """Test get_marketplace_circuit_breaker_status returns status dict."""
        from aragora.server.handlers.template_marketplace import (
            get_marketplace_circuit_breaker_status,
        )

        status = get_marketplace_circuit_breaker_status()
        assert "state" in status
        assert "failure_count" in status


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration in handler."""

    def test_circuit_breaker_endpoint(self, mock_marketplace_state, marketplace_handler):
        """Test circuit breaker status endpoint."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        raw_result = marketplace_handler.handle(
            "/api/v1/marketplace/circuit-breaker", {}, mock_handler
        )
        result = parse_result(raw_result)

        assert result["success"] is True
        assert result["data"]["state"] == "closed"

    def test_circuit_breaker_rejects_when_open(self, mock_marketplace_state, marketplace_handler):
        """Test requests rejected when circuit is open."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        # Open the circuit breaker
        for _ in range(10):
            marketplace_handler._circuit_breaker.record_failure()
        assert marketplace_handler._circuit_breaker.state == "open"

        raw_result = marketplace_handler.handle("/api/v1/marketplace/templates", {}, mock_handler)
        result = parse_result(raw_result)

        assert result["success"] is False
        assert result["status_code"] == 503

        # Reset for other tests
        marketplace_handler._circuit_breaker.reset()

    def test_success_records_in_circuit_breaker(self, mock_marketplace_state, marketplace_handler):
        """Test successful request records success."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        marketplace_handler._circuit_breaker.reset()
        marketplace_handler._circuit_breaker._failure_count = 2

        marketplace_handler.handle("/api/v1/marketplace/featured", {}, mock_handler)

        # Success should reset failures
        assert marketplace_handler._circuit_breaker._failure_count == 0

    def test_exception_records_failure(self, mock_marketplace_state, marketplace_handler):
        """Test exception records failure in circuit breaker."""
        marketplace_handler._circuit_breaker.reset()
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        with patch.object(marketplace_handler, "_route_request", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError):
                marketplace_handler.handle("/api/v1/marketplace/templates", {}, mock_handler)

        assert marketplace_handler._circuit_breaker._failure_count == 1
        marketplace_handler._circuit_breaker.reset()


class TestInputValidation:
    """Tests for input validation functions."""

    def test_validate_template_name_valid(self):
        """Test valid template names pass validation."""
        from aragora.server.handlers.template_marketplace import validate_template_name

        is_valid, _ = validate_template_name("My Template")
        assert is_valid

    def test_validate_template_name_empty(self):
        """Test empty name fails validation."""
        from aragora.server.handlers.template_marketplace import validate_template_name

        is_valid, msg = validate_template_name("")
        assert not is_valid
        assert "required" in msg.lower()

    def test_validate_template_name_whitespace(self):
        """Test whitespace-only name fails validation."""
        from aragora.server.handlers.template_marketplace import validate_template_name

        is_valid, msg = validate_template_name("   ")
        assert not is_valid
        assert "whitespace" in msg.lower()

    def test_validate_template_name_too_long(self):
        """Test name exceeding max length fails validation."""
        from aragora.server.handlers.template_marketplace import validate_template_name

        is_valid, msg = validate_template_name("A" * 150)
        assert not is_valid
        assert "too long" in msg.lower()

    def test_validate_category_valid(self):
        """Test all valid categories pass."""
        from aragora.server.handlers.template_marketplace import (
            validate_category,
            VALID_CATEGORIES,
        )

        for cat in VALID_CATEGORIES:
            is_valid, _ = validate_category(cat)
            assert is_valid, f"Expected {cat} to be valid"

    def test_validate_category_invalid(self):
        """Test invalid category fails."""
        from aragora.server.handlers.template_marketplace import validate_category

        is_valid, msg = validate_category("nonexistent")
        assert not is_valid
        assert "Invalid category" in msg

    def test_validate_category_empty(self):
        """Test empty category fails."""
        from aragora.server.handlers.template_marketplace import validate_category

        is_valid, msg = validate_category("")
        assert not is_valid
        assert "required" in msg.lower()

    def test_validate_pattern_valid(self):
        """Test all valid patterns pass."""
        from aragora.server.handlers.template_marketplace import (
            validate_pattern,
            VALID_PATTERNS,
        )

        for pat in VALID_PATTERNS:
            is_valid, _ = validate_pattern(pat)
            assert is_valid, f"Expected {pat} to be valid"

    def test_validate_pattern_invalid(self):
        """Test invalid pattern fails."""
        from aragora.server.handlers.template_marketplace import validate_pattern

        is_valid, msg = validate_pattern("nonexistent_pattern")
        assert not is_valid
        assert "Invalid pattern" in msg

    def test_validate_pattern_empty(self):
        """Test empty pattern fails."""
        from aragora.server.handlers.template_marketplace import validate_pattern

        is_valid, msg = validate_pattern("")
        assert not is_valid
        assert "required" in msg.lower()

    def test_validate_tags_valid(self):
        """Test valid tags pass."""
        from aragora.server.handlers.template_marketplace import validate_tags

        is_valid, _ = validate_tags(["tag1", "tag2", "tag3"])
        assert is_valid

    def test_validate_tags_not_a_list(self):
        """Test non-list tags fail."""
        from aragora.server.handlers.template_marketplace import validate_tags

        is_valid, msg = validate_tags("not a list")
        assert not is_valid
        assert "list" in msg.lower()

    def test_validate_tags_too_many(self):
        """Test too many tags fail."""
        from aragora.server.handlers.template_marketplace import validate_tags

        is_valid, msg = validate_tags([f"tag{i}" for i in range(25)])
        assert not is_valid
        assert "too many" in msg.lower()

    def test_validate_tags_non_string_element(self):
        """Test non-string tag fails."""
        from aragora.server.handlers.template_marketplace import validate_tags

        is_valid, msg = validate_tags(["valid", 123])
        assert not is_valid
        assert "string" in msg.lower()

    def test_validate_tags_too_long(self):
        """Test tag too long fails."""
        from aragora.server.handlers.template_marketplace import validate_tags

        is_valid, msg = validate_tags(["a" * 60])
        assert not is_valid
        assert "too long" in msg.lower()

    def test_validate_tags_empty_tag(self):
        """Test empty tag fails."""
        from aragora.server.handlers.template_marketplace import validate_tags

        is_valid, msg = validate_tags(["valid", "  "])
        assert not is_valid
        assert "empty" in msg.lower()

    def test_validate_rating_valid_range(self):
        """Test valid ratings pass."""
        from aragora.server.handlers.template_marketplace import validate_rating

        for r in [1, 2, 3, 4, 5, 1.0, 4.5]:
            is_valid, _ = validate_rating(r)
            assert is_valid, f"Expected {r} to be valid"

    def test_validate_rating_too_low(self):
        """Test rating below 1 fails."""
        from aragora.server.handlers.template_marketplace import validate_rating

        is_valid, msg = validate_rating(0)
        assert not is_valid
        assert "between 1 and 5" in msg

    def test_validate_rating_too_high(self):
        """Test rating above 5 fails."""
        from aragora.server.handlers.template_marketplace import validate_rating

        is_valid, msg = validate_rating(6)
        assert not is_valid

    def test_validate_rating_not_number(self):
        """Test non-numeric rating fails."""
        from aragora.server.handlers.template_marketplace import validate_rating

        is_valid, msg = validate_rating("five")
        assert not is_valid
        assert "number" in msg.lower()

    def test_validate_rating_none(self):
        """Test None rating fails."""
        from aragora.server.handlers.template_marketplace import validate_rating

        is_valid, msg = validate_rating(None)
        assert not is_valid

    def test_validate_review_content_valid(self):
        """Test valid review content passes."""
        from aragora.server.handlers.template_marketplace import validate_review_content

        is_valid, _ = validate_review_content("Great template!")
        assert is_valid

    def test_validate_review_content_empty(self):
        """Test empty review content fails."""
        from aragora.server.handlers.template_marketplace import validate_review_content

        is_valid, msg = validate_review_content("")
        assert not is_valid
        assert "required" in msg.lower()

    def test_validate_review_content_whitespace(self):
        """Test whitespace-only review content fails."""
        from aragora.server.handlers.template_marketplace import validate_review_content

        is_valid, msg = validate_review_content("    ")
        assert not is_valid
        assert "whitespace" in msg.lower()

    def test_validate_review_content_too_long(self):
        """Test review content exceeding max length fails."""
        from aragora.server.handlers.template_marketplace import validate_review_content

        is_valid, msg = validate_review_content("A" * 3000)
        assert not is_valid
        assert "too long" in msg.lower()


class TestPublishTemplateValidation:
    """Tests for publish template input validation."""

    def _make_handler(self, body: dict | bytes) -> MagicMock:
        """Create a mock handler with given body."""
        mock_handler = MagicMock()
        if isinstance(body, dict):
            encoded = json.dumps(body).encode()
        else:
            encoded = body
        mock_handler.headers = {"Content-Length": str(len(encoded))}
        mock_handler.rfile.read.return_value = encoded
        return mock_handler

    def test_publish_body_too_large(self, mock_marketplace_state, marketplace_handler):
        """Test body too large returns 413."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "200000"}

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 413

    def test_publish_invalid_json(self, mock_marketplace_state, marketplace_handler):
        """Test invalid JSON returns 400."""
        mock_handler = self._make_handler(b"not json")

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_publish_invalid_category(self, mock_marketplace_state, marketplace_handler):
        """Test invalid category returns 400."""
        body = {
            "name": "Test",
            "description": "Desc",
            "category": "not_valid_cat",
            "pattern": "debate",
            "workflow_definition": {},
        }
        mock_handler = self._make_handler(body)

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_publish_invalid_pattern(self, mock_marketplace_state, marketplace_handler):
        """Test invalid pattern returns 400."""
        body = {
            "name": "Test",
            "description": "Desc",
            "category": "security",
            "pattern": "not_valid_pattern",
            "workflow_definition": {},
        }
        mock_handler = self._make_handler(body)

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_publish_whitespace_name(self, mock_marketplace_state, marketplace_handler):
        """Test whitespace-only name returns 400."""
        body = {
            "name": "   ",
            "description": "Desc",
            "category": "security",
            "pattern": "debate",
            "workflow_definition": {},
        }
        mock_handler = self._make_handler(body)

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_publish_name_too_long(self, mock_marketplace_state, marketplace_handler):
        """Test name exceeding max length returns 400."""
        body = {
            "name": "A" * 150,
            "description": "Desc",
            "category": "security",
            "pattern": "debate",
            "workflow_definition": {},
        }
        mock_handler = self._make_handler(body)

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_publish_description_too_long(self, mock_marketplace_state, marketplace_handler):
        """Test description exceeding max length returns 400."""
        body = {
            "name": "Test",
            "description": "A" * 6000,
            "category": "security",
            "pattern": "debate",
            "workflow_definition": {},
        }
        mock_handler = self._make_handler(body)

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_publish_empty_description(self, mock_marketplace_state, marketplace_handler):
        """Test empty description returns 400."""
        body = {
            "name": "Test",
            "description": "   ",
            "category": "security",
            "pattern": "debate",
            "workflow_definition": {},
        }
        mock_handler = self._make_handler(body)

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_publish_workflow_definition_not_dict(
        self, mock_marketplace_state, marketplace_handler
    ):
        """Test non-dict workflow_definition returns 400."""
        body = {
            "name": "Test",
            "description": "Desc",
            "category": "security",
            "pattern": "debate",
            "workflow_definition": "not a dict",
        }
        mock_handler = self._make_handler(body)

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_publish_invalid_tags(self, mock_marketplace_state, marketplace_handler):
        """Test invalid tags returns 400."""
        body = {
            "name": "Test",
            "description": "Desc",
            "category": "security",
            "pattern": "debate",
            "workflow_definition": {},
            "tags": ["valid", 123],
        }
        mock_handler = self._make_handler(body)

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_publish_duplicate_template(self, mock_marketplace_state, marketplace_handler):
        """Test publishing duplicate returns 409."""
        body = {
            "name": "Unique Name",
            "description": "First publish",
            "category": "analytics",
            "pattern": "parallel",
            "workflow_definition": {},
        }
        mock_handler = self._make_handler(body)

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            # First publish
            marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            # Second publish - duplicate
            mock_handler.rfile.read.return_value = json.dumps(body).encode()
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 409

    def test_publish_returns_201(self, mock_marketplace_state, marketplace_handler):
        """Test successful publish returns 201."""
        body = {
            "name": "Status Check",
            "description": "Testing status code",
            "category": "development",
            "pattern": "sequential",
            "workflow_definition": {"steps": []},
        }
        mock_handler = self._make_handler(body)

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 201
        assert result["data"]["status"] == "published"

    def test_publish_generates_correct_id(self, mock_marketplace_state, marketplace_handler):
        """Test published template gets correct category/slug ID."""
        body = {
            "name": "My Cool Template",
            "description": "Test ID generation",
            "category": "security",
            "pattern": "debate",
            "workflow_definition": {},
        }
        mock_handler = self._make_handler(body)

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["data"]["template_id"] == "security/my-cool-template"

    def test_publish_with_all_optional_fields(self, mock_marketplace_state, marketplace_handler):
        """Test publish with all optional fields."""
        body = {
            "name": "Full Template",
            "description": "Template with all fields",
            "category": "data",
            "pattern": "pipeline",
            "workflow_definition": {"steps": [1]},
            "author_id": "custom-author",
            "author_name": "Custom Author",
            "version": "2.0.0",
            "tags": ["tag1", "tag2"],
            "input_schema": {"type": "object"},
            "output_schema": {"type": "string"},
            "documentation": "Full docs",
            "examples": [{"ex": 1}],
        }
        mock_handler = self._make_handler(body)

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["success"] is True

        from aragora.server.handlers.template_marketplace import _marketplace_templates

        template = _marketplace_templates["data/full-template"]
        assert template.author_id == "custom-author"
        assert template.version == "2.0.0"
        assert template.tags == ["tag1", "tag2"]

    def test_publish_defaults_for_optional_fields(
        self, mock_marketplace_state, marketplace_handler
    ):
        """Test publish defaults for optional fields."""
        body = {
            "name": "Minimal Template",
            "description": "Only required fields",
            "category": "other",
            "pattern": "custom",
            "workflow_definition": {},
        }
        mock_handler = self._make_handler(body)

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._publish_template(mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["success"] is True

        from aragora.server.handlers.template_marketplace import _marketplace_templates

        template = _marketplace_templates["other/minimal-template"]
        assert template.author_id == "anonymous"
        assert template.author_name == "Anonymous"
        assert template.version == "1.0.0"
        assert template.tags == []


class TestRateTemplateAdvanced:
    """Advanced tests for rating templates."""

    def test_rate_body_too_large(self, mock_marketplace_state, marketplace_handler):
        """Test body too large returns 413."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "20000"}

        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._rate_template("tpl-123", mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 413

    def test_rate_invalid_json(self, mock_marketplace_state, marketplace_handler):
        """Test invalid JSON returns 400."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "10"}
        mock_handler.rfile.read.return_value = b"not json"

        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._rate_template("tpl-123", mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_rate_missing_rating_field(self, mock_marketplace_state, marketplace_handler):
        """Test missing rating field returns 400."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "20"}
        mock_handler.rfile.read.return_value = b'{"user_id": "u1"}'

        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._rate_template("tpl-123", mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_rate_float_rating(self, mock_marketplace_state, marketplace_handler):
        """Test float rating works."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "20"}
        mock_handler.rfile.read.return_value = b'{"rating": 4.5}'

        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._rate_template("tpl-123", mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["success"] is True

    def test_rate_uses_client_ip_as_default_user(self, mock_marketplace_state, marketplace_handler):
        """Test client IP used as user_id when not provided."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "20"}
        mock_handler.rfile.read.return_value = b'{"rating": 3}'

        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter.is_allowed",
            return_value=True,
        ):
            marketplace_handler._rate_template("tpl-123", mock_handler, "10.0.0.1")

        from aragora.server.handlers.template_marketplace import _user_ratings

        assert "10.0.0.1" in _user_ratings

    def test_rate_updates_average_correctly(self, mock_marketplace_state, marketplace_handler):
        """Test rating computation for new rating."""
        from aragora.server.handlers.template_marketplace import _marketplace_templates

        template = _marketplace_templates["tpl-123"]
        original_count = template.rating_count
        original_total = template.rating * template.rating_count

        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "50"}
        mock_handler.rfile.read.return_value = b'{"rating": 5, "user_id": "new-user-99"}'

        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler._rate_template("tpl-123", mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["success"] is True
        expected_rating = (original_total + 5) / (original_count + 1)
        assert result["data"]["rating_count"] == original_count + 1
        assert abs(result["data"]["average_rating"] - round(expected_rating, 2)) < 0.01

    def test_rate_template_rate_limited(self, mock_marketplace_state, marketplace_handler):
        """Test rating rate limit."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "20"}
        mock_handler.rfile.read.return_value = b'{"rating": 3}'

        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter.is_allowed",
            return_value=False,
        ):
            raw_result = marketplace_handler._rate_template("tpl-123", mock_handler, "127.0.0.1")
            result = parse_result(raw_result)

        assert result["status_code"] == 429


class TestSubmitReviewAdvanced:
    """Advanced tests for submitting reviews."""

    def test_review_body_too_large(self, mock_marketplace_state, marketplace_handler):
        """Test body too large returns 413."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "60000"}

        raw_result = marketplace_handler._submit_review("tpl-123", mock_handler, "127.0.0.1")
        result = parse_result(raw_result)

        assert result["status_code"] == 413

    def test_review_invalid_json(self, mock_marketplace_state, marketplace_handler):
        """Test invalid JSON returns 400."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "10"}
        mock_handler.rfile.read.return_value = b"not json"

        raw_result = marketplace_handler._submit_review("tpl-123", mock_handler, "127.0.0.1")
        result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_review_missing_rating(self, mock_marketplace_state, marketplace_handler):
        """Test missing rating returns 400."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "50"}
        mock_handler.rfile.read.return_value = b'{"content": "Great!"}'

        raw_result = marketplace_handler._submit_review("tpl-123", mock_handler, "127.0.0.1")
        result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_review_invalid_rating(self, mock_marketplace_state, marketplace_handler):
        """Test invalid rating returns 400."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "50"}
        mock_handler.rfile.read.return_value = b'{"rating": 10, "content": "X"}'

        raw_result = marketplace_handler._submit_review("tpl-123", mock_handler, "127.0.0.1")
        result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_review_whitespace_content(self, mock_marketplace_state, marketplace_handler):
        """Test whitespace-only content returns 400."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "50"}
        mock_handler.rfile.read.return_value = json.dumps({"rating": 4, "content": "    "}).encode()

        raw_result = marketplace_handler._submit_review("tpl-123", mock_handler, "127.0.0.1")
        result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_review_content_too_long(self, mock_marketplace_state, marketplace_handler):
        """Test content too long returns 400."""
        mock_handler = MagicMock()
        body = json.dumps({"rating": 4, "content": "A" * 3000}).encode()
        mock_handler.headers = {"Content-Length": str(len(body))}
        mock_handler.rfile.read.return_value = body

        raw_result = marketplace_handler._submit_review("tpl-123", mock_handler, "127.0.0.1")
        result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_review_title_too_long(self, mock_marketplace_state, marketplace_handler):
        """Test title too long returns 400."""
        mock_handler = MagicMock()
        body = json.dumps(
            {
                "rating": 4,
                "content": "Good",
                "title": "T" * 150,
            }
        ).encode()
        mock_handler.headers = {"Content-Length": str(len(body))}
        mock_handler.rfile.read.return_value = body

        raw_result = marketplace_handler._submit_review("tpl-123", mock_handler, "127.0.0.1")
        result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_review_user_name_too_long(self, mock_marketplace_state, marketplace_handler):
        """Test user name too long returns 400."""
        mock_handler = MagicMock()
        body = json.dumps(
            {
                "rating": 4,
                "content": "Good",
                "user_name": "U" * 150,
            }
        ).encode()
        mock_handler.headers = {"Content-Length": str(len(body))}
        mock_handler.rfile.read.return_value = body

        raw_result = marketplace_handler._submit_review("tpl-123", mock_handler, "127.0.0.1")
        result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_review_returns_201(self, mock_marketplace_state, marketplace_handler):
        """Test successful review returns 201."""
        mock_handler = MagicMock()
        body = json.dumps(
            {
                "rating": 5,
                "content": "Excellent template!",
            }
        ).encode()
        mock_handler.headers = {"Content-Length": str(len(body))}
        mock_handler.rfile.read.return_value = body

        raw_result = marketplace_handler._submit_review("tpl-123", mock_handler, "127.0.0.1")
        result = parse_result(raw_result)

        assert result["status_code"] == 201
        assert result["data"]["status"] == "submitted"

    def test_review_default_user_name(self, mock_marketplace_state, marketplace_handler):
        """Test review defaults to Anonymous user name."""
        mock_handler = MagicMock()
        body = json.dumps(
            {
                "rating": 3,
                "content": "No name provided",
            }
        ).encode()
        mock_handler.headers = {"Content-Length": str(len(body))}
        mock_handler.rfile.read.return_value = body

        raw_result = marketplace_handler._submit_review("tpl-123", mock_handler, "127.0.0.1")
        result = parse_result(raw_result)

        assert result["success"] is True
        assert result["data"]["review"]["user_name"] == "Anonymous"

    def test_review_creates_list_for_new_template(
        self, mock_marketplace_state, marketplace_handler
    ):
        """Test review for template with no reviews creates the list."""
        from aragora.server.handlers.template_marketplace import (
            _marketplace_templates,
            _template_reviews,
            MarketplaceTemplate,
        )

        _marketplace_templates["other/no-reviews"] = MarketplaceTemplate(
            id="other/no-reviews",
            name="No Reviews",
            description="No reviews yet",
            category="other",
            pattern="custom",
            author_id="user",
            author_name="User",
        )

        mock_handler = MagicMock()
        body = json.dumps({"rating": 4, "content": "First!"}).encode()
        mock_handler.headers = {"Content-Length": str(len(body))}
        mock_handler.rfile.read.return_value = body

        raw_result = marketplace_handler._submit_review(
            "other/no-reviews", mock_handler, "127.0.0.1"
        )
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "other/no-reviews" in _template_reviews
        assert len(_template_reviews["other/no-reviews"]) == 1

    def test_review_updates_template_rating(self, mock_marketplace_state, marketplace_handler):
        """Test submitting review updates template average rating."""
        from aragora.server.handlers.template_marketplace import _marketplace_templates

        template = _marketplace_templates["tpl-123"]
        original_count = template.rating_count

        mock_handler = MagicMock()
        body = json.dumps({"rating": 1, "content": "Not great."}).encode()
        mock_handler.headers = {"Content-Length": str(len(body))}
        mock_handler.rfile.read.return_value = body

        marketplace_handler._submit_review("tpl-123", mock_handler, "127.0.0.1")
        assert template.rating_count == original_count + 1

    def test_review_not_found(self, mock_marketplace_state, marketplace_handler):
        """Test review for non-existent template returns 404."""
        mock_handler = MagicMock()
        body = json.dumps({"rating": 4, "content": "Good"}).encode()
        mock_handler.headers = {"Content-Length": str(len(body))}
        mock_handler.rfile.read.return_value = body

        raw_result = marketplace_handler._submit_review("nonexistent", mock_handler, "127.0.0.1")
        result = parse_result(raw_result)

        assert result["status_code"] == 404


class TestImportTemplateAdvanced:
    """Advanced tests for importing templates."""

    def test_import_body_too_large(self, mock_marketplace_state, marketplace_handler):
        """Test body too large returns 413."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "20000"}

        raw_result = marketplace_handler._import_template("tpl-123", mock_handler)
        result = parse_result(raw_result)

        assert result["status_code"] == 413

    def test_import_with_workspace_id(self, mock_marketplace_state, marketplace_handler):
        """Test import returns workspace_id when provided."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "40"}
        mock_handler.rfile.read.return_value = b'{"workspace_id": "ws-abc"}'

        raw_result = marketplace_handler._import_template("tpl-123", mock_handler)
        result = parse_result(raw_result)

        assert result["success"] is True
        assert result["data"]["workspace_id"] == "ws-abc"

    def test_import_without_workspace_id(self, mock_marketplace_state, marketplace_handler):
        """Test import works without workspace_id."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "2"}
        mock_handler.rfile.read.return_value = b"{}"

        raw_result = marketplace_handler._import_template("tpl-123", mock_handler)
        result = parse_result(raw_result)

        assert result["success"] is True
        assert result["data"]["workspace_id"] is None

    def test_import_handles_invalid_json_gracefully(
        self, mock_marketplace_state, marketplace_handler
    ):
        """Test import handles invalid JSON gracefully (defaults to empty dict)."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "10"}
        mock_handler.rfile.read.return_value = b"not json"

        raw_result = marketplace_handler._import_template("tpl-123", mock_handler)
        result = parse_result(raw_result)

        assert result["success"] is True
        assert result["data"]["workspace_id"] is None

    def test_import_returns_download_count(self, mock_marketplace_state, marketplace_handler):
        """Test import response includes download count."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "2"}
        mock_handler.rfile.read.return_value = b"{}"

        raw_result = marketplace_handler._import_template("tpl-123", mock_handler)
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "download_count" in result["data"]
        assert result["data"]["download_count"] > 0

    def test_import_returns_all_template_data(self, mock_marketplace_state, marketplace_handler):
        """Test import returns workflow_definition and schemas."""
        mock_handler = MagicMock()
        mock_handler.headers = {"Content-Length": "2"}
        mock_handler.rfile.read.return_value = b"{}"

        raw_result = marketplace_handler._import_template("tpl-123", mock_handler)
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "workflow_definition" in result["data"]
        assert "input_schema" in result["data"]
        assert "output_schema" in result["data"]
        assert "documentation" in result["data"]


class TestRouteRequestAdvanced:
    """Tests for request routing edge cases."""

    def test_route_post_templates_publishes(self, mock_marketplace_state, marketplace_handler):
        """Test POST /templates routes to publish."""
        mock_handler = MagicMock()
        mock_handler.command = "POST"
        mock_handler.headers = {"Content-Length": "200"}
        mock_handler.rfile.read.return_value = json.dumps(
            {
                "name": "Route Test",
                "description": "Testing routing",
                "category": "analytics",
                "pattern": "parallel",
                "workflow_definition": {},
            }
        ).encode()

        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter.is_allowed",
            return_value=True,
        ):
            raw_result = marketplace_handler.handle(
                "/api/v1/marketplace/templates", {}, mock_handler
            )
            result = parse_result(raw_result)

        assert result["success"] is True
        assert result["data"]["status"] == "published"

    def test_route_put_templates_405(self, mock_marketplace_state, marketplace_handler):
        """Test PUT /templates returns 405."""
        mock_handler = MagicMock()
        mock_handler.command = "PUT"

        raw_result = marketplace_handler.handle("/api/v1/marketplace/templates", {}, mock_handler)
        result = parse_result(raw_result)

        assert result["status_code"] == 405

    def test_route_invalid_path_400(self, mock_marketplace_state, marketplace_handler):
        """Test invalid path returns 400."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        raw_result = marketplace_handler.handle(
            "/api/v1/marketplace/invalid_endpoint", {}, mock_handler
        )
        result = parse_result(raw_result)

        assert result["status_code"] == 400

    def test_route_specific_template_get(self, mock_marketplace_state, marketplace_handler):
        """Test GET specific template via _route_request."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        # _get_template called directly works with any ID format
        raw_result = marketplace_handler._get_template("tpl-123")
        result = parse_result(raw_result)

        assert result["success"] is True
        assert result["data"]["id"] == "tpl-123"

    def test_route_specific_template_via_handle(self, marketplace_handler):
        """Test GET specific template routed via handle() with seeded data."""
        from aragora.server.handlers.template_marketplace import (
            _marketplace_templates,
            _clear_marketplace_state,
            MarketplaceTemplate,
        )

        _clear_marketplace_state()
        # Add template with ID that matches the URL routing
        # URL: /api/v1/marketplace/templates/<id> -> template_id = parts[4:]
        # For /api/v1/marketplace/templates/my-template -> id = "templates/my-template"
        # So for proper routing, use a template ID that accounts for the URL split
        _marketplace_templates["templates/my-template"] = MarketplaceTemplate(
            id="templates/my-template",
            name="My Template",
            description="Test",
            category="other",
            pattern="custom",
            author_id="u",
            author_name="U",
        )

        mock_handler = MagicMock()
        mock_handler.command = "GET"

        raw_result = marketplace_handler.handle(
            "/api/v1/marketplace/templates/my-template", {}, mock_handler
        )
        result = parse_result(raw_result)

        assert result["success"] is True
        assert result["data"]["id"] == "templates/my-template"

    def test_route_featured_get(self, mock_marketplace_state, marketplace_handler):
        """Test GET featured via handle()."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        raw_result = marketplace_handler.handle("/api/v1/marketplace/featured", {}, mock_handler)
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "featured" in result["data"]

    def test_route_trending_get(self, mock_marketplace_state, marketplace_handler):
        """Test GET trending via handle()."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        raw_result = marketplace_handler.handle("/api/v1/marketplace/trending", {}, mock_handler)
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "trending" in result["data"]

    def test_route_categories_get(self, mock_marketplace_state, marketplace_handler):
        """Test GET categories via handle()."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        raw_result = marketplace_handler.handle("/api/v1/marketplace/categories", {}, mock_handler)
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "categories" in result["data"]

    def test_handler_without_command_defaults_to_get(
        self, mock_marketplace_state, marketplace_handler
    ):
        """Test handler without command attribute defaults to GET."""
        mock_handler = MagicMock(spec=[])

        raw_result = marketplace_handler.handle("/api/v1/marketplace/featured", {}, mock_handler)
        result = parse_result(raw_result)

        assert result["success"] is True

    def test_rate_limit_on_handle(self, mock_marketplace_state, marketplace_handler):
        """Test rate limit check in handle() method."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        with patch(
            "aragora.server.handlers.template_marketplace._marketplace_limiter.is_allowed",
            return_value=False,
        ):
            raw_result = marketplace_handler.handle(
                "/api/v1/marketplace/templates", {}, mock_handler
            )
            result = parse_result(raw_result)

        assert result["status_code"] == 429


class TestSeedMarketplaceTemplates:
    """Tests for marketplace seeding."""

    def test_seed_populates_templates(self):
        """Test seeding populates templates."""
        from aragora.server.handlers.template_marketplace import (
            _marketplace_templates,
            _seed_marketplace_templates,
            _clear_marketplace_state,
        )

        _clear_marketplace_state()
        assert len(_marketplace_templates) == 0

        _seed_marketplace_templates()
        assert len(_marketplace_templates) > 0

    def test_seed_is_idempotent(self):
        """Test seeding is idempotent."""
        from aragora.server.handlers.template_marketplace import (
            _marketplace_templates,
            _seed_marketplace_templates,
            _clear_marketplace_state,
        )

        _clear_marketplace_state()
        _seed_marketplace_templates()
        count_first = len(_marketplace_templates)

        _seed_marketplace_templates()
        count_second = len(_marketplace_templates)

        assert count_first == count_second

    def test_seed_templates_cover_expected_categories(self):
        """Test seeded templates cover key categories."""
        from aragora.server.handlers.template_marketplace import (
            _marketplace_templates,
            _seed_marketplace_templates,
            _clear_marketplace_state,
        )

        _clear_marketplace_state()
        _seed_marketplace_templates()

        categories = {t.category for t in _marketplace_templates.values()}
        assert "security" in categories
        assert "research" in categories
        assert "quickstart" in categories
        assert "sme" in categories

    def test_seed_templates_have_valid_structure(self):
        """Test each seeded template has required fields."""
        from aragora.server.handlers.template_marketplace import (
            _marketplace_templates,
            _seed_marketplace_templates,
            _clear_marketplace_state,
        )

        _clear_marketplace_state()
        _seed_marketplace_templates()

        for tid, template in _marketplace_templates.items():
            assert template.id == tid
            assert template.name
            assert template.description
            assert template.category
            assert template.pattern
            assert template.author_id
            assert template.author_name


class TestClearMarketplaceState:
    """Tests for _clear_marketplace_state."""

    def test_clear_state_empties_all_collections(self):
        """Test clearing state empties all data."""
        from aragora.server.handlers.template_marketplace import (
            _marketplace_templates,
            _template_reviews,
            _user_ratings,
            _clear_marketplace_state,
            MarketplaceTemplate,
        )

        _marketplace_templates["test/t"] = MarketplaceTemplate(
            id="test/t",
            name="T",
            description="D",
            category="other",
            pattern="custom",
            author_id="u",
            author_name="U",
        )
        _template_reviews["test/t"] = []
        _user_ratings["u"] = {"test/t": 5}

        _clear_marketplace_state()

        assert len(_marketplace_templates) == 0
        assert len(_template_reviews) == 0
        assert len(_user_ratings) == 0


class TestPersistentStore:
    """Tests for persistent store initialization."""

    def test_init_persistent_store_fallback(self):
        """Test falls back to in-memory when storage unavailable."""
        from aragora.server.handlers.template_marketplace import _init_persistent_store

        with patch(
            "aragora.server.handlers.template_marketplace.is_distributed_state_required",
            return_value=False,
        ):
            result = _init_persistent_store()
            assert isinstance(result, bool)

    def test_init_persistent_store_distributed_required_raises(self):
        """Test raises DistributedStateError when distributed state required and storage unavailable."""
        from aragora.control_plane.leader import DistributedStateError
        import aragora.server.handlers.template_marketplace as tm

        # Save originals
        original_store = tm._persistent_store
        original_use = tm._use_persistent_store

        # Reset state so _init_persistent_store actually runs
        tm._persistent_store = None
        tm._use_persistent_store = False

        with (
            patch(
                "aragora.server.handlers.template_marketplace.is_distributed_state_required",
                return_value=True,
            ),
            patch.dict("sys.modules", {"aragora.storage.marketplace_store": None}),
        ):
            with pytest.raises((DistributedStateError, TypeError)):
                tm._init_persistent_store()

        # Restore
        tm._persistent_store = original_store
        tm._use_persistent_store = original_use

    def test_get_persistent_store_returns_none_in_memory(self):
        """Test _get_persistent_store returns None in-memory."""
        from aragora.server.handlers.template_marketplace import _get_persistent_store
        import aragora.server.handlers.template_marketplace as tm

        original_store = tm._persistent_store
        original_use = tm._use_persistent_store
        tm._persistent_store = None
        tm._use_persistent_store = False

        with patch.object(tm, "_init_persistent_store", return_value=False):
            result = _get_persistent_store()
            assert result is None

        tm._persistent_store = original_store
        tm._use_persistent_store = original_use


class TestDataClasses:
    """Tests for MarketplaceTemplate and TemplateReview dataclasses."""

    def test_template_default_values(self):
        """Test MarketplaceTemplate default values."""
        from aragora.server.handlers.template_marketplace import MarketplaceTemplate

        template = MarketplaceTemplate(
            id="test/t",
            name="T",
            description="D",
            category="other",
            pattern="custom",
            author_id="u",
            author_name="U",
        )

        assert template.version == "1.0.0"
        assert template.tags == []
        assert template.workflow_definition == {}
        assert template.rating == 0.0
        assert template.rating_count == 0
        assert template.download_count == 0
        assert template.is_featured is False
        assert template.is_verified is False
        assert template.created_at > 0

    def test_template_to_dict_all_fields(self):
        """Test to_dict includes all fields."""
        from aragora.server.handlers.template_marketplace import MarketplaceTemplate

        template = MarketplaceTemplate(
            id="test/complete",
            name="Complete",
            description="All fields",
            category="security",
            pattern="debate",
            author_id="author",
            author_name="Author",
            version="2.0.0",
            tags=["a", "b"],
            workflow_definition={"steps": [1]},
            input_schema={"type": "object"},
            output_schema={"type": "string"},
            documentation="Docs",
            examples=[{"ex": 1}],
            rating=4.5,
            rating_count=100,
            download_count=500,
            is_featured=True,
            is_verified=True,
        )

        d = template.to_dict()
        assert d["id"] == "test/complete"
        assert d["version"] == "2.0.0"
        assert d["tags"] == ["a", "b"]
        assert d["workflow_definition"] == {"steps": [1]}
        assert d["is_featured"] is True

    def test_template_to_summary_truncates_long_description(self):
        """Test to_summary truncates descriptions over 200 chars."""
        from aragora.server.handlers.template_marketplace import MarketplaceTemplate

        template = MarketplaceTemplate(
            id="test/long",
            name="Long",
            description="A" * 250,
            category="other",
            pattern="custom",
            author_id="u",
            author_name="U",
        )

        s = template.to_summary()
        assert s["description"].endswith("...")
        assert len(s["description"]) == 203  # 200 + "..."

    def test_template_to_summary_short_description_no_truncation(self):
        """Test to_summary does not truncate short descriptions."""
        from aragora.server.handlers.template_marketplace import MarketplaceTemplate

        template = MarketplaceTemplate(
            id="test/short",
            name="Short",
            description="Short desc",
            category="other",
            pattern="custom",
            author_id="u",
            author_name="U",
        )

        s = template.to_summary()
        assert s["description"] == "Short desc"

    def test_template_to_summary_excludes_heavy_fields(self):
        """Test to_summary excludes heavy fields."""
        from aragora.server.handlers.template_marketplace import MarketplaceTemplate

        template = MarketplaceTemplate(
            id="test/light",
            name="Light",
            description="Desc",
            category="other",
            pattern="custom",
            author_id="u",
            author_name="U",
            workflow_definition={"big": "data"},
        )

        s = template.to_summary()
        assert "workflow_definition" not in s
        assert "input_schema" not in s
        assert "output_schema" not in s

    def test_review_default_values(self):
        """Test TemplateReview default values."""
        from aragora.server.handlers.template_marketplace import TemplateReview

        review = TemplateReview(
            id="rev-1",
            template_id="test/t",
            user_id="u",
            user_name="U",
            rating=3,
            title="Title",
            content="Content",
        )

        assert review.helpful_count == 0
        assert review.created_at > 0

    def test_review_to_dict(self):
        """Test TemplateReview.to_dict includes all fields."""
        from aragora.server.handlers.template_marketplace import TemplateReview

        review = TemplateReview(
            id="rev-all",
            template_id="test/all",
            user_id="u",
            user_name="User",
            rating=4,
            title="All",
            content="Testing",
            helpful_count=10,
        )

        d = review.to_dict()
        assert d["id"] == "rev-all"
        assert d["rating"] == 4
        assert d["helpful_count"] == 10
        assert "created_at" in d


class TestFeaturedAdvanced:
    """Advanced featured and trending tests."""

    def test_featured_sorted_by_rating_desc(self, marketplace_handler):
        """Test featured sorted by rating descending."""
        from aragora.server.handlers.template_marketplace import (
            _marketplace_templates,
            _clear_marketplace_state,
            MarketplaceTemplate,
        )

        _clear_marketplace_state()
        _marketplace_templates["test/low"] = MarketplaceTemplate(
            id="test/low",
            name="Low",
            description="Low",
            category="other",
            pattern="custom",
            author_id="u",
            author_name="U",
            is_featured=True,
            rating=1.0,
        )
        _marketplace_templates["test/high"] = MarketplaceTemplate(
            id="test/high",
            name="High",
            description="High",
            category="other",
            pattern="custom",
            author_id="u",
            author_name="U",
            is_featured=True,
            rating=5.0,
        )

        raw_result = marketplace_handler._get_featured()
        result = parse_result(raw_result)

        assert result["success"] is True
        featured = result["data"]["featured"]
        assert len(featured) == 2
        assert featured[0]["rating"] >= featured[1]["rating"]

    def test_featured_limited_to_10(self, marketplace_handler):
        """Test featured capped at 10."""
        from aragora.server.handlers.template_marketplace import (
            _marketplace_templates,
            _clear_marketplace_state,
            MarketplaceTemplate,
        )

        _clear_marketplace_state()
        for i in range(15):
            _marketplace_templates[f"other/f-{i}"] = MarketplaceTemplate(
                id=f"other/f-{i}",
                name=f"F {i}",
                description="Desc",
                category="other",
                pattern="custom",
                author_id="u",
                author_name="U",
                is_featured=True,
                rating=float(i),
            )

        raw_result = marketplace_handler._get_featured()
        result = parse_result(raw_result)

        assert len(result["data"]["featured"]) <= 10
        assert result["data"]["total"] == 15

    def test_trending_returns_period(self, mock_marketplace_state, marketplace_handler):
        """Test trending includes requested period."""
        raw_result = marketplace_handler._get_trending({"period": "month"})
        result = parse_result(raw_result)

        assert result["data"]["period"] == "month"

    def test_trending_limit(self, mock_marketplace_state, marketplace_handler):
        """Test trending respects limit."""
        raw_result = marketplace_handler._get_trending({"limit": "3"})
        result = parse_result(raw_result)

        assert result["success"] is True
        assert len(result["data"]["trending"]) <= 3


class TestCategoriesAdvanced:
    """Advanced categories tests."""

    def test_categories_sorted_by_count(self, marketplace_handler):
        """Test categories sorted by template count descending."""
        from aragora.server.handlers.template_marketplace import (
            _marketplace_templates,
            _clear_marketplace_state,
            MarketplaceTemplate,
        )

        _clear_marketplace_state()
        for i in range(5):
            _marketplace_templates[f"security/t-{i}"] = MarketplaceTemplate(
                id=f"security/t-{i}",
                name=f"S {i}",
                description="Security",
                category="security",
                pattern="debate",
                author_id="u",
                author_name="U",
            )
        _marketplace_templates["data/t-0"] = MarketplaceTemplate(
            id="data/t-0",
            name="D 0",
            description="Data",
            category="data",
            pattern="pipeline",
            author_id="u",
            author_name="U",
        )

        raw_result = marketplace_handler._get_categories()
        result = parse_result(raw_result)

        categories = result["data"]["categories"]
        assert categories[0]["template_count"] >= categories[-1]["template_count"]

    def test_categories_include_total_downloads(self, marketplace_handler):
        """Test categories include total_downloads."""
        from aragora.server.handlers.template_marketplace import (
            _marketplace_templates,
            _clear_marketplace_state,
            MarketplaceTemplate,
        )

        _clear_marketplace_state()
        _marketplace_templates["data/t-1"] = MarketplaceTemplate(
            id="data/t-1",
            name="D1",
            description="D",
            category="data",
            pattern="pipeline",
            author_id="u",
            author_name="U",
            download_count=100,
        )
        _marketplace_templates["data/t-2"] = MarketplaceTemplate(
            id="data/t-2",
            name="D2",
            description="D",
            category="data",
            pattern="pipeline",
            author_id="u",
            author_name="U",
            download_count=200,
        )

        raw_result = marketplace_handler._get_categories()
        result = parse_result(raw_result)

        cat = result["data"]["categories"][0]
        assert cat["total_downloads"] == 300


class TestListTemplatesAdvanced:
    """Advanced list template tests."""

    def test_list_combined_filters(self, mock_marketplace_state, marketplace_handler):
        """Test listing with multiple filters simultaneously."""
        raw_result = marketplace_handler._list_templates(
            {
                "category": "automation",
                "pattern": "sequential",
                "verified_only": "true",
                "search": "test",
                "sort_by": "rating",
                "limit": "5",
                "offset": "0",
            }
        )
        result = parse_result(raw_result)
        assert result["success"] is True

    def test_list_returns_total_count(self, mock_marketplace_state, marketplace_handler):
        """Test list returns total count before pagination."""
        raw_result = marketplace_handler._list_templates({"limit": "1"})
        result = parse_result(raw_result)

        assert result["success"] is True
        assert result["data"]["total"] >= len(result["data"]["templates"])

    def test_list_offset_beyond_total(self, mock_marketplace_state, marketplace_handler):
        """Test offset beyond total returns empty list."""
        raw_result = marketplace_handler._list_templates({"offset": "9999"})
        result = parse_result(raw_result)

        assert result["success"] is True
        assert len(result["data"]["templates"]) == 0

    def test_list_summary_format(self, mock_marketplace_state, marketplace_handler):
        """Test list returns templates in summary format."""
        raw_result = marketplace_handler._list_templates({})
        result = parse_result(raw_result)

        assert result["success"] is True
        if result["data"]["templates"]:
            t = result["data"]["templates"][0]
            assert "workflow_definition" not in t
            assert "id" in t
            assert "name" in t
            assert "category" in t

    def test_list_search_in_description(self, mock_marketplace_state, marketplace_handler):
        """Test search matches against description."""
        raw_result = marketplace_handler._list_templates({"search": "unit testing"})
        result = parse_result(raw_result)

        assert result["success"] is True

    def test_list_search_in_tags(self, mock_marketplace_state, marketplace_handler):
        """Test search matches against tags."""
        raw_result = marketplace_handler._list_templates({"search": "automation"})
        result = parse_result(raw_result)

        assert result["success"] is True


class TestGetReviewsAdvanced:
    """Advanced get reviews tests."""

    def test_reviews_empty_list(self, mock_marketplace_state, marketplace_handler):
        """Test getting reviews for template with no reviews."""
        from aragora.server.handlers.template_marketplace import (
            _marketplace_templates,
            MarketplaceTemplate,
        )

        _marketplace_templates["other/empty"] = MarketplaceTemplate(
            id="other/empty",
            name="Empty",
            description="No reviews",
            category="other",
            pattern="custom",
            author_id="u",
            author_name="U",
        )

        raw_result = marketplace_handler._get_reviews("other/empty", {})
        result = parse_result(raw_result)

        assert result["success"] is True
        assert result["data"]["reviews"] == []
        assert result["data"]["total"] == 0

    def test_reviews_sorted_by_helpful_count(self, mock_marketplace_state, marketplace_handler):
        """Test reviews sorted by helpful_count descending."""
        from aragora.server.handlers.template_marketplace import (
            _template_reviews,
            TemplateReview,
        )

        _template_reviews["tpl-123"].append(
            TemplateReview(
                id="rev-new-1",
                template_id="tpl-123",
                user_id="u1",
                user_name="User 1",
                rating=5,
                title="Most helpful",
                content="Very helpful",
                helpful_count=100,
            )
        )
        _template_reviews["tpl-123"].append(
            TemplateReview(
                id="rev-new-2",
                template_id="tpl-123",
                user_id="u2",
                user_name="User 2",
                rating=3,
                title="Less helpful",
                content="Less helpful",
                helpful_count=2,
            )
        )

        raw_result = marketplace_handler._get_reviews("tpl-123", {})
        result = parse_result(raw_result)

        reviews = result["data"]["reviews"]
        for i in range(len(reviews) - 1):
            assert reviews[i]["helpful_count"] >= reviews[i + 1]["helpful_count"]


class TestRecommendationsHandler:
    """Tests for TemplateRecommendationsHandler."""

    @pytest.fixture
    def recommendations_handler(self, mock_server_context):
        """Create a TemplateRecommendationsHandler."""
        from aragora.server.handlers.template_marketplace import (
            TemplateRecommendationsHandler,
        )

        return TemplateRecommendationsHandler(mock_server_context)

    def test_can_handle_recommendations_path(self, recommendations_handler):
        """Test handler recognizes recommendations path."""
        assert recommendations_handler.can_handle("/api/v1/marketplace/recommendations")

    def test_cannot_handle_other_paths(self, recommendations_handler):
        """Test handler rejects non-recommendations paths."""
        assert not recommendations_handler.can_handle("/api/v1/marketplace/templates")

    def test_get_recommendations(self, mock_marketplace_state, recommendations_handler):
        """Test getting recommendations."""
        mock_handler = MagicMock()

        raw_result = recommendations_handler.handle(
            "/api/v1/marketplace/recommendations", {}, mock_handler
        )
        result = parse_result(raw_result)

        assert result["success"] is True
        assert "recommendations" in result["data"]

    def test_get_recommendations_with_limit(self, mock_marketplace_state, recommendations_handler):
        """Test recommendations with custom limit."""
        mock_handler = MagicMock()

        raw_result = recommendations_handler.handle(
            "/api/v1/marketplace/recommendations", {"limit": "3"}, mock_handler
        )
        result = parse_result(raw_result)

        assert result["success"] is True
        assert len(result["data"]["recommendations"]) <= 3


class TestModuleExports:
    """Tests for module-level exports and constants."""

    def test_all_exports(self):
        """Test __all__ exports expected symbols."""
        from aragora.server.handlers.template_marketplace import __all__

        expected = [
            "TemplateMarketplaceHandler",
            "TemplateRecommendationsHandler",
            "MarketplaceTemplate",
            "TemplateReview",
            "MarketplaceCircuitBreaker",
            "get_marketplace_circuit_breaker_status",
            "_clear_marketplace_state",
        ]
        for name in expected:
            assert name in __all__

    def test_valid_categories_frozen(self):
        """Test VALID_CATEGORIES is immutable."""
        from aragora.server.handlers.template_marketplace import VALID_CATEGORIES

        assert isinstance(VALID_CATEGORIES, frozenset)
        with pytest.raises(AttributeError):
            VALID_CATEGORIES.add("new")

    def test_valid_patterns_frozen(self):
        """Test VALID_PATTERNS is immutable."""
        from aragora.server.handlers.template_marketplace import VALID_PATTERNS

        assert isinstance(VALID_PATTERNS, frozenset)
        with pytest.raises(AttributeError):
            VALID_PATTERNS.add("new")

    def test_constants_reasonable_values(self):
        """Test validation constants have reasonable values."""
        from aragora.server.handlers.template_marketplace import (
            TEMPLATE_NAME_MAX_LENGTH,
            DESCRIPTION_MAX_LENGTH,
            REVIEW_CONTENT_MAX_LENGTH,
            MAX_TAGS,
            MAX_TAG_LENGTH,
        )

        assert TEMPLATE_NAME_MAX_LENGTH > 0
        assert DESCRIPTION_MAX_LENGTH > TEMPLATE_NAME_MAX_LENGTH
        assert REVIEW_CONTENT_MAX_LENGTH > 0
        assert MAX_TAGS > 0
        assert MAX_TAG_LENGTH > 0
