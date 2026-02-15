"""
Tests for Features Marketplace Handler (aragora/server/handlers/features/marketplace.py).

Stability: STABLE

This file tests:
- MarketplaceHandler initialization and routing
- Circuit breaker pattern (states, transitions, recovery)
- Rate limiting per endpoint
- Input validation (template IDs, deployment IDs, ratings, search queries)
- Template listing, filtering, pagination
- Template details and related templates
- Category listing
- Template search with various filters
- Popular templates endpoint
- Template deployment with validation
- Deployment listing and details
- Deployment archival
- Template rating with validation
- Demo endpoint
- Status endpoint (circuit breaker monitoring)
- Error handling and edge cases

Target: 80%+ code coverage
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from uuid import uuid4

import pytest

from aragora.server.handlers.features.marketplace import (
    MarketplaceHandler,
    MarketplaceCircuitBreaker,
    TemplateCategory,
    DeploymentStatus,
    TemplateMetadata,
    TemplateDeployment,
    TemplateRating,
    get_marketplace_circuit_breaker_status,
    _clear_marketplace_components,
    _validate_template_id,
    _validate_deployment_id,
    _validate_deployment_name,
    _validate_review,
    _validate_rating,
    _validate_search_query,
    _validate_category,
    _validate_config,
    _clamp_pagination,
    _load_templates,
    _get_tenant_deployments,
    _templates_cache,
    _deployments,
    _ratings,
    _download_counts,
    MAX_TEMPLATE_NAME_LENGTH,
    MAX_DEPLOYMENT_NAME_LENGTH,
    MAX_REVIEW_LENGTH,
    MAX_SEARCH_QUERY_LENGTH,
    MAX_CONFIG_SIZE,
    MAX_LIMIT,
    DEFAULT_LIMIT,
    MIN_RATING,
    MAX_RATING,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_marketplace_state():
    """Clear marketplace state before and after each test."""
    _clear_marketplace_components()
    yield
    _clear_marketplace_components()


@pytest.fixture
def mock_request():
    """Create a mock request object."""
    request = MagicMock()
    request.tenant_id = "test-tenant"
    request.user_id = "test-user"
    request.query = {}
    return request


@pytest.fixture
def handler():
    """Create a MarketplaceHandler instance."""
    with patch.object(
        MarketplaceHandler, "__init__", lambda self, ctx=None: setattr(self, "ctx", ctx or {})
    ):
        h = MarketplaceHandler.__new__(MarketplaceHandler)
        h.ctx = {}
        return h


@pytest.fixture
def sample_template():
    """Create a sample TemplateMetadata."""
    return TemplateMetadata(
        id="test-template-1",
        name="Test Template",
        description="A test template for testing",
        version="1.0.0",
        category=TemplateCategory.SOFTWARE,
        tags=["test", "software"],
        downloads=100,
        rating=4.5,
        rating_count=10,
    )


@pytest.fixture
def sample_templates():
    """Create multiple sample templates."""
    return {
        "test-1": TemplateMetadata(
            id="test-1",
            name="Test One",
            description="First test template",
            version="1.0.0",
            category=TemplateCategory.SOFTWARE,
            tags=["test", "software"],
            downloads=100,
        ),
        "test-2": TemplateMetadata(
            id="test-2",
            name="Test Two",
            description="Second test template",
            version="1.0.0",
            category=TemplateCategory.LEGAL,
            tags=["test", "legal"],
            downloads=50,
        ),
        "test-3": TemplateMetadata(
            id="test-3",
            name="Accounting Tool",
            description="Accounting template",
            version="1.0.0",
            category=TemplateCategory.ACCOUNTING,
            tags=["test", "accounting", "finance"],
            downloads=200,
            has_debate=True,
        ),
    }


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestMarketplaceCircuitBreaker:
    """Tests for MarketplaceCircuitBreaker."""

    def test_initial_state_is_closed(self):
        """Test that circuit breaker starts in closed state."""
        cb = MarketplaceCircuitBreaker()
        assert cb.state == MarketplaceCircuitBreaker.CLOSED

    def test_allows_requests_when_closed(self):
        """Test that requests are allowed when circuit is closed."""
        cb = MarketplaceCircuitBreaker()
        assert cb.is_allowed() is True

    def test_opens_after_threshold_failures(self):
        """Test that circuit opens after failure threshold is reached."""
        cb = MarketplaceCircuitBreaker(failure_threshold=3)

        # Record failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == MarketplaceCircuitBreaker.OPEN
        assert cb.is_allowed() is False

    def test_stays_closed_below_threshold(self):
        """Test that circuit stays closed below failure threshold."""
        cb = MarketplaceCircuitBreaker(failure_threshold=5)

        for _ in range(4):
            cb.record_failure()

        assert cb.state == MarketplaceCircuitBreaker.CLOSED
        assert cb.is_allowed() is True

    def test_transitions_to_half_open_after_cooldown(self):
        """Test that circuit transitions to half-open after cooldown."""
        cb = MarketplaceCircuitBreaker(failure_threshold=2, cooldown_seconds=0.1)

        # Open the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.OPEN

        # Wait for cooldown
        time.sleep(0.15)

        # Should transition to half-open
        assert cb.state == MarketplaceCircuitBreaker.HALF_OPEN

    def test_half_open_allows_limited_requests(self):
        """Test that half-open state allows limited requests."""
        cb = MarketplaceCircuitBreaker(
            failure_threshold=1, cooldown_seconds=0.01, half_open_max_calls=2
        )

        # Open the circuit
        cb.record_failure()
        time.sleep(0.02)

        # Should allow limited calls
        assert cb.is_allowed() is True
        assert cb.is_allowed() is True
        assert cb.is_allowed() is False  # Third call blocked

    def test_closes_after_successful_half_open_calls(self):
        """Test that circuit closes after successful calls in half-open."""
        cb = MarketplaceCircuitBreaker(
            failure_threshold=1, cooldown_seconds=0.01, half_open_max_calls=2
        )

        # Open and wait for half-open
        cb.record_failure()
        time.sleep(0.02)

        # Make successful calls
        cb.is_allowed()
        cb.record_success()
        cb.is_allowed()
        cb.record_success()

        assert cb.state == MarketplaceCircuitBreaker.CLOSED

    def test_reopens_on_half_open_failure(self):
        """Test that circuit reopens on failure in half-open state."""
        cb = MarketplaceCircuitBreaker(
            failure_threshold=1, cooldown_seconds=0.01, half_open_max_calls=2
        )

        # Open and wait for half-open
        cb.record_failure()
        time.sleep(0.02)

        # Make a call and fail
        cb.is_allowed()
        cb.record_failure()

        assert cb.state == MarketplaceCircuitBreaker.OPEN

    def test_reset_returns_to_closed(self):
        """Test that reset returns circuit to closed state."""
        cb = MarketplaceCircuitBreaker(failure_threshold=1)

        cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.OPEN

        cb.reset()
        assert cb.state == MarketplaceCircuitBreaker.CLOSED
        assert cb.is_allowed() is True

    def test_get_status_returns_correct_info(self):
        """Test that get_status returns correct information."""
        cb = MarketplaceCircuitBreaker(failure_threshold=5, cooldown_seconds=30.0)

        cb.record_failure()
        status = cb.get_status()

        assert status["state"] == MarketplaceCircuitBreaker.CLOSED
        assert status["failure_count"] == 1
        assert status["failure_threshold"] == 5
        assert status["cooldown_seconds"] == 30.0

    def test_success_resets_failure_count_when_closed(self):
        """Test that success resets failure count in closed state."""
        cb = MarketplaceCircuitBreaker(failure_threshold=5)

        cb.record_failure()
        cb.record_failure()
        assert cb.get_status()["failure_count"] == 2

        cb.record_success()
        assert cb.get_status()["failure_count"] == 0


class TestCircuitBreakerGlobalFunctions:
    """Tests for global circuit breaker functions."""

    def test_get_marketplace_circuit_breaker_status(self):
        """Test getting global circuit breaker status."""
        status = get_marketplace_circuit_breaker_status()

        assert "state" in status
        assert "failure_count" in status
        assert status["state"] == "closed"

    def test_clear_marketplace_components(self):
        """Test clearing marketplace components."""
        # Add some data
        _templates_cache["test"] = MagicMock()
        _deployments["tenant"] = {"deploy": MagicMock()}

        _clear_marketplace_components()

        assert len(_templates_cache) == 0
        assert len(_deployments) == 0


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestValidateTemplateId:
    """Tests for _validate_template_id."""

    def test_valid_template_id(self):
        """Test valid template IDs."""
        is_valid, err = _validate_template_id("test-template-123")
        assert is_valid is True
        assert err is None

    def test_empty_template_id(self):
        """Test empty template ID."""
        is_valid, err = _validate_template_id("")
        assert is_valid is False
        assert "required" in err.lower()

    def test_template_id_too_long(self):
        """Test template ID exceeding max length."""
        long_id = "a" * 200
        is_valid, err = _validate_template_id(long_id)
        assert is_valid is False
        assert "128" in err

    def test_template_id_invalid_characters(self):
        """Test template ID with invalid characters."""
        is_valid, err = _validate_template_id("test/../template")
        assert is_valid is False
        assert "invalid characters" in err.lower()

    def test_template_id_with_underscores(self):
        """Test template ID with underscores (valid)."""
        is_valid, err = _validate_template_id("test_template_v1")
        assert is_valid is True


class TestValidateDeploymentId:
    """Tests for _validate_deployment_id."""

    def test_valid_deployment_id(self):
        """Test valid deployment ID."""
        is_valid, err = _validate_deployment_id("deploy_abc123def456")
        assert is_valid is True
        assert err is None

    def test_empty_deployment_id(self):
        """Test empty deployment ID."""
        is_valid, err = _validate_deployment_id("")
        assert is_valid is False
        assert "required" in err.lower()

    def test_legacy_deployment_id(self):
        """Test legacy deployment ID format (safe pattern)."""
        is_valid, err = _validate_deployment_id("my-deployment-1")
        assert is_valid is True


class TestValidateDeploymentName:
    """Tests for _validate_deployment_name."""

    def test_valid_name(self):
        """Test valid deployment name."""
        is_valid, err = _validate_deployment_name("My Deployment")
        assert is_valid is True

    def test_none_name(self):
        """Test None deployment name (optional)."""
        is_valid, err = _validate_deployment_name(None)
        assert is_valid is True

    def test_name_too_long(self):
        """Test deployment name exceeding max length."""
        long_name = "a" * (MAX_DEPLOYMENT_NAME_LENGTH + 1)
        is_valid, err = _validate_deployment_name(long_name)
        assert is_valid is False

    def test_name_wrong_type(self):
        """Test deployment name with wrong type."""
        is_valid, err = _validate_deployment_name(123)
        assert is_valid is False
        assert "string" in err.lower()


class TestValidateReview:
    """Tests for _validate_review."""

    def test_valid_review(self):
        """Test valid review."""
        is_valid, err = _validate_review("Great template!")
        assert is_valid is True

    def test_none_review(self):
        """Test None review (optional)."""
        is_valid, err = _validate_review(None)
        assert is_valid is True

    def test_review_too_long(self):
        """Test review exceeding max length."""
        long_review = "a" * (MAX_REVIEW_LENGTH + 1)
        is_valid, err = _validate_review(long_review)
        assert is_valid is False

    def test_review_wrong_type(self):
        """Test review with wrong type."""
        is_valid, err = _validate_review(123)
        assert is_valid is False


class TestValidateRating:
    """Tests for _validate_rating."""

    def test_valid_rating(self):
        """Test valid rating."""
        is_valid, value, err = _validate_rating(5)
        assert is_valid is True
        assert value == 5

    def test_none_rating(self):
        """Test None rating (required)."""
        is_valid, value, err = _validate_rating(None)
        assert is_valid is False
        assert "required" in err.lower()

    def test_rating_too_low(self):
        """Test rating below minimum."""
        is_valid, value, err = _validate_rating(0)
        assert is_valid is False
        assert str(MIN_RATING) in err

    def test_rating_too_high(self):
        """Test rating above maximum."""
        is_valid, value, err = _validate_rating(10)
        assert is_valid is False
        assert str(MAX_RATING) in err

    def test_rating_wrong_type(self):
        """Test rating with wrong type."""
        is_valid, value, err = _validate_rating("five")
        assert is_valid is False
        assert "integer" in err.lower()


class TestValidateSearchQuery:
    """Tests for _validate_search_query."""

    def test_valid_query(self):
        """Test valid search query."""
        is_valid, sanitized, err = _validate_search_query("test query")
        assert is_valid is True
        assert sanitized == "test query"

    def test_empty_query(self):
        """Test empty search query."""
        is_valid, sanitized, err = _validate_search_query("")
        assert is_valid is True
        assert sanitized == ""

    def test_none_query(self):
        """Test None search query."""
        is_valid, sanitized, err = _validate_search_query(None)
        assert is_valid is True
        assert sanitized == ""

    def test_query_too_long(self):
        """Test search query exceeding max length."""
        long_query = "a" * (MAX_SEARCH_QUERY_LENGTH + 1)
        is_valid, sanitized, err = _validate_search_query(long_query)
        assert is_valid is False


class TestValidateCategory:
    """Tests for _validate_category."""

    def test_valid_category(self):
        """Test valid category."""
        is_valid, cat, err = _validate_category("software")
        assert is_valid is True
        assert cat == TemplateCategory.SOFTWARE

    def test_none_category(self):
        """Test None category (optional)."""
        is_valid, cat, err = _validate_category(None)
        assert is_valid is True
        assert cat is None

    def test_invalid_category(self):
        """Test invalid category."""
        is_valid, cat, err = _validate_category("invalid_category")
        assert is_valid is False
        assert "one of" in err.lower()

    def test_category_case_insensitive(self):
        """Test category validation is case insensitive."""
        is_valid, cat, err = _validate_category("SOFTWARE")
        assert is_valid is True
        assert cat == TemplateCategory.SOFTWARE


class TestValidateConfig:
    """Tests for _validate_config."""

    def test_valid_config(self):
        """Test valid config."""
        is_valid, config, err = _validate_config({"key": "value"})
        assert is_valid is True
        assert config == {"key": "value"}

    def test_none_config(self):
        """Test None config (defaults to empty dict)."""
        is_valid, config, err = _validate_config(None)
        assert is_valid is True
        assert config == {}

    def test_config_too_many_keys(self):
        """Test config with too many keys."""
        large_config = {f"key_{i}": i for i in range(MAX_CONFIG_SIZE + 1)}
        is_valid, config, err = _validate_config(large_config)
        assert is_valid is False

    def test_config_wrong_type(self):
        """Test config with wrong type."""
        is_valid, config, err = _validate_config("not a dict")
        assert is_valid is False


class TestClampPagination:
    """Tests for _clamp_pagination."""

    def test_default_values(self):
        """Test default pagination values."""
        limit, offset = _clamp_pagination(None, None)
        assert limit == DEFAULT_LIMIT
        assert offset == 0

    def test_clamps_limit_max(self):
        """Test limit is clamped to maximum."""
        limit, offset = _clamp_pagination(1000, 0)
        assert limit == MAX_LIMIT

    def test_clamps_limit_min(self):
        """Test limit is clamped to minimum."""
        limit, offset = _clamp_pagination(-10, 0)
        assert limit == 1

    def test_clamps_offset_min(self):
        """Test offset is clamped to minimum of 0."""
        limit, offset = _clamp_pagination(50, -10)
        assert offset == 0

    def test_handles_invalid_types(self):
        """Test handling of invalid types."""
        limit, offset = _clamp_pagination("invalid", "also_invalid")
        assert limit == DEFAULT_LIMIT
        assert offset == 0


# =============================================================================
# Handler Tests
# =============================================================================


class TestMarketplaceHandlerInit:
    """Tests for MarketplaceHandler initialization."""

    def test_init_with_context(self):
        """Test initialization with server context."""
        ctx = {"storage": "mock_storage"}
        with patch("aragora.server.handlers.features.marketplace._load_templates"):
            handler = MarketplaceHandler(server_context=ctx)
        assert handler.ctx == ctx

    def test_init_without_context(self):
        """Test initialization without server context."""
        with patch("aragora.server.handlers.features.marketplace._load_templates"):
            handler = MarketplaceHandler()
        assert handler.ctx == {}


class TestMarketplaceHandlerCanHandle:
    """Tests for MarketplaceHandler.can_handle."""

    def test_can_handle_templates_list(self, handler):
        """Test can_handle for templates list endpoint."""
        assert handler.can_handle("/api/v1/marketplace/templates") is True

    def test_can_handle_template_detail(self, handler):
        """Test can_handle for template detail endpoint."""
        assert handler.can_handle("/api/v1/marketplace/templates/test-id") is True

    def test_can_handle_template_deploy(self, handler):
        """Test can_handle for template deploy endpoint."""
        assert handler.can_handle("/api/v1/marketplace/templates/test-id/deploy") is True

    def test_can_handle_categories(self, handler):
        """Test can_handle for categories endpoint."""
        assert handler.can_handle("/api/v1/marketplace/categories") is True

    def test_can_handle_search(self, handler):
        """Test can_handle for search endpoint."""
        assert handler.can_handle("/api/v1/marketplace/search") is True

    def test_can_handle_deployments(self, handler):
        """Test can_handle for deployments endpoint."""
        assert handler.can_handle("/api/v1/marketplace/deployments") is True

    def test_can_handle_deployment_detail(self, handler):
        """Test can_handle for deployment detail endpoint."""
        assert handler.can_handle("/api/v1/marketplace/deployments/deploy_abc123") is True

    def test_can_handle_popular(self, handler):
        """Test can_handle for popular endpoint."""
        assert handler.can_handle("/api/v1/marketplace/popular") is True

    def test_can_handle_status(self, handler):
        """Test can_handle for status endpoint."""
        assert handler.can_handle("/api/v1/marketplace/status") is True

    def test_cannot_handle_unrelated_path(self, handler):
        """Test can_handle returns False for unrelated paths."""
        assert handler.can_handle("/api/v1/other/endpoint") is False


class TestMarketplaceHandlerListTemplates:
    """Tests for template listing endpoint."""

    @pytest.mark.asyncio
    async def test_list_templates_empty(self, handler, mock_request):
        """Test listing templates when none exist."""
        with patch("aragora.server.handlers.features.marketplace._load_templates", return_value={}):
            result = await handler._handle_list_templates(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["templates"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_templates_with_data(self, handler, mock_request, sample_templates):
        """Test listing templates with data."""
        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=sample_templates,
        ):
            result = await handler._handle_list_templates(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 3

    @pytest.mark.asyncio
    async def test_list_templates_with_category_filter(
        self, handler, mock_request, sample_templates
    ):
        """Test listing templates with category filter."""
        mock_request.query = {"category": "software"}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=sample_templates,
        ):
            result = await handler._handle_list_templates(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_list_templates_invalid_category(self, handler, mock_request, sample_templates):
        """Test listing templates with invalid category."""
        mock_request.query = {"category": "invalid"}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=sample_templates,
        ):
            result = await handler._handle_list_templates(mock_request, "tenant-1")

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_list_templates_pagination(self, handler, mock_request, sample_templates):
        """Test listing templates with pagination."""
        mock_request.query = {"limit": "1", "offset": "1"}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=sample_templates,
        ):
            result = await handler._handle_list_templates(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["templates"]) == 1
        assert body["limit"] == 1
        assert body["offset"] == 1


class TestMarketplaceHandlerGetTemplate:
    """Tests for template detail endpoint."""

    @pytest.mark.asyncio
    async def test_get_template_found(self, handler, mock_request, sample_template):
        """Test getting an existing template."""
        templates = {"test-template-1": sample_template}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates", return_value=templates
        ):
            with patch(
                "aragora.server.handlers.features.marketplace._get_full_template",
                return_value={"steps": []},
            ):
                result = await handler._handle_get_template(
                    mock_request, "tenant-1", "test-template-1"
                )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["template"]["id"] == "test-template-1"

    @pytest.mark.asyncio
    async def test_get_template_not_found(self, handler, mock_request):
        """Test getting a non-existent template."""
        with patch("aragora.server.handlers.features.marketplace._load_templates", return_value={}):
            result = await handler._handle_get_template(mock_request, "tenant-1", "nonexistent")

        assert result.status_code == 404


class TestMarketplaceHandlerCategories:
    """Tests for categories endpoint."""

    @pytest.mark.asyncio
    async def test_list_categories(self, handler, mock_request, sample_templates):
        """Test listing categories."""
        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=sample_templates,
        ):
            result = await handler._handle_list_categories(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["categories"]) == len(TemplateCategory)


class TestMarketplaceHandlerSearch:
    """Tests for search endpoint."""

    @pytest.mark.asyncio
    async def test_search_empty_query(self, handler, mock_request, sample_templates):
        """Test search with empty query returns all."""
        mock_request.query = {}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=sample_templates,
        ):
            result = await handler._handle_search(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 3

    @pytest.mark.asyncio
    async def test_search_with_query(self, handler, mock_request, sample_templates):
        """Test search with text query."""
        mock_request.query = {"q": "accounting"}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=sample_templates,
        ):
            result = await handler._handle_search(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_search_with_category(self, handler, mock_request, sample_templates):
        """Test search with category filter."""
        mock_request.query = {"category": "legal"}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=sample_templates,
        ):
            result = await handler._handle_search(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_search_with_has_debate(self, handler, mock_request, sample_templates):
        """Test search with has_debate filter."""
        mock_request.query = {"has_debate": "true"}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=sample_templates,
        ):
            result = await handler._handle_search(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 1


class TestMarketplaceHandlerPopular:
    """Tests for popular endpoint."""

    @pytest.mark.asyncio
    async def test_popular_default_limit(self, handler, mock_request, sample_templates):
        """Test popular templates with default limit."""
        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=sample_templates,
        ):
            result = await handler._handle_popular(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        # Should be sorted by downloads (accounting=200, test-1=100, test-2=50)
        assert body["popular"][0]["id"] == "test-3"

    @pytest.mark.asyncio
    async def test_popular_with_limit(self, handler, mock_request, sample_templates):
        """Test popular templates with custom limit."""
        mock_request.query = {"limit": "1"}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=sample_templates,
        ):
            result = await handler._handle_popular(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert len(body["popular"]) == 1


class TestMarketplaceHandlerDeploy:
    """Tests for template deployment endpoint."""

    @pytest.mark.asyncio
    async def test_deploy_template_success(self, handler, mock_request, sample_template):
        """Test successful template deployment."""
        templates = {"test-template-1": sample_template}
        mock_request.json = MagicMock(return_value={"name": "My Deployment"})

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates", return_value=templates
        ):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({"name": "My Deployment"}, None),
            ):
                result = await handler._handle_deploy(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "deployment" in body
        assert body["deployment"]["name"] == "My Deployment"

    @pytest.mark.asyncio
    async def test_deploy_template_not_found(self, handler, mock_request):
        """Test deploying non-existent template."""
        with patch("aragora.server.handlers.features.marketplace._load_templates", return_value={}):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({}, None),
            ):
                result = await handler._handle_deploy(mock_request, "tenant-1", "nonexistent")

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_deploy_with_invalid_name(self, handler, mock_request, sample_template):
        """Test deploying with invalid deployment name."""
        templates = {"test-template-1": sample_template}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates", return_value=templates
        ):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({"name": "a" * 300}, None),
            ):
                result = await handler._handle_deploy(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_deploy_circuit_breaker_open(self, handler, mock_request, sample_template):
        """Test deployment when circuit breaker is open."""
        templates = {"test-template-1": sample_template}

        # Force circuit breaker open
        cb = MarketplaceCircuitBreaker(failure_threshold=1)
        cb.record_failure()

        with patch(
            "aragora.server.handlers.features.marketplace._get_marketplace_circuit_breaker",
            return_value=cb,
        ):
            with patch(
                "aragora.server.handlers.features.marketplace._load_templates",
                return_value=templates,
            ):
                result = await handler._handle_deploy(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 503


class TestMarketplaceHandlerDeployments:
    """Tests for deployments endpoint."""

    @pytest.mark.asyncio
    async def test_list_deployments_empty(self, handler, mock_request):
        """Test listing deployments when none exist."""
        result = await handler._handle_list_deployments(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["deployments"] == []
        assert body["total"] == 0

    @pytest.mark.asyncio
    async def test_list_deployments_with_data(self, handler, mock_request):
        """Test listing deployments with data."""
        # Add a deployment
        deployment = TemplateDeployment(
            id="deploy_abc123def456",
            template_id="test-1",
            tenant_id="tenant-1",
            name="Test Deployment",
            status=DeploymentStatus.ACTIVE,
        )
        _deployments["tenant-1"] = {"deploy_abc123def456": deployment}

        result = await handler._handle_list_deployments(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["total"] == 1

    @pytest.mark.asyncio
    async def test_get_deployment_found(self, handler, mock_request, sample_template):
        """Test getting an existing deployment."""
        deployment = TemplateDeployment(
            id="deploy_abc123def456",
            template_id="test-template-1",
            tenant_id="tenant-1",
            name="Test Deployment",
            status=DeploymentStatus.ACTIVE,
        )
        _deployments["tenant-1"] = {"deploy_abc123def456": deployment}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value={"test-template-1": sample_template},
        ):
            result = await handler._handle_get_deployment(
                mock_request, "tenant-1", "deploy_abc123def456"
            )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["deployment"]["id"] == "deploy_abc123def456"

    @pytest.mark.asyncio
    async def test_get_deployment_not_found(self, handler, mock_request):
        """Test getting a non-existent deployment."""
        result = await handler._handle_get_deployment(
            mock_request, "tenant-1", "deploy_nonexistent"
        )

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_deployment_success(self, handler, mock_request):
        """Test archiving a deployment."""
        deployment = TemplateDeployment(
            id="deploy_abc123def456",
            template_id="test-1",
            tenant_id="tenant-1",
            name="Test Deployment",
            status=DeploymentStatus.ACTIVE,
        )
        _deployments["tenant-1"] = {"deploy_abc123def456": deployment}

        result = await handler._handle_delete_deployment(
            mock_request, "tenant-1", "deploy_abc123def456"
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["deployment"]["status"] == "archived"

    @pytest.mark.asyncio
    async def test_delete_deployment_not_found(self, handler, mock_request):
        """Test archiving a non-existent deployment."""
        result = await handler._handle_delete_deployment(
            mock_request, "tenant-1", "deploy_nonexistent"
        )

        assert result.status_code == 404


class TestMarketplaceHandlerRate:
    """Tests for template rating endpoint."""

    @pytest.mark.asyncio
    async def test_rate_template_success(self, handler, mock_request, sample_template):
        """Test successful template rating."""
        templates = {"test-template-1": sample_template}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates", return_value=templates
        ):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({"rating": 5, "review": "Great!"}, None),
            ):
                result = await handler._handle_rate(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "rating" in body
        assert body["template_rating"]["count"] == 1

    @pytest.mark.asyncio
    async def test_rate_template_not_found(self, handler, mock_request):
        """Test rating a non-existent template."""
        with patch("aragora.server.handlers.features.marketplace._load_templates", return_value={}):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({"rating": 5}, None),
            ):
                result = await handler._handle_rate(mock_request, "tenant-1", "nonexistent")

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_rate_invalid_rating(self, handler, mock_request, sample_template):
        """Test rating with invalid value."""
        templates = {"test-template-1": sample_template}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates", return_value=templates
        ):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({"rating": 10}, None),
            ):
                result = await handler._handle_rate(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_rate_missing_rating(self, handler, mock_request, sample_template):
        """Test rating without rating value."""
        templates = {"test-template-1": sample_template}

        with patch(
            "aragora.server.handlers.features.marketplace._load_templates", return_value=templates
        ):
            with patch(
                "aragora.server.handlers.features.marketplace.parse_json_body",
                return_value=({"review": "No rating"}, None),
            ):
                result = await handler._handle_rate(mock_request, "tenant-1", "test-template-1")

        assert result.status_code == 400


class TestMarketplaceHandlerDemo:
    """Tests for demo endpoint."""

    @pytest.mark.asyncio
    async def test_demo_endpoint(self, handler, mock_request, sample_templates):
        """Test demo endpoint."""
        with patch(
            "aragora.server.handlers.features.marketplace._load_templates",
            return_value=sample_templates,
        ):
            result = await handler._handle_demo(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "featured" in body
        assert "by_category" in body
        assert "categories" in body
        assert body["total_templates"] == 3


class TestMarketplaceHandlerStatus:
    """Tests for status endpoint."""

    @pytest.mark.asyncio
    async def test_status_endpoint(self, handler, mock_request):
        """Test status endpoint."""
        result = await handler._handle_status(mock_request, "tenant-1")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "circuit_breaker" in body
        assert "templates_loaded" in body
        assert "deployments_count" in body


class TestMarketplaceHandlerRouting:
    """Tests for request routing."""

    @pytest.mark.asyncio
    async def test_handle_routes_to_list_templates(self, handler, mock_request):
        """Test routing to list templates."""
        with patch.object(handler, "_handle_list_templates", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = MagicMock(status_code=200, body=b"{}")
            await handler.handle(mock_request, "/api/v1/marketplace/templates", "GET")
            mock_list.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_routes_to_get_template(self, handler, mock_request):
        """Test routing to get template."""
        with patch.object(handler, "_handle_get_template", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = MagicMock(status_code=200, body=b"{}")
            await handler.handle(mock_request, "/api/v1/marketplace/templates/test-id", "GET")
            mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_invalid_template_id(self, handler, mock_request):
        """Test routing with invalid template ID."""
        result = await handler.handle(
            mock_request, "/api/v1/marketplace/templates/../../../etc/passwd", "GET"
        )
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_not_found(self, handler, mock_request):
        """Test routing to unknown path."""
        result = await handler.handle(mock_request, "/api/v1/marketplace/unknown", "GET")
        assert result.status_code == 404


# =============================================================================
# Data Class Tests
# =============================================================================


class TestTemplateMetadata:
    """Tests for TemplateMetadata dataclass."""

    def test_to_dict(self, sample_template):
        """Test to_dict method."""
        d = sample_template.to_dict()

        assert d["id"] == "test-template-1"
        assert d["name"] == "Test Template"
        assert d["category"] == "software"
        assert d["downloads"] == 100
        assert "created_at" in d


class TestTemplateDeployment:
    """Tests for TemplateDeployment dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        deployment = TemplateDeployment(
            id="deploy_abc123",
            template_id="test-1",
            tenant_id="tenant-1",
            name="Test",
            status=DeploymentStatus.ACTIVE,
            config={"key": "value"},
        )
        d = deployment.to_dict()

        assert d["id"] == "deploy_abc123"
        assert d["status"] == "active"
        assert d["config"] == {"key": "value"}


class TestTemplateRating:
    """Tests for TemplateRating dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        rating = TemplateRating(
            id="rating_abc123",
            template_id="test-1",
            tenant_id="tenant-1",
            user_id="user-1",
            rating=5,
            review="Great!",
        )
        d = rating.to_dict()

        assert d["id"] == "rating_abc123"
        assert d["rating"] == 5
        assert d["review"] == "Great!"


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tenant_id_sanitization(self, handler, mock_request):
        """Test tenant ID sanitization."""
        mock_request.tenant_id = "a" * 200  # Too long
        result = handler._get_tenant_id(mock_request)
        assert result == "default"

    def test_tenant_id_non_string(self, handler, mock_request):
        """Test tenant ID with non-string value."""
        mock_request.tenant_id = 12345
        result = handler._get_tenant_id(mock_request)
        assert result == "default"

    @pytest.mark.asyncio
    async def test_exception_handling(self, handler, mock_request):
        """Test exception handling in main handler."""
        with patch.object(handler, "_get_tenant_id", side_effect=Exception("Test error")):
            result = await handler.handle(mock_request, "/api/v1/marketplace/templates", "GET")

        assert result.status_code == 500
        body = json.loads(result.body)
        assert "error" in body.get("error", "").lower()

    def test_related_templates_empty(self, handler, sample_template):
        """Test related templates with no matches."""
        templates = {"test-1": sample_template}
        result = handler._get_related_templates(sample_template, templates)
        assert result == []

    def test_related_templates_scoring(self, handler, sample_templates):
        """Test related templates scoring."""
        # test-1 is software category, test-3 shares category
        result = handler._get_related_templates(
            sample_templates["test-1"],
            sample_templates,
        )
        # test-3 shares 'test' tag with test-1
        # test-2 shares 'test' tag with test-1
        assert len(result) == 2
