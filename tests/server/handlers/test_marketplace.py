"""Tests for the Marketplace Handler."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.marketplace import (
    MarketplaceHandler,
    TemplateCategory,
    TemplateMetadata,
    TemplateDeployment,
    TemplateRating,
    DeploymentStatus,
    handle_marketplace,
    get_marketplace_handler,
    _load_templates,
    _templates_cache,
    _deployments,
    _ratings,
    CATEGORY_INFO,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before each test."""
    _templates_cache.clear()
    _deployments.clear()
    _ratings.clear()
    yield
    _templates_cache.clear()
    _deployments.clear()
    _ratings.clear()


@pytest.fixture
def handler():
    """Create a handler instance."""
    return MarketplaceHandler()


@pytest.fixture
def mock_request():
    """Create a mock request."""
    request = MagicMock()
    request.tenant_id = "test_tenant"
    request.query = {}
    return request


# =============================================================================
# Data Class Tests
# =============================================================================


class TestTemplateMetadata:
    """Test TemplateMetadata dataclass."""

    def test_creation(self):
        """Test metadata creation."""
        meta = TemplateMetadata(
            id="test_template",
            name="Test Template",
            description="A test template",
            version="1.0.0",
            category=TemplateCategory.ACCOUNTING,
            tags=["test", "accounting"],
        )

        assert meta.id == "test_template"
        assert meta.name == "Test Template"
        assert meta.category == TemplateCategory.ACCOUNTING
        assert "test" in meta.tags

    def test_to_dict(self):
        """Test conversion to dictionary."""
        meta = TemplateMetadata(
            id="test_template",
            name="Test Template",
            description="A test template",
            version="1.0.0",
            category=TemplateCategory.LEGAL,
        )

        result = meta.to_dict()
        assert result["id"] == "test_template"
        assert result["category"] == "legal"
        assert "created_at" in result


class TestTemplateDeployment:
    """Test TemplateDeployment dataclass."""

    def test_creation(self):
        """Test deployment creation."""
        deployment = TemplateDeployment(
            id="deploy_123",
            template_id="template_456",
            tenant_id="tenant_789",
            name="My Deployment",
            status=DeploymentStatus.ACTIVE,
        )

        assert deployment.id == "deploy_123"
        assert deployment.status == DeploymentStatus.ACTIVE
        assert deployment.run_count == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        deployment = TemplateDeployment(
            id="deploy_123",
            template_id="template_456",
            tenant_id="tenant_789",
            name="My Deployment",
            status=DeploymentStatus.PENDING,
        )

        result = deployment.to_dict()
        assert result["status"] == "pending"
        assert "deployed_at" in result


class TestTemplateRating:
    """Test TemplateRating dataclass."""

    def test_creation(self):
        """Test rating creation."""
        rating = TemplateRating(
            id="rating_123",
            template_id="template_456",
            tenant_id="tenant_789",
            user_id="user_001",
            rating=5,
            review="Great template!",
        )

        assert rating.rating == 5
        assert rating.review == "Great template!"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        rating = TemplateRating(
            id="rating_123",
            template_id="template_456",
            tenant_id="tenant_789",
            user_id="user_001",
            rating=4,
        )

        result = rating.to_dict()
        assert result["rating"] == 4
        assert "created_at" in result


# =============================================================================
# Handler Tests
# =============================================================================


class TestMarketplaceHandler:
    """Test MarketplaceHandler class."""

    def test_handler_routes(self):
        """Test handler has correct routes."""
        handler = MarketplaceHandler()
        assert "/api/v1/marketplace/templates" in handler.ROUTES
        assert "/api/v1/marketplace/categories" in handler.ROUTES
        assert "/api/v1/marketplace/search" in handler.ROUTES
        assert "/api/v1/marketplace/popular" in handler.ROUTES
        assert "/api/v1/marketplace/demo" in handler.ROUTES

    def test_get_handler_instance(self):
        """Test getting handler instance."""
        handler = get_marketplace_handler()
        assert isinstance(handler, MarketplaceHandler)


# =============================================================================
# List Templates Tests
# =============================================================================


class TestListTemplates:
    """Test listing templates."""

    @pytest.mark.asyncio
    async def test_list_all_templates(self, handler, mock_request):
        """Test listing all templates."""
        result = await handler.handle(mock_request, "/api/v1/marketplace/templates", "GET")

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "data" in data
        assert "templates" in data["data"]
        assert "total" in data["data"]

    @pytest.mark.asyncio
    async def test_list_templates_with_category_filter(self, handler, mock_request):
        """Test filtering by category."""
        mock_request.query = {"category": "accounting"}

        result = await handler.handle(mock_request, "/api/v1/marketplace/templates", "GET")

        assert result.status_code == 200
        data = json.loads(result.body)
        templates = data["data"]["templates"]
        # All returned templates should be in accounting category
        for t in templates:
            assert t["category"] == "accounting"

    @pytest.mark.asyncio
    async def test_list_templates_with_pagination(self, handler, mock_request):
        """Test pagination."""
        mock_request.query = {"limit": "5", "offset": "0"}

        result = await handler.handle(mock_request, "/api/v1/marketplace/templates", "GET")

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["data"]["limit"] == 5
        assert data["data"]["offset"] == 0


# =============================================================================
# List Categories Tests
# =============================================================================


class TestListCategories:
    """Test listing categories."""

    @pytest.mark.asyncio
    async def test_list_categories(self, handler, mock_request):
        """Test listing all categories."""
        result = await handler.handle(mock_request, "/api/v1/marketplace/categories", "GET")

        assert result.status_code == 200
        data = json.loads(result.body)
        categories = data["data"]["categories"]

        # Should have all template categories
        category_ids = [c["id"] for c in categories]
        assert "accounting" in category_ids
        assert "legal" in category_ids
        assert "healthcare" in category_ids

    def test_category_info_completeness(self):
        """Test that all categories have info."""
        for category in TemplateCategory:
            assert category in CATEGORY_INFO
            info = CATEGORY_INFO[category]
            assert "name" in info
            assert "description" in info
            assert "icon" in info
            assert "color" in info


# =============================================================================
# Search Tests
# =============================================================================


class TestSearch:
    """Test searching templates."""

    @pytest.mark.asyncio
    async def test_search_by_query(self, handler, mock_request):
        """Test searching by text query."""
        mock_request.query = {"q": "invoice"}

        result = await handler.handle(mock_request, "/api/v1/marketplace/search", "GET")

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "results" in data["data"]
        assert "query" in data["data"]
        assert data["data"]["query"] == "invoice"

    @pytest.mark.asyncio
    async def test_search_with_filters(self, handler, mock_request):
        """Test searching with multiple filters."""
        mock_request.query = {
            "q": "review",
            "category": "software",
        }

        result = await handler.handle(mock_request, "/api/v1/marketplace/search", "GET")

        assert result.status_code == 200
        data = json.loads(result.body)
        # Results should match filters
        for t in data["data"]["results"]:
            assert t["category"] == "software"

    @pytest.mark.asyncio
    async def test_search_empty_query(self, handler, mock_request):
        """Test searching with empty query returns all."""
        mock_request.query = {}

        result = await handler.handle(mock_request, "/api/v1/marketplace/search", "GET")

        assert result.status_code == 200


# =============================================================================
# Popular Templates Tests
# =============================================================================


class TestPopularTemplates:
    """Test getting popular templates."""

    @pytest.mark.asyncio
    async def test_get_popular(self, handler, mock_request):
        """Test getting popular templates."""
        result = await handler.handle(mock_request, "/api/v1/marketplace/popular", "GET")

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "popular" in data["data"]

    @pytest.mark.asyncio
    async def test_popular_with_limit(self, handler, mock_request):
        """Test limiting popular results."""
        mock_request.query = {"limit": "3"}

        result = await handler.handle(mock_request, "/api/v1/marketplace/popular", "GET")

        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["data"]["popular"]) <= 3


# =============================================================================
# Get Template Tests
# =============================================================================


class TestGetTemplate:
    """Test getting template details."""

    @pytest.mark.asyncio
    async def test_get_existing_template(self, handler, mock_request):
        """Test getting an existing template."""
        # First list templates to get an ID
        list_result = await handler.handle(mock_request, "/api/v1/marketplace/templates", "GET")
        list_data = json.loads(list_result.body)

        if list_data["data"]["templates"]:
            template_id = list_data["data"]["templates"][0]["id"]

            result = await handler.handle(
                mock_request, f"/api/v1/marketplace/templates/{template_id}", "GET"
            )

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["data"]["template"]["id"] == template_id

    @pytest.mark.asyncio
    async def test_get_nonexistent_template(self, handler, mock_request):
        """Test getting a template that doesn't exist."""
        result = await handler.handle(
            mock_request, "/api/v1/marketplace/templates/nonexistent_template", "GET"
        )

        assert result.status_code == 404


# =============================================================================
# Deploy Template Tests
# =============================================================================


class TestDeployTemplate:
    """Test deploying templates."""

    @pytest.mark.asyncio
    async def test_deploy_template(self, handler, mock_request):
        """Test deploying a template."""
        # First list templates to get an ID
        list_result = await handler.handle(mock_request, "/api/v1/marketplace/templates", "GET")
        list_data = json.loads(list_result.body)

        if list_data["data"]["templates"]:
            template_id = list_data["data"]["templates"][0]["id"]

            mock_request.json = AsyncMock(
                return_value={
                    "name": "My Custom Deployment",
                    "config": {"auto_approve_threshold": 1000},
                }
            )

            result = await handler.handle(
                mock_request, f"/api/v1/marketplace/templates/{template_id}/deploy", "POST"
            )

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "deployment" in data["data"]
            assert data["data"]["deployment"]["name"] == "My Custom Deployment"
            assert data["data"]["deployment"]["status"] == "active"

    @pytest.mark.asyncio
    async def test_deploy_nonexistent_template(self, handler, mock_request):
        """Test deploying a template that doesn't exist."""
        mock_request.json = AsyncMock(return_value={"name": "Test"})

        result = await handler.handle(
            mock_request, "/api/v1/marketplace/templates/nonexistent/deploy", "POST"
        )

        assert result.status_code == 404


# =============================================================================
# List Deployments Tests
# =============================================================================


class TestListDeployments:
    """Test listing deployments."""

    @pytest.mark.asyncio
    async def test_list_deployments_empty(self, handler, mock_request):
        """Test listing deployments when none exist."""
        result = await handler.handle(mock_request, "/api/v1/marketplace/deployments", "GET")

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["data"]["deployments"] == []
        assert data["data"]["total"] == 0

    @pytest.mark.asyncio
    async def test_list_deployments_after_deploy(self, handler, mock_request):
        """Test listing deployments after deploying."""
        # First deploy a template
        list_result = await handler.handle(mock_request, "/api/v1/marketplace/templates", "GET")
        list_data = json.loads(list_result.body)

        if list_data["data"]["templates"]:
            template_id = list_data["data"]["templates"][0]["id"]
            mock_request.json = AsyncMock(return_value={"name": "Test Deployment"})

            await handler.handle(
                mock_request, f"/api/v1/marketplace/templates/{template_id}/deploy", "POST"
            )

            # Now list deployments
            result = await handler.handle(mock_request, "/api/v1/marketplace/deployments", "GET")

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["data"]["total"] >= 1


# =============================================================================
# Rate Template Tests
# =============================================================================


class TestRateTemplate:
    """Test rating templates."""

    @pytest.mark.asyncio
    async def test_rate_template(self, handler, mock_request):
        """Test rating a template."""
        # First list templates to get an ID
        list_result = await handler.handle(mock_request, "/api/v1/marketplace/templates", "GET")
        list_data = json.loads(list_result.body)

        if list_data["data"]["templates"]:
            template_id = list_data["data"]["templates"][0]["id"]

            mock_request.json = AsyncMock(
                return_value={
                    "rating": 5,
                    "review": "Excellent template!",
                }
            )
            mock_request.user_id = "test_user"

            result = await handler.handle(
                mock_request, f"/api/v1/marketplace/templates/{template_id}/rate", "POST"
            )

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["data"]["rating"]["rating"] == 5
            assert data["data"]["template_rating"]["count"] == 1

    @pytest.mark.asyncio
    async def test_rate_invalid_rating(self, handler, mock_request):
        """Test rating with invalid value."""
        list_result = await handler.handle(mock_request, "/api/v1/marketplace/templates", "GET")
        list_data = json.loads(list_result.body)

        if list_data["data"]["templates"]:
            template_id = list_data["data"]["templates"][0]["id"]

            mock_request.json = AsyncMock(return_value={"rating": 10})

            result = await handler.handle(
                mock_request, f"/api/v1/marketplace/templates/{template_id}/rate", "POST"
            )

            assert result.status_code == 400


# =============================================================================
# Demo Endpoint Tests
# =============================================================================


class TestDemoEndpoint:
    """Test demo endpoint."""

    @pytest.mark.asyncio
    async def test_demo_endpoint(self, handler, mock_request):
        """Test getting demo marketplace data."""
        result = await handler.handle(mock_request, "/api/v1/marketplace/demo", "GET")

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "featured" in data["data"]
        assert "by_category" in data["data"]
        assert "categories" in data["data"]
        assert "total_templates" in data["data"]


# =============================================================================
# Handle Function Tests
# =============================================================================


class TestHandleFunction:
    """Test the module-level handle function."""

    @pytest.mark.asyncio
    async def test_entry_point(self):
        """Test the handle_marketplace entry point."""
        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {}

        result = await handle_marketplace(request, "/api/v1/marketplace/templates", "GET")

        assert result.status_code == 200


# =============================================================================
# Not Found Tests
# =============================================================================


class TestNotFoundRoute:
    """Test unknown routes."""

    @pytest.mark.asyncio
    async def test_unknown_route(self, handler, mock_request):
        """Test that unknown routes return 404."""
        result = await handler.handle(mock_request, "/api/v1/marketplace/unknown", "GET")

        assert result.status_code == 404


# =============================================================================
# Import Tests
# =============================================================================


class TestImports:
    """Test module imports."""

    def test_import_from_package(self):
        """Test importing from the features package."""
        from aragora.server.handlers.features import (
            MarketplaceHandler,
            handle_marketplace,
            get_marketplace_handler,
            TemplateCategory,
            DeploymentStatus,
        )

        assert MarketplaceHandler is not None
        assert handle_marketplace is not None
        assert TemplateCategory.ACCOUNTING is not None
        assert DeploymentStatus.ACTIVE is not None
