"""Tests for the features marketplace handler.

Covers all routes and behavior of MarketplaceHandler:
- GET  /api/v1/marketplace/templates         - List all available templates
- GET  /api/v1/marketplace/templates/{id}    - Get template details
- GET  /api/v1/marketplace/categories        - List template categories
- GET  /api/v1/marketplace/search            - Search templates
- POST /api/v1/marketplace/templates/{id}/deploy - Deploy a template
- GET  /api/v1/marketplace/deployments       - List deployed templates
- GET  /api/v1/marketplace/deployments/{id}  - Get deployment details
- DELETE /api/v1/marketplace/deployments/{id} - Archive a deployment
- GET  /api/v1/marketplace/popular           - Get popular templates
- POST /api/v1/marketplace/templates/{id}/rate - Rate a template
- GET  /api/v1/marketplace/demo              - Get demo marketplace data
- GET  /api/v1/marketplace/status            - Circuit breaker and health status
- can_handle() routing
- Tenant ID extraction
- Error and edge cases
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.features.marketplace.handler import MarketplaceHandler
from aragora.server.handlers.features.marketplace.models import (
    CATEGORY_INFO,
    DeploymentStatus,
    TemplateCategory,
    TemplateDeployment,
    TemplateMetadata,
    TemplateRating,
)
from aragora.server.handlers.features.marketplace.store import (
    _clear_marketplace_state,
    get_deployments,
    get_download_counts,
    get_ratings,
    _get_tenant_deployments,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    if hasattr(result, "body"):
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode("utf-8"))
        if isinstance(result.body, str):
            return json.loads(result.body)
    return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    if hasattr(result, "status_code"):
        return result.status_code
    return 200


# ---------------------------------------------------------------------------
# Mock Circuit Breaker
# ---------------------------------------------------------------------------


class MockCircuitBreaker:
    """Minimal mock circuit breaker that always allows by default."""

    def __init__(self, *, allowed: bool = True):
        self._allowed = allowed
        self._successes = 0
        self._failures = 0

    def is_allowed(self) -> bool:
        return self._allowed

    def record_success(self) -> None:
        self._successes += 1

    def record_failure(self) -> None:
        self._failures += 1

    def get_status(self) -> dict[str, Any]:
        return {
            "state": "closed" if self._allowed else "open",
            "failure_count": self._failures,
            "success_count": self._successes,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Sample templates used across all tests
_NOW = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _make_templates() -> dict[str, TemplateMetadata]:
    """Build a small set of sample templates for testing."""
    return {
        "tpl-software-1": TemplateMetadata(
            id="tpl-software-1",
            name="Code Review Workflow",
            description="Automated code review with debate",
            version="1.0.0",
            category=TemplateCategory.SOFTWARE,
            tags=["code-review", "automation", "ci"],
            downloads=150,
            rating=4.5,
            rating_count=10,
            steps_count=5,
            has_debate=True,
            has_human_checkpoint=False,
            created_at=_NOW,
            updated_at=_NOW,
        ),
        "tpl-legal-1": TemplateMetadata(
            id="tpl-legal-1",
            name="Contract Analysis",
            description="Legal contract review",
            version="2.0.0",
            category=TemplateCategory.LEGAL,
            tags=["contracts", "compliance"],
            downloads=80,
            rating=4.0,
            rating_count=5,
            steps_count=3,
            has_debate=False,
            has_human_checkpoint=True,
            created_at=_NOW,
            updated_at=_NOW,
        ),
        "tpl-healthcare-1": TemplateMetadata(
            id="tpl-healthcare-1",
            name="Clinical Review",
            description="Healthcare clinical decision review",
            version="1.2.0",
            category=TemplateCategory.HEALTHCARE,
            tags=["clinical", "hipaa"],
            downloads=50,
            rating=3.8,
            rating_count=3,
            steps_count=7,
            has_debate=True,
            has_human_checkpoint=True,
            created_at=_NOW,
            updated_at=_NOW,
        ),
    }


@pytest.fixture(autouse=True)
def _reset_marketplace_state():
    """Clear all marketplace in-memory state before and after each test."""
    _clear_marketplace_state()
    yield
    _clear_marketplace_state()


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiters between tests to avoid 429 cross-test pollution."""
    yield
    try:
        from aragora.server.handlers.utils import rate_limit as rl_mod

        for attr_name in dir(rl_mod):
            obj = getattr(rl_mod, attr_name, None)
            if hasattr(obj, "_requests") and isinstance(obj._requests, dict):
                obj._requests = defaultdict(list)
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def templates():
    """Return a fresh set of sample templates."""
    return _make_templates()


@pytest.fixture
def mock_cb():
    """Return a mock circuit breaker (allowed)."""
    return MockCircuitBreaker(allowed=True)


@pytest.fixture
def mock_cb_open():
    """Return a mock circuit breaker that rejects requests."""
    return MockCircuitBreaker(allowed=False)


@pytest.fixture
def handler(templates, mock_cb):
    """Create a MarketplaceHandler with patched template loading and circuit breaker."""
    with patch(
        "aragora.server.handlers.features.marketplace.handler._load_templates_proxy",
        return_value=templates,
    ), patch(
        "aragora.server.handlers.features.marketplace.handler._get_marketplace_circuit_breaker_proxy",
        return_value=mock_cb,
    ), patch(
        "aragora.server.handlers.features.marketplace.handler.get_marketplace_circuit_breaker_status",
        return_value=mock_cb.get_status(),
    ):
        h = MarketplaceHandler(server_context={})
        yield h


class MockRequest:
    """Mock request object carrying query params, body, tenant_id, and user_id."""

    def __init__(
        self,
        query: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        user_id: str = "test-user-001",
    ):
        self.query = query or {}
        self._body = body
        self.tenant_id = tenant_id
        self.user_id = user_id

    async def json(self) -> dict[str, Any]:
        return self._body or {}


def _req(
    query: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    tenant_id: str = "tenant-1",
) -> MockRequest:
    """Shorthand for building a MockRequest."""
    return MockRequest(query=query, body=body, tenant_id=tenant_id)


# ---------------------------------------------------------------------------
# Helper to invoke handler.handle with patching
# ---------------------------------------------------------------------------


async def _call(
    handler_instance: MarketplaceHandler,
    path: str,
    method: str = "GET",
    templates: dict | None = None,
    mock_cb: MockCircuitBreaker | None = None,
    request: MockRequest | None = None,
    cb_status: dict | None = None,
):
    """Invoke handler.handle with standard patches applied."""
    if templates is None:
        templates = _make_templates()
    if mock_cb is None:
        mock_cb = MockCircuitBreaker(allowed=True)
    if request is None:
        request = _req()
    if cb_status is None:
        cb_status = mock_cb.get_status()

    async def _parse_body(req, *, context=""):
        if hasattr(req, "_body") and req._body is not None:
            return req._body, None
        if hasattr(req, "json"):
            try:
                return await req.json(), None
            except Exception:
                return {}, None
        return {}, None

    with patch(
        "aragora.server.handlers.features.marketplace.handler._load_templates_proxy",
        return_value=templates,
    ), patch(
        "aragora.server.handlers.features.marketplace.handler._get_marketplace_circuit_breaker_proxy",
        return_value=mock_cb,
    ), patch(
        "aragora.server.handlers.features.marketplace.handler.get_marketplace_circuit_breaker_status",
        return_value=cb_status,
    ), patch(
        "aragora.server.handlers.features.marketplace.handler._parse_json_body_proxy",
        side_effect=_parse_body,
    ), patch(
        "aragora.server.handlers.features.marketplace.handler.get_deployments",
        wraps=get_deployments,
    ), patch(
        "aragora.server.handlers.features.marketplace.handler.get_ratings",
        wraps=get_ratings,
    ), patch(
        "aragora.server.handlers.features.marketplace.handler.get_download_counts",
        wraps=get_download_counts,
    ):
        return await handler_instance.handle(request, path, method)


# ===========================================================================
# can_handle() Tests
# ===========================================================================


class TestCanHandle:
    """Tests for MarketplaceHandler.can_handle routing."""

    def test_handles_templates_list(self, handler):
        assert handler.can_handle("/api/v1/marketplace/templates") is True

    def test_handles_template_by_id(self, handler):
        assert handler.can_handle("/api/v1/marketplace/templates/tpl-1") is True

    def test_handles_deploy(self, handler):
        assert handler.can_handle("/api/v1/marketplace/templates/tpl-1/deploy") is True

    def test_handles_rate(self, handler):
        assert handler.can_handle("/api/v1/marketplace/templates/tpl-1/rate") is True

    def test_handles_categories(self, handler):
        assert handler.can_handle("/api/v1/marketplace/categories") is True

    def test_handles_search(self, handler):
        assert handler.can_handle("/api/v1/marketplace/search") is True

    def test_handles_deployments(self, handler):
        assert handler.can_handle("/api/v1/marketplace/deployments") is True

    def test_handles_deployment_by_id(self, handler):
        assert handler.can_handle("/api/v1/marketplace/deployments/deploy-1") is True

    def test_handles_popular(self, handler):
        assert handler.can_handle("/api/v1/marketplace/popular") is True

    def test_handles_demo(self, handler):
        assert handler.can_handle("/api/v1/marketplace/demo") is True

    def test_handles_status(self, handler):
        assert handler.can_handle("/api/v1/marketplace/status") is True

    def test_rejects_unknown_path(self, handler):
        assert handler.can_handle("/api/v1/other/thing") is False

    def test_rejects_partial_match(self, handler):
        assert handler.can_handle("/api/v1/marketplace") is False


# ===========================================================================
# Tenant ID Extraction Tests
# ===========================================================================


class TestGetTenantId:
    """Tests for _get_tenant_id."""

    def test_extracts_tenant_id(self, handler):
        req = _req(tenant_id="my-tenant")
        assert handler._get_tenant_id(req) == "my-tenant"

    def test_default_when_none(self, handler):
        req = MockRequest(tenant_id=None)
        assert handler._get_tenant_id(req) == "default"

    def test_default_when_empty_string(self, handler):
        req = MockRequest(tenant_id="")
        assert handler._get_tenant_id(req) == "default"

    def test_default_when_not_string(self, handler):
        req = MockRequest()
        req.tenant_id = 12345
        assert handler._get_tenant_id(req) == "default"

    def test_default_when_too_long(self, handler):
        req = MockRequest(tenant_id="x" * 200)
        assert handler._get_tenant_id(req) == "default"


# ===========================================================================
# List Templates Tests
# ===========================================================================


class TestListTemplates:
    """Tests for GET /api/v1/marketplace/templates."""

    @pytest.mark.asyncio
    async def test_list_returns_200(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/templates", templates=templates)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_list_returns_all_templates(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/templates", templates=templates)
        body = _body(result)
        assert body["total"] == 3
        assert len(body["templates"]) == 3

    @pytest.mark.asyncio
    async def test_list_sorted_by_downloads(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/templates", templates=templates)
        body = _body(result)
        downloads = [t["downloads"] for t in body["templates"]]
        assert downloads == sorted(downloads, reverse=True)

    @pytest.mark.asyncio
    async def test_list_includes_pagination(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/templates", templates=templates)
        body = _body(result)
        assert "limit" in body
        assert "offset" in body
        assert body["offset"] == 0

    @pytest.mark.asyncio
    async def test_list_with_category_filter(self, handler, templates):
        req = _req(query={"category": "software"})
        result = await _call(
            handler, "/api/v1/marketplace/templates", templates=templates, request=req
        )
        body = _body(result)
        assert body["total"] == 1
        assert body["templates"][0]["category"] == "software"

    @pytest.mark.asyncio
    async def test_list_invalid_category_returns_400(self, handler, templates):
        req = _req(query={"category": "nonexistent"})
        result = await _call(
            handler, "/api/v1/marketplace/templates", templates=templates, request=req
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_with_custom_pagination(self, handler, templates):
        req = _req(query={"limit": "1", "offset": "1"})
        result = await _call(
            handler, "/api/v1/marketplace/templates", templates=templates, request=req
        )
        body = _body(result)
        assert body["limit"] == 1
        assert body["offset"] == 1
        assert len(body["templates"]) == 1

    @pytest.mark.asyncio
    async def test_list_with_invalid_limit(self, handler, templates):
        req = _req(query={"limit": "not_a_number"})
        result = await _call(
            handler, "/api/v1/marketplace/templates", templates=templates, request=req
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_list_empty_templates(self, handler):
        result = await _call(handler, "/api/v1/marketplace/templates", templates={})
        body = _body(result)
        assert body["total"] == 0
        assert body["templates"] == []


# ===========================================================================
# Get Template Detail Tests
# ===========================================================================


class TestGetTemplate:
    """Tests for GET /api/v1/marketplace/templates/{id}."""

    @pytest.mark.asyncio
    async def test_get_returns_200(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1",
            templates=templates,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_get_returns_template_data(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1",
            templates=templates,
        )
        body = _body(result)
        assert body["template"]["id"] == "tpl-software-1"
        assert body["template"]["name"] == "Code Review Workflow"

    @pytest.mark.asyncio
    async def test_get_includes_ratings(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1",
            templates=templates,
        )
        body = _body(result)
        assert "ratings" in body
        assert body["ratings"]["average"] == 4.5
        assert body["ratings"]["count"] == 10

    @pytest.mark.asyncio
    async def test_get_includes_related(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1",
            templates=templates,
        )
        body = _body(result)
        assert "related" in body
        # Should include the other 2 templates (related by tags or fallback)
        related_ids = [r["id"] for r in body["related"]]
        assert "tpl-software-1" not in related_ids

    @pytest.mark.asyncio
    async def test_get_not_found(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/nonexistent",
            templates=templates,
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_invalid_id_special_chars(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/../../etc/passwd",
            templates=templates,
        )
        # Path splits to more than 6 parts, but the ID extraction gets "..".
        # _validate_template_id rejects ".." because it starts with a dot.
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_get_invalid_id_empty_segment(self, handler, templates):
        # The path "/api/v1/marketplace/templates/" would have parts[5] == ""
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/",
            templates=templates,
        )
        # Empty string fails validation
        assert _status(result) in (400, 404)


# ===========================================================================
# List Categories Tests
# ===========================================================================


class TestListCategories:
    """Tests for GET /api/v1/marketplace/categories."""

    @pytest.mark.asyncio
    async def test_categories_returns_200(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/categories", templates=templates)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_categories_has_all_enums(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/categories", templates=templates)
        body = _body(result)
        category_ids = {c["id"] for c in body["categories"]}
        for cat in TemplateCategory:
            assert cat.value in category_ids

    @pytest.mark.asyncio
    async def test_categories_include_counts(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/categories", templates=templates)
        body = _body(result)
        software_cat = next(c for c in body["categories"] if c["id"] == "software")
        assert software_cat["template_count"] == 1

    @pytest.mark.asyncio
    async def test_categories_include_info_fields(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/categories", templates=templates)
        body = _body(result)
        for cat in body["categories"]:
            assert "name" in cat
            assert "description" in cat
            assert "icon" in cat
            assert "color" in cat


# ===========================================================================
# Search Tests
# ===========================================================================


class TestSearch:
    """Tests for GET /api/v1/marketplace/search."""

    @pytest.mark.asyncio
    async def test_search_returns_200(self, handler, templates):
        req = _req(query={"q": "code review"})
        result = await _call(
            handler, "/api/v1/marketplace/search", templates=templates, request=req
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_search_filters_by_query(self, handler, templates):
        req = _req(query={"q": "contract"})
        result = await _call(
            handler, "/api/v1/marketplace/search", templates=templates, request=req
        )
        body = _body(result)
        assert body["total"] == 1
        assert body["results"][0]["id"] == "tpl-legal-1"

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_all(self, handler, templates):
        req = _req(query={})
        result = await _call(
            handler, "/api/v1/marketplace/search", templates=templates, request=req
        )
        body = _body(result)
        assert body["total"] == 3

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self, handler, templates):
        req = _req(query={"q": "", "category": "healthcare"})
        result = await _call(
            handler, "/api/v1/marketplace/search", templates=templates, request=req
        )
        body = _body(result)
        assert body["total"] == 1
        assert body["results"][0]["category"] == "healthcare"

    @pytest.mark.asyncio
    async def test_search_with_tags_filter(self, handler, templates):
        req = _req(query={"tags": "hipaa"})
        result = await _call(
            handler, "/api/v1/marketplace/search", templates=templates, request=req
        )
        body = _body(result)
        assert body["total"] == 1
        assert body["results"][0]["id"] == "tpl-healthcare-1"

    @pytest.mark.asyncio
    async def test_search_has_debate_filter(self, handler, templates):
        req = _req(query={"has_debate": "true"})
        result = await _call(
            handler, "/api/v1/marketplace/search", templates=templates, request=req
        )
        body = _body(result)
        # tpl-software-1 and tpl-healthcare-1 have debates
        assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_search_has_checkpoint_filter(self, handler, templates):
        req = _req(query={"has_checkpoint": "true"})
        result = await _call(
            handler, "/api/v1/marketplace/search", templates=templates, request=req
        )
        body = _body(result)
        # tpl-legal-1 and tpl-healthcare-1 have human checkpoints
        assert body["total"] == 2

    @pytest.mark.asyncio
    async def test_search_invalid_category(self, handler, templates):
        req = _req(query={"category": "nonexistent"})
        result = await _call(
            handler, "/api/v1/marketplace/search", templates=templates, request=req
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_search_includes_query_echo(self, handler, templates):
        req = _req(query={"q": "review"})
        result = await _call(
            handler, "/api/v1/marketplace/search", templates=templates, request=req
        )
        body = _body(result)
        assert "query" in body

    @pytest.mark.asyncio
    async def test_search_no_results(self, handler, templates):
        req = _req(query={"q": "zzzznonexistentzzzz"})
        result = await _call(
            handler, "/api/v1/marketplace/search", templates=templates, request=req
        )
        body = _body(result)
        assert body["total"] == 0
        assert body["results"] == []


# ===========================================================================
# Popular Templates Tests
# ===========================================================================


class TestPopular:
    """Tests for GET /api/v1/marketplace/popular."""

    @pytest.mark.asyncio
    async def test_popular_returns_200(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/popular", templates=templates)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_popular_sorted_by_downloads_and_rating(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/popular", templates=templates)
        body = _body(result)
        popular = body["popular"]
        assert len(popular) == 3
        # Sorted by (downloads, rating) descending
        assert popular[0]["id"] == "tpl-software-1"

    @pytest.mark.asyncio
    async def test_popular_respects_limit(self, handler, templates):
        req = _req(query={"limit": "1"})
        result = await _call(
            handler, "/api/v1/marketplace/popular", templates=templates, request=req
        )
        body = _body(result)
        assert len(body["popular"]) == 1

    @pytest.mark.asyncio
    async def test_popular_caps_at_50(self, handler, templates):
        req = _req(query={"limit": "999"})
        result = await _call(
            handler, "/api/v1/marketplace/popular", templates=templates, request=req
        )
        body = _body(result)
        # Limit capped at 50, but we only have 3 templates
        assert len(body["popular"]) == 3


# ===========================================================================
# Deploy Template Tests
# ===========================================================================


class TestDeploy:
    """Tests for POST /api/v1/marketplace/templates/{id}/deploy."""

    @pytest.mark.asyncio
    async def test_deploy_returns_200(self, handler, templates, mock_cb):
        req = _req(body={"name": "My deployment", "config": {"key": "value"}})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_deploy_creates_deployment(self, handler, templates, mock_cb):
        req = _req(body={"name": "My deployment"})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        body = _body(result)
        assert "deployment" in body
        dep = body["deployment"]
        assert dep["template_id"] == "tpl-software-1"
        assert dep["status"] == "active"
        assert dep["name"] == "My deployment"

    @pytest.mark.asyncio
    async def test_deploy_increments_downloads(self, handler, templates, mock_cb):
        req = _req(body={})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        body = _body(result)
        # Template download count should be incremented
        assert body["template"]["downloads"] >= 1

    @pytest.mark.asyncio
    async def test_deploy_uses_template_name_as_fallback(self, handler, templates, mock_cb):
        req = _req(body={})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        body = _body(result)
        # When name is None, it falls back to template name
        assert body["deployment"]["name"] == "Code Review Workflow"

    @pytest.mark.asyncio
    async def test_deploy_with_config(self, handler, templates, mock_cb):
        req = _req(body={"config": {"env": "prod", "version": "2"}})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        body = _body(result)
        assert body["deployment"]["config"] == {"env": "prod", "version": "2"}

    @pytest.mark.asyncio
    async def test_deploy_not_found(self, handler, templates, mock_cb):
        req = _req(body={})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/nonexistent/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_deploy_invalid_config_type(self, handler, templates, mock_cb):
        req = _req(body={"config": "not-a-dict"})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_deploy_config_too_many_keys(self, handler, templates, mock_cb):
        config = {f"key_{i}": f"val_{i}" for i in range(60)}
        req = _req(body={"config": config})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_deploy_invalid_name_type(self, handler, templates, mock_cb):
        req = _req(body={"name": 12345})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_deploy_name_too_long(self, handler, templates, mock_cb):
        req = _req(body={"name": "x" * 300})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_deploy_circuit_breaker_open(self, handler, templates, mock_cb_open):
        req = _req(body={})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb_open,
            request=req,
        )
        assert _status(result) == 503
        assert "unavailable" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_deploy_records_success(self, handler, templates, mock_cb):
        req = _req(body={})
        await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert mock_cb._successes == 1


# ===========================================================================
# List Deployments Tests
# ===========================================================================


class TestListDeployments:
    """Tests for GET /api/v1/marketplace/deployments."""

    @pytest.mark.asyncio
    async def test_list_empty(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/deployments", templates=templates)
        body = _body(result)
        assert body["total"] == 0
        assert body["deployments"] == []

    @pytest.mark.asyncio
    async def test_list_after_deploy(self, handler, templates, mock_cb):
        # First deploy a template
        req = _req(body={"name": "My deploy"})
        await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        # Then list
        result = await _call(
            handler,
            "/api/v1/marketplace/deployments",
            templates=templates,
            request=_req(),
        )
        body = _body(result)
        assert body["total"] == 1
        assert body["deployments"][0]["name"] == "My deploy"


# ===========================================================================
# Get Deployment Detail Tests
# ===========================================================================


class TestGetDeployment:
    """Tests for GET /api/v1/marketplace/deployments/{id}."""

    @pytest.mark.asyncio
    async def test_get_deployment_found(self, handler, templates, mock_cb):
        # Deploy first
        req = _req(body={"name": "Test deploy"})
        deploy_result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        dep_id = _body(deploy_result)["deployment"]["id"]

        # Get it
        result = await _call(
            handler,
            f"/api/v1/marketplace/deployments/{dep_id}",
            templates=templates,
            request=_req(),
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["deployment"]["id"] == dep_id
        assert body["template"]["id"] == "tpl-software-1"

    @pytest.mark.asyncio
    async def test_get_deployment_not_found(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/deployments/nonexistent123",
            templates=templates,
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_deployment_invalid_id(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/deployments/bad id!!",
            templates=templates,
        )
        assert _status(result) == 400


# ===========================================================================
# Delete (Archive) Deployment Tests
# ===========================================================================


class TestDeleteDeployment:
    """Tests for DELETE /api/v1/marketplace/deployments/{id}."""

    @pytest.mark.asyncio
    async def test_delete_archives_deployment(self, handler, templates, mock_cb):
        # Deploy first
        req = _req(body={"name": "To archive"})
        deploy_result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-legal-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        dep_id = _body(deploy_result)["deployment"]["id"]

        # Delete it
        result = await _call(
            handler,
            f"/api/v1/marketplace/deployments/{dep_id}",
            method="DELETE",
            templates=templates,
            request=_req(),
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["deployment"]["status"] == "archived"
        assert "archived" in body.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_delete_deployment_not_found(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/deployments/nonexistent123",
            method="DELETE",
            templates=templates,
        )
        assert _status(result) == 404


# ===========================================================================
# Rate Template Tests
# ===========================================================================


class TestRateTemplate:
    """Tests for POST /api/v1/marketplace/templates/{id}/rate."""

    @pytest.mark.asyncio
    async def test_rate_returns_200(self, handler, templates, mock_cb):
        req = _req(body={"rating": 5, "review": "Excellent!"})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_rate_creates_rating(self, handler, templates, mock_cb):
        req = _req(body={"rating": 4, "review": "Good template"})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        body = _body(result)
        assert "rating" in body
        assert body["rating"]["rating"] == 4
        assert body["rating"]["review"] == "Good template"

    @pytest.mark.asyncio
    async def test_rate_updates_template_average(self, handler, templates, mock_cb):
        req = _req(body={"rating": 2})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        body = _body(result)
        assert "template_rating" in body
        assert body["template_rating"]["count"] == 1
        assert body["template_rating"]["average"] == 2.0

    @pytest.mark.asyncio
    async def test_rate_without_review(self, handler, templates, mock_cb):
        req = _req(body={"rating": 5})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["rating"]["review"] is None

    @pytest.mark.asyncio
    async def test_rate_missing_rating(self, handler, templates, mock_cb):
        req = _req(body={"review": "no rating given"})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_rate_below_min(self, handler, templates, mock_cb):
        req = _req(body={"rating": 0})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_rate_above_max(self, handler, templates, mock_cb):
        req = _req(body={"rating": 6})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_rate_non_integer(self, handler, templates, mock_cb):
        req = _req(body={"rating": "five"})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_rate_float_value(self, handler, templates, mock_cb):
        req = _req(body={"rating": 3.5})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_rate_review_too_long(self, handler, templates, mock_cb):
        req = _req(body={"rating": 4, "review": "x" * 2001})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_rate_review_non_string(self, handler, templates, mock_cb):
        req = _req(body={"rating": 4, "review": 12345})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_rate_template_not_found(self, handler, templates, mock_cb):
        req = _req(body={"rating": 5})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/nonexistent/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_rate_circuit_breaker_open(self, handler, templates, mock_cb_open):
        req = _req(body={"rating": 5})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb_open,
            request=req,
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_rate_records_cb_success(self, handler, templates, mock_cb):
        req = _req(body={"rating": 5})
        await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        assert mock_cb._successes == 1


# ===========================================================================
# Demo Data Tests
# ===========================================================================


class TestDemo:
    """Tests for GET /api/v1/marketplace/demo."""

    @pytest.mark.asyncio
    async def test_demo_returns_200(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/demo", templates=templates)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_demo_has_featured(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/demo", templates=templates)
        body = _body(result)
        assert "featured" in body
        assert len(body["featured"]) <= 6
        assert len(body["featured"]) > 0

    @pytest.mark.asyncio
    async def test_demo_has_by_category(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/demo", templates=templates)
        body = _body(result)
        assert "by_category" in body
        assert "software" in body["by_category"]

    @pytest.mark.asyncio
    async def test_demo_has_categories_info(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/demo", templates=templates)
        body = _body(result)
        assert "categories" in body
        assert len(body["categories"]) == len(TemplateCategory)

    @pytest.mark.asyncio
    async def test_demo_total_templates(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/demo", templates=templates)
        body = _body(result)
        assert body["total_templates"] == 3


# ===========================================================================
# Status / Health Tests
# ===========================================================================


class TestStatus:
    """Tests for GET /api/v1/marketplace/status."""

    @pytest.mark.asyncio
    async def test_status_returns_200(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/status", templates=templates)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_status_healthy_when_cb_closed(self, handler, templates):
        cb_status = {"state": "closed", "failure_count": 0}
        result = await _call(
            handler,
            "/api/v1/marketplace/status",
            templates=templates,
            cb_status=cb_status,
        )
        body = _body(result)
        assert body["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_status_degraded_when_cb_open(self, handler, templates):
        cb_status = {"state": "open", "failure_count": 5}
        result = await _call(
            handler,
            "/api/v1/marketplace/status",
            templates=templates,
            cb_status=cb_status,
        )
        body = _body(result)
        assert body["status"] == "degraded"

    @pytest.mark.asyncio
    async def test_status_includes_template_count(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/status", templates=templates)
        body = _body(result)
        assert body["templates_loaded"] == 3

    @pytest.mark.asyncio
    async def test_status_includes_circuit_breaker(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/status", templates=templates)
        body = _body(result)
        assert "circuit_breaker" in body

    @pytest.mark.asyncio
    async def test_status_includes_deployment_count(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/status", templates=templates)
        body = _body(result)
        assert "deployments_count" in body

    @pytest.mark.asyncio
    async def test_status_includes_ratings_count(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/status", templates=templates)
        body = _body(result)
        assert "ratings_count" in body


# ===========================================================================
# 404 / Unknown Route Tests
# ===========================================================================


class TestNotFound:
    """Tests for unknown routes returning 404."""

    @pytest.mark.asyncio
    async def test_unknown_path(self, handler, templates):
        result = await _call(handler, "/api/v1/marketplace/unknown", templates=templates)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_wrong_method_on_templates(self, handler, templates):
        result = await _call(
            handler, "/api/v1/marketplace/templates", method="DELETE", templates=templates
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_on_deploy_path(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="GET",
            templates=templates,
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_get_on_rate_path(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="GET",
            templates=templates,
        )
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_post_on_deployments_list(self, handler, templates):
        result = await _call(
            handler, "/api/v1/marketplace/deployments", method="POST", templates=templates
        )
        assert _status(result) == 404


# ===========================================================================
# Handler Initialization Tests
# ===========================================================================


class TestHandlerInit:
    """Tests for MarketplaceHandler initialization."""

    def test_init_with_server_context(self, templates, mock_cb):
        with patch(
            "aragora.server.handlers.features.marketplace.handler._load_templates_proxy",
            return_value=templates,
        ):
            h = MarketplaceHandler(server_context={"key": "value"})
        assert h.ctx == {"key": "value"}

    def test_init_with_none_context(self, templates, mock_cb):
        with patch(
            "aragora.server.handlers.features.marketplace.handler._load_templates_proxy",
            return_value=templates,
        ):
            h = MarketplaceHandler(server_context=None)
        assert h.ctx == {}

    def test_init_with_empty_context(self, templates, mock_cb):
        with patch(
            "aragora.server.handlers.features.marketplace.handler._load_templates_proxy",
            return_value=templates,
        ):
            h = MarketplaceHandler(server_context={})
        assert h.ctx == {}


# ===========================================================================
# Edge Cases and Security Tests
# ===========================================================================


class TestEdgeCases:
    """Edge case and security tests."""

    @pytest.mark.asyncio
    async def test_path_traversal_in_template_id(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/../../../etc/passwd",
            templates=templates,
        )
        # ".." fails the SAFE_ID_PATTERN because it starts with a dot
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_sql_injection_in_template_id(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/'; DROP TABLE templates;--",
            templates=templates,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_very_long_template_id(self, handler, templates):
        long_id = "a" * 200
        result = await _call(
            handler,
            f"/api/v1/marketplace/templates/{long_id}",
            templates=templates,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_null_bytes_in_template_id(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl\x00-1",
            templates=templates,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_unicode_template_id(self, handler, templates):
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/\u00e9\u00e8\u00ea",
            templates=templates,
        )
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_deploy_with_empty_body(self, handler, templates, mock_cb):
        req = _req(body={})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        # Empty body is valid: name defaults to template name, config defaults to {}
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_rate_with_empty_body(self, handler, templates, mock_cb):
        req = _req(body={})
        result = await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/rate",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req,
        )
        # Missing rating should be 400
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_handler_catches_internal_errors(self, handler, templates):
        """When _load_templates_proxy raises ValueError, handler returns 500."""
        mock_cb = MockCircuitBreaker(allowed=True)

        async def _parse_body(req, *, context=""):
            return {}, None

        with patch(
            "aragora.server.handlers.features.marketplace.handler._load_templates_proxy",
            side_effect=ValueError("broken"),
        ), patch(
            "aragora.server.handlers.features.marketplace.handler._get_marketplace_circuit_breaker_proxy",
            return_value=mock_cb,
        ), patch(
            "aragora.server.handlers.features.marketplace.handler._parse_json_body_proxy",
            side_effect=_parse_body,
        ):
            result = await handler.handle(_req(), "/api/v1/marketplace/templates", "GET")
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_multiple_deployments_for_same_template(self, handler, templates, mock_cb):
        """A template can be deployed multiple times."""
        for i in range(3):
            req = _req(body={"name": f"Deploy {i}"})
            await _call(
                handler,
                "/api/v1/marketplace/templates/tpl-software-1/deploy",
                method="POST",
                templates=templates,
                mock_cb=mock_cb,
                request=req,
            )
        result = await _call(
            handler, "/api/v1/marketplace/deployments", templates=templates, request=_req()
        )
        body = _body(result)
        assert body["total"] == 3

    @pytest.mark.asyncio
    async def test_multiple_ratings_update_average(self, handler, templates, mock_cb):
        """Multiple ratings correctly update the average."""
        for rating_val in [1, 3, 5]:
            req = _req(body={"rating": rating_val})
            result = await _call(
                handler,
                "/api/v1/marketplace/templates/tpl-software-1/rate",
                method="POST",
                templates=templates,
                mock_cb=mock_cb,
                request=req,
            )
        body = _body(result)
        # Average of 1, 3, 5 = 3.0
        assert body["template_rating"]["average"] == 3.0
        assert body["template_rating"]["count"] == 3

    @pytest.mark.asyncio
    async def test_tenant_isolation_deployments(self, handler, templates, mock_cb):
        """Deployments are isolated by tenant."""
        # Deploy as tenant-1
        req1 = MockRequest(body={"name": "T1 deploy"}, tenant_id="tenant-1")
        await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-software-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req1,
        )
        # Deploy as tenant-2
        req2 = MockRequest(body={"name": "T2 deploy"}, tenant_id="tenant-2")
        await _call(
            handler,
            "/api/v1/marketplace/templates/tpl-legal-1/deploy",
            method="POST",
            templates=templates,
            mock_cb=mock_cb,
            request=req2,
        )
        # List for tenant-1
        result = await _call(
            handler,
            "/api/v1/marketplace/deployments",
            templates=templates,
            request=MockRequest(tenant_id="tenant-1"),
        )
        body = _body(result)
        assert body["total"] == 1
        assert body["deployments"][0]["name"] == "T1 deploy"


# ===========================================================================
# Module-level Function Tests
# ===========================================================================


class TestModuleFunctions:
    """Tests for module-level helper functions."""

    def test_get_marketplace_handler(self):
        with patch(
            "aragora.server.handlers.features.marketplace.handler._load_templates_proxy",
            return_value={},
        ):
            from aragora.server.handlers.features.marketplace.handler import (
                get_marketplace_handler,
            )

            h = get_marketplace_handler()
            assert isinstance(h, MarketplaceHandler)

    @pytest.mark.asyncio
    async def test_handle_marketplace_delegates(self, templates):
        """handle_marketplace creates a handler and delegates to it."""
        from aragora.server.handlers.features.marketplace.handler import handle_marketplace

        req = _req()
        mock_cb = MockCircuitBreaker(allowed=True)

        async def _parse_body(r, *, context=""):
            return {}, None

        with patch(
            "aragora.server.handlers.features.marketplace.handler._load_templates_proxy",
            return_value=templates,
        ), patch(
            "aragora.server.handlers.features.marketplace.handler._get_marketplace_circuit_breaker_proxy",
            return_value=mock_cb,
        ), patch(
            "aragora.server.handlers.features.marketplace.handler.get_marketplace_circuit_breaker_status",
            return_value=mock_cb.get_status(),
        ), patch(
            "aragora.server.handlers.features.marketplace.handler._parse_json_body_proxy",
            side_effect=_parse_body,
        ):
            result = await handle_marketplace(req, "/api/v1/marketplace/categories", "GET")
        assert _status(result) == 200
