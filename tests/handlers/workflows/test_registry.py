"""Tests for TemplateRegistryHandler (aragora/server/handlers/workflows/registry.py).

Covers all routes and behavior of the TemplateRegistryHandler class:
- can_handle() routing
- GET    /api/v1/templates/registry              - Search/browse templates
- GET    /api/v1/templates/registry/{id}          - Get template by ID
- GET    /api/v1/templates/registry/{id}/analytics - Get template analytics
- POST   /api/v1/templates/registry              - Submit new template
- POST   /api/v1/templates/registry/{id}/install  - Install a template
- POST   /api/v1/templates/registry/{id}/rate     - Rate a template
- POST   /api/v1/templates/registry/{id}/approve  - Approve a template (admin)
- POST   /api/v1/templates/registry/{id}/reject   - Reject a template (admin)
- _extract_listing_id helper
- Error handling for ValueError, edge cases, missing fields
- ROUTES constant
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.workflows.registry import TemplateRegistryHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to TemplateRegistryHandler."""

    def __init__(
        self,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        content_type: str = "application/json",
    ):
        self.command = method
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers = {
                "Content-Length": str(len(raw)),
                "Content-Type": content_type,
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {
                "Content-Length": "2",
                "Content-Type": content_type,
            }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PATCH_MOD = "aragora.server.handlers.workflows.registry"


@pytest.fixture
def handler():
    """Create a TemplateRegistryHandler with default context."""
    return TemplateRegistryHandler(ctx={})


@pytest.fixture
def mock_http():
    """Factory for creating mock HTTP handlers."""

    def _create(method="GET", body=None, content_type="application/json"):
        return _MockHTTPHandler(method=method, body=body, content_type=content_type)

    return _create


def _make_listing(
    listing_id="lst-001",
    name="Test Template",
    description="A test template",
    category="general",
    author_id="user-1",
    status="approved",
):
    """Create a mock RegistryListing with to_dict()."""
    listing = MagicMock()
    listing.id = listing_id
    listing.name = name
    listing.description = description
    listing.category = category
    listing.author_id = author_id
    listing.status = status
    listing.to_dict.return_value = {
        "id": listing_id,
        "name": name,
        "description": description,
        "category": category,
        "author_id": author_id,
        "status": status,
    }
    return listing


def _make_mock_registry(
    search_results=None,
    get_result=None,
    analytics_result=None,
    submit_result="new-id",
    install_result=None,
    install_error=None,
    rate_error=None,
    approve_result=True,
    reject_result=True,
):
    """Create a mock TemplateRegistry."""
    registry = MagicMock()
    registry.search.return_value = search_results or []
    registry.get.return_value = get_result
    registry.get_analytics.return_value = analytics_result or {}
    registry.submit.return_value = submit_result
    if install_error:
        registry.install.side_effect = install_error
    else:
        registry.install.return_value = install_result or {}
    if rate_error:
        registry.rate.side_effect = rate_error
    registry.approve.return_value = approve_result
    registry.reject.return_value = reject_result
    return registry


# ---------------------------------------------------------------------------
# ROUTES constant
# ---------------------------------------------------------------------------


class TestRoutes:
    """Test the ROUTES constant has the expected entries."""

    def test_routes_include_registry(self):
        assert "/api/v1/templates/registry" in TemplateRegistryHandler.ROUTES

    def test_routes_include_registry_wildcard(self):
        assert "/api/v1/templates/registry/*" in TemplateRegistryHandler.ROUTES


# ---------------------------------------------------------------------------
# can_handle
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Test TemplateRegistryHandler.can_handle() routing."""

    def test_registry_root(self, handler):
        assert handler.can_handle("/api/v1/templates/registry") is True

    def test_registry_with_id(self, handler):
        assert handler.can_handle("/api/v1/templates/registry/lst-001") is True

    def test_registry_with_action(self, handler):
        assert handler.can_handle("/api/v1/templates/registry/lst-001/install") is True

    def test_registry_analytics(self, handler):
        assert handler.can_handle("/api/v1/templates/registry/lst-001/analytics") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/workflows") is False

    def test_templates_without_registry(self, handler):
        assert handler.can_handle("/api/v1/templates") is False

    def test_partial_match(self, handler):
        assert handler.can_handle("/api/v1/templates/registr") is False


# ---------------------------------------------------------------------------
# _extract_listing_id
# ---------------------------------------------------------------------------


class TestExtractListingId:
    """Test TemplateRegistryHandler._extract_listing_id helper."""

    def test_basic_id_extraction(self, handler):
        result = handler._extract_listing_id("/api/v1/templates/registry/lst-001")
        assert result == "lst-001"

    def test_id_with_action_suffix(self, handler):
        result = handler._extract_listing_id("/api/v1/templates/registry/lst-001/install")
        assert result == "lst-001"

    def test_no_id_returns_none(self, handler):
        result = handler._extract_listing_id("/api/v1/templates/registry")
        assert result is None

    def test_too_few_parts_returns_none(self, handler):
        result = handler._extract_listing_id("/api/v1/templates")
        assert result is None

    def test_different_prefix_with_registry_segment(self, handler):
        """_extract_listing_id only checks parts[3] == 'registry', not the full prefix."""
        result = handler._extract_listing_id("/api/v1/workflows/registry/lst-001")
        # The method does not validate parts[2], so this still extracts the ID
        assert result == "lst-001"

    def test_wrong_segment_not_registry_returns_none(self, handler):
        """If parts[3] is not 'registry', returns None."""
        result = handler._extract_listing_id("/api/v1/templates/other/lst-001")
        assert result is None

    def test_uuid_id(self, handler):
        result = handler._extract_listing_id(
            "/api/v1/templates/registry/550e8400-e29b-41d4-a716-446655440000"
        )
        assert result == "550e8400-e29b-41d4-a716-446655440000"

    def test_trailing_slash_still_returns_id(self, handler):
        # strip("/") removes the trailing slash, so the id is in parts[4]
        result = handler._extract_listing_id("/api/v1/templates/registry/lst-001/")
        assert result == "lst-001"


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    """Test TemplateRegistryHandler initialization."""

    def test_default_ctx(self):
        h = TemplateRegistryHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        ctx = {"key": "value"}
        h = TemplateRegistryHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx(self):
        h = TemplateRegistryHandler(ctx=None)
        assert h.ctx == {}


# ---------------------------------------------------------------------------
# GET /api/v1/templates/registry (search)
# ---------------------------------------------------------------------------


class TestSearch:
    """Test GET /api/v1/templates/registry."""

    def test_returns_empty_list(self, handler, mock_http):
        registry = _make_mock_registry(search_results=[])
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle("/api/v1/templates/registry", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["templates"] == []
        assert body["count"] == 0

    def test_returns_templates(self, handler, mock_http):
        lst1 = _make_listing("lst-001", "Template 1")
        lst2 = _make_listing("lst-002", "Template 2")
        registry = _make_mock_registry(search_results=[lst1, lst2])
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle("/api/v1/templates/registry", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 2
        assert body["templates"][0]["id"] == "lst-001"
        assert body["templates"][1]["id"] == "lst-002"

    def test_passes_query_param(self, handler, mock_http):
        registry = _make_mock_registry(search_results=[])
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle(
                "/api/v1/templates/registry",
                {"query": "contract"},
                mock_http(),
            )
        call_kwargs = registry.search.call_args[1]
        assert call_kwargs["query"] == "contract"

    def test_passes_category_param(self, handler, mock_http):
        registry = _make_mock_registry(search_results=[])
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle(
                "/api/v1/templates/registry",
                {"category": "legal"},
                mock_http(),
            )
        call_kwargs = registry.search.call_args[1]
        assert call_kwargs["category"] == "legal"

    def test_passes_status_param(self, handler, mock_http):
        registry = _make_mock_registry(search_results=[])
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle(
                "/api/v1/templates/registry",
                {"status": "pending"},
                mock_http(),
            )
        call_kwargs = registry.search.call_args[1]
        assert call_kwargs["status"] == "pending"

    def test_passes_limit_and_offset(self, handler, mock_http):
        registry = _make_mock_registry(search_results=[])
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle(
                "/api/v1/templates/registry",
                {"limit": "10", "offset": "5"},
                mock_http(),
            )
        call_kwargs = registry.search.call_args[1]
        assert call_kwargs["limit"] == 10
        assert call_kwargs["offset"] == 5

    def test_passes_tags_param_split_by_comma(self, handler, mock_http):
        registry = _make_mock_registry(search_results=[])
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle(
                "/api/v1/templates/registry",
                {"tags": "legal,review,compliance"},
                mock_http(),
            )
        call_kwargs = registry.search.call_args[1]
        assert call_kwargs["tags"] == ["legal", "review", "compliance"]

    def test_no_tags_passes_none(self, handler, mock_http):
        registry = _make_mock_registry(search_results=[])
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle(
                "/api/v1/templates/registry",
                {},
                mock_http(),
            )
        call_kwargs = registry.search.call_args[1]
        assert call_kwargs["tags"] is None

    def test_default_limit_is_20(self, handler, mock_http):
        registry = _make_mock_registry(search_results=[])
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle("/api/v1/templates/registry", {}, mock_http())
        call_kwargs = registry.search.call_args[1]
        assert call_kwargs["limit"] == 20

    def test_default_offset_is_0(self, handler, mock_http):
        registry = _make_mock_registry(search_results=[])
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle("/api/v1/templates/registry", {}, mock_http())
        call_kwargs = registry.search.call_args[1]
        assert call_kwargs["offset"] == 0

    def test_empty_query_passes_empty_string(self, handler, mock_http):
        registry = _make_mock_registry(search_results=[])
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle("/api/v1/templates/registry", {}, mock_http())
        call_kwargs = registry.search.call_args[1]
        assert call_kwargs["query"] == ""

    def test_unhandled_path_returns_none(self, handler, mock_http):
        result = handler.handle("/api/v1/workflows", {}, mock_http())
        assert result is None


# ---------------------------------------------------------------------------
# GET /api/v1/templates/registry/{id}
# ---------------------------------------------------------------------------


class TestGetListing:
    """Test GET /api/v1/templates/registry/{id}."""

    def test_returns_listing(self, handler, mock_http):
        listing = _make_listing("lst-001", "My Template")
        registry = _make_mock_registry(get_result=listing)
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle("/api/v1/templates/registry/lst-001", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "lst-001"
        assert body["name"] == "My Template"

    def test_not_found_returns_404(self, handler, mock_http):
        registry = _make_mock_registry(get_result=None)
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle("/api/v1/templates/registry/nonexistent", {}, mock_http())
        assert _status(result) == 404

    def test_not_found_message_contains_id(self, handler, mock_http):
        registry = _make_mock_registry(get_result=None)
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle("/api/v1/templates/registry/missing-123", {}, mock_http())
        body = _body(result)
        assert "missing-123" in body.get("error", "")

    def test_calls_registry_get(self, handler, mock_http):
        listing = _make_listing()
        registry = _make_mock_registry(get_result=listing)
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle("/api/v1/templates/registry/lst-001", {}, mock_http())
        registry.get.assert_called_once_with("lst-001")


# ---------------------------------------------------------------------------
# GET /api/v1/templates/registry/{id}/analytics
# ---------------------------------------------------------------------------


class TestGetAnalytics:
    """Test GET /api/v1/templates/registry/{id}/analytics."""

    def test_returns_analytics(self, handler, mock_http):
        listing = _make_listing("lst-001")
        analytics = {
            "listing_id": "lst-001",
            "install_count": 42,
            "rating_average": 4.5,
            "rating_count": 10,
            "recent_installs": [],
        }
        registry = _make_mock_registry(get_result=listing, analytics_result=analytics)
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle("/api/v1/templates/registry/lst-001/analytics", {}, mock_http())
        assert _status(result) == 200
        body = _body(result)
        assert body["install_count"] == 42
        assert body["rating_average"] == 4.5

    def test_not_found_returns_404(self, handler, mock_http):
        registry = _make_mock_registry(get_result=None)
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle(
                "/api/v1/templates/registry/nonexistent/analytics", {}, mock_http()
            )
        assert _status(result) == 404

    def test_calls_get_analytics(self, handler, mock_http):
        listing = _make_listing("lst-001")
        registry = _make_mock_registry(get_result=listing)
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle("/api/v1/templates/registry/lst-001/analytics", {}, mock_http())
        registry.get_analytics.assert_called_once_with("lst-001")


# ---------------------------------------------------------------------------
# GET routing edge cases
# ---------------------------------------------------------------------------


class TestGetRoutingEdgeCases:
    """Test GET handler routing edge cases."""

    def test_action_path_install_returns_none(self, handler, mock_http):
        """GET on /install should return None (POST-only action)."""
        result = handler.handle("/api/v1/templates/registry/lst-001/install", {}, mock_http())
        assert result is None

    def test_action_path_rate_returns_none(self, handler, mock_http):
        """GET on /rate should return None (POST-only action)."""
        result = handler.handle("/api/v1/templates/registry/lst-001/rate", {}, mock_http())
        assert result is None

    def test_action_path_approve_returns_none(self, handler, mock_http):
        """GET on /approve should return None (POST-only action)."""
        result = handler.handle("/api/v1/templates/registry/lst-001/approve", {}, mock_http())
        assert result is None

    def test_action_path_reject_returns_none(self, handler, mock_http):
        """GET on /reject should return None (POST-only action)."""
        result = handler.handle("/api/v1/templates/registry/lst-001/reject", {}, mock_http())
        assert result is None

    def test_path_ending_with_registry_routes_to_search(self, handler, mock_http):
        """Path ending with /registry routes to search, not get."""
        registry = _make_mock_registry(search_results=[])
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle("/api/v1/templates/registry", {}, mock_http())
        assert _status(result) == 200
        registry.search.assert_called_once()


# ---------------------------------------------------------------------------
# POST /api/v1/templates/registry (submit)
# ---------------------------------------------------------------------------


class TestSubmitTemplate:
    """Test POST /api/v1/templates/registry."""

    def test_submits_template(self, handler, mock_http):
        body = {
            "name": "My New Template",
            "description": "A great template",
            "category": "legal",
            "tags": ["legal", "review"],
            "template_data": {"steps": []},
            "author_id": "user-1",
            "version": "2.0.0",
        }
        registry = _make_mock_registry(submit_result="new-listing-id")
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 201
        resp = _body(result)
        assert resp["id"] == "new-listing-id"
        assert resp["status"] == "pending"

    def test_submit_passes_all_fields(self, handler, mock_http):
        body = {
            "name": "Template X",
            "description": "Desc",
            "category": "code",
            "tags": ["code", "security"],
            "template_data": {"key": "value"},
            "author_id": "author-42",
            "version": "3.1.0",
        }
        registry = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry",
                {},
                mock_http(method="POST", body=body),
            )
        call_kwargs = registry.submit.call_args[1]
        assert call_kwargs["name"] == "Template X"
        assert call_kwargs["description"] == "Desc"
        assert call_kwargs["category"] == "code"
        assert call_kwargs["tags"] == ["code", "security"]
        assert call_kwargs["template_data"] == {"key": "value"}
        assert call_kwargs["author_id"] == "author-42"
        assert call_kwargs["version"] == "3.1.0"

    def test_submit_missing_name_returns_400(self, handler, mock_http):
        body = {"description": "No name here"}
        result = handler.handle_post(
            "/api/v1/templates/registry",
            {},
            mock_http(method="POST", body=body),
        )
        assert _status(result) == 400
        assert "name" in _body(result).get("error", "").lower()

    def test_submit_empty_name_returns_400(self, handler, mock_http):
        body = {"name": ""}
        result = handler.handle_post(
            "/api/v1/templates/registry",
            {},
            mock_http(method="POST", body=body),
        )
        assert _status(result) == 400

    def test_submit_defaults(self, handler, mock_http):
        """Submit with only name, verify defaults are applied."""
        body = {"name": "Minimal Template"}
        registry = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 201
        call_kwargs = registry.submit.call_args[1]
        assert call_kwargs["description"] == ""
        assert call_kwargs["category"] == "general"
        assert call_kwargs["tags"] == []
        assert call_kwargs["template_data"] == {}
        assert call_kwargs["version"] == "1.0.0"

    def test_submit_author_from_query_params(self, handler, mock_http):
        """author_id falls back to user_id query param if not in body."""
        body = {"name": "Template"}
        registry = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry",
                {"user_id": "query-user"},
                mock_http(method="POST", body=body),
            )
        call_kwargs = registry.submit.call_args[1]
        assert call_kwargs["author_id"] == "query-user"

    def test_submit_author_from_body_takes_precedence(self, handler, mock_http):
        """author_id in body takes precedence over query param."""
        body = {"name": "Template", "author_id": "body-author"}
        registry = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry",
                {"user_id": "query-user"},
                mock_http(method="POST", body=body),
            )
        call_kwargs = registry.submit.call_args[1]
        assert call_kwargs["author_id"] == "body-author"

    def test_submit_default_author_is_anonymous(self, handler, mock_http):
        """No author_id in body or query defaults to anonymous."""
        body = {"name": "Template"}
        registry = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry",
                {},
                mock_http(method="POST", body=body),
            )
        call_kwargs = registry.submit.call_args[1]
        assert call_kwargs["author_id"] == "anonymous"

    def test_unhandled_path_returns_none(self, handler, mock_http):
        result = handler.handle_post("/api/v1/unrelated", {}, mock_http(method="POST", body={}))
        assert result is None


# ---------------------------------------------------------------------------
# POST /api/v1/templates/registry/{id}/install
# ---------------------------------------------------------------------------


class TestInstallTemplate:
    """Test POST /api/v1/templates/registry/{id}/install."""

    def test_installs_template(self, handler, mock_http):
        body = {"user_id": "user-1"}
        template_data = {"steps": [{"id": "s1"}]}
        registry = _make_mock_registry(install_result=template_data)
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/lst-001/install",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 200
        resp = _body(result)
        assert resp["installed"] is True
        assert resp["template_data"] == template_data

    def test_install_passes_user_id_from_body(self, handler, mock_http):
        body = {"user_id": "user-42"}
        registry = _make_mock_registry(install_result={})
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry/lst-001/install",
                {},
                mock_http(method="POST", body=body),
            )
        registry.install.assert_called_once_with("lst-001", user_id="user-42")

    def test_install_user_id_from_query_param(self, handler, mock_http):
        body = {}
        registry = _make_mock_registry(install_result={})
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry/lst-001/install",
                {"user_id": "query-user"},
                mock_http(method="POST", body=body),
            )
        registry.install.assert_called_once_with("lst-001", user_id="query-user")

    def test_install_not_found_returns_404(self, handler, mock_http):
        body = {}
        registry = _make_mock_registry(install_error=ValueError("Template not found"))
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/nonexistent/install",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 404

    def test_install_not_approved_returns_404(self, handler, mock_http):
        body = {}
        registry = _make_mock_registry(install_error=ValueError("Template is not approved"))
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/lst-pending/install",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 404

    def test_install_default_user_id_empty(self, handler, mock_http):
        """Without user_id in body or query, defaults to empty string."""
        body = {}
        registry = _make_mock_registry(install_result={})
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry/lst-001/install",
                {},
                mock_http(method="POST", body=body),
            )
        registry.install.assert_called_once_with("lst-001", user_id="")


# ---------------------------------------------------------------------------
# POST /api/v1/templates/registry/{id}/rate
# ---------------------------------------------------------------------------


class TestRateTemplate:
    """Test POST /api/v1/templates/registry/{id}/rate."""

    def test_rates_template(self, handler, mock_http):
        body = {"rating": 5, "review": "Excellent!", "user_id": "user-1"}
        registry = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/lst-001/rate",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 200
        resp = _body(result)
        assert resp["rated"] is True
        assert resp["listing_id"] == "lst-001"
        assert resp["rating"] == 5

    def test_rate_passes_all_fields(self, handler, mock_http):
        body = {"rating": 3, "review": "Decent", "user_id": "reviewer-1"}
        registry = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry/lst-001/rate",
                {},
                mock_http(method="POST", body=body),
            )
        registry.rate.assert_called_once_with(
            "lst-001", user_id="reviewer-1", rating=3, review="Decent"
        )

    def test_rate_missing_rating_returns_400(self, handler, mock_http):
        body = {"review": "No rating"}
        result = handler.handle_post(
            "/api/v1/templates/registry/lst-001/rate",
            {},
            mock_http(method="POST", body=body),
        )
        assert _status(result) == 400
        assert "rating" in _body(result).get("error", "").lower()

    def test_rate_non_integer_rating_returns_400(self, handler, mock_http):
        body = {"rating": "not-a-number"}
        result = handler.handle_post(
            "/api/v1/templates/registry/lst-001/rate",
            {},
            mock_http(method="POST", body=body),
        )
        assert _status(result) == 400
        assert "integer" in _body(result).get("error", "").lower()

    def test_rate_float_string_returns_400(self, handler, mock_http):
        body = {"rating": "4.5"}
        result = handler.handle_post(
            "/api/v1/templates/registry/lst-001/rate",
            {},
            mock_http(method="POST", body=body),
        )
        assert _status(result) == 400

    def test_rate_none_rating_returns_400(self, handler, mock_http):
        body = {"rating": None}
        result = handler.handle_post(
            "/api/v1/templates/registry/lst-001/rate",
            {},
            mock_http(method="POST", body=body),
        )
        assert _status(result) == 400

    def test_rate_out_of_range_returns_400(self, handler, mock_http):
        """Rating out of 1-5 range is caught by registry.rate() ValueError."""
        body = {"rating": 0}
        registry = _make_mock_registry(rate_error=ValueError("Rating must be between 1 and 5"))
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/lst-001/rate",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 400

    def test_rate_high_value_returns_400(self, handler, mock_http):
        body = {"rating": 6}
        registry = _make_mock_registry(rate_error=ValueError("Rating must be between 1 and 5"))
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/lst-001/rate",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 400

    def test_rate_user_id_from_query(self, handler, mock_http):
        body = {"rating": 4}
        registry = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry/lst-001/rate",
                {"user_id": "query-rater"},
                mock_http(method="POST", body=body),
            )
        call_kwargs = registry.rate.call_args
        assert call_kwargs[1]["user_id"] == "query-rater"

    def test_rate_without_review(self, handler, mock_http):
        """Review is optional."""
        body = {"rating": 5, "user_id": "u1"}
        registry = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry/lst-001/rate",
                {},
                mock_http(method="POST", body=body),
            )
        call_kwargs = registry.rate.call_args[1]
        assert call_kwargs["review"] is None

    def test_rate_integer_as_string_is_accepted(self, handler, mock_http):
        """String like '4' is accepted because int('4') works."""
        body = {"rating": "4"}
        registry = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/lst-001/rate",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 200
        resp = _body(result)
        assert resp["rating"] == 4


# ---------------------------------------------------------------------------
# POST /api/v1/templates/registry/{id}/approve
# ---------------------------------------------------------------------------


class TestApproveTemplate:
    """Test POST /api/v1/templates/registry/{id}/approve."""

    def test_approves_template(self, handler, mock_http):
        body = {"approved_by": "admin-1"}
        registry = _make_mock_registry(approve_result=True)
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/lst-001/approve",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 200
        resp = _body(result)
        assert resp["approved"] is True
        assert resp["listing_id"] == "lst-001"

    def test_approve_passes_approved_by(self, handler, mock_http):
        body = {"approved_by": "admin-42"}
        registry = _make_mock_registry(approve_result=True)
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry/lst-001/approve",
                {},
                mock_http(method="POST", body=body),
            )
        registry.approve.assert_called_once_with("lst-001", approved_by="admin-42")

    def test_approve_not_found_returns_404(self, handler, mock_http):
        body = {"approved_by": "admin-1"}
        registry = _make_mock_registry(approve_result=False)
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/nonexistent/approve",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 404

    def test_approve_user_id_from_query(self, handler, mock_http):
        body = {}
        registry = _make_mock_registry(approve_result=True)
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry/lst-001/approve",
                {"user_id": "query-admin"},
                mock_http(method="POST", body=body),
            )
        registry.approve.assert_called_once_with("lst-001", approved_by="query-admin")

    def test_approve_default_empty_approved_by(self, handler, mock_http):
        body = {}
        registry = _make_mock_registry(approve_result=True)
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry/lst-001/approve",
                {},
                mock_http(method="POST", body=body),
            )
        registry.approve.assert_called_once_with("lst-001", approved_by="")


# ---------------------------------------------------------------------------
# POST /api/v1/templates/registry/{id}/reject
# ---------------------------------------------------------------------------


class TestRejectTemplate:
    """Test POST /api/v1/templates/registry/{id}/reject."""

    def test_rejects_template(self, handler, mock_http):
        body = {"reason": "Low quality"}
        registry = _make_mock_registry(reject_result=True)
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/lst-001/reject",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 200
        resp = _body(result)
        assert resp["rejected"] is True
        assert resp["listing_id"] == "lst-001"

    def test_reject_passes_reason(self, handler, mock_http):
        body = {"reason": "Insufficient documentation"}
        registry = _make_mock_registry(reject_result=True)
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry/lst-001/reject",
                {},
                mock_http(method="POST", body=body),
            )
        registry.reject.assert_called_once_with("lst-001", reason="Insufficient documentation")

    def test_reject_not_found_returns_404(self, handler, mock_http):
        body = {}
        registry = _make_mock_registry(reject_result=False)
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/nonexistent/reject",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 404

    def test_reject_without_reason(self, handler, mock_http):
        """Reason is optional."""
        body = {}
        registry = _make_mock_registry(reject_result=True)
        with patch.object(handler, "_get_registry", return_value=registry):
            handler.handle_post(
                "/api/v1/templates/registry/lst-001/reject",
                {},
                mock_http(method="POST", body=body),
            )
        registry.reject.assert_called_once_with("lst-001", reason=None)

    def test_reject_404_message_includes_id(self, handler, mock_http):
        body = {}
        registry = _make_mock_registry(reject_result=False)
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/missing-id/reject",
                {},
                mock_http(method="POST", body=body),
            )
        body_resp = _body(result)
        assert "missing-id" in body_resp.get("error", "")


# ---------------------------------------------------------------------------
# POST routing edge cases
# ---------------------------------------------------------------------------


class TestPostRoutingEdgeCases:
    """Test POST handler routing edge cases."""

    def test_post_unhandled_path_returns_none(self, handler, mock_http):
        result = handler.handle_post("/api/v1/workflows", {}, mock_http(method="POST", body={}))
        assert result is None

    def test_post_with_listing_id_but_no_action_returns_none(self, handler, mock_http):
        """POST to /api/v1/templates/registry/{id} with no action suffix returns None."""
        body = {"name": "Ignored"}
        registry = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/lst-001",
                {},
                mock_http(method="POST", body=body),
            )
        assert result is None

    def test_post_unknown_action_returns_none(self, handler, mock_http):
        """POST to unknown action path with a listing ID returns None."""
        body = {}
        registry = _make_mock_registry()
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/lst-001/unknown",
                {},
                mock_http(method="POST", body=body),
            )
        assert result is None


# ---------------------------------------------------------------------------
# POST error handling via @handle_errors decorator
# ---------------------------------------------------------------------------


class TestPostErrorHandling:
    """Test that @handle_errors decorator catches exceptions on POST endpoints."""

    def test_runtime_error_returns_500(self, handler, mock_http):
        """Unrecognized exception should be caught by @handle_errors."""
        body = {"name": "Template"}
        registry = MagicMock()
        registry.submit.side_effect = RuntimeError("unexpected")
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 500

    def test_os_error_returns_500(self, handler, mock_http):
        body = {"name": "Template"}
        registry = MagicMock()
        registry.submit.side_effect = OSError("disk failure")
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 500

    def test_value_error_returns_400(self, handler, mock_http):
        body = {"name": "Template"}
        registry = MagicMock()
        registry.submit.side_effect = ValueError("bad value")
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry",
                {},
                mock_http(method="POST", body=body),
            )
        assert _status(result) == 400

    def test_install_os_error_caught(self, handler, mock_http):
        body = {}
        registry = MagicMock()
        registry.install.side_effect = OSError("db down")
        with patch.object(handler, "_get_registry", return_value=registry):
            result = handler.handle_post(
                "/api/v1/templates/registry/lst-001/install",
                {},
                mock_http(method="POST", body=body),
            )
        # OSError maps to 500 in the exception map
        assert _status(result) == 500


# ---------------------------------------------------------------------------
# POST body validation
# ---------------------------------------------------------------------------


class TestPostBodyValidation:
    """Test POST body validation edge cases."""

    def test_invalid_json_body_returns_error(self, handler, mock_http):
        """Non-JSON body should return an error from read_json_body_validated."""
        mock = mock_http(method="POST", content_type="text/plain")
        mock.rfile.read.return_value = b"not json"
        mock.headers = {"Content-Length": "8", "Content-Type": "text/plain"}
        result = handler.handle_post(
            "/api/v1/templates/registry",
            {},
            mock,
        )
        # Should get a 400 or 415 for invalid content type
        assert _status(result) in (400, 415)

    def test_empty_body_for_submit_returns_400(self, handler, mock_http):
        """Empty body {} has no name field, should return 400."""
        result = handler.handle_post(
            "/api/v1/templates/registry",
            {},
            mock_http(method="POST", body={}),
        )
        assert _status(result) == 400
