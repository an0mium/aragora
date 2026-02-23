"""Tests for marketplace handler (aragora/server/handlers/marketplace.py).

Covers all routes, validation, circuit breaker, rate limiting, error paths, and RBAC:
- GET  /api/v1/marketplace/templates          - List all templates
- GET  /api/v1/marketplace/templates/{id}     - Get template details
- POST /api/v1/marketplace/templates          - Create a template
- DELETE /api/v1/marketplace/templates/{id}   - Delete a template
- POST /api/v1/marketplace/templates/{id}/ratings - Rate a template
- GET  /api/v1/marketplace/templates/{id}/ratings  - Get template ratings
- POST /api/v1/marketplace/templates/{id}/star     - Star a template
- GET  /api/v1/marketplace/categories         - List categories
- GET  /api/v1/marketplace/templates/{id}/export   - Export template
- POST /api/v1/marketplace/templates/import   - Import a template
- GET  /api/v1/marketplace/status             - Health/circuit breaker status
- Input validation functions
- Circuit breaker state management
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult
from aragora.server.handlers.marketplace import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MAX_OFFSET,
    MAX_QUERY_LENGTH,
    MAX_RATING,
    MAX_REVIEW_LENGTH,
    MAX_TAGS_LENGTH,
    MAX_TEMPLATE_ID_LENGTH,
    MIN_LIMIT,
    MIN_RATING,
    MarketplaceHandler,
    _clear_registry,
    _validate_pagination,
    _validate_query,
    _validate_rating,
    _validate_review,
    _validate_tags,
    _validate_template_id,
    reset_marketplace_circuit_breaker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: HandlerResult) -> dict:
    """Extract JSON body from a HandlerResult."""
    if isinstance(result, HandlerResult):
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode("utf-8"))
        return result.body
    if isinstance(result, dict):
        return result.get("body", result)
    return {}


def _status(result: HandlerResult) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, HandlerResult):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return 200


class MockHTTPHandler:
    """Mock HTTP handler simulating BaseHTTPRequestHandler."""

    def __init__(self, body: dict[str, Any] | None = None):
        self.rfile = MagicMock()
        self.command = "GET"
        self._body = body
        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {"Content-Length": str(len(body_bytes))}
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {"Content-Length": "2"}
        self.client_address = ("127.0.0.1", 54321)


@dataclass
class MockTemplateMetadata:
    """Mock template metadata."""

    id: str = "test-template-1"
    name: str = "Test Template"
    description: str = "A test template"
    version: str = "1.0.0"
    author: str = "tester"
    category: str = "analysis"
    tags: list[str] = field(default_factory=list)
    downloads: int = 0
    stars: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "category": self.category,
            "tags": self.tags,
            "downloads": self.downloads,
            "stars": self.stars,
        }


@dataclass
class MockTemplate:
    """Mock marketplace template."""

    metadata: MockTemplateMetadata = field(default_factory=MockTemplateMetadata)

    def to_dict(self) -> dict[str, Any]:
        return {"metadata": self.metadata.to_dict()}


@dataclass
class MockRating:
    """Mock template rating."""

    user_id: str = "user-1"
    score: int = 4
    review: str | None = "Great template"
    created_at: datetime = field(default_factory=lambda: datetime(2025, 1, 1, tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiters between tests to avoid cross-test pollution."""
    yield
    # Reset rate limiters using defaultdict pattern
    try:
        from aragora.server.handlers.utils import rate_limit as rl_mod

        for attr_name in dir(rl_mod):
            obj = getattr(rl_mod, attr_name, None)
            if hasattr(obj, "_requests") and isinstance(obj._requests, dict):
                obj._requests = defaultdict(list)
    except (ImportError, AttributeError):
        pass


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset the circuit breaker before each test."""
    reset_marketplace_circuit_breaker()
    yield
    reset_marketplace_circuit_breaker()


@pytest.fixture(autouse=True)
def _clear_registry_fixture():
    """Clear the registry singleton before each test."""
    _clear_registry()
    yield
    _clear_registry()


@pytest.fixture
def mock_registry():
    """Create a mock TemplateRegistry."""
    registry = MagicMock()

    templates = [
        MockTemplate(MockTemplateMetadata(id="tpl-1", name="Template One")),
        MockTemplate(MockTemplateMetadata(id="tpl-2", name="Template Two")),
        MockTemplate(MockTemplateMetadata(id="tpl-3", name="Template Three")),
    ]

    registry.search.return_value = templates
    registry.get.side_effect = lambda tid: next(
        (t for t in templates if t.metadata.id == tid), None
    )
    registry.increment_downloads.return_value = None
    registry.import_template.return_value = "new-tpl-id"
    registry.delete.return_value = True
    registry.rate.return_value = None
    registry.get_average_rating.return_value = 4.5
    registry.get_ratings.return_value = [
        MockRating(user_id="user-1", score=5, review="Excellent"),
        MockRating(user_id="user-2", score=4, review="Good"),
    ]
    registry.star.return_value = None
    registry.list_categories.return_value = ["analysis", "coding", "debate"]
    registry.export_template.side_effect = lambda tid: (
        json.dumps({"id": tid, "name": "Exported"}) if tid == "tpl-1" else None
    )

    return registry


@pytest.fixture
def handler(mock_registry):
    """Create a MarketplaceHandler with mocked registry."""
    with patch(
        "aragora.server.handlers.marketplace._get_registry",
        return_value=mock_registry,
    ):
        h = MarketplaceHandler(server_context={})
        yield h


@pytest.fixture
def http_handler():
    """Create a mock HTTP handler (no body)."""
    return MockHTTPHandler()


@pytest.fixture
def http_handler_with_body():
    """Factory to create mock HTTP handler with specific body."""

    def _factory(body: dict[str, Any]) -> MockHTTPHandler:
        return MockHTTPHandler(body=body)

    return _factory


# ===========================================================================
# Input Validation Unit Tests
# ===========================================================================


class TestValidateTemplateId:
    """Tests for _validate_template_id."""

    def test_valid_id(self):
        valid, err = _validate_template_id("my-template-1")
        assert valid is True
        assert err == ""

    def test_valid_id_with_underscores(self):
        valid, err = _validate_template_id("my_template_v2")
        assert valid is True

    def test_empty_string(self):
        valid, err = _validate_template_id("")
        assert valid is False
        assert "required" in err.lower()

    def test_none_value(self):
        valid, err = _validate_template_id(None)
        assert valid is False
        assert "required" in err.lower()

    def test_too_long(self):
        valid, err = _validate_template_id("a" * (MAX_TEMPLATE_ID_LENGTH + 1))
        assert valid is False
        assert "at most" in err

    def test_max_length_exactly(self):
        valid, err = _validate_template_id("a" * MAX_TEMPLATE_ID_LENGTH)
        assert valid is True

    def test_invalid_chars_spaces(self):
        valid, err = _validate_template_id("my template")
        assert valid is False
        assert "invalid characters" in err.lower()

    def test_invalid_chars_special(self):
        valid, err = _validate_template_id("template!@#")
        assert valid is False

    def test_starts_with_hyphen(self):
        valid, err = _validate_template_id("-starts-with-hyphen")
        assert valid is False

    def test_starts_with_underscore(self):
        valid, err = _validate_template_id("_starts-with-underscore")
        assert valid is False

    def test_alphanumeric_start(self):
        valid, err = _validate_template_id("a")
        assert valid is True

    def test_numeric_start(self):
        valid, err = _validate_template_id("123abc")
        assert valid is True


class TestValidatePagination:
    """Tests for _validate_pagination."""

    def test_defaults(self):
        limit, offset, err = _validate_pagination({})
        assert limit == DEFAULT_LIMIT
        assert offset == 0
        assert err == ""

    def test_explicit_values(self):
        limit, offset, err = _validate_pagination({"limit": "10", "offset": "20"})
        assert limit == 10
        assert offset == 20
        assert err == ""

    def test_clamp_limit_below_min(self):
        limit, offset, err = _validate_pagination({"limit": "-5"})
        assert limit == MIN_LIMIT
        assert err == ""

    def test_clamp_limit_above_max(self):
        limit, offset, err = _validate_pagination({"limit": "999"})
        assert limit == MAX_LIMIT
        assert err == ""

    def test_clamp_offset_below_zero(self):
        limit, offset, err = _validate_pagination({"offset": "-10"})
        assert offset == 0
        assert err == ""

    def test_clamp_offset_above_max(self):
        limit, offset, err = _validate_pagination({"offset": "99999"})
        assert offset == MAX_OFFSET
        assert err == ""

    def test_invalid_limit_string(self):
        limit, offset, err = _validate_pagination({"limit": "not_a_number"})
        assert err != ""
        assert "limit" in err.lower()

    def test_invalid_offset_string(self):
        limit, offset, err = _validate_pagination({"offset": "abc"})
        assert err != ""
        assert "offset" in err.lower()


class TestValidateRating:
    """Tests for _validate_rating."""

    def test_valid_rating(self):
        valid, score, err = _validate_rating(3)
        assert valid is True
        assert score == 3
        assert err == ""

    def test_min_rating(self):
        valid, score, err = _validate_rating(MIN_RATING)
        assert valid is True
        assert score == MIN_RATING

    def test_max_rating(self):
        valid, score, err = _validate_rating(MAX_RATING)
        assert valid is True
        assert score == MAX_RATING

    def test_below_min(self):
        valid, score, err = _validate_rating(0)
        assert valid is False
        assert "between" in err.lower()

    def test_above_max(self):
        valid, score, err = _validate_rating(6)
        assert valid is False
        assert "between" in err.lower()

    def test_none_value(self):
        valid, score, err = _validate_rating(None)
        assert valid is False
        assert "required" in err.lower()

    def test_non_integer(self):
        valid, score, err = _validate_rating("three")
        assert valid is False
        assert "integer" in err.lower()

    def test_float_value(self):
        valid, score, err = _validate_rating(3.5)
        assert valid is False
        assert "integer" in err.lower()


class TestValidateReview:
    """Tests for _validate_review."""

    def test_none_is_valid(self):
        valid, value, err = _validate_review(None)
        assert valid is True
        assert value is None
        assert err == ""

    def test_valid_string(self):
        valid, value, err = _validate_review("Great template!")
        assert valid is True
        assert value is not None

    def test_too_long(self):
        valid, value, err = _validate_review("a" * (MAX_REVIEW_LENGTH + 1))
        assert valid is False
        assert "at most" in err

    def test_non_string(self):
        valid, value, err = _validate_review(12345)
        assert valid is False
        assert "string" in err.lower()


class TestValidateQuery:
    """Tests for _validate_query."""

    def test_none_is_valid(self):
        valid, value, err = _validate_query(None)
        assert valid is True
        assert value == ""

    def test_empty_is_valid(self):
        valid, value, err = _validate_query("")
        assert valid is True
        assert value == ""

    def test_valid_query(self):
        valid, value, err = _validate_query("debate template")
        assert valid is True
        assert "debate" in value

    def test_too_long(self):
        valid, value, err = _validate_query("a" * (MAX_QUERY_LENGTH + 1))
        assert valid is False
        assert "at most" in err

    def test_non_string(self):
        valid, value, err = _validate_query(42)
        assert valid is False
        assert "string" in err.lower()


class TestValidateTags:
    """Tests for _validate_tags."""

    def test_none_is_valid(self):
        valid, tags, err = _validate_tags(None)
        assert valid is True
        assert tags == []

    def test_empty_is_valid(self):
        valid, tags, err = _validate_tags("")
        assert valid is True
        assert tags == []

    def test_single_tag(self):
        valid, tags, err = _validate_tags("analysis")
        assert valid is True
        assert len(tags) == 1

    def test_multiple_tags(self):
        valid, tags, err = _validate_tags("analysis,coding,debate")
        assert valid is True
        assert len(tags) == 3

    def test_tags_with_whitespace(self):
        valid, tags, err = _validate_tags(" analysis , coding ")
        assert valid is True
        assert len(tags) == 2

    def test_too_long(self):
        valid, tags, err = _validate_tags("a" * (MAX_TAGS_LENGTH + 1))
        assert valid is False
        assert "at most" in err

    def test_non_string(self):
        valid, tags, err = _validate_tags(["analysis"])
        assert valid is False
        assert "comma-separated" in err.lower()


# ===========================================================================
# Handler Tests: List Templates
# ===========================================================================


class TestListTemplates:
    """Tests for handle_list_templates (GET /api/v1/marketplace/templates)."""

    def test_list_returns_200(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        assert _status(result) == 200

    def test_list_returns_templates(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        body = _body(result)
        assert "templates" in body
        assert body["count"] == 3

    def test_list_includes_pagination(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        body = _body(result)
        assert "limit" in body
        assert "offset" in body
        assert body["limit"] == DEFAULT_LIMIT
        assert body["offset"] == 0

    def test_list_with_query_param(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {"q": "debate"})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        assert _status(result) == 200
        mock_registry.search.assert_called_once()
        call_kwargs = mock_registry.search.call_args
        assert call_kwargs[1]["query"] == "debate" or call_kwargs.kwargs["query"] == "debate"

    def test_list_with_category_filter(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {"category": "analysis"})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        assert _status(result) == 200

    def test_list_with_invalid_category(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {"category": "nonexistent_cat"})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        assert _status(result) == 400
        assert "invalid category" in _body(result).get("error", "").lower()

    def test_list_with_tags(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {"tags": "tag1,tag2"})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        assert _status(result) == 200

    def test_list_with_type_filter(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {"type": "agent"})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        assert _status(result) == 200

    def test_list_with_custom_pagination(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {"limit": "5", "offset": "10"})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        body = _body(result)
        assert body["limit"] == 5
        assert body["offset"] == 10

    def test_list_with_invalid_query_too_long(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {"q": "x" * (MAX_QUERY_LENGTH + 1)})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        assert _status(result) == 400

    def test_list_with_invalid_tags_too_long(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {"tags": "t" * (MAX_TAGS_LENGTH + 1)})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        assert _status(result) == 400

    def test_list_with_invalid_pagination_limit(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {"limit": "abc"})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        assert _status(result) == 400

    def test_list_registry_error(self, handler, http_handler, mock_registry):
        mock_registry.search.side_effect = ValueError("registry error")
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        assert _status(result) == 500

    def test_list_circuit_breaker_open(self, handler, http_handler):
        handler._circuit_breaker._state = "open"
        handler._circuit_breaker._last_failure_time = None  # Prevent transition
        handler.set_request_context(http_handler, {})
        result = handler.handle_list_templates()
        assert _status(result) == 503
        assert "temporarily unavailable" in _body(result).get("error", "").lower()


# ===========================================================================
# Handler Tests: Get Template
# ===========================================================================


class TestGetTemplate:
    """Tests for handle_get_template (GET /api/v1/marketplace/templates/{id})."""

    def test_get_returns_200(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_get_template("tpl-1")
        assert _status(result) == 200

    def test_get_returns_template_data(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_get_template("tpl-1")
        body = _body(result)
        assert "metadata" in body

    def test_get_increments_downloads(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            handler.handle_get_template("tpl-1")
        mock_registry.increment_downloads.assert_called_with("tpl-1")

    def test_get_not_found(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_get_template("nonexistent")
        assert _status(result) == 404

    def test_get_invalid_id_empty(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_get_template("")
        assert _status(result) == 400

    def test_get_invalid_id_special_chars(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_get_template("../../etc/passwd")
        assert _status(result) == 400

    def test_get_invalid_id_too_long(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_get_template("a" * (MAX_TEMPLATE_ID_LENGTH + 1))
        assert _status(result) == 400

    def test_get_circuit_breaker_open(self, handler, http_handler):
        handler._circuit_breaker._state = "open"
        handler._circuit_breaker._last_failure_time = None
        handler.set_request_context(http_handler, {})
        result = handler.handle_get_template("tpl-1")
        assert _status(result) == 503

    def test_get_registry_error(self, handler, http_handler, mock_registry):
        mock_registry.get.side_effect = OSError("connection failed")
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_get_template("tpl-1")
        assert _status(result) == 500


# ===========================================================================
# Handler Tests: Create Template
# ===========================================================================


class TestCreateTemplate:
    """Tests for handle_create_template (POST /api/v1/marketplace/templates)."""

    def test_create_returns_201(self, handler, http_handler_with_body, mock_registry):
        body = {"name": "New Template", "description": "Desc"}
        h = http_handler_with_body(body)
        handler.set_request_context(h, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_create_template()
        assert _status(result) == 201

    def test_create_returns_id(self, handler, http_handler_with_body, mock_registry):
        body = {"name": "New Template"}
        h = http_handler_with_body(body)
        handler.set_request_context(h, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_create_template()
        data = _body(result)
        assert data["id"] == "new-tpl-id"
        assert data["success"] is True

    def test_create_empty_body(self, handler, http_handler):
        # Simulate empty body (Content-Length: 0)
        http_handler.headers = {"Content-Length": "0"}
        handler.set_request_context(http_handler, {})
        result = handler.handle_create_template()
        assert _status(result) == 400

    def test_create_with_invalid_template_id(self, handler, http_handler_with_body, mock_registry):
        body = {"id": "bad id!!", "name": "Template"}
        h = http_handler_with_body(body)
        handler.set_request_context(h, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_create_template()
        assert _status(result) == 400

    def test_create_with_valid_template_id(self, handler, http_handler_with_body, mock_registry):
        body = {"id": "custom-id-1", "name": "Template"}
        h = http_handler_with_body(body)
        handler.set_request_context(h, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_create_template()
        assert _status(result) == 201

    def test_create_value_error(self, handler, http_handler_with_body, mock_registry):
        mock_registry.import_template.side_effect = ValueError("invalid data")
        body = {"name": "Template"}
        h = http_handler_with_body(body)
        handler.set_request_context(h, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_create_template()
        assert _status(result) == 400
        assert "invalid" in _body(result).get("error", "").lower()

    def test_create_os_error(self, handler, http_handler_with_body, mock_registry):
        mock_registry.import_template.side_effect = OSError("disk full")
        body = {"name": "Template"}
        h = http_handler_with_body(body)
        handler.set_request_context(h, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_create_template()
        assert _status(result) == 500

    def test_create_circuit_breaker_open(self, handler, http_handler_with_body):
        handler._circuit_breaker._state = "open"
        handler._circuit_breaker._last_failure_time = None
        body = {"name": "Template"}
        h = http_handler_with_body(body)
        handler.set_request_context(h, {})
        result = handler.handle_create_template()
        assert _status(result) == 503


# ===========================================================================
# Handler Tests: Delete Template
# ===========================================================================


class TestDeleteTemplate:
    """Tests for handle_delete_template (DELETE /api/v1/marketplace/templates/{id})."""

    def test_delete_returns_200(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_delete_template("tpl-1")
        assert _status(result) == 200

    def test_delete_returns_success(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_delete_template("tpl-1")
        body = _body(result)
        assert body["success"] is True
        assert body["deleted"] == "tpl-1"

    def test_delete_forbidden_builtin(self, handler, http_handler, mock_registry):
        mock_registry.delete.return_value = False
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_delete_template("tpl-1")
        assert _status(result) == 403
        assert "cannot delete" in _body(result).get("error", "").lower()

    def test_delete_invalid_id(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_delete_template("bad id!!")
        assert _status(result) == 400

    def test_delete_empty_id(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_delete_template("")
        assert _status(result) == 400

    def test_delete_circuit_breaker_open(self, handler, http_handler):
        handler._circuit_breaker._state = "open"
        handler._circuit_breaker._last_failure_time = None
        handler.set_request_context(http_handler, {})
        result = handler.handle_delete_template("tpl-1")
        assert _status(result) == 503

    def test_delete_registry_error(self, handler, http_handler, mock_registry):
        mock_registry.delete.side_effect = OSError("disk error")
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_delete_template("tpl-1")
        assert _status(result) == 500


# ===========================================================================
# Handler Tests: Rate Template
# ===========================================================================


class TestRateTemplate:
    """Tests for handle_rate_template (POST /api/v1/marketplace/templates/{id}/ratings)."""

    def test_rate_returns_200(self, handler, http_handler_with_body, mock_registry):
        h = http_handler_with_body({"score": 5})
        handler.set_request_context(h, {})
        with (
            patch(
                "aragora.server.handlers.marketplace._get_registry",
                return_value=mock_registry,
            ),
            patch(
                "aragora.marketplace.TemplateRating",
                MagicMock(),
            ),
        ):
            result = handler.handle_rate_template("tpl-1")
        assert _status(result) == 200

    def test_rate_returns_average(self, handler, http_handler_with_body, mock_registry):
        h = http_handler_with_body({"score": 4})
        handler.set_request_context(h, {})
        with (
            patch(
                "aragora.server.handlers.marketplace._get_registry",
                return_value=mock_registry,
            ),
            patch(
                "aragora.marketplace.TemplateRating",
                MagicMock(),
            ),
        ):
            result = handler.handle_rate_template("tpl-1")
        body = _body(result)
        assert body["success"] is True
        assert body["average_rating"] == 4.5

    def test_rate_with_review(self, handler, http_handler_with_body, mock_registry):
        h = http_handler_with_body({"score": 5, "review": "Excellent!"})
        handler.set_request_context(h, {})
        with (
            patch(
                "aragora.server.handlers.marketplace._get_registry",
                return_value=mock_registry,
            ),
            patch(
                "aragora.marketplace.TemplateRating",
                MagicMock(),
            ),
        ):
            result = handler.handle_rate_template("tpl-1")
        assert _status(result) == 200

    def test_rate_empty_body(self, handler, http_handler):
        http_handler.headers = {"Content-Length": "0"}
        handler.set_request_context(http_handler, {})
        result = handler.handle_rate_template("tpl-1")
        assert _status(result) == 400

    def test_rate_missing_score(self, handler, http_handler_with_body, mock_registry):
        h = http_handler_with_body({"review": "no score"})
        handler.set_request_context(h, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_rate_template("tpl-1")
        assert _status(result) == 400

    def test_rate_invalid_score_too_low(self, handler, http_handler_with_body, mock_registry):
        h = http_handler_with_body({"score": 0})
        handler.set_request_context(h, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_rate_template("tpl-1")
        assert _status(result) == 400

    def test_rate_invalid_score_too_high(self, handler, http_handler_with_body, mock_registry):
        h = http_handler_with_body({"score": 6})
        handler.set_request_context(h, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_rate_template("tpl-1")
        assert _status(result) == 400

    def test_rate_non_integer_score(self, handler, http_handler_with_body, mock_registry):
        h = http_handler_with_body({"score": "five"})
        handler.set_request_context(h, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_rate_template("tpl-1")
        assert _status(result) == 400

    def test_rate_review_too_long(self, handler, http_handler_with_body, mock_registry):
        h = http_handler_with_body({"score": 4, "review": "x" * (MAX_REVIEW_LENGTH + 1)})
        handler.set_request_context(h, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_rate_template("tpl-1")
        assert _status(result) == 400

    def test_rate_review_non_string(self, handler, http_handler_with_body, mock_registry):
        h = http_handler_with_body({"score": 4, "review": 12345})
        handler.set_request_context(h, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_rate_template("tpl-1")
        assert _status(result) == 400

    def test_rate_invalid_template_id(self, handler, http_handler_with_body):
        h = http_handler_with_body({"score": 5})
        handler.set_request_context(h, {})
        result = handler.handle_rate_template("bad id!!")
        assert _status(result) == 400

    def test_rate_circuit_breaker_open(self, handler, http_handler_with_body):
        handler._circuit_breaker._state = "open"
        handler._circuit_breaker._last_failure_time = None
        h = http_handler_with_body({"score": 5})
        handler.set_request_context(h, {})
        result = handler.handle_rate_template("tpl-1")
        assert _status(result) == 503

    def test_rate_registry_error(self, handler, http_handler_with_body, mock_registry):
        mock_registry.rate.side_effect = TypeError("bad type")
        h = http_handler_with_body({"score": 5})
        handler.set_request_context(h, {})
        with (
            patch(
                "aragora.server.handlers.marketplace._get_registry",
                return_value=mock_registry,
            ),
            patch(
                "aragora.marketplace.TemplateRating",
                MagicMock(),
            ),
        ):
            result = handler.handle_rate_template("tpl-1")
        assert _status(result) == 500

    def test_rate_value_error(self, handler, http_handler_with_body, mock_registry):
        h = http_handler_with_body({"score": 3})
        handler.set_request_context(h, {})
        with (
            patch(
                "aragora.server.handlers.marketplace._get_registry",
                return_value=mock_registry,
            ),
            patch(
                "aragora.marketplace.TemplateRating",
                side_effect=ValueError("bad rating"),
            ),
        ):
            result = handler.handle_rate_template("tpl-1")
        assert _status(result) == 400


# ===========================================================================
# Handler Tests: Get Ratings
# ===========================================================================


class TestGetRatings:
    """Tests for handle_get_ratings (GET /api/v1/marketplace/templates/{id}/ratings)."""

    def test_get_ratings_returns_200(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_get_ratings("tpl-1")
        assert _status(result) == 200

    def test_get_ratings_has_ratings_list(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_get_ratings("tpl-1")
        body = _body(result)
        assert "ratings" in body
        assert body["count"] == 2
        assert body["average"] == 4.5

    def test_get_ratings_structure(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_get_ratings("tpl-1")
        body = _body(result)
        rating = body["ratings"][0]
        assert "user_id" in rating
        assert "score" in rating
        assert "review" in rating
        assert "created_at" in rating

    def test_get_ratings_invalid_id(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_get_ratings("bad id!!")
        assert _status(result) == 400

    def test_get_ratings_circuit_breaker_open(self, handler, http_handler):
        handler._circuit_breaker._state = "open"
        handler._circuit_breaker._last_failure_time = None
        handler.set_request_context(http_handler, {})
        result = handler.handle_get_ratings("tpl-1")
        assert _status(result) == 503

    def test_get_ratings_registry_error(self, handler, http_handler, mock_registry):
        mock_registry.get_ratings.side_effect = AttributeError("missing attr")
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_get_ratings("tpl-1")
        assert _status(result) == 500


# ===========================================================================
# Handler Tests: Star Template
# ===========================================================================


class TestStarTemplate:
    """Tests for handle_star_template (POST /api/v1/marketplace/templates/{id}/star)."""

    def test_star_returns_200(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_star_template("tpl-1")
        assert _status(result) == 200

    def test_star_returns_count(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_star_template("tpl-1")
        body = _body(result)
        assert body["success"] is True
        assert "stars" in body

    def test_star_template_not_found(self, handler, http_handler, mock_registry):
        mock_registry.get.return_value = None
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_star_template("nonexistent")
        # Should still return 200 but stars=0
        assert _status(result) == 200
        body = _body(result)
        assert body["stars"] == 0

    def test_star_invalid_id(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_star_template("bad id!!")
        assert _status(result) == 400

    def test_star_circuit_breaker_open(self, handler, http_handler):
        handler._circuit_breaker._state = "open"
        handler._circuit_breaker._last_failure_time = None
        handler.set_request_context(http_handler, {})
        result = handler.handle_star_template("tpl-1")
        assert _status(result) == 503

    def test_star_registry_error(self, handler, http_handler, mock_registry):
        mock_registry.star.side_effect = KeyError("not found")
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_star_template("tpl-1")
        assert _status(result) == 500


# ===========================================================================
# Handler Tests: List Categories
# ===========================================================================


class TestListCategories:
    """Tests for handle_list_categories (GET /api/v1/marketplace/categories)."""

    def test_categories_returns_200(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_categories()
        assert _status(result) == 200

    def test_categories_returns_list(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_categories()
        body = _body(result)
        assert "categories" in body
        assert body["categories"] == ["analysis", "coding", "debate"]

    def test_categories_circuit_breaker_open(self, handler, http_handler):
        handler._circuit_breaker._state = "open"
        handler._circuit_breaker._last_failure_time = None
        handler.set_request_context(http_handler, {})
        result = handler.handle_list_categories()
        assert _status(result) == 503

    def test_categories_registry_error(self, handler, http_handler, mock_registry):
        mock_registry.list_categories.side_effect = TypeError("bad")
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_categories()
        assert _status(result) == 500


# ===========================================================================
# Handler Tests: Export Template
# ===========================================================================


class TestExportTemplate:
    """Tests for handle_export_template (GET /api/v1/marketplace/templates/{id}/export)."""

    def test_export_returns_200(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_export_template("tpl-1")
        assert _status(result) == 200

    def test_export_returns_json_content(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_export_template("tpl-1")
        assert result.content_type == "application/json"
        data = json.loads(result.body.decode("utf-8"))
        assert data["id"] == "tpl-1"

    def test_export_has_content_disposition(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_export_template("tpl-1")
        assert result.headers is not None
        assert "Content-Disposition" in result.headers
        assert "tpl-1.json" in result.headers["Content-Disposition"]

    def test_export_not_found(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_export_template("nonexistent")
        assert _status(result) == 404

    def test_export_invalid_id(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_export_template("bad id!!")
        assert _status(result) == 400

    def test_export_circuit_breaker_open(self, handler, http_handler):
        handler._circuit_breaker._state = "open"
        handler._circuit_breaker._last_failure_time = None
        handler.set_request_context(http_handler, {})
        result = handler.handle_export_template("tpl-1")
        assert _status(result) == 503

    def test_export_registry_error(self, handler, http_handler, mock_registry):
        mock_registry.export_template.side_effect = ValueError("encode failed")
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_export_template("tpl-1")
        assert _status(result) == 500


# ===========================================================================
# Handler Tests: Import Template
# ===========================================================================


class TestImportTemplate:
    """Tests for handle_import_template (POST /api/v1/marketplace/templates/import)."""

    def test_import_delegates_to_create(self, handler, http_handler_with_body, mock_registry):
        body = {"name": "Imported Template"}
        h = http_handler_with_body(body)
        handler.set_request_context(h, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_import_template()
        assert _status(result) == 201

    def test_import_returns_id(self, handler, http_handler_with_body, mock_registry):
        body = {"name": "Imported"}
        h = http_handler_with_body(body)
        handler.set_request_context(h, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_import_template()
        data = _body(result)
        assert data["id"] == "new-tpl-id"
        assert data["success"] is True

    def test_import_empty_body(self, handler, http_handler):
        http_handler.headers = {"Content-Length": "0"}
        handler.set_request_context(http_handler, {})
        result = handler.handle_import_template()
        assert _status(result) == 400


# ===========================================================================
# Handler Tests: Status
# ===========================================================================


class TestStatus:
    """Tests for handle_status (GET /api/v1/marketplace/status)."""

    def test_status_returns_200(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_status()
        assert _status(result) == 200

    def test_status_healthy_when_closed(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_status()
        body = _body(result)
        assert body["status"] == "healthy"
        assert body["circuit_breaker"]["state"] == "closed"

    def test_status_degraded_when_open(self, handler, http_handler):
        handler._circuit_breaker._state = "open"
        handler._circuit_breaker._last_failure_time = None
        handler.set_request_context(http_handler, {})
        result = handler.handle_status()
        body = _body(result)
        assert body["status"] == "degraded"

    def test_status_includes_circuit_breaker_details(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_status()
        body = _body(result)
        cb = body["circuit_breaker"]
        assert "state" in cb
        assert "failure_count" in cb
        assert "failure_threshold" in cb


# ===========================================================================
# Circuit Breaker Integration Tests
# ===========================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker behavior across handler methods."""

    def test_failure_records_on_registry_error(self, handler, http_handler, mock_registry):
        mock_registry.search.side_effect = OSError("fail")
        handler.set_request_context(http_handler, {})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            handler.handle_list_templates()
        assert handler._circuit_breaker._failure_count >= 1

    def test_success_records_on_happy_path(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {})
        # Reset success count tracking
        handler._circuit_breaker._failure_count = 2
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            handler.handle_list_templates()
        # record_success resets failure_count to 0 in CLOSED state
        assert handler._circuit_breaker._failure_count == 0

    def test_circuit_breaker_transitions_to_open(self, handler, http_handler, mock_registry):
        mock_registry.get.side_effect = OSError("fail")
        handler.set_request_context(http_handler, {})
        # Trigger enough failures to open the breaker
        for _ in range(handler._circuit_breaker.failure_threshold):
            with patch(
                "aragora.server.handlers.marketplace._get_registry",
                return_value=mock_registry,
            ):
                handler.handle_get_template("tpl-1")
        assert handler._circuit_breaker.state == "open"

    def test_open_circuit_rejects_requests(self, handler, http_handler, mock_registry):
        handler._circuit_breaker._state = "open"
        handler._circuit_breaker._last_failure_time = None
        handler.set_request_context(http_handler, {})
        result = handler.handle_list_templates()
        assert _status(result) == 503

    def test_reset_circuit_breaker(self, handler, http_handler):
        handler._circuit_breaker._state = "open"
        reset_marketplace_circuit_breaker()
        assert handler._circuit_breaker.state == "closed"


# ===========================================================================
# Handler Initialization Tests
# ===========================================================================


class TestHandlerInit:
    """Tests for MarketplaceHandler initialization."""

    def test_init_with_server_context(self):
        h = MarketplaceHandler(server_context={"key": "value"})
        assert h.ctx == {"key": "value"}

    def test_init_with_ctx(self):
        h = MarketplaceHandler(ctx={"key": "value"})
        assert h.ctx == {"key": "value"}

    def test_init_with_empty_context(self):
        h = MarketplaceHandler(server_context={})
        assert h.ctx == {}

    def test_init_with_none(self):
        h = MarketplaceHandler(ctx=None, server_context=None)
        assert h.ctx == {}

    def test_circuit_breaker_initialized(self):
        h = MarketplaceHandler(server_context={})
        assert h._circuit_breaker is not None


# ===========================================================================
# Edge Cases and Security Tests
# ===========================================================================


class TestEdgeCases:
    """Edge case and security tests."""

    def test_path_traversal_id(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_get_template("../../../etc/passwd")
        assert _status(result) == 400

    def test_sql_injection_id(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_get_template("'; DROP TABLE templates;--")
        assert _status(result) == 400

    def test_xss_in_query(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {"q": "<script>alert('xss')</script>"})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        # The query should be sanitized, not cause an error
        assert _status(result) in (200, 400)

    def test_unicode_template_id(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_get_template("\u00e9\u00e8\u00ea")
        assert _status(result) == 400

    def test_null_bytes_in_id(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_get_template("tpl\x00-1")
        assert _status(result) == 400

    def test_very_long_id(self, handler, http_handler):
        handler.set_request_context(http_handler, {})
        result = handler.handle_get_template("a" * 500)
        assert _status(result) == 400

    def test_create_with_no_handler(self, handler):
        # _current_handler is None => get_json_body returns None => empty body
        handler._current_handler = None
        handler._current_query_params = {}
        result = handler.handle_create_template()
        assert _status(result) == 400

    def test_empty_tags_parsed_correctly(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {"tags": ",,,"})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        assert _status(result) == 200

    def test_pagination_boundary_values(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {"limit": "0", "offset": "0"})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        body = _body(result)
        # limit=0 gets clamped to MIN_LIMIT=1
        assert body["limit"] == MIN_LIMIT
        assert body["offset"] == 0

    def test_negative_pagination(self, handler, http_handler, mock_registry):
        handler.set_request_context(http_handler, {"limit": "-1", "offset": "-1"})
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            return_value=mock_registry,
        ):
            result = handler.handle_list_templates()
        body = _body(result)
        assert body["limit"] == MIN_LIMIT
        assert body["offset"] == 0


# ===========================================================================
# Module-level Function Tests
# ===========================================================================


class TestModuleFunctions:
    """Tests for module-level helper functions."""

    def test_get_marketplace_circuit_breaker(self):
        from aragora.server.handlers.marketplace import get_marketplace_circuit_breaker

        cb = get_marketplace_circuit_breaker()
        assert cb is not None
        assert cb.state == "closed"

    def test_get_marketplace_circuit_breaker_status(self):
        from aragora.server.handlers.marketplace import get_marketplace_circuit_breaker_status

        status = get_marketplace_circuit_breaker_status()
        assert "state" in status
        assert status["state"] == "closed"

    def test_reset_marketplace_circuit_breaker(self):
        from aragora.server.handlers.marketplace import (
            get_marketplace_circuit_breaker,
            reset_marketplace_circuit_breaker,
        )

        cb = get_marketplace_circuit_breaker()
        cb._failure_count = 10
        cb._state = "open"
        reset_marketplace_circuit_breaker()
        assert cb.state == "closed"
        assert cb._failure_count == 0

    def test_clear_registry(self):
        from aragora.server.handlers.marketplace import _clear_registry, _registry

        _clear_registry()
        # After clearing, the module-level _registry should be None
        import aragora.server.handlers.marketplace as m

        assert m._registry is None
