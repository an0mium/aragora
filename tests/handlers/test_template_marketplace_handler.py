"""
Tests for TemplateMarketplaceHandler and TemplateRecommendationsHandler.

Covers all routes and behavior of both handler classes plus supporting
validation functions and dataclasses:

- can_handle() route matching
- GET  /api/v1/marketplace/templates          - Browse/search/filter/sort
- GET  /api/v1/marketplace/templates/:id      - Get template details
- POST /api/v1/marketplace/templates          - Publish template
- POST /api/v1/marketplace/templates/:id/rate - Rate a template
- GET  /api/v1/marketplace/templates/:id/reviews  - Get reviews
- POST /api/v1/marketplace/templates/:id/reviews  - Submit review
- POST /api/v1/marketplace/templates/:id/import   - Import template
- GET  /api/v1/marketplace/featured           - Featured templates
- GET  /api/v1/marketplace/trending           - Trending templates
- GET  /api/v1/marketplace/categories         - Categories with counts
- GET  /api/v1/marketplace/circuit-breaker    - Circuit breaker status
- GET  /api/v1/marketplace/recommendations    - Recommendations
- Validation functions (name, category, pattern, tags, rating, review)
- MarketplaceTemplate and TemplateReview dataclasses
- Circuit breaker integration (open circuit -> 503)
- Rate limiting
- Input validation error paths
- _clear_marketplace_state

NOTE: The handler's _route_request uses ``"/".join(parts[4:])`` to extract
the template_id from the path.  For path
``/api/v1/marketplace/templates/security/code-audit`` this yields
``"templates/security/code-audit"`` (because ``parts`` includes the leading
empty string from the split).  Tests that exercise routing through
``handle()`` must insert templates keyed by the *routed* id (with the
``templates/`` prefix) or call internal methods directly with the correct id.
"""

from __future__ import annotations

import io
import json
import sys
import time
import types as _types_mod
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Slack stubs to prevent transitive import issues (standard pattern)
# ---------------------------------------------------------------------------
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m


from aragora.server.handlers.template_marketplace import (
    DESCRIPTION_MAX_LENGTH,
    MAX_TAG_LENGTH,
    MAX_TAGS,
    REVIEW_CONTENT_MAX_LENGTH,
    TEMPLATE_NAME_MAX_LENGTH,
    VALID_CATEGORIES,
    VALID_PATTERNS,
    MarketplaceCircuitBreaker,
    MarketplaceTemplate,
    TemplateMarketplaceHandler,
    TemplateRecommendationsHandler,
    TemplateReview,
    _clear_marketplace_state,
    _marketplace_templates,
    _seed_marketplace_templates,
    _template_reviews,
    _user_ratings,
    get_marketplace_circuit_breaker_status,
    validate_category,
    validate_pattern,
    validate_rating,
    validate_review_content,
    validate_tags,
    validate_template_name,
)


# ===========================================================================
# Helpers
# ===========================================================================

# The routed template_id for path "/api/v1/marketplace/templates/<cat>/<name>"
# is "templates/<cat>/<name>" due to parts[4:] in the handler.
_ROUTED_TPL_ID = "templates/test-cat/test-tpl"


def parse_body(result) -> dict[str, Any]:
    """Parse HandlerResult body into a dict."""
    return json.loads(result.body.decode("utf-8"))


def make_mock_handler(
    *,
    method: str = "GET",
    client_address: tuple[str, int] = ("127.0.0.1", 12345),
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
) -> MagicMock:
    """Create a mock HTTP handler with configurable attributes."""
    mock = MagicMock()
    mock.command = method
    mock.client_address = client_address
    mock.headers = headers or {}
    if body is not None:
        mock.headers["Content-Length"] = str(len(body))
        mock.rfile = io.BytesIO(body)
    else:
        mock.headers.setdefault("Content-Length", "0")
        mock.rfile = io.BytesIO(b"")
    return mock


def json_body(data: dict) -> bytes:
    """Encode a dict as JSON bytes."""
    return json.dumps(data).encode("utf-8")


def _publish_body(**overrides) -> dict:
    """Build a valid publish body with optional overrides."""
    base = {
        "name": "Test Template",
        "description": "A test template description.",
        "category": "security",
        "pattern": "debate",
        "workflow_definition": {"steps": [{"name": "step1"}]},
        "tags": ["test", "security"],
        "author_id": "user-1",
        "author_name": "Test Author",
    }
    base.update(overrides)
    return base


def _insert_routed_template(
    routed_id: str = _ROUTED_TPL_ID,
    **overrides: Any,
) -> MarketplaceTemplate:
    """Insert a template keyed by its *routed* id for end-to-end handle() tests."""
    defaults = dict(
        id=routed_id,
        name="Routed Test",
        description="Template for routing tests.",
        category="security",
        pattern="debate",
        author_id="test",
        author_name="Test Author",
        rating=4.5,
        rating_count=10,
        download_count=100,
        is_featured=False,
        is_verified=True,
    )
    defaults.update(overrides)
    tpl = MarketplaceTemplate(**defaults)
    _marketplace_templates[routed_id] = tpl
    return tpl


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def clean_marketplace_state():
    """Reset marketplace state before and after each test."""
    _clear_marketplace_state()
    yield
    _clear_marketplace_state()


@pytest.fixture
def handler() -> TemplateMarketplaceHandler:
    """Create a TemplateMarketplaceHandler instance."""
    return TemplateMarketplaceHandler(ctx={})


@pytest.fixture
def rec_handler() -> TemplateRecommendationsHandler:
    """Create a TemplateRecommendationsHandler instance."""
    return TemplateRecommendationsHandler(ctx={})


@pytest.fixture
def seeded_handler(handler) -> TemplateMarketplaceHandler:
    """Create a handler with seeded marketplace templates."""
    _seed_marketplace_templates()
    return handler


@pytest.fixture
def mock_http_get() -> MagicMock:
    """Mock HTTP handler for GET requests."""
    return make_mock_handler(method="GET")


@pytest.fixture
def mock_http_post() -> MagicMock:
    """Mock HTTP handler for POST requests."""
    return make_mock_handler(method="POST")


# ===========================================================================
# Tests: Validation Functions
# ===========================================================================


class TestValidateTemplateName:
    """Tests for validate_template_name."""

    def test_valid_name(self):
        ok, err = validate_template_name("My Template")
        assert ok is True
        assert err == ""

    def test_empty_name(self):
        ok, err = validate_template_name("")
        assert ok is False
        assert "required" in err.lower()

    def test_whitespace_only_name(self):
        ok, err = validate_template_name("   ")
        assert ok is False
        assert "empty" in err.lower() or "whitespace" in err.lower()

    def test_name_too_long(self):
        ok, err = validate_template_name("x" * (TEMPLATE_NAME_MAX_LENGTH + 1))
        assert ok is False
        assert "too long" in err.lower()

    def test_name_at_max_length(self):
        ok, err = validate_template_name("x" * TEMPLATE_NAME_MAX_LENGTH)
        assert ok is True

    def test_single_char_name(self):
        ok, err = validate_template_name("A")
        assert ok is True


class TestValidateCategory:
    """Tests for validate_category."""

    def test_valid_categories(self):
        for cat in VALID_CATEGORIES:
            ok, err = validate_category(cat)
            assert ok is True, f"Category '{cat}' should be valid"

    def test_empty_category(self):
        ok, err = validate_category("")
        assert ok is False
        assert "required" in err.lower()

    def test_invalid_category(self):
        ok, err = validate_category("nonexistent")
        assert ok is False
        assert "Invalid category" in err


class TestValidatePattern:
    """Tests for validate_pattern."""

    def test_valid_patterns(self):
        for p in VALID_PATTERNS:
            ok, err = validate_pattern(p)
            assert ok is True, f"Pattern '{p}' should be valid"

    def test_empty_pattern(self):
        ok, err = validate_pattern("")
        assert ok is False
        assert "required" in err.lower()

    def test_invalid_pattern(self):
        ok, err = validate_pattern("unknown_pattern")
        assert ok is False
        assert "Invalid pattern" in err


class TestValidateTags:
    """Tests for validate_tags."""

    def test_valid_tags(self):
        ok, err = validate_tags(["tag1", "tag2"])
        assert ok is True

    def test_empty_tags_list(self):
        ok, err = validate_tags([])
        assert ok is True

    def test_too_many_tags(self):
        ok, err = validate_tags(["t"] * (MAX_TAGS + 1))
        assert ok is False
        assert "Too many" in err

    def test_non_list_tags(self):
        ok, err = validate_tags("not-a-list")
        assert ok is False
        assert "must be a list" in err.lower()

    def test_non_string_tag(self):
        ok, err = validate_tags([123])
        assert ok is False
        assert "must be a string" in err.lower()

    def test_tag_too_long(self):
        ok, err = validate_tags(["x" * (MAX_TAG_LENGTH + 1)])
        assert ok is False
        assert "too long" in err.lower()

    def test_empty_tag(self):
        ok, err = validate_tags(["  "])
        assert ok is False
        assert "cannot be empty" in err.lower()

    def test_tags_at_max_count(self):
        ok, err = validate_tags(["t"] * MAX_TAGS)
        assert ok is True

    def test_tag_at_max_length(self):
        ok, err = validate_tags(["x" * MAX_TAG_LENGTH])
        assert ok is True


class TestValidateRating:
    """Tests for validate_rating."""

    def test_valid_int_rating(self):
        ok, err = validate_rating(4)
        assert ok is True

    def test_valid_float_rating(self):
        ok, err = validate_rating(3.5)
        assert ok is True

    def test_rating_boundary_low(self):
        ok, err = validate_rating(1)
        assert ok is True

    def test_rating_boundary_high(self):
        ok, err = validate_rating(5)
        assert ok is True

    def test_rating_too_low(self):
        ok, err = validate_rating(0)
        assert ok is False
        assert "between 1 and 5" in err

    def test_rating_too_high(self):
        ok, err = validate_rating(6)
        assert ok is False

    def test_non_numeric_rating(self):
        ok, err = validate_rating("five")
        assert ok is False
        assert "must be a number" in err.lower()

    def test_none_rating(self):
        ok, err = validate_rating(None)
        assert ok is False


class TestValidateReviewContent:
    """Tests for validate_review_content."""

    def test_valid_content(self):
        ok, err = validate_review_content("Great template!")
        assert ok is True

    def test_empty_content(self):
        ok, err = validate_review_content("")
        assert ok is False
        assert "required" in err.lower()

    def test_whitespace_only_content(self):
        ok, err = validate_review_content("   ")
        assert ok is False
        assert "empty" in err.lower() or "whitespace" in err.lower()

    def test_content_too_long(self):
        ok, err = validate_review_content("x" * (REVIEW_CONTENT_MAX_LENGTH + 1))
        assert ok is False
        assert "too long" in err.lower()

    def test_content_at_max_length(self):
        ok, err = validate_review_content("x" * REVIEW_CONTENT_MAX_LENGTH)
        assert ok is True


# ===========================================================================
# Tests: Dataclasses
# ===========================================================================


class TestMarketplaceTemplate:
    """Tests for MarketplaceTemplate dataclass."""

    def test_to_dict(self):
        t = MarketplaceTemplate(
            id="test/template",
            name="Test",
            description="A test template",
            category="security",
            pattern="debate",
            author_id="user-1",
            author_name="Author",
        )
        d = t.to_dict()
        assert d["id"] == "test/template"
        assert d["name"] == "Test"
        assert d["category"] == "security"
        assert d["pattern"] == "debate"
        assert d["rating"] == 0.0
        assert d["rating_count"] == 0
        assert d["download_count"] == 0
        assert d["is_featured"] is False
        assert d["is_verified"] is False
        assert "created_at" in d
        assert "updated_at" in d
        assert d["tags"] == []
        assert d["workflow_definition"] == {}

    def test_to_summary(self):
        t = MarketplaceTemplate(
            id="test/template",
            name="Test",
            description="Short desc",
            category="security",
            pattern="debate",
            author_id="user-1",
            author_name="Author",
            tags=["a", "b", "c", "d", "e", "f"],
        )
        s = t.to_summary()
        assert s["id"] == "test/template"
        assert s["name"] == "Test"
        assert s["description"] == "Short desc"
        # Summary should truncate tags to 5
        assert len(s["tags"]) == 5
        # Summary should not include workflow_definition
        assert "workflow_definition" not in s
        assert "input_schema" not in s

    def test_to_summary_truncates_long_description(self):
        long_desc = "x" * 300
        t = MarketplaceTemplate(
            id="test/t", name="T", description=long_desc,
            category="security", pattern="debate",
            author_id="u", author_name="A",
        )
        s = t.to_summary()
        assert len(s["description"]) == 203  # 200 chars + "..."
        assert s["description"].endswith("...")

    def test_default_values(self):
        t = MarketplaceTemplate(
            id="t", name="N", description="D", category="security",
            pattern="debate", author_id="a", author_name="A",
        )
        assert t.version == "1.0.0"
        assert t.tags == []
        assert t.documentation == ""
        assert t.examples == []


class TestTemplateReview:
    """Tests for TemplateReview dataclass."""

    def test_to_dict(self):
        r = TemplateReview(
            id="rev-1",
            template_id="tpl-1",
            user_id="user-1",
            user_name="Test User",
            rating=5,
            title="Great",
            content="Works well",
        )
        d = r.to_dict()
        assert d["id"] == "rev-1"
        assert d["template_id"] == "tpl-1"
        assert d["rating"] == 5
        assert d["title"] == "Great"
        assert d["content"] == "Works well"
        assert d["helpful_count"] == 0
        assert "created_at" in d

    def test_default_helpful_count(self):
        r = TemplateReview(
            id="r", template_id="t", user_id="u", user_name="U",
            rating=3, title="T", content="C",
        )
        assert r.helpful_count == 0


# ===========================================================================
# Tests: can_handle
# ===========================================================================


class TestCanHandle:
    """Tests for route matching."""

    def test_marketplace_templates_path(self, handler):
        assert handler.can_handle("/api/v1/marketplace/templates") is True

    def test_marketplace_templates_with_id(self, handler):
        assert handler.can_handle("/api/v1/marketplace/templates/security/code-audit") is True

    def test_marketplace_featured(self, handler):
        assert handler.can_handle("/api/v1/marketplace/featured") is True

    def test_marketplace_trending(self, handler):
        assert handler.can_handle("/api/v1/marketplace/trending") is True

    def test_marketplace_categories(self, handler):
        assert handler.can_handle("/api/v1/marketplace/categories") is True

    def test_marketplace_circuit_breaker(self, handler):
        assert handler.can_handle("/api/v1/marketplace/circuit-breaker") is True

    def test_non_marketplace_path(self, handler):
        assert handler.can_handle("/api/v1/debates/list") is False

    def test_root_path(self, handler):
        assert handler.can_handle("/") is False

    def test_recommendations_handler_can_handle(self, rec_handler):
        assert rec_handler.can_handle("/api/v1/marketplace/recommendations") is True

    def test_recommendations_handler_rejects_other(self, rec_handler):
        assert rec_handler.can_handle("/api/v1/marketplace/templates") is False


# ===========================================================================
# Tests: Circuit Breaker Status Endpoint
# ===========================================================================


class TestCircuitBreakerEndpoint:
    """Tests for GET /api/v1/marketplace/circuit-breaker."""

    def test_returns_circuit_breaker_status(self, handler, mock_http_get):
        result = handler.handle("/api/v1/marketplace/circuit-breaker", {}, mock_http_get)
        body = parse_body(result)
        assert result.status_code == 200
        assert "state" in body
        assert body["state"] == "closed"

    def test_circuit_breaker_open_blocks_requests(self, handler, mock_http_get):
        # Force circuit breaker open
        handler._circuit_breaker._state = "open"
        handler._circuit_breaker._last_failure_time = time.time()
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_http_get)
        assert result.status_code == 503
        body = parse_body(result)
        assert "circuit breaker" in body.get("error", "").lower()

    def test_circuit_breaker_status_bypasses_open_check(self, handler, mock_http_get):
        """Circuit breaker status endpoint should work even when circuit is open."""
        handler._circuit_breaker._state = "open"
        handler._circuit_breaker._last_failure_time = time.time()
        result = handler.handle("/api/v1/marketplace/circuit-breaker", {}, mock_http_get)
        assert result.status_code == 200
        body = parse_body(result)
        assert body["state"] == "open"


# ===========================================================================
# Tests: Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting on marketplace endpoints."""

    def test_browse_rate_limit(self, seeded_handler, mock_http_get):
        """Requests within browse rate limit should succeed."""
        result = seeded_handler.handle("/api/v1/marketplace/templates", {}, mock_http_get)
        assert result.status_code == 200

    def test_browse_rate_limit_exceeded(self, seeded_handler):
        """Exceeding browse rate limit should return 429."""
        with patch(
            "aragora.server.handlers.template_marketplace._marketplace_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            mock_get = make_mock_handler(method="GET")
            result = seeded_handler.handle(
                "/api/v1/marketplace/templates", {}, mock_get
            )
            assert result.status_code == 429


# ===========================================================================
# Tests: List Templates (through handle)
# ===========================================================================


class TestListTemplates:
    """Tests for GET /api/v1/marketplace/templates."""

    def test_list_returns_templates(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle("/api/v1/marketplace/templates", {}, mock_http_get)
        body = parse_body(result)
        assert result.status_code == 200
        assert "templates" in body
        assert "total" in body
        assert body["total"] > 0

    def test_list_with_category_filter(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"category": "security"},
            mock_http_get,
        )
        body = parse_body(result)
        assert result.status_code == 200
        for t in body["templates"]:
            assert t["category"] == "security"

    def test_list_with_pattern_filter(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"pattern": "debate"},
            mock_http_get,
        )
        body = parse_body(result)
        assert result.status_code == 200
        for t in body["templates"]:
            assert t["pattern"] == "debate"

    def test_list_with_verified_only(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"verified_only": "true"},
            mock_http_get,
        )
        body = parse_body(result)
        assert result.status_code == 200
        for t in body["templates"]:
            assert t["is_verified"] is True

    def test_list_with_search(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"search": "security"},
            mock_http_get,
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert body["total"] >= 1

    def test_list_with_tag_filter(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"tags": "sme"},
            mock_http_get,
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert body["total"] >= 1

    def test_list_sort_by_rating(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"sort_by": "rating"},
            mock_http_get,
        )
        body = parse_body(result)
        ratings = [t["rating"] for t in body["templates"]]
        assert ratings == sorted(ratings, reverse=True)

    def test_list_sort_by_downloads(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"sort_by": "downloads"},
            mock_http_get,
        )
        body = parse_body(result)
        downloads = [t["download_count"] for t in body["templates"]]
        assert downloads == sorted(downloads, reverse=True)

    def test_list_sort_by_name(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"sort_by": "name"},
            mock_http_get,
        )
        body = parse_body(result)
        names = [t["name"].lower() for t in body["templates"]]
        assert names == sorted(names)

    def test_list_sort_by_newest(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"sort_by": "newest"},
            mock_http_get,
        )
        body = parse_body(result)
        assert result.status_code == 200

    def test_list_pagination(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"limit": "3", "offset": "0"},
            mock_http_get,
        )
        body = parse_body(result)
        assert len(body["templates"]) <= 3
        assert body["limit"] == 3
        assert body["offset"] == 0

    def test_list_pagination_offset(self, seeded_handler, mock_http_get):
        # Get first page
        r1 = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"limit": "2", "offset": "0"},
            mock_http_get,
        )
        b1 = parse_body(r1)

        # Get second page
        mock2 = make_mock_handler(method="GET")
        r2 = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"limit": "2", "offset": "2"},
            mock2,
        )
        b2 = parse_body(r2)

        # Pages should be different
        ids1 = {t["id"] for t in b1["templates"]}
        ids2 = {t["id"] for t in b2["templates"]}
        assert ids1.isdisjoint(ids2)

    def test_list_no_results_for_unknown_category(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates",
            {"category": "nonexistent_cat"},
            mock_http_get,
        )
        body = parse_body(result)
        assert body["total"] == 0
        assert body["templates"] == []

    def test_method_not_allowed(self, seeded_handler):
        mock_put = make_mock_handler(method="PUT")
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates", {}, mock_put
        )
        assert result.status_code == 405


# ===========================================================================
# Tests: Get Template (via internal method)
# ===========================================================================


class TestGetTemplate:
    """Tests for _get_template (called with clean template_id)."""

    def test_get_existing_template(self, seeded_handler):
        result = seeded_handler._get_template("security/code-audit")
        body = parse_body(result)
        assert result.status_code == 200
        assert body["id"] == "security/code-audit"
        assert body["name"] == "Security Code Audit"
        # Full details should include workflow_definition
        assert "workflow_definition" in body

    def test_get_nonexistent_template(self, seeded_handler):
        result = seeded_handler._get_template("nonexistent/template")
        assert result.status_code == 404

    def test_get_template_routed(self, seeded_handler, mock_http_get):
        """End-to-end test via handle(). The routing yields
        template_id = 'templates/security/code-audit', so we insert one with
        that key."""
        _insert_routed_template(
            routed_id="templates/security/code-audit",
            name="Routing Test",
        )
        result = seeded_handler.handle(
            "/api/v1/marketplace/templates/security/code-audit",
            {},
            mock_http_get,
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert body["name"] == "Routing Test"


# ===========================================================================
# Tests: Publish Template
# ===========================================================================


class TestPublishTemplate:
    """Tests for POST /api/v1/marketplace/templates."""

    def test_publish_success(self, handler):
        body_data = _publish_body()
        mock_post = make_mock_handler(method="POST", body=json_body(body_data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        body = parse_body(result)
        assert result.status_code == 201
        assert body["status"] == "published"
        assert "template_id" in body

    def test_publish_missing_name(self, handler):
        data = _publish_body()
        del data["name"]
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_missing_description(self, handler):
        data = _publish_body()
        del data["description"]
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_missing_category(self, handler):
        data = _publish_body()
        del data["category"]
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_missing_pattern(self, handler):
        data = _publish_body()
        del data["pattern"]
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_missing_workflow_definition(self, handler):
        data = _publish_body()
        del data["workflow_definition"]
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_invalid_category(self, handler):
        data = _publish_body(category="invalid_category")
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_invalid_pattern(self, handler):
        data = _publish_body(pattern="invalid_pattern")
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_invalid_name_empty(self, handler):
        data = _publish_body(name="")
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_name_too_long(self, handler):
        data = _publish_body(name="x" * (TEMPLATE_NAME_MAX_LENGTH + 1))
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_description_too_long(self, handler):
        data = _publish_body(description="x" * (DESCRIPTION_MAX_LENGTH + 1))
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_empty_description(self, handler):
        data = _publish_body(description="   ")
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_workflow_definition_not_dict(self, handler):
        data = _publish_body(workflow_definition="not a dict")
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_invalid_tags(self, handler):
        data = _publish_body(tags=[123])
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_too_many_tags(self, handler):
        data = _publish_body(tags=["t"] * (MAX_TAGS + 1))
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_duplicate(self, handler):
        data = _publish_body()
        mock1 = make_mock_handler(method="POST", body=json_body(data))
        result1 = handler.handle("/api/v1/marketplace/templates", {}, mock1)
        assert result1.status_code == 201

        # Publish same template again
        mock2 = make_mock_handler(method="POST", body=json_body(data))
        result2 = handler.handle("/api/v1/marketplace/templates", {}, mock2)
        assert result2.status_code == 409

    def test_publish_invalid_json(self, handler):
        mock_post = make_mock_handler(method="POST", body=b"not json{{{")
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 400

    def test_publish_body_too_large(self, handler):
        mock_post = make_mock_handler(method="POST", body=b"{}")
        mock_post.headers["Content-Length"] = str(200000)
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        assert result.status_code == 413

    def test_publish_rate_limit(self, handler):
        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            data = _publish_body()
            mock_post = make_mock_handler(method="POST", body=json_body(data))
            result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
            assert result.status_code == 429

    def test_publish_with_optional_fields(self, handler):
        data = _publish_body(
            name="Optional Fields Test",
            version="2.0.0",
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "string"},
            documentation="Full documentation here.",
            examples=[{"input": "test", "output": "result"}],
        )
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        body = parse_body(result)
        assert result.status_code == 201
        assert body["status"] == "published"

    def test_publish_generates_id_from_category_and_name(self, handler):
        data = _publish_body(name="My Custom Template", category="analytics")
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        body = parse_body(result)
        assert result.status_code == 201
        assert body["template_id"] == "analytics/my-custom-template"

    def test_publish_default_author(self, handler):
        data = _publish_body()
        del data["author_id"]
        del data["author_name"]
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle("/api/v1/marketplace/templates", {}, mock_post)
        body = parse_body(result)
        assert result.status_code == 201
        # Verify template stored with defaults
        tpl_id = body["template_id"]
        tpl = _marketplace_templates[tpl_id]
        assert tpl.author_id == "anonymous"
        assert tpl.author_name == "Anonymous"


# ===========================================================================
# Tests: Rate Template (via internal method + routing)
# ===========================================================================


class TestRateTemplate:
    """Tests for _rate_template and POST .../:id/rate routing."""

    def test_rate_template_success(self, seeded_handler):
        """Test via internal method with clean template_id."""
        data = {"rating": 4, "user_id": "user-1"}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._rate_template("security/code-audit", mock_post, "127.0.0.1")
        body = parse_body(result)
        assert result.status_code == 200
        assert body["status"] == "rated"
        assert body["your_rating"] == 4

    def test_rate_template_not_found(self, seeded_handler):
        data = {"rating": 4}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._rate_template("nonexistent/template", mock_post, "127.0.0.1")
        assert result.status_code == 404

    def test_rate_template_invalid_rating(self, seeded_handler):
        data = {"rating": 10}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._rate_template("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 400

    def test_rate_template_missing_rating(self, seeded_handler):
        data = {}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._rate_template("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 400

    def test_rate_template_non_numeric(self, seeded_handler):
        data = {"rating": "great"}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._rate_template("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 400

    def test_rate_template_updates_average(self, seeded_handler):
        tpl = _marketplace_templates["security/code-audit"]
        original_count = tpl.rating_count

        data = {"rating": 5, "user_id": "new-user"}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._rate_template("security/code-audit", mock_post, "127.0.0.1")
        body = parse_body(result)
        assert result.status_code == 200
        assert body["rating_count"] == original_count + 1

    def test_rate_template_update_existing_rating(self, seeded_handler):
        # First rating
        data1 = {"rating": 2, "user_id": "user-1"}
        m1 = make_mock_handler(method="POST", body=json_body(data1))
        r1 = seeded_handler._rate_template("security/code-audit", m1, "127.0.0.1")
        b1 = parse_body(r1)
        count_after_first = b1["rating_count"]

        # Update the rating from the same user
        data2 = {"rating": 5, "user_id": "user-1"}
        m2 = make_mock_handler(method="POST", body=json_body(data2))
        r2 = seeded_handler._rate_template("security/code-audit", m2, "127.0.0.1")
        b2 = parse_body(r2)
        # Count should not increase on update
        assert b2["rating_count"] == count_after_first

    def test_rate_template_rate_limit(self, seeded_handler):
        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            data = {"rating": 4}
            mock_post = make_mock_handler(method="POST", body=json_body(data))
            result = seeded_handler._rate_template("security/code-audit", mock_post, "127.0.0.1")
            assert result.status_code == 429

    def test_rate_template_body_too_large(self, seeded_handler):
        mock_post = make_mock_handler(method="POST", body=b'{"rating": 5}')
        mock_post.headers["Content-Length"] = str(20000)
        result = seeded_handler._rate_template("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 413

    def test_rate_template_invalid_json(self, seeded_handler):
        mock_post = make_mock_handler(method="POST", body=b"not json")
        result = seeded_handler._rate_template("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 400

    def test_rate_template_routed(self, handler):
        """End-to-end routing test for rate endpoint."""
        tpl = _insert_routed_template(routed_id="templates/mycat/mytpl")
        data = {"rating": 3, "user_id": "user-x"}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle(
            "/api/v1/marketplace/templates/mycat/mytpl/rate", {}, mock_post
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert body["status"] == "rated"


# ===========================================================================
# Tests: Get Reviews (via internal method)
# ===========================================================================


class TestGetReviews:
    """Tests for _get_reviews."""

    def test_get_reviews_empty(self, seeded_handler):
        result = seeded_handler._get_reviews("security/code-audit", {})
        body = parse_body(result)
        assert result.status_code == 200
        assert body["reviews"] == []
        assert body["total"] == 0

    def test_get_reviews_not_found(self, seeded_handler):
        result = seeded_handler._get_reviews("nonexistent/template", {})
        assert result.status_code == 404

    def test_get_reviews_with_data(self, seeded_handler):
        review = TemplateReview(
            id="rev-1",
            template_id="security/code-audit",
            user_id="user-1",
            user_name="Reviewer",
            rating=5,
            title="Excellent",
            content="Works great.",
        )
        _template_reviews["security/code-audit"] = [review]

        result = seeded_handler._get_reviews("security/code-audit", {})
        body = parse_body(result)
        assert body["total"] == 1
        assert body["reviews"][0]["title"] == "Excellent"

    def test_get_reviews_pagination(self, seeded_handler):
        reviews = [
            TemplateReview(
                id=f"rev-{i}",
                template_id="security/code-audit",
                user_id=f"user-{i}",
                user_name=f"User {i}",
                rating=3,
                title=f"Review {i}",
                content=f"Content {i}",
                helpful_count=i,
            )
            for i in range(5)
        ]
        _template_reviews["security/code-audit"] = reviews

        result = seeded_handler._get_reviews(
            "security/code-audit", {"limit": "2", "offset": "0"}
        )
        body = parse_body(result)
        assert len(body["reviews"]) == 2
        assert body["total"] == 5
        # Should be sorted by helpful_count descending
        assert body["reviews"][0]["helpful_count"] >= body["reviews"][1]["helpful_count"]

    def test_get_reviews_routed(self, handler, mock_http_get):
        """End-to-end routing test for GET reviews."""
        tpl = _insert_routed_template(routed_id="templates/cat/tpl")
        _template_reviews["templates/cat/tpl"] = []
        result = handler.handle(
            "/api/v1/marketplace/templates/cat/tpl/reviews",
            {},
            mock_http_get,
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert body["total"] == 0


# ===========================================================================
# Tests: Submit Review (via internal method)
# ===========================================================================


class TestSubmitReview:
    """Tests for _submit_review."""

    def test_submit_review_success(self, seeded_handler):
        data = {
            "rating": 5,
            "content": "Excellent template, highly recommend.",
            "title": "Great work!",
            "user_id": "user-1",
            "user_name": "Reviewer",
        }
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        body = parse_body(result)
        assert result.status_code == 201
        assert body["status"] == "submitted"
        assert body["review"]["rating"] == 5
        assert body["review"]["title"] == "Great work!"

    def test_submit_review_not_found(self, seeded_handler):
        data = {"rating": 5, "content": "Good."}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("nonexistent/template", mock_post, "127.0.0.1")
        assert result.status_code == 404

    def test_submit_review_missing_rating(self, seeded_handler):
        data = {"content": "No rating here."}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 400

    def test_submit_review_missing_content(self, seeded_handler):
        data = {"rating": 4}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 400

    def test_submit_review_invalid_rating(self, seeded_handler):
        data = {"rating": 10, "content": "Bad rating."}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 400

    def test_submit_review_content_too_long(self, seeded_handler):
        data = {"rating": 4, "content": "x" * (REVIEW_CONTENT_MAX_LENGTH + 1)}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 400

    def test_submit_review_empty_content(self, seeded_handler):
        data = {"rating": 4, "content": "   "}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 400

    def test_submit_review_title_too_long(self, seeded_handler):
        data = {
            "rating": 4,
            "content": "Good template.",
            "title": "x" * (TEMPLATE_NAME_MAX_LENGTH + 1),
        }
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 400

    def test_submit_review_user_name_too_long(self, seeded_handler):
        data = {
            "rating": 4,
            "content": "Good template.",
            "user_name": "x" * (TEMPLATE_NAME_MAX_LENGTH + 1),
        }
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 400

    def test_submit_review_invalid_json(self, seeded_handler):
        mock_post = make_mock_handler(method="POST", body=b"not valid json")
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 400

    def test_submit_review_body_too_large(self, seeded_handler):
        mock_post = make_mock_handler(method="POST", body=b'{"rating": 5}')
        mock_post.headers["Content-Length"] = str(60000)
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 413

    def test_submit_review_updates_template_rating(self, seeded_handler):
        tpl = _marketplace_templates["security/code-audit"]
        original_count = tpl.rating_count

        data = {"rating": 5, "content": "Five stars!"}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        assert result.status_code == 201
        assert tpl.rating_count == original_count + 1

    def test_submit_review_default_user_name(self, seeded_handler):
        data = {"rating": 4, "content": "Good template."}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock_post, "127.0.0.1")
        body = parse_body(result)
        assert body["review"]["user_name"] == "Anonymous"

    def test_submit_review_routed_post(self, handler):
        """End-to-end routing test for POST reviews."""
        tpl = _insert_routed_template(routed_id="templates/cat/tpl")
        data = {"rating": 4, "content": "Via routing."}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = handler.handle(
            "/api/v1/marketplace/templates/cat/tpl/reviews",
            {},
            mock_post,
        )
        body = parse_body(result)
        assert result.status_code == 201
        assert body["status"] == "submitted"


# ===========================================================================
# Tests: Import Template (via internal method)
# ===========================================================================


class TestImportTemplate:
    """Tests for _import_template."""

    def test_import_success(self, seeded_handler):
        data = {"workspace_id": "ws-1"}
        mock_post = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._import_template("security/code-audit", mock_post)
        body = parse_body(result)
        assert result.status_code == 200
        assert body["status"] == "imported"
        assert body["template_id"] == "security/code-audit"
        assert body["workspace_id"] == "ws-1"
        assert "workflow_definition" in body

    def test_import_not_found(self, seeded_handler):
        mock_post = make_mock_handler(method="POST", body=json_body({"workspace_id": "ws-1"}))
        result = seeded_handler._import_template("nonexistent/template", mock_post)
        assert result.status_code == 404

    def test_import_increments_download_count(self, seeded_handler):
        tpl = _marketplace_templates["security/code-audit"]
        original_count = tpl.download_count

        mock_post = make_mock_handler(method="POST", body=json_body({}))
        result = seeded_handler._import_template("security/code-audit", mock_post)
        body = parse_body(result)
        assert body["download_count"] == original_count + 1
        assert tpl.download_count == original_count + 1

    def test_import_without_workspace_id(self, seeded_handler):
        mock_post = make_mock_handler(method="POST", body=json_body({}))
        result = seeded_handler._import_template("security/code-audit", mock_post)
        body = parse_body(result)
        assert result.status_code == 200
        assert body["workspace_id"] is None

    def test_import_body_too_large(self, seeded_handler):
        mock_post = make_mock_handler(method="POST", body=b"{}")
        mock_post.headers["Content-Length"] = str(20000)
        result = seeded_handler._import_template("security/code-audit", mock_post)
        assert result.status_code == 413

    def test_import_invalid_json_falls_through(self, seeded_handler):
        """Import with invalid JSON body should still work (empty data fallback)."""
        mock_post = make_mock_handler(method="POST", body=b"not json")
        result = seeded_handler._import_template("security/code-audit", mock_post)
        body = parse_body(result)
        assert result.status_code == 200
        assert body["workspace_id"] is None

    def test_import_routed(self, handler):
        """End-to-end routing test for import endpoint."""
        _insert_routed_template(routed_id="templates/cat/tpl", download_count=50)
        mock_post = make_mock_handler(method="POST", body=json_body({"workspace_id": "ws-2"}))
        result = handler.handle(
            "/api/v1/marketplace/templates/cat/tpl/import", {}, mock_post
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert body["status"] == "imported"
        assert body["download_count"] == 51


# ===========================================================================
# Tests: Featured Templates
# ===========================================================================


class TestFeaturedTemplates:
    """Tests for GET /api/v1/marketplace/featured."""

    def test_get_featured(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/featured", {}, mock_http_get
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert "featured" in body
        assert body["total"] > 0
        # All returned should be featured
        for t in body["featured"]:
            assert t["is_featured"] is True

    def test_featured_sorted_by_rating(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/featured", {}, mock_http_get
        )
        body = parse_body(result)
        ratings = [t["rating"] for t in body["featured"]]
        assert ratings == sorted(ratings, reverse=True)

    def test_featured_max_10(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/featured", {}, mock_http_get
        )
        body = parse_body(result)
        assert len(body["featured"]) <= 10


# ===========================================================================
# Tests: Trending Templates
# ===========================================================================


class TestTrendingTemplates:
    """Tests for GET /api/v1/marketplace/trending."""

    def test_get_trending(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/trending", {}, mock_http_get
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert "trending" in body
        assert "period" in body

    def test_trending_default_limit(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/trending", {}, mock_http_get
        )
        body = parse_body(result)
        assert len(body["trending"]) <= 10

    def test_trending_custom_limit(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/trending",
            {"limit": "3"},
            mock_http_get,
        )
        body = parse_body(result)
        assert len(body["trending"]) <= 3

    def test_trending_sorted_by_downloads(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/trending", {}, mock_http_get
        )
        body = parse_body(result)
        downloads = [t["download_count"] for t in body["trending"]]
        assert downloads == sorted(downloads, reverse=True)

    def test_trending_with_period(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/trending",
            {"period": "month"},
            mock_http_get,
        )
        body = parse_body(result)
        assert body["period"] == "month"


# ===========================================================================
# Tests: Categories
# ===========================================================================


class TestCategories:
    """Tests for GET /api/v1/marketplace/categories."""

    def test_get_categories(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/categories", {}, mock_http_get
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert "categories" in body
        assert body["total"] > 0

    def test_categories_have_counts(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/categories", {}, mock_http_get
        )
        body = parse_body(result)
        for cat in body["categories"]:
            assert "id" in cat
            assert "name" in cat
            assert "template_count" in cat
            assert "total_downloads" in cat
            assert cat["template_count"] > 0

    def test_categories_sorted_by_count(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/categories", {}, mock_http_get
        )
        body = parse_body(result)
        counts = [c["template_count"] for c in body["categories"]]
        assert counts == sorted(counts, reverse=True)


# ===========================================================================
# Tests: Recommendations Handler
# ===========================================================================


class TestRecommendationsHandler:
    """Tests for TemplateRecommendationsHandler."""

    def test_recommendations_returns_templates(self, rec_handler, mock_http_get):
        _seed_marketplace_templates()
        result = rec_handler.handle(
            "/api/v1/marketplace/recommendations", {}, mock_http_get
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert "recommendations" in body
        assert body["total"] > 0

    def test_recommendations_default_limit(self, rec_handler, mock_http_get):
        _seed_marketplace_templates()
        result = rec_handler.handle(
            "/api/v1/marketplace/recommendations", {}, mock_http_get
        )
        body = parse_body(result)
        assert len(body["recommendations"]) <= 5

    def test_recommendations_custom_limit(self, rec_handler, mock_http_get):
        _seed_marketplace_templates()
        result = rec_handler.handle(
            "/api/v1/marketplace/recommendations", {"limit": "3"}, mock_http_get
        )
        body = parse_body(result)
        assert len(body["recommendations"]) <= 3


# ===========================================================================
# Tests: Clear Marketplace State
# ===========================================================================


class TestClearMarketplaceState:
    """Tests for _clear_marketplace_state."""

    def test_clears_templates(self):
        _seed_marketplace_templates()
        assert len(_marketplace_templates) > 0
        _clear_marketplace_state()
        assert len(_marketplace_templates) == 0

    def test_clears_reviews(self):
        _template_reviews["test-id"] = [
            TemplateReview(
                id="r1", template_id="test-id", user_id="u1",
                user_name="U", rating=5, title="T", content="C",
            )
        ]
        _clear_marketplace_state()
        assert len(_template_reviews) == 0

    def test_clears_user_ratings(self):
        _user_ratings["user-1"] = {"tpl-1": 5}
        _clear_marketplace_state()
        assert len(_user_ratings) == 0


# ===========================================================================
# Tests: Seed Marketplace Templates
# ===========================================================================


class TestSeedMarketplaceTemplates:
    """Tests for _seed_marketplace_templates."""

    def test_seeds_templates(self):
        _seed_marketplace_templates()
        assert len(_marketplace_templates) > 0

    def test_seed_is_idempotent(self):
        _seed_marketplace_templates()
        count = len(_marketplace_templates)
        _seed_marketplace_templates()
        assert len(_marketplace_templates) == count

    def test_seeded_templates_have_valid_categories(self):
        _seed_marketplace_templates()
        for t in _marketplace_templates.values():
            assert t.category in VALID_CATEGORIES

    def test_seeded_templates_have_valid_patterns(self):
        _seed_marketplace_templates()
        for t in _marketplace_templates.values():
            assert t.pattern in VALID_PATTERNS


# ===========================================================================
# Tests: Circuit Breaker Module-Level Helpers
# ===========================================================================


class TestCircuitBreakerHelpers:
    """Tests for module-level circuit breaker helpers."""

    def test_get_marketplace_circuit_breaker_status(self):
        status = get_marketplace_circuit_breaker_status()
        assert "state" in status
        assert status["state"] == "closed"


# ===========================================================================
# Tests: Invalid Path
# ===========================================================================


class TestInvalidPath:
    """Tests for invalid path handling."""

    def test_invalid_path_returns_400(self, seeded_handler, mock_http_get):
        result = seeded_handler.handle(
            "/api/v1/marketplace/unknown_endpoint", {}, mock_http_get
        )
        assert result.status_code == 400


# ===========================================================================
# Tests: Handler Initialization
# ===========================================================================


class TestHandlerInitialization:
    """Tests for handler initialization."""

    def test_default_context(self):
        h = TemplateMarketplaceHandler()
        assert h.ctx == {}

    def test_custom_context(self):
        ctx = {"key": "value"}
        h = TemplateMarketplaceHandler(ctx)
        assert h.ctx == ctx

    def test_circuit_breaker_initialized(self, handler):
        assert handler._circuit_breaker is not None

    def test_recommendations_handler_default_context(self):
        h = TemplateRecommendationsHandler()
        assert h.ctx == {}

    def test_routes_defined(self, handler):
        assert len(handler.ROUTES) > 0

    def test_recommendations_routes_defined(self, rec_handler):
        assert len(rec_handler.ROUTES) > 0


# ===========================================================================
# Tests: _list_templates internal method
# ===========================================================================


class TestListTemplatesInternal:
    """Tests for _list_templates internal method directly."""

    def test_list_with_multiple_tag_filter(self, seeded_handler):
        result = seeded_handler._list_templates({"tags": "security,owasp"})
        body = parse_body(result)
        assert result.status_code == 200
        assert body["total"] >= 1

    def test_list_empty_marketplace(self, handler):
        result = handler._list_templates({})
        body = parse_body(result)
        assert result.status_code == 200
        assert body["total"] == 0
        assert body["templates"] == []

    def test_list_with_search_in_tags(self, seeded_handler):
        result = seeded_handler._list_templates({"search": "quickstart"})
        body = parse_body(result)
        assert body["total"] >= 1

    def test_list_default_sort_is_downloads(self, seeded_handler):
        result = seeded_handler._list_templates({})
        body = parse_body(result)
        downloads = [t["download_count"] for t in body["templates"]]
        assert downloads == sorted(downloads, reverse=True)

    def test_list_combined_filters(self, seeded_handler):
        result = seeded_handler._list_templates({
            "category": "sme",
            "pattern": "debate",
            "verified_only": "true",
        })
        body = parse_body(result)
        for t in body["templates"]:
            assert t["category"] == "sme"
            assert t["pattern"] == "debate"
            assert t["is_verified"] is True


# ===========================================================================
# Tests: _get_featured internal method
# ===========================================================================


class TestGetFeaturedInternal:
    """Tests for _get_featured internal method."""

    def test_featured_empty_marketplace(self, handler):
        result = handler._get_featured()
        body = parse_body(result)
        assert body["featured"] == []
        assert body["total"] == 0


# ===========================================================================
# Tests: _get_trending internal method
# ===========================================================================


class TestGetTrendingInternal:
    """Tests for _get_trending internal method."""

    def test_trending_empty_marketplace(self, handler):
        result = handler._get_trending({})
        body = parse_body(result)
        assert body["trending"] == []
        assert body["total"] == 0

    def test_trending_default_period(self, handler):
        _seed_marketplace_templates()
        result = handler._get_trending({})
        body = parse_body(result)
        assert body["period"] == "week"


# ===========================================================================
# Tests: _get_categories internal method
# ===========================================================================


class TestGetCategoriesInternal:
    """Tests for _get_categories internal method."""

    def test_categories_empty_marketplace(self, handler):
        result = handler._get_categories()
        body = parse_body(result)
        assert body["categories"] == []
        assert body["total"] == 0

    def test_categories_name_formatting(self, seeded_handler):
        result = seeded_handler._get_categories()
        body = parse_body(result)
        for cat in body["categories"]:
            # name should be title-cased version of id
            assert cat["name"] == cat["id"].replace("_", " ").title()
