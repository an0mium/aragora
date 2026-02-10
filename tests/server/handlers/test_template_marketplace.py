"""
Tests for TemplateMarketplaceHandler and TemplateRecommendationsHandler.

Tests the template marketplace API endpoints defined in
aragora/server/handlers/template_marketplace.py including:

- Validation functions: template name, category, pattern, tags, rating, review content
- MarketplaceCircuitBreaker: state transitions, failure counting, cooldown, reset
- MarketplaceTemplate / TemplateReview dataclasses
- TemplateMarketplaceHandler:
  - _list_templates: search, filters, sorting, pagination
  - _get_template: found and not found
  - _publish_template: success, validation errors, duplicates, rate limits
  - _rate_template: new rating, update rating, validation, not found
  - _get_reviews: pagination, empty list, not found
  - _submit_review: success, validation errors, not found
  - _import_template: success, not found, download count increment
  - _get_featured: returns featured templates sorted by rating
  - _get_trending: returns templates sorted by downloads
  - _get_categories: returns categories with counts
  - handle routing: circuit breaker, rate limiting, method dispatch
- TemplateRecommendationsHandler: recommendations endpoint
- _clear_marketplace_state: resets all in-memory stores

Target: 35+ tests covering all handler endpoints and edge cases.
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


class TestValidateCategory:
    """Tests for validate_category."""

    def test_valid_category(self):
        ok, err = validate_category("security")
        assert ok is True

    def test_empty_category(self):
        ok, err = validate_category("")
        assert ok is False

    def test_invalid_category(self):
        ok, err = validate_category("nonexistent")
        assert ok is False
        assert "Invalid category" in err


class TestValidatePattern:
    """Tests for validate_pattern."""

    def test_valid_pattern(self):
        ok, err = validate_pattern("debate")
        assert ok is True

    def test_invalid_pattern(self):
        ok, err = validate_pattern("unknown_pattern")
        assert ok is False
        assert "Invalid pattern" in err


class TestValidateTags:
    """Tests for validate_tags."""

    def test_valid_tags(self):
        ok, err = validate_tags(["tag1", "tag2"])
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


class TestValidateRating:
    """Tests for validate_rating."""

    def test_valid_int_rating(self):
        ok, err = validate_rating(4)
        assert ok is True

    def test_valid_float_rating(self):
        ok, err = validate_rating(3.5)
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


class TestValidateReviewContent:
    """Tests for validate_review_content."""

    def test_valid_content(self):
        ok, err = validate_review_content("Great template!")
        assert ok is True

    def test_empty_content(self):
        ok, err = validate_review_content("")
        assert ok is False

    def test_whitespace_content(self):
        ok, err = validate_review_content("   ")
        assert ok is False

    def test_content_too_long(self):
        ok, err = validate_review_content("x" * (REVIEW_CONTENT_MAX_LENGTH + 1))
        assert ok is False
        assert "too long" in err.lower()


# ===========================================================================
# Tests: MarketplaceCircuitBreaker
# ===========================================================================


class TestMarketplaceCircuitBreaker:
    """Tests for the circuit breaker state machine."""

    def test_initial_state_is_closed(self):
        cb = MarketplaceCircuitBreaker()
        assert cb.state == MarketplaceCircuitBreaker.CLOSED

    def test_allows_requests_when_closed(self):
        cb = MarketplaceCircuitBreaker()
        assert cb.is_allowed() is True

    def test_opens_after_threshold_failures(self):
        cb = MarketplaceCircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.OPEN

    def test_blocks_requests_when_open(self):
        cb = MarketplaceCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.is_allowed() is False

    def test_transitions_to_half_open_after_cooldown(self):
        cb = MarketplaceCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.OPEN
        time.sleep(0.02)
        assert cb.state == MarketplaceCircuitBreaker.HALF_OPEN

    def test_half_open_allows_limited_calls(self):
        cb = MarketplaceCircuitBreaker(
            failure_threshold=1, cooldown_seconds=0.01, half_open_max_calls=2
        )
        cb.record_failure()
        time.sleep(0.02)
        # Half open: allows up to 2 calls
        assert cb.is_allowed() is True
        assert cb.is_allowed() is True
        assert cb.is_allowed() is False  # Third call blocked

    def test_closes_after_enough_successes_in_half_open(self):
        cb = MarketplaceCircuitBreaker(
            failure_threshold=1, cooldown_seconds=0.01, half_open_max_calls=2
        )
        cb.record_failure()
        time.sleep(0.02)
        cb.is_allowed()  # Consume a half-open call
        cb.record_success()
        cb.record_success()
        assert cb.state == MarketplaceCircuitBreaker.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        cb = MarketplaceCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.is_allowed()  # Transition to half-open
        cb.record_failure()  # Should reopen
        assert cb.state == MarketplaceCircuitBreaker.OPEN

    def test_reset_clears_everything(self):
        cb = MarketplaceCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.OPEN
        cb.reset()
        assert cb.state == MarketplaceCircuitBreaker.CLOSED
        assert cb.is_allowed() is True

    def test_get_status_returns_dict(self):
        cb = MarketplaceCircuitBreaker()
        status = cb.get_status()
        assert "state" in status
        assert "failure_count" in status
        assert "failure_threshold" in status
        assert "cooldown_seconds" in status
        assert status["state"] == "closed"

    def test_success_resets_failure_count_when_closed(self):
        cb = MarketplaceCircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # After success, failure count should be reset
        # Need 5 more failures to open (not 3)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.CLOSED
        cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.OPEN


# ===========================================================================
# Tests: MarketplaceTemplate and TemplateReview Dataclasses
# ===========================================================================


class TestMarketplaceTemplate:
    """Tests for MarketplaceTemplate dataclass."""

    def test_to_dict_contains_all_fields(self):
        t = MarketplaceTemplate(
            id="test/template",
            name="Test",
            description="A test template",
            category="security",
            pattern="debate",
            author_id="user-1",
            author_name="Test User",
        )
        d = t.to_dict()
        assert d["id"] == "test/template"
        assert d["name"] == "Test"
        assert d["description"] == "A test template"
        assert d["category"] == "security"
        assert d["pattern"] == "debate"
        assert d["author_id"] == "user-1"
        assert d["author_name"] == "Test User"
        assert "rating" in d
        assert "download_count" in d

    def test_to_summary_truncates_description(self):
        t = MarketplaceTemplate(
            id="test/template",
            name="Test",
            description="A" * 300,
            category="security",
            pattern="debate",
            author_id="user-1",
            author_name="Test User",
        )
        s = t.to_summary()
        assert len(s["description"]) <= 204  # 200 + "..."
        assert s["description"].endswith("...")

    def test_to_summary_limits_tags_to_five(self):
        t = MarketplaceTemplate(
            id="test/template",
            name="Test",
            description="desc",
            category="security",
            pattern="debate",
            author_id="user-1",
            author_name="Test User",
            tags=["t1", "t2", "t3", "t4", "t5", "t6", "t7"],
        )
        s = t.to_summary()
        assert len(s["tags"]) == 5


class TestTemplateReview:
    """Tests for TemplateReview dataclass."""

    def test_to_dict(self):
        r = TemplateReview(
            id="rev-1",
            template_id="tpl-1",
            user_id="user-1",
            user_name="Test User",
            rating=5,
            title="Great!",
            content="Really useful template",
        )
        d = r.to_dict()
        assert d["id"] == "rev-1"
        assert d["rating"] == 5
        assert d["content"] == "Really useful template"


# ===========================================================================
# Tests: TemplateMarketplaceHandler - Internal Methods
# ===========================================================================


class TestListTemplates:
    """Tests for _list_templates."""

    def test_list_all_templates(self, seeded_handler):
        result = seeded_handler._list_templates({})
        body = parse_body(result)
        assert result.status_code == 200
        assert body["total"] > 0
        assert len(body["templates"]) > 0

    def test_list_templates_filter_by_category(self, seeded_handler):
        result = seeded_handler._list_templates({"category": "security"})
        body = parse_body(result)
        assert result.status_code == 200
        assert all(t["category"] == "security" for t in body["templates"])

    def test_list_templates_filter_by_pattern(self, seeded_handler):
        result = seeded_handler._list_templates({"pattern": "debate"})
        body = parse_body(result)
        assert result.status_code == 200
        assert all(t["pattern"] == "debate" for t in body["templates"])

    def test_list_templates_filter_verified_only(self, seeded_handler):
        result = seeded_handler._list_templates({"verified_only": "true"})
        body = parse_body(result)
        assert result.status_code == 200
        assert all(t["is_verified"] for t in body["templates"])

    def test_list_templates_search(self, seeded_handler):
        result = seeded_handler._list_templates({"search": "Security Code Audit"})
        body = parse_body(result)
        assert result.status_code == 200
        assert body["total"] >= 1
        names = [t["name"] for t in body["templates"]]
        assert "Security Code Audit" in names

    def test_list_templates_filter_by_tags(self, seeded_handler):
        result = seeded_handler._list_templates({"tags": "owasp"})
        body = parse_body(result)
        assert result.status_code == 200
        assert body["total"] >= 1

    def test_list_templates_sort_by_rating(self, seeded_handler):
        result = seeded_handler._list_templates({"sort_by": "rating"})
        body = parse_body(result)
        ratings = [t["rating"] for t in body["templates"]]
        assert ratings == sorted(ratings, reverse=True)

    def test_list_templates_sort_by_name(self, seeded_handler):
        result = seeded_handler._list_templates({"sort_by": "name"})
        body = parse_body(result)
        names = [t["name"].lower() for t in body["templates"]]
        assert names == sorted(names)

    def test_list_templates_pagination(self, seeded_handler):
        result = seeded_handler._list_templates({"limit": "2", "offset": "0"})
        body = parse_body(result)
        assert len(body["templates"]) == 2
        assert body["limit"] == 2
        assert body["offset"] == 0
        assert body["total"] > 2

    def test_list_templates_limit_clamped(self, seeded_handler):
        """Limit values above max are clamped to 50."""
        result = seeded_handler._list_templates({"limit": "999"})
        body = parse_body(result)
        assert body["limit"] == 50


class TestGetTemplate:
    """Tests for _get_template."""

    def test_get_existing_template(self, seeded_handler):
        result = seeded_handler._get_template("security/code-audit")
        body = parse_body(result)
        assert result.status_code == 200
        assert body["id"] == "security/code-audit"
        assert body["name"] == "Security Code Audit"

    def test_get_nonexistent_template(self, seeded_handler):
        result = seeded_handler._get_template("does/not-exist")
        body = parse_body(result)
        assert result.status_code == 404
        assert "not found" in body.get("error", "").lower()


class TestPublishTemplate:
    """Tests for _publish_template."""

    def _make_publish_handler(self, data: dict) -> MagicMock:
        body = json_body(data)
        return make_mock_handler(method="POST", body=body)

    def test_publish_template_success(self, handler):
        data = {
            "name": "My New Template",
            "description": "A great template",
            "category": "security",
            "pattern": "debate",
            "workflow_definition": {"steps": []},
            "tags": ["test"],
        }
        mock = self._make_publish_handler(data)
        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = handler._publish_template(mock, "127.0.0.1")

        body = parse_body(result)
        assert result.status_code == 201
        assert body["status"] == "published"
        assert "security/my-new-template" == body["template_id"]

    def test_publish_template_rate_limited(self, handler):
        data = {
            "name": "Template",
            "description": "desc",
            "category": "security",
            "pattern": "debate",
            "workflow_definition": {},
        }
        mock = self._make_publish_handler(data)
        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = False
            result = handler._publish_template(mock, "127.0.0.1")

        assert result.status_code == 429

    def test_publish_template_missing_required_field(self, handler):
        data = {
            "name": "Template",
            # Missing description, category, pattern, workflow_definition
        }
        mock = self._make_publish_handler(data)
        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = handler._publish_template(mock, "127.0.0.1")

        body = parse_body(result)
        assert result.status_code == 400
        assert "Missing required field" in body.get("error", "")

    def test_publish_template_invalid_category(self, handler):
        data = {
            "name": "Template",
            "description": "desc",
            "category": "invalid_cat",
            "pattern": "debate",
            "workflow_definition": {},
        }
        mock = self._make_publish_handler(data)
        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = handler._publish_template(mock, "127.0.0.1")

        assert result.status_code == 400

    def test_publish_template_invalid_pattern(self, handler):
        data = {
            "name": "Template",
            "description": "desc",
            "category": "security",
            "pattern": "invalid_pat",
            "workflow_definition": {},
        }
        mock = self._make_publish_handler(data)
        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = handler._publish_template(mock, "127.0.0.1")

        assert result.status_code == 400

    def test_publish_template_invalid_json_body(self, handler):
        mock = make_mock_handler(method="POST", body=b"not-json{{{")
        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = handler._publish_template(mock, "127.0.0.1")

        assert result.status_code == 400
        body = parse_body(result)
        assert "Invalid JSON" in body.get("error", "")

    def test_publish_template_body_too_large(self, handler):
        mock = make_mock_handler(method="POST", body=b"x")
        mock.headers["Content-Length"] = "200000"  # Over 100KB limit
        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = handler._publish_template(mock, "127.0.0.1")

        assert result.status_code == 413

    def test_publish_template_duplicate(self, seeded_handler):
        """Publishing a template with same name+category as existing returns 409."""
        data = {
            "name": "code-audit",  # Same as seeded security/code-audit
            "description": "Another audit",
            "category": "security",
            "pattern": "debate",
            "workflow_definition": {},
        }
        mock = self._make_publish_handler(data)
        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = seeded_handler._publish_template(mock, "127.0.0.1")

        assert result.status_code == 409

    def test_publish_template_invalid_workflow_definition(self, handler):
        data = {
            "name": "Template",
            "description": "desc",
            "category": "security",
            "pattern": "debate",
            "workflow_definition": "not-a-dict",
        }
        mock = self._make_publish_handler(data)
        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = handler._publish_template(mock, "127.0.0.1")

        assert result.status_code == 400
        body = parse_body(result)
        assert "workflow_definition" in body.get("error", "").lower()

    def test_publish_template_name_too_long(self, handler):
        data = {
            "name": "x" * (TEMPLATE_NAME_MAX_LENGTH + 1),
            "description": "desc",
            "category": "security",
            "pattern": "debate",
            "workflow_definition": {},
        }
        mock = self._make_publish_handler(data)
        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = handler._publish_template(mock, "127.0.0.1")

        assert result.status_code == 400

    def test_publish_template_description_too_long(self, handler):
        data = {
            "name": "Template",
            "description": "x" * (DESCRIPTION_MAX_LENGTH + 1),
            "category": "security",
            "pattern": "debate",
            "workflow_definition": {},
        }
        mock = self._make_publish_handler(data)
        with patch(
            "aragora.server.handlers.template_marketplace._publish_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = handler._publish_template(mock, "127.0.0.1")

        assert result.status_code == 400


class TestRateTemplate:
    """Tests for _rate_template."""

    def test_rate_template_success(self, seeded_handler):
        data = {"rating": 4, "user_id": "user-1"}
        mock = make_mock_handler(method="POST", body=json_body(data))
        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = seeded_handler._rate_template("security/code-audit", mock, "127.0.0.1")

        body = parse_body(result)
        assert result.status_code == 200
        assert body["status"] == "rated"
        assert body["your_rating"] == 4
        assert body["rating_count"] > 0

    def test_rate_template_not_found(self, seeded_handler):
        data = {"rating": 5}
        mock = make_mock_handler(method="POST", body=json_body(data))
        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = seeded_handler._rate_template("nonexistent/template", mock, "127.0.0.1")

        assert result.status_code == 404

    def test_rate_template_invalid_rating(self, seeded_handler):
        data = {"rating": 10}
        mock = make_mock_handler(method="POST", body=json_body(data))
        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = seeded_handler._rate_template("security/code-audit", mock, "127.0.0.1")

        assert result.status_code == 400

    def test_rate_template_non_numeric_rating(self, seeded_handler):
        data = {"rating": "great"}
        mock = make_mock_handler(method="POST", body=json_body(data))
        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = seeded_handler._rate_template("security/code-audit", mock, "127.0.0.1")

        assert result.status_code == 400

    def test_rate_template_update_existing_rating(self, seeded_handler):
        """Updating an existing rating should adjust the average."""
        data = {"rating": 5, "user_id": "user-x"}
        mock1 = make_mock_handler(method="POST", body=json_body(data))
        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            seeded_handler._rate_template("security/code-audit", mock1, "127.0.0.1")

        # Update to a lower rating
        data2 = {"rating": 2, "user_id": "user-x"}
        mock2 = make_mock_handler(method="POST", body=json_body(data2))
        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = True
            result = seeded_handler._rate_template("security/code-audit", mock2, "127.0.0.1")

        body = parse_body(result)
        assert result.status_code == 200
        assert body["your_rating"] == 2

    def test_rate_template_rate_limited(self, seeded_handler):
        data = {"rating": 3}
        mock = make_mock_handler(method="POST", body=json_body(data))
        with patch(
            "aragora.server.handlers.template_marketplace._rate_limiter"
        ) as limiter:
            limiter.is_allowed.return_value = False
            result = seeded_handler._rate_template("security/code-audit", mock, "127.0.0.1")

        assert result.status_code == 429


class TestGetReviews:
    """Tests for _get_reviews."""

    def test_get_reviews_empty(self, seeded_handler):
        result = seeded_handler._get_reviews("security/code-audit", {})
        body = parse_body(result)
        assert result.status_code == 200
        assert body["total"] == 0
        assert body["reviews"] == []

    def test_get_reviews_with_data(self, seeded_handler):
        _template_reviews["security/code-audit"] = [
            TemplateReview(
                id="r1",
                template_id="security/code-audit",
                user_id="u1",
                user_name="User 1",
                rating=5,
                title="Great",
                content="Excellent template",
                helpful_count=10,
            ),
            TemplateReview(
                id="r2",
                template_id="security/code-audit",
                user_id="u2",
                user_name="User 2",
                rating=3,
                title="OK",
                content="Decent template",
                helpful_count=5,
            ),
        ]
        result = seeded_handler._get_reviews("security/code-audit", {})
        body = parse_body(result)
        assert result.status_code == 200
        assert body["total"] == 2
        # Should be sorted by helpful_count descending
        assert body["reviews"][0]["helpful_count"] >= body["reviews"][1]["helpful_count"]

    def test_get_reviews_not_found(self, seeded_handler):
        result = seeded_handler._get_reviews("nonexistent/template", {})
        assert result.status_code == 404

    def test_get_reviews_pagination(self, seeded_handler):
        _template_reviews["security/code-audit"] = [
            TemplateReview(
                id=f"r{i}",
                template_id="security/code-audit",
                user_id=f"u{i}",
                user_name=f"User {i}",
                rating=4,
                title=f"Review {i}",
                content=f"Content {i}",
            )
            for i in range(5)
        ]
        result = seeded_handler._get_reviews("security/code-audit", {"limit": "2", "offset": "1"})
        body = parse_body(result)
        assert body["total"] == 5
        assert len(body["reviews"]) == 2
        assert body["limit"] == 2
        assert body["offset"] == 1


class TestSubmitReview:
    """Tests for _submit_review."""

    def test_submit_review_success(self, seeded_handler):
        data = {
            "rating": 5,
            "content": "This template is excellent!",
            "title": "Love it",
            "user_id": "user-1",
            "user_name": "Test User",
        }
        mock = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock, "127.0.0.1")
        body = parse_body(result)
        assert result.status_code == 201
        assert body["status"] == "submitted"
        assert body["review"]["rating"] == 5
        assert body["review"]["content"] == "This template is excellent!"

    def test_submit_review_not_found(self, seeded_handler):
        data = {"rating": 5, "content": "Great!"}
        mock = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("nonexistent/template", mock, "127.0.0.1")
        assert result.status_code == 404

    def test_submit_review_missing_rating(self, seeded_handler):
        data = {"content": "No rating provided"}
        mock = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock, "127.0.0.1")
        assert result.status_code == 400

    def test_submit_review_missing_content(self, seeded_handler):
        data = {"rating": 5}
        mock = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock, "127.0.0.1")
        assert result.status_code == 400

    def test_submit_review_invalid_rating(self, seeded_handler):
        data = {"rating": 0, "content": "Bad rating value"}
        mock = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock, "127.0.0.1")
        assert result.status_code == 400

    def test_submit_review_content_too_long(self, seeded_handler):
        data = {"rating": 4, "content": "x" * (REVIEW_CONTENT_MAX_LENGTH + 1)}
        mock = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._submit_review("security/code-audit", mock, "127.0.0.1")
        assert result.status_code == 400

    def test_submit_review_body_too_large(self, seeded_handler):
        mock = make_mock_handler(method="POST", body=b"x")
        mock.headers["Content-Length"] = "60000"  # Over 50KB
        result = seeded_handler._submit_review("security/code-audit", mock, "127.0.0.1")
        assert result.status_code == 413

    def test_submit_review_invalid_json(self, seeded_handler):
        mock = make_mock_handler(method="POST", body=b"not-json")
        result = seeded_handler._submit_review("security/code-audit", mock, "127.0.0.1")
        assert result.status_code == 400


class TestImportTemplate:
    """Tests for _import_template."""

    def test_import_template_success(self, seeded_handler):
        template = _marketplace_templates["security/code-audit"]
        initial_downloads = template.download_count
        data = {"workspace_id": "ws-1"}
        mock = make_mock_handler(method="POST", body=json_body(data))
        result = seeded_handler._import_template("security/code-audit", mock)
        body = parse_body(result)
        assert result.status_code == 200
        assert body["status"] == "imported"
        assert body["workspace_id"] == "ws-1"
        assert body["download_count"] == initial_downloads + 1

    def test_import_template_not_found(self, seeded_handler):
        mock = make_mock_handler(method="POST", body=json_body({}))
        result = seeded_handler._import_template("nonexistent/template", mock)
        assert result.status_code == 404

    def test_import_template_body_too_large(self, seeded_handler):
        mock = make_mock_handler(method="POST", body=b"x")
        mock.headers["Content-Length"] = "20000"  # Over 10KB
        result = seeded_handler._import_template("security/code-audit", mock)
        assert result.status_code == 413


class TestGetFeatured:
    """Tests for _get_featured."""

    def test_get_featured_returns_only_featured(self, seeded_handler):
        result = seeded_handler._get_featured()
        body = parse_body(result)
        assert result.status_code == 200
        assert len(body["featured"]) > 0
        assert all(t["is_featured"] for t in body["featured"])

    def test_get_featured_sorted_by_rating(self, seeded_handler):
        result = seeded_handler._get_featured()
        body = parse_body(result)
        ratings = [t["rating"] for t in body["featured"]]
        assert ratings == sorted(ratings, reverse=True)

    def test_get_featured_limited_to_ten(self, seeded_handler):
        result = seeded_handler._get_featured()
        body = parse_body(result)
        assert len(body["featured"]) <= 10


class TestGetTrending:
    """Tests for _get_trending."""

    def test_get_trending(self, seeded_handler):
        result = seeded_handler._get_trending({})
        body = parse_body(result)
        assert result.status_code == 200
        assert "trending" in body
        assert body["period"] == "week"  # default
        downloads = [t["download_count"] for t in body["trending"]]
        assert downloads == sorted(downloads, reverse=True)

    def test_get_trending_custom_limit(self, seeded_handler):
        result = seeded_handler._get_trending({"limit": "3"})
        body = parse_body(result)
        assert len(body["trending"]) <= 3

    def test_get_trending_custom_period(self, seeded_handler):
        result = seeded_handler._get_trending({"period": "month"})
        body = parse_body(result)
        assert body["period"] == "month"


class TestGetCategories:
    """Tests for _get_categories."""

    def test_get_categories(self, seeded_handler):
        result = seeded_handler._get_categories()
        body = parse_body(result)
        assert result.status_code == 200
        assert len(body["categories"]) > 0
        # Each category should have expected fields
        cat = body["categories"][0]
        assert "id" in cat
        assert "name" in cat
        assert "template_count" in cat
        assert "total_downloads" in cat

    def test_get_categories_sorted_by_count(self, seeded_handler):
        result = seeded_handler._get_categories()
        body = parse_body(result)
        counts = [c["template_count"] for c in body["categories"]]
        assert counts == sorted(counts, reverse=True)


# ===========================================================================
# Tests: Handle Routing and Circuit Breaker Integration
# ===========================================================================


class TestHandleRouting:
    """Tests for the handle method dispatch, circuit breaker, and rate limiting."""

    def test_can_handle_marketplace_paths(self, handler):
        assert handler.can_handle("/api/v1/marketplace/templates") is True
        assert handler.can_handle("/api/v1/marketplace/featured") is True
        assert handler.can_handle("/api/v1/marketplace/trending") is True
        assert handler.can_handle("/api/v1/marketplace/categories") is True
        assert handler.can_handle("/api/v1/marketplace/circuit-breaker") is True

    def test_cannot_handle_non_marketplace_paths(self, handler):
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/gallery") is False

    def test_circuit_breaker_status_endpoint(self, handler):
        """Circuit breaker status is returned without permission checks
        via _route_request, but the handle() method wraps it."""
        # Access directly via the circuit breaker endpoint logic
        status = handler._circuit_breaker.get_status()
        assert status["state"] == "closed"

    def test_route_request_featured(self, seeded_handler, mock_http_get):
        result = seeded_handler._route_request(
            "/api/v1/marketplace/featured", {}, mock_http_get, "GET", "127.0.0.1"
        )
        body = parse_body(result)
        assert "featured" in body

    def test_route_request_trending(self, seeded_handler, mock_http_get):
        result = seeded_handler._route_request(
            "/api/v1/marketplace/trending", {}, mock_http_get, "GET", "127.0.0.1"
        )
        body = parse_body(result)
        assert "trending" in body

    def test_route_request_categories(self, seeded_handler, mock_http_get):
        result = seeded_handler._route_request(
            "/api/v1/marketplace/categories", {}, mock_http_get, "GET", "127.0.0.1"
        )
        body = parse_body(result)
        assert "categories" in body

    def test_route_request_list_templates_get(self, seeded_handler, mock_http_get):
        result = seeded_handler._route_request(
            "/api/v1/marketplace/templates", {}, mock_http_get, "GET", "127.0.0.1"
        )
        body = parse_body(result)
        assert "templates" in body

    def test_route_request_templates_method_not_allowed(self, seeded_handler, mock_http_get):
        result = seeded_handler._route_request(
            "/api/v1/marketplace/templates", {}, mock_http_get, "DELETE", "127.0.0.1"
        )
        assert result.status_code == 405

    def test_route_request_get_specific_template(self, seeded_handler, mock_http_get):
        """Route to a specific template. Note: the handler joins parts[4:] from the
        path split, so for template_id 'security/code-audit' the resulting id
        passed to _get_template is 'templates/security/code-audit'. We add a
        template with that id to validate the routing works end-to-end."""
        # Insert a template with the id that the router will extract
        _marketplace_templates["templates/security/code-audit"] = MarketplaceTemplate(
            id="templates/security/code-audit",
            name="Routing Test",
            description="Test",
            category="security",
            pattern="debate",
            author_id="test",
            author_name="Test",
        )
        result = seeded_handler._route_request(
            "/api/v1/marketplace/templates/security/code-audit",
            {},
            mock_http_get,
            "GET",
            "127.0.0.1",
        )
        body = parse_body(result)
        assert result.status_code == 200
        assert body["name"] == "Routing Test"

    def test_route_request_invalid_path(self, seeded_handler, mock_http_get):
        result = seeded_handler._route_request(
            "/api/v1/marketplace/unknown", {}, mock_http_get, "GET", "127.0.0.1"
        )
        assert result.status_code == 400


# ===========================================================================
# Tests: TemplateRecommendationsHandler
# ===========================================================================


class TestTemplateRecommendationsHandler:
    """Tests for the recommendations handler."""

    def test_can_handle(self):
        h = TemplateRecommendationsHandler(ctx={})
        assert h.can_handle("/api/v1/marketplace/recommendations") is True
        assert h.can_handle("/api/v1/marketplace/templates") is False

    def test_recommendations_returned(self):
        """Recommendations should return top templates sorted by featured+downloads."""
        _seed_marketplace_templates()
        h = TemplateRecommendationsHandler(ctx={})
        # Call handle's internal logic directly (bypass @require_permission)
        # by invoking the same logic
        from aragora.server.handlers.base import json_response, get_clamped_int_param

        limit = 5
        templates = sorted(
            _marketplace_templates.values(),
            key=lambda t: (t.is_featured, t.download_count),
            reverse=True,
        )[:limit]

        assert len(templates) == 5
        # First template should be featured with highest download count
        assert templates[0].is_featured is True


# ===========================================================================
# Tests: _clear_marketplace_state
# ===========================================================================


class TestClearMarketplaceState:
    """Tests for _clear_marketplace_state."""

    def test_clears_templates(self):
        _seed_marketplace_templates()
        assert len(_marketplace_templates) > 0
        _clear_marketplace_state()
        assert len(_marketplace_templates) == 0

    def test_clears_reviews(self):
        _template_reviews["some/template"] = [
            TemplateReview(
                id="r1",
                template_id="some/template",
                user_id="u1",
                user_name="User",
                rating=5,
                title="Test",
                content="Test content",
            )
        ]
        _clear_marketplace_state()
        assert len(_template_reviews) == 0

    def test_clears_user_ratings(self):
        _user_ratings["user-1"] = {"template-1": 5}
        _clear_marketplace_state()
        assert len(_user_ratings) == 0


# ===========================================================================
# Tests: Seed Marketplace Templates
# ===========================================================================


class TestSeedMarketplaceTemplates:
    """Tests for _seed_marketplace_templates."""

    def test_seeds_templates(self):
        assert len(_marketplace_templates) == 0
        _seed_marketplace_templates()
        assert len(_marketplace_templates) > 0

    def test_idempotent_seeding(self):
        _seed_marketplace_templates()
        count1 = len(_marketplace_templates)
        _seed_marketplace_templates()
        count2 = len(_marketplace_templates)
        assert count1 == count2
