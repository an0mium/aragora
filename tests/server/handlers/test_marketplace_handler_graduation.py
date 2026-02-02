"""
Tests for MarketplaceHandler graduation from EXPERIMENTAL to STABLE.

This test file provides comprehensive coverage for the upgraded MarketplaceHandler
at aragora/server/handlers/marketplace.py, targeting 80%+ overall coverage.

Areas covered:
- Circuit breaker pattern for registry access resilience
- Rate limiting on all endpoints
- Comprehensive input validation (IDs, pagination, ratings, review text)
- RBAC permission enforcement
- Error handling and circuit breaker failure recording
- Status/health endpoint
"""

from __future__ import annotations

import json
import sys
import time
import types as _types_mod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Slack stubs to prevent transitive import issues
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

from aragora.server.handlers.marketplace import (
    MarketplaceHandler,
    MarketplaceCircuitBreaker,
    _validate_template_id,
    _validate_pagination,
    _validate_rating,
    _validate_review,
    _validate_query,
    _validate_tags,
    _get_circuit_breaker,
    get_marketplace_circuit_breaker,
    get_marketplace_circuit_breaker_status,
    reset_marketplace_circuit_breaker,
    _clear_registry,
    SAFE_ID_PATTERN,
    MAX_TEMPLATE_ID_LENGTH,
    MAX_QUERY_LENGTH,
    MAX_TAGS_LENGTH,
    MAX_REVIEW_LENGTH,
    MIN_RATING,
    MAX_RATING,
    DEFAULT_LIMIT,
    MIN_LIMIT,
    MAX_LIMIT,
    MAX_OFFSET,
)


# ===========================================================================
# Mocks
# ===========================================================================


@dataclass
class MockTemplateMetadata:
    """Mock template metadata."""

    stars: int = 5
    downloads: int = 42


@dataclass
class MockTemplate:
    """Mock marketplace template."""

    id: str = "tpl-001"
    name: str = "Test Template"
    category: str = "debate"
    template_type: str = "workflow"
    tags: list = field(default_factory=lambda: ["test", "example"])
    metadata: MockTemplateMetadata = field(default_factory=MockTemplateMetadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "type": self.template_type,
            "tags": self.tags,
            "stars": self.metadata.stars,
            "downloads": self.metadata.downloads,
        }


@dataclass
class MockRating:
    """Mock template rating."""

    user_id: str = "user-123"
    template_id: str = "tpl-001"
    score: int = 4
    review: str = "Great template!"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MockRegistry:
    """Mock TemplateRegistry for testing."""

    def __init__(self):
        self._templates: dict[str, MockTemplate] = {}
        self._ratings: dict[str, list[MockRating]] = {}
        self._downloads: dict[str, int] = {}

    def search(
        self,
        query: str | None = None,
        category: Any = None,
        template_type: str | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MockTemplate]:
        results = list(self._templates.values())
        return results[offset : offset + limit]

    def get(self, template_id: str) -> MockTemplate | None:
        return self._templates.get(template_id)

    def import_template(self, json_str: str) -> str:
        data = json.loads(json_str)
        tpl_id = data.get("id", "tpl-new")
        self._templates[tpl_id] = MockTemplate(id=tpl_id, name=data.get("name", "Imported"))
        return tpl_id

    def delete(self, template_id: str) -> bool:
        if template_id in self._templates:
            del self._templates[template_id]
            return True
        return False

    def rate(self, rating: Any) -> None:
        tid = rating.template_id
        if tid not in self._ratings:
            self._ratings[tid] = []
        self._ratings[tid].append(rating)

    def get_ratings(self, template_id: str) -> list[MockRating]:
        return self._ratings.get(template_id, [])

    def get_average_rating(self, template_id: str) -> float:
        ratings = self._ratings.get(template_id, [])
        if not ratings:
            return 0.0
        return sum(r.score for r in ratings) / len(ratings)

    def star(self, template_id: str) -> None:
        tpl = self._templates.get(template_id)
        if tpl:
            tpl.metadata.stars += 1

    def increment_downloads(self, template_id: str) -> None:
        self._downloads[template_id] = self._downloads.get(template_id, 0) + 1

    def list_categories(self) -> list[str]:
        return ["debate", "workflow", "analysis", "compliance"]

    def export_template(self, template_id: str) -> str | None:
        tpl = self._templates.get(template_id)
        if tpl is None:
            return None
        return json.dumps(tpl.to_dict())


@dataclass
class MockAuthUser:
    """Mock authenticated user."""

    id: str = "user-123"
    email: str = "test@example.com"


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def clear_state():
    """Clear state before and after each test."""
    reset_marketplace_circuit_breaker()
    _clear_registry()
    yield
    reset_marketplace_circuit_breaker()
    _clear_registry()


@pytest.fixture
def registry():
    """Create a mock template registry."""
    r = MockRegistry()
    r._templates["tpl-001"] = MockTemplate(id="tpl-001", name="Debate Flow")
    r._templates["tpl-002"] = MockTemplate(id="tpl-002", name="Compliance Check")
    r._templates["tpl-003"] = MockTemplate(id="tpl-003", name="Risk Analysis")
    return r


@pytest.fixture
def handler(registry):
    """Create a MarketplaceHandler with mocked dependencies."""
    h = MarketplaceHandler({})
    h._current_query_params = {}
    return h


def get_body(result) -> dict:
    """Extract JSON body from a HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


# ===========================================================================
# Tests: Input Validation
# ===========================================================================


class TestInputValidation:
    """Tests for input validation functions."""

    def test_validate_template_id_valid(self):
        """Valid template IDs should pass validation."""
        valid_ids = ["tpl-001", "valid_id", "abc123", "ABC-def_123"]
        for tid in valid_ids:
            is_valid, err = _validate_template_id(tid)
            assert is_valid is True, f"Expected {tid} to be valid"
            assert err == ""

    def test_validate_template_id_empty(self):
        """Empty template ID should fail validation."""
        is_valid, err = _validate_template_id("")
        assert is_valid is False
        assert "required" in err.lower()

    def test_validate_template_id_too_long(self):
        """Too long template ID should fail validation."""
        long_id = "a" * (MAX_TEMPLATE_ID_LENGTH + 1)
        is_valid, err = _validate_template_id(long_id)
        assert is_valid is False
        assert str(MAX_TEMPLATE_ID_LENGTH) in err

    def test_validate_template_id_invalid_chars(self):
        """Template IDs with invalid characters should fail validation."""
        invalid_ids = ["has space", "has/slash", "../evil", "has.dot", "has@symbol"]
        for tid in invalid_ids:
            is_valid, err = _validate_template_id(tid)
            assert is_valid is False, f"Expected {tid} to be invalid"
            assert "invalid" in err.lower()

    def test_validate_template_id_none(self):
        """None template ID should fail validation."""
        is_valid, err = _validate_template_id(None)
        assert is_valid is False
        assert "required" in err.lower()

    def test_validate_pagination_defaults(self):
        """Default pagination should work correctly."""
        limit, offset, err = _validate_pagination({})
        assert limit == DEFAULT_LIMIT
        assert offset == 0
        assert err == ""

    def test_validate_pagination_valid(self):
        """Valid pagination parameters should be accepted."""
        limit, offset, err = _validate_pagination({"limit": "25", "offset": "10"})
        assert limit == 25
        assert offset == 10
        assert err == ""

    def test_validate_pagination_clamped(self):
        """Out of range pagination should be clamped."""
        limit, offset, err = _validate_pagination({"limit": "9999", "offset": "-5"})
        assert limit == MAX_LIMIT
        assert offset == 0
        assert err == ""

    def test_validate_pagination_invalid_limit(self):
        """Invalid limit should return error."""
        limit, offset, err = _validate_pagination({"limit": "not-a-number"})
        assert "integer" in err.lower()

    def test_validate_pagination_invalid_offset(self):
        """Invalid offset should return error."""
        limit, offset, err = _validate_pagination({"offset": "not-a-number"})
        assert "integer" in err.lower()

    def test_validate_rating_valid(self):
        """Valid ratings should pass validation."""
        for value in [1, 2, 3, 4, 5]:
            is_valid, rating, err = _validate_rating(value)
            assert is_valid is True
            assert rating == value
            assert err == ""

    def test_validate_rating_none(self):
        """None rating should fail validation."""
        is_valid, rating, err = _validate_rating(None)
        assert is_valid is False
        assert "required" in err.lower()

    def test_validate_rating_out_of_range(self):
        """Out of range ratings should fail validation."""
        for value in [0, 6, -1, 10]:
            is_valid, rating, err = _validate_rating(value)
            assert is_valid is False
            assert str(MIN_RATING) in err or str(MAX_RATING) in err

    def test_validate_rating_not_integer(self):
        """Non-integer ratings should fail validation."""
        is_valid, rating, err = _validate_rating("five")
        assert is_valid is False
        assert "integer" in err.lower()

    def test_validate_review_valid(self):
        """Valid reviews should pass validation."""
        is_valid, review, err = _validate_review("Great template!")
        assert is_valid is True
        assert review is not None
        assert err == ""

    def test_validate_review_none(self):
        """None review should pass validation."""
        is_valid, review, err = _validate_review(None)
        assert is_valid is True
        assert review is None
        assert err == ""

    def test_validate_review_too_long(self):
        """Too long review should fail validation."""
        long_review = "x" * (MAX_REVIEW_LENGTH + 1)
        is_valid, review, err = _validate_review(long_review)
        assert is_valid is False
        assert str(MAX_REVIEW_LENGTH) in err

    def test_validate_review_not_string(self):
        """Non-string review should fail validation."""
        is_valid, review, err = _validate_review(12345)
        assert is_valid is False
        assert "string" in err.lower()

    def test_validate_query_valid(self):
        """Valid queries should pass validation."""
        is_valid, query, err = _validate_query("search term")
        assert is_valid is True
        assert query is not None
        assert err == ""

    def test_validate_query_empty(self):
        """Empty query should pass validation."""
        is_valid, query, err = _validate_query("")
        assert is_valid is True
        assert query == ""
        assert err == ""

    def test_validate_query_none(self):
        """None query should pass validation."""
        is_valid, query, err = _validate_query(None)
        assert is_valid is True
        assert query == ""
        assert err == ""

    def test_validate_query_too_long(self):
        """Too long query should fail validation."""
        long_query = "x" * (MAX_QUERY_LENGTH + 1)
        is_valid, query, err = _validate_query(long_query)
        assert is_valid is False
        assert str(MAX_QUERY_LENGTH) in err

    def test_validate_query_not_string(self):
        """Non-string query should fail validation."""
        is_valid, query, err = _validate_query(12345)
        assert is_valid is False
        assert "string" in err.lower()

    def test_validate_tags_valid(self):
        """Valid tags should pass validation."""
        is_valid, tags, err = _validate_tags("python,automation,testing")
        assert is_valid is True
        assert len(tags) == 3
        assert err == ""

    def test_validate_tags_empty(self):
        """Empty tags should pass validation."""
        is_valid, tags, err = _validate_tags("")
        assert is_valid is True
        assert tags == []
        assert err == ""

    def test_validate_tags_none(self):
        """None tags should pass validation."""
        is_valid, tags, err = _validate_tags(None)
        assert is_valid is True
        assert tags == []
        assert err == ""

    def test_validate_tags_too_long(self):
        """Too long tags string should fail validation."""
        long_tags = "x" * (MAX_TAGS_LENGTH + 1)
        is_valid, tags, err = _validate_tags(long_tags)
        assert is_valid is False
        assert str(MAX_TAGS_LENGTH) in err

    def test_validate_tags_not_string(self):
        """Non-string tags should fail validation."""
        is_valid, tags, err = _validate_tags(["python", "automation"])
        assert is_valid is False
        assert "string" in err.lower()


# ===========================================================================
# Tests: Circuit Breaker
# ===========================================================================


class TestCircuitBreaker:
    """Tests for MarketplaceCircuitBreaker."""

    def test_circuit_breaker_initial_state(self):
        """Circuit breaker should start in CLOSED state."""
        cb = MarketplaceCircuitBreaker()
        assert cb.state == MarketplaceCircuitBreaker.CLOSED
        assert cb.can_proceed() is True

    def test_circuit_breaker_opens_after_failures(self):
        """Circuit breaker should open after failure threshold."""
        cb = MarketplaceCircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.OPEN
        assert cb.can_proceed() is False

    def test_circuit_breaker_transitions_to_half_open(self):
        """Circuit breaker should transition to HALF_OPEN after cooldown."""
        cb = MarketplaceCircuitBreaker(failure_threshold=1, cooldown_seconds=0.1)
        cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.OPEN

        time.sleep(0.15)
        assert cb.state == MarketplaceCircuitBreaker.HALF_OPEN
        assert cb.can_proceed() is True

    def test_circuit_breaker_closes_after_success_in_half_open(self):
        """Circuit breaker should close after successful calls in HALF_OPEN."""
        cb = MarketplaceCircuitBreaker(
            failure_threshold=1, cooldown_seconds=0.1, half_open_max_calls=2
        )
        cb.record_failure()
        time.sleep(0.15)

        # Allow test calls
        assert cb.can_proceed() is True
        cb.record_success()
        assert cb.can_proceed() is True
        cb.record_success()

        assert cb.state == MarketplaceCircuitBreaker.CLOSED

    def test_circuit_breaker_reopens_on_failure_in_half_open(self):
        """Circuit breaker should reopen on failure in HALF_OPEN."""
        cb = MarketplaceCircuitBreaker(failure_threshold=1, cooldown_seconds=0.1)
        cb.record_failure()
        time.sleep(0.15)

        assert cb.state == MarketplaceCircuitBreaker.HALF_OPEN
        cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.OPEN

    def test_circuit_breaker_half_open_limits_calls(self):
        """Circuit breaker should limit calls in HALF_OPEN."""
        cb = MarketplaceCircuitBreaker(
            failure_threshold=1, cooldown_seconds=0.1, half_open_max_calls=2
        )
        cb.record_failure()
        time.sleep(0.15)

        assert cb.can_proceed() is True  # 1st call
        assert cb.can_proceed() is True  # 2nd call
        assert cb.can_proceed() is False  # 3rd call should be blocked

    def test_circuit_breaker_reset(self):
        """Circuit breaker reset should return to CLOSED state."""
        cb = MarketplaceCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.OPEN

        cb.reset()
        assert cb.state == MarketplaceCircuitBreaker.CLOSED
        assert cb.get_status()["failure_count"] == 0

    def test_circuit_breaker_status(self):
        """Circuit breaker status should include all relevant info."""
        cb = MarketplaceCircuitBreaker(failure_threshold=5, cooldown_seconds=30.0)
        cb.record_failure()

        status = cb.get_status()
        assert status["state"] == MarketplaceCircuitBreaker.CLOSED
        assert status["failure_count"] == 1
        assert status["failure_threshold"] == 5
        assert status["cooldown_seconds"] == 30.0
        assert status["last_failure_time"] is not None

    def test_global_circuit_breaker(self):
        """Global circuit breaker should be accessible."""
        cb = get_marketplace_circuit_breaker()
        assert cb is not None
        assert isinstance(cb, MarketplaceCircuitBreaker)

    def test_global_circuit_breaker_status(self):
        """Global circuit breaker status should be accessible."""
        status = get_marketplace_circuit_breaker_status()
        assert "state" in status
        assert "failure_count" in status

    def test_is_allowed_alias(self):
        """is_allowed should be an alias for can_proceed."""
        cb = MarketplaceCircuitBreaker()
        assert cb.is_allowed() == cb.can_proceed()


# ===========================================================================
# Tests: Handler Methods with Circuit Breaker
# ===========================================================================


class TestHandlerWithCircuitBreaker:
    """Tests for handler methods with circuit breaker integration."""

    def test_list_templates_circuit_breaker_open(self, handler, registry):
        """List templates should return 503 when circuit breaker is open."""
        for _ in range(5):
            handler._circuit_breaker.record_failure()

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_list_templates()

        assert result.status_code == HTTPStatus.SERVICE_UNAVAILABLE

    def test_get_template_circuit_breaker_open(self, handler, registry):
        """Get template should return 503 when circuit breaker is open."""
        for _ in range(5):
            handler._circuit_breaker.record_failure()

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_get_template("tpl-001")

        assert result.status_code == HTTPStatus.SERVICE_UNAVAILABLE

    def test_list_templates_success_records_success(self, handler, registry):
        """Successful list templates should record circuit breaker success."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_list_templates()

        assert result.status_code == HTTPStatus.OK
        assert handler._circuit_breaker.get_status()["failure_count"] == 0

    def test_list_templates_exception_records_failure(self, handler):
        """Exception during list templates should record circuit breaker failure."""
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            side_effect=RuntimeError("DB down"),
        ):
            result = handler.handle_list_templates()

        assert result.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert handler._circuit_breaker.get_status()["failure_count"] == 1


# ===========================================================================
# Tests: Handler Methods with Rate Limiting
# ===========================================================================


class TestHandlerWithRateLimiting:
    """Tests for handler methods with rate limiting."""

    def test_list_templates_has_rate_limit_decorator(self, handler):
        """List templates should have rate limit decorator."""
        assert hasattr(handler.handle_list_templates, "_rate_limited")

    def test_get_template_has_rate_limit_decorator(self, handler):
        """Get template should have rate limit decorator."""
        assert hasattr(handler.handle_get_template, "_rate_limited")

    def test_create_template_has_rate_limit_decorator(self, handler):
        """Create template should have rate limit decorator."""
        assert hasattr(handler.handle_create_template, "_rate_limited")

    def test_delete_template_has_rate_limit_decorator(self, handler):
        """Delete template should have rate limit decorator."""
        assert hasattr(handler.handle_delete_template, "_rate_limited")

    def test_rate_template_has_rate_limit_decorator(self, handler):
        """Rate template should have rate limit decorator."""
        assert hasattr(handler.handle_rate_template, "_rate_limited")

    def test_star_template_has_rate_limit_decorator(self, handler):
        """Star template should have rate limit decorator."""
        assert hasattr(handler.handle_star_template, "_rate_limited")

    def test_export_template_has_rate_limit_decorator(self, handler):
        """Export template should have rate limit decorator."""
        assert hasattr(handler.handle_export_template, "_rate_limited")

    def test_import_template_has_rate_limit_decorator(self, handler):
        """Import template should have rate limit decorator."""
        assert hasattr(handler.handle_import_template, "_rate_limited")


# ===========================================================================
# Tests: Handler Methods with Input Validation
# ===========================================================================


class TestHandlerWithInputValidation:
    """Tests for handler methods with input validation."""

    def test_get_template_invalid_id(self, handler, registry):
        """Get template should return 400 for invalid template ID."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_get_template("../../../etc/passwd")

        assert result.status_code == HTTPStatus.BAD_REQUEST
        body = get_body(result)
        assert "invalid" in body.get("error", "").lower()

    def test_get_template_empty_id(self, handler, registry):
        """Get template should return 400 for empty template ID."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_get_template("")

        assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_delete_template_invalid_id(self, handler, registry):
        """Delete template should return 400 for invalid template ID."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
        ):
            result = handler.handle_delete_template("has space")

        assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_rate_template_invalid_id(self, handler, registry):
        """Rate template should return 400 for invalid template ID."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value={"score": 4}),
        ):
            result = handler.handle_rate_template("has/slash")

        assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_rate_template_invalid_score(self, handler, registry):
        """Rate template should return 400 for invalid score."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value={"score": 10}),
        ):
            result = handler.handle_rate_template("tpl-001")

        assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_rate_template_invalid_review(self, handler, registry):
        """Rate template should return 400 for invalid review."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value={"score": 4, "review": 12345}),
        ):
            result = handler.handle_rate_template("tpl-001")

        assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_list_templates_invalid_pagination(self, handler, registry):
        """List templates should return 400 for invalid pagination."""
        handler._current_query_params = {"limit": "not-a-number"}

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_list_templates()

        assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_list_templates_invalid_query(self, handler, registry):
        """List templates should return 400 for invalid search query."""
        handler._current_query_params = {}

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "get_query_param", return_value=12345),
        ):
            result = handler.handle_list_templates()

        assert result.status_code == HTTPStatus.BAD_REQUEST


# ===========================================================================
# Tests: Status Endpoint
# ===========================================================================


class TestStatusEndpoint:
    """Tests for the status endpoint."""

    def test_status_healthy(self, handler):
        """Status should report healthy when circuit breaker is closed."""
        result = handler.handle_status()

        assert result.status_code == HTTPStatus.OK
        body = get_body(result)
        assert body["status"] == "healthy"
        assert body["circuit_breaker"]["state"] == "closed"

    def test_status_degraded(self, handler):
        """Status should report degraded when circuit breaker is open."""
        for _ in range(5):
            handler._circuit_breaker.record_failure()

        result = handler.handle_status()

        assert result.status_code == HTTPStatus.OK
        body = get_body(result)
        assert body["status"] == "degraded"
        assert body["circuit_breaker"]["state"] == "open"


# ===========================================================================
# Tests: Existing Handler Methods (upgraded versions)
# ===========================================================================


class TestListTemplates:
    """Tests for listing marketplace templates."""

    def test_list_templates_happy_path(self, handler, registry):
        """List templates returns all templates with count."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_list_templates()

        body = get_body(result)
        assert result.status_code == HTTPStatus.OK
        assert body["count"] == 3
        assert len(body["templates"]) == 3
        assert body["limit"] == DEFAULT_LIMIT
        assert body["offset"] == 0

    def test_list_templates_with_pagination(self, handler, registry):
        """List templates respects limit and offset parameters."""
        handler._current_query_params = {"limit": "1", "offset": "1"}

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_list_templates()

        body = get_body(result)
        assert result.status_code == HTTPStatus.OK
        assert body["limit"] == 1
        assert body["offset"] == 1

    def test_list_templates_with_valid_tags(self, handler, registry):
        """List templates with valid tags filter."""
        handler._current_query_params = {"tags": "test,example"}

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(
                handler,
                "get_query_param",
                side_effect=lambda k: handler._current_query_params.get(k),
            ),
        ):
            result = handler.handle_list_templates()

        assert result.status_code == HTTPStatus.OK

    def test_list_templates_internal_error(self, handler):
        """List templates returns 500 on internal error."""
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            side_effect=RuntimeError("DB down"),
        ):
            result = handler.handle_list_templates()

        assert result.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


class TestGetTemplate:
    """Tests for getting a single template."""

    def test_get_template_happy_path(self, handler, registry):
        """Get template returns template data and increments downloads."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_get_template("tpl-001")

        body = get_body(result)
        assert result.status_code == HTTPStatus.OK
        assert body["id"] == "tpl-001"
        assert body["name"] == "Debate Flow"
        # Download count should have been incremented
        assert registry._downloads.get("tpl-001", 0) == 1

    def test_get_template_not_found(self, handler, registry):
        """Get template returns 404 for unknown ID."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_get_template("nonexistent-valid-id")

        body = get_body(result)
        assert result.status_code == HTTPStatus.NOT_FOUND
        assert "not found" in body.get("error", "").lower()

    def test_get_template_internal_error(self, handler):
        """Get template returns 500 on internal error."""
        with patch(
            "aragora.server.handlers.marketplace._get_registry",
            side_effect=RuntimeError("fail"),
        ):
            result = handler.handle_get_template("tpl-001")

        assert result.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


class TestCreateTemplate:
    """Tests for creating a template."""

    def test_create_template_happy_path(self, handler, registry):
        """Create template succeeds with valid body and auth."""
        handler._current_handler = MagicMock()
        body_data = {"id": "tpl-new", "name": "New Template"}

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value=body_data),
        ):
            result = handler.handle_create_template()

        body = get_body(result)
        assert result.status_code == HTTPStatus.CREATED
        assert body["success"] is True
        assert body["id"] == "tpl-new"

    def test_create_template_no_auth(self, handler, registry):
        """Create template returns 401 when not authenticated."""
        handler._current_handler = MagicMock()
        from aragora.server.handlers.base import error_response

        auth_error = error_response("Authentication required", 401)

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(None, auth_error)),
        ):
            result = handler.handle_create_template()

        assert result.status_code == 401

    def test_create_template_no_body(self, handler, registry):
        """Create template returns 400 when body is empty."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value=None),
        ):
            result = handler.handle_create_template()

        body = get_body(result)
        assert result.status_code == HTTPStatus.BAD_REQUEST
        assert "body" in body.get("error", "").lower()

    def test_create_template_invalid_id(self, handler, registry):
        """Create template returns 400 for invalid template ID."""
        handler._current_handler = MagicMock()
        body_data = {"id": "../evil", "name": "Evil Template"}

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value=body_data),
        ):
            result = handler.handle_create_template()

        assert result.status_code == HTTPStatus.BAD_REQUEST


class TestDeleteTemplate:
    """Tests for deleting a template."""

    def test_delete_template_happy_path(self, handler, registry):
        """Delete template succeeds for existing template."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
        ):
            result = handler.handle_delete_template("tpl-001")

        body = get_body(result)
        assert result.status_code == HTTPStatus.OK
        assert body["success"] is True
        assert body["deleted"] == "tpl-001"
        assert "tpl-001" not in registry._templates

    def test_delete_template_forbidden_builtin(self, handler):
        """Delete template returns 403 for non-deletable (built-in) template."""
        handler._current_handler = MagicMock()
        mock_reg = MockRegistry()
        mock_reg.delete = MagicMock(return_value=False)

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_reg),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
        ):
            result = handler.handle_delete_template("builtin-001")

        assert result.status_code == HTTPStatus.FORBIDDEN


class TestRateTemplate:
    """Tests for rating a template."""

    def test_rate_template_happy_path(self, handler, registry):
        """Rate template succeeds with valid score."""
        handler._current_handler = MagicMock()

        mock_rating_cls = MagicMock(return_value=MockRating(score=4, review="Nice!"))
        mock_marketplace_mod = MagicMock(TemplateRating=mock_rating_cls)

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value={"score": 4, "review": "Nice!"}),
            patch.dict("sys.modules", {"aragora.marketplace": mock_marketplace_mod}),
        ):
            result = handler.handle_rate_template("tpl-001")

        body = get_body(result)
        assert result.status_code == HTTPStatus.OK
        assert body["success"] is True
        assert "average_rating" in body

    def test_rate_template_no_body(self, handler, registry):
        """Rate template returns 400 when body is empty."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(handler, "get_json_body", return_value=None),
        ):
            result = handler.handle_rate_template("tpl-001")

        assert result.status_code == HTTPStatus.BAD_REQUEST


class TestStarTemplate:
    """Tests for starring a template."""

    def test_star_template_happy_path(self, handler, registry):
        """Star template increments star count."""
        handler._current_handler = MagicMock()
        initial_stars = registry._templates["tpl-001"].metadata.stars

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
        ):
            result = handler.handle_star_template("tpl-001")

        body = get_body(result)
        assert result.status_code == HTTPStatus.OK
        assert body["success"] is True
        assert body["stars"] == initial_stars + 1


class TestListCategories:
    """Tests for listing template categories."""

    def test_list_categories_happy_path(self, handler, registry):
        """List categories returns available categories."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_list_categories()

        body = get_body(result)
        assert result.status_code == HTTPStatus.OK
        assert "categories" in body
        assert len(body["categories"]) > 0


class TestExportImportTemplate:
    """Tests for template export and import."""

    def test_export_template_happy_path(self, handler, registry):
        """Export template returns downloadable JSON."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_export_template("tpl-001")

        assert result.status_code == HTTPStatus.OK
        assert result.content_type == "application/json"
        assert result.headers.get("Content-Disposition") is not None
        assert "tpl-001.json" in result.headers["Content-Disposition"]

        body = json.loads(result.body.decode("utf-8"))
        assert body["id"] == "tpl-001"

    def test_export_template_not_found(self, handler, registry):
        """Export template returns 404 for unknown template."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_export_template("nonexistent-valid-id")

        body = get_body(result)
        assert result.status_code == HTTPStatus.NOT_FOUND
        assert "not found" in body.get("error", "").lower()

    def test_export_template_invalid_id(self, handler, registry):
        """Export template returns 400 for invalid template ID."""
        with patch("aragora.server.handlers.marketplace._get_registry", return_value=registry):
            result = handler.handle_export_template("../etc/passwd")

        assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_import_template_delegates_to_create(self, handler, registry):
        """Import template delegates to handle_create_template."""
        handler._current_handler = MagicMock()

        with (
            patch("aragora.server.handlers.marketplace._get_registry", return_value=registry),
            patch.object(handler, "require_auth_or_error", return_value=(MockAuthUser(), None)),
            patch.object(
                handler, "get_json_body", return_value={"id": "tpl-imported", "name": "Imported"}
            ),
        ):
            result = handler.handle_import_template()

        body = get_body(result)
        assert result.status_code == HTTPStatus.CREATED
        assert body["success"] is True


# ===========================================================================
# Tests: Constants
# ===========================================================================


class TestConstants:
    """Tests to verify constant values are sane."""

    def test_pagination_limits(self):
        """Test pagination limit constants."""
        assert MIN_LIMIT >= 1
        assert DEFAULT_LIMIT > 0
        assert MAX_LIMIT >= DEFAULT_LIMIT
        assert MAX_OFFSET > 0

    def test_rating_bounds(self):
        """Test rating bound constants."""
        assert MIN_RATING >= 1
        assert MAX_RATING >= MIN_RATING
        assert MAX_RATING <= 10

    def test_string_length_limits(self):
        """Test string length limit constants."""
        assert MAX_TEMPLATE_ID_LENGTH > 0
        assert MAX_QUERY_LENGTH > 0
        assert MAX_TAGS_LENGTH > 0
        assert MAX_REVIEW_LENGTH > 0

    def test_safe_id_pattern(self):
        """Test the safe ID regex pattern."""
        assert SAFE_ID_PATTERN.match("valid-id") is not None
        assert SAFE_ID_PATTERN.match("valid_id_123") is not None
        assert SAFE_ID_PATTERN.match("UPPER") is not None
        assert SAFE_ID_PATTERN.match("") is None
        assert SAFE_ID_PATTERN.match("../evil") is None
        assert SAFE_ID_PATTERN.match("has space") is None
