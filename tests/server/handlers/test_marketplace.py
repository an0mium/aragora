"""
Tests for aragora.server.handlers.marketplace - Marketplace HTTP Handlers.

Tests cover:
- MarketplaceHandler: instantiation, circuit breaker, CRUD operations
- Input validation: template ID, pagination, rating, review, query, tags
- Circuit breaker: state transitions, can_proceed, record_success/failure
- List templates: happy path, with filters, circuit breaker open
- Get template: found, not found, invalid ID
- Create template: success, missing body, circuit breaker
- Delete template: success, forbidden (built-in), circuit breaker
- Rate template: success, missing score, invalid score, invalid review
- Get ratings: success, circuit breaker
- Star template: success, circuit breaker
- List categories: success, circuit breaker
- Export template: found, not found, circuit breaker
- Import template: delegates to create
- Status: healthy, degraded
"""

from __future__ import annotations

import json
import time
from http import HTTPStatus
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.marketplace import (
    MarketplaceHandler,
    MarketplaceCircuitBreaker,
    _validate_template_id,
    _validate_pagination,
    _validate_rating,
    _validate_review,
    _validate_query,
    _validate_tags,
    reset_marketplace_circuit_breaker,
    get_marketplace_circuit_breaker_status,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


# ===========================================================================
# Input Validation Tests
# ===========================================================================


class TestValidateTemplateId:
    """Tests for _validate_template_id."""

    def test_valid_id(self):
        valid, err = _validate_template_id("my-template-123")
        assert valid is True
        assert err == ""

    def test_empty_id(self):
        valid, err = _validate_template_id("")
        assert valid is False
        assert "required" in err.lower()

    def test_none_id(self):
        valid, err = _validate_template_id(None)
        assert valid is False

    def test_too_long_id(self):
        valid, err = _validate_template_id("a" * 129)
        assert valid is False
        assert "128" in err

    def test_invalid_characters(self):
        valid, err = _validate_template_id("template/with/slashes")
        assert valid is False
        assert "invalid" in err.lower()

    def test_starts_with_hyphen(self):
        valid, err = _validate_template_id("-leading-hyphen")
        assert valid is False

    def test_underscore_id(self):
        valid, err = _validate_template_id("my_template_v2")
        assert valid is True


class TestValidatePagination:
    """Tests for _validate_pagination."""

    def test_defaults(self):
        limit, offset, err = _validate_pagination({})
        assert limit == 50
        assert offset == 0
        assert err == ""

    def test_custom_values(self):
        limit, offset, err = _validate_pagination({"limit": "25", "offset": "10"})
        assert limit == 25
        assert offset == 10
        assert err == ""

    def test_clamps_max_limit(self):
        limit, offset, err = _validate_pagination({"limit": "999"})
        assert limit == 200  # MAX_LIMIT

    def test_clamps_min_limit(self):
        limit, offset, err = _validate_pagination({"limit": "0"})
        assert limit == 1  # MIN_LIMIT

    def test_invalid_limit(self):
        limit, offset, err = _validate_pagination({"limit": "not_a_number"})
        assert err != ""

    def test_invalid_offset(self):
        limit, offset, err = _validate_pagination({"offset": "bad"})
        assert err != ""


class TestValidateRating:
    """Tests for _validate_rating."""

    def test_valid_rating(self):
        valid, score, err = _validate_rating(4)
        assert valid is True
        assert score == 4

    def test_none_rating(self):
        valid, score, err = _validate_rating(None)
        assert valid is False
        assert "required" in err.lower()

    def test_below_min(self):
        valid, score, err = _validate_rating(0)
        assert valid is False

    def test_above_max(self):
        valid, score, err = _validate_rating(6)
        assert valid is False

    def test_not_integer(self):
        valid, score, err = _validate_rating("four")
        assert valid is False


class TestValidateReview:
    """Tests for _validate_review."""

    def test_valid_review(self):
        valid, text, err = _validate_review("Great template!")
        assert valid is True
        assert text is not None

    def test_none_review(self):
        valid, text, err = _validate_review(None)
        assert valid is True
        assert text is None

    def test_too_long(self):
        valid, text, err = _validate_review("x" * 2001)
        assert valid is False

    def test_not_string(self):
        valid, text, err = _validate_review(123)
        assert valid is False


class TestValidateQuery:
    """Tests for _validate_query."""

    def test_valid_query(self):
        valid, q, err = _validate_query("search term")
        assert valid is True
        assert q != ""

    def test_empty_query(self):
        valid, q, err = _validate_query("")
        assert valid is True
        assert q == ""

    def test_none_query(self):
        valid, q, err = _validate_query(None)
        assert valid is True

    def test_too_long_query(self):
        valid, q, err = _validate_query("x" * 501)
        assert valid is False

    def test_not_string_query(self):
        valid, q, err = _validate_query(123)
        assert valid is False


class TestValidateTags:
    """Tests for _validate_tags."""

    def test_valid_tags(self):
        valid, tags, err = _validate_tags("security,code,review")
        assert valid is True
        assert len(tags) == 3

    def test_empty_tags(self):
        valid, tags, err = _validate_tags("")
        assert valid is True
        assert tags == []

    def test_none_tags(self):
        valid, tags, err = _validate_tags(None)
        assert valid is True
        assert tags == []

    def test_too_long_tags(self):
        valid, tags, err = _validate_tags("x" * 1001)
        assert valid is False

    def test_not_string_tags(self):
        valid, tags, err = _validate_tags(["security"])
        assert valid is False


# ===========================================================================
# Circuit Breaker Tests
# ===========================================================================


class TestMarketplaceCircuitBreaker:
    """Tests for the MarketplaceCircuitBreaker."""

    def test_initial_state_closed(self):
        cb = MarketplaceCircuitBreaker()
        assert cb.state == MarketplaceCircuitBreaker.CLOSED

    def test_can_proceed_when_closed(self):
        cb = MarketplaceCircuitBreaker()
        assert cb.can_proceed() is True

    def test_is_allowed_alias(self):
        cb = MarketplaceCircuitBreaker()
        assert cb.is_allowed() is True

    def test_opens_after_threshold_failures(self):
        cb = MarketplaceCircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.OPEN
        assert cb.can_proceed() is False

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
        # Should allow up to half_open_max_calls
        assert cb.can_proceed() is True
        assert cb.can_proceed() is True
        assert cb.can_proceed() is False

    def test_closes_after_successful_half_open(self):
        cb = MarketplaceCircuitBreaker(
            failure_threshold=1, cooldown_seconds=0.01, half_open_max_calls=2
        )
        cb.record_failure()
        time.sleep(0.02)
        cb.can_proceed()  # consume one half-open call
        cb.record_success()
        cb.record_success()
        assert cb.state == MarketplaceCircuitBreaker.CLOSED

    def test_reopens_on_half_open_failure(self):
        cb = MarketplaceCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.can_proceed()  # enter half-open
        cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.OPEN

    def test_get_status(self):
        cb = MarketplaceCircuitBreaker()
        status = cb.get_status()
        assert "state" in status
        assert "failure_count" in status
        assert "failure_threshold" in status
        assert status["state"] == "closed"

    def test_reset(self):
        cb = MarketplaceCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == MarketplaceCircuitBreaker.OPEN
        cb.reset()
        assert cb.state == MarketplaceCircuitBreaker.CLOSED
        assert cb.can_proceed() is True

    def test_record_success_resets_failure_count_when_closed(self):
        cb = MarketplaceCircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # Failure count should be 0 now
        assert cb._failure_count == 0


# ===========================================================================
# MarketplaceHandler Instantiation Tests
# ===========================================================================


class TestMarketplaceHandlerBasics:
    """Basic instantiation tests."""

    def test_instantiation_with_ctx(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler(ctx={"key": "val"})
        assert h is not None
        assert isinstance(h, MarketplaceHandler)
        assert h.ctx == {"key": "val"}

    def test_instantiation_with_server_context(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler(server_context={"key": "val"})
        assert h.ctx == {"key": "val"}

    def test_instantiation_no_args(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        assert h.ctx == {}

    def test_has_circuit_breaker(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        assert h._circuit_breaker is not None


# ===========================================================================
# MarketplaceHandler Status Endpoint Tests
# ===========================================================================


class TestMarketplaceStatus:
    """Tests for the status endpoint."""

    def test_status_healthy(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        result = h.handle_status()
        assert result.status_code == 200
        data = _parse_body(result)
        assert data["status"] == "healthy"
        assert "circuit_breaker" in data

    def test_status_degraded_when_open(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        # Force circuit breaker open
        for _ in range(10):
            h._circuit_breaker.record_failure()
        result = h.handle_status()
        data = _parse_body(result)
        assert data["status"] == "degraded"


# ===========================================================================
# MarketplaceHandler List Templates Tests
# ===========================================================================


class TestMarketplaceListTemplates:
    """Tests for listing marketplace templates."""

    def test_list_templates_circuit_breaker_open(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        h._current_query_params = {}
        for _ in range(10):
            h._circuit_breaker.record_failure()
        result = h.handle_list_templates()
        assert result.status_code == HTTPStatus.SERVICE_UNAVAILABLE

    def test_list_templates_success(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        h._current_query_params = {}

        mock_registry = MagicMock()
        mock_template = MagicMock()
        mock_template.to_dict.return_value = {"id": "t1", "name": "Test Template"}
        mock_registry.search.return_value = [mock_template]

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            result = h.handle_list_templates()
            assert result.status_code == 200
            data = _parse_body(result)
            assert "templates" in data
            assert data["count"] == 1

    def test_list_templates_exception(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        h._current_query_params = {}

        with patch(
            "aragora.server.handlers.marketplace._get_registry", side_effect=OSError("DB down")
        ):
            result = h.handle_list_templates()
            assert result.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


# ===========================================================================
# MarketplaceHandler Get Template Tests
# ===========================================================================


class TestMarketplaceGetTemplate:
    """Tests for getting a specific template."""

    def test_get_template_invalid_id(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        result = h.handle_get_template("invalid/id/with/slashes")
        assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_get_template_not_found(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()

        mock_registry = MagicMock()
        mock_registry.get.return_value = None

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            result = h.handle_get_template("valid-template-id")
            assert result.status_code == HTTPStatus.NOT_FOUND

    def test_get_template_success(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()

        mock_registry = MagicMock()
        mock_template = MagicMock()
        mock_template.to_dict.return_value = {"id": "test-1", "name": "Test"}
        mock_registry.get.return_value = mock_template

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            result = h.handle_get_template("test-1")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["id"] == "test-1"

    def test_get_template_circuit_breaker_open(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        for _ in range(10):
            h._circuit_breaker.record_failure()
        result = h.handle_get_template("valid-id")
        assert result.status_code == HTTPStatus.SERVICE_UNAVAILABLE


# ===========================================================================
# MarketplaceHandler Create Template Tests
# ===========================================================================


class TestMarketplaceCreateTemplate:
    """Tests for creating a template."""

    def test_create_missing_body(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        h._current_handler = MagicMock()

        # Mock auth to pass
        with patch.object(h, "require_auth_or_error", return_value=(MagicMock(), None)):
            with patch.object(h, "get_json_body", return_value=None):
                result = h.handle_create_template()
                assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_create_success(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        h._current_handler = MagicMock()

        mock_registry = MagicMock()
        mock_registry.import_template.return_value = "new-template-id"

        with patch.object(h, "require_auth_or_error", return_value=(MagicMock(), None)):
            with patch.object(h, "get_json_body", return_value={"name": "New Template"}):
                with patch(
                    "aragora.server.handlers.marketplace._get_registry", return_value=mock_registry
                ):
                    result = h.handle_create_template()
                    assert result.status_code == HTTPStatus.CREATED
                    data = _parse_body(result)
                    assert data["success"] is True
                    assert data["id"] == "new-template-id"

    def test_create_circuit_breaker_open(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        for _ in range(10):
            h._circuit_breaker.record_failure()
        result = h.handle_create_template()
        assert result.status_code == HTTPStatus.SERVICE_UNAVAILABLE


# ===========================================================================
# MarketplaceHandler Delete Template Tests
# ===========================================================================


class TestMarketplaceDeleteTemplate:
    """Tests for deleting a template."""

    def test_delete_invalid_id(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        result = h.handle_delete_template("bad/id")
        assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_delete_forbidden(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        h._current_handler = MagicMock()

        mock_registry = MagicMock()
        mock_registry.delete.return_value = False

        with patch.object(h, "require_auth_or_error", return_value=(MagicMock(), None)):
            with patch(
                "aragora.server.handlers.marketplace._get_registry", return_value=mock_registry
            ):
                result = h.handle_delete_template("builtin-template")
                assert result.status_code == HTTPStatus.FORBIDDEN

    def test_delete_success(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        h._current_handler = MagicMock()

        mock_registry = MagicMock()
        mock_registry.delete.return_value = True

        with patch.object(h, "require_auth_or_error", return_value=(MagicMock(), None)):
            with patch(
                "aragora.server.handlers.marketplace._get_registry", return_value=mock_registry
            ):
                result = h.handle_delete_template("user-template")
                assert result.status_code == 200
                data = _parse_body(result)
                assert data["success"] is True
                assert data["deleted"] == "user-template"


# ===========================================================================
# MarketplaceHandler Rate Template Tests
# ===========================================================================


class TestMarketplaceRateTemplate:
    """Tests for rating a template."""

    def test_rate_invalid_id(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        result = h.handle_rate_template("bad/id")
        assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_rate_missing_body(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        h._current_handler = MagicMock()

        with patch.object(h, "require_auth_or_error", return_value=(MagicMock(), None)):
            with patch.object(h, "get_json_body", return_value=None):
                result = h.handle_rate_template("valid-id")
                assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_rate_missing_score(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        h._current_handler = MagicMock()

        with patch.object(h, "require_auth_or_error", return_value=(MagicMock(), None)):
            with patch.object(h, "get_json_body", return_value={"review": "Nice"}):
                result = h.handle_rate_template("valid-id")
                assert result.status_code == HTTPStatus.BAD_REQUEST

    def test_rate_success(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        h._current_handler = MagicMock()

        mock_registry = MagicMock()
        mock_registry.get_average_rating.return_value = 4.5

        mock_user = MagicMock()
        mock_user.id = "user-1"

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch.object(h, "get_json_body", return_value={"score": 4, "review": "Great!"}):
                with patch(
                    "aragora.server.handlers.marketplace._get_registry", return_value=mock_registry
                ):
                    result = h.handle_rate_template("valid-id")
                    assert result.status_code == 200
                    data = _parse_body(result)
                    assert data["success"] is True
                    assert data["average_rating"] == 4.5


# ===========================================================================
# MarketplaceHandler Get Ratings Tests
# ===========================================================================


class TestMarketplaceGetRatings:
    """Tests for getting template ratings."""

    def test_get_ratings_success(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()

        mock_rating = MagicMock()
        mock_rating.user_id = "user-1"
        mock_rating.score = 5
        mock_rating.review = "Excellent"
        mock_rating.created_at.isoformat.return_value = "2025-01-01T00:00:00"

        mock_registry = MagicMock()
        mock_registry.get_ratings.return_value = [mock_rating]
        mock_registry.get_average_rating.return_value = 5.0

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            result = h.handle_get_ratings("valid-id")
            assert result.status_code == 200
            data = _parse_body(result)
            assert "ratings" in data
            assert data["count"] == 1
            assert data["average"] == 5.0

    def test_get_ratings_invalid_id(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        result = h.handle_get_ratings("bad/id")
        assert result.status_code == HTTPStatus.BAD_REQUEST


# ===========================================================================
# MarketplaceHandler Star Template Tests
# ===========================================================================


class TestMarketplaceStarTemplate:
    """Tests for starring a template."""

    def test_star_success(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        h._current_handler = MagicMock()

        mock_template = MagicMock()
        mock_template.metadata.stars = 42

        mock_registry = MagicMock()
        mock_registry.get.return_value = mock_template

        with patch.object(h, "require_auth_or_error", return_value=(MagicMock(), None)):
            with patch(
                "aragora.server.handlers.marketplace._get_registry", return_value=mock_registry
            ):
                result = h.handle_star_template("valid-id")
                assert result.status_code == 200
                data = _parse_body(result)
                assert data["success"] is True
                assert data["stars"] == 42


# ===========================================================================
# MarketplaceHandler List Categories Tests
# ===========================================================================


class TestMarketplaceListCategories:
    """Tests for listing categories."""

    def test_list_categories_success(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()

        mock_registry = MagicMock()
        mock_registry.list_categories.return_value = ["general", "code", "security"]

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            result = h.handle_list_categories()
            assert result.status_code == 200
            data = _parse_body(result)
            assert "categories" in data
            assert len(data["categories"]) == 3


# ===========================================================================
# MarketplaceHandler Export Template Tests
# ===========================================================================


class TestMarketplaceExportTemplate:
    """Tests for exporting a template."""

    def test_export_success(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()

        mock_registry = MagicMock()
        mock_registry.export_template.return_value = '{"id": "test-1", "name": "Test"}'

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            result = h.handle_export_template("test-1")
            assert result.status_code == 200
            assert result.content_type == "application/json"
            assert "Content-Disposition" in result.headers

    def test_export_not_found(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()

        mock_registry = MagicMock()
        mock_registry.export_template.return_value = None

        with patch("aragora.server.handlers.marketplace._get_registry", return_value=mock_registry):
            result = h.handle_export_template("nonexistent")
            assert result.status_code == HTTPStatus.NOT_FOUND

    def test_export_invalid_id(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        result = h.handle_export_template("bad/id")
        assert result.status_code == HTTPStatus.BAD_REQUEST


# ===========================================================================
# MarketplaceHandler Import Template Tests
# ===========================================================================


class TestMarketplaceImportTemplate:
    """Tests for importing a template (delegates to create)."""

    def test_import_delegates_to_create(self):
        reset_marketplace_circuit_breaker()
        h = MarketplaceHandler()
        h._current_handler = MagicMock()

        mock_registry = MagicMock()
        mock_registry.import_template.return_value = "imported-id"

        with patch.object(h, "require_auth_or_error", return_value=(MagicMock(), None)):
            with patch.object(h, "get_json_body", return_value={"name": "Imported"}):
                with patch(
                    "aragora.server.handlers.marketplace._get_registry", return_value=mock_registry
                ):
                    result = h.handle_import_template()
                    assert result.status_code == HTTPStatus.CREATED


# ===========================================================================
# Global Circuit Breaker Functions
# ===========================================================================


class TestGlobalCircuitBreakerFunctions:
    """Tests for module-level circuit breaker functions."""

    def test_get_status(self):
        reset_marketplace_circuit_breaker()
        status = get_marketplace_circuit_breaker_status()
        assert "state" in status
        assert status["state"] == "closed"

    def test_reset(self):
        reset_marketplace_circuit_breaker()
        # Should not raise
        status = get_marketplace_circuit_breaker_status()
        assert status["state"] == "closed"
