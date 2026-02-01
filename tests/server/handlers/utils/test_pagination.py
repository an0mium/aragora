"""Tests for pagination utilities in responses module."""

from __future__ import annotations

import json
import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
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

import pytest

from aragora.server.handlers.utils.responses import (
    paginated_response,
    parse_pagination_params,
    normalize_pagination_response,
)


# =============================================================================
# Test paginated_response
# =============================================================================


class TestPaginatedResponse:
    """Tests for paginated_response function."""

    def test_paginated_response_format(self):
        """Should create response with standard pagination format."""
        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = paginated_response(
            items,
            total=10,
            limit=3,
            offset=0,
        )

        assert result.status_code == 200
        assert result.content_type == "application/json"

        body = json.loads(result.body)
        assert "data" in body
        assert "pagination" in body
        assert body["data"] == items
        assert body["pagination"]["total"] == 10
        assert body["pagination"]["limit"] == 3
        assert body["pagination"]["offset"] == 0

    def test_paginated_response_has_more_true(self):
        """Should set has_more to true when more items remain."""
        items = [{"id": 1}, {"id": 2}]
        result = paginated_response(
            items,
            total=10,
            limit=2,
            offset=0,
        )

        body = json.loads(result.body)
        assert body["pagination"]["has_more"] is True

    def test_paginated_response_has_more_false(self):
        """Should set has_more to false when no more items remain."""
        # Case 1: At last page with exact items
        items = [{"id": 9}, {"id": 10}]
        result = paginated_response(
            items,
            total=10,
            limit=2,
            offset=8,
        )

        body = json.loads(result.body)
        assert body["pagination"]["has_more"] is False

        # Case 2: Last page with fewer items than limit
        items = [{"id": 10}]
        result = paginated_response(
            items,
            total=10,
            limit=3,
            offset=9,
        )

        body = json.loads(result.body)
        assert body["pagination"]["has_more"] is False

        # Case 3: All items fit on one page
        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = paginated_response(
            items,
            total=3,
            limit=10,
            offset=0,
        )

        body = json.loads(result.body)
        assert body["pagination"]["has_more"] is False

    def test_paginated_response_with_headers(self):
        """Should include custom headers."""
        result = paginated_response(
            [{"id": 1}],
            total=1,
            limit=10,
            offset=0,
            headers={"X-Request-Id": "abc123"},
        )

        assert result.headers["X-Request-Id"] == "abc123"

    def test_paginated_response_empty_items(self):
        """Should handle empty items list."""
        result = paginated_response(
            [],
            total=0,
            limit=20,
            offset=0,
        )

        body = json.loads(result.body)
        assert body["data"] == []
        assert body["pagination"]["total"] == 0
        assert body["pagination"]["has_more"] is False

    def test_paginated_response_middle_page(self):
        """Should correctly handle middle pages."""
        items = [{"id": 21}, {"id": 22}, {"id": 23}, {"id": 24}, {"id": 25}]
        result = paginated_response(
            items,
            total=100,
            limit=5,
            offset=20,
        )

        body = json.loads(result.body)
        assert body["pagination"]["offset"] == 20
        assert body["pagination"]["has_more"] is True
        # offset=20 + 5 items = 25, which is less than 100 total


# =============================================================================
# Test parse_pagination_params
# =============================================================================


class TestParsePaginationParams:
    """Tests for parse_pagination_params function."""

    def test_parse_pagination_default_values(self):
        """Should use default values when params not provided."""
        limit, offset = parse_pagination_params({})

        assert limit == 20  # default_limit
        assert offset == 0

    def test_parse_pagination_custom_defaults(self):
        """Should use custom default values."""
        limit, offset = parse_pagination_params(
            {},
            default_limit=50,
            max_limit=200,
        )

        assert limit == 50
        assert offset == 0

    def test_parse_pagination_extracts_values(self):
        """Should extract limit and offset from query params."""
        limit, offset = parse_pagination_params({"limit": "15", "offset": "30"})

        assert limit == 15
        assert offset == 30

    def test_parse_pagination_handles_int_values(self):
        """Should handle integer values in params."""
        limit, offset = parse_pagination_params({"limit": 25, "offset": 50})

        assert limit == 25
        assert offset == 50

    def test_parse_pagination_max_limit_enforced(self):
        """Should enforce max_limit."""
        limit, offset = parse_pagination_params(
            {"limit": "500"},
            max_limit=100,
        )

        assert limit == 100  # Clamped to max_limit

        # Custom max_limit
        limit, offset = parse_pagination_params(
            {"limit": "300"},
            max_limit=200,
        )

        assert limit == 200

    def test_parse_pagination_negative_values(self):
        """Should handle negative values gracefully."""
        # Negative limit - should use default
        limit, offset = parse_pagination_params(
            {"limit": "-5"},
            default_limit=20,
        )
        assert limit == 20  # Falls back to default

        # Negative offset - should clamp to 0
        limit, offset = parse_pagination_params({"offset": "-10"})
        assert offset == 0

        # Both negative
        limit, offset = parse_pagination_params(
            {"limit": "-1", "offset": "-100"},
            default_limit=25,
        )
        assert limit == 25
        assert offset == 0

    def test_parse_pagination_zero_limit(self):
        """Should handle zero limit."""
        limit, offset = parse_pagination_params(
            {"limit": "0"},
            default_limit=20,
        )

        # Zero is invalid, should use default
        assert limit == 20

    def test_parse_pagination_invalid_values(self):
        """Should handle non-numeric values gracefully."""
        limit, offset = parse_pagination_params(
            {"limit": "abc", "offset": "xyz"},
            default_limit=15,
        )

        assert limit == 15  # Falls back to default
        assert offset == 0

    def test_parse_pagination_list_values(self):
        """Should handle list values (as from some query parsers)."""
        limit, offset = parse_pagination_params({"limit": ["10", "20"], "offset": ["5"]})

        # Should take first value
        assert limit == 10
        assert offset == 5

    def test_parse_pagination_empty_list(self):
        """Should handle empty list values."""
        limit, offset = parse_pagination_params(
            {"limit": [], "offset": []},
            default_limit=20,
        )

        assert limit == 20
        assert offset == 0


# =============================================================================
# Test normalize_pagination_response
# =============================================================================


class TestNormalizePaginationResponse:
    """Tests for normalize_pagination_response function."""

    def test_normalize_pagination_items_key(self):
        """Should normalize 'items' key to 'data'."""
        response = {
            "items": [1, 2, 3],
            "total": 10,
            "limit": 3,
            "offset": 0,
        }

        normalized = normalize_pagination_response(response)

        assert "data" in normalized
        assert "items" not in normalized
        assert normalized["data"] == [1, 2, 3]
        assert normalized["pagination"]["total"] == 10
        assert normalized["pagination"]["limit"] == 3
        assert normalized["pagination"]["offset"] == 0

    def test_normalize_pagination_results_key(self):
        """Should normalize 'results' key to 'data'."""
        response = {
            "results": [{"id": 1}, {"id": 2}],
            "total": 5,
        }

        normalized = normalize_pagination_response(response)

        assert "data" in normalized
        assert "results" not in normalized
        assert normalized["data"] == [{"id": 1}, {"id": 2}]

    def test_normalize_pagination_total_count_key(self):
        """Should normalize 'total_count' key to 'total' in pagination."""
        response = {
            "items": [1, 2],
            "total_count": 100,
        }

        normalized = normalize_pagination_response(response)

        assert normalized["pagination"]["total"] == 100
        assert "total_count" not in normalized

    def test_normalize_pagination_count_key(self):
        """Should normalize 'count' key to 'total' in pagination."""
        response = {
            "data": ["a", "b", "c"],
            "count": 50,
        }

        normalized = normalize_pagination_response(response)

        assert normalized["pagination"]["total"] == 50
        assert "count" not in normalized

    def test_normalize_pagination_already_standard(self):
        """Should return already standardized format unchanged."""
        response = {
            "data": [1, 2, 3],
            "pagination": {
                "total": 100,
                "limit": 10,
                "offset": 0,
                "has_more": True,
            },
        }

        normalized = normalize_pagination_response(response)

        assert normalized == response

    def test_normalize_pagination_calculates_has_more(self):
        """Should calculate has_more when normalizing."""
        # Has more items
        response = {
            "items": [1, 2, 3],
            "total": 10,
            "limit": 3,
            "offset": 0,
        }

        normalized = normalize_pagination_response(response)

        # offset (0) + len(items) (3) < total (10) -> has_more = True
        assert normalized["pagination"]["has_more"] is True

        # No more items
        response = {
            "items": [8, 9, 10],
            "total": 10,
            "limit": 3,
            "offset": 7,
        }

        normalized = normalize_pagination_response(response)

        # offset (7) + len(items) (3) >= total (10) -> has_more = False
        assert normalized["pagination"]["has_more"] is False

    def test_normalize_pagination_empty_response(self):
        """Should handle response with no recognized keys."""
        response = {"unknown_key": "value"}

        normalized = normalize_pagination_response(response)

        assert normalized["data"] == []
        assert normalized["pagination"]["total"] == 0
        assert normalized["pagination"]["has_more"] is False

    def test_normalize_pagination_defaults_limit_to_items_length(self):
        """Should default limit to length of items."""
        response = {
            "items": [1, 2, 3, 4, 5],
            "total": 10,
        }

        normalized = normalize_pagination_response(response)

        assert normalized["pagination"]["limit"] == 5

    def test_normalize_pagination_defaults_offset_to_zero(self):
        """Should default offset to zero."""
        response = {
            "items": [1, 2, 3],
            "total": 3,
        }

        normalized = normalize_pagination_response(response)

        assert normalized["pagination"]["offset"] == 0
