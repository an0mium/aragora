"""
Tests for the Response Schema helpers.

Tests consistent API response formatting utilities.
"""

import json
import pytest

from aragora.server.response_schema import (
    paginated_response,
    list_response,
    item_response,
    success_response,
    created_response,
    deleted_response,
    not_found_response,
    validation_error_response,
    rate_limit_response,
    server_error_response,
)


class TestPaginatedResponse:
    """Tests for paginated_response helper."""

    def test_basic_pagination(self):
        """Test basic paginated response structure."""
        items = [{"id": 1}, {"id": 2}]
        result = paginated_response(items=items, total=10, offset=0, limit=2)

        data = json.loads(result.body)
        assert data["items"] == items
        assert data["total"] == 10
        assert data["offset"] == 0
        assert data["limit"] == 2
        assert data["has_more"] is True
        assert result.status_code == 200

    def test_last_page(self):
        """Test pagination on last page."""
        items = [{"id": 9}, {"id": 10}]
        result = paginated_response(items=items, total=10, offset=8, limit=2)

        data = json.loads(result.body)
        assert data["has_more"] is False

    def test_custom_item_key(self):
        """Test custom item key name."""
        items = [{"name": "debate1"}]
        result = paginated_response(items=items, total=1, offset=0, limit=10, item_key="debates")

        data = json.loads(result.body)
        assert "debates" in data
        assert data["debates"] == items
        assert "items" not in data

    def test_extra_fields(self):
        """Test adding extra fields to response."""
        result = paginated_response(
            items=[],
            total=0,
            offset=0,
            limit=10,
            extra={"query": "test", "filters": {"active": True}},
        )

        data = json.loads(result.body)
        assert data["query"] == "test"
        assert data["filters"]["active"] is True

    def test_empty_list(self):
        """Test empty paginated response."""
        result = paginated_response(items=[], total=0, offset=0, limit=10)

        data = json.loads(result.body)
        assert data["items"] == []
        assert data["total"] == 0
        assert data["has_more"] is False


class TestListResponse:
    """Tests for list_response helper."""

    def test_basic_list(self):
        """Test basic list response."""
        items = [{"id": 1}, {"id": 2}, {"id": 3}]
        result = list_response(items=items)

        data = json.loads(result.body)
        assert data["items"] == items
        assert data["total"] == 3

    def test_custom_key(self):
        """Test custom item key."""
        items = ["agent1", "agent2"]
        result = list_response(items=items, item_key="agents")

        data = json.loads(result.body)
        assert "agents" in data
        assert data["agents"] == items

    def test_without_total(self):
        """Test list without total count."""
        result = list_response(items=[1, 2, 3], include_total=False)

        data = json.loads(result.body)
        assert "total" not in data

    def test_with_extra(self):
        """Test list with extra fields."""
        result = list_response(
            items=["a", "b"],
            extra={"source": "cache", "timestamp": "2024-01-01"},
        )

        data = json.loads(result.body)
        assert data["source"] == "cache"
        assert data["timestamp"] == "2024-01-01"


class TestItemResponse:
    """Tests for item_response helper."""

    def test_dict_item(self):
        """Test returning a dict item."""
        item = {"id": "123", "name": "Test", "status": "active"}
        result = item_response(item)

        data = json.loads(result.body)
        assert data == item
        assert result.status_code == 200

    def test_custom_status(self):
        """Test custom status code."""
        result = item_response({"id": "123"}, status=201)
        assert result.status_code == 201

    def test_extra_fields_merged(self):
        """Test extra fields are merged into dict item."""
        item = {"id": "123"}
        result = item_response(item, extra={"_links": {"self": "/api/123"}})

        data = json.loads(result.body)
        assert data["id"] == "123"
        assert data["_links"]["self"] == "/api/123"

    def test_non_dict_item_wrapped(self):
        """Test non-dict items are wrapped."""
        result = item_response("simple_value")

        data = json.loads(result.body)
        assert data["data"] == "simple_value"


class TestSuccessResponse:
    """Tests for success_response helper."""

    def test_with_message(self):
        """Test success response with message."""
        result = success_response(message="Operation completed")

        data = json.loads(result.body)
        assert data["message"] == "Operation completed"

    def test_with_data(self):
        """Test success response with data."""
        result = success_response(data={"count": 5, "updated": True})

        data = json.loads(result.body)
        assert data["count"] == 5
        assert data["updated"] is True

    def test_with_both(self):
        """Test success with message and data."""
        result = success_response(
            data={"id": "123"},
            message="Created successfully",
            status=201,
        )

        data = json.loads(result.body)
        assert data["id"] == "123"
        assert data["message"] == "Created successfully"
        assert result.status_code == 201


class TestCreatedResponse:
    """Tests for created_response helper."""

    def test_basic_created(self):
        """Test basic 201 response."""
        item = {"id": "abc123", "name": "New Item"}
        result = created_response(item)

        data = json.loads(result.body)
        assert data == item
        assert result.status_code == 201
        assert "Location" in result.headers

    def test_custom_id_field(self):
        """Test custom ID field for location."""
        item = {"debate_id": "xyz", "topic": "Test"}
        result = created_response(item, id_field="debate_id")

        assert "/api/xyz" in result.headers.get("Location", "")


class TestDeletedResponse:
    """Tests for deleted_response helper."""

    def test_default_message(self):
        """Test default delete message."""
        result = deleted_response()

        data = json.loads(result.body)
        assert data["message"] == "Deleted successfully"
        assert data["deleted"] is True

    def test_custom_message(self):
        """Test custom delete message."""
        result = deleted_response(message="Agent removed from debate")

        data = json.loads(result.body)
        assert data["message"] == "Agent removed from debate"


class TestNotFoundResponse:
    """Tests for not_found_response helper."""

    def test_basic_not_found(self):
        """Test basic 404 response."""
        result = not_found_response()

        data = json.loads(result.body)
        assert "not found" in data["error"]
        assert data["code"] == "NOT_FOUND"
        assert result.status_code == 404

    def test_with_resource_type(self):
        """Test 404 with resource type."""
        result = not_found_response(resource="Debate")

        data = json.loads(result.body)
        assert "Debate not found" in data["error"]

    def test_with_identifier(self):
        """Test 404 with identifier."""
        result = not_found_response(resource="Agent", identifier="claude-3")

        data = json.loads(result.body)
        assert "Agent 'claude-3' not found" in data["error"]


class TestValidationErrorResponse:
    """Tests for validation_error_response helper."""

    def test_single_error(self):
        """Test single validation error."""
        result = validation_error_response("Invalid email format")

        data = json.loads(result.body)
        assert data["error"] == "Validation failed"
        assert data["code"] == "VALIDATION_ERROR"
        assert "Invalid email format" in data["details"]
        assert result.status_code == 400

    def test_multiple_errors(self):
        """Test multiple validation errors."""
        errors = ["Name is required", "Limit must be positive"]
        result = validation_error_response(errors)

        data = json.loads(result.body)
        assert len(data["details"]) == 2
        assert "Name is required" in data["details"]

    def test_with_field(self):
        """Test validation error with field name."""
        result = validation_error_response("Must be numeric", field="limit")

        data = json.loads(result.body)
        assert data["field"] == "limit"


class TestRateLimitResponse:
    """Tests for rate_limit_response helper."""

    def test_basic_rate_limit(self):
        """Test basic rate limit response."""
        result = rate_limit_response()

        data = json.loads(result.body)
        assert data["code"] == "RATE_LIMITED"
        assert data["retry_after"] == 60
        assert result.status_code == 429
        assert result.headers.get("Retry-After") == "60"

    def test_custom_retry_after(self):
        """Test custom retry-after value."""
        result = rate_limit_response(retry_after=120)

        data = json.loads(result.body)
        assert data["retry_after"] == 120
        assert result.headers.get("Retry-After") == "120"


class TestServerErrorResponse:
    """Tests for server_error_response helper."""

    def test_basic_error(self):
        """Test basic 500 response."""
        result = server_error_response()

        data = json.loads(result.body)
        assert data["error"] == "Internal server error"
        assert data["code"] == "INTERNAL_ERROR"
        assert result.status_code == 500

    def test_with_trace_id(self):
        """Test error with trace ID."""
        result = server_error_response(trace_id="abc123")

        data = json.loads(result.body)
        assert data["trace_id"] == "abc123"

    def test_custom_message(self):
        """Test custom error message."""
        result = server_error_response(message="Database unavailable")

        data = json.loads(result.body)
        assert data["error"] == "Database unavailable"


class TestResponseContentType:
    """Tests for response content type."""

    def test_all_responses_are_json(self):
        """All response helpers should return JSON content type."""
        responses = [
            paginated_response([], 0, 0, 10),
            list_response([]),
            item_response({}),
            success_response(),
            created_response({"id": "1"}),
            deleted_response(),
            not_found_response(),
            validation_error_response("error"),
            rate_limit_response(),
            server_error_response(),
        ]

        for response in responses:
            assert response.content_type == "application/json"


class TestResponseIntegration:
    """Integration tests for response helpers."""

    def test_realistic_debate_list(self):
        """Test realistic debate list response."""
        debates = [
            {"id": "d1", "topic": "AI Safety", "status": "active"},
            {"id": "d2", "topic": "Ethics", "status": "completed"},
        ]
        result = paginated_response(
            items=debates,
            total=100,
            offset=0,
            limit=2,
            item_key="debates",
            extra={"filters": {"status": "all"}},
        )

        data = json.loads(result.body)
        assert len(data["debates"]) == 2
        assert data["has_more"] is True
        assert data["filters"]["status"] == "all"

    def test_realistic_agent_detail(self):
        """Test realistic agent detail response."""
        agent = {
            "name": "claude-visionary",
            "elo": 1850,
            "matches": 42,
            "win_rate": 0.71,
        }
        result = item_response(
            agent,
            extra={"_links": {"matches": "/api/agents/claude-visionary/matches"}},
        )

        data = json.loads(result.body)
        assert data["name"] == "claude-visionary"
        assert data["_links"]["matches"].endswith("/matches")
