"""
Tests for aragora.server.openapi.helpers module.

Covers:
- Rate limit tiers and headers
- Error examples
- Response builders (_ok_response, _array_response, _error_response)
- Rate limited endpoint documentation
- Authentication requirements
- STANDARD_ERRORS with examples
"""

from __future__ import annotations

import pytest

from aragora.server.openapi.helpers import (
    AUTH_REQUIREMENTS,
    ERROR_EXAMPLES,
    RATE_LIMIT_HEADERS,
    RATE_LIMIT_TIERS,
    STANDARD_ERRORS,
    _array_response,
    _error_response,
    _ok_response,
    _rate_limited_endpoint,
)


# =============================================================================
# TestRateLimitTiers
# =============================================================================


class TestRateLimitTiers:
    """Tests for RATE_LIMIT_TIERS configuration."""

    def test_has_free_tier(self):
        """Should have free tier defined."""
        assert "free" in RATE_LIMIT_TIERS

    def test_has_pro_tier(self):
        """Should have pro tier defined."""
        assert "pro" in RATE_LIMIT_TIERS

    def test_has_enterprise_tier(self):
        """Should have enterprise tier defined."""
        assert "enterprise" in RATE_LIMIT_TIERS

    def test_free_tier_limits(self):
        """Free tier should have reasonable limits."""
        free = RATE_LIMIT_TIERS["free"]
        assert "requests_per_minute" in free
        assert "requests_per_hour" in free
        assert "debates_per_day" in free
        assert "concurrent_debates" in free
        assert free["requests_per_minute"] > 0
        assert free["concurrent_debates"] >= 1

    def test_pro_tier_higher_than_free(self):
        """Pro tier should have higher limits than free."""
        free = RATE_LIMIT_TIERS["free"]
        pro = RATE_LIMIT_TIERS["pro"]
        assert pro["requests_per_minute"] > free["requests_per_minute"]
        assert pro["debates_per_day"] > free["debates_per_day"]

    def test_enterprise_tier_highest(self):
        """Enterprise tier should have highest limits."""
        pro = RATE_LIMIT_TIERS["pro"]
        enterprise = RATE_LIMIT_TIERS["enterprise"]
        assert enterprise["requests_per_minute"] > pro["requests_per_minute"]
        # -1 means unlimited
        assert enterprise["debates_per_day"] == -1


# =============================================================================
# TestRateLimitHeaders
# =============================================================================


class TestRateLimitHeaders:
    """Tests for RATE_LIMIT_HEADERS definitions."""

    @pytest.mark.parametrize(
        "header",
        [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "X-RateLimit-RetryAfter",
        ],
    )
    def test_standard_headers_defined(self, header: str):
        """Standard rate limit headers should be defined."""
        assert header in RATE_LIMIT_HEADERS
        assert isinstance(RATE_LIMIT_HEADERS[header], str)


# =============================================================================
# TestErrorExamples
# =============================================================================


class TestErrorExamples:
    """Tests for ERROR_EXAMPLES definitions."""

    @pytest.mark.parametrize("status", ["400", "401", "403", "404", "402", "429", "500"])
    def test_error_examples_has_status(self, status: str):
        """ERROR_EXAMPLES should have examples for common status codes."""
        assert status in ERROR_EXAMPLES

    def test_400_has_multiple_examples(self):
        """400 errors should have multiple examples."""
        examples = ERROR_EXAMPLES["400"]
        assert "invalid_json" in examples
        assert "missing_field" in examples
        assert "invalid_value" in examples

    def test_error_example_has_code(self):
        """Error examples should include error code."""
        example = ERROR_EXAMPLES["400"]["invalid_json"]["value"]
        assert "code" in example
        assert example["code"] == "INVALID_JSON"

    def test_error_example_has_trace_id(self):
        """Error examples should include trace_id."""
        example = ERROR_EXAMPLES["401"]["missing_token"]["value"]
        assert "trace_id" in example

    def test_429_rate_limited_has_retry_after(self):
        """429 rate limit example should include retry_after."""
        example = ERROR_EXAMPLES["429"]["rate_limited"]["value"]
        assert "retry_after" in example
        assert example["retry_after"] > 0


# =============================================================================
# TestOkResponseHelper
# =============================================================================


class TestOkResponseHelper:
    """Tests for _ok_response helper function."""

    def test_basic_response(self):
        """Should create basic response with description."""
        result = _ok_response("Success")
        assert result["description"] == "Success"

    def test_without_schema(self):
        """Response without schema should have no content."""
        result = _ok_response("No content")
        assert "content" not in result

    def test_with_schema_reference(self):
        """Should handle schema reference string."""
        result = _ok_response("Agent", "Agent")
        content = result["content"]["application/json"]
        assert "$ref" in content["schema"]
        assert content["schema"]["$ref"] == "#/components/schemas/Agent"

    def test_with_inline_schema(self):
        """Should handle inline schema dict."""
        inline_schema = {"name": {"type": "string"}, "count": {"type": "integer"}}
        result = _ok_response("Custom", inline_schema)
        content = result["content"]["application/json"]["schema"]
        assert content["type"] == "object"
        assert "name" in content["properties"]
        assert "count" in content["properties"]


# =============================================================================
# TestArrayResponseHelper
# =============================================================================


class TestArrayResponseHelper:
    """Tests for _array_response helper function."""

    def test_creates_array_structure(self):
        """Should create array response structure."""
        result = _array_response("List of items", "Item")
        assert result["description"] == "List of items"
        schema = result["content"]["application/json"]["schema"]
        assert schema["type"] == "object"
        assert "items" in schema["properties"]

    def test_items_is_array_type(self):
        """Items property should be an array."""
        result = _array_response("List", "Agent")
        items_prop = result["content"]["application/json"]["schema"]["properties"]["items"]
        assert items_prop["type"] == "array"

    def test_includes_total(self):
        """Should include total property."""
        result = _array_response("List", "Agent")
        props = result["content"]["application/json"]["schema"]["properties"]
        assert "total" in props
        assert props["total"]["type"] == "integer"

    def test_with_schema_reference(self):
        """Should handle schema reference."""
        result = _array_response("Debates", "Debate")
        items_schema = result["content"]["application/json"]["schema"]["properties"]["items"][
            "items"
        ]
        assert items_schema["$ref"] == "#/components/schemas/Debate"

    def test_with_inline_schema(self):
        """Should handle inline schema dict."""
        inline_schema = {"id": {"type": "string"}, "name": {"type": "string"}}
        result = _array_response("Custom list", inline_schema)
        items_schema = result["content"]["application/json"]["schema"]["properties"]["items"][
            "items"
        ]
        assert items_schema["type"] == "object"
        assert "id" in items_schema["properties"]


# =============================================================================
# TestErrorResponseHelper
# =============================================================================


class TestErrorResponseHelper:
    """Tests for _error_response helper function."""

    def test_creates_error_structure(self):
        """Should create error response structure."""
        result = _error_response("400", "Bad request")
        assert result["description"] == "Bad request"
        assert "content" in result

    def test_references_error_schema(self):
        """Should reference Error schema."""
        result = _error_response("404", "Not found")
        schema = result["content"]["application/json"]["schema"]
        assert schema["$ref"] == "#/components/schemas/Error"

    def test_includes_examples_when_available(self):
        """Should include examples from ERROR_EXAMPLES."""
        result = _error_response("400", "Bad request")
        content = result["content"]["application/json"]
        assert "examples" in content
        assert "invalid_json" in content["examples"]

    def test_no_examples_for_unknown_status(self):
        """Should handle status without examples gracefully."""
        result = _error_response("418", "I'm a teapot")
        content = result["content"]["application/json"]
        assert "examples" not in content


# =============================================================================
# TestRateLimitedEndpoint
# =============================================================================


class TestRateLimitedEndpoint:
    """Tests for _rate_limited_endpoint helper function."""

    def test_adds_rate_limit_to_description(self):
        """Should add rate limit info to description."""
        operation = {"description": "Get agents"}
        result = _rate_limited_endpoint(operation, tier="free")
        assert "Rate Limit" in result["description"]
        assert "60" in result["description"]  # free tier default
        assert "free" in result["description"]

    def test_handles_no_description(self):
        """Should work when operation has no description."""
        operation = {}
        result = _rate_limited_endpoint(operation, tier="pro")
        assert "description" in result
        assert "Rate Limit" in result["description"]

    def test_uses_tier_limits(self):
        """Should use correct tier limits."""
        operation = {"description": "Test"}
        result = _rate_limited_endpoint(operation, tier="pro")
        assert "300" in result["description"]  # pro tier

    def test_custom_limit_override(self):
        """Should allow custom limit override."""
        operation = {"description": "Special endpoint"}
        result = _rate_limited_endpoint(operation, tier="free", custom_limit=10)
        assert "10" in result["description"]

    def test_adds_headers_to_success_responses(self):
        """Should add rate limit headers to 2xx responses."""
        operation = {
            "description": "Test",
            "responses": {
                "200": {"description": "OK"},
                "400": {"description": "Error"},
            },
        }
        result = _rate_limited_endpoint(operation, tier="free")
        assert "headers" in result["responses"]["200"]
        assert "X-RateLimit-Limit" in result["responses"]["200"]["headers"]
        assert "X-RateLimit-Remaining" in result["responses"]["200"]["headers"]
        assert "headers" not in result["responses"]["400"]

    def test_different_windows(self):
        """Should support different time windows."""
        operation = {"description": "Test"}
        result = _rate_limited_endpoint(operation, tier="free", window="hour")
        assert "hour" in result["description"]
        assert "1000" in result["description"]  # free tier per hour


# =============================================================================
# TestAuthRequirements
# =============================================================================


class TestAuthRequirements:
    """Tests for AUTH_REQUIREMENTS definitions."""

    def test_has_none_level(self):
        """Should have 'none' auth level."""
        assert "none" in AUTH_REQUIREMENTS
        assert AUTH_REQUIREMENTS["none"]["security"] == []

    def test_has_optional_level(self):
        """Should have 'optional' auth level."""
        assert "optional" in AUTH_REQUIREMENTS
        # Empty dict in security array means optional
        assert {} in AUTH_REQUIREMENTS["optional"]["security"]

    def test_has_required_level(self):
        """Should have 'required' auth level."""
        assert "required" in AUTH_REQUIREMENTS
        security = AUTH_REQUIREMENTS["required"]["security"]
        assert len(security) > 0
        assert "bearerAuth" in security[0]

    def test_has_admin_level(self):
        """Should have 'admin' auth level."""
        assert "admin" in AUTH_REQUIREMENTS
        security = AUTH_REQUIREMENTS["admin"]["security"]
        assert "bearerAuth" in security[0]
        assert "admin" in security[0]["bearerAuth"]

    def test_all_have_descriptions(self):
        """All auth levels should have descriptions."""
        for level, config in AUTH_REQUIREMENTS.items():
            assert "description" in config, f"Auth level '{level}' missing description"


# =============================================================================
# TestStandardErrorsWithExamples
# =============================================================================


class TestStandardErrorsWithExamples:
    """Tests for STANDARD_ERRORS in helpers module (with examples)."""

    @pytest.mark.parametrize("status", ["400", "401", "403", "404", "402", "429", "500"])
    def test_standard_error_exists(self, status: str):
        """Should have all standard error codes."""
        assert status in STANDARD_ERRORS

    def test_errors_have_descriptions(self):
        """All errors should have descriptions."""
        for status, response in STANDARD_ERRORS.items():
            assert "description" in response
            assert len(response["description"]) > 0

    def test_errors_have_examples(self):
        """Errors with examples should include them."""
        # 400 should have examples
        response = STANDARD_ERRORS["400"]
        examples = response["content"]["application/json"].get("examples")
        assert examples is not None
        assert len(examples) > 0

    def test_error_reference_error_schema(self):
        """All errors should reference Error schema."""
        for status, response in STANDARD_ERRORS.items():
            schema = response["content"]["application/json"]["schema"]
            assert schema["$ref"] == "#/components/schemas/Error"
