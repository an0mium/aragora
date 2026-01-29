"""Tests for validation decorators."""

import pytest

from aragora.server.validation.decorators import (
    validate_post_body,
    validate_query_params,
    validate_request,
)


class MockHandler:
    """Mock handler class for testing decorators."""

    pass


# ============================================================================
# Tests for validate_request decorator
# ============================================================================


class TestValidateRequest:
    """Tests for the validate_request decorator."""

    def test_passes_with_no_validation(self):
        """Should pass through when no validation is configured."""

        class Handler:
            @validate_request()
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/test", {}, MockHandler())
        assert result == {"success": True}

    def test_required_params_missing(self):
        """Should return error when required param is missing."""

        class Handler:
            @validate_request(required_params=["task"])
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/test", {}, MockHandler())
        assert result["status"] == 400
        assert "Missing required parameter: task" in result["error"]

    def test_required_params_present(self):
        """Should pass when required param is present."""

        class Handler:
            @validate_request(required_params=["task"])
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/test", {"task": "test task"}, MockHandler())
        assert result == {"success": True}

    def test_required_params_empty_list(self):
        """Should return error when required param is empty list."""

        class Handler:
            @validate_request(required_params=["items"])
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/test", {"items": []}, MockHandler())
        assert result["status"] == 400
        assert "Missing required parameter: items" in result["error"]

    def test_path_validator_debate_id_valid(self):
        """Should pass with valid debate_id."""

        def validate_id(val):
            return (True, None) if len(val) > 0 else (False, "Invalid ID")

        class Handler:
            @validate_request(path_validators={"debate_id": validate_id})
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/debates/abc123", {}, MockHandler())
        assert result == {"success": True}

    def test_path_validator_debate_id_invalid(self):
        """Should return error with invalid debate_id."""

        def validate_id(val):
            return (False, "Invalid debate ID format")

        class Handler:
            @validate_request(path_validators={"debate_id": validate_id})
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/debates/bad", {}, MockHandler())
        assert result["status"] == 400
        assert "Invalid debate ID format" in result["error"]

    def test_path_validator_agent_valid(self):
        """Should pass with valid agent name."""

        def validate_agent(val):
            return (True, None) if val.isalnum() else (False, "Invalid agent name")

        class Handler:
            @validate_request(path_validators={"agent": validate_agent})
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/agent/claude", {}, MockHandler())
        assert result == {"success": True}

    def test_path_validator_from_kwargs(self):
        """Should use path segment from kwargs if provided."""

        def validate_id(val):
            return (True, None) if val == "valid" else (False, "Invalid")

        class Handler:
            @validate_request(path_validators={"custom_id": validate_id})
            def handle(self, path, query, handler, custom_id=None):
                return {"success": True, "id": custom_id}

        h = Handler()
        result = h.handle("/api/test", {}, MockHandler(), custom_id="valid")
        assert result["success"] is True

    def test_schema_validation_valid(self):
        """Should pass with valid body against schema."""
        # Custom schema format: field -> {type, required, ...}
        schema = {
            "task": {"type": "string", "required": True, "min_length": 1},
        }

        class Handler:
            @validate_request(schema=schema)
            def handle(self, path, query, body, handler):
                return {"success": True, "task": body["task"]}

        h = Handler()
        result = h.handle("/api/test", {}, {"task": "test"}, MockHandler())
        assert result["success"] is True

    def test_schema_validation_invalid(self):
        """Should return error with invalid body."""
        schema = {
            "task": {"type": "string", "required": True, "min_length": 1},
        }

        class Handler:
            @validate_request(schema=schema)
            def handle(self, path, query, body, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/test", {}, {"not_task": "value"}, MockHandler())
        assert result["status"] == 400

    def test_kwargs_style_call(self):
        """Should work with kwargs-style calls."""

        class Handler:
            @validate_request(required_params=["task"])
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle(path="/api/test", query={"task": "test"}, handler=MockHandler())
        assert result == {"success": True}


# ============================================================================
# Tests for validate_post_body decorator
# ============================================================================


class TestValidatePostBody:
    """Tests for the validate_post_body decorator."""

    def test_valid_body(self):
        """Should pass with valid body."""
        # Custom schema format: field -> {type, required, ...}
        schema = {
            "name": {"type": "string", "required": True, "min_length": 1},
        }

        class Handler:
            @validate_post_body(schema)
            def handle(self, body, handler):
                return {"success": True, "name": body["name"]}

        h = Handler()
        result = h.handle({"name": "test"}, MockHandler())
        assert result["success"] is True
        assert result["name"] == "test"

    def test_invalid_body_missing_required(self):
        """Should return error when required field is missing."""
        schema = {
            "name": {"type": "string", "required": True, "min_length": 1},
        }

        class Handler:
            @validate_post_body(schema)
            def handle(self, body, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({}, MockHandler())
        assert result["status"] == 400

    def test_non_dict_body(self):
        """Should return error when body is not a dict."""
        # Empty schema - just checking body is dict
        schema = {}

        class Handler:
            @validate_post_body(schema)
            def handle(self, body, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("not a dict", MockHandler())
        assert result["status"] == 400
        assert "must be a JSON object" in result["error"]

    def test_body_from_kwargs(self):
        """Should work with body passed as kwarg."""
        schema = {
            "task": {"type": "string", "required": True, "min_length": 1},
        }

        class Handler:
            @validate_post_body(schema)
            def handle(self, body, handler):
                return {"success": True}

        h = Handler()
        result = h.handle(body={"task": "test"}, handler=MockHandler())
        assert result["success"] is True


# ============================================================================
# Tests for validate_query_params decorator
# ============================================================================


class TestValidateQueryParams:
    """Tests for the validate_query_params decorator."""

    def test_no_validation(self):
        """Should pass through with no validation configured."""

        class Handler:
            @validate_query_params()
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({}, MockHandler())
        assert result == {"success": True}

    def test_required_missing(self):
        """Should return error when required param is missing."""

        class Handler:
            @validate_query_params(required=["agent"])
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({}, MockHandler())
        assert result["status"] == 400
        assert "Missing required parameter: agent" in result["error"]

    def test_required_present(self):
        """Should pass when required param is present."""

        class Handler:
            @validate_query_params(required=["agent"])
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"agent": "claude"}, MockHandler())
        assert result == {"success": True}

    def test_required_empty_list(self):
        """Should return error when required param is empty list."""

        class Handler:
            @validate_query_params(required=["ids"])
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"ids": []}, MockHandler())
        assert result["status"] == 400

    def test_int_param_valid(self):
        """Should pass with valid int param."""

        class Handler:
            @validate_query_params(int_params={"limit": (10, 1, 100)})
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"limit": "50"}, MockHandler())
        assert result == {"success": True}

    def test_int_param_below_min(self):
        """Should return error when int param below minimum."""

        class Handler:
            @validate_query_params(int_params={"limit": (10, 1, 100)})
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"limit": "0"}, MockHandler())
        assert result["status"] == 400
        assert "must be between 1 and 100" in result["error"]

    def test_int_param_above_max(self):
        """Should return error when int param above maximum."""

        class Handler:
            @validate_query_params(int_params={"limit": (10, 1, 100)})
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"limit": "200"}, MockHandler())
        assert result["status"] == 400
        assert "must be between 1 and 100" in result["error"]

    def test_int_param_not_integer(self):
        """Should return error when int param is not valid integer."""

        class Handler:
            @validate_query_params(int_params={"limit": (10, 1, 100)})
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"limit": "abc"}, MockHandler())
        assert result["status"] == 400
        assert "must be an integer" in result["error"]

    def test_int_param_from_list(self):
        """Should extract int param from list (common in query strings)."""

        class Handler:
            @validate_query_params(int_params={"limit": (10, 1, 100)})
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"limit": ["25"]}, MockHandler())
        assert result == {"success": True}

    def test_string_param_valid(self):
        """Should pass with valid string param."""

        class Handler:
            @validate_query_params(string_params={"sort": ("created_at", 64)})
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"sort": "updated_at"}, MockHandler())
        assert result == {"success": True}

    def test_string_param_too_long(self):
        """Should return error when string param exceeds max length."""

        class Handler:
            @validate_query_params(string_params={"sort": ("default", 10)})
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"sort": "this_is_way_too_long"}, MockHandler())
        assert result["status"] == 400
        assert "exceeds maximum length of 10" in result["error"]

    def test_string_param_from_list(self):
        """Should extract string param from list."""

        class Handler:
            @validate_query_params(string_params={"sort": ("default", 64)})
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"sort": ["created_at"]}, MockHandler())
        assert result == {"success": True}

    def test_string_param_empty_list_uses_default(self):
        """Should use default when string param is empty list."""

        class Handler:
            @validate_query_params(string_params={"sort": ("default", 64)})
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"sort": []}, MockHandler())
        assert result == {"success": True}

    def test_query_from_positional_args(self):
        """Should extract query from positional args."""

        class Handler:
            @validate_query_params(required=["task"])
            def handle(self, path, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle("/api/test", {"task": "value"}, MockHandler())
        assert result == {"success": True}

    def test_combined_validation(self):
        """Should validate all param types together."""

        class Handler:
            @validate_query_params(
                required=["agent"],
                int_params={"limit": (10, 1, 100)},
                string_params={"sort": ("created_at", 64)},
            )
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        result = h.handle({"agent": "claude", "limit": "50", "sort": "name"}, MockHandler())
        assert result == {"success": True}

    def test_combined_validation_fails_on_first_error(self):
        """Should return error for first validation failure."""

        class Handler:
            @validate_query_params(
                required=["agent"],
                int_params={"limit": (10, 1, 100)},
            )
            def handle(self, query, handler):
                return {"success": True}

        h = Handler()
        # Missing required, but also invalid int - should fail on required first
        result = h.handle({"limit": "abc"}, MockHandler())
        assert result["status"] == 400
        assert "Missing required parameter: agent" in result["error"]
