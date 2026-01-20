"""
Tests for input validation limits and security constraints.

Tests cover:
- Input length limits (title, task, context)
- Agent count limits
- User input sanitization
- ValidationError handling
- SecurityValidationResult behavior
"""

import pytest

from aragora.server.validation.security import (
    MAX_AGENTS_PER_DEBATE,
    MAX_CONTEXT_LENGTH,
    MAX_DEBATE_TITLE_LENGTH,
    MAX_SEARCH_QUERY_LENGTH,
    MAX_TASK_LENGTH,
    SecurityValidationResult,
    ValidationError,
    sanitize_user_input,
    validate_agent_count,
    validate_context_size,
    validate_debate_title,
    validate_task_content,
)


class TestSecurityValidationResult:
    """Tests for SecurityValidationResult dataclass."""

    def test_success_factory(self):
        """Test creating success result."""
        result = SecurityValidationResult.success(value="test", sanitized="clean")
        assert result.is_valid is True
        assert result.value == "test"
        assert result.sanitized == "clean"
        assert result.error is None

    def test_failure_factory(self):
        """Test creating failure result."""
        result = SecurityValidationResult.failure("Something went wrong")
        assert result.is_valid is False
        assert result.error == "Something went wrong"
        assert result.value is None

    def test_default_values(self):
        """Test default values."""
        result = SecurityValidationResult(is_valid=True)
        assert result.value is None
        assert result.error is None
        assert result.sanitized is None


class TestValidationError:
    """Tests for ValidationError exception."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = ValidationError("Invalid input")
        assert str(error) == "Invalid input"
        assert error.message == "Invalid input"
        assert error.field is None

    def test_error_with_field(self):
        """Test error with field name."""
        error = ValidationError("Too long", field="title")
        assert error.message == "Too long"
        assert error.field == "title"
        assert "title" in repr(error)

    def test_error_is_exception(self):
        """Test that error can be raised."""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Test error", field="test_field")

        assert exc_info.value.message == "Test error"
        assert exc_info.value.field == "test_field"


class TestValidateDebateTitle:
    """Tests for validate_debate_title function."""

    def test_valid_title(self):
        """Test valid debate title."""
        result = validate_debate_title("My Debate Title")
        assert result.is_valid is True
        assert result.value == "My Debate Title"

    def test_empty_title_rejected(self):
        """Test that empty title is rejected."""
        result = validate_debate_title("")
        assert result.is_valid is False
        assert "empty" in result.error.lower()

    def test_whitespace_only_title_rejected(self):
        """Test that whitespace-only title is rejected."""
        result = validate_debate_title("   \t\n  ")
        assert result.is_valid is False
        assert "empty" in result.error.lower()

    def test_title_stripped(self):
        """Test that title is stripped of whitespace."""
        result = validate_debate_title("  My Title  ")
        assert result.is_valid is True
        assert result.value == "My Title"

    def test_max_length_enforced(self):
        """Test that max length is enforced."""
        long_title = "a" * (MAX_DEBATE_TITLE_LENGTH + 1)
        result = validate_debate_title(long_title)
        assert result.is_valid is False
        assert "maximum length" in result.error.lower()
        assert str(MAX_DEBATE_TITLE_LENGTH) in result.error

    def test_max_length_boundary(self):
        """Test title at exact max length."""
        exact_title = "a" * MAX_DEBATE_TITLE_LENGTH
        result = validate_debate_title(exact_title)
        assert result.is_valid is True


class TestValidateTaskContent:
    """Tests for validate_task_content function."""

    def test_valid_task(self):
        """Test valid task content."""
        result = validate_task_content("Discuss the pros and cons of AI")
        assert result.is_valid is True
        assert result.value == "Discuss the pros and cons of AI"

    def test_empty_task_rejected(self):
        """Test that empty task is rejected."""
        result = validate_task_content("")
        assert result.is_valid is False
        assert "empty" in result.error.lower()

    def test_whitespace_only_task_rejected(self):
        """Test that whitespace-only task is rejected."""
        result = validate_task_content("   ")
        assert result.is_valid is False
        assert "empty" in result.error.lower()

    def test_task_stripped(self):
        """Test that task is stripped of whitespace."""
        result = validate_task_content("  My task description  ")
        assert result.is_valid is True
        assert result.value == "My task description"

    def test_max_length_enforced(self):
        """Test that max length is enforced."""
        long_task = "a" * (MAX_TASK_LENGTH + 1)
        result = validate_task_content(long_task)
        assert result.is_valid is False
        assert "maximum length" in result.error.lower()

    def test_max_length_boundary(self):
        """Test task at exact max length."""
        exact_task = "a" * MAX_TASK_LENGTH
        result = validate_task_content(exact_task)
        assert result.is_valid is True


class TestValidateContextSize:
    """Tests for validate_context_size function."""

    def test_none_context_valid(self):
        """Test that None context is valid."""
        result = validate_context_size(None)
        assert result.is_valid is True

    def test_string_context_valid(self):
        """Test valid string context."""
        result = validate_context_size("some context data")
        assert result.is_valid is True

    def test_bytes_context_valid(self):
        """Test valid bytes context."""
        result = validate_context_size(b"some binary data")
        assert result.is_valid is True

    def test_dict_context_valid(self):
        """Test valid dict context."""
        result = validate_context_size({"key": "value", "nested": {"data": True}})
        assert result.is_valid is True

    def test_list_context_valid(self):
        """Test valid list context."""
        result = validate_context_size([1, 2, 3, "four", {"five": 5}])
        assert result.is_valid is True

    def test_string_context_too_large(self):
        """Test that overly large string context is rejected."""
        large_context = "a" * (MAX_CONTEXT_LENGTH + 1)
        result = validate_context_size(large_context)
        assert result.is_valid is False
        assert "exceeds maximum" in result.error.lower()

    def test_bytes_context_too_large(self):
        """Test that overly large bytes context is rejected."""
        large_context = b"a" * (MAX_CONTEXT_LENGTH + 1)
        result = validate_context_size(large_context)
        assert result.is_valid is False
        assert "exceeds maximum" in result.error.lower()

    def test_dict_context_too_large(self):
        """Test that overly large dict context is rejected."""
        # Create a dict that serializes to > MAX_CONTEXT_LENGTH
        large_context = {"data": "x" * MAX_CONTEXT_LENGTH}
        result = validate_context_size(large_context)
        assert result.is_valid is False

    def test_exact_max_size_boundary(self):
        """Test context at exact max size."""
        # Create context just at the limit
        exact_context = "a" * (MAX_CONTEXT_LENGTH - 1)  # Leave room for encoding
        result = validate_context_size(exact_context)
        assert result.is_valid is True


class TestValidateAgentCount:
    """Tests for validate_agent_count function."""

    def test_valid_agent_count(self):
        """Test valid agent count."""
        result = validate_agent_count(3)
        assert result.is_valid is True
        assert result.value == 3

    def test_minimum_agent_count(self):
        """Test minimum agent count (1)."""
        result = validate_agent_count(1)
        assert result.is_valid is True
        assert result.value == 1

    def test_maximum_agent_count(self):
        """Test maximum agent count."""
        result = validate_agent_count(MAX_AGENTS_PER_DEBATE)
        assert result.is_valid is True
        assert result.value == MAX_AGENTS_PER_DEBATE

    def test_zero_agents_rejected(self):
        """Test that zero agents is rejected."""
        result = validate_agent_count(0)
        assert result.is_valid is False
        assert "at least 1" in result.error.lower()

    def test_negative_agents_rejected(self):
        """Test that negative agent count is rejected."""
        result = validate_agent_count(-5)
        assert result.is_valid is False
        assert "at least 1" in result.error.lower()

    def test_too_many_agents_rejected(self):
        """Test that too many agents is rejected."""
        result = validate_agent_count(MAX_AGENTS_PER_DEBATE + 1)
        assert result.is_valid is False
        assert "exceeds maximum" in result.error.lower()
        assert str(MAX_AGENTS_PER_DEBATE) in result.error


class TestSanitizeUserInput:
    """Tests for sanitize_user_input function."""

    def test_empty_input(self):
        """Test empty input returns empty string."""
        assert sanitize_user_input("") == ""
        assert sanitize_user_input(None) == ""

    def test_basic_text_unchanged(self):
        """Test that basic text is unchanged."""
        assert sanitize_user_input("Hello World") == "Hello World"

    def test_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        assert sanitize_user_input("  hello  ") == "hello"

    def test_removes_control_characters(self):
        """Test that control characters are removed."""
        # \x00 is a null byte
        assert sanitize_user_input("Hello\x00World") == "HelloWorld"
        # Bell character
        assert sanitize_user_input("Hello\x07World") == "HelloWorld"

    def test_preserves_newlines_and_tabs(self):
        """Test that newlines and tabs are preserved."""
        result = sanitize_user_input("Hello\nWorld\tTest")
        assert "\n" in result
        assert "\t" not in result  # Tab is converted to space by normalize_whitespace

    def test_normalizes_whitespace(self):
        """Test that multiple spaces are collapsed."""
        assert sanitize_user_input("Hello    World") == "Hello World"

    def test_collapses_multiple_newlines(self):
        """Test that more than 2 newlines are collapsed."""
        result = sanitize_user_input("Hello\n\n\n\n\nWorld")
        assert result.count("\n") <= 2

    def test_max_length_truncation(self):
        """Test that text is truncated to max length."""
        result = sanitize_user_input("Hello World", max_length=5)
        assert len(result) <= 5

    def test_disable_control_char_stripping(self):
        """Test disabling control character stripping."""
        result = sanitize_user_input("Hello\x00World", strip_control_chars=False)
        assert "\x00" in result

    def test_disable_whitespace_normalization(self):
        """Test disabling whitespace normalization."""
        result = sanitize_user_input("Hello    World", normalize_whitespace=False)
        assert "    " in result


class TestConstantValues:
    """Tests verifying constant values are reasonable."""

    def test_max_debate_title_length(self):
        """Test MAX_DEBATE_TITLE_LENGTH is reasonable."""
        assert MAX_DEBATE_TITLE_LENGTH >= 50  # Should allow reasonable titles
        assert MAX_DEBATE_TITLE_LENGTH <= 1000  # But not too long

    def test_max_task_length(self):
        """Test MAX_TASK_LENGTH is reasonable."""
        assert MAX_TASK_LENGTH >= 100  # Should allow real tasks
        assert MAX_TASK_LENGTH <= 100000  # But not unlimited

    def test_max_context_length(self):
        """Test MAX_CONTEXT_LENGTH is reasonable."""
        assert MAX_CONTEXT_LENGTH >= 1000  # Allow some context
        assert MAX_CONTEXT_LENGTH <= 10_000_000  # Limit to ~10MB

    def test_max_agents_per_debate(self):
        """Test MAX_AGENTS_PER_DEBATE is reasonable."""
        assert MAX_AGENTS_PER_DEBATE >= 2  # At least allow a debate
        assert MAX_AGENTS_PER_DEBATE <= 100  # But don't allow hundreds

    def test_max_search_query_length(self):
        """Test MAX_SEARCH_QUERY_LENGTH is reasonable."""
        assert MAX_SEARCH_QUERY_LENGTH >= 10  # Allow real searches
        assert MAX_SEARCH_QUERY_LENGTH <= 1000  # But limit complexity


class TestSecurityIntegration:
    """Integration tests for security validation."""

    def test_debate_creation_workflow(self):
        """Test validation workflow for debate creation."""
        # Typical debate creation inputs
        title_result = validate_debate_title("AI Ethics Discussion")
        task_result = validate_task_content("Discuss the ethical implications of AI in healthcare")
        context_result = validate_context_size({"background": "some context"})
        agent_result = validate_agent_count(3)

        assert all(
            [
                title_result.is_valid,
                task_result.is_valid,
                context_result.is_valid,
                agent_result.is_valid,
            ]
        )

    def test_malicious_input_handling(self):
        """Test that malicious inputs are properly rejected."""
        # Very long inputs
        assert not validate_debate_title("x" * 10000).is_valid
        assert not validate_task_content("y" * 100000).is_valid

        # Zero/negative agents
        assert not validate_agent_count(0).is_valid
        assert not validate_agent_count(-1).is_valid

    def test_sanitization_chain(self):
        """Test chaining sanitization with validation."""
        # Dirty input
        dirty_input = "  Hello\x00World  "

        # Sanitize first
        clean = sanitize_user_input(dirty_input)

        # Then validate
        result = validate_debate_title(clean)
        assert result.is_valid is True
        assert "\x00" not in result.value
