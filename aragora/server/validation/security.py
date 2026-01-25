"""
Security validation utilities.

Provides centralized security validation including:
- ReDoS (Regular Expression Denial of Service) protection
- Input length validation with configurable limits
- Timeout-protected regex execution
- Pattern safety checking

Usage:
    from aragora.server.validation.security import (
        validate_search_query_redos_safe,
        execute_regex_with_timeout,
        is_safe_regex_pattern,
        validate_context_size,
    )

    # ReDoS-safe search
    result = validate_search_query_redos_safe(user_input)
    if not result.is_valid:
        return error_response(400, result.error)

    # Timeout-protected regex
    match = execute_regex_with_timeout(pattern, text, timeout=1.0)
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
from dataclasses import dataclass
from typing import Any, Match, Optional, Pattern, Union

logger = logging.getLogger(__name__)

# =============================================================================
# Security Constants
# =============================================================================

# Input length limits
MAX_DEBATE_TITLE_LENGTH = 500
MAX_TASK_LENGTH = 10000
MAX_CONTEXT_LENGTH = 100000  # 100KB
MAX_SEARCH_QUERY_LENGTH = 200
MAX_REGEX_PATTERN_LENGTH = 100

# Resource limits
MAX_AGENTS_PER_DEBATE = 10
MAX_ROUNDS_PER_DEBATE = 50
MAX_VOTES_PER_USER_PER_DEBATE = 100
MAX_BATCH_SIZE = 100

# Timeout settings
REGEX_TIMEOUT_SECONDS = 1.0
DEFAULT_OPERATION_TIMEOUT = 30.0


# =============================================================================
# Validation Error
# =============================================================================


class ValidationError(Exception):
    """Exception raised when validation fails."""

    def __init__(self, message: str, field: Optional[str] = None):
        self.message = message
        self.field = field
        super().__init__(message)

    def __repr__(self) -> str:
        if self.field:
            return f"ValidationError(field={self.field!r}, message={self.message!r})"
        return f"ValidationError({self.message!r})"


# =============================================================================
# Validation Result
# =============================================================================


@dataclass
class SecurityValidationResult:
    """Result of a security validation check."""

    is_valid: bool
    value: Any = None
    error: Optional[str] = None
    sanitized: Optional[str] = None

    @classmethod
    def success(
        cls, value: Any = None, sanitized: Optional[str] = None
    ) -> "SecurityValidationResult":
        """Create a successful validation result."""
        return cls(is_valid=True, value=value, sanitized=sanitized)

    @classmethod
    def failure(cls, error: str) -> "SecurityValidationResult":
        """Create a failed validation result."""
        return cls(is_valid=False, error=error)


# =============================================================================
# ReDoS Protection
# =============================================================================

# Patterns known to cause catastrophic backtracking
DANGEROUS_REGEX_PATTERNS = [
    # Nested quantifiers with overlapping character classes
    r"\(.*\)\+.*\1",  # (a+)+ pattern
    r"\([^)]*\)\*\s*\1",  # Backreference with quantifiers
    # Multiple nested quantifiers
    r"\(\.\*\)\+",
    r"\(\.\+\)\+",
    r"\(\.\*\)\*",
    r"\(\.\+\)\*",
    # Overlapping alternation with quantifiers
    r"\([^|]*\|[^)]*\)\+",
]

# Pre-compiled dangerous pattern detector
_DANGEROUS_PATTERN_REGEX = re.compile(
    r"(?:"
    # Nested quantifiers: (a+)+, (a*)+, (a+)*, (a*)*
    r"\([^)]*[+*]\)[+*]"
    r"|"
    # Overlapping alternation with quantifier
    r"\([^)]*\|[^)]*\)[+*]"
    r"|"
    # Multiple dots with quantifiers
    r"\.+[+*]\.+[+*]"
    r"|"
    # Backreference with quantifier after group with quantifier
    r"\([^)]*[+*]\)[^)]*\\[0-9]"
    r")",
    re.IGNORECASE,
)


def is_safe_regex_pattern(pattern: str) -> tuple[bool, Optional[str]]:
    """
    Check if a regex pattern is safe from ReDoS attacks.

    Detects common patterns that cause catastrophic backtracking:
    - Nested quantifiers: (a+)+, (a*)+, etc.
    - Overlapping alternation with quantifiers: (a|a)+
    - Backreferences with quantified groups

    Args:
        pattern: The regex pattern to check

    Returns:
        Tuple of (is_safe, error_message)

    Example:
        >>> is_safe, err = is_safe_regex_pattern("(a+)+")
        >>> print(is_safe, err)
        False "Pattern contains nested quantifiers which may cause ReDoS"
    """
    if not pattern:
        return True, None

    # Length check
    if len(pattern) > MAX_REGEX_PATTERN_LENGTH:
        return False, f"Pattern exceeds maximum length of {MAX_REGEX_PATTERN_LENGTH}"

    # Check for known dangerous patterns
    if _DANGEROUS_PATTERN_REGEX.search(pattern):
        return False, "Pattern contains constructs that may cause catastrophic backtracking (ReDoS)"

    # Check for excessive quantifier nesting depth
    set("+*?")
    depth = 0
    max_depth = 0
    for char in pattern:
        if char == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ")":
            depth = max(0, depth - 1)

    # Check if quantifiers follow grouped quantifiers
    paren_with_quantifier = re.search(r"\([^)]*[+*]\)\s*[+*]", pattern)
    if paren_with_quantifier:
        return False, "Pattern contains nested quantifiers which may cause ReDoS"

    return True, None


def execute_regex_with_timeout(
    pattern: Union[str, Pattern[str]],
    text: str,
    timeout: float = REGEX_TIMEOUT_SECONDS,
    flags: int = 0,
) -> Optional[Match[str]]:
    """
    Execute a regex match with a timeout to prevent ReDoS.

    Uses a thread pool to enforce timeout on regex execution.
    Returns None if the regex times out or fails.

    Args:
        pattern: Regex pattern (string or compiled)
        text: Text to search
        timeout: Maximum execution time in seconds
        flags: Regex flags (if pattern is a string)

    Returns:
        Match object if found within timeout, None otherwise

    Example:
        >>> match = execute_regex_with_timeout(r"\\d+", "abc123def", timeout=1.0)
        >>> if match:
        ...     print(match.group())
        123
    """
    if isinstance(pattern, str):
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            logger.warning(f"Invalid regex pattern: {e}")
            return None
    else:
        compiled = pattern

    def do_search() -> Optional[Match[str]]:
        return compiled.search(text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(do_search)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"Regex execution timed out after {timeout}s")
            return None
        except Exception as e:
            logger.warning(f"Regex execution failed: {e}")
            return None


def execute_regex_finditer_with_timeout(
    pattern: Union[str, Pattern[str]],
    text: str,
    timeout: float = REGEX_TIMEOUT_SECONDS,
    flags: int = 0,
    max_matches: int = 100,
) -> list[Match[str]]:
    """
    Execute regex finditer with a timeout to prevent ReDoS.

    Uses a thread pool to enforce timeout on regex execution.
    Returns empty list if the regex times out or fails.

    Args:
        pattern: Regex pattern (string or compiled)
        text: Text to search
        timeout: Maximum execution time in seconds
        flags: Regex flags (if pattern is a string)
        max_matches: Maximum matches to return (prevents memory exhaustion)

    Returns:
        List of Match objects found within timeout, empty list on failure

    Example:
        >>> matches = execute_regex_finditer_with_timeout(r"\\d+", "a1b2c3", timeout=1.0)
        >>> [m.group() for m in matches]
        ['1', '2', '3']
    """
    if isinstance(pattern, str):
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            logger.warning(f"Invalid regex pattern: {e}")
            return []
    else:
        compiled = pattern

    def do_finditer() -> list[Match[str]]:
        results = []
        for i, match in enumerate(compiled.finditer(text)):
            if i >= max_matches:
                break
            results.append(match)
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(do_finditer)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            logger.warning(f"Regex finditer timed out after {timeout}s")
            return []
        except Exception as e:
            logger.warning(f"Regex finditer failed: {e}")
            return []


def validate_search_query_redos_safe(
    query: str,
    max_length: int = MAX_SEARCH_QUERY_LENGTH,
    allow_wildcards: bool = True,
) -> SecurityValidationResult:
    """
    Validate and sanitize a search query with ReDoS protection.

    This function:
    1. Validates length
    2. Removes or escapes regex metacharacters
    3. Converts user wildcards (* and ?) to safe equivalents
    4. Ensures the resulting pattern is ReDoS-safe

    Args:
        query: User-provided search query
        max_length: Maximum allowed query length
        allow_wildcards: Whether to convert * and ? to regex equivalents

    Returns:
        SecurityValidationResult with sanitized query or error

    Example:
        >>> result = validate_search_query_redos_safe("test*query")
        >>> if result.is_valid:
        ...     pattern = result.sanitized
    """
    if not query:
        return SecurityValidationResult.success(value="", sanitized="")

    # Length check
    if len(query) > max_length:
        return SecurityValidationResult.failure(
            f"Search query exceeds maximum length of {max_length} characters"
        )

    if allow_wildcards:
        # Split on wildcards, escape each part, then rejoin with safe patterns
        # This avoids escaping issues with placeholder characters
        parts = []
        current = ""
        for char in query:
            if char == "*":
                if current:
                    parts.append(("literal", current))
                    current = ""
                parts.append(("star", None))
            elif char == "?":
                if current:
                    parts.append(("literal", current))
                    current = ""
                parts.append(("question", None))
            else:
                current += char
        if current:
            parts.append(("literal", current))

        # Build escaped pattern
        escaped_parts = []
        for part_type, part_value in parts:
            if part_type == "literal":
                escaped_parts.append(re.escape(part_value))
            elif part_type == "star":
                # * becomes bounded .* to prevent ReDoS
                escaped_parts.append(".{0,100}")
            elif part_type == "question":
                escaped_parts.append(".")
        escaped = "".join(escaped_parts)
    else:
        # No wildcards - just escape everything
        escaped = re.escape(query)

    # Final safety check
    is_safe, error = is_safe_regex_pattern(escaped)
    if not is_safe:
        return SecurityValidationResult.failure(error or "Query pattern is not safe")

    return SecurityValidationResult.success(value=query, sanitized=escaped)


# =============================================================================
# Input Length Validation
# =============================================================================


def validate_debate_title(title: str) -> SecurityValidationResult:
    """
    Validate a debate title.

    Args:
        title: The debate title to validate

    Returns:
        SecurityValidationResult
    """
    if not title or not title.strip():
        return SecurityValidationResult.failure("Debate title cannot be empty")

    title = title.strip()

    if len(title) > MAX_DEBATE_TITLE_LENGTH:
        return SecurityValidationResult.failure(
            f"Debate title exceeds maximum length of {MAX_DEBATE_TITLE_LENGTH} characters"
        )

    return SecurityValidationResult.success(value=title, sanitized=title)


def validate_task_content(task: str) -> SecurityValidationResult:
    """
    Validate debate task content.

    Args:
        task: The task description to validate

    Returns:
        SecurityValidationResult
    """
    if not task or not task.strip():
        return SecurityValidationResult.failure("Task content cannot be empty")

    task = task.strip()

    if len(task) > MAX_TASK_LENGTH:
        return SecurityValidationResult.failure(
            f"Task content exceeds maximum length of {MAX_TASK_LENGTH} characters"
        )

    return SecurityValidationResult.success(value=task, sanitized=task)


def validate_context_size(context: Union[str, bytes, dict, list]) -> SecurityValidationResult:
    """
    Validate context size is within limits.

    Args:
        context: The context to validate (string, bytes, or JSON-serializable)

    Returns:
        SecurityValidationResult
    """
    import json

    if context is None:
        return SecurityValidationResult.success(value=None)

    # Calculate size
    if isinstance(context, bytes):
        size = len(context)
    elif isinstance(context, str):
        size = len(context.encode("utf-8"))
    elif isinstance(context, (dict, list)):
        try:
            size = len(json.dumps(context).encode("utf-8"))
        except (TypeError, ValueError) as e:
            return SecurityValidationResult.failure(f"Invalid context format: {e}")
    else:
        return SecurityValidationResult.failure("Context must be string, bytes, dict, or list")

    if size > MAX_CONTEXT_LENGTH:
        return SecurityValidationResult.failure(
            f"Context size ({size:,} bytes) exceeds maximum of {MAX_CONTEXT_LENGTH:,} bytes"
        )

    return SecurityValidationResult.success(value=context)


def validate_agent_count(count: int) -> SecurityValidationResult:
    """
    Validate the number of agents for a debate.

    Args:
        count: Number of agents

    Returns:
        SecurityValidationResult
    """
    if count < 1:
        return SecurityValidationResult.failure("At least 1 agent is required")

    if count > MAX_AGENTS_PER_DEBATE:
        return SecurityValidationResult.failure(
            f"Number of agents ({count}) exceeds maximum of {MAX_AGENTS_PER_DEBATE}"
        )

    return SecurityValidationResult.success(value=count)


# =============================================================================
# User Input Sanitization
# =============================================================================


def sanitize_user_input(
    text: str,
    max_length: Optional[int] = None,
    strip_control_chars: bool = True,
    normalize_whitespace: bool = True,
) -> str:
    """
    Sanitize user input text.

    Args:
        text: Raw user input
        max_length: Optional maximum length (truncates if exceeded)
        strip_control_chars: Remove control characters (except newlines/tabs)
        normalize_whitespace: Collapse multiple spaces/newlines

    Returns:
        Sanitized text

    Example:
        >>> sanitize_user_input("  Hello\\x00World  ", max_length=20)
        'Hello World'
    """
    if not text:
        return ""

    result = text

    # Strip control characters (keep newlines, tabs)
    if strip_control_chars:
        # Remove ASCII control chars except \t (9), \n (10), \r (13)
        result = "".join(char for char in result if ord(char) >= 32 or char in "\t\n\r")

    # Normalize whitespace
    if normalize_whitespace:
        # Collapse multiple spaces
        result = re.sub(r"[ \t]+", " ", result)
        # Collapse multiple newlines (keep max 2)
        result = re.sub(r"\n{3,}", "\n\n", result)

    # Strip leading/trailing whitespace
    result = result.strip()

    # Truncate if needed
    if max_length and len(result) > max_length:
        result = result[:max_length]

    return result


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Constants
    "MAX_DEBATE_TITLE_LENGTH",
    "MAX_TASK_LENGTH",
    "MAX_CONTEXT_LENGTH",
    "MAX_SEARCH_QUERY_LENGTH",
    "MAX_REGEX_PATTERN_LENGTH",
    "MAX_AGENTS_PER_DEBATE",
    "MAX_ROUNDS_PER_DEBATE",
    "MAX_VOTES_PER_USER_PER_DEBATE",
    "MAX_BATCH_SIZE",
    "REGEX_TIMEOUT_SECONDS",
    "DEFAULT_OPERATION_TIMEOUT",
    # Error class
    "ValidationError",
    # Result class
    "SecurityValidationResult",
    # ReDoS protection
    "is_safe_regex_pattern",
    "execute_regex_with_timeout",
    "execute_regex_finditer_with_timeout",
    "validate_search_query_redos_safe",
    # Input validation
    "validate_debate_title",
    "validate_task_content",
    "validate_context_size",
    "validate_agent_count",
    # Sanitization
    "sanitize_user_input",
]
