"""Centralized error sanitization utilities.

Provides consistent error message handling across the server:
- sanitize_error_text: Redacts sensitive data from error strings
- safe_error_message: Maps exceptions to user-friendly messages
"""

import logging
import re

logger = logging.getLogger(__name__)

# Patterns for redacting sensitive data in error messages
_SENSITIVE_PATTERNS = [
    (r'sk-[a-zA-Z0-9]{20,}', '<REDACTED_KEY>'),  # OpenAI API keys
    (r'AIza[a-zA-Z0-9_-]{35}', '<REDACTED_KEY>'),  # Google API keys
    (r'["\']?api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+["\']?', 'api_key=<REDACTED>'),
    (r'["\']?authorization["\']?\s*[:=]\s*["\']?Bearer\s+[\w.-]+["\']?', 'authorization=<REDACTED>'),
    (r'["\']?token["\']?\s*[:=]\s*["\']?[\w.-]+["\']?', 'token=<REDACTED>'),
    (r'["\']?secret["\']?\s*[:=]\s*["\']?[\w-]+["\']?', 'secret=<REDACTED>'),
    (r'x-api-key:\s*[\w-]+', 'x-api-key: <REDACTED>'),
]


def sanitize_error_text(error_text: str, max_length: int = 500) -> str:
    """Sanitize error text to remove potential secrets.

    - Redacts patterns that look like API keys or tokens
    - Truncates to prevent log flooding
    - Preserves useful diagnostic info (status codes, error types)

    Args:
        error_text: Raw error message that may contain secrets
        max_length: Maximum length before truncation

    Returns:
        Sanitized error text safe for logging/display
    """
    sanitized = error_text

    # Apply all redaction patterns
    for pattern, replacement in _SENSITIVE_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    # Truncate long messages
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "... [truncated]"

    return sanitized


def safe_error_message(e: Exception, context: str = "") -> str:
    """Return a sanitized error message for client responses.

    Logs the full error server-side while returning a generic message to clients.
    This prevents information disclosure of internal details like file paths,
    stack traces, or sensitive configuration.

    Args:
        e: The exception that occurred
        context: Optional context string for logging (e.g., "debate creation")

    Returns:
        User-friendly error message safe to return to clients
    """
    # Log full details server-side for debugging
    logger.error(f"Error in {context}: {type(e).__name__}: {e}", exc_info=True)

    # Map common exceptions to user-friendly messages
    error_type = type(e).__name__
    if error_type in ("FileNotFoundError", "OSError"):
        return "Resource not found"
    elif error_type in ("json.JSONDecodeError", "ValueError"):
        return "Invalid data format"
    elif error_type in ("PermissionError",):
        return "Access denied"
    elif error_type in ("TimeoutError", "asyncio.TimeoutError"):
        return "Operation timed out"
    else:
        return "An error occurred"
