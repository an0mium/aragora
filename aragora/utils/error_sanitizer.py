"""
Error sanitization utilities.

Provides centralized sensitive data redaction for error messages.
Used by both agents and server components.
"""

import re


# Patterns for redacting sensitive data in error messages
SENSITIVE_PATTERNS = [
    # API keys
    (r"sk-[a-zA-Z0-9]{20,}", "<REDACTED_KEY>"),  # OpenAI API keys
    (r"AIza[a-zA-Z0-9_-]{35}", "<REDACTED_KEY>"),  # Google API keys
    (r"xai-[a-zA-Z0-9]{20,}", "<REDACTED_KEY>"),  # xAI/Grok API keys
    (r"key-[a-zA-Z0-9]{20,}", "<REDACTED_KEY>"),  # Generic key pattern
    # Key-value patterns
    (r'["\']?api[_-]?key["\']?\s*[:=]\s*["\']?[\w-]+["\']?', "api_key=<REDACTED>"),
    (
        r'["\']?authorization["\']?\s*[:=]\s*["\']?Bearer\s+[\w.-]+["\']?',
        "authorization=<REDACTED>",
    ),
    (r'["\']?token["\']?\s*[:=]\s*["\']?[\w.-]+["\']?', "token=<REDACTED>"),
    (r'["\']?secret["\']?\s*[:=]\s*["\']?[\w-]+["\']?', "secret=<REDACTED>"),
    (r'["\']?password["\']?\s*[:=]\s*["\']?[\w-]+["\']?', "password=<REDACTED>"),
    # Header patterns
    (r"x-api-key:\s*[\w-]+", "x-api-key: <REDACTED>"),
    (r"x-goog-api-key:\s*[\w-]+", "x-goog-api-key: <REDACTED>"),
    (r"Authorization:\s*Bearer\s+[\w.-]+", "Authorization: Bearer <REDACTED>"),
]


def sanitize_error(error_text: str, max_length: int = 500) -> str:
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
    sanitized = str(error_text)

    # Apply all redaction patterns
    for pattern, replacement in SENSITIVE_PATTERNS:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    # Truncate long messages
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "... [truncated]"

    return sanitized


# Alias for backwards compatibility
sanitize_error_text = sanitize_error
