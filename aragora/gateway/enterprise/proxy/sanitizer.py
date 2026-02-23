"""
Request sanitization for the enterprise proxy.

Provides security-focused sanitization of HTTP headers and body content
for both outgoing requests and audit logging, including pattern-based
redaction and injection detection.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

from .config import SanitizationSettings
from .exceptions import SanitizationError
from .models import ProxyRequest

logger = logging.getLogger(__name__)


class RequestSanitizer:
    """Sanitizes requests and responses for security and logging."""

    def __init__(self, settings: SanitizationSettings) -> None:
        """Initialize sanitizer.

        Args:
            settings: Sanitization settings.
        """
        self.settings = settings
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in settings.redact_body_patterns
        ]

    def sanitize_headers(
        self,
        headers: Mapping[str, str],
        for_logging: bool = False,
    ) -> dict[str, str]:
        """Sanitize headers.

        Args:
            headers: Headers to sanitize.
            for_logging: If True, redact sensitive values for logging.

        Returns:
            Sanitized headers dictionary.
        """
        result = {}
        redact_headers = {h.lower() for h in self.settings.redact_headers}
        strip_headers = {h.lower() for h in self.settings.strip_sensitive_headers}

        for key, value in headers.items():
            key_lower = key.lower()

            # Strip sensitive headers from outgoing requests
            if key_lower in strip_headers:
                continue

            # Redact sensitive values for logging
            if for_logging and key_lower in redact_headers:
                result[key] = "[REDACTED]"
            else:
                result[key] = value

        return result

    def sanitize_body_for_logging(self, body: bytes | None) -> str:
        """Sanitize body content for logging.

        Args:
            body: Request/response body.

        Returns:
            Sanitized body string for logging.
        """
        if body is None:
            return ""

        # Truncate if too large
        if len(body) > self.settings.max_body_log_size:
            try:
                text = body[: self.settings.max_body_log_size].decode("utf-8", errors="replace")
            except (UnicodeDecodeError, ValueError) as e:
                logger.debug("Failed to decode truncated body as UTF-8: %s: %s", type(e).__name__, e)
                text = f"[Binary data, {len(body)} bytes]"
            return f"{text}... [truncated, {len(body)} bytes total]"

        try:
            text = body.decode("utf-8", errors="replace")
        except (UnicodeDecodeError, ValueError) as e:
            logger.debug("Failed to decode body as UTF-8: %s: %s", type(e).__name__, e)
            return f"[Binary data, {len(body)} bytes]"

        # Apply redaction patterns
        for pattern in self._compiled_patterns:
            text = pattern.sub('"[REDACTED]"', text)

        return text

    def validate_request(self, request: ProxyRequest) -> None:
        """Validate request for security issues.

        Args:
            request: Request to validate.

        Raises:
            SanitizationError: If validation fails.
        """
        # Check for header injection attempts
        for key, value in request.headers.items():
            if "\n" in key or "\r" in key or "\n" in value or "\r" in value:
                raise SanitizationError(
                    "Header injection attempt detected",
                    framework=request.framework,
                    details={"header": key},
                )

        # Check URL for common injection patterns
        if request.url:
            suspicious_patterns = ["<script", "javascript:", "data:", "file://"]
            url_lower = request.url.lower()
            for pattern in suspicious_patterns:
                if pattern in url_lower:
                    raise SanitizationError(
                        "Suspicious URL pattern detected",
                        framework=request.framework,
                        details={"pattern": pattern},
                    )


__all__ = [
    "RequestSanitizer",
]
