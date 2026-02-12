"""
Security handlers for Aragora server.

Provides endpoints for security monitoring and reporting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from aragora.server.handlers.utils.responses import HandlerResult, json_response

logger = logging.getLogger(__name__)


@dataclass
class CSPViolation:
    """A Content-Security-Policy violation report."""

    document_uri: str
    violated_directive: str
    blocked_uri: str
    source_file: str | None = None
    line_number: int | None = None
    column_number: int | None = None
    original_policy: str | None = None
    disposition: str | None = None
    referrer: str | None = None
    status_code: int | None = None

    @classmethod
    def from_report(cls, report: dict[str, Any]) -> CSPViolation:
        """Parse a CSP violation from browser report format.

        Browsers send CSP violations in slightly different formats.
        This handles both the standard format and Chrome's format.
        """
        # Chrome wraps the report in a "csp-report" key
        data = report.get("csp-report", report)

        return cls(
            document_uri=data.get("document-uri", data.get("documentURI", "")),
            violated_directive=data.get("violated-directive", data.get("violatedDirective", "")),
            blocked_uri=data.get("blocked-uri", data.get("blockedURL", "")),
            source_file=data.get("source-file", data.get("sourceFile")),
            line_number=data.get("line-number", data.get("lineNumber")),
            column_number=data.get("column-number", data.get("columnNumber")),
            original_policy=data.get("original-policy", data.get("originalPolicy")),
            disposition=data.get("disposition"),
            referrer=data.get("referrer"),
            status_code=data.get("status-code", data.get("statusCode")),
        )


class CSPReportHandler:
    """Handler for Content-Security-Policy violation reports.

    Receives and logs CSP violations from browsers when they detect
    policy violations. This helps identify XSS attempts and
    misconfigured CSP policies.

    The handler accepts POST requests with:
    - Content-Type: application/csp-report (standard)
    - Content-Type: application/json (fallback)

    Example usage:
        handler = CSPReportHandler()

        # In route registration:
        routes["/api/csp-report"] = handler.handle_report
    """

    def __init__(
        self,
        ctx: dict | None = None,
        *,
        log_violations: bool = True,
        store_violations: bool = False,
        max_violations_stored: int = 1000,
        on_violation: Any | None = None,
    ) -> None:
        """Initialize the CSP report handler.

        Args:
            ctx: Handler context (unused, for compatibility)
            log_violations: Whether to log violations to the application logger
            store_violations: Whether to store violations in memory for analysis
            max_violations_stored: Maximum violations to keep in memory
            on_violation: Optional callback called for each violation
        """
        self.ctx = ctx or {}
        self.log_violations = log_violations
        self.store_violations = store_violations
        self.max_violations_stored = max_violations_stored
        self.on_violation = on_violation
        self._violations: list[CSPViolation] = []

    async def handle_report(
        self,
        body: bytes,
        headers: dict[str, str] | None = None,
    ) -> HandlerResult:
        """Handle a CSP violation report from a browser.

        Args:
            body: Raw request body (JSON)
            headers: Request headers

        Returns:
            Empty 204 response (CSP reports don't need response data)
        """
        try:
            report = json.loads(body.decode("utf-8"))
            violation = CSPViolation.from_report(report)

            if self.log_violations:
                logger.warning(
                    "CSP violation: directive=%s blocked=%s source=%s:%s",
                    violation.violated_directive,
                    violation.blocked_uri,
                    violation.source_file,
                    violation.line_number,
                )

            if self.store_violations:
                self._violations.append(violation)
                # Trim if we have too many
                if len(self._violations) > self.max_violations_stored:
                    self._violations = self._violations[-self.max_violations_stored :]

            if self.on_violation:
                try:
                    if hasattr(self.on_violation, "__call__"):
                        result = self.on_violation(violation)
                        if hasattr(result, "__await__"):
                            await result
                except Exception as e:
                    logger.error("Error in CSP violation callback: %s", e)

        except json.JSONDecodeError:
            logger.warning("Invalid JSON in CSP report")
        except Exception as e:
            logger.error("Error processing CSP report: %s", e)

        # Always return 204 No Content - CSP reports are fire-and-forget
        return HandlerResult(
            status_code=204,
            content_type="text/plain",
            body=b"",
            headers={},
        )

    async def get_violations(self) -> HandlerResult:
        """Get stored CSP violations.

        Returns:
            JSON response with violation data
        """
        if not self.store_violations:
            return json_response(
                {"error": "Violation storage not enabled"},
                status=400,
            )

        violations = [
            {
                "document_uri": v.document_uri,
                "violated_directive": v.violated_directive,
                "blocked_uri": v.blocked_uri,
                "source_file": v.source_file,
                "line_number": v.line_number,
            }
            for v in self._violations
        ]

        return json_response(
            {
                "violations": violations,
                "count": len(violations),
            }
        )

    def clear_violations(self) -> None:
        """Clear stored violations."""
        self._violations.clear()


# Default handler instance
csp_report_handler = CSPReportHandler(log_violations=True, store_violations=True)

__all__ = [
    "CSPViolation",
    "CSPReportHandler",
    "csp_report_handler",
]
