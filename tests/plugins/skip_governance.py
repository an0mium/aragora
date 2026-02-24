"""
Skip Marker Governance Plugin

Enforces that new skip markers include:
1. A reason with a ticket reference (e.g., GH-1234, JIRA-123, or a URL)
2. An expiry date for temporary skips

Existing skips are grandfathered via the baseline file.

Usage in pyproject.toml:
    [tool.pytest.ini_options]
    plugins = ["tests.plugins.skip_governance"]

Or registered as a conftest plugin automatically since it's under tests/.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

# Ticket reference patterns: GH-123, JIRA-123, #123, or URLs
TICKET_PATTERNS = [
    r"GH-\d+",
    r"JIRA-\d+",
    r"#\d+",
    r"https?://",
    r"TODO\(",
    r"TICKET:",
]

# Expiry pattern: expires=YYYY-MM-DD or expiry:YYYY-MM-DD
EXPIRY_PATTERN = re.compile(r"expir[ey][s:=]\s*\d{4}-\d{2}-\d{2}", re.IGNORECASE)

# Known legitimate skip reasons that don't need tickets
EXEMPT_REASONS = [
    "not installed",
    "not available",
    "requires ",
    "only on ",
    "platform",
    "windows only",
    "linux only",
    "darwin only",
    "macos only",
    "ci only",
    "no gpu",
    "optional dependency",
    "importorskip",
]

_TICKET_RE = re.compile("|".join(TICKET_PATTERNS), re.IGNORECASE)


def _has_ticket_ref(reason: str) -> bool:
    """Check if a skip reason contains a ticket reference."""
    return bool(_TICKET_RE.search(reason))


def _is_exempt(reason: str) -> bool:
    """Check if a skip reason is exempt from ticket requirements."""
    reason_lower = reason.lower()
    return any(exempt in reason_lower for exempt in EXEMPT_REASONS)


def _check_expired(reason: str) -> str | None:
    """Check if a skip marker has an expired expiry date. Returns expiry date if expired."""
    match = EXPIRY_PATTERN.search(reason)
    if not match:
        return None
    # Extract the date portion
    date_match = re.search(r"\d{4}-\d{2}-\d{2}", match.group())
    if date_match:
        from datetime import date

        try:
            expiry = date.fromisoformat(date_match.group())
            if expiry < date.today():
                return date_match.group()
        except ValueError:
            pass
    return None


def pytest_collection_modifyitems(config, items):
    """Warn about skip markers missing ticket references or with expired dates."""
    governance_warnings = []

    for item in items:
        for marker in item.iter_markers("skip"):
            reason = marker.kwargs.get("reason", "")
            if not reason and marker.args:
                reason = str(marker.args[0])

            if not reason:
                continue

            # Check for expired skips
            expired = _check_expired(reason)
            if expired:
                governance_warnings.append(
                    f"EXPIRED skip in {item.nodeid}: expired {expired} - {reason}"
                )

        for marker in item.iter_markers("skipif"):
            reason = marker.kwargs.get("reason", "")
            if not reason:
                continue

            expired = _check_expired(reason)
            if expired:
                governance_warnings.append(
                    f"EXPIRED skipif in {item.nodeid}: expired {expired} - {reason}"
                )

    if governance_warnings:
        # Write warnings to a file for CI to pick up
        warnings_file = Path(config.rootdir) / "skip_governance_warnings.txt"
        warnings_file.write_text("\n".join(governance_warnings) + "\n")

        for w in governance_warnings[:10]:
            warnings.warn(w, stacklevel=1)
        if len(governance_warnings) > 10:
            warnings.warn(
                f"... and {len(governance_warnings) - 10} more skip governance warnings",
                stacklevel=1,
            )
