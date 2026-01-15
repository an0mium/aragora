"""
Cognitive Firewall: Security and Telemetry Components.

Provides security-related classes for debate orchestration:
- SecurityBarrier: Dynamic secret redaction for telemetry streams
- TelemetryVerifier: Runtime capability verification for telemetry
"""

import re
from typing import Any


class SecurityBarrier:
    """
    Dynamic secret redaction for telemetry streams.

    Identifies and redacts potentially sensitive content from agent thoughts
    before broadcasting to WebSocket clients.
    """

    # Patterns that might indicate sensitive content
    DEFAULT_PATTERNS = [
        # API keys and tokens
        r"(?i)(api[_-]?key|token|secret|password|auth)['\"]?\s*[:=]\s*['\"]?[\w\-]+",
        r"(?i)bearer\s+[\w\-\.]+",
        r"sk-[a-zA-Z0-9\-_]{10,}",  # OpenAI-style keys (sk-proj-, sk-ant-, etc.)
        r"AIza[a-zA-Z0-9_\-]{35}",  # Google API keys
        # Environment variables
        r"(?i)(ANTHROPIC|OPENAI|GEMINI|GROK|XAI)[_\s]*[A-Z_]*\s*=\s*[\w\-]+",
        # URLs with credentials
        r"https?://[^:]+:[^@]+@",
        # Private keys
        r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",
    ]

    def __init__(self, patterns: list[str] | None = None, redaction_marker: str = "[REDACTED]"):
        """
        Initialize SecurityBarrier.

        Args:
            patterns: Regex patterns to identify sensitive content
            redaction_marker: Text to replace sensitive content with
        """
        self._patterns = [re.compile(p) for p in (patterns or self.DEFAULT_PATTERNS)]
        self._redaction_marker = redaction_marker
        self._custom_patterns: list = []
        # Cache combined patterns to avoid list concatenation on every redact() call
        self._all_patterns_cache: list | None = None

    def add_pattern(self, pattern: str) -> None:
        """Add a custom redaction pattern."""
        self._custom_patterns.append(re.compile(pattern))
        # Invalidate cache when patterns change
        self._all_patterns_cache = None

    def refresh_patterns(self) -> None:
        """Refresh patterns (e.g., from environment or config)."""
        # Invalidate cache on refresh
        self._all_patterns_cache = None
        # Could be extended to load patterns from config file

    def _get_all_patterns(self) -> list:
        """Get combined patterns with caching."""
        if self._all_patterns_cache is None:
            self._all_patterns_cache = self._patterns + self._custom_patterns
        return self._all_patterns_cache

    def redact(self, content: str) -> str:
        """
        Redact sensitive content from a string.

        Args:
            content: The content to redact

        Returns:
            Content with sensitive patterns replaced
        """
        if not content:
            return content

        result = content
        all_patterns = self._get_all_patterns()

        for pattern in all_patterns:
            result = pattern.sub(self._redaction_marker, result)

        return result

    def redact_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively redact sensitive content from a dictionary.

        Args:
            data: Dictionary to redact

        Returns:
            New dictionary with sensitive content redacted
        """
        if not data:
            return data

        result: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.redact(value)
            elif isinstance(value, dict):
                result[key] = self.redact_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    (
                        self.redact(v)
                        if isinstance(v, str)
                        else self.redact_dict(v) if isinstance(v, dict) else v
                    )
                    for v in value
                ]
            else:
                result[key] = value

        return result

    def contains_sensitive(self, content: str) -> bool:
        """Check if content contains potentially sensitive patterns."""
        if not content:
            return False

        all_patterns = self._patterns + self._custom_patterns
        return any(pattern.search(content) for pattern in all_patterns)


class TelemetryVerifier:
    """
    Runtime capability verification for telemetry.

    Checks that agents and systems have the required capabilities
    before allowing telemetry operations.
    """

    # Required capabilities for different telemetry levels
    CAPABILITY_REQUIREMENTS = {
        "thought_streaming": ["generate", "name"],
        "capability_probe": ["generate", "name", "model"],
        "diagnostic": ["name"],
    }

    def __init__(self) -> None:
        """Initialize TelemetryVerifier."""
        self._capability_cache: dict[str, set[str]] = {}
        self._verification_results: list[dict] = []

    def verify_agent(
        self, agent, required_capabilities: list[str] | None = None
    ) -> tuple[bool, list[str]]:
        """
        Verify an agent has required capabilities.

        Args:
            agent: The agent to verify
            required_capabilities: List of required capability names

        Returns:
            Tuple of (all_present, missing_capabilities)
        """
        if required_capabilities is None:
            required_capabilities = self.CAPABILITY_REQUIREMENTS.get("thought_streaming", [])

        agent_name = getattr(agent, "name", str(agent))
        missing = []

        for cap in required_capabilities:
            if not hasattr(agent, cap) or getattr(agent, cap) is None:
                missing.append(cap)

        # Cache result
        self._capability_cache[agent_name] = set(required_capabilities) - set(missing)

        # Record verification
        self._verification_results.append(
            {
                "agent": agent_name,
                "required": required_capabilities,
                "missing": missing,
                "passed": len(missing) == 0,
            }
        )

        return len(missing) == 0, missing

    def verify_telemetry_level(self, level: str, agent) -> bool:
        """
        Verify an agent supports a specific telemetry level.

        Args:
            level: Telemetry level name (e.g., "thought_streaming")
            agent: The agent to verify

        Returns:
            True if agent supports the level
        """
        requirements = self.CAPABILITY_REQUIREMENTS.get(level, [])
        passed, _ = self.verify_agent(agent, requirements)
        return passed

    def get_verification_report(self) -> dict:
        """Get a summary of all verification results."""
        if not self._verification_results:
            return {"total": 0, "passed": 0, "failed": 0, "agents": []}

        passed = sum(1 for r in self._verification_results if r["passed"])
        return {
            "total": len(self._verification_results),
            "passed": passed,
            "failed": len(self._verification_results) - passed,
            "agents": self._verification_results,
        }

    def clear_cache(self) -> None:
        """Clear the capability cache."""
        self._capability_cache.clear()
        self._verification_results.clear()
