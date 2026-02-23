"""
Output Filter - PII and secret redaction from agent outputs.

Scans and redacts sensitive information from external agent outputs:
- API keys and tokens
- Credit card numbers
- Social security numbers
- Email addresses
- Phone numbers
- IP addresses
- Custom patterns per tenant

Security Model:
1. All outputs pass through filter before returning to caller
2. Redacted content replaced with [REDACTED:TYPE] markers
3. Original content never logged or stored
4. Redaction counts tracked for monitoring
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SensitiveDataType(str, Enum):
    """Types of sensitive data that can be detected."""

    API_KEY = "api_key"
    AWS_KEY = "aws_key"
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    IP_ADDRESS = "ip_address"
    JWT_TOKEN = "jwt_token"
    PASSWORD = "password"
    PRIVATE_KEY = "private_key"
    CUSTOM = "custom"


@dataclass
class SensitivePattern:
    """Pattern for detecting sensitive data."""

    name: str
    pattern: str  # Regex pattern
    data_type: SensitiveDataType
    replacement: str = "[REDACTED:{type}]"
    enabled: bool = True
    priority: int = 0  # Higher priority patterns checked first


@dataclass
class RedactionResult:
    """Result of redaction operation."""

    original_length: int
    redacted_length: int
    redaction_count: int
    redacted_types: dict[str, int] = field(default_factory=dict)


# Default patterns for common sensitive data
DEFAULT_PATTERNS: list[SensitivePattern] = [
    # API Keys
    SensitivePattern(
        name="openai_api_key",
        pattern=r"sk-[a-zA-Z0-9]{20,}",
        data_type=SensitiveDataType.API_KEY,
        priority=10,
    ),
    SensitivePattern(
        name="anthropic_api_key",
        pattern=r"sk-ant-[a-zA-Z0-9-]{20,}",
        data_type=SensitiveDataType.API_KEY,
        priority=10,
    ),
    SensitivePattern(
        name="generic_api_key",
        pattern=r"(?i)(api[_-]?key|apikey)['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9_-]{20,})['\"]?",
        data_type=SensitiveDataType.API_KEY,
        priority=5,
    ),
    # AWS Keys
    SensitivePattern(
        name="aws_access_key",
        pattern=r"AKIA[0-9A-Z]{16}",
        data_type=SensitiveDataType.AWS_KEY,
        priority=10,
    ),
    SensitivePattern(
        name="aws_secret_key",
        pattern=r"(?i)(aws[_-]?secret[_-]?access[_-]?key)['\"]?\s*[:=]\s*['\"]?([a-zA-Z0-9/+=]{40})['\"]?",
        data_type=SensitiveDataType.AWS_KEY,
        priority=10,
    ),
    # Credit Cards
    SensitivePattern(
        name="credit_card",
        pattern=r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
        data_type=SensitiveDataType.CREDIT_CARD,
        priority=10,
    ),
    # SSN
    SensitivePattern(
        name="ssn",
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        data_type=SensitiveDataType.SSN,
        priority=10,
    ),
    # Email (optional - may want to allow in some contexts)
    SensitivePattern(
        name="email",
        pattern=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        data_type=SensitiveDataType.EMAIL,
        enabled=False,  # Disabled by default - enable per tenant
        priority=1,
    ),
    # Phone Numbers
    SensitivePattern(
        name="phone_us",
        pattern=r"\b(?:\+1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b",
        data_type=SensitiveDataType.PHONE,
        enabled=False,
        priority=1,
    ),
    # IP Addresses
    SensitivePattern(
        name="ip_address",
        pattern=r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        data_type=SensitiveDataType.IP_ADDRESS,
        enabled=False,
        priority=1,
    ),
    # JWT Tokens
    SensitivePattern(
        name="jwt_token",
        pattern=r"eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*",
        data_type=SensitiveDataType.JWT_TOKEN,
        priority=10,
    ),
    # Private Keys
    SensitivePattern(
        name="private_key",
        pattern=r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        data_type=SensitiveDataType.PRIVATE_KEY,
        priority=10,
    ),
    # Password patterns
    SensitivePattern(
        name="password_assignment",
        pattern=r"(?i)(password|passwd|pwd)['\"]?\s*[:=]\s*['\"]?([^\s'\"]{8,})['\"]?",
        data_type=SensitiveDataType.PASSWORD,
        priority=5,
    ),
]


class OutputFilter:
    """
    Filter for redacting sensitive information from agent outputs.

    Scans text for patterns matching sensitive data types and replaces
    them with redaction markers. Supports custom patterns per tenant.

    Usage:
        filter = OutputFilter()

        # Add custom pattern
        filter.add_pattern(SensitivePattern(
            name="internal_id",
            pattern=r"ACME-[0-9]{10}",
            data_type=SensitiveDataType.CUSTOM,
        ))

        # Redact output
        redacted, result = await filter.redact(agent_output)
    """

    def __init__(
        self,
        patterns: list[SensitivePattern] | None = None,
        enable_default_patterns: bool = True,
    ):
        self._patterns: list[SensitivePattern] = []

        if enable_default_patterns:
            # Deep copy so enable/disable_pattern doesn't mutate the shared
            # DEFAULT_PATTERNS list (which would leak state across instances).
            self._patterns.extend(copy.copy(p) for p in DEFAULT_PATTERNS)

        if patterns:
            self._patterns.extend(patterns)

        # Sort by priority (higher first)
        self._patterns.sort(key=lambda p: -p.priority)

        # Compile patterns for performance
        self._compiled: dict[str, re.Pattern[str]] = {}
        for pattern in self._patterns:
            if pattern.enabled:
                try:
                    self._compiled[pattern.name] = re.compile(pattern.pattern)
                except re.error as e:
                    logger.error("Invalid pattern %s: %s", pattern.name, e)

    def add_pattern(self, pattern: SensitivePattern) -> None:
        """Add a custom pattern."""
        self._patterns.append(pattern)
        self._patterns.sort(key=lambda p: -p.priority)
        if pattern.enabled:
            try:
                self._compiled[pattern.name] = re.compile(pattern.pattern)
            except re.error as e:
                logger.error("Invalid pattern %s: %s", pattern.name, e)

    def enable_pattern(self, name: str) -> bool:
        """Enable a pattern by name."""
        for pattern in self._patterns:
            if pattern.name == name:
                pattern.enabled = True
                if name not in self._compiled:
                    self._compiled[name] = re.compile(pattern.pattern)
                return True
        return False

    def disable_pattern(self, name: str) -> bool:
        """Disable a pattern by name."""
        for pattern in self._patterns:
            if pattern.name == name:
                pattern.enabled = False
                self._compiled.pop(name, None)
                return True
        return False

    async def redact(
        self,
        text: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, RedactionResult]:
        """
        Redact sensitive information from text.

        Args:
            text: Text to scan and redact
            context: Optional context for conditional redaction

        Returns:
            Tuple of (redacted text, redaction result)
        """
        original_length = len(text)
        redaction_count = 0
        redacted_types: dict[str, int] = {}

        result_text = text

        for pattern in self._patterns:
            if not pattern.enabled or pattern.name not in self._compiled:
                continue

            compiled = self._compiled[pattern.name]
            matches = compiled.findall(result_text)

            if matches:
                # Create replacement text
                replacement = pattern.replacement.format(type=pattern.data_type.value)

                # Replace all matches
                result_text = compiled.sub(replacement, result_text)

                # Track statistics
                match_count = len(matches)
                redaction_count += match_count
                redacted_types[pattern.data_type.value] = (
                    redacted_types.get(pattern.data_type.value, 0) + match_count
                )

                logger.debug("Redacted %s matches for pattern %s", match_count, pattern.name)

        result = RedactionResult(
            original_length=original_length,
            redacted_length=len(result_text),
            redaction_count=redaction_count,
            redacted_types=redacted_types,
        )

        if redaction_count > 0:
            logger.info("Redacted %s sensitive items: %s", redaction_count, redacted_types)

        return result_text, result

    def list_patterns(self) -> list[dict[str, Any]]:
        """List all configured patterns."""
        return [
            {
                "name": p.name,
                "data_type": p.data_type.value,
                "enabled": p.enabled,
                "priority": p.priority,
            }
            for p in self._patterns
        ]

    @classmethod
    def create_strict(cls) -> OutputFilter:
        """Create a filter with all patterns enabled."""
        filter_instance = cls()
        for pattern in filter_instance._patterns:
            filter_instance.enable_pattern(pattern.name)
        return filter_instance

    @classmethod
    def create_minimal(cls) -> OutputFilter:
        """Create a filter with only critical patterns (keys, tokens)."""
        filter_instance = cls(enable_default_patterns=False)
        critical_types = {
            SensitiveDataType.API_KEY,
            SensitiveDataType.AWS_KEY,
            SensitiveDataType.JWT_TOKEN,
            SensitiveDataType.PRIVATE_KEY,
            SensitiveDataType.PASSWORD,
        }
        for pattern in DEFAULT_PATTERNS:
            if pattern.data_type in critical_types:
                filter_instance.add_pattern(pattern)
        return filter_instance
