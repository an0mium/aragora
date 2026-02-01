"""
OpenClaw Capability Mapper.

Bidirectional mapping between OpenClaw capability identifiers
and Aragora SkillCapability enums.

OpenClaw capabilities (from SKILL.md requires):
    browser, file_read, file_write, code_execution, shell,
    web_search, screenshot, api, database

Aragora capabilities (from skills.base.SkillCapability):
    WEB_FETCH, READ_LOCAL, WRITE_LOCAL, CODE_EXECUTION,
    SHELL_EXECUTION, WEB_SEARCH, EXTERNAL_API, READ_DATABASE, etc.
"""

from __future__ import annotations

import logging

from aragora.skills.base import SkillCapability

logger = logging.getLogger(__name__)

# OpenClaw capability -> Aragora SkillCapability
_OPENCLAW_TO_ARAGORA: dict[str, SkillCapability] = {
    "browser": SkillCapability.WEB_FETCH,
    "file_read": SkillCapability.READ_LOCAL,
    "file_write": SkillCapability.WRITE_LOCAL,
    "code_execution": SkillCapability.CODE_EXECUTION,
    "shell": SkillCapability.SHELL_EXECUTION,
    "web_search": SkillCapability.WEB_SEARCH,
    "screenshot": SkillCapability.WEB_FETCH,
    "api": SkillCapability.EXTERNAL_API,
    "database": SkillCapability.READ_DATABASE,
    "embedding": SkillCapability.EMBEDDING,
    "llm": SkillCapability.LLM_INFERENCE,
}

# Aragora SkillCapability -> OpenClaw capability
_ARAGORA_TO_OPENCLAW: dict[SkillCapability, str] = {
    SkillCapability.WEB_FETCH: "browser",
    SkillCapability.READ_LOCAL: "file_read",
    SkillCapability.WRITE_LOCAL: "file_write",
    SkillCapability.CODE_EXECUTION: "code_execution",
    SkillCapability.SHELL_EXECUTION: "shell",
    SkillCapability.WEB_SEARCH: "web_search",
    SkillCapability.EXTERNAL_API: "api",
    SkillCapability.READ_DATABASE: "database",
    SkillCapability.WRITE_DATABASE: "database",
    SkillCapability.EMBEDDING: "embedding",
    SkillCapability.LLM_INFERENCE: "llm",
    SkillCapability.NETWORK: "api",
    SkillCapability.SYSTEM_INFO: "shell",
}


class CapabilityMapper:
    """Bidirectional mapper between OpenClaw and Aragora capabilities."""

    @staticmethod
    def to_aragora(openclaw_capabilities: list[str]) -> list[SkillCapability]:
        """
        Convert OpenClaw capabilities to Aragora SkillCapability list.

        Args:
            openclaw_capabilities: List of OpenClaw capability strings
                (e.g., ["browser", "file_read", "shell"])

        Returns:
            List of corresponding Aragora SkillCapability values.
            Unknown capabilities are logged and skipped.
        """
        result: list[SkillCapability] = []
        seen: set[SkillCapability] = set()

        for cap in openclaw_capabilities:
            cap_lower = cap.lower().strip()
            aragora_cap = _OPENCLAW_TO_ARAGORA.get(cap_lower)

            if aragora_cap is None:
                logger.warning(f"Unknown OpenClaw capability: {cap}")
                continue

            if aragora_cap not in seen:
                result.append(aragora_cap)
                seen.add(aragora_cap)

        return result

    @staticmethod
    def to_openclaw(aragora_capabilities: list[SkillCapability]) -> list[str]:
        """
        Convert Aragora SkillCapability list to OpenClaw capability strings.

        Args:
            aragora_capabilities: List of Aragora SkillCapability values.

        Returns:
            List of corresponding OpenClaw capability strings.
            Unknown capabilities are logged and skipped.
        """
        result: list[str] = []
        seen: set[str] = set()

        for cap in aragora_capabilities:
            openclaw_cap = _ARAGORA_TO_OPENCLAW.get(cap)

            if openclaw_cap is None:
                logger.warning(f"No OpenClaw mapping for Aragora capability: {cap.value}")
                continue

            if openclaw_cap not in seen:
                result.append(openclaw_cap)
                seen.add(openclaw_cap)

        return result

    @staticmethod
    def is_supported(openclaw_capability: str) -> bool:
        """Check if an OpenClaw capability has an Aragora mapping."""
        return openclaw_capability.lower().strip() in _OPENCLAW_TO_ARAGORA

    @staticmethod
    def all_openclaw_capabilities() -> list[str]:
        """List all known OpenClaw capabilities."""
        return sorted(_OPENCLAW_TO_ARAGORA.keys())

    @staticmethod
    def all_aragora_capabilities() -> list[SkillCapability]:
        """List all Aragora capabilities with OpenClaw mappings."""
        return sorted(set(_ARAGORA_TO_OPENCLAW.keys()), key=lambda c: c.value)
