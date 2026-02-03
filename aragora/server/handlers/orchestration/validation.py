"""
Security validation and RBAC permission constants for orchestration.

Provides path-traversal prevention for source IDs and channel IDs,
and defines fine-grained RBAC permission strings.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# =============================================================================
# RBAC Permission Constants for Orchestration
# =============================================================================

# Core orchestration permissions
PERM_ORCH_DELIBERATE = "orchestration:deliberate:create"
PERM_ORCH_KNOWLEDGE_READ = "orchestration:knowledge:read"
PERM_ORCH_CHANNELS_WRITE = "orchestration:channels:write"
PERM_ORCH_ADMIN = "orchestration:admin"

# Source-type specific permissions (for fine-grained access control)
PERM_KNOWLEDGE_SLACK = "orchestration:knowledge:slack"
PERM_KNOWLEDGE_CONFLUENCE = "orchestration:knowledge:confluence"
PERM_KNOWLEDGE_GITHUB = "orchestration:knowledge:github"
PERM_KNOWLEDGE_JIRA = "orchestration:knowledge:jira"
PERM_KNOWLEDGE_DOCUMENT = "orchestration:knowledge:document"

# Channel-type specific permissions
PERM_CHANNEL_SLACK = "orchestration:channel:slack"
PERM_CHANNEL_TEAMS = "orchestration:channel:teams"
PERM_CHANNEL_DISCORD = "orchestration:channel:discord"
PERM_CHANNEL_TELEGRAM = "orchestration:channel:telegram"
PERM_CHANNEL_EMAIL = "orchestration:channel:email"
PERM_CHANNEL_WEBHOOK = "orchestration:channel:webhook"

# =============================================================================
# Source ID Validation (Path Traversal Prevention)
# =============================================================================

# Safe source_id pattern: alphanumeric, hyphens, underscores, colons, slashes (no ..)
# Allows formats like: owner/repo/pr/123, PROJ-123, channel_id, page-id
SAFE_SOURCE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-/:@.#]+$")

# Maximum source_id length to prevent DoS
MAX_SOURCE_ID_LENGTH = 256


class SourceIdValidationError(ValueError):
    """Raised when source_id validation fails."""

    pass


def safe_source_id(source_id: str) -> str:
    """
    Validate and sanitize a source_id to prevent path traversal attacks.

    Args:
        source_id: The source identifier to validate

    Returns:
        The validated source_id (unchanged if valid)

    Raises:
        SourceIdValidationError: If source_id contains dangerous patterns
    """
    if not source_id:
        raise SourceIdValidationError("source_id cannot be empty")

    if len(source_id) > MAX_SOURCE_ID_LENGTH:
        raise SourceIdValidationError(
            f"source_id too long: {len(source_id)} > {MAX_SOURCE_ID_LENGTH}"
        )

    # Check for path traversal sequences
    if ".." in source_id:
        logger.warning("[SECURITY] Path traversal attempt in source_id: %r", source_id[:50])
        raise SourceIdValidationError("source_id contains path traversal sequence (..)")

    # Check for absolute paths
    if source_id.startswith("/"):
        logger.warning("[SECURITY] Absolute path in source_id: %r", source_id[:50])
        raise SourceIdValidationError("source_id cannot start with /")

    # Check for Windows absolute paths
    if len(source_id) > 1 and source_id[1] == ":" and source_id[0].isalpha():
        logger.warning("[SECURITY] Windows absolute path in source_id: %r", source_id[:50])
        raise SourceIdValidationError("source_id cannot be a Windows absolute path")

    # Check for null bytes
    if "\x00" in source_id:
        logger.warning("[SECURITY] Null byte in source_id: %r", source_id[:50])
        raise SourceIdValidationError("source_id contains null byte")

    # Check against safe pattern
    if not SAFE_SOURCE_ID_PATTERN.match(source_id):
        logger.warning("[SECURITY] Invalid characters in source_id: %r", source_id[:50])
        raise SourceIdValidationError(
            "source_id contains invalid characters (only alphanumeric, -, _, :, /, @, ., # allowed)"
        )

    return source_id


def validate_channel_id(channel_id: str, channel_type: str) -> str:
    """
    Validate a channel_id based on the channel type.

    Args:
        channel_id: The channel identifier to validate
        channel_type: The type of channel (slack, teams, webhook, etc.)

    Returns:
        The validated channel_id

    Raises:
        ValueError: If channel_id is invalid for the channel type
    """
    if not channel_id:
        raise ValueError("channel_id cannot be empty")

    if len(channel_id) > MAX_SOURCE_ID_LENGTH:
        raise ValueError(f"channel_id too long: {len(channel_id)} > {MAX_SOURCE_ID_LENGTH}")

    # Check for null bytes
    if "\x00" in channel_id:
        raise ValueError("channel_id contains null byte")

    # For webhooks, validate it's a proper URL
    if channel_type == "webhook":
        if not channel_id.startswith(("http://", "https://")):
            raise ValueError("webhook channel_id must be a valid URL")
        # Additional URL validation
        if ".." in channel_id or "\\" in channel_id:
            raise ValueError("webhook URL contains invalid characters")
    else:
        # For non-webhook channels, use similar validation as source_id
        if ".." in channel_id or channel_id.startswith("/"):
            raise ValueError("channel_id contains invalid path characters")

    return channel_id
