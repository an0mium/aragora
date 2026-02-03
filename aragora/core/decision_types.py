"""
Decision type enumerations and default settings.

Extracted from decision.py for modularity.
"""

from __future__ import annotations

from enum import Enum

from aragora.config.settings import get_settings

# Resolve defaults once per import (environment-driven)
_DEFAULT_SETTINGS = get_settings()
_DEFAULT_DECISION_ROUNDS = _DEFAULT_SETTINGS.debate.default_rounds
_DEFAULT_DECISION_CONSENSUS = _DEFAULT_SETTINGS.debate.default_consensus
_DEFAULT_DECISION_MAX_AGENTS = _DEFAULT_SETTINGS.debate.max_agents_per_debate


def _default_decision_agents() -> list[str]:
    """Get default agents from settings or fallback."""
    agents = _DEFAULT_SETTINGS.agent.default_agent_list
    return agents if agents else ["anthropic-api", "openai-api"]


class DecisionType(str, Enum):
    """Types of decision-making processes available."""

    DEBATE = "debate"  # Multi-agent debate with consensus
    WORKFLOW = "workflow"  # DAG-based workflow execution
    GAUNTLET = "gauntlet"  # Adversarial validation pipeline
    QUICK = "quick"  # Fast single-agent response
    AUTO = "auto"  # Auto-detect based on content


class InputSource(str, Enum):
    """Source channel for the decision request."""

    # Chat platforms
    SLACK = "slack"
    DISCORD = "discord"
    TEAMS = "teams"
    GOOGLE_CHAT = "google_chat"
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"

    # Direct interfaces
    HTTP_API = "http_api"
    WEBSOCKET = "websocket"
    CLI = "cli"

    # Voice
    VOICE = "voice"
    VOICE_SLACK = "voice_slack"
    VOICE_TELEGRAM = "voice_telegram"
    VOICE_WHATSAPP = "voice_whatsapp"

    # Email
    EMAIL = "email"
    GMAIL = "gmail"

    # Cloud storage
    GOOGLE_DRIVE = "google_drive"
    ONEDRIVE = "onedrive"
    SHAREPOINT = "sharepoint"
    DROPBOX = "dropbox"
    S3 = "s3"

    # Enterprise integrations
    JIRA = "jira"
    GITHUB = "github"
    SERVICENOW = "servicenow"
    CONFLUENCE = "confluence"
    NOTION = "notion"

    # Event streams
    KAFKA = "kafka"
    RABBITMQ = "rabbitmq"
    WEBHOOK = "webhook"

    # Internal
    WORKFLOW = "workflow"  # Triggered by another workflow
    SCHEDULED = "scheduled"  # Scheduled task
    INTERNAL = "internal"  # System-generated


class Priority(str, Enum):
    """Request priority levels."""

    CRITICAL = "critical"  # Immediate processing
    HIGH = "high"  # Fast-track queue
    NORMAL = "normal"  # Standard processing
    LOW = "low"  # Background processing
    BATCH = "batch"  # Batch with similar requests


class ResponseFormat(str, Enum):
    """Format for the response delivery."""

    FULL = "full"  # Complete response with reasoning
    SUMMARY = "summary"  # Condensed summary
    NOTIFICATION = "notification"  # Brief notification
    VOICE = "voice"  # Audio/TTS response
    VOICE_WITH_TEXT = "voice_with_text"  # Both audio and text


__all__ = [
    "DecisionType",
    "InputSource",
    "Priority",
    "ResponseFormat",
    # Default settings
    "_DEFAULT_DECISION_ROUNDS",
    "_DEFAULT_DECISION_CONSENSUS",
    "_DEFAULT_DECISION_MAX_AGENTS",
    "_default_decision_agents",
]
