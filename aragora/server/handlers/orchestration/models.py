"""
Data models for the orchestration handler.

Defines enums, request/response dataclasses, and knowledge/channel
source types used throughout orchestration.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from aragora.config import MAX_ROUNDS


class TeamStrategy(Enum):
    """Strategy for selecting the agent team."""

    SPECIFIED = "specified"  # Use explicitly provided agents
    BEST_FOR_DOMAIN = "best_for_domain"  # Auto-select based on domain
    DIVERSE = "diverse"  # Maximize agent diversity
    FAST = "fast"  # Optimize for speed
    RANDOM = "random"  # Random selection


class OutputFormat(Enum):
    """Format for vetted decisionmaking output."""

    STANDARD = "standard"  # Default JSON response
    DECISION_RECEIPT = "decision_receipt"  # Full audit receipt
    SUMMARY = "summary"  # Condensed summary
    GITHUB_REVIEW = "github_review"  # GitHub PR review format
    SLACK_MESSAGE = "slack_message"  # Slack-formatted message


@dataclass
class KnowledgeContextSource:
    """A source of knowledge context for vetted decisionmaking."""

    source_type: str  # slack, confluence, github, document, etc.
    source_id: str  # Channel ID, page ID, PR number, etc.
    lookback_minutes: int = 60
    max_items: int = 50

    @classmethod
    def from_string(cls, source_str: str) -> KnowledgeContextSource:
        """Parse from 'type:id' format."""
        if ":" not in source_str:
            return cls(source_type="document", source_id=source_str)
        parts = source_str.split(":", 1)
        return cls(source_type=parts[0], source_id=parts[1])


@dataclass
class OutputChannel:
    """A channel to route vetted decisionmaking results to."""

    channel_type: str  # slack, teams, discord, telegram, email, webhook
    channel_id: str  # Channel ID, email address, webhook URL
    thread_id: str | None = None  # Optional thread/conversation ID

    @classmethod
    def from_string(cls, channel_str: str) -> OutputChannel:
        """Parse from 'type:id' or 'type:id:thread_id' format.

        Special handling for URLs (webhook:https://...) to avoid splitting on
        the protocol colon.
        """
        if channel_str.startswith(("http://", "https://")):
            return cls(channel_type="webhook", channel_id=channel_str)
        if ":" not in channel_str:
            return cls(channel_type="webhook", channel_id=channel_str)

        # Split on first colon only to get type
        parts = channel_str.split(":", 1)
        channel_type = parts[0].lower()

        # For webhooks with URLs, the rest is the channel_id
        if (
            channel_type == "webhook"
            or channel_type in ("http", "https")
            or parts[1].startswith("http")
        ):
            return cls(
                channel_type="webhook" if channel_type in ("http", "https") else channel_type,
                channel_id=parts[1] if channel_type == "webhook" else channel_str,
            )

        # For other types, check for thread_id (type:id:thread_id)
        remaining = parts[1]
        if ":" in remaining:
            id_parts = remaining.split(":", 1)
            return cls(channel_type=channel_type, channel_id=id_parts[0], thread_id=id_parts[1])

        return cls(channel_type=channel_type, channel_id=remaining)


@dataclass
class OrchestrationRequest:
    """Request for a unified orchestration vetted decisionmaking session."""

    question: str
    knowledge_sources: list[KnowledgeContextSource] = field(default_factory=list)
    workspaces: list[str] = field(default_factory=list)
    team_strategy: TeamStrategy = TeamStrategy.BEST_FOR_DOMAIN
    agents: list[str] = field(default_factory=list)
    output_channels: list[OutputChannel] = field(default_factory=list)
    output_format: OutputFormat = OutputFormat.STANDARD
    require_consensus: bool = True
    priority: str = "normal"
    max_rounds: int = MAX_ROUNDS
    timeout_seconds: float = 300.0
    template: str | None = None
    notify: bool = True  # Auto-notify on debate completion (server default)
    dry_run: bool = False  # Return cost estimate only without executing
    metadata: dict[str, Any] = field(default_factory=dict)

    # Assigned by system
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrchestrationRequest:
        """Create from request payload."""
        # Parse knowledge sources
        knowledge_sources = []
        for source in data.get("knowledge_sources", []):
            if isinstance(source, str):
                knowledge_sources.append(KnowledgeContextSource.from_string(source))
            elif isinstance(source, dict):
                knowledge_sources.append(
                    KnowledgeContextSource(
                        source_type=source.get("type", "document"),
                        source_id=source.get("id", ""),
                        lookback_minutes=source.get("lookback_minutes", 60),
                        max_items=source.get("max_items", 50),
                    )
                )

        # Also support nested knowledge_context format
        if "knowledge_context" in data:
            ctx = data["knowledge_context"]
            for source in ctx.get("sources", []):
                if isinstance(source, str):
                    knowledge_sources.append(KnowledgeContextSource.from_string(source))

        # Parse output channels
        output_channels = []
        for channel in data.get("output_channels", []):
            if isinstance(channel, str):
                output_channels.append(OutputChannel.from_string(channel))
            elif isinstance(channel, dict):
                output_channels.append(
                    OutputChannel(
                        channel_type=channel.get("type", "webhook"),
                        channel_id=channel.get("id", ""),
                        thread_id=channel.get("thread_id"),
                    )
                )

        # Parse team strategy
        strategy_str = data.get("team_strategy", "best_for_domain")
        try:
            team_strategy = TeamStrategy(strategy_str)
        except ValueError:
            team_strategy = TeamStrategy.BEST_FOR_DOMAIN

        # Parse output format
        format_str = data.get("output_format", "standard")
        try:
            output_format = OutputFormat(format_str)
        except ValueError:
            output_format = OutputFormat.STANDARD

        return cls(
            question=data.get("question", ""),
            knowledge_sources=knowledge_sources,
            workspaces=data.get(
                "workspaces", data.get("knowledge_context", {}).get("workspaces", [])
            ),
            team_strategy=team_strategy,
            agents=data.get("agents", []),
            output_channels=output_channels,
            output_format=output_format,
            require_consensus=data.get("require_consensus", True),
            priority=data.get("priority", "normal"),
            max_rounds=data.get("max_rounds", MAX_ROUNDS),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            template=data.get("template"),
            notify=data.get("notify", True),
            dry_run=data.get("dry_run", False),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OrchestrationResult:
    """Result of a unified orchestration vetted decisionmaking session."""

    request_id: str
    success: bool
    consensus_reached: bool = False
    final_answer: str | None = None
    confidence: float | None = None
    agents_participated: list[str] = field(default_factory=list)
    rounds_completed: int = 0
    duration_seconds: float = 0.0
    knowledge_context_used: list[str] = field(default_factory=list)
    channels_notified: list[str] = field(default_factory=list)
    receipt_id: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "success": self.success,
            "consensus_reached": self.consensus_reached,
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "agents_participated": self.agents_participated,
            "rounds_completed": self.rounds_completed,
            "duration_seconds": self.duration_seconds,
            "knowledge_context_used": self.knowledge_context_used,
            "channels_notified": self.channels_notified,
            "receipt_id": self.receipt_id,
            "error": self.error,
            "created_at": self.created_at,
        }
