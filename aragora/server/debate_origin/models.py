"""Debate origin data model."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DebateOrigin:
    """Origin information for a debate."""

    debate_id: str
    platform: str  # telegram, whatsapp, slack, discord, teams, email, web
    channel_id: str  # Chat ID, channel ID, thread ID, etc.
    user_id: str  # User who initiated the debate
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional threading info
    thread_id: str | None = None
    message_id: str | None = None

    # Session tracking for multi-channel support
    session_id: str | None = None

    # Result routing
    result_sent: bool = False
    result_sent_at: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "platform": self.platform,
            "channel_id": self.channel_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "thread_id": self.thread_id,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "result_sent": self.result_sent,
            "result_sent_at": self.result_sent_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DebateOrigin:
        return cls(
            debate_id=data["debate_id"],
            platform=data["platform"],
            channel_id=data["channel_id"],
            user_id=data["user_id"],
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
            thread_id=data.get("thread_id"),
            message_id=data.get("message_id"),
            session_id=data.get("session_id"),
            result_sent=data.get("result_sent", False),
            result_sent_at=data.get("result_sent_at"),
        )
