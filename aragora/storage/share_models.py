"""Debate sharing data models.

These models live in the storage layer so that lower-layer modules (e.g.
``aragora.storage.share_store``) can reference them without importing
``aragora.server``. The server-side sharing handler
(``aragora.server.handlers.social.sharing``) re-exports them for backward
compatibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DebateVisibility(str, Enum):
    """Visibility level for a debate."""

    PRIVATE = "private"  # Only creator can access
    TEAM = "team"  # Organization members can access
    PUBLIC = "public"  # Anyone with link can access


@dataclass
class ShareSettings:
    """Sharing settings for a debate."""

    debate_id: str
    visibility: DebateVisibility = DebateVisibility.PRIVATE
    share_token: str | None = None
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None  # None = no expiration
    allow_comments: bool = False
    allow_forking: bool = False
    view_count: int = 0
    owner_id: str | None = None
    org_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "debate_id": self.debate_id,
            "visibility": self.visibility.value,
            "share_token": self.share_token,
            "share_url": self._get_share_url() if self.share_token else None,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "is_expired": self.is_expired,
            "allow_comments": self.allow_comments,
            "allow_forking": self.allow_forking,
            "view_count": self.view_count,
        }

    def _get_share_url(self) -> str:
        """Generate the share URL."""
        # This would be configured via settings in production
        return f"/api/v1/shared/{self.share_token}"

    @property
    def is_expired(self) -> bool:
        """Check if the share link has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShareSettings:
        """Create from dictionary."""
        return cls(
            debate_id=data["debate_id"],
            visibility=DebateVisibility(data.get("visibility", "private")),
            share_token=data.get("share_token"),
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            allow_comments=data.get("allow_comments", False),
            allow_forking=data.get("allow_forking", False),
            view_count=data.get("view_count", 0),
            owner_id=data.get("owner_id"),
            org_id=data.get("org_id"),
        )


__all__ = [
    "DebateVisibility",
    "ShareSettings",
]
