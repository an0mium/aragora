"""
Debate Sharing Handler.

Provides endpoints for sharing debates with different visibility levels:
- private: Only accessible by the creator
- team: Accessible by organization members
- public: Accessible via shareable link

Endpoints:
    GET  /api/debates/{id}/share          - Get sharing settings
    POST /api/debates/{id}/share          - Update sharing settings
    GET  /api/shared/{token}              - Access shared debate
    POST /api/debates/{id}/share/revoke   - Revoke all share links
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from aragora.server.validation.schema import SHARE_UPDATE_SCHEMA, validate_against_schema

from ..base import (
    BaseHandler,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
)
from ..utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)


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
    share_token: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # None = no expiration
    allow_comments: bool = False
    allow_forking: bool = False
    view_count: int = 0
    owner_id: Optional[str] = None
    org_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
        return f"/api/shared/{self.share_token}"

    @property
    def is_expired(self) -> bool:
        """Check if the share link has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ShareSettings":
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


MAX_SHARE_SETTINGS = 10000  # Prevent unbounded memory growth


class ShareStore:
    """In-memory store for sharing settings (thread-safe).

    In production, this would be backed by a database.
    """

    def __init__(self):
        self._settings: Dict[str, ShareSettings] = {}
        self._tokens: Dict[str, str] = {}  # token -> debate_id
        self._lock = threading.Lock()

    def get(self, debate_id: str) -> Optional[ShareSettings]:
        """Get sharing settings for a debate (thread-safe)."""
        with self._lock:
            return self._settings.get(debate_id)

    def get_by_token(self, token: str) -> Optional[ShareSettings]:
        """Get sharing settings by share token (thread-safe)."""
        with self._lock:
            debate_id = self._tokens.get(token)
            if debate_id:
                return self._settings.get(debate_id)
            return None

    def save(self, settings: ShareSettings) -> None:
        """Save sharing settings (thread-safe with size limit)."""
        with self._lock:
            # Enforce max size with LRU eviction (by created_at)
            if (
                settings.debate_id not in self._settings
                and len(self._settings) >= MAX_SHARE_SETTINGS
            ):
                # Remove oldest 10% by created_at
                sorted_items = sorted(self._settings.items(), key=lambda x: x[1].created_at)
                remove_count = max(1, len(sorted_items) // 10)
                for debate_id_to_remove, s in sorted_items[:remove_count]:
                    del self._settings[debate_id_to_remove]
                    if s.share_token:
                        self._tokens.pop(s.share_token, None)
                logger.debug(f"ShareStore evicted {remove_count} oldest entries")

            self._settings[settings.debate_id] = settings
            if settings.share_token:
                self._tokens[settings.share_token] = settings.debate_id

    def delete(self, debate_id: str) -> bool:
        """Delete sharing settings (thread-safe)."""
        with self._lock:
            settings = self._settings.pop(debate_id, None)
            if settings and settings.share_token:
                self._tokens.pop(settings.share_token, None)
            return settings is not None

    def revoke_token(self, debate_id: str) -> bool:
        """Revoke the share token for a debate (thread-safe)."""
        with self._lock:
            settings = self._settings.get(debate_id)
            if settings and settings.share_token:
                self._tokens.pop(settings.share_token, None)
                settings.share_token = None
                return True
            return False

    def increment_view_count(self, debate_id: str) -> None:
        """Increment the view count for a shared debate (thread-safe)."""
        with self._lock:
            settings = self._settings.get(debate_id)
            if settings:
                settings.view_count += 1


# Global store instance with thread-safe initialization
# Can be either in-memory ShareStore or SQLite-backed ShareLinkStore
# Use Any to allow dynamic ShareLinkStore assignment without import cycle
_share_store: Optional[Any] = None
_share_store_lock = threading.Lock()


def get_share_store() -> Any:
    """Get the global share store instance (thread-safe).

    Uses SQLite-backed ShareLinkStore for production persistence,
    with fallback to in-memory ShareStore if database unavailable.
    """
    global _share_store
    if _share_store is None:
        with _share_store_lock:
            # Double-check after acquiring lock
            if _share_store is None:
                try:
                    from aragora.config.legacy import DATA_DIR
                    from aragora.storage.share_store import ShareLinkStore

                    db_path = DATA_DIR / "share_links.db"
                    _share_store = ShareLinkStore(db_path)
                    logger.info(f"Using SQLite ShareLinkStore: {db_path}")
                except Exception as e:
                    logger.warning(f"Failed to init ShareLinkStore, using in-memory: {e}")
                    _share_store = ShareStore()
    return _share_store


class SharingHandler(BaseHandler):
    """Handler for debate sharing endpoints."""

    ROUTES = [
        "/api/debates/*/share",
        "/api/debates/*/share/revoke",
        "/api/shared/*",
    ]

    # Require auth for all endpoints except shared view
    AUTH_REQUIRED_ENDPOINTS = [
        "/share",
        "/share/revoke",
    ]

    def __init__(self, server_context: dict = None):
        super().__init__(server_context or {})
        self._store = get_share_store()

    def handle(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Handle GET requests."""
        # Shared debate access (public endpoint)
        if path.startswith("/api/shared/"):
            token = path.split("/api/shared/")[1].rstrip("/")
            return self._get_shared_debate(token, query_params)

        # Get sharing settings
        if path.endswith("/share"):
            debate_id, err = self._extract_debate_id(path)
            if err:
                return error_response(err, 400)
            return self._get_share_settings(debate_id, handler)

        return None

    def handle_post(self, path: str, query_params: dict, handler) -> Optional[HandlerResult]:
        """Handle POST requests."""
        # Revoke share link
        if path.endswith("/share/revoke"):
            debate_id, err = self._extract_debate_id(path)
            if err:
                return error_response(err, 400)
            return self._revoke_share(debate_id, handler)

        # Update sharing settings
        if path.endswith("/share"):
            debate_id, err = self._extract_debate_id(path)
            if err:
                return error_response(err, 400)
            return self._update_share_settings(debate_id, handler)

        return None

    def _extract_debate_id(self, path: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract debate ID from path."""
        try:
            # Path format: /api/debates/{id}/share or /api/debates/{id}/share/revoke
            parts = path.split("/")
            # Find 'debates' and get the next part
            for i, part in enumerate(parts):
                if part == "debates" and i + 1 < len(parts):
                    debate_id = parts[i + 1]
                    if debate_id and debate_id not in ("share", "revoke", ""):
                        return debate_id, None
            return None, "Could not extract debate ID from path"
        except Exception as e:
            return None, str(e)

    @handle_errors("get share settings")
    def _get_share_settings(self, debate_id: str, handler) -> HandlerResult:
        """Get sharing settings for a debate.

        Returns:
            Current sharing settings including visibility, share URL, etc.
        """
        # Check authorization
        user = self.get_current_user(handler)
        if not user:
            return error_response("Authentication required", 401)

        # Get or create settings
        settings = self._store.get(debate_id)
        if not settings:
            settings = ShareSettings(
                debate_id=debate_id,
                owner_id=user.id,
                org_id=user.org_id,
            )
            self._store.save(settings)

        # Verify ownership
        if settings.owner_id and settings.owner_id != user.id:
            # Allow org members to view team-visible debates
            if settings.visibility != DebateVisibility.TEAM or settings.org_id != user.org_id:
                return error_response("Not authorized to view sharing settings", 403)

        return json_response(settings.to_dict())

    @rate_limit(rpm=30, limiter_name="share_update")
    @handle_errors("update share settings")
    def _update_share_settings(self, debate_id: str, handler) -> HandlerResult:
        """Update sharing settings for a debate.

        POST body:
            {
                "visibility": "private" | "team" | "public",
                "expires_in_hours": int,  # Optional: hours until link expires
                "allow_comments": bool,   # Optional
                "allow_forking": bool     # Optional
            }

        Returns:
            Updated sharing settings including new share URL if public.
        """
        user = self.get_current_user(handler)
        if not user:
            return error_response("Authentication required", 401)

        body = self.read_json_body(handler)
        if body is None:
            return error_response("Invalid or missing JSON body", 400)

        # Schema validation for input sanitization
        validation_result = validate_against_schema(body, SHARE_UPDATE_SCHEMA)
        if not validation_result.is_valid:
            return error_response(validation_result.error, 400)

        # Get or create settings
        settings = self._store.get(debate_id)
        if not settings:
            settings = ShareSettings(
                debate_id=debate_id,
                owner_id=user.id,
                org_id=user.org_id,
            )

        # Verify ownership
        if settings.owner_id and settings.owner_id != user.id:
            return error_response("Not authorized to modify sharing settings", 403)

        # Update visibility
        visibility_str = body.get("visibility")
        if visibility_str:
            try:
                new_visibility = DebateVisibility(visibility_str)
                old_visibility = settings.visibility
                settings.visibility = new_visibility

                # Generate share token if making public
                if new_visibility == DebateVisibility.PUBLIC and not settings.share_token:
                    settings.share_token = self._generate_share_token(debate_id)

                # Revoke token if making private
                if new_visibility == DebateVisibility.PRIVATE and settings.share_token:
                    self._store.revoke_token(debate_id)
                    settings.share_token = None

                logger.info(
                    f"Debate {debate_id} visibility changed: {old_visibility.value} -> {new_visibility.value}"
                )
            except ValueError:
                return error_response(
                    f"Invalid visibility. Must be: {', '.join(v.value for v in DebateVisibility)}",
                    400,
                )

        # Update expiration
        expires_in_hours = body.get("expires_in_hours")
        if expires_in_hours is not None:
            if expires_in_hours <= 0:
                settings.expires_at = None  # No expiration
            else:
                settings.expires_at = time.time() + (expires_in_hours * 3600)

        # Update other settings
        if "allow_comments" in body:
            settings.allow_comments = bool(body["allow_comments"])
        if "allow_forking" in body:
            settings.allow_forking = bool(body["allow_forking"])

        # Save
        self._store.save(settings)

        return json_response(
            {
                "success": True,
                "settings": settings.to_dict(),
            }
        )

    @rate_limit(rpm=60, limiter_name="shared_debate_access")
    @handle_errors("get shared debate")
    def _get_shared_debate(self, token: str, query_params: dict) -> HandlerResult:
        """Access a shared debate via token.

        This is a public endpoint - no authentication required.
        Returns debate data if the share link is valid.
        """
        settings = self._store.get_by_token(token)

        if not settings:
            # Log failed lookups to detect potential enumeration attacks
            # Use debug level to avoid log spam from legitimate 404s
            logger.debug(f"Share token not found: {token[:8]}...")
            return error_response("Share link not found", 404)

        if settings.is_expired:
            return error_response("Share link has expired", 410)

        if settings.visibility != DebateVisibility.PUBLIC:
            return error_response("Debate is no longer shared", 403)

        # Increment view count
        self._store.increment_view_count(settings.debate_id)

        # Get debate data
        debate_data = self._get_debate_data(settings.debate_id)
        if not debate_data:
            return error_response("Debate not found", 404)

        return json_response(
            {
                "debate": debate_data,
                "sharing": {
                    "allow_comments": settings.allow_comments,
                    "allow_forking": settings.allow_forking,
                    "view_count": settings.view_count,
                },
            }
        )

    @handle_errors("revoke share")
    def _revoke_share(self, debate_id: str, handler) -> HandlerResult:
        """Revoke all share links for a debate.

        POST body: {} (empty)

        Returns:
            {
                "success": true,
                "message": "Share links revoked"
            }
        """
        user = self.get_current_user(handler)
        if not user:
            return error_response("Authentication required", 401)

        settings = self._store.get(debate_id)
        if not settings:
            return error_response("No sharing settings found", 404)

        # Verify ownership
        if settings.owner_id and settings.owner_id != user.id:
            return error_response("Not authorized to revoke sharing", 403)

        # Revoke token and set to private
        self._store.revoke_token(debate_id)
        settings.visibility = DebateVisibility.PRIVATE
        self._store.save(settings)

        logger.info(f"Share links revoked for debate {debate_id}")

        return json_response(
            {
                "success": True,
                "message": "Share links revoked",
            }
        )

    def _generate_share_token(self, debate_id: str) -> str:
        """Generate a secure share token."""
        # Use secrets for cryptographically secure token
        return secrets.token_urlsafe(16)

    def _get_debate_data(self, debate_id: str) -> Optional[Dict[str, Any]]:
        """Get debate data for sharing.

        Fetches the debate artifact from the DebateStorage database.
        """
        try:
            from aragora.server.storage import get_debates_db

            db = get_debates_db()
            if db:
                return db.get(debate_id)
        except Exception as e:
            logger.warning(f"Could not fetch debate {debate_id}: {e}")

        return None


__all__ = [
    "SharingHandler",
    "ShareSettings",
    "DebateVisibility",
    "ShareStore",
    "get_share_store",
]
