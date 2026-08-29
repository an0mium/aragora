"""OAuth2 user-context token management for the X (Twitter) API v2.

Bookmarks and likes are only readable with an OAuth2 *user-context* token
(scopes ``bookmark.read`` / ``like.read``), not the app-only bearer token the
:class:`~aragora.connectors.twitter.TwitterConnector` uses for search.

Tokens live in ``.aragora/x_intake/oauth.json`` (written by
``scripts/x_oauth_setup.py``). X rotates the refresh token on every refresh,
so the rotated pair is persisted immediately after each refresh — losing a
rotated refresh token strands the grant.

Environment variables ``X_OAUTH2_ACCESS_TOKEN`` / ``X_OAUTH2_REFRESH_TOKEN`` /
``X_OAUTH2_CLIENT_ID`` seed the store when the file does not exist yet.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

TOKEN_URL = "https://api.twitter.com/2/oauth2/token"
DEFAULT_TOKEN_PATH = Path(".aragora/x_intake/oauth.json")

__all__ = ["XOAuthTokenStore", "XOAuthTokens"]


@dataclass
class XOAuthTokens:
    """An OAuth2 user-context token pair."""

    access_token: str
    refresh_token: str = ""
    client_id: str = ""
    expires_at: float = 0.0

    @property
    def is_expired(self) -> bool:
        # 60s of slack so a token never expires mid-request
        return bool(self.expires_at) and time.time() > self.expires_at - 60


class XOAuthTokenStore:
    """Load, refresh, and persist X OAuth2 user-context tokens."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_TOKEN_PATH

    def load(self) -> XOAuthTokens | None:
        """Load tokens from file, falling back to environment seeds."""
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                return XOAuthTokens(
                    access_token=data.get("access_token", ""),
                    refresh_token=data.get("refresh_token", ""),
                    client_id=data.get("client_id", ""),
                    expires_at=float(data.get("expires_at", 0.0)),
                )
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                logger.warning("Could not read X OAuth token file %s: %s", self.path, exc)

        access = os.environ.get("X_OAUTH2_ACCESS_TOKEN", "")
        if not access and not os.environ.get("X_OAUTH2_REFRESH_TOKEN"):
            return None
        return XOAuthTokens(
            access_token=access,
            refresh_token=os.environ.get("X_OAUTH2_REFRESH_TOKEN", ""),
            client_id=os.environ.get("X_OAUTH2_CLIENT_ID", ""),
        )

    def save(self, tokens: XOAuthTokens) -> None:
        """Persist tokens (0600) — must run immediately after every rotation."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "access_token": tokens.access_token,
            "refresh_token": tokens.refresh_token,
            "client_id": tokens.client_id,
            "expires_at": tokens.expires_at,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            self.path.chmod(0o600)
        except OSError:  # pragma: no cover - platform-dependent
            pass

    async def refresh(self, tokens: XOAuthTokens) -> XOAuthTokens | None:
        """Refresh the access token; persists the rotated pair on success."""
        if not tokens.refresh_token or not tokens.client_id:
            logger.warning("X OAuth refresh needs refresh_token and client_id")
            return None
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not available; cannot refresh X OAuth token")
            return None

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    TOKEN_URL,
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": tokens.refresh_token,
                        "client_id": tokens.client_id,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("X OAuth token refresh failed: %s", exc)
            return None

        refreshed = XOAuthTokens(
            access_token=data.get("access_token", ""),
            # X rotates the refresh token; fall back to the old one if absent
            refresh_token=data.get("refresh_token", tokens.refresh_token),
            client_id=tokens.client_id,
            expires_at=time.time() + float(data.get("expires_in", 7200)),
        )
        if not refreshed.access_token:
            logger.warning("X OAuth refresh response had no access_token")
            return None
        self.save(refreshed)
        logger.info("X OAuth token refreshed (expires in %ss)", data.get("expires_in"))
        return refreshed

    async def get_valid(self) -> XOAuthTokens | None:
        """Return a non-expired token pair, refreshing if needed."""
        tokens = self.load()
        if tokens is None:
            return None
        if tokens.is_expired or not tokens.access_token:
            return await self.refresh(tokens)
        return tokens
