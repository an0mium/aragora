"""
Authentication helpers for Aragora server.

Provides optional token-based access control for WebSocket subscriptions and API endpoints.
Tokens can be set via environment variables or runtime configuration.
"""

import hashlib
import hmac
import os
import secrets
import time
from typing import Optional, Dict, Any
from urllib.parse import parse_qs


class AuthConfig:
    """Configuration for authentication."""

    def __init__(self):
        self.enabled = False
        self.api_token: Optional[str] = None
        self.token_ttl = 3600  # 1 hour default
        self.allowed_origins: list[str] = ["*"]  # CORS origins

    def configure_from_env(self):
        """Configure from environment variables."""
        self.api_token = os.getenv("ARAGORA_API_TOKEN")
        if self.api_token:
            self.enabled = True

        ttl_str = os.getenv("ARAGORA_TOKEN_TTL", "3600")
        try:
            self.token_ttl = int(ttl_str)
        except ValueError:
            pass

        origins = os.getenv("ARAGORA_ALLOWED_ORIGINS")
        if origins:
            self.allowed_origins = [o.strip() for o in origins.split(",")]

    def generate_token(self, loop_id: str = "", expires_in: int = None) -> str:
        """Generate a signed token for access."""
        if not self.api_token:
            return ""

        if expires_in is None:
            expires_in = self.token_ttl

        expires = int(time.time()) + expires_in
        payload = f"{loop_id}:{expires}"
        signature = hmac.new(
            self.api_token.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return f"{payload}:{signature}"

    def validate_token(self, token: str, loop_id: str = "") -> bool:
        """Validate a token."""
        if not self.api_token or not token:
            return not self.enabled  # If auth disabled, allow; if enabled, require token

        try:
            payload, signature = token.rsplit(":", 1)
            loop_part, expires_str = payload.rsplit(":", 1)
            expires = int(expires_str)

            # Check expiration
            if time.time() > expires:
                return False

            # Verify signature
            expected = hmac.new(
                self.api_token.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected):
                return False

            # Check loop_id if specified
            if loop_id and loop_part != loop_id:
                return False

            return True

        except (ValueError, IndexError):
            return False

    def extract_token_from_request(self, headers: Dict[str, str], query_params: Dict[str, list]) -> Optional[str]:
        """Extract token from Authorization header or query params."""
        # Check Authorization header
        auth_header = headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]

        # Check query params
        tokens = query_params.get("token", [])
        if tokens:
            return tokens[0]

        return None


# Global auth config instance
auth_config = AuthConfig()
auth_config.configure_from_env()


def check_auth(headers: Dict[str, Any], query_string: str = "", loop_id: str = "") -> bool:
    """
    Check authentication for a request.

    Args:
        headers: HTTP headers dict
        query_string: Raw query string
        loop_id: Optional loop ID to validate against token

    Returns:
        True if authenticated or auth disabled
    """
    if not auth_config.enabled:
        return True

    query_params = parse_qs(query_string.lstrip("?")) if query_string else {}
    token = auth_config.extract_token_from_request(headers, query_params)

    return auth_config.validate_token(token or "", loop_id)


def generate_shareable_link(base_url: str, loop_id: str, expires_in: int = 3600) -> str:
    """Generate a shareable link with embedded token."""
    token = auth_config.generate_token(loop_id, expires_in)
    if not token:
        return base_url

    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}token={token}"