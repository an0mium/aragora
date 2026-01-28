"""
GitHub OAuth Provider - GitHub OAuth 2.0 implementation.

Handles:
- Authorization URL generation for GitHub consent screen
- Token exchange with GitHub
- User info retrieval from GitHub API (including email fallback)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from aragora.server.handlers.oauth_providers.base import (
    OAuthProvider,
    OAuthProviderConfig,
    OAuthTokens,
    OAuthUserInfo,
    _get_secret,
    _is_production,
)

logger = logging.getLogger(__name__)

# GitHub OAuth endpoints
GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USERINFO_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"


class GitHubOAuthProvider(OAuthProvider):
    """
    GitHub OAuth 2.0 provider.

    Supports:
    - OAuth 2.0 authorization code flow
    - User info retrieval (basic profile + email)
    - Private email retrieval via emails API

    Note: GitHub does not support token refresh. Tokens are long-lived
    but can be revoked by the user at any time.
    """

    PROVIDER_NAME = "github"

    def _load_config_from_env(self) -> OAuthProviderConfig:
        """Load GitHub OAuth configuration from environment."""
        redirect_uri = _get_secret("GITHUB_OAUTH_REDIRECT_URI", "")
        if not redirect_uri and not _is_production():
            redirect_uri = "http://localhost:8080/api/auth/oauth/github/callback"

        return OAuthProviderConfig(
            client_id=_get_secret("GITHUB_OAUTH_CLIENT_ID", ""),
            client_secret=_get_secret("GITHUB_OAUTH_CLIENT_SECRET", ""),
            redirect_uri=redirect_uri,
            scopes=["read:user", "user:email"],
            authorization_endpoint=GITHUB_AUTH_URL,
            token_endpoint=GITHUB_TOKEN_URL,
            userinfo_endpoint=GITHUB_USERINFO_URL,
        )

    def get_authorization_url(
        self,
        state: str,
        redirect_uri: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        **kwargs,
    ) -> str:
        """
        Generate GitHub OAuth authorization URL.

        Args:
            state: CSRF protection state parameter
            redirect_uri: Override redirect URI
            scopes: Override scopes
            **kwargs: Additional parameters:
                - allow_signup: Whether to allow sign-up during auth (default True)
                - login: Pre-fill username

        Returns:
            Authorization URL to redirect user to
        """
        params = {
            "client_id": self._config.client_id,
            "redirect_uri": redirect_uri or self._config.redirect_uri,
            "scope": " ".join(scopes or self._config.scopes),
            "state": state,
        }

        # Optional GitHub-specific parameters
        if "allow_signup" in kwargs:
            params["allow_signup"] = str(kwargs["allow_signup"]).lower()
        if "login" in kwargs:
            params["login"] = kwargs["login"]

        return self._build_authorization_url(GITHUB_AUTH_URL, params)

    def exchange_code(
        self,
        code: str,
        redirect_uri: Optional[str] = None,
    ) -> OAuthTokens:
        """
        Exchange authorization code for tokens.

        Args:
            code: Authorization code from callback
            redirect_uri: Redirect URI used in authorization

        Returns:
            OAuth tokens (access_token only - GitHub doesn't provide refresh tokens)
        """
        data = {
            "code": code,
            "client_id": self._config.client_id,
            "client_secret": self._config.client_secret,
            "redirect_uri": redirect_uri or self._config.redirect_uri,
        }

        # GitHub requires Accept header for JSON response
        headers = {"Accept": "application/json"}
        return self._request_tokens(GITHUB_TOKEN_URL, data, headers)

    def get_user_info(self, access_token: str) -> OAuthUserInfo:
        """
        Get user information from GitHub API.

        Retrieves basic user info and falls back to emails API if email
        is not public.

        Args:
            access_token: Access token from exchange

        Returns:
            User information including email, name, picture
        """
        # GitHub API requires specific Accept header
        headers = {"Accept": "application/json"}

        # Get basic user info
        user_data = self._request_user_info(GITHUB_USERINFO_URL, access_token, headers)

        # Get email - may need to fetch from emails endpoint
        email = user_data.get("email")
        email_verified = False

        if not email:
            email, email_verified = self._fetch_primary_email(access_token)

        if not email:
            raise ValueError("Could not retrieve email from GitHub")

        return OAuthUserInfo(
            provider=self.PROVIDER_NAME,
            provider_user_id=str(user_data["id"]),
            email=email,
            email_verified=email_verified,
            name=user_data.get("name") or user_data.get("login"),
            picture=user_data.get("avatar_url"),
            raw_data=user_data,
        )

    def _fetch_primary_email(self, access_token: str) -> tuple[Optional[str], bool]:
        """
        Fetch primary email from GitHub emails API.

        Args:
            access_token: Access token

        Returns:
            Tuple of (email, email_verified)
        """
        try:
            client = self._get_http_client()
            response = client.get(
                GITHUB_EMAILS_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
            emails = response.json()

            # Find primary verified email
            for email_entry in emails:
                if email_entry.get("primary") and email_entry.get("verified"):
                    return email_entry.get("email"), True

            # Fallback to any verified email
            for email_entry in emails:
                if email_entry.get("verified"):
                    return email_entry.get("email"), True

            # Last resort: any email
            if emails:
                return emails[0].get("email"), False

            return None, False

        except Exception as e:
            logger.warning(f"[{self.PROVIDER_NAME}] Failed to fetch emails: {e}")
            return None, False

    def get_user_repos(
        self,
        access_token: str,
        visibility: str = "all",
        sort: str = "updated",
        per_page: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        Get user's repositories (if repo scope is granted).

        Args:
            access_token: Access token
            visibility: "all", "public", or "private"
            sort: "created", "updated", "pushed", or "full_name"
            per_page: Number of repos per page (max 100)

        Returns:
            List of repository data
        """
        try:
            client = self._get_http_client()
            response = client.get(
                "https://api.github.com/user/repos",
                params={
                    "visibility": visibility,
                    "sort": sort,
                    "per_page": min(per_page, 100),
                },
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json",
                },
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"[{self.PROVIDER_NAME}] Failed to fetch repos: {e}")
            return []


__all__ = ["GitHubOAuthProvider"]
