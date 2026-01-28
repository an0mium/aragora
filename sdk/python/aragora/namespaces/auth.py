"""
Auth Namespace API

Provides methods for authentication and user session management.

Features:
- User registration and login
- Token refresh and logout
- Password reset workflows
- MFA setup and verification
- OAuth authentication flows
- Session management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class AuthAPI:
    """
    Synchronous Auth API.

    Provides methods for authentication operations:
    - User registration and login
    - Token management
    - Password reset
    - MFA configuration
    - OAuth flows
    - Session management

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> token = client.auth.login(email="user@example.com", password="secret")
        >>> user = client.auth.get_current_user()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Registration and Login
    # ===========================================================================

    def register(
        self,
        email: str,
        password: str,
        name: str | None = None,
    ) -> dict[str, Any]:
        """
        Register a new user account.

        Args:
            email: User email address
            password: Password (min 8 chars)
            name: Optional display name

        Returns:
            Dict with user info and confirmation status
        """
        data = {"email": email, "password": password}
        if name:
            data["name"] = name

        return self._client.request("POST", "/api/v1/auth/register", json=data)

    def login(
        self,
        email: str,
        password: str,
        mfa_code: str | None = None,
    ) -> dict[str, Any]:
        """
        Authenticate and get access token.

        Args:
            email: User email
            password: User password
            mfa_code: Optional MFA code if enabled

        Returns:
            Dict with access_token, refresh_token, expires_in, and token_type
        """
        data = {"email": email, "password": password}
        if mfa_code:
            data["mfa_code"] = mfa_code

        return self._client.request("POST", "/api/v1/auth/login", json=data)

    def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """
        Refresh the access token.

        Args:
            refresh_token: The refresh token

        Returns:
            Dict with new access_token, refresh_token, expires_in
        """
        return self._client.request(
            "POST",
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )

    def logout(self) -> dict[str, Any]:
        """
        Logout and invalidate current token.

        Returns:
            Dict with success status
        """
        return self._client.request("POST", "/api/v1/auth/logout")

    def logout_all(self) -> dict[str, Any]:
        """
        Logout from all sessions.

        Returns:
            Dict with logged_out status and sessions_revoked count
        """
        return self._client.request("POST", "/api/v1/auth/logout/all")

    # ===========================================================================
    # User Profile
    # ===========================================================================

    def get_current_user(self) -> dict[str, Any]:
        """
        Get the current authenticated user.

        Returns:
            Dict with user profile information
        """
        return self._client.request("GET", "/api/v1/auth/me")

    def update_profile(
        self,
        name: str | None = None,
        avatar_url: str | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update user profile.

        Args:
            name: New display name
            avatar_url: New avatar URL
            preferences: User preferences dict

        Returns:
            Updated user profile
        """
        data = {}
        if name is not None:
            data["name"] = name
        if avatar_url is not None:
            data["avatar_url"] = avatar_url
        if preferences is not None:
            data["preferences"] = preferences

        return self._client.request("PATCH", "/api/v1/auth/me", json=data)

    # ===========================================================================
    # Password Management
    # ===========================================================================

    def change_password(
        self,
        current_password: str,
        new_password: str,
    ) -> dict[str, Any]:
        """
        Change the user's password.

        Args:
            current_password: Current password for verification
            new_password: New password (min 8 chars)

        Returns:
            Dict with success status
        """
        return self._client.request(
            "POST",
            "/api/v1/auth/password/change",
            json={
                "current_password": current_password,
                "new_password": new_password,
            },
        )

    def request_password_reset(self, email: str) -> dict[str, Any]:
        """
        Request a password reset email.

        Args:
            email: Email address to send reset link to

        Returns:
            Dict with success status
        """
        return self._client.request(
            "POST",
            "/api/v1/auth/password/forgot",
            json={"email": email},
        )

    def reset_password(self, token: str, new_password: str) -> dict[str, Any]:
        """
        Reset password using reset token.

        Args:
            token: Password reset token from email
            new_password: New password

        Returns:
            Dict with success status
        """
        return self._client.request(
            "POST",
            "/api/v1/auth/password/reset",
            json={"token": token, "new_password": new_password},
        )

    # ===========================================================================
    # Email Verification
    # ===========================================================================

    def verify_email(self, token: str) -> dict[str, Any]:
        """
        Verify email address with token.

        Args:
            token: Verification token from email

        Returns:
            Dict with verified status
        """
        return self._client.request(
            "POST",
            "/api/v1/auth/verify-email",
            json={"token": token},
        )

    def resend_verification(self) -> dict[str, Any]:
        """
        Resend email verification link.

        Returns:
            Dict with success status
        """
        return self._client.request("POST", "/api/v1/auth/verify-email/resend")

    # ===========================================================================
    # MFA (Multi-Factor Authentication)
    # ===========================================================================

    def setup_mfa(self, method: str = "totp") -> dict[str, Any]:
        """
        Start MFA setup process.

        Args:
            method: MFA method (totp, hotp)

        Returns:
            Dict with secret, qr_code_uri, and backup_codes
        """
        return self._client.request(
            "POST",
            "/api/v1/auth/mfa/setup",
            json={"method": method},
        )

    def verify_mfa_setup(self, code: str) -> dict[str, Any]:
        """
        Verify MFA setup with code.

        Args:
            code: MFA code from authenticator app

        Returns:
            Dict with verified status
        """
        return self._client.request(
            "POST",
            "/api/v1/auth/mfa/verify",
            json={"code": code},
        )

    def enable_mfa(self, code: str) -> dict[str, Any]:
        """
        Enable MFA after verification.

        Args:
            code: MFA code to confirm

        Returns:
            Dict with enabled status
        """
        return self._client.request(
            "POST",
            "/api/v1/auth/mfa/enable",
            json={"code": code},
        )

    def disable_mfa(self, password: str) -> dict[str, Any]:
        """
        Disable MFA (requires password confirmation).

        Args:
            password: Current password for verification

        Returns:
            Dict with success status
        """
        return self._client.request(
            "POST",
            "/api/v1/auth/mfa/disable",
            json={"password": password},
        )

    def generate_backup_codes(self) -> dict[str, Any]:
        """
        Generate new backup codes.

        Returns:
            Dict with codes array
        """
        return self._client.request("POST", "/api/v1/auth/mfa/backup-codes")

    # ===========================================================================
    # OAuth
    # ===========================================================================

    def get_oauth_url(
        self,
        provider: str,
        redirect_uri: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """
        Get OAuth authorization URL.

        Args:
            provider: OAuth provider (google, github, microsoft)
            redirect_uri: Callback URL
            state: CSRF state parameter

        Returns:
            Dict with auth_url
        """
        params = {"provider": provider}
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        if state:
            params["state"] = state

        return self._client.request("GET", "/api/v1/auth/oauth/url", params=params)

    def complete_oauth(
        self,
        provider: str,
        code: str,
        state: str | None = None,
    ) -> dict[str, Any]:
        """
        Complete OAuth flow with authorization code.

        Args:
            provider: OAuth provider
            code: Authorization code from callback
            state: CSRF state parameter

        Returns:
            Dict with access_token, refresh_token
        """
        data = {"provider": provider, "code": code}
        if state:
            data["state"] = state

        return self._client.request("POST", "/api/v1/auth/oauth/callback", json=data)

    def list_oauth_providers(self) -> dict[str, Any]:
        """
        List available OAuth providers.

        Returns:
            Dict with providers array
        """
        return self._client.request("GET", "/api/v1/auth/oauth/providers")

    # ===========================================================================
    # Sessions
    # ===========================================================================

    def list_sessions(self) -> dict[str, Any]:
        """
        List active sessions.

        Returns:
            Dict with sessions array
        """
        return self._client.request("GET", "/api/v1/auth/sessions")

    def revoke_session(self, session_id: str) -> dict[str, Any]:
        """
        Revoke a specific session.

        Args:
            session_id: Session ID to revoke

        Returns:
            Dict with success status
        """
        return self._client.request("DELETE", f"/api/v1/auth/sessions/{session_id}")

    # ===========================================================================
    # API Keys
    # ===========================================================================

    def list_api_keys(self) -> dict[str, Any]:
        """
        List API keys for the current user.

        Returns:
            Dict with keys array
        """
        return self._client.request("GET", "/api/v1/auth/api-keys")

    def create_api_key(
        self,
        name: str,
        scopes: list[str] | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a new API key.

        Args:
            name: Key name for identification
            scopes: Optional permission scopes
            expires_at: Optional expiration (ISO 8601)

        Returns:
            Dict with key info including the secret (only shown once)
        """
        data: dict[str, Any] = {"name": name}
        if scopes:
            data["scopes"] = scopes
        if expires_at:
            data["expires_at"] = expires_at

        return self._client.request("POST", "/api/v1/auth/api-keys", json=data)

    def revoke_api_key(self, key_id: str) -> dict[str, Any]:
        """
        Revoke an API key.

        Args:
            key_id: API key ID

        Returns:
            Dict with success status
        """
        return self._client.request("DELETE", f"/api/v1/auth/api-keys/{key_id}")


class AsyncAuthAPI:
    """
    Asynchronous Auth API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     token = await client.auth.login(email="user@example.com", password="secret")
        ...     user = await client.auth.get_current_user()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Registration and Login
    # ===========================================================================

    async def register(
        self,
        email: str,
        password: str,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Register a new user account."""
        data = {"email": email, "password": password}
        if name:
            data["name"] = name

        return await self._client.request("POST", "/api/v1/auth/register", json=data)

    async def login(
        self,
        email: str,
        password: str,
        mfa_code: str | None = None,
    ) -> dict[str, Any]:
        """Authenticate and get access token."""
        data = {"email": email, "password": password}
        if mfa_code:
            data["mfa_code"] = mfa_code

        return await self._client.request("POST", "/api/v1/auth/login", json=data)

    async def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh the access token."""
        return await self._client.request(
            "POST",
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )

    async def logout(self) -> dict[str, Any]:
        """Logout and invalidate current token."""
        return await self._client.request("POST", "/api/v1/auth/logout")

    async def logout_all(self) -> dict[str, Any]:
        """Logout from all sessions."""
        return await self._client.request("POST", "/api/v1/auth/logout/all")

    # ===========================================================================
    # User Profile
    # ===========================================================================

    async def get_current_user(self) -> dict[str, Any]:
        """Get the current authenticated user."""
        return await self._client.request("GET", "/api/v1/auth/me")

    async def update_profile(
        self,
        name: str | None = None,
        avatar_url: str | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update user profile."""
        data = {}
        if name is not None:
            data["name"] = name
        if avatar_url is not None:
            data["avatar_url"] = avatar_url
        if preferences is not None:
            data["preferences"] = preferences

        return await self._client.request("PATCH", "/api/v1/auth/me", json=data)

    # ===========================================================================
    # Password Management
    # ===========================================================================

    async def change_password(
        self,
        current_password: str,
        new_password: str,
    ) -> dict[str, Any]:
        """Change the user's password."""
        return await self._client.request(
            "POST",
            "/api/v1/auth/password/change",
            json={
                "current_password": current_password,
                "new_password": new_password,
            },
        )

    async def request_password_reset(self, email: str) -> dict[str, Any]:
        """Request a password reset email."""
        return await self._client.request(
            "POST",
            "/api/v1/auth/password/forgot",
            json={"email": email},
        )

    async def reset_password(self, token: str, new_password: str) -> dict[str, Any]:
        """Reset password using reset token."""
        return await self._client.request(
            "POST",
            "/api/v1/auth/password/reset",
            json={"token": token, "new_password": new_password},
        )

    # ===========================================================================
    # Email Verification
    # ===========================================================================

    async def verify_email(self, token: str) -> dict[str, Any]:
        """Verify email address with token."""
        return await self._client.request(
            "POST",
            "/api/v1/auth/verify-email",
            json={"token": token},
        )

    async def resend_verification(self) -> dict[str, Any]:
        """Resend email verification link."""
        return await self._client.request("POST", "/api/v1/auth/verify-email/resend")

    # ===========================================================================
    # MFA (Multi-Factor Authentication)
    # ===========================================================================

    async def setup_mfa(self, method: str = "totp") -> dict[str, Any]:
        """Start MFA setup process."""
        return await self._client.request(
            "POST",
            "/api/v1/auth/mfa/setup",
            json={"method": method},
        )

    async def verify_mfa_setup(self, code: str) -> dict[str, Any]:
        """Verify MFA setup with code."""
        return await self._client.request(
            "POST",
            "/api/v1/auth/mfa/verify",
            json={"code": code},
        )

    async def enable_mfa(self, code: str) -> dict[str, Any]:
        """Enable MFA after verification."""
        return await self._client.request(
            "POST",
            "/api/v1/auth/mfa/enable",
            json={"code": code},
        )

    async def disable_mfa(self, password: str) -> dict[str, Any]:
        """Disable MFA (requires password confirmation)."""
        return await self._client.request(
            "POST",
            "/api/v1/auth/mfa/disable",
            json={"password": password},
        )

    async def generate_backup_codes(self) -> dict[str, Any]:
        """Generate new backup codes."""
        return await self._client.request("POST", "/api/v1/auth/mfa/backup-codes")

    # ===========================================================================
    # OAuth
    # ===========================================================================

    async def get_oauth_url(
        self,
        provider: str,
        redirect_uri: str | None = None,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Get OAuth authorization URL."""
        params = {"provider": provider}
        if redirect_uri:
            params["redirect_uri"] = redirect_uri
        if state:
            params["state"] = state

        return await self._client.request("GET", "/api/v1/auth/oauth/url", params=params)

    async def complete_oauth(
        self,
        provider: str,
        code: str,
        state: str | None = None,
    ) -> dict[str, Any]:
        """Complete OAuth flow with authorization code."""
        data = {"provider": provider, "code": code}
        if state:
            data["state"] = state

        return await self._client.request("POST", "/api/v1/auth/oauth/callback", json=data)

    async def list_oauth_providers(self) -> dict[str, Any]:
        """List available OAuth providers."""
        return await self._client.request("GET", "/api/v1/auth/oauth/providers")

    # ===========================================================================
    # Sessions
    # ===========================================================================

    async def list_sessions(self) -> dict[str, Any]:
        """List active sessions."""
        return await self._client.request("GET", "/api/v1/auth/sessions")

    async def revoke_session(self, session_id: str) -> dict[str, Any]:
        """Revoke a specific session."""
        return await self._client.request("DELETE", f"/api/v1/auth/sessions/{session_id}")

    # ===========================================================================
    # API Keys
    # ===========================================================================

    async def list_api_keys(self) -> dict[str, Any]:
        """List API keys for the current user."""
        return await self._client.request("GET", "/api/v1/auth/api-keys")

    async def create_api_key(
        self,
        name: str,
        scopes: list[str] | None = None,
        expires_at: str | None = None,
    ) -> dict[str, Any]:
        """Create a new API key."""
        data: dict[str, Any] = {"name": name}
        if scopes:
            data["scopes"] = scopes
        if expires_at:
            data["expires_at"] = expires_at

        return await self._client.request("POST", "/api/v1/auth/api-keys", json=data)

    async def revoke_api_key(self, key_id: str) -> dict[str, Any]:
        """Revoke an API key."""
        return await self._client.request("DELETE", f"/api/v1/auth/api-keys/{key_id}")
