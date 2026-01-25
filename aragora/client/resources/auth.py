"""
Auth API resource for the Aragora client.

Provides methods for authentication and MFA:
- Login and token management
- MFA setup and verification
- API key management
- Session management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..client import AragoraClient

logger = logging.getLogger(__name__)


@dataclass
class User:
    """A user profile."""

    id: str
    email: str
    name: Optional[str] = None
    mfa_enabled: bool = False
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    roles: List[str] = field(default_factory=list)


@dataclass
class Session:
    """An authenticated session."""

    session_id: str
    user_id: str
    expires_at: datetime
    created_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class APIKey:
    """An API key."""

    id: str
    name: str
    key_prefix: str
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    scopes: List[str] = field(default_factory=list)


@dataclass
class MFASetupResult:
    """Result of MFA setup initiation."""

    secret: str
    qr_code_url: str
    backup_codes: List[str] = field(default_factory=list)


class AuthAPI:
    """API interface for authentication and MFA."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Authentication
    # =========================================================================

    def login(
        self,
        email: str,
        password: str,
        mfa_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Authenticate a user.

        Args:
            email: User email
            password: User password
            mfa_code: MFA code if MFA is enabled

        Returns:
            Authentication result with tokens
        """
        body: Dict[str, Any] = {
            "email": email,
            "password": password,
        }
        if mfa_code:
            body["mfa_code"] = mfa_code

        return self._client._post("/api/v1/auth/login", body)

    async def login_async(
        self,
        email: str,
        password: str,
        mfa_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of login()."""
        body: Dict[str, Any] = {
            "email": email,
            "password": password,
        }
        if mfa_code:
            body["mfa_code"] = mfa_code

        return await self._client._post_async("/api/v1/auth/login", body)

    def logout(self) -> bool:
        """Log out the current session."""
        self._client._post("/api/v1/auth/logout", {})
        return True

    async def logout_async(self) -> bool:
        """Async version of logout()."""
        await self._client._post_async("/api/v1/auth/logout", {})
        return True

    def refresh_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh an access token.

        Args:
            refresh_token: The refresh token

        Returns:
            New access token
        """
        body = {"refresh_token": refresh_token}
        return self._client._post("/api/v1/auth/refresh", body)

    async def refresh_token_async(self, refresh_token: str) -> Dict[str, Any]:
        """Async version of refresh_token()."""
        body = {"refresh_token": refresh_token}
        return await self._client._post_async("/api/v1/auth/refresh", body)

    def get_current_user(self) -> User:
        """Get the current authenticated user."""
        response = self._client._get("/api/v1/auth/me")
        return User(**response)

    async def get_current_user_async(self) -> User:
        """Async version of get_current_user()."""
        response = await self._client._get_async("/api/v1/auth/me")
        return User(**response)

    def update_profile(
        self,
        name: Optional[str] = None,
        email: Optional[str] = None,
    ) -> User:
        """Update current user profile."""
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if email is not None:
            body["email"] = email

        response = self._client._patch("/api/v1/auth/me", body)
        return User(**response)

    async def update_profile_async(
        self,
        name: Optional[str] = None,
        email: Optional[str] = None,
    ) -> User:
        """Async version of update_profile()."""
        body: Dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if email is not None:
            body["email"] = email

        response = await self._client._patch_async("/api/v1/auth/me", body)
        return User(**response)

    def change_password(
        self,
        current_password: str,
        new_password: str,
    ) -> bool:
        """
        Change the current user's password.

        Args:
            current_password: Current password
            new_password: New password

        Returns:
            True if successful
        """
        body = {
            "current_password": current_password,
            "new_password": new_password,
        }
        self._client._post("/api/v1/auth/change-password", body)
        return True

    async def change_password_async(
        self,
        current_password: str,
        new_password: str,
    ) -> bool:
        """Async version of change_password()."""
        body = {
            "current_password": current_password,
            "new_password": new_password,
        }
        await self._client._post_async("/api/v1/auth/change-password", body)
        return True

    # =========================================================================
    # MFA
    # =========================================================================

    def setup_mfa(self) -> MFASetupResult:
        """
        Initiate MFA setup.

        Returns:
            MFA setup information including secret and QR code
        """
        response = self._client._post("/api/v1/auth/mfa/setup", {})
        return MFASetupResult(
            secret=response.get("secret", ""),
            qr_code_url=response.get("qr_code_url", response.get("qr_uri", "")),
            backup_codes=response.get("backup_codes", []),
        )

    async def setup_mfa_async(self) -> MFASetupResult:
        """Async version of setup_mfa()."""
        response = await self._client._post_async("/api/v1/auth/mfa/setup", {})
        return MFASetupResult(
            secret=response.get("secret", ""),
            qr_code_url=response.get("qr_code_url", response.get("qr_uri", "")),
            backup_codes=response.get("backup_codes", []),
        )

    def verify_mfa_setup(self, code: str) -> bool:
        """
        Verify MFA setup with a code.

        Args:
            code: TOTP code from authenticator app

        Returns:
            True if verified successfully
        """
        body = {"code": code}
        response = self._client._post("/api/v1/auth/mfa/verify", body)
        return response.get("verified", False)

    async def verify_mfa_setup_async(self, code: str) -> bool:
        """Async version of verify_mfa_setup()."""
        body = {"code": code}
        response = await self._client._post_async("/api/v1/auth/mfa/verify", body)
        return response.get("verified", False)

    def enable_mfa(self, code: str) -> bool:
        """
        Enable MFA after setup verification.

        Args:
            code: TOTP code from authenticator app

        Returns:
            True if enabled successfully
        """
        body = {"code": code}
        response = self._client._post("/api/v1/auth/mfa/enable", body)
        return response.get("enabled", False)

    async def enable_mfa_async(self, code: str) -> bool:
        """Async version of enable_mfa()."""
        body = {"code": code}
        response = await self._client._post_async("/api/v1/auth/mfa/enable", body)
        return response.get("enabled", False)

    def disable_mfa(self, code: str) -> bool:
        """
        Disable MFA.

        Args:
            code: TOTP code to verify identity

        Returns:
            True if disabled successfully
        """
        body = {"code": code}
        self._client._post("/api/v1/auth/mfa/disable", body)
        return True

    async def disable_mfa_async(self, code: str) -> bool:
        """Async version of disable_mfa()."""
        body = {"code": code}
        await self._client._post_async("/api/v1/auth/mfa/disable", body)
        return True

    def get_backup_codes(self) -> List[str]:
        """Generate new MFA backup codes."""
        response = self._client._post("/api/v1/auth/mfa/backup-codes", {})
        return response.get("codes", [])

    async def get_backup_codes_async(self) -> List[str]:
        """Async version of get_backup_codes()."""
        response = await self._client._post_async("/api/v1/auth/mfa/backup-codes", {})
        return response.get("codes", [])

    def get_mfa_status(self) -> Dict[str, Any]:
        """Get MFA status for current user."""
        return self._client._get("/api/v1/auth/mfa/status")

    async def get_mfa_status_async(self) -> Dict[str, Any]:
        """Async version of get_mfa_status()."""
        return await self._client._get_async("/api/v1/auth/mfa/status")

    # =========================================================================
    # Sessions
    # =========================================================================

    def list_sessions(self) -> List[Session]:
        """List active sessions for the current user."""
        response = self._client._get("/api/v1/auth/sessions")
        sessions = response.get("sessions", [])
        return [Session(**s) for s in sessions]

    async def list_sessions_async(self) -> List[Session]:
        """Async version of list_sessions()."""
        response = await self._client._get_async("/api/v1/auth/sessions")
        sessions = response.get("sessions", [])
        return [Session(**s) for s in sessions]

    def revoke_session(self, session_id: str) -> bool:
        """Revoke a specific session."""
        self._client._delete(f"/api/v1/auth/sessions/{session_id}")
        return True

    async def revoke_session_async(self, session_id: str) -> bool:
        """Async version of revoke_session()."""
        await self._client._delete_async(f"/api/v1/auth/sessions/{session_id}")
        return True

    def revoke_all_sessions(self, except_current: bool = True) -> int:
        """
        Revoke all sessions.

        Args:
            except_current: Keep the current session active

        Returns:
            Number of sessions revoked
        """
        params = {"except_current": except_current}
        response = self._client._delete("/api/v1/auth/sessions", params)
        return response.get("revoked_count", 0)

    async def revoke_all_sessions_async(self, except_current: bool = True) -> int:
        """Async version of revoke_all_sessions()."""
        params = {"except_current": except_current}
        response = await self._client._delete_async("/api/v1/auth/sessions", params)
        return response.get("revoked_count", 0)

    # =========================================================================
    # API Keys
    # =========================================================================

    def list_api_keys(self) -> List[APIKey]:
        """List API keys for the current user."""
        response = self._client._get("/api/v1/auth/api-keys")
        keys = response.get("keys", response.get("api_keys", []))
        return [APIKey(**k) for k in keys]

    async def list_api_keys_async(self) -> List[APIKey]:
        """Async version of list_api_keys()."""
        response = await self._client._get_async("/api/v1/auth/api-keys")
        keys = response.get("keys", response.get("api_keys", []))
        return [APIKey(**k) for k in keys]

    def create_api_key(
        self,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create a new API key.

        Args:
            name: Name for the API key
            scopes: Permission scopes
            expires_in_days: Days until expiration (None = never)

        Returns:
            Created API key (full key only shown once)
        """
        body: Dict[str, Any] = {"name": name}
        if scopes:
            body["scopes"] = scopes
        if expires_in_days:
            body["expires_in_days"] = expires_in_days

        return self._client._post("/api/v1/auth/api-keys", body)

    async def create_api_key_async(
        self,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Async version of create_api_key()."""
        body: Dict[str, Any] = {"name": name}
        if scopes:
            body["scopes"] = scopes
        if expires_in_days:
            body["expires_in_days"] = expires_in_days

        return await self._client._post_async("/api/v1/auth/api-keys", body)

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        self._client._delete(f"/api/v1/auth/api-keys/{key_id}")
        return True

    async def revoke_api_key_async(self, key_id: str) -> bool:
        """Async version of revoke_api_key()."""
        await self._client._delete_async(f"/api/v1/auth/api-keys/{key_id}")
        return True


__all__ = [
    "AuthAPI",
    "User",
    "Session",
    "APIKey",
    "MFASetupResult",
]
