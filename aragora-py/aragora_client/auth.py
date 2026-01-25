"""Authentication API for the Aragora SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from aragora_client.client import AragoraClient


# Request/Response models
class LoginRequest(BaseModel):
    """Login request."""

    email: str
    password: str


class RegisterRequest(BaseModel):
    """Register request."""

    email: str
    password: str
    name: str | None = None


class AuthToken(BaseModel):
    """Authentication token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshRequest(BaseModel):
    """Token refresh request."""

    refresh_token: str


class VerifyEmailRequest(BaseModel):
    """Email verification request."""

    token: str


class ChangePasswordRequest(BaseModel):
    """Change password request."""

    current_password: str
    new_password: str


class ForgotPasswordRequest(BaseModel):
    """Forgot password request."""

    email: str


class ResetPasswordRequest(BaseModel):
    """Reset password request."""

    token: str
    new_password: str


class User(BaseModel):
    """User profile."""

    id: str
    email: str
    name: str | None = None
    email_verified: bool = False
    mfa_enabled: bool = False
    created_at: str | None = None
    updated_at: str | None = None


class UpdateProfileRequest(BaseModel):
    """Update profile request."""

    name: str | None = None
    avatar_url: str | None = None


class OAuthUrlParams(BaseModel):
    """OAuth URL parameters."""

    provider: str
    redirect_uri: str
    state: str | None = None
    scope: str | None = None


class OAuthUrl(BaseModel):
    """OAuth authorization URL response."""

    url: str
    state: str


class OAuthCallbackRequest(BaseModel):
    """OAuth callback request."""

    provider: str
    code: str
    state: str | None = None
    redirect_uri: str | None = None


class MFASetupRequest(BaseModel):
    """MFA setup request."""

    method: str = "totp"


class MFASetupResponse(BaseModel):
    """MFA setup response."""

    secret: str
    qr_code_uri: str
    backup_codes: list[str] | None = None


class MFAVerifyRequest(BaseModel):
    """MFA verification request."""

    code: str


class MFAVerifyResponse(BaseModel):
    """MFA verification response."""

    verified: bool
    backup_codes: list[str] | None = None


class APIKey(BaseModel):
    """API key model."""

    id: str
    name: str
    key_prefix: str
    created_at: str
    expires_at: str | None = None
    last_used_at: str | None = None
    scopes: list[str] = []


class CreateAPIKeyRequest(BaseModel):
    """Create API key request."""

    name: str
    scopes: list[str] | None = None
    expires_in_days: int | None = None


class CreateAPIKeyResponse(BaseModel):
    """Create API key response."""

    key: str
    api_key: APIKey


class Session(BaseModel):
    """User session."""

    id: str
    user_agent: str | None = None
    ip_address: str | None = None
    created_at: str
    last_active_at: str
    is_current: bool = False


class AuthAPI:
    """API for authentication operations."""

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    async def register(
        self,
        email: str,
        password: str,
        *,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Register a new user."""
        request = RegisterRequest(email=email, password=password, name=name)
        return await self._client._post("/api/v1/auth/register", request.model_dump())

    async def login(self, email: str, password: str) -> AuthToken:
        """Login with email and password."""
        request = LoginRequest(email=email, password=password)
        data = await self._client._post("/api/v1/auth/login", request.model_dump())
        return AuthToken.model_validate(data)

    async def refresh_token(self, refresh_token: str) -> AuthToken:
        """Refresh an access token using a refresh token."""
        request = RefreshRequest(refresh_token=refresh_token)
        data = await self._client._post("/api/v1/auth/refresh", request.model_dump())
        return AuthToken.model_validate(data)

    async def logout(self) -> None:
        """Logout and invalidate the current session."""
        await self._client._post("/api/v1/auth/logout", {})

    async def logout_all(self) -> dict[str, Any]:
        """Logout from all sessions."""
        return await self._client._post("/api/v1/auth/logout-all", {})

    async def verify_email(self, token: str) -> dict[str, Any]:
        """Verify email address with a verification token."""
        request = VerifyEmailRequest(token=token)
        return await self._client._post(
            "/api/v1/auth/verify-email", request.model_dump()
        )

    async def get_current_user(self) -> User:
        """Get the current authenticated user's profile."""
        data = await self._client._get("/api/v1/auth/me")
        return User.model_validate(data)

    async def update_profile(
        self,
        *,
        name: str | None = None,
        avatar_url: str | None = None,
    ) -> dict[str, Any]:
        """Update the current user's profile."""
        request = UpdateProfileRequest(name=name, avatar_url=avatar_url)
        return await self._client._patch("/api/v1/auth/me", request.model_dump())

    async def change_password(
        self,
        current_password: str,
        new_password: str,
    ) -> None:
        """Change the current user's password."""
        request = ChangePasswordRequest(
            current_password=current_password, new_password=new_password
        )
        await self._client._post("/api/v1/auth/change-password", request.model_dump())

    async def request_password_reset(self, email: str) -> None:
        """Request a password reset email."""
        request = ForgotPasswordRequest(email=email)
        await self._client._post("/api/v1/auth/forgot-password", request.model_dump())

    async def reset_password(self, token: str, new_password: str) -> None:
        """Reset password using a reset token."""
        request = ResetPasswordRequest(token=token, new_password=new_password)
        await self._client._post("/api/v1/auth/reset-password", request.model_dump())

    async def get_oauth_url(
        self,
        provider: str,
        redirect_uri: str,
        *,
        state: str | None = None,
        scope: str | None = None,
    ) -> OAuthUrl:
        """Get an OAuth authorization URL for a provider."""
        params: dict[str, str] = {
            "provider": provider,
            "redirect_uri": redirect_uri,
        }
        if state:
            params["state"] = state
        if scope:
            params["scope"] = scope
        data = await self._client._get("/api/v1/auth/oauth/authorize", params=params)
        return OAuthUrl.model_validate(data)

    async def complete_oauth(
        self,
        provider: str,
        code: str,
        *,
        state: str | None = None,
        redirect_uri: str | None = None,
    ) -> AuthToken:
        """Complete OAuth flow with authorization code."""
        request = OAuthCallbackRequest(
            provider=provider,
            code=code,
            state=state,
            redirect_uri=redirect_uri,
        )
        data = await self._client._post(
            "/api/v1/auth/oauth/callback", request.model_dump()
        )
        return AuthToken.model_validate(data)

    async def setup_mfa(self, method: str = "totp") -> MFASetupResponse:
        """Setup multi-factor authentication."""
        request = MFASetupRequest(method=method)
        data = await self._client._post("/api/v1/auth/mfa/setup", request.model_dump())
        return MFASetupResponse.model_validate(data)

    async def verify_mfa_setup(self, code: str) -> MFAVerifyResponse:
        """Verify MFA setup with a code."""
        request = MFAVerifyRequest(code=code)
        data = await self._client._post("/api/v1/auth/mfa/verify", request.model_dump())
        return MFAVerifyResponse.model_validate(data)

    async def disable_mfa(self) -> None:
        """Disable multi-factor authentication."""
        await self._client._delete("/api/v1/auth/mfa")

    async def verify_mfa(self, code: str) -> AuthToken:
        """Verify MFA code during login."""
        request = MFAVerifyRequest(code=code)
        data = await self._client._post(
            "/api/v1/auth/mfa/challenge", request.model_dump()
        )
        return AuthToken.model_validate(data)

    # API Key management
    async def list_api_keys(self) -> list[APIKey]:
        """List all API keys for the current user."""
        data = await self._client._get("/api/v1/auth/api-keys")
        return [APIKey.model_validate(k) for k in data.get("api_keys", [])]

    async def create_api_key(
        self,
        name: str,
        *,
        scopes: list[str] | None = None,
        expires_in_days: int | None = None,
    ) -> CreateAPIKeyResponse:
        """Create a new API key."""
        request = CreateAPIKeyRequest(
            name=name, scopes=scopes, expires_in_days=expires_in_days
        )
        data = await self._client._post("/api/v1/auth/api-keys", request.model_dump())
        return CreateAPIKeyResponse.model_validate(data)

    async def revoke_api_key(self, key_id: str) -> None:
        """Revoke an API key."""
        await self._client._delete(f"/api/v1/auth/api-keys/{key_id}")

    # Session management
    async def list_sessions(self) -> list[Session]:
        """List all active sessions for the current user."""
        data = await self._client._get("/api/v1/auth/sessions")
        return [Session.model_validate(s) for s in data.get("sessions", [])]

    async def revoke_session(self, session_id: str) -> None:
        """Revoke a specific session."""
        await self._client._delete(f"/api/v1/auth/sessions/{session_id}")

    async def revoke_all_sessions(self) -> dict[str, Any]:
        """Revoke all sessions except the current one."""
        return await self._client._post("/api/v1/auth/sessions/revoke-all", {})
