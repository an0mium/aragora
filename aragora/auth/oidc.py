"""
OpenID Connect (OIDC) Authentication Provider for Aragora.

Implements OIDC/OAuth 2.0 for SSO with common providers:
- Azure AD
- Okta
- Google Workspace
- Auth0
- Keycloak
- Generic OIDC providers

Usage:
    from aragora.auth.oidc import OIDCProvider, OIDCConfig

    config = OIDCConfig(
        client_id="your-client-id",
        client_secret="your-client-secret",
        issuer_url="https://login.microsoftonline.com/tenant-id/v2.0",
        callback_url="https://aragora.example.com/auth/callback",
    )

    provider = OIDCProvider(config)
    auth_url = await provider.get_authorization_url(state="...")
    user = await provider.authenticate(code="...")
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode, urljoin

from .sso import (
    SSOProvider,
    SSOProviderType,
    SSOConfig,
    SSOUser,
    SSOError,
    SSOAuthenticationError,
    SSOConfigurationError,
)

logger = logging.getLogger(__name__)

# Optional: PyJWT for token validation
try:
    import jwt
    from jwt import PyJWKClient
    HAS_JWT = True
except ImportError:
    jwt = None
    PyJWKClient = None
    HAS_JWT = False

# Optional: httpx for async HTTP
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    httpx = None
    HAS_HTTPX = False


class OIDCError(SSOError):
    """OIDC-specific error."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "OIDC_ERROR", details)


@dataclass
class OIDCConfig(SSOConfig):
    """
    OpenID Connect configuration.

    Extends base SSOConfig with OIDC-specific settings.
    """

    # Client credentials
    client_id: str = ""
    client_secret: str = ""

    # OIDC Discovery
    issuer_url: str = ""  # Used for auto-discovery via .well-known/openid-configuration

    # Manual endpoint configuration (optional, auto-discovered from issuer)
    authorization_endpoint: str = ""
    token_endpoint: str = ""
    userinfo_endpoint: str = ""
    jwks_uri: str = ""
    end_session_endpoint: str = ""

    # Scopes
    scopes: List[str] = field(default_factory=lambda: ["openid", "email", "profile"])

    # PKCE (Proof Key for Code Exchange)
    use_pkce: bool = True

    # Token validation
    validate_tokens: bool = True
    allowed_audiences: List[str] = field(default_factory=list)

    # Claim mapping (OIDC claim -> user field)
    claim_mapping: Dict[str, str] = field(default_factory=lambda: {
        "sub": "id",
        "email": "email",
        "name": "name",
        "given_name": "first_name",
        "family_name": "last_name",
        "preferred_username": "username",
        "groups": "groups",
        "roles": "roles",
    })

    def __post_init__(self):
        if not self.provider_type:
            self.provider_type = SSOProviderType.OIDC

    def validate(self) -> List[str]:
        """Validate OIDC configuration."""
        errors = super().validate()

        if not self.client_id:
            errors.append("client_id is required")

        if not self.client_secret:
            errors.append("client_secret is required")

        if not self.issuer_url:
            if not self.authorization_endpoint or not self.token_endpoint:
                errors.append("issuer_url or explicit endpoints are required")

        return errors


# Well-known provider configurations
PROVIDER_CONFIGS: Dict[str, Dict[str, str]] = {
    "azure_ad": {
        "authorization_endpoint": "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize",
        "token_endpoint": "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token",
        "userinfo_endpoint": "https://graph.microsoft.com/oidc/userinfo",
        "jwks_uri": "https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys",
    },
    "okta": {
        "authorization_endpoint": "{domain}/oauth2/v1/authorize",
        "token_endpoint": "{domain}/oauth2/v1/token",
        "userinfo_endpoint": "{domain}/oauth2/v1/userinfo",
        "jwks_uri": "{domain}/oauth2/v1/keys",
    },
    "google": {
        "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_endpoint": "https://oauth2.googleapis.com/token",
        "userinfo_endpoint": "https://openidconnect.googleapis.com/v1/userinfo",
        "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
    },
    "github": {
        "authorization_endpoint": "https://github.com/login/oauth/authorize",
        "token_endpoint": "https://github.com/login/oauth/access_token",
        "userinfo_endpoint": "https://api.github.com/user",
        # GitHub doesn't use JWKS (uses opaque tokens)
    },
}


class OIDCProvider(SSOProvider):
    """
    OpenID Connect provider implementation.

    Supports:
    - Authorization Code flow with PKCE
    - Token validation via JWKS
    - Auto-discovery via .well-known
    - Common IdP presets (Azure AD, Okta, Google)
    """

    def __init__(self, config: OIDCConfig):
        super().__init__(config)
        self.config: OIDCConfig = config

        # Validate config
        errors = config.validate()
        if errors:
            raise SSOConfigurationError(
                f"Invalid OIDC configuration: {', '.join(errors)}",
                {"errors": errors}
            )

        # PKCE state (code_verifier stored by state)
        self._pkce_store: Dict[str, str] = {}

        # Discovery cache
        self._discovery_cache: Optional[Dict[str, Any]] = None
        self._discovery_cached_at: float = 0

        # JWKS client
        self._jwks_client: Optional[Any] = None

    @property
    def provider_type(self) -> SSOProviderType:
        return self.config.provider_type

    async def _discover_endpoints(self) -> Dict[str, Any]:
        """Fetch OIDC discovery document."""
        # Check cache (1 hour TTL)
        if self._discovery_cache and time.time() - self._discovery_cached_at < 3600:
            return self._discovery_cache

        if not self.config.issuer_url:
            return {}

        discovery_url = urljoin(
            self.config.issuer_url.rstrip("/") + "/",
            ".well-known/openid-configuration"
        )

        try:
            if HAS_HTTPX:
                async with httpx.AsyncClient() as client:
                    response = await client.get(discovery_url, timeout=10.0)
                    response.raise_for_status()
                    self._discovery_cache = response.json()
            else:
                # Fallback to sync requests
                import urllib.request
                with urllib.request.urlopen(discovery_url, timeout=10) as resp:
                    self._discovery_cache = json.loads(resp.read().decode())

            self._discovery_cached_at = time.time()
            logger.debug(f"OIDC discovery successful for {self.config.issuer_url}")
            return self._discovery_cache

        except Exception as e:
            logger.warning(f"OIDC discovery failed: {e}")
            return {}

    async def _get_endpoint(self, name: str) -> str:
        """Get endpoint URL, preferring config over discovery."""
        # Check config first
        config_value = getattr(self.config, name, "")
        if config_value:
            return config_value

        # Try discovery
        discovery = await self._discover_endpoints()
        return discovery.get(name, "")

    def _generate_pkce(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge."""
        # Generate random code verifier (43-128 chars)
        code_verifier = secrets.token_urlsafe(64)

        # Create code challenge (SHA256 + base64url)
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        code_challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

        return code_verifier, code_challenge

    async def get_authorization_url(
        self,
        state: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        nonce: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate OIDC authorization URL.

        Args:
            state: CSRF state parameter
            redirect_uri: Override callback URL
            scopes: Override scopes
            nonce: ID token nonce (auto-generated if not provided)

        Returns:
            Authorization URL to redirect user to
        """
        auth_endpoint = await self._get_endpoint("authorization_endpoint")
        if not auth_endpoint:
            raise SSOConfigurationError("No authorization_endpoint configured or discovered")

        # Generate state if not provided
        if not state:
            state = self.generate_state()
        else:
            self._state_store[state] = time.time()

        # Build parameters
        params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri or self.config.callback_url,
            "scope": " ".join(scopes or self.config.scopes),
            "state": state,
        }

        # Add nonce for ID token validation
        if not nonce:
            nonce = secrets.token_urlsafe(16)
        params["nonce"] = nonce

        # PKCE
        if self.config.use_pkce:
            code_verifier, code_challenge = self._generate_pkce()
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"
            self._pkce_store[state] = code_verifier

        # Provider-specific parameters
        if self.config.provider_type == SSOProviderType.AZURE_AD:
            params["response_mode"] = "query"

        # Add any extra parameters
        params.update(kwargs)

        return f"{auth_endpoint}?{urlencode(params)}"

    async def authenticate(
        self,
        code: Optional[str] = None,
        state: Optional[str] = None,
        **kwargs,
    ) -> SSOUser:
        """
        Authenticate user from OIDC callback.

        Args:
            code: Authorization code from IdP
            state: State parameter for CSRF validation

        Returns:
            Authenticated user

        Raises:
            SSOAuthenticationError: If authentication fails
        """
        if not code:
            raise SSOAuthenticationError("No authorization code provided")

        # Validate state
        if state and not self.validate_state(state):
            raise SSOAuthenticationError(
                "Invalid or expired state parameter",
                {"code": "INVALID_STATE"}
            )

        # Get PKCE code verifier
        code_verifier = None
        if self.config.use_pkce and state:
            code_verifier = self._pkce_store.pop(state, None)

        # Exchange code for tokens
        tokens = await self._exchange_code(code, code_verifier)

        # Get user info
        user = await self._get_user_info(tokens)

        # Check domain restriction
        if not self.is_domain_allowed(user.email):
            raise SSOAuthenticationError(
                f"Email domain not allowed: {user.email.split('@')[-1]}",
                {"code": "DOMAIN_NOT_ALLOWED"}
            )

        logger.info(f"OIDC authentication successful for {user.email}")
        return user

    async def _exchange_code(
        self,
        code: str,
        code_verifier: Optional[str],
    ) -> Dict[str, Any]:
        """Exchange authorization code for tokens."""
        token_endpoint = await self._get_endpoint("token_endpoint")
        if not token_endpoint:
            raise SSOConfigurationError("No token_endpoint configured or discovered")

        data = {
            "grant_type": "authorization_code",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "code": code,
            "redirect_uri": self.config.callback_url,
        }

        if code_verifier:
            data["code_verifier"] = code_verifier

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        # GitHub needs Accept header
        if self.config.provider_type == SSOProviderType.GITHUB:
            headers["Accept"] = "application/json"

        try:
            if HAS_HTTPX:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        token_endpoint,
                        data=data,
                        headers=headers,
                        timeout=30.0
                    )
                    response.raise_for_status()
                    return response.json()
            else:
                # Fallback to sync
                import urllib.request
                import urllib.parse

                req_data = urllib.parse.urlencode(data).encode()
                req = urllib.request.Request(
                    token_endpoint,
                    data=req_data,
                    headers=headers,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode())

        except Exception as e:
            logger.error(f"Token exchange failed: {e}")
            raise SSOAuthenticationError(f"Token exchange failed: {e}")

    async def _get_user_info(self, tokens: Dict[str, Any]) -> SSOUser:
        """Get user info from tokens or userinfo endpoint."""
        access_token = tokens.get("access_token")
        id_token = tokens.get("id_token")

        claims: Dict[str, Any] = {}

        # Parse ID token if available
        if id_token and HAS_JWT:
            try:
                # Validate and decode ID token
                claims = await self._validate_id_token(id_token)
            except Exception as e:
                logger.warning(f"ID token validation failed: {e}")
                # Fall back to userinfo endpoint

        # Fetch from userinfo endpoint if needed
        if not claims.get("email"):
            userinfo = await self._fetch_userinfo(access_token)
            claims.update(userinfo)

        # Map claims to user
        return self._claims_to_user(claims, tokens)

    async def _validate_id_token(self, id_token: str) -> Dict[str, Any]:
        """Validate and decode ID token using JWKS."""
        if not HAS_JWT:
            raise SSOError("PyJWT required for ID token validation")

        jwks_uri = await self._get_endpoint("jwks_uri")
        if not jwks_uri:
            # Decode without validation (not recommended)
            logger.warning("No JWKS URI - decoding ID token without signature validation")
            return jwt.decode(id_token, options={"verify_signature": False})

        # Get or create JWKS client
        if not self._jwks_client:
            self._jwks_client = PyJWKClient(jwks_uri)

        # Get signing key
        signing_key = self._jwks_client.get_signing_key_from_jwt(id_token)

        # Decode and validate
        audiences = self.config.allowed_audiences or [self.config.client_id]

        return jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=audiences,
            issuer=self.config.issuer_url,
        )

    async def _fetch_userinfo(self, access_token: str) -> Dict[str, Any]:
        """Fetch user info from userinfo endpoint."""
        userinfo_endpoint = await self._get_endpoint("userinfo_endpoint")
        if not userinfo_endpoint:
            return {}

        headers = {"Authorization": f"Bearer {access_token}"}

        try:
            if HAS_HTTPX:
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        userinfo_endpoint,
                        headers=headers,
                        timeout=10.0
                    )
                    response.raise_for_status()
                    return response.json()
            else:
                import urllib.request
                req = urllib.request.Request(userinfo_endpoint, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return json.loads(resp.read().decode())

        except Exception as e:
            logger.warning(f"Userinfo fetch failed: {e}")
            return {}

    def _claims_to_user(
        self,
        claims: Dict[str, Any],
        tokens: Dict[str, Any],
    ) -> SSOUser:
        """Map OIDC claims to SSOUser."""
        mapping = self.config.claim_mapping

        # Extract basic fields
        user_id = claims.get(self._find_claim_key(claims, "sub", mapping), "")
        email = claims.get(self._find_claim_key(claims, "email", mapping), "")
        name = claims.get(self._find_claim_key(claims, "name", mapping), "")
        first_name = claims.get(self._find_claim_key(claims, "given_name", mapping), "")
        last_name = claims.get(self._find_claim_key(claims, "family_name", mapping), "")
        username = claims.get(self._find_claim_key(claims, "preferred_username", mapping), "")

        # Extract roles/groups (may be nested or list)
        roles = self._extract_list_claim(claims, "roles", mapping)
        groups = self._extract_list_claim(claims, "groups", mapping)

        # Handle Azure AD group claims
        if "wids" in claims:  # Azure AD role IDs
            roles.extend(claims["wids"])

        return SSOUser(
            id=user_id,
            email=email,
            name=name,
            first_name=first_name,
            last_name=last_name,
            username=username,
            roles=self.map_roles(roles),
            groups=self.map_groups(groups),
            provider_type=self.config.provider_type.value,
            provider_id=self.config.issuer_url or self.config.client_id,
            access_token=tokens.get("access_token"),
            refresh_token=tokens.get("refresh_token"),
            id_token=tokens.get("id_token"),
            token_expires_at=time.time() + tokens.get("expires_in", 3600),
            raw_claims=claims,
        )

    def _find_claim_key(
        self,
        claims: Dict[str, Any],
        target: str,
        mapping: Dict[str, str],
    ) -> str:
        """Find the claim key that maps to target field."""
        for claim_key, field_name in mapping.items():
            if field_name == target and claim_key in claims:
                return claim_key
        return target  # Fallback to direct lookup

    def _extract_list_claim(
        self,
        claims: Dict[str, Any],
        target: str,
        mapping: Dict[str, str],
    ) -> List[str]:
        """Extract a list-valued claim."""
        key = self._find_claim_key(claims, target, mapping)
        value = claims.get(key, [])

        if isinstance(value, list):
            return [str(v) for v in value]
        elif isinstance(value, str):
            return [value]
        return []

    async def refresh_token(self, user: SSOUser) -> Optional[SSOUser]:
        """Refresh access token using refresh token."""
        if not user.refresh_token:
            return None

        token_endpoint = await self._get_endpoint("token_endpoint")
        if not token_endpoint:
            return None

        data = {
            "grant_type": "refresh_token",
            "client_id": self.config.client_id,
            "client_secret": self.config.client_secret,
            "refresh_token": user.refresh_token,
        }

        try:
            if HAS_HTTPX:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        token_endpoint,
                        data=data,
                        timeout=30.0
                    )
                    response.raise_for_status()
                    tokens = response.json()
            else:
                import urllib.request
                import urllib.parse
                req_data = urllib.parse.urlencode(data).encode()
                req = urllib.request.Request(
                    token_endpoint,
                    data=req_data,
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    tokens = json.loads(resp.read().decode())

            # Update user with new tokens
            return SSOUser(
                id=user.id,
                email=user.email,
                name=user.name,
                first_name=user.first_name,
                last_name=user.last_name,
                username=user.username,
                roles=user.roles,
                groups=user.groups,
                provider_type=user.provider_type,
                provider_id=user.provider_id,
                access_token=tokens.get("access_token"),
                refresh_token=tokens.get("refresh_token", user.refresh_token),
                id_token=tokens.get("id_token"),
                token_expires_at=time.time() + tokens.get("expires_in", 3600),
                raw_claims=user.raw_claims,
            )

        except Exception as e:
            logger.warning(f"Token refresh failed: {e}")
            return None

    async def logout(self, user: SSOUser) -> Optional[str]:
        """Get logout URL for IdP-initiated logout."""
        end_session = await self._get_endpoint("end_session_endpoint")
        if not end_session:
            return self.config.logout_url or None

        params = {}
        if user.id_token:
            params["id_token_hint"] = user.id_token
        if self.config.post_logout_redirect_url:
            params["post_logout_redirect_uri"] = self.config.post_logout_redirect_url

        if params:
            return f"{end_session}?{urlencode(params)}"
        return end_session


__all__ = [
    "OIDCError",
    "OIDCConfig",
    "OIDCProvider",
    "PROVIDER_CONFIGS",
    "HAS_JWT",
    "HAS_HTTPX",
]
