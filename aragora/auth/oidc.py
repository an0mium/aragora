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
from types import ModuleType
from typing import Any
from urllib.parse import urlencode, urljoin

from .sso import (
    SSOAuthenticationError,
    SSOConfig,
    SSOConfigurationError,
    SSOError,
    SSOProvider,
    SSOProviderType,
    SSOUser,
)

logger = logging.getLogger(__name__)

# PyJWT for token validation (always available)
import jwt
from jwt import PyJWKClient

HAS_JWT = True

# Optional: httpx for async HTTP
httpx: ModuleType | None = None
try:
    import httpx as _httpx

    httpx = _httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


def _is_production_mode() -> bool:
    """Check if running in production mode.

    SECURITY: Defaults to production mode (secure by default).
    Set ARAGORA_ENV=development to enable dev mode behaviors.
    """
    import os

    env = os.environ.get("ARAGORA_ENV", "production").lower()
    return env not in ("development", "dev", "local", "test")


def _allow_dev_auth_fallback() -> bool:
    """Check if dev auth fallback is explicitly allowed.

    SECURITY: Even in dev mode, auth fallback must be explicitly enabled.
    This prevents accidental exposure in misconfigured environments.
    """
    import os

    # Must be in dev mode AND have explicit fallback enabled
    if _is_production_mode():
        return False
    return os.environ.get("ARAGORA_ALLOW_DEV_AUTH_FALLBACK", "").lower() in ("1", "true", "yes")


def validate_oidc_security_settings() -> None:
    """Validate OIDC security settings at startup.

    SECURITY: This function should be called during OIDCProvider initialization
    to ensure dangerous configurations are rejected early.

    Raises:
        SSOConfigurationError: If ARAGORA_ENV=production and
            ARAGORA_ALLOW_DEV_AUTH_FALLBACK is set (any value).
    """
    import os

    env = os.environ.get("ARAGORA_ENV", "production").lower()
    fallback_setting = os.environ.get("ARAGORA_ALLOW_DEV_AUTH_FALLBACK")

    # SECURITY: Reject any combination of production mode with fallback enabled
    if env == "production" and fallback_setting is not None:
        logger.critical(
            "SECURITY VIOLATION: ARAGORA_ALLOW_DEV_AUTH_FALLBACK is set in production mode. "
            "This setting must not be present in production environments. "
            "Remove the ARAGORA_ALLOW_DEV_AUTH_FALLBACK environment variable."
        )
        raise SSOConfigurationError(
            "SECURITY VIOLATION: ARAGORA_ALLOW_DEV_AUTH_FALLBACK cannot be set when "
            "ARAGORA_ENV=production. This combination bypasses ID token validation security. "
            "Remove the ARAGORA_ALLOW_DEV_AUTH_FALLBACK environment variable.",
            {
                "error_code": "INSECURE_AUTH_FALLBACK_IN_PRODUCTION",
                "aragora_env": env,
                "fallback_setting_present": True,
            },
        )

    # Warn in non-production if fallback is enabled
    if not _is_production_mode() and fallback_setting is not None:
        fallback_enabled = fallback_setting.lower() in ("1", "true", "yes")
        if fallback_enabled:
            logger.warning(
                "SECURITY WARNING: ARAGORA_ALLOW_DEV_AUTH_FALLBACK is enabled. ID token validation failures will fall back to userinfo endpoint. This is INSECURE and should NEVER be used in production. Current ARAGORA_ENV=%s", env
            )


class OIDCError(SSOError):
    """OIDC-specific error."""

    def __init__(self, message: str, details: dict | None = None):
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
    scopes: list[str] = field(default_factory=lambda: ["openid", "email", "profile"])

    # PKCE (Proof Key for Code Exchange)
    use_pkce: bool = True

    # Token validation
    validate_tokens: bool = True
    allowed_audiences: list[str] = field(default_factory=list)
    # JWT algorithms allowed for ID token validation - defaults to RS256 only
    # Allowing multiple algorithms can enable algorithm confusion attacks
    allowed_algorithms: list[str] = field(default_factory=lambda: ["RS256"])

    # Claim mapping (OIDC claim -> user field)
    claim_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "sub": "id",
            "email": "email",
            "name": "name",
            "given_name": "first_name",
            "family_name": "last_name",
            "preferred_username": "username",
            "groups": "groups",
            "roles": "roles",
        }
    )

    # Azure AD specific
    tenant_id: str = ""

    # Google specific (hosted domain)
    hd: str = ""

    def __post_init__(self) -> None:
        if not self.provider_type:
            self.provider_type = SSOProviderType.OIDC

    @classmethod
    def for_azure_ad(
        cls,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        callback_url: str,
        **kwargs: Any,
    ) -> OIDCConfig:
        """Create configuration for Azure AD / Microsoft Entra ID."""
        return cls(
            provider_type=SSOProviderType.AZURE_AD,
            entity_id=client_id,
            client_id=client_id,
            client_secret=client_secret,
            callback_url=callback_url,
            issuer_url=f"https://login.microsoftonline.com/{tenant_id}/v2.0",
            tenant_id=tenant_id,
            scopes=["openid", "email", "profile", "User.Read"],
            **kwargs,
        )

    @classmethod
    def for_okta(
        cls,
        org_url: str,
        client_id: str,
        client_secret: str,
        callback_url: str,
        **kwargs: Any,
    ) -> OIDCConfig:
        """Create configuration for Okta."""
        # Normalize org_url (remove trailing slash)
        org_url = org_url.rstrip("/")
        return cls(
            provider_type=SSOProviderType.OKTA,
            entity_id=client_id,
            client_id=client_id,
            client_secret=client_secret,
            callback_url=callback_url,
            issuer_url=org_url,
            scopes=["openid", "email", "profile", "groups"],
            **kwargs,
        )

    @classmethod
    def for_google(
        cls,
        client_id: str,
        client_secret: str,
        callback_url: str,
        hd: str = "",
        **kwargs: Any,
    ) -> OIDCConfig:
        """Create configuration for Google Workspace.

        Args:
            hd: Hosted domain restriction (e.g., "example.com")
        """
        return cls(
            provider_type=SSOProviderType.GOOGLE,
            entity_id=client_id,
            client_id=client_id,
            client_secret=client_secret,
            callback_url=callback_url,
            issuer_url="https://accounts.google.com",
            hd=hd,
            scopes=["openid", "email", "profile"],
            **kwargs,
        )

    def validate(self) -> list[str]:
        """Validate OIDC configuration."""
        errors = super().validate()

        if not self.client_id:
            errors.append("client_id is required")

        if not self.client_secret:
            errors.append("client_secret is required")

        if not self.issuer_url:
            if not self.authorization_endpoint or not self.token_endpoint:
                errors.append("issuer_url or explicit endpoints are required")

        # Validate allowed algorithms - reject insecure symmetric algorithms
        # HMAC algorithms (HS*) are insecure for OIDC because the secret is shared
        # The "none" algorithm means no signature at all
        insecure_algorithms = {"HS256", "HS384", "HS512", "none"}
        for alg in self.allowed_algorithms:
            if alg in insecure_algorithms:
                errors.append(
                    f"Algorithm '{alg}' is insecure for OIDC ID token validation. "
                    "Use asymmetric algorithms like RS256 or ES256."
                )

        return errors


# Well-known provider configurations
PROVIDER_CONFIGS: dict[str, dict[str, str]] = {
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
                f"Invalid OIDC configuration: {', '.join(errors)}", {"errors": errors}
            )

        # SECURITY: Validate environment security settings at startup
        # This catches dangerous configurations like fallback enabled in production
        validate_oidc_security_settings()

        # SECURITY: Require JWT library in production for proper token validation
        if not HAS_JWT and _is_production_mode():
            raise SSOConfigurationError(
                "PyJWT library required for OIDC in production. Install with: pip install PyJWT",
                {"missing_dependency": "PyJWT"},
            )

        # PKCE state (code_verifier stored by state) - bounded OrderedDict
        from collections import OrderedDict

        self._pkce_store: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._pkce_store_max_size: int = 10_000
        self._pkce_entry_ttl: float = 600.0  # 10 minutes

        # Nonce store (nonce stored by state for ID token validation) - bounded OrderedDict
        self._nonce_store: OrderedDict[str, tuple[str, float]] = OrderedDict()

        # Discovery cache
        self._discovery_cache: dict[str, Any] | None = None
        self._discovery_cached_at: float = 0

        # JWKS client
        self._jwks_client: Any | None = None

    @property
    def provider_type(self) -> SSOProviderType:
        return self.config.provider_type

    async def _discover_endpoints(self) -> dict[str, Any]:
        """Fetch OIDC discovery document."""
        # Check cache (1 hour TTL)
        if self._discovery_cache and time.time() - self._discovery_cached_at < 3600:
            return self._discovery_cache

        if not self.config.issuer_url:
            return {}

        discovery_url = urljoin(
            self.config.issuer_url.rstrip("/") + "/", ".well-known/openid-configuration"
        )

        try:
            if HAS_HTTPX:
                if httpx is None:
                    raise RuntimeError("httpx unavailable despite HAS_HTTPX")
                from aragora.server.http_client_pool import get_http_pool

                pool = get_http_pool()
                async with pool.get_session("oidc") as client:
                    response = await client.get(discovery_url, timeout=10.0)
                    response.raise_for_status()
                    self._discovery_cache = response.json()
            else:
                raise SSOConfigurationError(
                    "httpx library required for OIDC. Install with: pip install httpx"
                )

            self._discovery_cached_at = time.time()
            logger.debug("OIDC discovery successful for %s", self.config.issuer_url)
            return self._discovery_cache

        except json.JSONDecodeError as e:
            logger.warning("OIDC discovery failed - invalid JSON: %s", e)
            return {}
        except (OSError, TimeoutError) as e:
            logger.warning("OIDC discovery failed - network error: %s", e)
            return {}
        except RuntimeError as e:
            logger.warning("OIDC discovery failed - runtime error: %s", e)
            return {}

    async def _get_endpoint(self, name: str) -> str:
        """Get endpoint URL, preferring config over discovery."""
        # Check config first
        config_value = getattr(self.config, name, "")
        if config_value:
            return config_value

        # Try discovery
        discovery = await self._discover_endpoints()
        result: str = discovery.get(name, "")
        return result

    def _evict_bounded_store(self, store: dict) -> None:
        """Evict expired and overflow entries from a bounded OrderedDict store.

        Each value is expected to be a tuple of (data, timestamp).
        Entries older than ``_pkce_entry_ttl`` are removed first.
        If the store still exceeds ``_pkce_store_max_size``, the oldest
        entries are dropped until the size is within limits.
        """
        now = time.time()
        # Remove expired entries
        expired_keys = [k for k, v in store.items() if now - v[1] > self._pkce_entry_ttl]
        for k in expired_keys:
            del store[k]
        # Enforce max size by evicting oldest entries
        while len(store) >= self._pkce_store_max_size:
            store.popitem(last=False)

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
        state: str | None = None,
        redirect_uri: str | None = None,
        scopes: list[str] | None = None,
        nonce: str | None = None,
        **kwargs: Any,
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

        # Store nonce keyed by state for later validation
        self._evict_bounded_store(self._nonce_store)
        self._nonce_store[state] = (nonce, time.time())

        # PKCE
        if self.config.use_pkce:
            code_verifier, code_challenge = self._generate_pkce()
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"
            self._evict_bounded_store(self._pkce_store)
            self._pkce_store[state] = (code_verifier, time.time())

        # Provider-specific parameters
        if self.config.provider_type == SSOProviderType.AZURE_AD:
            params["response_mode"] = "query"

        # Add any extra parameters
        params.update(kwargs)

        return f"{auth_endpoint}?{urlencode(params)}"

    async def authenticate(
        self,
        code: str | None = None,
        saml_response: str | None = None,
        state: str | None = None,
        **kwargs: Any,
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

        # SECURITY: State parameter is REQUIRED to prevent CSRF attacks.
        # An attacker could trick a victim into completing an OAuth flow
        # initiated by the attacker if state validation is skipped.
        if not state:
            raise SSOAuthenticationError(
                "Missing state parameter - state is required to prevent CSRF attacks",
                {"code": "MISSING_STATE"},
            )

        # Validate state
        if not self.validate_state(state):
            raise SSOAuthenticationError(
                "Invalid or expired state parameter", {"code": "INVALID_STATE"}
            )

        # Get PKCE code verifier
        code_verifier = None
        if self.config.use_pkce and state:
            pkce_entry = self._pkce_store.pop(state, None)
            if pkce_entry is not None:
                code_verifier = pkce_entry[0]  # (verifier, timestamp)

        # Retrieve stored nonce for this state (for ID token validation)
        expected_nonce: str | None = None
        nonce_entry = self._nonce_store.pop(state, None) if state else None
        if nonce_entry is not None:
            expected_nonce = nonce_entry[0]  # (nonce, timestamp)

        # Exchange code for tokens
        tokens = await self._exchange_code(code, code_verifier)

        # Get user info (pass expected nonce for ID token validation)
        user = await self._get_user_info(tokens, expected_nonce=expected_nonce)

        # Check domain restriction
        if not self.is_domain_allowed(user.email):
            raise SSOAuthenticationError(
                f"Email domain not allowed: {user.email.split('@')[-1]}",
                {"code": "DOMAIN_NOT_ALLOWED"},
            )

        logger.info("OIDC authentication successful for user_id=%s", user.id)
        return user

    async def _exchange_code(
        self,
        code: str,
        code_verifier: str | None,
    ) -> dict[str, Any]:
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
                if httpx is None:
                    raise RuntimeError("httpx unavailable despite HAS_HTTPX")
                from aragora.server.http_client_pool import get_http_pool

                pool = get_http_pool()
                async with pool.get_session("oidc") as client:
                    response = await client.post(
                        token_endpoint, data=data, headers=headers, timeout=30.0
                    )
                    response.raise_for_status()
                    result: dict[str, Any] = response.json()
                    return result
            else:
                raise SSOConfigurationError(
                    "httpx library required for OIDC. Install with: pip install httpx"
                )

        except json.JSONDecodeError as e:
            logger.error("Token exchange failed - invalid JSON response: %s", e)
            raise SSOAuthenticationError("Token exchange failed - invalid response from provider")
        except (OSError, TimeoutError) as e:
            logger.error("Token exchange failed - network error: %s", e)
            raise SSOAuthenticationError("Token exchange failed - network error")
        except ValueError as e:
            logger.error("Token exchange failed - invalid data: %s", e)
            raise SSOAuthenticationError("Token exchange failed")

    async def _get_user_info(
        self, tokens: dict[str, Any], expected_nonce: str | None = None
    ) -> SSOUser:
        """Get user info from tokens or userinfo endpoint."""
        access_token = tokens.get("access_token")
        id_token = tokens.get("id_token")

        claims: dict[str, Any] = {}

        # Parse ID token if available
        if id_token and HAS_JWT:
            try:
                # Validate and decode ID token
                claims = await self._validate_id_token(id_token)

                # SECURITY: Validate nonce claim to prevent ID token replay attacks.
                # The nonce was generated during get_authorization_url and stored
                # keyed by state. The IdP must echo it back in the ID token.
                if expected_nonce is not None:
                    token_nonce = claims.get("nonce")
                    if not token_nonce:
                        raise SSOAuthenticationError(
                            "ID token missing required nonce claim",
                            {"code": "MISSING_NONCE"},
                        )
                    if not secrets.compare_digest(str(token_nonce), expected_nonce):
                        raise SSOAuthenticationError(
                            "ID token nonce does not match expected value - "
                            "possible token replay attack",
                            {"code": "NONCE_MISMATCH"},
                        )
            except (
                ValueError,
                KeyError,
                TypeError,
                jwt.exceptions.InvalidSignatureError,
                jwt.exceptions.DecodeError,
                jwt.exceptions.InvalidTokenError,
                jwt.exceptions.ExpiredSignatureError,
                jwt.exceptions.InvalidAudienceError,
                jwt.exceptions.InvalidIssuerError,
                jwt.exceptions.InvalidAlgorithmError,
                jwt.exceptions.InvalidKeyError,
                jwt.exceptions.MissingRequiredClaimError,
                jwt.exceptions.PyJWKClientError,
                jwt.exceptions.PyJWKError,
                jwt.exceptions.PyJWTError,
            ) as e:
                # Token parsing/validation errors - catch all JWT exceptions for defense in depth
                # SECURITY: Never silently fall back to userinfo - could allow signature bypass
                #
                # DEFENSE IN DEPTH: Multiple explicit checks to prevent fallback in production:
                # 1. _is_production_mode() - returns True if ARAGORA_ENV is not dev/test/local
                # 2. _allow_dev_auth_fallback() - returns False in production (double-check)
                # 3. Explicit production check below as final safeguard
                is_production = _is_production_mode()
                fallback_allowed = _allow_dev_auth_fallback()

                # SECURITY: Explicit production guard - NEVER allow fallback in production
                if is_production:
                    logger.error(
                        "ID token validation failed in PRODUCTION mode: %s. "
                        "Fallback is BLOCKED for security.", e
                    )
                    raise SSOAuthenticationError(
                        "ID token validation failed. "
                        "Token validation is required in production mode."
                    )

                # SECURITY: Even in non-production, require explicit opt-in
                if not fallback_allowed:
                    logger.error("ID token validation failed: %s", e)
                    raise SSOAuthenticationError(
                        "ID token validation failed. "
                        "Set ARAGORA_ENV=development and ARAGORA_ALLOW_DEV_AUTH_FALLBACK=1 "
                        "to allow fallback to userinfo endpoint (NOT recommended for production)."
                    )

                # At this point: NOT production AND fallback explicitly allowed
                # SECURITY: Even in dev fallback, validate token structure and expiry
                # to reject completely forged or expired tokens
                if id_token:
                    try:
                        fallback_claims = jwt.decode(
                            id_token,
                            options={
                                "verify_signature": False,
                                "verify_exp": True,
                            },
                            algorithms=self.config.allowed_algorithms,
                        )
                        # Require minimal identity claims
                        if not fallback_claims.get("sub") and not fallback_claims.get("email"):
                            raise SSOAuthenticationError(
                                "ID token missing required identity claims (sub or email) "
                                "even in dev fallback mode",
                                {"code": "MISSING_IDENTITY_CLAIMS"},
                            )
                        # Use decoded claims as baseline
                        claims = fallback_claims
                    except jwt.exceptions.ExpiredSignatureError:
                        raise SSOAuthenticationError(
                            "ID token is expired — rejected even in dev fallback mode",
                            {"code": "TOKEN_EXPIRED"},
                        )
                    except jwt.exceptions.DecodeError:
                        raise SSOAuthenticationError(
                            "ID token has invalid structure — rejected even in dev fallback mode",
                            {"code": "INVALID_TOKEN_STRUCTURE"},
                        )

                logger.warning(
                    "ID token validation failed, using userinfo fallback (dev mode): %s. This is INSECURE - do not use in production!", e
                )
                # Emit security audit event for the fallback if audit system is available
                try:
                    from aragora.server.middleware.audit_logger import audit_security_event

                    audit_security_event(
                        event_type="oidc_token_validation_fallback",
                        actor="system",
                        details={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "issuer": self.config.issuer_url,
                            "fallback": "userinfo_endpoint",
                            "aragora_env": "development",
                        },
                    )
                except ImportError:
                    pass  # Audit system not available - warning log above is sufficient

        # Fetch from userinfo endpoint if needed
        if not claims.get("email"):
            userinfo = await self._fetch_userinfo(access_token)
            claims.update(userinfo)

        # Map claims to user
        return self._claims_to_user(claims, tokens)

    async def _validate_id_token(self, id_token: str) -> dict[str, Any]:
        """Validate and decode ID token using JWKS."""
        if not HAS_JWT:
            raise SSOError("PyJWT required for ID token validation")

        jwks_uri = await self._get_endpoint("jwks_uri")
        if not jwks_uri:
            # SECURITY: Fail closed - never accept unverified tokens
            logger.error("JWKS URI not available - cannot verify ID token signature")
            raise SSOAuthenticationError(
                "ID token validation failed: JWKS URI not configured. "
                "Configure issuer_url or jwks_uri for secure token validation."
            )

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
            algorithms=self.config.allowed_algorithms,
            audience=audiences,
            issuer=self.config.issuer_url,
        )

    async def _fetch_userinfo(self, access_token: str) -> dict[str, Any]:
        """Fetch user info from userinfo endpoint."""
        userinfo_endpoint = await self._get_endpoint("userinfo_endpoint")
        if not userinfo_endpoint:
            return {}

        headers = {"Authorization": f"Bearer {access_token}"}

        try:
            if HAS_HTTPX:
                if httpx is None:
                    raise RuntimeError("httpx unavailable despite HAS_HTTPX")
                from aragora.server.http_client_pool import get_http_pool

                pool = get_http_pool()
                async with pool.get_session("oidc") as client:
                    response = await client.get(userinfo_endpoint, headers=headers, timeout=10.0)
                    response.raise_for_status()
                    return response.json()
            else:
                raise SSOConfigurationError(
                    "httpx library required for OIDC. Install with: pip install httpx"
                )

        except json.JSONDecodeError as e:
            logger.warning("Userinfo fetch failed - invalid JSON: %s", e)
            return {}
        except (OSError, TimeoutError) as e:
            logger.warning("Userinfo fetch failed - network error: %s", e)
            return {}

    def _claims_to_user(
        self,
        claims: dict[str, Any],
        tokens: dict[str, Any],
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
        claims: dict[str, Any],
        target: str,
        mapping: dict[str, str],
    ) -> str:
        """Find the claim key that maps to target field."""
        for claim_key, field_name in mapping.items():
            if field_name == target and claim_key in claims:
                return claim_key
        return target  # Fallback to direct lookup

    def _extract_list_claim(
        self,
        claims: dict[str, Any],
        target: str,
        mapping: dict[str, str],
    ) -> list[str]:
        """Extract a list-valued claim."""
        key = self._find_claim_key(claims, target, mapping)
        value = claims.get(key, [])

        if isinstance(value, list):
            return [str(v) for v in value]
        elif isinstance(value, str):
            return [value]
        return []

    async def refresh_token(self, user: SSOUser) -> SSOUser | None:
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
                if httpx is None:
                    raise RuntimeError("httpx unavailable despite HAS_HTTPX")
                from aragora.server.http_client_pool import get_http_pool

                pool = get_http_pool()
                async with pool.get_session("oidc") as client:
                    response = await client.post(token_endpoint, data=data, timeout=30.0)
                    response.raise_for_status()
                    tokens = response.json()
            else:
                raise SSOConfigurationError(
                    "httpx library required for OIDC. Install with: pip install httpx"
                )

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

        except json.JSONDecodeError as e:
            logger.warning("Token refresh failed - invalid JSON: %s", e)
            return None
        except (OSError, TimeoutError) as e:
            logger.warning("Token refresh failed - network error: %s", e)
            return None
        except (KeyError, TypeError) as e:
            logger.warning("Token refresh failed - invalid response: %s", e)
            return None

    async def logout(self, user: SSOUser) -> str | None:
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
    "validate_oidc_security_settings",
    "_is_production_mode",
    "_allow_dev_auth_fallback",
]
