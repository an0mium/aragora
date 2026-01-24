"""
Teams SSO Authentication.

Provides single sign-on authentication for Microsoft Teams users via Azure AD.
Handles:
- Token exchange from Teams Bot Framework activities
- Azure AD token validation (when OBO flow is needed)
- User resolution via identity bridge

Usage:
    from aragora.auth.teams_sso import TeamsSSO

    sso = TeamsSSO()
    user = await sso.authenticate_from_activity(activity)

Environment Variables:
    AZURE_AD_CLIENT_ID: Azure AD app client ID
    AZURE_AD_CLIENT_SECRET: Azure AD app client secret
    AZURE_AD_TENANT_ID: Azure AD tenant ID (or 'common' for multi-tenant)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from aragora.auth.sso import SSOUser

logger = logging.getLogger(__name__)

# Azure AD configuration
AZURE_AD_CLIENT_ID = os.environ.get("AZURE_AD_CLIENT_ID", "")
AZURE_AD_CLIENT_SECRET = os.environ.get("AZURE_AD_CLIENT_SECRET", "")
AZURE_AD_TENANT_ID = os.environ.get("AZURE_AD_TENANT_ID", "common")

# Token validation endpoints
AZURE_AD_AUTHORITY = f"https://login.microsoftonline.com/{AZURE_AD_TENANT_ID}"
AZURE_AD_TOKEN_URL = f"{AZURE_AD_AUTHORITY}/oauth2/v2.0/token"
AZURE_AD_JWKS_URL = f"{AZURE_AD_AUTHORITY}/discovery/v2.0/keys"


@dataclass
class TeamsTokenInfo:
    """Information extracted from a Teams/Azure AD token."""

    oid: str  # Azure AD object ID
    tid: str  # Tenant ID
    upn: Optional[str] = None  # User principal name
    email: Optional[str] = None
    name: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    preferred_username: Optional[str] = None
    iss: Optional[str] = None  # Issuer
    aud: Optional[str] = None  # Audience
    exp: Optional[int] = None  # Expiration
    iat: Optional[int] = None  # Issued at
    raw_claims: Dict[str, Any] = None

    def __post_init__(self):
        if self.raw_claims is None:
            self.raw_claims = {}

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.exp is None:
            return False
        return time.time() > self.exp


class TeamsSSO:
    """
    Teams Single Sign-On handler.

    Authenticates Teams users via Azure AD, either by:
    1. Extracting user info from Bot Framework activities (implicit SSO)
    2. Validating Azure AD tokens for explicit authentication
    """

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        """Initialize Teams SSO handler.

        Args:
            client_id: Azure AD client ID (or uses env var)
            client_secret: Azure AD client secret (or uses env var)
            tenant_id: Azure AD tenant ID (or uses env var)
        """
        self.client_id = client_id or AZURE_AD_CLIENT_ID
        self.client_secret = client_secret or AZURE_AD_CLIENT_SECRET
        self.tenant_id = tenant_id or AZURE_AD_TENANT_ID
        self._identity_bridge = None
        self._jwks_cache: Optional[Any] = None  # PyJWKClient when available
        self._jwks_cache_time: float = 0

    def _get_identity_bridge(self):
        """Lazy-load the identity bridge."""
        if self._identity_bridge is None:
            from aragora.connectors.chat.teams_identity import get_teams_identity_bridge

            self._identity_bridge = get_teams_identity_bridge()
        return self._identity_bridge

    async def authenticate_from_activity(
        self,
        activity: Dict[str, Any],
        create_user_if_missing: bool = True,
    ) -> Optional["SSOUser"]:
        """
        Authenticate a Teams user from a Bot Framework activity.

        This is the primary authentication method for Teams bot interactions.
        Extracts the Azure AD object ID from the activity and resolves to an
        Aragora user via the identity bridge.

        Args:
            activity: Bot Framework activity dictionary
            create_user_if_missing: Create Aragora user if not found

        Returns:
            SSOUser if authentication successful, None otherwise
        """

        # Extract user info from activity
        from_data = activity.get("from", {})
        aad_object_id = from_data.get("aadObjectId")

        if not aad_object_id:
            logger.debug("No aadObjectId in activity, cannot authenticate")
            return None

        # Get tenant ID
        conversation = activity.get("conversation", {})
        channel_data = activity.get("channelData", {})
        tenant_id = (
            conversation.get("tenantId")
            or channel_data.get("tenant", {}).get("id")
            or self.tenant_id
        )

        if not tenant_id or tenant_id == "common":
            logger.warning("No tenant ID available for Teams SSO")
            # Still try with what we have
            tenant_id = ""

        # Try to resolve existing user
        bridge = self._get_identity_bridge()
        user = await bridge.resolve_user(aad_object_id, tenant_id)

        if user:
            logger.debug(f"Resolved Teams user: {user.id}")
            return user

        # User not found - optionally create
        if not create_user_if_missing:
            logger.debug(f"Teams user not found and creation disabled: {aad_object_id}")
            return None

        # Extract more info and sync user
        from aragora.connectors.chat.teams_identity import TeamsUserInfo

        teams_user = TeamsUserInfo(
            aad_object_id=aad_object_id,
            tenant_id=tenant_id,
            display_name=from_data.get("name"),
        )

        user = await bridge.sync_user_from_teams(teams_user, create_if_missing=True)
        if user:
            logger.info(f"Created and linked Teams user: {user.id}")
        return user

    async def validate_token(
        self,
        token: str,
    ) -> Optional[TeamsTokenInfo]:
        """
        Validate an Azure AD token.

        Used when explicit token validation is needed, such as:
        - Tab app authentication
        - On-behalf-of (OBO) flows
        - Message extension authentication

        Args:
            token: Azure AD access token or ID token

        Returns:
            TeamsTokenInfo if valid, None if invalid
        """
        try:
            import jwt

            # Decode without verification first to get header
            unverified = jwt.decode(token, options={"verify_signature": False})

            # Get signing key from JWKS
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")

            if not kid:
                logger.warning("Token missing kid in header")
                return None

            # Fetch JWKS (with caching)
            signing_key = await self._get_signing_key(kid)
            if not signing_key:
                logger.warning(f"Could not find signing key: {kid}")
                return None

            # Verify token
            audience = self.client_id or unverified.get("aud")
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                audience=audience,
            )

            return TeamsTokenInfo(
                oid=claims.get("oid", ""),
                tid=claims.get("tid", ""),
                upn=claims.get("upn"),
                email=claims.get("email") or claims.get("preferred_username"),
                name=claims.get("name"),
                given_name=claims.get("given_name"),
                family_name=claims.get("family_name"),
                preferred_username=claims.get("preferred_username"),
                iss=claims.get("iss"),
                aud=claims.get("aud"),
                exp=claims.get("exp"),
                iat=claims.get("iat"),
                raw_claims=claims,
            )

        except ImportError:
            logger.warning("PyJWT not installed, cannot validate tokens")
            return None
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            return None

    async def _get_signing_key(self, kid: str) -> Optional[Any]:
        """Get signing key from JWKS cache or fetch fresh.

        Args:
            kid: Key ID from token header

        Returns:
            Signing key if found
        """
        try:
            from jwt import PyJWKClient

            # Check cache (refresh every 24 hours)
            if self._jwks_cache and (time.time() - self._jwks_cache_time) < 86400:
                try:
                    key = self._jwks_cache.get_signing_key(kid)
                    return key.key
                except Exception:
                    pass  # Key not in cache, refetch

            # Fetch fresh JWKS
            jwks_url = f"https://login.microsoftonline.com/{self.tenant_id}/discovery/v2.0/keys"
            self._jwks_cache = PyJWKClient(jwks_url)
            self._jwks_cache_time = time.time()

            key = self._jwks_cache.get_signing_key(kid)
            return key.key

        except ImportError:
            logger.warning("PyJWT with cryptography not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to get signing key: {e}")
            return None

    async def exchange_token(
        self,
        token: str,
        scopes: Optional[list] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Exchange a Teams SSO token using on-behalf-of (OBO) flow.

        This is used when you need to call other APIs (e.g., Microsoft Graph)
        on behalf of the authenticated user.

        Args:
            token: Teams SSO token
            scopes: Scopes to request (default: User.Read)

        Returns:
            Token response dict with access_token if successful
        """
        if not self.client_id or not self.client_secret:
            logger.warning("Azure AD credentials not configured for OBO flow")
            return None

        scopes = scopes or ["User.Read"]

        try:
            import httpx

            token_url = f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/v2.0/token"

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    token_url,
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "assertion": token,
                        "scope": " ".join(scopes),
                        "requested_token_use": "on_behalf_of",
                    },
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(
                        f"OBO token exchange failed: {response.status_code} - {response.text}"
                    )
                    return None

        except ImportError:
            logger.warning("httpx not installed for OBO flow")
            return None
        except Exception as e:
            logger.error(f"OBO token exchange error: {e}")
            return None

    async def get_user_from_graph(
        self,
        access_token: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch user profile from Microsoft Graph API.

        Args:
            access_token: Access token with User.Read scope

        Returns:
            User profile from Graph API
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    "https://graph.microsoft.com/v1.0/me",
                    headers={"Authorization": f"Bearer {access_token}"},
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Graph API call failed: {response.status_code}")
                    return None

        except ImportError:
            logger.warning("httpx not installed for Graph API")
            return None
        except Exception as e:
            logger.error(f"Graph API error: {e}")
            return None


# Singleton instance
_sso: Optional[TeamsSSO] = None


def get_teams_sso() -> TeamsSSO:
    """Get or create the Teams SSO singleton."""
    global _sso
    if _sso is None:
        _sso = TeamsSSO()
    return _sso
