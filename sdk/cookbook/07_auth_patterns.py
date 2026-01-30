#!/usr/bin/env python3
"""
07_auth_patterns.py - Authentication patterns for the Aragora SDK.

This example demonstrates:
- JWT authentication with token refresh
- API key authentication
- OAuth 2.0 flows for enterprise SSO

Usage:
    python 07_auth_patterns.py --dry-run
    python 07_auth_patterns.py --method jwt
    python 07_auth_patterns.py --method oauth
"""

import argparse
import asyncio
import os
from aragora_sdk import ArenaClient, DebateConfig, Agent
from aragora_sdk.auth import (
    JWTAuth,
    APIKeyAuth,
    OAuthClient,
    AuthError,
)


# -----------------------------------------------------------------------------
# Pattern 1: JWT Authentication with automatic token refresh
# -----------------------------------------------------------------------------


async def jwt_authentication_example(dry_run: bool = False) -> dict:
    """Demonstrate JWT authentication with automatic refresh."""

    print("\n=== JWT Authentication ===")

    if dry_run:
        print("[DRY RUN] Would authenticate with JWT")
        print("[DRY RUN] Token would auto-refresh before expiry")
        return {"method": "jwt", "status": "dry_run"}

    # Configure JWT auth with refresh
    jwt_auth = JWTAuth(
        token=os.environ.get("ARAGORA_JWT_TOKEN"),
        refresh_token=os.environ.get("ARAGORA_REFRESH_TOKEN"),
        # Auto-refresh when token has < 5 minutes left
        refresh_threshold_seconds=300,
    )

    # Token refresh handler for when tokens expire
    async def on_token_refresh(new_token: str, new_refresh: str) -> None:
        print(f"Token refreshed. New token expires at: {jwt_auth.token_expiry}")
        # Optionally persist new tokens
        # await save_tokens(new_token, new_refresh)

    jwt_auth.on_refresh = on_token_refresh

    # Create client with JWT auth
    client = ArenaClient(auth=jwt_auth)

    # The client will automatically refresh tokens as needed
    result = await client.run_debate(
        DebateConfig(
            topic="JWT auth test",
            agents=[Agent(name="claude", model="claude-sonnet-4-20250514")],
            rounds=1,
        )
    )

    return {"method": "jwt", "status": "success", "result": result.to_dict()}


# -----------------------------------------------------------------------------
# Pattern 2: API Key Authentication (simpler, for server-to-server)
# -----------------------------------------------------------------------------


async def api_key_authentication_example(dry_run: bool = False) -> dict:
    """Demonstrate API key authentication."""

    print("\n=== API Key Authentication ===")

    if dry_run:
        print("[DRY RUN] Would authenticate with API key")
        print("[DRY RUN] API key sent in X-API-Key header")
        return {"method": "api_key", "status": "dry_run"}

    # Simple API key auth
    api_key_auth = APIKeyAuth(
        api_key=os.environ.get("ARAGORA_API_KEY"),
        header_name="X-API-Key",  # Default header
    )

    # Create client with API key auth
    client = ArenaClient(auth=api_key_auth)

    result = await client.run_debate(
        DebateConfig(
            topic="API key auth test",
            agents=[Agent(name="claude", model="claude-sonnet-4-20250514")],
            rounds=1,
        )
    )

    return {"method": "api_key", "status": "success", "result": result.to_dict()}


# -----------------------------------------------------------------------------
# Pattern 3: OAuth 2.0 for Enterprise SSO
# -----------------------------------------------------------------------------


async def oauth_authentication_example(dry_run: bool = False) -> dict:
    """Demonstrate OAuth 2.0 authentication flow."""

    print("\n=== OAuth 2.0 Authentication ===")

    if dry_run:
        print("[DRY RUN] Would initiate OAuth flow")
        print("[DRY RUN] Steps: authorize -> callback -> token exchange")
        return {"method": "oauth", "status": "dry_run"}

    # Configure OAuth client
    oauth = OAuthClient(
        client_id=os.environ.get("ARAGORA_OAUTH_CLIENT_ID"),
        client_secret=os.environ.get("ARAGORA_OAUTH_CLIENT_SECRET"),
        authorize_url="https://auth.example.com/oauth/authorize",
        token_url="https://auth.example.com/oauth/token",
        redirect_uri="http://localhost:8000/callback",
        scopes=["debates:read", "debates:write", "knowledge:read"],
    )

    # Step 1: Get authorization URL (user visits this in browser)
    auth_url = oauth.get_authorization_url(state="random_state_string")
    print(f"Authorization URL: {auth_url}")

    # Step 2: After user authorizes, exchange code for token
    # In a real app, this happens in your callback handler
    # code = request.query_params.get("code")
    # tokens = await oauth.exchange_code(code)

    # Step 3: Use tokens with client
    # client = ArenaClient(auth=oauth.get_auth_handler())

    return {"method": "oauth", "auth_url": auth_url, "status": "pending_authorization"}


async def run_auth_example(method: str, dry_run: bool = False) -> dict:
    """Run the specified authentication example."""

    if method == "jwt":
        return await jwt_authentication_example(dry_run)
    elif method == "api_key":
        return await api_key_authentication_example(dry_run)
    elif method == "oauth":
        return await oauth_authentication_example(dry_run)
    else:
        # Run all examples
        results = {}
        for m in ["jwt", "api_key", "oauth"]:
            try:
                results[m] = await run_auth_example(m, dry_run)
            except AuthError as e:
                results[m] = {"status": "error", "error": str(e)}
        return results


def main():
    parser = argparse.ArgumentParser(description="Authentication patterns demo")
    parser.add_argument(
        "--method",
        choices=["jwt", "api_key", "oauth", "all"],
        default="all",
        help="Auth method to demonstrate",
    )
    parser.add_argument("--dry-run", action="store_true", help="Test without API calls")
    args = parser.parse_args()

    result = asyncio.run(run_auth_example(args.method, args.dry_run))
    return result


if __name__ == "__main__":
    main()
