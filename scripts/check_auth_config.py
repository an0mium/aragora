#!/usr/bin/env python3
"""
Check authentication configuration for Aragora.

Run this script to verify your OAuth and JWT configuration is correct:
    python scripts/check_auth_config.py

Exit codes:
    0 - All required config is present
    1 - Missing required configuration
"""

import os
import sys


def check_config():
    """Check authentication configuration and report issues."""
    issues = []
    warnings = []

    # Check JWT Secret
    jwt_secret = os.environ.get("ARAGORA_JWT_SECRET", "")
    if not jwt_secret:
        issues.append(
            "ARAGORA_JWT_SECRET is not set. Generate one with:\n"
            '    python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    elif len(jwt_secret) < 32:
        issues.append(
            f"ARAGORA_JWT_SECRET is too short ({len(jwt_secret)} chars). "
            "Minimum 32 characters required."
        )

    # Check OAuth configuration
    oauth_vars = {
        "GOOGLE_OAUTH_CLIENT_ID": "Google OAuth client ID",
        "GOOGLE_OAUTH_CLIENT_SECRET": "Google OAuth client secret",
        "GOOGLE_OAUTH_REDIRECT_URI": "Google OAuth redirect URI",
        "OAUTH_SUCCESS_URL": "OAuth success redirect URL",
        "OAUTH_ERROR_URL": "OAuth error redirect URL",
    }

    missing_oauth = []
    for var, desc in oauth_vars.items():
        if not os.environ.get(var):
            missing_oauth.append(f"  - {var}: {desc}")

    if missing_oauth:
        warnings.append(
            "OAuth not fully configured (optional unless using social login):\n"
            + "\n".join(missing_oauth)
        )

    # Check environment mode
    env_mode = os.environ.get("ARAGORA_ENVIRONMENT", "development")
    if env_mode == "production":
        print("Environment: PRODUCTION")
        if not jwt_secret:
            issues.append("ARAGORA_JWT_SECRET is REQUIRED in production mode!")
    else:
        print(f"Environment: {env_mode}")

    # Report results
    print()

    if issues:
        print("ERRORS (must fix):")
        for issue in issues:
            print(f"  [X] {issue}")
        print()

    if warnings:
        print("WARNINGS (optional):")
        for warning in warnings:
            print(f"  [!] {warning}")
        print()

    if not issues and not warnings:
        print("All authentication configuration looks good!")
    elif not issues:
        print("Core authentication is configured. Warnings above are optional.")

    # Summary
    print()
    print("Configuration Summary:")
    print(
        f"  JWT Secret: {'Set (' + str(len(jwt_secret)) + ' chars)' if jwt_secret else 'NOT SET'}"
    )
    print(f"  OAuth: {'Configured' if not missing_oauth else 'Partial/Missing'}")

    return 0 if not issues else 1


if __name__ == "__main__":
    # Load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    sys.exit(check_config())
