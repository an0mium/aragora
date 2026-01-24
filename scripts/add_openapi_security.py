#!/usr/bin/env python3
"""
Add security definitions to OpenAPI spec endpoints.

Adds bearerAuth security to endpoints that require authentication.
"""

import json
import sys
from pathlib import Path

# Endpoints that should be public (no auth required)
PUBLIC_PATHS = [
    "/api/health",
    "/api/ready",
    "/api/metrics",
    "/api/openapi",
    "/api/slack/oauth",
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/forgot-password",
    "/api/v1/auth/reset-password",
    "/api/v1/auth/verify-email",
    "/api/integrations/slack/callback",
    "/api/integrations/slack/install",
    "/api/webhooks/",  # Webhooks typically use their own auth
]

# Paths that are explicitly public
PUBLIC_PATH_PREFIXES = [
    "/api/health",
    "/api/ready",
    "/api/metrics",
    "/api/docs",
    "/api/openapi",
]


def is_public_endpoint(path: str, method: str) -> bool:
    """Determine if an endpoint should be public."""
    path_lower = path.lower()

    # Check exact matches
    for public_path in PUBLIC_PATHS:
        if path_lower == public_path.lower():
            return True

    # Check prefixes
    for prefix in PUBLIC_PATH_PREFIXES:
        if path_lower.startswith(prefix.lower()):
            return True

    # OAuth callbacks are public
    if "callback" in path_lower and "oauth" in path_lower:
        return True

    # Webhooks typically use their own auth
    if "/webhooks/" in path_lower:
        return True

    return False


def add_security_definitions(spec: dict) -> tuple[dict, int, int, int]:
    """Add security definitions to endpoints missing them.

    Returns: (updated_spec, added_count, existing_count, public_count)
    """
    added = 0
    existing = 0
    public = 0

    # Ensure security scheme exists
    if "components" not in spec:
        spec["components"] = {}
    if "securitySchemes" not in spec["components"]:
        spec["components"]["securitySchemes"] = {}

    if "bearerAuth" not in spec["components"]["securitySchemes"]:
        spec["components"]["securitySchemes"]["bearerAuth"] = {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Bearer token authentication. Include token in Authorization header.",
        }

    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if not isinstance(details, dict):
                continue
            if method.lower() not in ("get", "post", "put", "patch", "delete", "head", "options"):
                continue

            if details.get("security") is not None:
                existing += 1
            elif is_public_endpoint(path, method):
                # Mark as explicitly public
                details["security"] = []  # Empty array = no auth required
                public += 1
            else:
                # Add bearer auth requirement
                details["security"] = [{"bearerAuth": []}]
                added += 1

    return spec, added, existing, public


def main():
    """Main entry point."""
    spec_path = Path("docs/api/openapi.json")

    if not spec_path.exists():
        print(f"Error: {spec_path} not found")
        sys.exit(1)

    print(f"Reading {spec_path}...")
    with open(spec_path, "r") as f:
        spec = json.load(f)

    print("Adding security definitions...")
    spec, added, existing, public = add_security_definitions(spec)

    print(f"Writing {spec_path}...")
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)

    print("\nResults:")
    print(f"  - Already had security: {existing}")
    print(f"  - Added bearerAuth: {added}")
    print(f"  - Marked as public: {public}")
    print(f"  - Total endpoints: {added + existing + public}")


if __name__ == "__main__":
    main()
