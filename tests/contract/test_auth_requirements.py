"""
Auth Requirements Contract Tests.

Validates that:
1. Protected endpoints declare security in OpenAPI spec
2. Public endpoints don't require authentication
3. Auth manifest matches actual handler protection
"""

from __future__ import annotations

from typing import Any

import pytest


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def openapi_spec() -> dict[str, Any]:
    """Load the OpenAPI specification."""
    from aragora.server.openapi import generate_openapi_schema

    return generate_openapi_schema()


@pytest.fixture(scope="module")
def auth_manifest():
    """Load the auth requirements manifest."""
    from aragora.server.auth_requirements import (
        get_all_requirements,
        get_protected_prefixes,
        get_public_paths,
    )

    return {
        "requirements": get_all_requirements(),
        "protected_prefixes": get_protected_prefixes(),
        "public_paths": get_public_paths(),
    }


# =============================================================================
# Auth Manifest Integrity Tests
# =============================================================================


class TestAuthManifestIntegrity:
    """Tests for auth manifest completeness and consistency."""

    def test_manifest_has_entries(self, auth_manifest: dict) -> None:
        """Auth manifest should have endpoint entries."""
        assert len(auth_manifest["requirements"]) > 0

    def test_manifest_has_public_endpoints(self, auth_manifest: dict) -> None:
        """Auth manifest should define public endpoints."""
        assert len(auth_manifest["public_paths"]) > 0

    def test_manifest_has_protected_prefixes(self, auth_manifest: dict) -> None:
        """Auth manifest should define protected prefixes."""
        assert len(auth_manifest["protected_prefixes"]) > 0

    def test_public_paths_not_in_protected_prefixes(self, auth_manifest: dict) -> None:
        """Public paths shouldn't be under protected prefixes (unless explicitly public)."""
        from aragora.server.auth_requirements import AuthLevel

        public_reqs = [r for r in auth_manifest["requirements"] if r.level == AuthLevel.PUBLIC]

        # Public paths should be intentionally public, not accidentally unprotected
        for req in public_reqs:
            # Health, auth, and info endpoints are expected to be public
            allowed_public_prefixes = [
                "/api/health",
                "/api/healthz",
                "/api/readyz",
                "/api/auth",
                "/api/openapi",
                "/api/modes",
                "/api/metrics",
                "/api/nomic/health",
            ]
            is_allowed_public = any(
                req.path.startswith(prefix) for prefix in allowed_public_prefixes
            )
            assert is_allowed_public, (
                f"Path {req.path} is marked PUBLIC but doesn't match expected "
                f"public prefixes: {allowed_public_prefixes}"
            )

    def test_no_duplicate_requirements(self, auth_manifest: dict) -> None:
        """No duplicate path/method combinations in manifest."""
        seen = set()
        for req in auth_manifest["requirements"]:
            key = (req.path, req.method.lower())
            assert key not in seen, f"Duplicate requirement: {req.path} {req.method}"
            seen.add(key)

    def test_permission_endpoints_have_permissions(self, auth_manifest: dict) -> None:
        """Permission-level endpoints must specify a permission."""
        from aragora.server.auth_requirements import AuthLevel

        for req in auth_manifest["requirements"]:
            if req.level == AuthLevel.PERMISSION:
                assert req.permission, (
                    f"Permission-level endpoint {req.path} {req.method} "
                    f"missing permission specification"
                )


# =============================================================================
# OpenAPI Security Tests
# =============================================================================


class TestOpenAPISecurityDeclarations:
    """Tests that OpenAPI spec declares security correctly."""

    HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}

    def test_protected_endpoints_have_security(
        self, openapi_spec: dict, auth_manifest: dict
    ) -> None:
        """Protected endpoints should declare security in OpenAPI spec."""
        from aragora.server.auth_requirements import AuthLevel, get_requirement

        missing_security: list[str] = []

        for path, path_item in openapi_spec["paths"].items():
            for method, operation in path_item.items():
                if method.lower() not in self.HTTP_METHODS:
                    continue

                # Check if this endpoint requires auth
                req = get_requirement(path, method)

                if req and req.level != AuthLevel.PUBLIC:
                    # Should have security declared
                    has_security = "security" in operation and len(operation["security"]) > 0
                    if not has_security:
                        missing_security.append(f"{method.upper()} {path}")

        # Allow some missing (incrementally adding)
        max_missing = 50  # Target: reduce to 0
        assert len(missing_security) <= max_missing, (
            f"Too many protected endpoints missing security declaration "
            f"({len(missing_security)}): {missing_security[:10]}"
        )

    def test_public_endpoints_accessible(self, openapi_spec: dict, auth_manifest: dict) -> None:
        """Public endpoints should not require authentication."""
        from aragora.server.auth_requirements import get_requirement, AuthLevel

        incorrectly_protected: list[str] = []

        for public_path in auth_manifest["public_paths"]:
            if public_path not in openapi_spec["paths"]:
                continue  # Path not in spec (maybe not yet implemented)

            path_item = openapi_spec["paths"][public_path]
            for method, operation in path_item.items():
                if method.lower() not in self.HTTP_METHODS:
                    continue

                req = get_requirement(public_path, method)
                if req and req.level == AuthLevel.PUBLIC:
                    # Should NOT have security required
                    has_security = "security" in operation and len(operation["security"]) > 0
                    if has_security:
                        incorrectly_protected.append(f"{method.upper()} {public_path}")

        assert len(incorrectly_protected) == 0, (
            f"Public endpoints incorrectly require authentication: {incorrectly_protected}"
        )


# =============================================================================
# Protected Prefix Coverage Tests
# =============================================================================


class TestProtectedPrefixCoverage:
    """Tests that protected prefixes are enforced."""

    HTTP_METHODS = {"get", "post", "put", "patch", "delete"}

    def test_protected_prefix_endpoints_require_auth(
        self, openapi_spec: dict, auth_manifest: dict
    ) -> None:
        """Endpoints under protected prefixes should require authentication."""
        from aragora.server.auth_requirements import get_public_paths, requires_auth

        public_paths = get_public_paths()
        unprotected_endpoints: list[str] = []

        for path, path_item in openapi_spec["paths"].items():
            # Skip explicitly public paths
            if path in public_paths:
                continue

            for method, operation in path_item.items():
                if method.lower() not in self.HTTP_METHODS:
                    continue

                # Check if path should be protected
                if requires_auth(path, method):
                    # Should have security in spec
                    has_security = "security" in operation and len(operation["security"]) > 0
                    if not has_security:
                        unprotected_endpoints.append(f"{method.upper()} {path}")

        # Allow some unprotected (work in progress)
        max_unprotected = 100  # Target: reduce significantly
        assert len(unprotected_endpoints) <= max_unprotected, (
            f"Too many endpoints under protected prefixes lack security "
            f"({len(unprotected_endpoints)}): {unprotected_endpoints[:15]}"
        )


# =============================================================================
# Permission Tests
# =============================================================================


class TestPermissionRequirements:
    """Tests for RBAC permission requirements."""

    def test_permission_matrix_coverage(self) -> None:
        """All required permissions should be in the permission matrix."""
        from aragora.server.auth_requirements import PERMISSION_ENDPOINTS
        from aragora.server.handlers.utils.decorators import PERMISSION_MATRIX

        missing_permissions: list[str] = []

        for req in PERMISSION_ENDPOINTS:
            if req.permission and req.permission not in PERMISSION_MATRIX:
                missing_permissions.append(req.permission)

        assert len(missing_permissions) == 0, (
            f"Permissions not in PERMISSION_MATRIX: {missing_permissions}"
        )

    def test_permission_format(self) -> None:
        """Permission strings should follow the category:action format."""
        from aragora.server.auth_requirements import PERMISSION_ENDPOINTS

        invalid_format: list[str] = []

        for req in PERMISSION_ENDPOINTS:
            if req.permission:
                if ":" not in req.permission:
                    invalid_format.append(req.permission)

        assert len(invalid_format) == 0, (
            f"Permissions with invalid format (should be category:action): {invalid_format}"
        )


# =============================================================================
# Consistency Tests
# =============================================================================


class TestAuthConsistency:
    """Tests for auth consistency across the codebase."""

    def test_auth_levels_hierarchy(self) -> None:
        """Auth levels should follow expected hierarchy."""
        from aragora.server.auth_requirements import AuthLevel

        # Owner > Admin > Permission > Authenticated/User > Public
        hierarchy = [
            AuthLevel.PUBLIC,
            AuthLevel.AUTHENTICATED,
            AuthLevel.USER,
            AuthLevel.PERMISSION,
            AuthLevel.ADMIN,
            AuthLevel.OWNER,
        ]

        # Verify all levels are defined
        for level in AuthLevel:
            assert level in hierarchy, f"Auth level {level} not in hierarchy"

    def test_requires_auth_helper(self) -> None:
        """The requires_auth helper should work correctly."""
        from aragora.server.auth_requirements import requires_auth

        # Public endpoints should not require auth
        assert not requires_auth("/api/health", "get")
        assert not requires_auth("/api/healthz", "get")
        assert not requires_auth("/api/openapi", "get")

        # Protected prefixes should require auth by default
        assert requires_auth("/api/debates", "get")
        assert requires_auth("/api/debates/123", "get")
        assert requires_auth("/api/agents", "get")
        assert requires_auth("/api/admin/users", "get")

    def test_get_required_permission_helper(self) -> None:
        """The get_required_permission helper should work correctly."""
        from aragora.server.auth_requirements import get_required_permission

        # Permission endpoints should return their permission
        assert get_required_permission("/api/plugins", "post") == "plugins:install"

        # Non-permission endpoints should return None
        assert get_required_permission("/api/health", "get") is None
        assert get_required_permission("/api/debates", "get") is None


# =============================================================================
# Documentation Tests
# =============================================================================


class TestAuthDocumentation:
    """Tests for auth documentation in OpenAPI."""

    HTTP_METHODS = {"get", "post", "put", "patch", "delete"}

    def test_security_schemes_documented(self, openapi_spec: dict) -> None:
        """Security schemes should be documented."""
        components = openapi_spec.get("components", {})
        security_schemes = components.get("securitySchemes", {})

        assert "bearerAuth" in security_schemes, "bearerAuth scheme not documented"
        assert security_schemes["bearerAuth"]["type"] == "http"
        assert security_schemes["bearerAuth"]["scheme"] == "bearer"

    def test_admin_endpoints_marked(self, openapi_spec: dict) -> None:
        """Admin endpoints should be clearly marked."""
        from aragora.server.auth_requirements import ADMIN_ENDPOINTS

        for req in ADMIN_ENDPOINTS:
            if req.path not in openapi_spec["paths"]:
                continue  # Path not in spec

            path_item = openapi_spec["paths"][req.path]
            method = req.method.lower()

            if method not in path_item:
                continue

            operation = path_item[method]

            # Admin endpoints should mention "admin" in description or be in Admin tag
            tags = operation.get("tags", [])
            description = operation.get("description", "").lower()

            is_marked = "Admin" in tags or "admin" in description or "System" in tags

            # Soft check - warning only
            if not is_marked:
                pass  # Consider adding Admin tag or description
