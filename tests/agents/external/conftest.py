"""Fixtures for external agent handler tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def mock_auth_for_external_agent_tests(request, monkeypatch):
    """Bypass RBAC authentication for external agent handler unit tests.

    This autouse fixture ensures @require_permission decorated methods
    receive an AuthorizationContext automatically.
    """
    if "no_auto_auth" in [m.name for m in request.node.iter_markers()]:
        yield
        return

    try:
        from aragora.rbac.models import AuthorizationContext
        from aragora.rbac import decorators

        mock_auth_ctx = AuthorizationContext(
            user_id="test-user-001",
            user_email="test@example.com",
            org_id="test-org-001",
            roles={"admin", "owner"},
            permissions={"*"},
        )

        original_get_context = decorators._get_context_from_args

        def patched_get_context_from_args(args, kwargs, context_param):
            """Return mock context with full permissions.

            Always return our mock context with full permissions to bypass
            RBAC checks in unit tests.
            """
            # Always return the mock context with full permissions
            return mock_auth_ctx

        monkeypatch.setattr(decorators, "_get_context_from_args", patched_get_context_from_args)
    except (ImportError, AttributeError):
        pass

    yield
