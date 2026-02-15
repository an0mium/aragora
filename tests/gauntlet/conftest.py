"""Shared fixtures for gauntlet tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _bypass_rbac_for_gauntlet_tests(monkeypatch):
    """Bypass RBAC permission checks for gauntlet handler tests."""
    try:
        from aragora.rbac import decorators
        from aragora.rbac.models import AuthorizationContext

        mock_auth_ctx = AuthorizationContext(
            user_id="test-user-001",
            org_id="test-org-001",
            roles={"admin", "owner"},
            permissions={"*"},
        )

        original_get_context = decorators._get_context_from_args

        def patched_get_context(args, kwargs, context_param):
            result = original_get_context(args, kwargs, context_param)
            if result is None:
                return mock_auth_ctx
            return result

        monkeypatch.setattr(decorators, "_get_context_from_args", patched_get_context)
    except (ImportError, AttributeError):
        pass
