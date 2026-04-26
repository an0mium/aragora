"""Feature-flag coverage for experimental Agent Bridge registry wiring."""

from __future__ import annotations

import importlib
import os
from unittest.mock import patch

import pytest

from aragora.config.settings import reset_settings


def _reload_admin_registry(*, enabled: bool):
    env = {k: v for k, v in os.environ.items() if k != "ARAGORA_FEATURE_AGENT_BRIDGE"}
    if enabled:
        env["ARAGORA_FEATURE_AGENT_BRIDGE"] = "true"
    with patch.dict(os.environ, env, clear=True):
        reset_settings()
        import aragora.server.handler_registry.admin as admin_registry

        return importlib.reload(admin_registry)


@pytest.fixture(autouse=True)
def _restore_default_registry():
    yield
    _reload_admin_registry(enabled=False)
    reset_settings()


def test_agent_bridge_handler_registry_disabled_by_default() -> None:
    admin_registry = _reload_admin_registry(enabled=False)

    registry_names = [name for name, _ in admin_registry.ADMIN_HANDLER_REGISTRY]
    assert "_agent_bridge_handler" not in registry_names
    assert admin_registry.AgentBridgeHandler is None


def test_agent_bridge_handler_registry_enabled_by_feature_flag() -> None:
    admin_registry = _reload_admin_registry(enabled=True)

    registry_names = [name for name, _ in admin_registry.ADMIN_HANDLER_REGISTRY]
    assert "_agent_bridge_handler" in registry_names
    assert admin_registry.AgentBridgeHandler is not None
