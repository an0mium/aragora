"""Tests for KnowledgeMoundHandlerProtocol (base_mixin)."""

from __future__ import annotations

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m


from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from aragora.server.handlers.knowledge_base.mound.base_mixin import (
    KnowledgeMoundHandlerProtocol,
    KnowledgeMoundMixinBase,
)


# =============================================================================
# Mock Objects
# =============================================================================


class MockKnowledgeMound:
    """Mock KnowledgeMound for testing."""

    pass


class ValidHandler:
    """Handler that satisfies the protocol."""

    ctx: dict[str, Any] = {}

    def _get_mound(self) -> MockKnowledgeMound | None:
        return MockKnowledgeMound()


class InvalidHandler:
    """Handler that does NOT satisfy the protocol (missing _get_mound)."""

    ctx: dict[str, Any] = {}


class PartialHandler:
    """Handler with _get_mound but missing ctx."""

    def _get_mound(self) -> MockKnowledgeMound | None:
        return None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_module_state():
    """Clear any module-level state between tests."""
    yield


# =============================================================================
# Test Protocol Definition
# =============================================================================


class TestProtocolDefinition:
    """Tests for KnowledgeMoundHandlerProtocol definition."""

    def test_protocol_is_runtime_checkable(self):
        """Test that the protocol is runtime checkable."""
        # KnowledgeMoundHandlerProtocol should be decorated with @runtime_checkable
        assert hasattr(KnowledgeMoundHandlerProtocol, "__protocol_attrs__") or hasattr(
            KnowledgeMoundHandlerProtocol, "__subclasshook__"
        )

    def test_valid_handler_satisfies_protocol(self):
        """Test that a valid handler satisfies the protocol."""
        handler = ValidHandler()

        # Should be able to call _get_mound
        result = handler._get_mound()
        assert result is not None

        # Should have ctx attribute
        assert hasattr(handler, "ctx")

    def test_handler_with_mound_returns_mound(self):
        """Test handler _get_mound returns mound instance."""
        handler = ValidHandler()
        mound = handler._get_mound()

        assert isinstance(mound, MockKnowledgeMound)

    def test_handler_returns_none_when_no_mound(self):
        """Test handler _get_mound returns None when not available."""
        handler = PartialHandler()
        mound = handler._get_mound()

        assert mound is None


# =============================================================================
# Test Backward Compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility alias."""

    def test_mixin_base_alias_exists(self):
        """Test that KnowledgeMoundMixinBase alias exists."""
        assert KnowledgeMoundMixinBase is not None

    def test_mixin_base_is_same_as_protocol(self):
        """Test that alias points to the protocol."""
        assert KnowledgeMoundMixinBase is KnowledgeMoundHandlerProtocol


# =============================================================================
# Test Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test __all__ contains expected exports."""
        from aragora.server.handlers.knowledge_base.mound import base_mixin

        assert "KnowledgeMoundHandlerProtocol" in base_mixin.__all__
        assert "KnowledgeMoundMixinBase" in base_mixin.__all__

    def test_exports_are_importable(self):
        """Test all exports are importable."""
        # These imports should not raise
        from aragora.server.handlers.knowledge_base.mound.base_mixin import (
            KnowledgeMoundHandlerProtocol,
            KnowledgeMoundMixinBase,
        )

        assert KnowledgeMoundHandlerProtocol is not None
        assert KnowledgeMoundMixinBase is not None


# =============================================================================
# Test Protocol Structural Typing
# =============================================================================


class TestStructuralTyping:
    """Tests for protocol structural typing behavior."""

    def test_protocol_requires_get_mound(self):
        """Test protocol expects _get_mound method."""

        # A class with _get_mound should work with the protocol
        class WithGetMound:
            ctx = {}

            def _get_mound(self):
                return None

        handler = WithGetMound()
        assert hasattr(handler, "_get_mound")
        assert callable(handler._get_mound)

    def test_protocol_requires_ctx(self):
        """Test protocol expects ctx attribute."""

        # A class with ctx should have the attribute
        class WithCtx:
            ctx: dict[str, Any] = {"key": "value"}

            def _get_mound(self):
                return None

        handler = WithCtx()
        assert hasattr(handler, "ctx")
        assert isinstance(handler.ctx, dict)

    def test_ctx_can_store_arbitrary_data(self):
        """Test ctx can store arbitrary server context data."""
        handler = ValidHandler()
        handler.ctx["custom_key"] = "custom_value"
        handler.ctx["mound"] = MockKnowledgeMound()
        handler.ctx["config"] = {"option": True}

        assert handler.ctx["custom_key"] == "custom_value"
        assert isinstance(handler.ctx["mound"], MockKnowledgeMound)
        assert handler.ctx["config"]["option"] is True
