"""
Tests for Phase S security hardening: production-default ARAGORA_ENV.

Validates that:
1. Connector/handler modules default ARAGORA_ENV to "production" (not "development")
2. JWT verification uses production-safe defaults
3. Server startup logs appropriate warnings when ARAGORA_ENV is unset
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Webhook security helper tests (aragora.connectors.chat.webhook_security)
# ---------------------------------------------------------------------------


class TestGetEnvironment:
    """Tests for get_environment() production default."""

    @patch.dict(os.environ, {}, clear=True)
    def test_default_is_production_when_unset(self):
        """ARAGORA_ENV unset should default to 'production'."""
        from aragora.connectors.chat.webhook_security import get_environment

        assert get_environment() == "production"

    @patch.dict(os.environ, {"ARAGORA_ENV": "development"})
    def test_returns_development_when_set(self):
        """ARAGORA_ENV=development should be returned as-is."""
        from aragora.connectors.chat.webhook_security import get_environment

        assert get_environment() == "development"

    @patch.dict(os.environ, {"ARAGORA_ENV": "staging"})
    def test_returns_staging_when_set(self):
        """ARAGORA_ENV=staging should be returned as-is."""
        from aragora.connectors.chat.webhook_security import get_environment

        assert get_environment() == "staging"

    @patch.dict(os.environ, {"ARAGORA_ENV": "Production"})
    def test_lowercases_env_value(self):
        """get_environment() should lowercase the value."""
        from aragora.connectors.chat.webhook_security import get_environment

        assert get_environment() == "production"

    @patch.dict(os.environ, {"ARAGORA_ENV": "PRODUCTION"})
    def test_lowercases_uppercase_value(self):
        """get_environment() should lowercase fully uppercase value."""
        from aragora.connectors.chat.webhook_security import get_environment

        assert get_environment() == "production"


class TestIsProductionEnvironment:
    """Tests for is_production_environment()."""

    @patch.dict(os.environ, {}, clear=True)
    def test_true_when_env_unset(self):
        """Default (production) should be treated as production."""
        from aragora.connectors.chat.webhook_security import is_production_environment

        assert is_production_environment() is True

    @patch.dict(os.environ, {"ARAGORA_ENV": "production"})
    def test_true_for_production(self):
        from aragora.connectors.chat.webhook_security import is_production_environment

        assert is_production_environment() is True

    @patch.dict(os.environ, {"ARAGORA_ENV": "prod"})
    def test_true_for_prod(self):
        from aragora.connectors.chat.webhook_security import is_production_environment

        assert is_production_environment() is True

    @patch.dict(os.environ, {"ARAGORA_ENV": "staging"})
    def test_true_for_staging(self):
        from aragora.connectors.chat.webhook_security import is_production_environment

        assert is_production_environment() is True

    @patch.dict(os.environ, {"ARAGORA_ENV": "stage"})
    def test_true_for_stage(self):
        from aragora.connectors.chat.webhook_security import is_production_environment

        assert is_production_environment() is True

    @patch.dict(os.environ, {"ARAGORA_ENV": "development"})
    def test_false_for_development(self):
        from aragora.connectors.chat.webhook_security import is_production_environment

        assert is_production_environment() is False

    @patch.dict(os.environ, {"ARAGORA_ENV": "test"})
    def test_false_for_test(self):
        from aragora.connectors.chat.webhook_security import is_production_environment

        assert is_production_environment() is False


class TestIsWebhookVerificationRequired:
    """Tests for is_webhook_verification_required()."""

    @patch.dict(os.environ, {}, clear=True)
    def test_required_when_env_unset(self):
        """Default production mode must require verification."""
        from aragora.connectors.chat.webhook_security import is_webhook_verification_required

        assert is_webhook_verification_required() is True

    @patch.dict(os.environ, {"ARAGORA_ENV": "production"})
    def test_required_in_production(self):
        from aragora.connectors.chat.webhook_security import is_webhook_verification_required

        assert is_webhook_verification_required() is True

    @patch.dict(
        os.environ, {"ARAGORA_ENV": "production", "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": "true"}
    )
    def test_override_ignored_in_production(self):
        """ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS must be ignored in production."""
        from aragora.connectors.chat.webhook_security import is_webhook_verification_required

        assert is_webhook_verification_required() is True

    @patch.dict(os.environ, {"ARAGORA_ENV": "development"})
    def test_required_in_dev_without_override(self):
        """Without the override flag, verification is still required in dev."""
        from aragora.connectors.chat.webhook_security import is_webhook_verification_required

        assert is_webhook_verification_required() is True

    @patch.dict(
        os.environ, {"ARAGORA_ENV": "development", "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": "true"}
    )
    def test_not_required_in_dev_with_override(self):
        """Override flag works only in development."""
        from aragora.connectors.chat.webhook_security import is_webhook_verification_required

        assert is_webhook_verification_required() is False

    @patch.dict(
        os.environ, {"ARAGORA_ENV": "development", "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": "1"}
    )
    def test_not_required_in_dev_with_override_1(self):
        """Override flag '1' also works in development."""
        from aragora.connectors.chat.webhook_security import is_webhook_verification_required

        assert is_webhook_verification_required() is False


class TestShouldAllowUnverified:
    """Tests for should_allow_unverified()."""

    @patch.dict(os.environ, {}, clear=True)
    def test_disallowed_when_env_unset(self):
        """Default production mode must not allow unverified webhooks."""
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        assert should_allow_unverified("slack") is False

    @patch.dict(os.environ, {"ARAGORA_ENV": "production"})
    def test_disallowed_in_production(self):
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        assert should_allow_unverified("teams") is False

    @patch.dict(
        os.environ, {"ARAGORA_ENV": "production", "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": "true"}
    )
    def test_override_ignored_in_production(self):
        """Even with override flag set, production must reject unverified."""
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        assert should_allow_unverified("discord") is False

    @patch.dict(
        os.environ, {"ARAGORA_ENV": "development", "ARAGORA_ALLOW_UNVERIFIED_WEBHOOKS": "true"}
    )
    def test_allowed_in_dev_with_override(self):
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        assert should_allow_unverified("slack") is True

    @patch.dict(os.environ, {"ARAGORA_ENV": "development"})
    def test_disallowed_in_dev_without_override(self):
        from aragora.connectors.chat.webhook_security import should_allow_unverified

        assert should_allow_unverified("slack") is False


# ---------------------------------------------------------------------------
# 2. JWT verify production default tests
# ---------------------------------------------------------------------------


class TestJWTProductionDefault:
    """Tests for jwt_verify._IS_PRODUCTION module-level flag."""

    def test_is_production_true_when_unset(self):
        """_IS_PRODUCTION must be True when ARAGORA_ENV is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # _IS_PRODUCTION is set at module load time, so we must reload
            import aragora.connectors.chat.jwt_verify as jwt_mod

            importlib.reload(jwt_mod)
            assert jwt_mod._IS_PRODUCTION is True

    def test_is_production_true_when_production(self):
        """_IS_PRODUCTION must be True when ARAGORA_ENV=production."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}, clear=True):
            import aragora.connectors.chat.jwt_verify as jwt_mod

            importlib.reload(jwt_mod)
            assert jwt_mod._IS_PRODUCTION is True

    def test_is_production_true_when_prod(self):
        """_IS_PRODUCTION must be True when ARAGORA_ENV=prod."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "prod"}, clear=True):
            import aragora.connectors.chat.jwt_verify as jwt_mod

            importlib.reload(jwt_mod)
            assert jwt_mod._IS_PRODUCTION is True

    def test_is_production_false_when_development(self):
        """_IS_PRODUCTION must be False when ARAGORA_ENV=development."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=True):
            import aragora.connectors.chat.jwt_verify as jwt_mod

            importlib.reload(jwt_mod)
            assert jwt_mod._IS_PRODUCTION is False

    def test_is_production_false_when_test(self):
        """_IS_PRODUCTION must be False when ARAGORA_ENV=test."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "test"}, clear=True):
            import aragora.connectors.chat.jwt_verify as jwt_mod

            importlib.reload(jwt_mod)
            assert jwt_mod._IS_PRODUCTION is False


# ---------------------------------------------------------------------------
# 3. Startup validation tests (unified_server._init_subsystems)
# ---------------------------------------------------------------------------


def _mock_handler_stores():
    """Return the dict init_handler_stores is expected to produce."""
    return {
        "document_store": None,
        "audio_store": None,
        "video_generator": None,
        "twitter_connector": None,
        "youtube_connector": None,
        "user_store": None,
        "usage_tracker": None,
    }


def _run_init_subsystems(env_dict, caplog, pre_hook=None):
    """Run UnifiedServer._init_subsystems with heavy mocking.

    Because _init_subsystems uses local imports (from aragora.server.initialization ...),
    we must patch at the *source* module so the local import picks up the mock.
    We also patch the CrossDebateMemory / KnowledgeMound local imports via
    sys.modules manipulation.

    Args:
        env_dict: os.environ contents (used with clear=True).
        caplog: pytest caplog fixture.
        pre_hook: optional callable(UnifiedHandler) to configure handler attrs
                  before _init_subsystems runs its summary block.
    """
    import sys

    from aragora.server.unified_server import UnifiedHandler, UnifiedServer

    mock_registry = MagicMock()

    with (
        patch.dict(os.environ, env_dict, clear=True),
        patch("aragora.server.initialization.initialize_subsystems", return_value=mock_registry),
        patch(
            "aragora.server.initialization.init_handler_stores", return_value=_mock_handler_stores()
        ),
        patch.object(UnifiedServer, "_init_decision_router"),
        # Block the CrossDebateMemory and KnowledgeMound local imports so they
        # don't trigger real filesystem/database work.
        patch.dict(
            sys.modules,
            {
                "aragora.memory.cross_debate_rlm": MagicMock(),
                "aragora.knowledge.mound": MagicMock(),
            },
        ),
    ):
        if pre_hook is not None:
            pre_hook(UnifiedHandler)

        instance = object.__new__(UnifiedServer)

        with caplog.at_level(logging.DEBUG, logger="aragora.server.unified_server"):
            instance._init_subsystems(Path("/tmp/test_nomic"))

    return caplog.messages


class TestStartupEnvWarning:
    """Tests for ARAGORA_ENV warning in _init_subsystems."""

    def test_warning_logged_when_env_unset(self, caplog):
        """A warning must be logged when ARAGORA_ENV is not set."""
        msgs = _run_init_subsystems({}, caplog)
        assert any("ARAGORA_ENV is not set" in m for m in msgs), (
            f"Expected ARAGORA_ENV warning in logs, got: {msgs}"
        )

    def test_no_warning_when_env_set_production(self, caplog):
        """No ARAGORA_ENV warning when the variable is explicitly set to production."""
        msgs = _run_init_subsystems({"ARAGORA_ENV": "production"}, caplog)
        assert not any("ARAGORA_ENV is not set" in m for m in msgs), (
            f"Unexpected ARAGORA_ENV warning when env is set: {msgs}"
        )

    def test_no_warning_when_env_set_development(self, caplog):
        """No ARAGORA_ENV warning when the variable is explicitly set to development."""
        msgs = _run_init_subsystems({"ARAGORA_ENV": "development"}, caplog)
        assert not any("ARAGORA_ENV is not set" in m for m in msgs), (
            f"Unexpected ARAGORA_ENV warning when env is set: {msgs}"
        )


class TestSubsystemSummaryLogging:
    """Tests for subsystem initialization summary logging."""

    def test_all_subsystems_initialized_logs_info(self, caplog):
        """When all subsystems initialize, an info-level summary is logged."""

        def set_all(handler):
            handler.cross_debate_memory = MagicMock()
            handler.knowledge_mound = MagicMock()
            handler.decision_router = MagicMock()
            handler.continuum_memory = MagicMock()

        msgs = _run_init_subsystems({"ARAGORA_ENV": "development"}, caplog, pre_hook=set_all)
        assert any("All" in m and "core subsystems initialized successfully" in m for m in msgs), (
            f"Expected success summary in logs, got: {msgs}"
        )

    def test_missing_subsystems_logs_warning(self, caplog):
        """When some subsystems fail to initialize, a warning with names is logged."""
        from aragora.server.unified_server import UnifiedHandler, UnifiedServer

        mock_registry = MagicMock()
        # Make continuum_memory None so it shows as missing too
        mock_registry.continuum_memory = None

        original_cdm = getattr(UnifiedHandler, "cross_debate_memory", None)
        original_km = getattr(UnifiedHandler, "knowledge_mound", None)
        original_dr = getattr(UnifiedHandler, "decision_router", None)
        original_cm = getattr(UnifiedHandler, "continuum_memory", None)

        # Patch imports so CrossDebateMemory and KnowledgeMound raise ImportError
        def _import_raiser(name, *args, **kwargs):
            if name in ("aragora.memory.cross_debate_rlm", "aragora.knowledge.mound"):
                raise ImportError(f"test: {name} unavailable")
            return original_import(name, *args, **kwargs)

        import builtins

        original_import = builtins.__import__

        try:
            with (
                patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=True),
                patch(
                    "aragora.server.initialization.initialize_subsystems",
                    return_value=mock_registry,
                ),
                patch(
                    "aragora.server.initialization.init_handler_stores",
                    return_value=_mock_handler_stores(),
                ),
                patch.object(UnifiedServer, "_init_decision_router"),
                patch.object(builtins, "__import__", side_effect=_import_raiser),
            ):
                instance = object.__new__(UnifiedServer)

                with caplog.at_level(logging.WARNING, logger="aragora.server.unified_server"):
                    instance._init_subsystems(Path("/tmp/test_nomic"))

            missing_msgs = [m for m in caplog.messages if "Missing" in m]
            assert len(missing_msgs) > 0, (
                f"Expected missing subsystems warning, got: {caplog.messages}"
            )
            assert any(
                "cross_debate_memory" in m or "knowledge_mound" in m for m in missing_msgs
            ), f"Expected specific missing subsystem names, got: {missing_msgs}"
        finally:
            # Restore class attributes
            UnifiedHandler.cross_debate_memory = original_cdm
            UnifiedHandler.knowledge_mound = original_km
            UnifiedHandler.decision_router = original_dr
            UnifiedHandler.continuum_memory = original_cm
