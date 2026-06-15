"""Tests for deprecated import path shims.

Each shim module emits a DeprecationWarning on import and re-exports symbols
from the new canonical location. Tests verify:
1. Importing the deprecated path triggers DeprecationWarning
2. The imported object ``is`` the same as from the canonical path (where applicable)

See docs/BREAKING_CHANGES.md "Module Relocations" for the full list.
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_fresh(module_name: str):
    """Import a module after removing it from sys.modules so the warning fires again."""
    # Remove the target module and any cached sub-imports to ensure fresh load
    to_remove = [
        key for key in sys.modules if key == module_name or key.startswith(module_name + ".")
    ]
    for key in to_remove:
        del sys.modules[key]
    return importlib.import_module(module_name)


def _assert_deprecation_warning(module_name: str, expected_substring: str | None = None):
    """Import *module_name* and assert that at least one DeprecationWarning is raised.

    Returns the imported module.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod = _import_fresh(module_name)

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warnings, (
        f"Expected DeprecationWarning when importing {module_name!r}, "
        f"but only got: {[w.category.__name__ for w in caught]}"
    )

    if expected_substring:
        messages = [str(w.message) for w in dep_warnings]
        assert any(expected_substring in msg for msg in messages), (
            f"Expected substring {expected_substring!r} in DeprecationWarning messages, "
            f"but got: {messages}"
        )

    return mod


# ---------------------------------------------------------------------------
# 1. aragora.schedulers -> aragora.scheduler
# ---------------------------------------------------------------------------


class TestSchedulersShim:
    """aragora.schedulers is deprecated in favor of aragora.scheduler."""

    def test_import_triggers_deprecation_warning(self) -> None:
        _assert_deprecation_warning("aragora.schedulers", "aragora.scheduler")

    def test_cleanup_result_is_same_object(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.schedulers")
        from aragora.scheduler.receipt_retention import CleanupResult

        assert shim.CleanupResult is CleanupResult

    def test_cleanup_stats_is_same_object(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.schedulers")
        from aragora.scheduler.receipt_retention import CleanupStats

        assert shim.CleanupStats is CleanupStats

    def test_receipt_retention_scheduler_is_same_object(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.schedulers")
        from aragora.scheduler.receipt_retention import ReceiptRetentionScheduler

        assert shim.ReceiptRetentionScheduler is ReceiptRetentionScheduler

    def test_get_receipt_retention_scheduler_is_same_object(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.schedulers")
        from aragora.scheduler.receipt_retention import get_receipt_retention_scheduler

        assert shim.get_receipt_retention_scheduler is get_receipt_retention_scheduler

    def test_slack_token_refresh_scheduler_is_same_object(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.schedulers")
        from aragora.scheduler.slack_token_refresh import SlackTokenRefreshScheduler

        assert shim.SlackTokenRefreshScheduler is SlackTokenRefreshScheduler

    def test_settlement_review_scheduler_is_same_object(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.schedulers")
        from aragora.scheduler.settlement_review import SettlementReviewScheduler

        assert shim.SettlementReviewScheduler is SettlementReviewScheduler


# ---------------------------------------------------------------------------
# 2. aragora.operations -> aragora.ops
# ---------------------------------------------------------------------------


class TestOperationsShim:
    """aragora.operations is deprecated in favor of aragora.ops."""

    def test_import_triggers_deprecation_warning(self) -> None:
        _assert_deprecation_warning("aragora.operations", "aragora.ops")

    def test_key_rotation_scheduler_is_same_object(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.operations")
        from aragora.ops.key_rotation import KeyRotationScheduler

        assert shim.KeyRotationScheduler is KeyRotationScheduler

    def test_key_rotation_config_is_same_object(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.operations")
        from aragora.ops.key_rotation import KeyRotationConfig

        assert shim.KeyRotationConfig is KeyRotationConfig

    def test_key_rotation_result_is_same_object(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.operations")
        from aragora.ops.key_rotation import KeyRotationResult

        assert shim.KeyRotationResult is KeyRotationResult

    def test_get_key_rotation_scheduler_is_same_object(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.operations")
        from aragora.ops.key_rotation import get_key_rotation_scheduler

        assert shim.get_key_rotation_scheduler is get_key_rotation_scheduler


# ---------------------------------------------------------------------------
# 3. aragora.gateway.decision_router -> aragora.core.decision_router
# ---------------------------------------------------------------------------


class TestGatewayDecisionRouterShim:
    """aragora.gateway.decision_router emits DeprecationWarning."""

    def test_import_triggers_deprecation_warning(self) -> None:
        _assert_deprecation_warning(
            "aragora.gateway.decision_router",
            "aragora.core.decision_router",
        )

    def test_decision_router_class_available(self) -> None:
        """DecisionRouter should be importable from the deprecated path."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.gateway.decision_router")
        assert hasattr(shim, "DecisionRouter")

    def test_route_destination_enum_available(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.gateway.decision_router")
        assert hasattr(shim, "RouteDestination")
        assert shim.RouteDestination.DEBATE.value == "debate"

    def test_routing_criteria_available(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.gateway.decision_router")
        assert hasattr(shim, "RoutingCriteria")


# ---------------------------------------------------------------------------
# 4. aragora.observability.logging -> aragora.logging_config
# ---------------------------------------------------------------------------


class TestObservabilityLoggingShim:
    """aragora.observability.logging emits DeprecationWarning."""

    def test_import_triggers_deprecation_warning(self) -> None:
        _assert_deprecation_warning(
            "aragora.observability.logging",
            "aragora.logging_config",
        )

    def test_get_logger_available(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.observability.logging")
        assert hasattr(shim, "get_logger")
        assert callable(shim.get_logger)

    def test_configure_logging_available(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.observability.logging")
        assert hasattr(shim, "configure_logging")
        assert callable(shim.configure_logging)

    def test_structured_logger_available(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.observability.logging")
        assert hasattr(shim, "StructuredLogger")

    def test_correlation_context_available(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.observability.logging")
        assert hasattr(shim, "correlation_context")


# ---------------------------------------------------------------------------
# 5. aragora.connectors.email.gmail_sync -> aragora.connectors.enterprise.communication.gmail
# ---------------------------------------------------------------------------


class TestGmailSyncShim:
    """aragora.connectors.email.gmail_sync emits DeprecationWarning."""

    def test_import_triggers_deprecation_warning(self) -> None:
        _assert_deprecation_warning(
            "aragora.connectors.email.gmail_sync",
            "aragora.connectors.enterprise.communication.gmail",
        )

    def test_gmail_sync_service_available(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.connectors.email.gmail_sync")
        assert hasattr(shim, "GmailSyncService")

    def test_gmail_sync_config_available(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.connectors.email.gmail_sync")
        assert hasattr(shim, "GmailSyncConfig")

    def test_sync_status_enum_available(self) -> None:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            shim = _import_fresh("aragora.connectors.email.gmail_sync")
        assert hasattr(shim, "SyncStatus")
        assert shim.SyncStatus.IDLE.value == "idle"


# ---------------------------------------------------------------------------
# 6. aragora.modes.gauntlet -> aragora.gauntlet
# ---------------------------------------------------------------------------


class TestModesGauntletShim:
    """aragora.modes.gauntlet should emit DeprecationWarning."""

    def test_import_triggers_deprecation_warning(self) -> None:
        _assert_deprecation_warning("aragora.modes.gauntlet", "aragora.gauntlet")

    def test_canonical_gauntlet_importable(self) -> None:
        """The canonical aragora.gauntlet module should import without warnings."""
        # Just verify the canonical path works
        mod = importlib.import_module("aragora.gauntlet")
        assert hasattr(mod, "GauntletRunner")
