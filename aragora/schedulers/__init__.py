"""Deprecated compatibility shim for :mod:`aragora.scheduler`."""

from __future__ import annotations

import importlib
import warnings

warnings.warn(
    "aragora.schedulers is deprecated. Use aragora.scheduler instead.",
    DeprecationWarning,
    stacklevel=2,
)

_canonical = importlib.import_module("aragora.scheduler")
_receipt_retention = importlib.import_module("aragora.scheduler.receipt_retention")
_slack_token_refresh = importlib.import_module("aragora.scheduler.slack_token_refresh")
_settlement_review = importlib.import_module("aragora.scheduler.settlement_review")
_extra_exports = {
    "CleanupResult": _receipt_retention,
    "CleanupStats": _receipt_retention,
    "ReceiptRetentionScheduler": _receipt_retention,
    "get_receipt_retention_scheduler": _receipt_retention,
    "set_receipt_retention_scheduler": _receipt_retention,
    "RefreshResult": _slack_token_refresh,
    "RefreshStats": _slack_token_refresh,
    "SlackTokenRefreshScheduler": _slack_token_refresh,
    "SettlementReviewResult": _settlement_review,
    "SettlementReviewScheduler": _settlement_review,
    "SettlementReviewStats": _settlement_review,
    "get_settlement_review_scheduler": _settlement_review,
    "set_settlement_review_scheduler": _settlement_review,
}

__all__ = list(dict.fromkeys([*getattr(_canonical, "__all__", ()), *_extra_exports]))


def __getattr__(name: str):
    module = _extra_exports.get(name)
    if module is not None:
        return getattr(module, name)
    return getattr(_canonical, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
