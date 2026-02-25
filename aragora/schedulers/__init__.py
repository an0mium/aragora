"""
Background schedulers for Aragora.

DEPRECATED: This package is a backward compatibility shim.
Import from `aragora.scheduler` instead.

    # New style (preferred)
    from aragora.scheduler import AccessReviewScheduler
    from aragora.scheduler.receipt_retention import ReceiptRetentionScheduler

    # Old style (still works but will be removed)
    from aragora.schedulers import ReceiptRetentionScheduler
"""

import warnings

warnings.warn(
    "aragora.schedulers is deprecated. "
    "Import from aragora.scheduler instead. "
    "This package will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new location for backward compatibility
from aragora.scheduler.receipt_retention import (
    CleanupResult,
    CleanupStats,
    ReceiptRetentionScheduler,
    get_receipt_retention_scheduler,
)
from aragora.scheduler.settlement_review import (
    SettlementReviewResult,
    SettlementReviewScheduler,
    SettlementReviewStats,
    get_settlement_review_scheduler,
)
from aragora.scheduler.slack_token_refresh import SlackTokenRefreshScheduler

__all__ = [
    "SlackTokenRefreshScheduler",
    "ReceiptRetentionScheduler",
    "CleanupResult",
    "CleanupStats",
    "get_receipt_retention_scheduler",
    "SettlementReviewResult",
    "SettlementReviewStats",
    "SettlementReviewScheduler",
    "get_settlement_review_scheduler",
]
