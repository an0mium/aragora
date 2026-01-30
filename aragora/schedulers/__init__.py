"""
Background schedulers for Aragora.

DEPRECATED: This package is a backward compatibility shim.
Import from `aragora.scheduler` instead for enterprise schedulers.

    # New style (preferred)
    from aragora.scheduler import AccessReviewScheduler

    # Old style (still works for specific schedulers)
    from aragora.schedulers import ReceiptRetentionScheduler

Note: receipt_retention, slack_token_refresh, and teams_token_refresh
schedulers remain in this package until migrated to aragora.scheduler.
"""

import warnings

warnings.warn(
    "aragora.schedulers is deprecated. "
    "For enterprise schedulers, import from aragora.scheduler instead. "
    "This package will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

from aragora.schedulers.receipt_retention import (
    CleanupResult,
    CleanupStats,
    ReceiptRetentionScheduler,
    get_receipt_retention_scheduler,
)
from aragora.schedulers.slack_token_refresh import SlackTokenRefreshScheduler

__all__ = [
    "SlackTokenRefreshScheduler",
    "ReceiptRetentionScheduler",
    "CleanupResult",
    "CleanupStats",
    "get_receipt_retention_scheduler",
]
