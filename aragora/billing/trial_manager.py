"""
Trial Manager for Aragora.

Manages free trial lifecycle including:
- Trial initialization for new signups
- Trial status checking and enforcement
- Trial conversion to paid subscriptions
- Trial expiration handling and notifications
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Protocol

from .models import Organization, SubscriptionTier

logger = logging.getLogger(__name__)

# Trial configuration defaults
DEFAULT_TRIAL_DURATION_DAYS = 7
DEFAULT_TRIAL_DEBATES_LIMIT = 10


class UserStore(Protocol):
    """Protocol for user storage backends."""

    def get_organization(self, org_id: str) -> Organization | None:
        """Get organization by ID."""
        ...

    def save_organization(self, org: Organization) -> None:
        """Save organization."""
        ...


@dataclass
class TrialStatus:
    """Current trial status for an organization."""

    is_active: bool
    is_expired: bool
    days_remaining: int
    debates_remaining: int
    debates_used: int
    debates_limit: int
    started_at: datetime | None
    expires_at: datetime | None
    converted: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "is_active": self.is_active,
            "is_expired": self.is_expired,
            "days_remaining": self.days_remaining,
            "debates_remaining": self.debates_remaining,
            "debates_used": self.debates_used,
            "debates_limit": self.debates_limit,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "converted": self.converted,
        }


class TrialManager:
    """
    Manages free trial lifecycle.

    Usage:
        trial_mgr = TrialManager(user_store)

        # Start trial for new signup
        trial_mgr.start_trial(org)

        # Check trial status
        status = trial_mgr.get_trial_status(org)

        # Check if action is allowed
        can_debate = trial_mgr.can_start_debate(org)

        # Record debate usage
        trial_mgr.record_debate(org)

        # Convert trial to paid
        trial_mgr.convert_trial(org, SubscriptionTier.STARTER)
    """

    def __init__(
        self,
        user_store: UserStore | None = None,
        trial_duration_days: int = DEFAULT_TRIAL_DURATION_DAYS,
        trial_debates_limit: int = DEFAULT_TRIAL_DEBATES_LIMIT,
    ):
        """
        Initialize TrialManager.

        Args:
            user_store: Optional user store for persisting changes
            trial_duration_days: Trial duration in days (default 7)
            trial_debates_limit: Maximum debates during trial (default 10)
        """
        self.user_store = user_store
        self.trial_duration_days = trial_duration_days
        self.trial_debates_limit = trial_debates_limit

    def start_trial(self, org: Organization) -> TrialStatus:
        """
        Start a free trial for an organization.

        Args:
            org: Organization to start trial for

        Returns:
            TrialStatus with trial details
        """
        if org.trial_started_at is not None:
            logger.warning("Trial already started for org %s", org.id)
            return self.get_trial_status(org)

        org.start_trial(
            duration_days=self.trial_duration_days,
            debates_limit=self.trial_debates_limit,
        )

        if self.user_store:
            self.user_store.save_organization(org)

        logger.info(
            "Started trial for org %s: %s days, %s debates",
            org.id,
            self.trial_duration_days,
            self.trial_debates_limit,
        )

        return self.get_trial_status(org)

    def get_trial_status(self, org: Organization) -> TrialStatus:
        """
        Get current trial status for an organization.

        Args:
            org: Organization to check

        Returns:
            TrialStatus with current trial details
        """
        return TrialStatus(
            is_active=org.is_in_trial,
            is_expired=org.is_trial_expired,
            days_remaining=org.trial_days_remaining,
            debates_remaining=org.trial_debates_remaining,
            debates_used=org.debates_used_this_month,
            debates_limit=org.trial_debates_limit,
            started_at=org.trial_started_at,
            expires_at=org.trial_expires_at,
            converted=org.trial_converted,
        )

    def can_start_debate(self, org: Organization) -> tuple[bool, str]:
        """
        Check if organization can start a new debate.

        Args:
            org: Organization to check

        Returns:
            Tuple of (allowed, reason) where reason explains denial
        """
        # Not in trial - use normal tier limits
        if not org.is_in_trial:
            if org.is_trial_expired and not org.trial_converted:
                return False, "Trial has expired. Please upgrade to continue."
            # Normal tier limit check
            if org.is_at_limit:
                return (
                    False,
                    f"Monthly debate limit reached ({org.limits.debates_per_month} debates)",
                )
            return True, ""

        # In trial - check trial-specific limits
        if org.is_at_limit:
            return False, f"Trial debate limit reached ({org.trial_debates_limit} debates)"

        return True, ""

    def record_debate(self, org: Organization) -> bool:
        """
        Record a debate for the organization.

        Args:
            org: Organization that ran a debate

        Returns:
            True if recorded successfully, False if at limit
        """
        success = org.increment_debates(1)

        if success and self.user_store:
            self.user_store.save_organization(org)

        return success

    def convert_trial(
        self,
        org: Organization,
        new_tier: SubscriptionTier,
    ) -> bool:
        """
        Convert a trial to a paid subscription.

        Args:
            org: Organization to convert
            new_tier: New subscription tier

        Returns:
            True if converted successfully
        """
        if not org.is_in_trial and not org.is_trial_expired:
            logger.warning("Cannot convert non-trial org %s", org.id)
            return False

        org.convert_trial(new_tier)

        if self.user_store:
            self.user_store.save_organization(org)

        logger.info("Converted trial for org %s to tier %s", org.id, new_tier.value)
        return True

    def extend_trial(
        self,
        org: Organization,
        additional_days: int = 7,
    ) -> TrialStatus:
        """
        Extend a trial by additional days.

        Args:
            org: Organization to extend
            additional_days: Days to add to trial

        Returns:
            Updated TrialStatus
        """
        if org.trial_expires_at is None:
            logger.warning("Cannot extend - no trial exists for org %s", org.id)
            return self.get_trial_status(org)

        org.trial_expires_at = org.trial_expires_at + timedelta(days=additional_days)
        org.updated_at = datetime.now(timezone.utc)

        if self.user_store:
            self.user_store.save_organization(org)

        logger.info("Extended trial for org %s by %s days", org.id, additional_days)
        return self.get_trial_status(org)

    def get_expiring_trials(
        self,
        user_store: UserStore,
        days_ahead: int = 2,
    ) -> list[Organization]:
        """
        Get organizations with trials expiring soon.

        Args:
            user_store: User store to query
            days_ahead: Days ahead to check for expiring trials

        Returns:
            List of organizations with trials expiring within days_ahead
        """
        # This is a placeholder - actual implementation depends on user store
        # supporting query by trial_expires_at range
        logger.debug("Would query for trials expiring within %s days", days_ahead)
        return []


# Module-level singleton for convenience
_default_trial_manager: TrialManager | None = None


def get_trial_manager() -> TrialManager:
    """Get the default TrialManager instance."""
    global _default_trial_manager
    if _default_trial_manager is None:
        _default_trial_manager = TrialManager()
    return _default_trial_manager


def set_trial_manager(manager: TrialManager) -> None:
    """Set the default TrialManager instance."""
    global _default_trial_manager
    _default_trial_manager = manager


__all__ = [
    "TrialManager",
    "TrialStatus",
    "get_trial_manager",
    "set_trial_manager",
    "DEFAULT_TRIAL_DURATION_DAYS",
    "DEFAULT_TRIAL_DEBATES_LIMIT",
]
