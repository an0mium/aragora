"""
Billing Data Models.

Core data structures for user management, organizations, and subscriptions.
"""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from aragora.exceptions import ConfigurationError

# Try to import bcrypt for secure password hashing
try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False

logger = logging.getLogger(__name__)

# Password hash version prefixes for migration support
HASH_VERSION_SHA256 = "sha256:"
HASH_VERSION_BCRYPT = "bcrypt:"
HASH_VERSION_CURRENT = HASH_VERSION_BCRYPT if HAS_BCRYPT else HASH_VERSION_SHA256
BCRYPT_ROUNDS = 12  # Cost factor for bcrypt

# Security: Allow SHA-256 fallback only when explicitly enabled
# Set ARAGORA_ALLOW_INSECURE_PASSWORDS=1 for testing (NOT for production)
# Note: bcrypt is a required dependency, so fallback should never be needed
ALLOW_INSECURE_PASSWORDS = (
    os.environ.get("ARAGORA_ALLOW_INSECURE_PASSWORDS", "").lower() in ("1", "true", "yes")
)


class SubscriptionTier(Enum):
    """Available subscription tiers."""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class TierLimits:
    """Limits for a subscription tier."""

    debates_per_month: int
    users_per_org: int
    api_access: bool
    all_agents: bool
    custom_agents: bool
    sso_enabled: bool
    audit_logs: bool
    priority_support: bool
    price_monthly_cents: int  # Price in cents

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "debates_per_month": self.debates_per_month,
            "users_per_org": self.users_per_org,
            "api_access": self.api_access,
            "all_agents": self.all_agents,
            "custom_agents": self.custom_agents,
            "sso_enabled": self.sso_enabled,
            "audit_logs": self.audit_logs,
            "priority_support": self.priority_support,
            "price_monthly_cents": self.price_monthly_cents,
        }


# Tier configurations
TIER_LIMITS: dict[SubscriptionTier, TierLimits] = {
    SubscriptionTier.FREE: TierLimits(
        debates_per_month=10,
        users_per_org=1,
        api_access=False,
        all_agents=False,
        custom_agents=False,
        sso_enabled=False,
        audit_logs=False,
        priority_support=False,
        price_monthly_cents=0,
    ),
    SubscriptionTier.STARTER: TierLimits(
        debates_per_month=50,
        users_per_org=2,
        api_access=False,
        all_agents=False,
        custom_agents=False,
        sso_enabled=False,
        audit_logs=False,
        priority_support=False,
        price_monthly_cents=9900,  # $99
    ),
    SubscriptionTier.PROFESSIONAL: TierLimits(
        debates_per_month=200,
        users_per_org=10,
        api_access=True,
        all_agents=True,
        custom_agents=False,
        sso_enabled=False,
        audit_logs=True,
        priority_support=False,
        price_monthly_cents=29900,  # $299
    ),
    SubscriptionTier.ENTERPRISE: TierLimits(
        debates_per_month=999999,  # Unlimited
        users_per_org=999999,
        api_access=True,
        all_agents=True,
        custom_agents=True,
        sso_enabled=True,
        audit_logs=True,
        priority_support=True,
        price_monthly_cents=99900,  # $999 base
    ),
}


def _hash_password_sha256(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Legacy SHA-256 password hashing (for backward compatibility).

    Args:
        password: Plain text password
        salt: Optional salt (generated if not provided)

    Returns:
        Tuple of (hash, salt)
    """
    if salt is None:
        salt = secrets.token_hex(32)
    hash_input = f"{salt}{password}".encode("utf-8")
    password_hash = hashlib.sha256(hash_input).hexdigest()
    return password_hash, salt


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash a password using bcrypt.

    In production, bcrypt is REQUIRED. SHA-256 fallback is only allowed
    when ARAGORA_ALLOW_INSECURE_PASSWORDS=1 is set (for testing only).

    Args:
        password: Plain text password
        salt: Optional salt (only used for SHA-256 fallback in dev)

    Returns:
        Tuple of (versioned_hash, salt)
        - For bcrypt: salt is empty string (embedded in hash)
        - For SHA-256: salt is the random salt used

    Raises:
        RuntimeError: If bcrypt is not installed and insecure fallback not enabled
    """
    if HAS_BCRYPT:
        # Use bcrypt (salt is embedded in the hash)
        password_bytes = password.encode("utf-8")
        hashed = bcrypt.hashpw(password_bytes, bcrypt.gensalt(rounds=BCRYPT_ROUNDS))
        return f"{HASH_VERSION_BCRYPT}{hashed.decode('utf-8')}", ""
    elif ALLOW_INSECURE_PASSWORDS:
        # Fall back to SHA-256 ONLY in development/testing
        legacy_hash, salt = _hash_password_sha256(password, salt)
        logger.warning(
            "SECURITY WARNING: Using SHA-256 for password hashing. "
            "This is insecure for production. Install bcrypt: pip install bcrypt"
        )
        return f"{HASH_VERSION_SHA256}{legacy_hash}", salt
    else:
        # Production: fail if bcrypt not available
        raise ConfigurationError(
            component="Password Hashing",
            reason="bcrypt is required for secure password hashing but is not installed. "
                   "Install it with: pip install bcrypt. "
                   "For development/testing only, set ARAGORA_ALLOW_INSECURE_PASSWORDS=1"
        )


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    """
    Verify a password against a stored hash with automatic version detection.

    Supports:
    - bcrypt: hashes prefixed with "bcrypt:"
    - sha256: hashes prefixed with "sha256:" or legacy unprefixed 64-char hex

    Args:
        password: Plain text password to verify
        password_hash: Stored hash (may include version prefix)
        salt: Stored salt (used for SHA-256, ignored for bcrypt)

    Returns:
        True if password matches
    """
    if password_hash.startswith(HASH_VERSION_BCRYPT):
        # Modern bcrypt verification
        if not HAS_BCRYPT:
            logger.error("Cannot verify bcrypt hash: bcrypt not installed")
            return False
        stored_hash = password_hash[len(HASH_VERSION_BCRYPT):].encode("utf-8")
        try:
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash)
        except Exception as e:
            logger.error(f"bcrypt verification failed: {e}")
            return False

    elif password_hash.startswith(HASH_VERSION_SHA256):
        # Prefixed SHA-256
        actual_hash = password_hash[len(HASH_VERSION_SHA256):]
        computed_hash, _ = _hash_password_sha256(password, salt)
        return secrets.compare_digest(computed_hash, actual_hash)

    elif len(password_hash) == 64:
        # Legacy unprefixed SHA-256 (64-char hex)
        computed_hash, _ = _hash_password_sha256(password, salt)
        return secrets.compare_digest(computed_hash, password_hash)

    else:
        logger.warning(f"Unknown password hash format (length={len(password_hash)})")
        return False


def needs_rehash(password_hash: str) -> bool:
    """
    Check if a password hash should be upgraded to the current algorithm.

    Call this after successful password verification to determine if
    the hash should be updated. This enables transparent migration
    from SHA-256 to bcrypt.

    Args:
        password_hash: The stored password hash

    Returns:
        True if hash should be regenerated with current algorithm
    """
    if not password_hash:
        return True

    # If bcrypt is available and hash isn't bcrypt, it needs rehash
    if HAS_BCRYPT and not password_hash.startswith(HASH_VERSION_BCRYPT):
        return True

    # If bcrypt isn't available but hash is bcrypt, can't rehash (keep as-is)
    if not HAS_BCRYPT and password_hash.startswith(HASH_VERSION_BCRYPT):
        return False

    return False


@dataclass
class User:
    """A user account."""

    id: str = field(default_factory=lambda: str(uuid4()))
    email: str = ""
    password_hash: str = ""
    password_salt: str = ""
    name: str = ""
    org_id: Optional[str] = None
    role: str = "member"  # owner, admin, member
    is_active: bool = True
    email_verified: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login_at: Optional[datetime] = None

    # API access (secure storage: hash + prefix for identification)
    api_key_hash: Optional[str] = None  # SHA-256 hash of the key
    api_key_prefix: Optional[str] = None  # First 12 chars for identification (ara_xxxx...)
    api_key_created_at: Optional[datetime] = None
    api_key_expires_at: Optional[datetime] = None  # Expiration time

    # Legacy field for backward compatibility during migration
    api_key: Optional[str] = None  # DEPRECATED: Plaintext key, will be removed

    # MFA/2FA fields
    mfa_secret: Optional[str] = None  # TOTP secret (encrypted)
    mfa_enabled: bool = False
    mfa_backup_codes: Optional[str] = None  # JSON-encoded list of hashed backup codes

    # Token revocation - increment to invalidate all existing tokens
    token_version: int = 1

    def set_password(self, password: str) -> None:
        """Set user password."""
        self.password_hash, self.password_salt = hash_password(password)
        self.updated_at = datetime.utcnow()

    def verify_password(self, password: str) -> bool:
        """Verify user password."""
        return verify_password(password, self.password_hash, self.password_salt)

    def needs_password_rehash(self) -> bool:
        """Check if password hash should be upgraded to current algorithm."""
        return needs_rehash(self.password_hash)

    def upgrade_password_hash(self, password: str) -> bool:
        """
        Upgrade password hash to current algorithm if needed.

        Call this after successful password verification to transparently
        migrate from SHA-256 to bcrypt.

        Args:
            password: The verified plaintext password

        Returns:
            True if hash was upgraded, False if no upgrade needed
        """
        if not self.needs_password_rehash():
            return False
        self.password_hash, self.password_salt = hash_password(password)
        self.updated_at = datetime.utcnow()
        logger.info(f"Password hash upgraded for user {self.id}")
        return True

    def generate_api_key(self, expires_days: int = 365) -> str:
        """
        Generate a new API key for this user.

        The plaintext key is returned once and never stored. Only the
        SHA-256 hash is persisted for verification.

        Args:
            expires_days: Days until key expires (default 365)

        Returns:
            The plaintext API key (only returned once, never stored)
        """
        api_key = f"ara_{secrets.token_urlsafe(32)}"

        # Store hash, not plaintext
        self.api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        self.api_key_prefix = api_key[:12]  # "ara_" + 8 chars for identification
        self.api_key_created_at = datetime.utcnow()
        self.api_key_expires_at = datetime.utcnow() + timedelta(days=expires_days)
        self.updated_at = datetime.utcnow()

        # Clear legacy field
        self.api_key = None

        return api_key  # Returned to user once, never stored

    def verify_api_key(self, api_key: str) -> bool:
        """
        Verify an API key against stored hash.

        Checks both hash match and expiration.

        Args:
            api_key: The plaintext API key to verify

        Returns:
            True if key is valid and not expired
        """
        if not self.api_key_hash:
            # Check legacy plaintext key for backward compatibility
            if self.api_key and secrets.compare_digest(self.api_key, api_key):
                logger.warning(f"Legacy plaintext API key used for user {self.id}")
                return True
            return False

        # Check expiration
        if self.api_key_expires_at and datetime.utcnow() > self.api_key_expires_at:
            logger.debug(f"API key expired for user {self.id}")
            return False

        # Verify hash
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return secrets.compare_digest(key_hash, self.api_key_hash)

    def is_api_key_expired(self) -> bool:
        """Check if API key is expired."""
        if not self.api_key_expires_at:
            return False
        return datetime.utcnow() > self.api_key_expires_at

    def revoke_api_key(self) -> None:
        """Revoke the user's API key."""
        self.api_key_hash = None
        self.api_key_prefix = None
        self.api_key_created_at = None
        self.api_key_expires_at = None
        self.api_key = None  # Also clear legacy field
        self.updated_at = datetime.utcnow()

    def to_dict(self, include_sensitive: bool = False) -> dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "org_id": self.org_id,
            "role": self.role,
            "is_active": self.is_active,
            "email_verified": self.email_verified,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "has_api_key": self.api_key_hash is not None or self.api_key is not None,
            "token_version": self.token_version,
        }
        if include_sensitive:
            # API key is only available if stored in legacy format
            data["api_key"] = self.api_key
            data["api_key_prefix"] = self.api_key_prefix
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "User":
        """Create from dictionary."""
        user = cls(
            id=data.get("id", str(uuid4())),
            email=data.get("email", ""),
            password_hash=data.get("password_hash", ""),
            password_salt=data.get("password_salt", ""),
            name=data.get("name", ""),
            org_id=data.get("org_id"),
            role=data.get("role", "member"),
            is_active=data.get("is_active", True),
            email_verified=data.get("email_verified", False),
            api_key=data.get("api_key"),
            token_version=data.get("token_version", 1),
        )
        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                user.created_at = datetime.fromisoformat(data["created_at"])
            else:
                user.created_at = data["created_at"]
        if "updated_at" in data and data["updated_at"]:
            if isinstance(data["updated_at"], str):
                user.updated_at = datetime.fromisoformat(data["updated_at"])
            else:
                user.updated_at = data["updated_at"]
        if "last_login_at" in data and data["last_login_at"]:
            if isinstance(data["last_login_at"], str):
                user.last_login_at = datetime.fromisoformat(data["last_login_at"])
            else:
                user.last_login_at = data["last_login_at"]
        if "api_key_created_at" in data and data["api_key_created_at"]:
            if isinstance(data["api_key_created_at"], str):
                user.api_key_created_at = datetime.fromisoformat(data["api_key_created_at"])
            else:
                user.api_key_created_at = data["api_key_created_at"]
        return user


@dataclass
class Organization:
    """An organization (team/company)."""

    id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    slug: str = ""  # URL-friendly name
    tier: SubscriptionTier = SubscriptionTier.FREE
    owner_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # Billing
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None

    # Usage tracking (reset monthly)
    debates_used_this_month: int = 0
    billing_cycle_start: datetime = field(default_factory=datetime.utcnow)

    # Settings
    settings: dict[str, Any] = field(default_factory=dict)

    @property
    def limits(self) -> TierLimits:
        """Get limits for this organization's tier."""
        return TIER_LIMITS[self.tier]

    @property
    def debates_remaining(self) -> int:
        """Get remaining debates this month."""
        return max(0, self.limits.debates_per_month - self.debates_used_this_month)

    @property
    def is_at_limit(self) -> bool:
        """Check if organization has reached debate limit."""
        return self.debates_used_this_month >= self.limits.debates_per_month

    def increment_debates(self, count: int = 1) -> bool:
        """
        Increment debate count.

        Returns:
            True if successful, False if at limit
        """
        if self.is_at_limit:
            return False
        self.debates_used_this_month += count
        self.updated_at = datetime.utcnow()
        return True

    def reset_monthly_usage(self) -> None:
        """Reset monthly usage counters."""
        self.debates_used_this_month = 0
        self.billing_cycle_start = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "slug": self.slug,
            "tier": self.tier.value,
            "owner_id": self.owner_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "stripe_customer_id": self.stripe_customer_id,
            "stripe_subscription_id": self.stripe_subscription_id,
            "debates_used_this_month": self.debates_used_this_month,
            "billing_cycle_start": self.billing_cycle_start.isoformat(),
            "debates_remaining": self.debates_remaining,
            "is_at_limit": self.is_at_limit,
            "limits": self.limits.to_dict(),
            "settings": self.settings,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Organization":
        """Create from dictionary."""
        org = cls(
            id=data.get("id", str(uuid4())),
            name=data.get("name", ""),
            slug=data.get("slug", ""),
            tier=SubscriptionTier(data.get("tier", "free")),
            owner_id=data.get("owner_id"),
            stripe_customer_id=data.get("stripe_customer_id"),
            stripe_subscription_id=data.get("stripe_subscription_id"),
            debates_used_this_month=data.get("debates_used_this_month", 0),
            settings=data.get("settings", {}),
        )
        if "created_at" in data and data["created_at"]:
            if isinstance(data["created_at"], str):
                org.created_at = datetime.fromisoformat(data["created_at"])
            else:
                org.created_at = data["created_at"]
        if "updated_at" in data and data["updated_at"]:
            if isinstance(data["updated_at"], str):
                org.updated_at = datetime.fromisoformat(data["updated_at"])
            else:
                org.updated_at = data["updated_at"]
        if "billing_cycle_start" in data and data["billing_cycle_start"]:
            if isinstance(data["billing_cycle_start"], str):
                org.billing_cycle_start = datetime.fromisoformat(data["billing_cycle_start"])
            else:
                org.billing_cycle_start = data["billing_cycle_start"]
        return org


@dataclass
class Subscription:
    """A subscription record."""

    id: str = field(default_factory=lambda: str(uuid4()))
    org_id: str = ""
    tier: SubscriptionTier = SubscriptionTier.FREE
    status: str = "active"  # active, canceled, past_due, trialing
    stripe_subscription_id: Optional[str] = None
    stripe_price_id: Optional[str] = None

    # Billing period
    current_period_start: datetime = field(default_factory=datetime.utcnow)
    current_period_end: datetime = field(
        default_factory=lambda: datetime.utcnow() + timedelta(days=30)
    )
    cancel_at_period_end: bool = False

    # Trial
    trial_start: Optional[datetime] = None
    trial_end: Optional[datetime] = None

    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_active(self) -> bool:
        """Check if subscription is active."""
        return self.status in ("active", "trialing")

    @property
    def is_trialing(self) -> bool:
        """Check if subscription is in trial."""
        if self.status != "trialing":
            return False
        if self.trial_end is None:
            return False
        return datetime.utcnow() < self.trial_end

    @property
    def days_until_renewal(self) -> int:
        """Get days until next renewal."""
        delta = self.current_period_end - datetime.utcnow()
        return max(0, delta.days)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "org_id": self.org_id,
            "tier": self.tier.value,
            "status": self.status,
            "stripe_subscription_id": self.stripe_subscription_id,
            "stripe_price_id": self.stripe_price_id,
            "current_period_start": self.current_period_start.isoformat(),
            "current_period_end": self.current_period_end.isoformat(),
            "cancel_at_period_end": self.cancel_at_period_end,
            "trial_start": self.trial_start.isoformat() if self.trial_start else None,
            "trial_end": self.trial_end.isoformat() if self.trial_end else None,
            "is_active": self.is_active,
            "is_trialing": self.is_trialing,
            "days_until_renewal": self.days_until_renewal,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Subscription":
        """Create from dictionary."""
        sub = cls(
            id=data.get("id", str(uuid4())),
            org_id=data.get("org_id", ""),
            tier=SubscriptionTier(data.get("tier", "free")),
            status=data.get("status", "active"),
            stripe_subscription_id=data.get("stripe_subscription_id"),
            stripe_price_id=data.get("stripe_price_id"),
            cancel_at_period_end=data.get("cancel_at_period_end", False),
        )
        for field_name in ["current_period_start", "current_period_end", "trial_start",
                          "trial_end", "created_at", "updated_at"]:
            if field_name in data and data[field_name]:
                value = data[field_name]
                if isinstance(value, str):
                    value = datetime.fromisoformat(value)
                setattr(sub, field_name, value)
        return sub


@dataclass
class OrganizationInvitation:
    """An organization invitation for a user.

    Invitations are sent to email addresses. When the user registers or
    logs in with that email, they can accept the invitation to join.
    Invitations expire after a configurable number of days.
    """

    id: str = field(default_factory=lambda: str(uuid4()))
    org_id: str = ""
    email: str = ""  # Email address of invitee
    role: str = "member"  # Role to assign on acceptance (member, admin)
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    invited_by: Optional[str] = None  # User ID of inviter
    status: str = "pending"  # pending, accepted, expired, revoked
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: datetime = field(
        default_factory=lambda: datetime.utcnow() + timedelta(days=7)
    )
    accepted_by: Optional[str] = None  # User ID who accepted the invitation
    accepted_at: Optional[datetime] = None

    @property
    def is_expired(self) -> bool:
        """Check if invitation has expired."""
        return datetime.utcnow() > self.expires_at

    @property
    def is_pending(self) -> bool:
        """Check if invitation is still pending and valid."""
        return self.status == "pending" and not self.is_expired

    def accept(self) -> bool:
        """Mark invitation as accepted.

        Returns:
            True if successfully accepted, False if already processed or expired
        """
        if not self.is_pending:
            return False
        self.status = "accepted"
        self.accepted_at = datetime.utcnow()
        return True

    def revoke(self) -> bool:
        """Revoke the invitation.

        Returns:
            True if successfully revoked, False if already processed
        """
        if self.status != "pending":
            return False
        self.status = "revoked"
        return True

    def to_dict(self, include_token: bool = False) -> dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "id": self.id,
            "org_id": self.org_id,
            "email": self.email,
            "role": self.role,
            "invited_by": self.invited_by,
            "status": self.status,
            "is_pending": self.is_pending,
            "is_expired": self.is_expired,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "accepted_at": self.accepted_at.isoformat() if self.accepted_at else None,
        }
        if include_token:
            data["token"] = self.token
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OrganizationInvitation":
        """Create from dictionary."""
        inv = cls(
            id=data.get("id", str(uuid4())),
            org_id=data.get("org_id", ""),
            email=data.get("email", "").lower(),
            role=data.get("role", "member"),
            token=data.get("token", secrets.token_urlsafe(32)),
            invited_by=data.get("invited_by"),
            status=data.get("status", "pending"),
        )
        for field_name in ["created_at", "expires_at", "accepted_at"]:
            if field_name in data and data[field_name]:
                value = data[field_name]
                if isinstance(value, str):
                    value = datetime.fromisoformat(value)
                setattr(inv, field_name, value)
        return inv


def generate_slug(name: str) -> str:
    """Generate URL-friendly slug from name."""
    import re
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug[:50] or "org"


__all__ = [
    "SubscriptionTier",
    "TierLimits",
    "TIER_LIMITS",
    "User",
    "Organization",
    "Subscription",
    "OrganizationInvitation",
    "hash_password",
    "verify_password",
    "needs_rehash",
    "generate_slug",
    "HAS_BCRYPT",
]
