"""
Token Rotation Policy Enforcement.

Provides automatic token rotation policies for enterprise security:
- Rotation after N uses
- Rotation after time period
- Suspicious activity triggered rotation
- Token binding validation (IP, user agent)

Usage:
    from aragora.auth.token_rotation import get_rotation_manager

    manager = get_rotation_manager()

    # Check if rotation is required
    if manager.requires_rotation(user_id, token_jti):
        new_tokens = await refresh_tokens(...)

    # Track token usage
    manager.record_usage(user_id, token_jti, ip_address, user_agent)

    # Check for suspicious activity
    if manager.is_suspicious(user_id, token_jti, ip_address):
        await revoke_token(token_jti)
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RotationReason(str, Enum):
    """Reasons for token rotation."""

    MAX_USES_EXCEEDED = "max_uses_exceeded"
    TIME_BASED = "time_based"
    IP_CHANGE = "ip_change"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ADMIN_REQUEST = "admin_request"
    SECURITY_POLICY = "security_policy"


@dataclass
class TokenUsageRecord:
    """Record of a token's usage."""

    token_jti: str
    user_id: str
    first_used: float  # Unix timestamp
    last_used: float  # Unix timestamp
    use_count: int = 0
    ip_addresses: Set[str] = field(default_factory=set)
    user_agents: Set[str] = field(default_factory=set)
    locations: Set[str] = field(default_factory=set)  # Geo locations if available
    suspicious_flags: List[str] = field(default_factory=list)

    def add_usage(
        self,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        location: Optional[str] = None,
    ) -> None:
        """Record a usage event."""
        self.use_count += 1
        self.last_used = time.time()

        if ip_address:
            self.ip_addresses.add(ip_address)
        if user_agent:
            # Hash user agent for privacy
            ua_hash = hashlib.sha256(user_agent.encode()).hexdigest()[:16]
            self.user_agents.add(ua_hash)
        if location:
            self.locations.add(location)


@dataclass
class RotationPolicy:
    """Token rotation policy configuration."""

    # Usage-based rotation
    max_uses: int = 100  # Rotate after N uses (0 = disabled)
    max_uses_per_hour: int = 50  # Rate limit per hour

    # Time-based rotation
    max_age_seconds: int = 86400  # Rotate after 24 hours (0 = disabled)
    idle_rotation_seconds: int = 3600  # Rotate if idle > 1 hour (0 = disabled)

    # IP-based security
    allow_ip_change: bool = True  # Allow token use from different IPs
    max_ips_per_token: int = 5  # Max IPs per token before forced rotation
    ip_change_requires_rotation: bool = False  # Force rotation on IP change

    # Suspicious activity thresholds
    max_failed_validations: int = 5  # Max failures before flag
    rapid_use_threshold: int = 10  # Uses per second threshold
    geo_velocity_threshold_km: float = 500  # km/hour for impossible travel

    # Binding
    bind_to_ip: bool = False  # Strict IP binding (one IP only)
    bind_to_user_agent: bool = False  # Strict user agent binding

    @classmethod
    def strict(cls) -> "RotationPolicy":
        """Create a strict security policy."""
        return cls(
            max_uses=50,
            max_uses_per_hour=30,
            max_age_seconds=3600,  # 1 hour
            idle_rotation_seconds=900,  # 15 minutes
            allow_ip_change=False,
            max_ips_per_token=1,
            ip_change_requires_rotation=True,
            bind_to_ip=True,
            bind_to_user_agent=True,
        )

    @classmethod
    def standard(cls) -> "RotationPolicy":
        """Create a standard security policy."""
        return cls()

    @classmethod
    def relaxed(cls) -> "RotationPolicy":
        """Create a relaxed policy for development."""
        return cls(
            max_uses=1000,
            max_age_seconds=604800,  # 7 days
            idle_rotation_seconds=0,  # Disabled
            allow_ip_change=True,
            max_ips_per_token=20,
            ip_change_requires_rotation=False,
        )


@dataclass
class RotationCheckResult:
    """Result of checking if rotation is required."""

    requires_rotation: bool
    reason: Optional[RotationReason] = None
    details: str = ""
    is_suspicious: bool = False
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "requires_rotation": self.requires_rotation,
            "reason": self.reason.value if self.reason else None,
            "details": self.details,
            "is_suspicious": self.is_suspicious,
            "recommendations": self.recommendations,
        }


class TokenRotationManager:
    """
    Manages token rotation policies and usage tracking.

    Thread-safe implementation with configurable policies.
    """

    def __init__(
        self,
        policy: Optional[RotationPolicy] = None,
        on_rotation_required: Optional[Callable[[str, str, RotationReason], None]] = None,
        on_suspicious_activity: Optional[Callable[[str, str, List[str]], None]] = None,
    ):
        """Initialize the rotation manager.

        Args:
            policy: Rotation policy to enforce
            on_rotation_required: Callback when rotation is required
            on_suspicious_activity: Callback when suspicious activity detected
        """
        self.policy = policy or RotationPolicy.standard()
        self.on_rotation_required = on_rotation_required
        self.on_suspicious_activity = on_suspicious_activity

        # Token usage tracking: token_jti -> TokenUsageRecord
        self._usage: Dict[str, TokenUsageRecord] = {}
        self._lock = threading.Lock()

        # Failed validation tracking: token_jti -> count
        self._failed_validations: Dict[str, int] = {}

        # Recent usage times for rate limiting: token_jti -> list of timestamps
        self._recent_uses: Dict[str, List[float]] = {}

        # Cleanup tracking
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes

    def record_usage(
        self,
        user_id: str,
        token_jti: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        location: Optional[str] = None,
    ) -> RotationCheckResult:
        """Record token usage and check if rotation is needed.

        Args:
            user_id: User identifier
            token_jti: JWT token identifier
            ip_address: Client IP address
            user_agent: Client user agent string
            location: Geo location if available

        Returns:
            RotationCheckResult indicating if rotation is needed
        """
        now = time.time()

        with self._lock:
            self._maybe_cleanup()

            # Get or create usage record
            if token_jti not in self._usage:
                self._usage[token_jti] = TokenUsageRecord(
                    token_jti=token_jti,
                    user_id=user_id,
                    first_used=now,
                    last_used=now,
                )

            record = self._usage[token_jti]
            previous_ips = record.ip_addresses.copy()
            record.add_usage(ip_address, user_agent, location)

            # Track recent uses for rate limiting
            if token_jti not in self._recent_uses:
                self._recent_uses[token_jti] = []
            self._recent_uses[token_jti].append(now)
            # Keep only last minute
            self._recent_uses[token_jti] = [t for t in self._recent_uses[token_jti] if now - t < 60]

        # Check rotation requirements
        return self._check_rotation(record, ip_address, previous_ips)

    def requires_rotation(
        self,
        user_id: str,
        token_jti: str,
    ) -> RotationCheckResult:
        """Check if a token requires rotation without recording usage.

        Args:
            user_id: User identifier
            token_jti: JWT token identifier

        Returns:
            RotationCheckResult indicating if rotation is needed
        """
        with self._lock:
            record = self._usage.get(token_jti)
            if not record:
                return RotationCheckResult(
                    requires_rotation=False,
                    details="No usage record found",
                )
            return self._check_rotation(record, None, set())

    def record_failed_validation(self, token_jti: str) -> bool:
        """Record a failed token validation.

        Args:
            token_jti: JWT token identifier

        Returns:
            True if token should be revoked due to excessive failures
        """
        with self._lock:
            if token_jti not in self._failed_validations:
                self._failed_validations[token_jti] = 0
            self._failed_validations[token_jti] += 1

            if self._failed_validations[token_jti] >= self.policy.max_failed_validations:
                record = self._usage.get(token_jti)
                if record:
                    record.suspicious_flags.append("excessive_failed_validations")

                    if self.on_suspicious_activity:
                        self.on_suspicious_activity(
                            record.user_id,
                            token_jti,
                            record.suspicious_flags,
                        )
                return True
            return False

    def is_suspicious(
        self,
        user_id: str,
        token_jti: str,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Check if token usage is suspicious.

        Args:
            user_id: User identifier
            token_jti: JWT token identifier
            ip_address: Current IP address

        Returns:
            True if activity is suspicious
        """
        with self._lock:
            record = self._usage.get(token_jti)
            if not record:
                return False

            # Check for suspicious flags
            if record.suspicious_flags:
                return True

            # Check rate limiting
            recent = self._recent_uses.get(token_jti, [])
            if len(recent) > self.policy.rapid_use_threshold:
                # More than threshold uses in last second
                one_second_ago = time.time() - 1
                recent_second = [t for t in recent if t > one_second_ago]
                if len(recent_second) > self.policy.rapid_use_threshold:
                    record.suspicious_flags.append("rapid_use_detected")
                    return True

            # Check excessive IP diversity
            if len(record.ip_addresses) > self.policy.max_ips_per_token:
                record.suspicious_flags.append("excessive_ip_diversity")
                return True

            return False

    def get_usage_stats(self, token_jti: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for a token.

        Args:
            token_jti: JWT token identifier

        Returns:
            Usage statistics or None if not found
        """
        with self._lock:
            record = self._usage.get(token_jti)
            if not record:
                return None

            return {
                "token_jti": record.token_jti[:8] + "...",
                "user_id": record.user_id,
                "first_used": datetime.fromtimestamp(record.first_used, timezone.utc).isoformat(),
                "last_used": datetime.fromtimestamp(record.last_used, timezone.utc).isoformat(),
                "use_count": record.use_count,
                "unique_ips": len(record.ip_addresses),
                "unique_user_agents": len(record.user_agents),
                "suspicious_flags": record.suspicious_flags,
                "age_seconds": time.time() - record.first_used,
            }

    def clear_token(self, token_jti: str) -> None:
        """Clear usage tracking for a token (on revocation).

        Args:
            token_jti: JWT token identifier
        """
        with self._lock:
            self._usage.pop(token_jti, None)
            self._failed_validations.pop(token_jti, None)
            self._recent_uses.pop(token_jti, None)

    def _check_rotation(
        self,
        record: TokenUsageRecord,
        current_ip: Optional[str],
        previous_ips: Set[str],
    ) -> RotationCheckResult:
        """Check if rotation is required based on policy."""
        now = time.time()
        recommendations: List[str] = []

        # Check max uses
        if self.policy.max_uses > 0 and record.use_count >= self.policy.max_uses:
            return RotationCheckResult(
                requires_rotation=True,
                reason=RotationReason.MAX_USES_EXCEEDED,
                details=f"Token used {record.use_count} times (max: {self.policy.max_uses})",
            )

        # Check age
        age = now - record.first_used
        if self.policy.max_age_seconds > 0 and age >= self.policy.max_age_seconds:
            return RotationCheckResult(
                requires_rotation=True,
                reason=RotationReason.TIME_BASED,
                details=f"Token age {age:.0f}s exceeds max {self.policy.max_age_seconds}s",
            )

        # Check idle time
        idle = now - record.last_used
        if self.policy.idle_rotation_seconds > 0 and idle >= self.policy.idle_rotation_seconds:
            return RotationCheckResult(
                requires_rotation=True,
                reason=RotationReason.TIME_BASED,
                details=f"Token idle {idle:.0f}s exceeds max {self.policy.idle_rotation_seconds}s",
            )

        # Check IP change
        if current_ip and previous_ips:
            if current_ip not in previous_ips:
                if self.policy.ip_change_requires_rotation:
                    return RotationCheckResult(
                        requires_rotation=True,
                        reason=RotationReason.IP_CHANGE,
                        details=f"IP changed from {previous_ips} to {current_ip}",
                        is_suspicious=not self.policy.allow_ip_change,
                    )

                if not self.policy.allow_ip_change:
                    return RotationCheckResult(
                        requires_rotation=True,
                        reason=RotationReason.SUSPICIOUS_ACTIVITY,
                        details=f"IP change not allowed: {previous_ips} -> {current_ip}",
                        is_suspicious=True,
                    )

                if len(record.ip_addresses) >= self.policy.max_ips_per_token:
                    return RotationCheckResult(
                        requires_rotation=True,
                        reason=RotationReason.SUSPICIOUS_ACTIVITY,
                        details=f"Max IPs ({self.policy.max_ips_per_token}) exceeded",
                        is_suspicious=True,
                    )

        # Check suspicious flags
        if record.suspicious_flags:
            return RotationCheckResult(
                requires_rotation=True,
                reason=RotationReason.SUSPICIOUS_ACTIVITY,
                details=f"Suspicious activity: {', '.join(record.suspicious_flags)}",
                is_suspicious=True,
            )

        # Suggest rotation if approaching limits
        if self.policy.max_uses > 0 and record.use_count >= self.policy.max_uses * 0.8:
            recommendations.append(
                f"Token at {record.use_count}/{self.policy.max_uses} uses, consider rotating"
            )

        if self.policy.max_age_seconds > 0 and age >= self.policy.max_age_seconds * 0.8:
            recommendations.append(
                f"Token age {age:.0f}s approaching max {self.policy.max_age_seconds}s"
            )

        return RotationCheckResult(
            requires_rotation=False,
            recommendations=recommendations,
        )

    def _maybe_cleanup(self) -> None:
        """Periodic cleanup of old records."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        stale_threshold = 86400 * 7  # 7 days

        # Clean up old usage records
        to_remove = [
            jti for jti, record in self._usage.items() if now - record.last_used > stale_threshold
        ]

        for jti in to_remove:
            del self._usage[jti]
            self._failed_validations.pop(jti, None)
            self._recent_uses.pop(jti, None)

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} stale token records")


# Singleton instance
_rotation_manager: Optional[TokenRotationManager] = None
_manager_lock = threading.Lock()


def get_rotation_manager(
    policy: Optional[RotationPolicy] = None,
) -> TokenRotationManager:
    """Get the global token rotation manager."""
    global _rotation_manager
    if _rotation_manager is None:
        with _manager_lock:
            if _rotation_manager is None:
                # Use policy from environment or default
                env_policy = os.getenv("ARAGORA_TOKEN_ROTATION_POLICY", "standard")
                if policy is None:
                    if env_policy == "strict":
                        policy = RotationPolicy.strict()
                    elif env_policy == "relaxed":
                        policy = RotationPolicy.relaxed()
                    else:
                        policy = RotationPolicy.standard()
                _rotation_manager = TokenRotationManager(policy=policy)
    return _rotation_manager


def reset_rotation_manager() -> None:
    """Reset the rotation manager (for testing)."""
    global _rotation_manager
    with _manager_lock:
        _rotation_manager = None


__all__ = [
    "RotationReason",
    "TokenUsageRecord",
    "RotationPolicy",
    "RotationCheckResult",
    "TokenRotationManager",
    "get_rotation_manager",
    "reset_rotation_manager",
]
