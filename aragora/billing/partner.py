"""
Partner API Program.

Provides infrastructure for partner integrations, API key management,
revenue sharing, and usage attribution.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PartnerTier(Enum):
    """Partner program tiers."""

    STARTER = "starter"  # Free tier, basic API access
    DEVELOPER = "developer"  # Paid tier, higher limits
    BUSINESS = "business"  # Enterprise features, priority support
    ENTERPRISE = "enterprise"  # Custom terms, dedicated support


class PartnerStatus(Enum):
    """Partner account status."""

    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"


@dataclass
class PartnerLimits:
    """Rate limits and quotas for partner tiers."""

    requests_per_minute: int
    requests_per_day: int
    debates_per_month: int
    max_agents_per_debate: int
    max_rounds: int
    webhook_endpoints: int
    revenue_share_percent: float  # Percentage of referred user spend


PARTNER_TIER_LIMITS: Dict[PartnerTier, PartnerLimits] = {
    PartnerTier.STARTER: PartnerLimits(
        requests_per_minute=60,
        requests_per_day=1000,
        debates_per_month=100,
        max_agents_per_debate=3,
        max_rounds=3,
        webhook_endpoints=1,
        revenue_share_percent=0.0,
    ),
    PartnerTier.DEVELOPER: PartnerLimits(
        requests_per_minute=300,
        requests_per_day=10000,
        debates_per_month=1000,
        max_agents_per_debate=5,
        max_rounds=5,
        webhook_endpoints=5,
        revenue_share_percent=10.0,
    ),
    PartnerTier.BUSINESS: PartnerLimits(
        requests_per_minute=1000,
        requests_per_day=100000,
        debates_per_month=10000,
        max_agents_per_debate=10,
        max_rounds=10,
        webhook_endpoints=20,
        revenue_share_percent=15.0,
    ),
    PartnerTier.ENTERPRISE: PartnerLimits(
        requests_per_minute=5000,
        requests_per_day=1000000,
        debates_per_month=100000,
        max_agents_per_debate=20,
        max_rounds=20,
        webhook_endpoints=100,
        revenue_share_percent=20.0,
    ),
}


@dataclass
class Partner:
    """Partner account."""

    partner_id: str
    name: str
    email: str
    company: Optional[str]
    tier: PartnerTier
    status: PartnerStatus
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    referral_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "partner_id": self.partner_id,
            "name": self.name,
            "email": self.email,
            "company": self.company,
            "tier": self.tier.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "webhook_url": self.webhook_url,
            "referral_code": self.referral_code,
        }


@dataclass
class APIKey:
    """Partner API key."""

    key_id: str
    partner_id: str
    key_prefix: str  # First 8 chars for identification
    key_hash: str  # SHA-256 hash of full key
    name: str
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excludes key_hash)."""
        return {
            "key_id": self.key_id,
            "partner_id": self.partner_id,
            "key_prefix": self.key_prefix,
            "name": self.name,
            "scopes": self.scopes,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "is_active": self.is_active,
        }


@dataclass
class UsageRecord:
    """Partner API usage record."""

    record_id: str
    partner_id: str
    key_id: str
    endpoint: str
    method: str
    status_code: int
    latency_ms: int
    tokens_used: int
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RevenueShare:
    """Revenue share record for referred users."""

    share_id: str
    partner_id: str
    referred_user_id: str
    period_start: datetime
    period_end: datetime
    referred_spend_usd: float
    share_percent: float
    share_amount_usd: float
    status: str  # pending, paid, cancelled
    paid_at: Optional[datetime] = None


# Available API scopes
API_SCOPES = [
    "debates:read",
    "debates:write",
    "agents:read",
    "gauntlet:run",
    "memory:read",
    "memory:write",
    "analytics:read",
    "webhooks:manage",
    "knowledge:read",
    "knowledge:write",
]


class PartnerStore:
    """Storage for partner data."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize partner store."""

        self._db_path = db_path or ":memory:"
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS partners (
                    partner_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    company TEXT,
                    tier TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata_json TEXT,
                    webhook_url TEXT,
                    webhook_secret TEXT,
                    referral_code TEXT UNIQUE
                );

                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    partner_id TEXT NOT NULL,
                    key_prefix TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    name TEXT NOT NULL,
                    scopes_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    last_used_at TEXT,
                    is_active INTEGER DEFAULT 1,
                    FOREIGN KEY (partner_id) REFERENCES partners(partner_id)
                );

                CREATE TABLE IF NOT EXISTS usage_records (
                    record_id TEXT PRIMARY KEY,
                    partner_id TEXT NOT NULL,
                    key_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    tokens_used INTEGER DEFAULT 0,
                    timestamp TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (partner_id) REFERENCES partners(partner_id)
                );

                CREATE TABLE IF NOT EXISTS revenue_shares (
                    share_id TEXT PRIMARY KEY,
                    partner_id TEXT NOT NULL,
                    referred_user_id TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    referred_spend_usd REAL NOT NULL,
                    share_percent REAL NOT NULL,
                    share_amount_usd REAL NOT NULL,
                    status TEXT NOT NULL,
                    paid_at TEXT,
                    FOREIGN KEY (partner_id) REFERENCES partners(partner_id)
                );

                CREATE INDEX IF NOT EXISTS idx_api_keys_partner ON api_keys(partner_id);
                CREATE INDEX IF NOT EXISTS idx_usage_partner_time ON usage_records(partner_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_revenue_partner_period ON revenue_shares(partner_id, period_start);
            """
            )
            conn.commit()

    def create_partner(self, partner: Partner) -> Partner:
        """Create a new partner."""
        import json

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO partners (
                    partner_id, name, email, company, tier, status,
                    created_at, updated_at, metadata_json, webhook_url,
                    webhook_secret, referral_code
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    partner.partner_id,
                    partner.name,
                    partner.email,
                    partner.company,
                    partner.tier.value,
                    partner.status.value,
                    partner.created_at.isoformat(),
                    partner.updated_at.isoformat(),
                    json.dumps(partner.metadata),
                    partner.webhook_url,
                    partner.webhook_secret,
                    partner.referral_code,
                ),
            )
            conn.commit()
        return partner

    def get_partner(self, partner_id: str) -> Optional[Partner]:
        """Get partner by ID."""
        import json

        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM partners WHERE partner_id = ?", (partner_id,)
            ).fetchone()

            if not row:
                return None

            return Partner(
                partner_id=row["partner_id"],
                name=row["name"],
                email=row["email"],
                company=row["company"],
                tier=PartnerTier(row["tier"]),
                status=PartnerStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                metadata=json.loads(row["metadata_json"] or "{}"),
                webhook_url=row["webhook_url"],
                webhook_secret=row["webhook_secret"],
                referral_code=row["referral_code"],
            )

    def get_partner_by_email(self, email: str) -> Optional[Partner]:
        """Get partner by email."""
        import json

        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM partners WHERE email = ?", (email,)).fetchone()

            if not row:
                return None

            return Partner(
                partner_id=row["partner_id"],
                name=row["name"],
                email=row["email"],
                company=row["company"],
                tier=PartnerTier(row["tier"]),
                status=PartnerStatus(row["status"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                updated_at=datetime.fromisoformat(row["updated_at"]),
                metadata=json.loads(row["metadata_json"] or "{}"),
                webhook_url=row["webhook_url"],
                webhook_secret=row["webhook_secret"],
                referral_code=row["referral_code"],
            )

    def update_partner(self, partner: Partner) -> Partner:
        """Update partner."""
        import json

        partner.updated_at = datetime.now(timezone.utc)
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE partners SET
                    name = ?, email = ?, company = ?, tier = ?, status = ?,
                    updated_at = ?, metadata_json = ?, webhook_url = ?,
                    webhook_secret = ?, referral_code = ?
                WHERE partner_id = ?
                """,
                (
                    partner.name,
                    partner.email,
                    partner.company,
                    partner.tier.value,
                    partner.status.value,
                    partner.updated_at.isoformat(),
                    json.dumps(partner.metadata),
                    partner.webhook_url,
                    partner.webhook_secret,
                    partner.referral_code,
                    partner.partner_id,
                ),
            )
            conn.commit()
        return partner

    def create_api_key(self, api_key: APIKey) -> APIKey:
        """Create API key record."""
        import json

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO api_keys (
                    key_id, partner_id, key_prefix, key_hash, name,
                    scopes_json, created_at, expires_at, last_used_at, is_active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    api_key.key_id,
                    api_key.partner_id,
                    api_key.key_prefix,
                    api_key.key_hash,
                    api_key.name,
                    json.dumps(api_key.scopes),
                    api_key.created_at.isoformat(),
                    api_key.expires_at.isoformat() if api_key.expires_at else None,
                    api_key.last_used_at.isoformat() if api_key.last_used_at else None,
                    1 if api_key.is_active else 0,
                ),
            )
            conn.commit()
        return api_key

    def get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash."""
        import json

        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)).fetchone()

            if not row:
                return None

            return APIKey(
                key_id=row["key_id"],
                partner_id=row["partner_id"],
                key_prefix=row["key_prefix"],
                key_hash=row["key_hash"],
                name=row["name"],
                scopes=json.loads(row["scopes_json"]),
                created_at=datetime.fromisoformat(row["created_at"]),
                expires_at=(
                    datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
                ),
                last_used_at=(
                    datetime.fromisoformat(row["last_used_at"]) if row["last_used_at"] else None
                ),
                is_active=bool(row["is_active"]),
            )

    def list_partner_keys(self, partner_id: str) -> List[APIKey]:
        """List all API keys for a partner."""
        import json

        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM api_keys WHERE partner_id = ? ORDER BY created_at DESC",
                (partner_id,),
            ).fetchall()

            return [
                APIKey(
                    key_id=row["key_id"],
                    partner_id=row["partner_id"],
                    key_prefix=row["key_prefix"],
                    key_hash=row["key_hash"],
                    name=row["name"],
                    scopes=json.loads(row["scopes_json"]),
                    created_at=datetime.fromisoformat(row["created_at"]),
                    expires_at=(
                        datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
                    ),
                    last_used_at=(
                        datetime.fromisoformat(row["last_used_at"]) if row["last_used_at"] else None
                    ),
                    is_active=bool(row["is_active"]),
                )
                for row in rows
            ]

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        with self._get_connection() as conn:
            cursor = conn.execute("UPDATE api_keys SET is_active = 0 WHERE key_id = ?", (key_id,))
            conn.commit()
            return cursor.rowcount > 0

    def record_usage(self, record: UsageRecord) -> None:
        """Record API usage."""
        import json

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO usage_records (
                    record_id, partner_id, key_id, endpoint, method,
                    status_code, latency_ms, tokens_used, timestamp, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.record_id,
                    record.partner_id,
                    record.key_id,
                    record.endpoint,
                    record.method,
                    record.status_code,
                    record.latency_ms,
                    record.tokens_used,
                    record.timestamp.isoformat(),
                    json.dumps(record.metadata),
                ),
            )
            conn.commit()

    def get_usage_stats(self, partner_id: str, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get usage statistics for a partner."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) as total_requests,
                    SUM(tokens_used) as total_tokens,
                    AVG(latency_ms) as avg_latency,
                    SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as errors
                FROM usage_records
                WHERE partner_id = ?
                    AND timestamp >= ?
                    AND timestamp <= ?
                """,
                (partner_id, start.isoformat(), end.isoformat()),
            ).fetchone()

            return {
                "total_requests": row["total_requests"] or 0,
                "total_tokens": row["total_tokens"] or 0,
                "avg_latency_ms": round(row["avg_latency"] or 0, 2),
                "error_count": row["errors"] or 0,
                "error_rate": (
                    (row["errors"] or 0) / row["total_requests"] if row["total_requests"] else 0
                ),
            }


class PartnerAPI:
    """Partner API management."""

    def __init__(self, store: Optional[PartnerStore] = None):
        """Initialize partner API."""
        self._store = store or PartnerStore()

    def register_partner(
        self,
        name: str,
        email: str,
        company: Optional[str] = None,
        tier: PartnerTier = PartnerTier.STARTER,
    ) -> Partner:
        """Register a new partner."""
        now = datetime.now(timezone.utc)
        partner_id = f"partner_{secrets.token_hex(12)}"
        referral_code = secrets.token_urlsafe(8).upper()

        partner = Partner(
            partner_id=partner_id,
            name=name,
            email=email,
            company=company,
            tier=tier,
            status=PartnerStatus.PENDING,
            created_at=now,
            updated_at=now,
            referral_code=referral_code,
        )

        return self._store.create_partner(partner)

    def activate_partner(self, partner_id: str) -> Optional[Partner]:
        """Activate a pending partner."""
        partner = self._store.get_partner(partner_id)
        if not partner:
            return None

        partner.status = PartnerStatus.ACTIVE
        return self._store.update_partner(partner)

    def create_api_key(
        self,
        partner_id: str,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
    ) -> tuple[APIKey, str]:
        """
        Create a new API key for a partner.

        Returns (APIKey, raw_key) - raw_key is only returned once.
        """
        partner = self._store.get_partner(partner_id)
        if not partner or partner.status != PartnerStatus.ACTIVE:
            raise ValueError("Partner not found or not active")

        # Generate secure key
        raw_key = f"ara_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_prefix = raw_key[:12]

        now = datetime.now(timezone.utc)
        expires_at = None
        if expires_in_days:
            from datetime import timedelta

            expires_at = now + timedelta(days=expires_in_days)

        # Default to all scopes for the tier
        if scopes is None:
            scopes = API_SCOPES.copy()

        api_key = APIKey(
            key_id=f"key_{secrets.token_hex(8)}",
            partner_id=partner_id,
            key_prefix=key_prefix,
            key_hash=key_hash,
            name=name,
            scopes=scopes,
            created_at=now,
            expires_at=expires_at,
            last_used_at=None,
            is_active=True,
        )

        self._store.create_api_key(api_key)
        return api_key, raw_key

    def validate_api_key(self, raw_key: str) -> Optional[tuple[APIKey, Partner]]:
        """
        Validate an API key.

        Returns (APIKey, Partner) if valid, None otherwise.
        """
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        api_key = self._store.get_api_key_by_hash(key_hash)

        if not api_key:
            return None

        if not api_key.is_active:
            return None

        if api_key.expires_at and api_key.expires_at < datetime.now(timezone.utc):
            return None

        partner = self._store.get_partner(api_key.partner_id)
        if not partner or partner.status != PartnerStatus.ACTIVE:
            return None

        return api_key, partner

    def check_rate_limit(self, partner: Partner) -> tuple[bool, Dict[str, Any]]:
        """
        Check if partner is within rate limits.

        Returns (allowed, limit_info).
        """
        limits = PARTNER_TIER_LIMITS[partner.tier]
        now = datetime.now(timezone.utc)

        # Get usage for current minute and day

        minute_start = now.replace(second=0, microsecond=0)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        minute_stats = self._store.get_usage_stats(partner.partner_id, minute_start, now)
        day_stats = self._store.get_usage_stats(partner.partner_id, day_start, now)

        limit_info = {
            "requests_this_minute": minute_stats["total_requests"],
            "requests_this_day": day_stats["total_requests"],
            "limit_per_minute": limits.requests_per_minute,
            "limit_per_day": limits.requests_per_day,
            "remaining_minute": max(0, limits.requests_per_minute - minute_stats["total_requests"]),
            "remaining_day": max(0, limits.requests_per_day - day_stats["total_requests"]),
        }

        allowed = (
            minute_stats["total_requests"] < limits.requests_per_minute
            and day_stats["total_requests"] < limits.requests_per_day
        )

        return allowed, limit_info

    def record_request(
        self,
        partner_id: str,
        key_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        latency_ms: int,
        tokens_used: int = 0,
    ) -> None:
        """Record an API request."""
        record = UsageRecord(
            record_id=f"req_{secrets.token_hex(8)}",
            partner_id=partner_id,
            key_id=key_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            timestamp=datetime.now(timezone.utc),
        )
        self._store.record_usage(record)

    def get_partner_stats(self, partner_id: str, days: int = 30) -> Dict[str, Any]:
        """Get partner statistics."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days)

        partner = self._store.get_partner(partner_id)
        if not partner:
            raise ValueError("Partner not found")

        usage = self._store.get_usage_stats(partner_id, start, now)
        keys = self._store.list_partner_keys(partner_id)
        limits = PARTNER_TIER_LIMITS[partner.tier]

        return {
            "partner": partner.to_dict(),
            "usage": usage,
            "keys": {
                "total": len(keys),
                "active": len([k for k in keys if k.is_active]),
            },
            "limits": {
                "requests_per_minute": limits.requests_per_minute,
                "requests_per_day": limits.requests_per_day,
                "debates_per_month": limits.debates_per_month,
                "revenue_share_percent": limits.revenue_share_percent,
            },
            "period": {
                "start": start.isoformat(),
                "end": now.isoformat(),
                "days": days,
            },
        }

    def generate_webhook_secret(self, partner_id: str) -> str:
        """Generate a webhook secret for a partner."""
        partner = self._store.get_partner(partner_id)
        if not partner:
            raise ValueError("Partner not found")

        secret = f"whsec_{secrets.token_urlsafe(32)}"
        partner.webhook_secret = secret
        self._store.update_partner(partner)
        return secret

    def verify_webhook_signature(
        self, partner_id: str, payload: bytes, signature: str, timestamp: str
    ) -> bool:
        """Verify a webhook signature."""
        partner = self._store.get_partner(partner_id)
        if not partner or not partner.webhook_secret:
            return False

        # Verify timestamp is recent (within 5 minutes)
        try:
            ts = int(timestamp)
            if abs(time.time() - ts) > 300:
                return False
        except ValueError:
            return False

        # Compute expected signature
        signed_payload = f"{timestamp}.{payload.decode()}"
        expected = hmac.new(
            partner.webhook_secret.encode(),
            signed_payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(f"v1={expected}", signature)


# Singleton instance
_partner_api: Optional[PartnerAPI] = None


def get_partner_api() -> PartnerAPI:
    """Get or create the partner API singleton."""
    global _partner_api
    if _partner_api is None:
        _partner_api = PartnerAPI()
    return _partner_api
