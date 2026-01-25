"""
Consent Management System.

Implements GDPR-compliant consent tracking:
- Purpose-based consent tracking
- Consent lifecycle management (grant, revoke, history)
- Right to be forgotten (bulk revocation)
- Data export for GDPR DSARs

Usage:
    from aragora.privacy.consent import ConsentManager, ConsentPurpose

    manager = ConsentManager()

    # Grant consent
    record = manager.grant_consent("user_123", ConsentPurpose.ANALYTICS, "v1.0")

    # Check consent
    if manager.check_consent("user_123", ConsentPurpose.ANALYTICS):
        # Process analytics...

    # Revoke consent
    manager.revoke_consent("user_123", ConsentPurpose.ANALYTICS)

    # Export all consent data (GDPR DSAR)
    export = manager.export_consent_data("user_123")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


class ConsentPurpose(str, Enum):
    """Purpose for which consent is requested."""

    DEBATE_PROCESSING = "debate_processing"  # Core debate functionality
    KNOWLEDGE_STORAGE = "knowledge_storage"  # Storing knowledge/memories
    ANALYTICS = "analytics"  # Usage analytics
    MARKETING = "marketing"  # Marketing communications
    THIRD_PARTY_SHARING = "third_party_sharing"  # Sharing with third parties
    AI_TRAINING = "ai_training"  # Using data for AI improvement
    PERSONALIZATION = "personalization"  # Personalized experience
    RESEARCH = "research"  # Research purposes


class ConsentStatus(str, Enum):
    """Status of a consent record."""

    ACTIVE = "active"  # Consent is currently valid
    REVOKED = "revoked"  # Consent has been revoked
    EXPIRED = "expired"  # Consent has expired


@dataclass
class ConsentRecord:
    """
    Record of consent given or revoked by a user.

    Attributes:
        id: Unique record identifier
        user_id: User who gave/revoked consent
        purpose: Purpose for which consent was given
        granted: Whether consent was granted
        granted_at: When consent was granted
        revoked_at: When consent was revoked (if applicable)
        version: Policy version consent was given for
        ip_address: IP address at time of consent
        user_agent: Browser/client info at time of consent
        status: Current status of the consent
        metadata: Additional metadata
    """

    id: str
    user_id: str
    purpose: ConsentPurpose
    granted: bool
    granted_at: datetime | None
    version: str
    revoked_at: datetime | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    status: ConsentStatus = ConsentStatus.ACTIVE
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "purpose": self.purpose.value,
            "granted": self.granted,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "version": self.version,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "status": self.status.value,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConsentRecord:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            purpose=ConsentPurpose(data["purpose"]),
            granted=data["granted"],
            granted_at=(
                datetime.fromisoformat(data["granted_at"]) if data.get("granted_at") else None
            ),
            revoked_at=(
                datetime.fromisoformat(data["revoked_at"]) if data.get("revoked_at") else None
            ),
            version=data["version"],
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            status=ConsentStatus(data.get("status", "active")),
            expires_at=(
                datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
            ),
            metadata=data.get("metadata", {}),
        )

    @property
    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if self.status != ConsentStatus.ACTIVE:
            return False
        if not self.granted:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True


@dataclass
class ConsentExport:
    """Exported consent data for GDPR DSARs."""

    user_id: str
    records: list[ConsentRecord]
    exported_at: datetime
    export_format: str = "json"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "records": [r.to_dict() for r in self.records],
            "exported_at": self.exported_at.isoformat(),
            "export_format": self.export_format,
            "record_count": len(self.records),
        }

    def to_json(self, pretty: bool = True) -> str:
        """Export as JSON string."""
        indent = 2 if pretty else None
        return json.dumps(self.to_dict(), indent=indent)


class ConsentStore:
    """
    Persistence layer for consent records.

    Uses a file-based JSON store by default. Can be subclassed
    for database storage.
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        """
        Initialize consent store.

        Args:
            storage_path: Path to store consent data (default: ~/.aragora/consent.json)
        """
        if storage_path is None:
            storage_path = Path.home() / ".aragora" / "consent.json"
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        self._records: dict[str, dict[str, ConsentRecord]] = {}
        self._load()

    def save(self, record: ConsentRecord) -> None:
        """Save a consent record."""
        if record.user_id not in self._records:
            self._records[record.user_id] = {}

        self._records[record.user_id][record.purpose.value] = record
        self._persist()

        logger.debug(
            "Saved consent record: user=%s, purpose=%s, granted=%s",
            record.user_id,
            record.purpose.value,
            record.granted,
        )

    def get(self, user_id: str, purpose: ConsentPurpose) -> ConsentRecord | None:
        """Get consent record for a user and purpose."""
        user_records = self._records.get(user_id, {})
        return user_records.get(purpose.value)

    def get_all_for_user(self, user_id: str) -> list[ConsentRecord]:
        """Get all consent records for a user."""
        user_records = self._records.get(user_id, {})
        return list(user_records.values())

    def delete_all_for_user(self, user_id: str) -> int:
        """Delete all consent records for a user (right to be forgotten)."""
        if user_id not in self._records:
            return 0

        count = len(self._records[user_id])
        del self._records[user_id]
        self._persist()

        logger.info("Deleted %d consent records for user: %s", count, user_id)
        return count

    def get_users_with_consent(self, purpose: ConsentPurpose) -> list[str]:
        """Get all users who have granted consent for a purpose."""
        users = []
        for user_id, user_records in self._records.items():
            record = user_records.get(purpose.value)
            if record and record.is_valid:
                users.append(user_id)
        return users

    def _load(self) -> None:
        """Load records from storage."""
        if not self.storage_path.exists():
            self._records = {}
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            self._records = {}
            for user_id, user_data in data.get("records", {}).items():
                self._records[user_id] = {
                    purpose: ConsentRecord.from_dict(record_data)
                    for purpose, record_data in user_data.items()
                }

            logger.debug("Loaded %d users from consent store", len(self._records))

        except Exception as e:
            logger.error("Failed to load consent store: %s", e)
            self._records = {}

    def _persist(self) -> None:
        """Persist records to storage."""
        try:
            data = {
                "records": {
                    user_id: {purpose: record.to_dict() for purpose, record in user_records.items()}
                    for user_id, user_records in self._records.items()
                },
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error("Failed to persist consent store: %s", e)


class ConsentManager:
    """
    Manages consent lifecycle for users.

    Provides methods for:
    - Granting consent
    - Revoking consent
    - Checking consent status
    - Consent history
    - GDPR data export
    - Right to be forgotten
    """

    def __init__(self, store: ConsentStore | None = None) -> None:
        """
        Initialize consent manager.

        Args:
            store: Consent store to use (default: creates FileConsentStore)
        """
        self.store = store or ConsentStore()
        self._history: dict[str, list[ConsentRecord]] = {}

    def grant_consent(
        self,
        user_id: str,
        purpose: ConsentPurpose,
        version: str,
        ip_address: str | None = None,
        user_agent: str | None = None,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConsentRecord:
        """
        Grant consent for a specific purpose.

        Args:
            user_id: User granting consent
            purpose: Purpose for consent
            version: Policy version consented to
            ip_address: IP address at time of consent
            user_agent: Browser/client info
            expires_at: When consent expires (None = never)
            metadata: Additional metadata

        Returns:
            Created ConsentRecord
        """
        record = ConsentRecord(
            id=str(uuid4()),
            user_id=user_id,
            purpose=purpose,
            granted=True,
            granted_at=datetime.now(timezone.utc),
            version=version,
            ip_address=ip_address,
            user_agent=user_agent,
            status=ConsentStatus.ACTIVE,
            expires_at=expires_at,
            metadata=metadata or {},
        )

        self.store.save(record)
        self._add_to_history(user_id, record)

        logger.info(
            "Consent granted: user=%s, purpose=%s, version=%s",
            user_id,
            purpose.value,
            version,
        )

        return record

    def revoke_consent(
        self,
        user_id: str,
        purpose: ConsentPurpose,
    ) -> ConsentRecord | None:
        """
        Revoke consent for a specific purpose.

        Args:
            user_id: User revoking consent
            purpose: Purpose to revoke

        Returns:
            Updated ConsentRecord or None if no consent existed
        """
        record = self.store.get(user_id, purpose)
        if record is None:
            logger.warning("No consent to revoke: user=%s, purpose=%s", user_id, purpose.value)
            return None

        record.granted = False
        record.revoked_at = datetime.now(timezone.utc)
        record.status = ConsentStatus.REVOKED

        self.store.save(record)
        self._add_to_history(user_id, record)

        logger.info("Consent revoked: user=%s, purpose=%s", user_id, purpose.value)

        return record

    def check_consent(self, user_id: str, purpose: ConsentPurpose) -> bool:
        """
        Check if user has valid consent for a purpose.

        Args:
            user_id: User to check
            purpose: Purpose to check

        Returns:
            True if user has valid consent
        """
        record = self.store.get(user_id, purpose)
        if record is None:
            return False

        return record.is_valid

    def get_consent_status(self, user_id: str, purpose: ConsentPurpose) -> ConsentRecord | None:
        """
        Get the current consent status for a user and purpose.

        Args:
            user_id: User to check
            purpose: Purpose to check

        Returns:
            Current ConsentRecord or None
        """
        return self.store.get(user_id, purpose)

    def get_all_consents(self, user_id: str) -> list[ConsentRecord]:
        """
        Get all consent records for a user.

        Args:
            user_id: User to get consents for

        Returns:
            List of ConsentRecords
        """
        return self.store.get_all_for_user(user_id)

    def get_consent_history(self, user_id: str) -> list[ConsentRecord]:
        """
        Get consent change history for a user.

        Args:
            user_id: User to get history for

        Returns:
            List of ConsentRecords in chronological order
        """
        return self._history.get(user_id, [])

    def get_users_with_consent(self, purpose: ConsentPurpose) -> list[str]:
        """
        Get all users who have granted consent for a purpose.

        Args:
            purpose: Purpose to check

        Returns:
            List of user IDs with valid consent
        """
        return self.store.get_users_with_consent(purpose)

    def bulk_revoke_for_user(self, user_id: str) -> int:
        """
        Revoke all consents for a user (right to be forgotten).

        Args:
            user_id: User to revoke all consents for

        Returns:
            Number of consents revoked
        """
        records = self.store.get_all_for_user(user_id)
        revoked_count = 0

        for record in records:
            if record.granted:
                self.revoke_consent(user_id, record.purpose)
                revoked_count += 1

        logger.info("Bulk revoked %d consents for user: %s", revoked_count, user_id)

        return revoked_count

    def delete_user_data(self, user_id: str) -> int:
        """
        Delete all consent data for a user (GDPR right to erasure).

        Args:
            user_id: User to delete data for

        Returns:
            Number of records deleted
        """
        # First revoke all active consents
        self.bulk_revoke_for_user(user_id)

        # Then delete all records
        count = self.store.delete_all_for_user(user_id)

        # Clear history
        if user_id in self._history:
            del self._history[user_id]

        logger.info("Deleted all consent data for user: %s", user_id)

        return count

    def export_consent_data(self, user_id: str) -> ConsentExport:
        """
        Export all consent data for a user (GDPR data export).

        Args:
            user_id: User to export data for

        Returns:
            ConsentExport with all user's consent records
        """
        records = self.store.get_all_for_user(user_id)

        export = ConsentExport(
            user_id=user_id,
            records=records,
            exported_at=datetime.now(timezone.utc),
        )

        logger.info("Exported consent data for user: %s (%d records)", user_id, len(records))

        return export

    def verify_consent(
        self,
        user_id: str,
        required_purposes: list[ConsentPurpose],
    ) -> tuple[bool, list[ConsentPurpose]]:
        """
        Verify user has consent for all required purposes.

        Args:
            user_id: User to check
            required_purposes: List of required consents

        Returns:
            Tuple of (all_consented, missing_purposes)
        """
        missing = []

        for purpose in required_purposes:
            if not self.check_consent(user_id, purpose):
                missing.append(purpose)

        return len(missing) == 0, missing

    def _add_to_history(self, user_id: str, record: ConsentRecord) -> None:
        """Add a record to the history."""
        if user_id not in self._history:
            self._history[user_id] = []

        # Create a copy for history
        history_record = ConsentRecord(
            id=record.id,
            user_id=record.user_id,
            purpose=record.purpose,
            granted=record.granted,
            granted_at=record.granted_at,
            revoked_at=record.revoked_at,
            version=record.version,
            ip_address=record.ip_address,
            user_agent=record.user_agent,
            status=record.status,
            expires_at=record.expires_at,
            metadata=record.metadata.copy(),
        )

        self._history[user_id].append(history_record)


# Global consent manager instance
_consent_manager: ConsentManager | None = None


def get_consent_manager() -> ConsentManager:
    """Get or create the global consent manager."""
    global _consent_manager
    if _consent_manager is None:
        _consent_manager = ConsentManager()
    return _consent_manager


def set_consent_manager(manager: ConsentManager) -> None:
    """Set the global consent manager."""
    global _consent_manager
    _consent_manager = manager


__all__ = [
    "ConsentExport",
    "ConsentManager",
    "ConsentPurpose",
    "ConsentRecord",
    "ConsentStatus",
    "ConsentStore",
    "get_consent_manager",
    "set_consent_manager",
]
