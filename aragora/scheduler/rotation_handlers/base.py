"""
Base rotation handler interface.

All provider-specific handlers should inherit from RotationHandler.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import logging

logger = logging.getLogger(__name__)


class RotationStatus(Enum):
    """Status of a rotation operation."""

    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"  # New secret works but old wasn't revoked
    SKIPPED = "skipped"  # Not due for rotation
    PENDING = "pending"  # In progress


@dataclass
class RotationResult:
    """Result of a secret rotation operation."""

    status: RotationStatus
    secret_id: str
    secret_type: str
    old_version: str | None = None
    new_version: str | None = None
    rotated_at: datetime = field(default_factory=datetime.utcnow)
    grace_period_ends: datetime | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "secret_id": self.secret_id,
            "secret_type": self.secret_type,
            "old_version": self.old_version,
            "new_version": self.new_version,
            "rotated_at": self.rotated_at.isoformat(),
            "grace_period_ends": (
                self.grace_period_ends.isoformat() if self.grace_period_ends else None
            ),
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


class RotationHandler(ABC):
    """
    Abstract base class for secret rotation handlers.

    Each provider (database, OAuth, API key, etc.) should implement
    a concrete handler that knows how to:
    1. Generate new credentials
    2. Validate they work
    3. Update storage
    4. Revoke old credentials
    """

    def __init__(
        self,
        grace_period_hours: int = 24,
        max_retries: int = 3,
    ):
        """
        Initialize rotation handler.

        Args:
            grace_period_hours: Hours old credentials remain valid after rotation
            max_retries: Maximum retry attempts for failed operations
        """
        self.grace_period_hours = grace_period_hours
        self.max_retries = max_retries

    @property
    @abstractmethod
    def secret_type(self) -> str:
        """Return the type of secret this handler manages."""

    @abstractmethod
    async def generate_new_credentials(
        self, secret_id: str, metadata: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate new credentials.

        Args:
            secret_id: Identifier for the secret
            metadata: Additional context (provider, region, etc.)

        Returns:
            Tuple of (new_secret_value, updated_metadata)

        Raises:
            RotationError: If generation fails
        """

    @abstractmethod
    async def validate_credentials(
        self, secret_id: str, secret_value: str, metadata: dict[str, Any]
    ) -> bool:
        """
        Validate that credentials work.

        Args:
            secret_id: Identifier for the secret
            secret_value: The credential value to test
            metadata: Additional context

        Returns:
            True if credentials are valid, False otherwise
        """

    @abstractmethod
    async def revoke_old_credentials(
        self, secret_id: str, old_value: str, metadata: dict[str, Any]
    ) -> bool:
        """
        Revoke old credentials after grace period.

        Args:
            secret_id: Identifier for the secret
            old_value: The old credential value to revoke
            metadata: Additional context

        Returns:
            True if revocation succeeded, False otherwise
        """

    async def rotate(
        self,
        secret_id: str,
        current_value: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> RotationResult:
        """
        Execute full rotation lifecycle.

        Args:
            secret_id: Identifier for the secret
            current_value: Current secret value (for revocation)
            metadata: Additional context

        Returns:
            RotationResult with status and details
        """
        metadata = metadata or {}
        old_version = metadata.get("version", "unknown")

        try:
            # Step 1: Generate new credentials
            logger.info(f"Generating new credentials for {secret_id}")
            new_value, updated_metadata = await self.generate_new_credentials(secret_id, metadata)
            new_version = updated_metadata.get(
                "version", "v" + datetime.utcnow().strftime("%Y%m%d%H%M%S")
            )

            # Step 2: Validate new credentials
            logger.info(f"Validating new credentials for {secret_id}")
            is_valid = await self.validate_credentials(secret_id, new_value, updated_metadata)

            if not is_valid:
                return RotationResult(
                    status=RotationStatus.FAILED,
                    secret_id=secret_id,
                    secret_type=self.secret_type,
                    old_version=old_version,
                    error_message="New credentials validation failed",
                )

            # Step 3: Calculate grace period
            grace_period_ends = datetime.utcnow()
            from datetime import timedelta

            grace_period_ends += timedelta(hours=self.grace_period_hours)

            # Step 4: Return success (revocation happens after grace period)
            logger.info(
                f"Rotation successful for {secret_id}, grace period ends at {grace_period_ends}"
            )

            return RotationResult(
                status=RotationStatus.SUCCESS,
                secret_id=secret_id,
                secret_type=self.secret_type,
                old_version=old_version,
                new_version=new_version,
                grace_period_ends=grace_period_ends,
                metadata={
                    "new_value": new_value,  # Caller should store this securely
                    **updated_metadata,
                },
            )

        except Exception as e:
            logger.exception(f"Rotation failed for {secret_id}: {e}")
            return RotationResult(
                status=RotationStatus.FAILED,
                secret_id=secret_id,
                secret_type=self.secret_type,
                old_version=old_version,
                error_message=str(e),
            )

    async def cleanup_expired(
        self,
        secret_id: str,
        old_value: str,
        metadata: dict[str, Any] | None = None,
    ) -> RotationResult:
        """
        Revoke old credentials after grace period has expired.

        Args:
            secret_id: Identifier for the secret
            old_value: The old credential value to revoke
            metadata: Additional context

        Returns:
            RotationResult with revocation status
        """
        metadata = metadata or {}

        try:
            success = await self.revoke_old_credentials(secret_id, old_value, metadata)

            if success:
                logger.info(f"Successfully revoked old credentials for {secret_id}")
                return RotationResult(
                    status=RotationStatus.SUCCESS,
                    secret_id=secret_id,
                    secret_type=self.secret_type,
                    metadata={"action": "revocation"},
                )
            else:
                logger.warning(f"Failed to revoke old credentials for {secret_id}")
                return RotationResult(
                    status=RotationStatus.PARTIAL,
                    secret_id=secret_id,
                    secret_type=self.secret_type,
                    error_message="Revocation failed",
                )

        except Exception as e:
            logger.exception(f"Revocation error for {secret_id}: {e}")
            return RotationResult(
                status=RotationStatus.FAILED,
                secret_id=secret_id,
                secret_type=self.secret_type,
                error_message=str(e),
            )


class RotationError(Exception):
    """Error during secret rotation."""

    def __init__(self, message: str, secret_id: str | None = None):
        super().__init__(message)
        self.secret_id = secret_id
