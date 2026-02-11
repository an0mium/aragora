"""
Integration store abstract base class.

Defines the IntegrationStoreBackend ABC that all backend implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from aragora.storage.integration_models import IntegrationConfig, UserIdMapping


class IntegrationStoreBackend(ABC):
    """Abstract base for integration storage backends."""

    @abstractmethod
    async def get(
        self, integration_type: str, user_id: str = "default"
    ) -> IntegrationConfig | None:
        """Get integration configuration."""
        pass

    @abstractmethod
    async def save(self, config: IntegrationConfig) -> None:
        """Save integration configuration."""
        pass

    @abstractmethod
    async def delete(self, integration_type: str, user_id: str = "default") -> bool:
        """Delete integration configuration. Returns True if deleted."""
        pass

    @abstractmethod
    async def list_for_user(self, user_id: str = "default") -> list[IntegrationConfig]:
        """List all integrations for a user."""
        pass

    @abstractmethod
    async def list_all(self, limit: int = 1000) -> list[IntegrationConfig]:
        """List all integrations (admin use).

        Args:
            limit: Maximum number of results (default 1000 to prevent unbounded queries)
        """
        pass

    # User ID mapping methods
    @abstractmethod
    async def get_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> UserIdMapping | None:
        """Get user ID mapping for a platform."""
        pass

    @abstractmethod
    async def save_user_mapping(self, mapping: UserIdMapping) -> None:
        """Save user ID mapping."""
        pass

    @abstractmethod
    async def delete_user_mapping(
        self, email: str, platform: str, user_id: str = "default"
    ) -> bool:
        """Delete user ID mapping."""
        pass

    @abstractmethod
    async def list_user_mappings(
        self, platform: str | None = None, user_id: str = "default"
    ) -> list[UserIdMapping]:
        """List user ID mappings, optionally filtered by platform."""
        pass

    async def close(self) -> None:
        """Close connections (optional to implement)."""
        pass


__all__ = [
    "IntegrationStoreBackend",
]
