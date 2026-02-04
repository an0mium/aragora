"""Abstract base class for unified inbox storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class UnifiedInboxStoreBackend(ABC):
    """Abstract base for unified inbox storage backends."""

    @abstractmethod
    async def save_account(self, tenant_id: str, account: dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def get_account(self, tenant_id: str, account_id: str) -> Optional[dict[str, Any]]:
        pass

    @abstractmethod
    async def list_accounts(self, tenant_id: str) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    async def delete_account(self, tenant_id: str, account_id: str) -> bool:
        pass

    @abstractmethod
    async def update_account_fields(
        self, tenant_id: str, account_id: str, updates: dict[str, Any]
    ) -> None:
        pass

    @abstractmethod
    async def increment_account_counts(
        self,
        tenant_id: str,
        account_id: str,
        total_delta: int = 0,
        unread_delta: int = 0,
        sync_error_delta: int = 0,
    ) -> None:
        pass

    @abstractmethod
    async def save_message(self, tenant_id: str, message: dict[str, Any]) -> tuple[str, bool]:
        pass

    @abstractmethod
    async def get_message(self, tenant_id: str, message_id: str) -> Optional[dict[str, Any]]:
        pass

    @abstractmethod
    async def list_messages(
        self,
        tenant_id: str,
        limit: int | None = None,
        offset: int = 0,
        priority_tier: str | None = None,
        account_id: str | None = None,
        unread_only: bool = False,
        search: str | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        pass

    @abstractmethod
    async def delete_message(self, tenant_id: str, message_id: str) -> bool:
        pass

    @abstractmethod
    async def update_message_flags(
        self,
        tenant_id: str,
        message_id: str,
        is_read: bool | None = None,
        is_starred: bool | None = None,
    ) -> bool:
        pass

    @abstractmethod
    async def update_message_triage(
        self,
        tenant_id: str,
        message_id: str,
        triage_action: str | None,
        triage_rationale: str | None,
    ) -> None:
        pass

    @abstractmethod
    async def save_triage_result(self, tenant_id: str, triage: dict[str, Any]) -> None:
        pass

    @abstractmethod
    async def get_triage_result(self, tenant_id: str, message_id: str) -> Optional[dict[str, Any]]:
        pass
