"""
Storage interface protocols for unifying persistence layers.

These protocols provide a minimal contract for storage backends so that
server, CLI, and memory layers can converge on a shared interface over time.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StorageInterface(Protocol):
    """Synchronous storage interface."""

    def save(self, key: str, data: dict[str, Any]) -> None: ...

    def get(self, key: str) -> dict[str, Any] | None: ...

    def delete(self, key: str) -> bool: ...

    def query(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...


@runtime_checkable
class AsyncStorageInterface(Protocol):
    """Asynchronous storage interface."""

    async def save(self, key: str, data: dict[str, Any]) -> None: ...

    async def get(self, key: str) -> dict[str, Any] | None: ...

    async def delete(self, key: str) -> bool: ...

    async def query(self, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...


__all__ = ["StorageInterface", "AsyncStorageInterface"]
