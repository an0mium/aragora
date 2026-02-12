"""Bulkhead pattern for adapter isolation."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation.

    Limits concurrent operations per adapter to prevent cascade failures.
    """

    max_concurrent_calls: int = 10
    max_wait_seconds: float = 5.0


class AdapterBulkhead:
    """
    Bulkhead pattern for isolating adapter operations.

    Limits the number of concurrent calls to an adapter to prevent
    resource exhaustion and cascade failures.

    Usage:
        bulkhead = AdapterBulkhead("continuum", max_concurrent=10)

        async with bulkhead.acquire():
            result = await adapter.operation()
    """

    def __init__(
        self,
        adapter_name: str,
        config: BulkheadConfig | None = None,
    ):
        """Initialize bulkhead.

        Args:
            adapter_name: Name of the adapter
            config: Bulkhead configuration
        """
        self.adapter_name = adapter_name
        self.config = config or BulkheadConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_calls)
        self._active_calls = 0
        self._rejected_calls = 0
        self._total_calls = 0

    @property
    def active_calls(self) -> int:
        """Get number of active calls."""
        return self._active_calls

    @property
    def available_permits(self) -> int:
        """Get number of available permits."""
        return self.config.max_concurrent_calls - self._active_calls

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Acquire a permit from the bulkhead.

        Raises:
            BulkheadFullError: If bulkhead is full and wait times out
        """
        self._total_calls += 1

        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self.config.max_wait_seconds,
            )
            if not acquired:
                self._rejected_calls += 1
                raise BulkheadFullError(
                    self.adapter_name,
                    self.config.max_concurrent_calls,
                )
        except asyncio.TimeoutError:
            self._rejected_calls += 1
            raise BulkheadFullError(
                self.adapter_name,
                self.config.max_concurrent_calls,
                f"Bulkhead full after waiting {self.config.max_wait_seconds}s",
            )

        self._active_calls += 1
        try:
            yield
        finally:
            self._active_calls -= 1
            self._semaphore.release()

    def get_stats(self) -> dict[str, Any]:
        """Get bulkhead statistics."""
        return {
            "adapter_name": self.adapter_name,
            "max_concurrent_calls": self.config.max_concurrent_calls,
            "active_calls": self._active_calls,
            "available_permits": self.available_permits,
            "total_calls": self._total_calls,
            "rejected_calls": self._rejected_calls,
            "rejection_rate": (
                self._rejected_calls / self._total_calls if self._total_calls > 0 else 0.0
            ),
        }


class BulkheadFullError(Exception):
    """Raised when bulkhead is at capacity."""

    def __init__(
        self,
        adapter_name: str,
        max_concurrent: int,
        message: str | None = None,
    ):
        self.adapter_name = adapter_name
        self.max_concurrent = max_concurrent
        super().__init__(
            message or f"Bulkhead '{adapter_name}' full (max {max_concurrent} concurrent calls)"
        )


# Global registry of adapter bulkheads
_adapter_bulkheads: dict[str, AdapterBulkhead] = {}


def get_adapter_bulkhead(
    adapter_name: str,
    config: BulkheadConfig | None = None,
) -> AdapterBulkhead:
    """Get or create a bulkhead for an adapter.

    Args:
        adapter_name: Name of the adapter
        config: Optional configuration (only used if creating new)

    Returns:
        AdapterBulkhead instance
    """
    if adapter_name not in _adapter_bulkheads:
        _adapter_bulkheads[adapter_name] = AdapterBulkhead(adapter_name, config)
    return _adapter_bulkheads[adapter_name]


def get_all_adapter_bulkhead_stats() -> dict[str, dict[str, Any]]:
    """Get statistics for all adapter bulkheads."""
    return {name: bh.get_stats() for name, bh in _adapter_bulkheads.items()}
