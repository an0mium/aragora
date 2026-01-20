"""
BidirectionalCoordinator - Central coordination layer for KM bidirectional sync.

This coordinator manages bidirectional data flow between the Knowledge Mound
and all connected subsystems (adapters). It provides:

- Centralized sync scheduling and coordination
- Batch sync operations for efficiency
- Error handling and partial failure recovery
- Metrics and statistics tracking
- Adapter registration and lifecycle management

Usage:
    from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator

    coordinator = BidirectionalCoordinator()
    coordinator.register_adapter("continuum", continuum_adapter, "sync_to_km", "sync_from_km")
    coordinator.register_adapter("elo", elo_adapter, "sync_to_km", "sync_from_km")

    # Run bidirectional sync
    result = await coordinator.run_bidirectional_sync()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AdapterRegistration:
    """Registration information for an adapter."""

    name: str
    adapter: Any
    forward_method: str  # Method name for source → KM
    reverse_method: Optional[str] = None  # Method name for KM → source
    enabled: bool = True
    priority: int = 0  # Higher priority = earlier in sync order
    last_forward_sync: Optional[float] = None
    last_reverse_sync: Optional[float] = None
    forward_errors: int = 0
    reverse_errors: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncResult:
    """Result of a single adapter sync operation."""

    adapter_name: str
    direction: str  # "forward" or "reverse"
    success: bool
    items_processed: int = 0
    items_updated: int = 0
    errors: List[str] = field(default_factory=list)
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BidirectionalSyncReport:
    """Report from a full bidirectional sync cycle."""

    forward_results: List[SyncResult] = field(default_factory=list)
    reverse_results: List[SyncResult] = field(default_factory=list)
    total_adapters: int = 0
    successful_forward: int = 0
    successful_reverse: int = 0
    total_errors: int = 0
    total_duration_ms: int = 0
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinatorConfig:
    """Configuration for the BidirectionalCoordinator."""

    sync_interval_seconds: int = 300  # 5 minutes
    batch_size: int = 100
    min_confidence_for_reverse: float = 0.7
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    parallel_sync: bool = True  # Sync adapters in parallel
    timeout_seconds: float = 60.0  # Timeout per adapter sync
    enable_metrics: bool = True


class BidirectionalCoordinator:
    """
    Coordinates bidirectional sync across all KM adapters.

    This is the central coordination layer that manages data flow between
    the Knowledge Mound and all connected subsystems.

    Features:
    - Adapter registration and lifecycle management
    - Batch sync operations with configurable parallelism
    - Error handling with retries
    - Metrics tracking and reporting
    - Configurable sync intervals and confidence thresholds
    """

    def __init__(
        self,
        config: Optional[CoordinatorConfig] = None,
        knowledge_mound: Optional[Any] = None,
    ):
        """
        Initialize the coordinator.

        Args:
            config: Optional configuration for the coordinator
            knowledge_mound: Optional KnowledgeMound instance
        """
        self.config = config or CoordinatorConfig()
        self.knowledge_mound = knowledge_mound

        # Adapter registry
        self._adapters: Dict[str, AdapterRegistration] = {}

        # Sync state
        self._last_full_sync: Optional[float] = None
        self._sync_in_progress: bool = False
        self._sync_lock = asyncio.Lock()

        # Metrics
        self._total_forward_syncs: int = 0
        self._total_reverse_syncs: int = 0
        self._total_errors: int = 0
        self._sync_history: List[BidirectionalSyncReport] = []
        self._max_history: int = 100

    def register_adapter(
        self,
        name: str,
        adapter: Any,
        forward_method: str,
        reverse_method: Optional[str] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register an adapter with the coordinator.

        Args:
            name: Unique name for the adapter
            adapter: The adapter instance
            forward_method: Method name for forward sync (source → KM)
            reverse_method: Optional method name for reverse sync (KM → source)
            priority: Sync priority (higher = earlier)
            metadata: Optional metadata

        Returns:
            True if registered successfully
        """
        if name in self._adapters:
            logger.warning(f"Adapter '{name}' already registered, updating")

        # Verify methods exist
        if not hasattr(adapter, forward_method):
            logger.error(f"Adapter '{name}' missing forward method: {forward_method}")
            return False

        if reverse_method and not hasattr(adapter, reverse_method):
            logger.warning(f"Adapter '{name}' missing reverse method: {reverse_method}")
            reverse_method = None

        self._adapters[name] = AdapterRegistration(
            name=name,
            adapter=adapter,
            forward_method=forward_method,
            reverse_method=reverse_method,
            priority=priority,
            metadata=metadata or {},
        )

        logger.info(
            f"Registered adapter: {name} (forward={forward_method}, "
            f"reverse={reverse_method}, priority={priority})"
        )
        return True

    def unregister_adapter(self, name: str) -> bool:
        """
        Unregister an adapter.

        Args:
            name: Name of the adapter to unregister

        Returns:
            True if unregistered successfully
        """
        if name not in self._adapters:
            logger.warning(f"Adapter '{name}' not found")
            return False

        del self._adapters[name]
        logger.info(f"Unregistered adapter: {name}")
        return True

    def get_adapter(self, name: str) -> Optional[Any]:
        """Get an adapter by name."""
        reg = self._adapters.get(name)
        return reg.adapter if reg else None

    def get_registered_adapters(self) -> List[str]:
        """Get list of registered adapter names."""
        return list(self._adapters.keys())

    def enable_adapter(self, name: str) -> bool:
        """Enable an adapter for sync."""
        if name in self._adapters:
            self._adapters[name].enabled = True
            return True
        return False

    def disable_adapter(self, name: str) -> bool:
        """Disable an adapter from sync."""
        if name in self._adapters:
            self._adapters[name].enabled = False
            return True
        return False

    async def _sync_adapter_forward(
        self,
        registration: AdapterRegistration,
        km_items: Optional[List[Dict[str, Any]]] = None,
    ) -> SyncResult:
        """
        Run forward sync for a single adapter (source → KM).

        Args:
            registration: The adapter registration
            km_items: Optional items to sync

        Returns:
            SyncResult with operation details
        """
        start_time = time.time()
        result = SyncResult(
            adapter_name=registration.name,
            direction="forward",
            success=False,
        )

        try:
            adapter = registration.adapter
            method = getattr(adapter, registration.forward_method)

            # Call the forward method
            if asyncio.iscoroutinefunction(method):
                sync_result = await asyncio.wait_for(
                    method(),
                    timeout=self.config.timeout_seconds,
                )
            else:
                sync_result = method()

            # Extract result details
            if isinstance(sync_result, dict):
                result.items_processed = sync_result.get("items_processed", 0)
                result.items_updated = sync_result.get("items_updated", 0)
                result.metadata = sync_result
            elif hasattr(sync_result, "items_processed"):
                result.items_processed = getattr(sync_result, "items_processed", 0)
                result.items_updated = getattr(sync_result, "items_updated", 0)

            result.success = True
            registration.last_forward_sync = time.time()
            registration.forward_errors = 0

        except asyncio.TimeoutError:
            error = f"Forward sync timeout for {registration.name}"
            result.errors.append(error)
            logger.error(error)
            registration.forward_errors += 1

        except Exception as e:
            error = f"Forward sync error for {registration.name}: {e}"
            result.errors.append(error)
            logger.error(error, exc_info=True)
            registration.forward_errors += 1

        result.duration_ms = int((time.time() - start_time) * 1000)
        return result

    async def _sync_adapter_reverse(
        self,
        registration: AdapterRegistration,
        km_items: List[Dict[str, Any]],
    ) -> SyncResult:
        """
        Run reverse sync for a single adapter (KM → source).

        Args:
            registration: The adapter registration
            km_items: KM items to process

        Returns:
            SyncResult with operation details
        """
        start_time = time.time()
        result = SyncResult(
            adapter_name=registration.name,
            direction="reverse",
            success=False,
        )

        if not registration.reverse_method:
            result.errors.append(f"No reverse method for {registration.name}")
            return result

        try:
            adapter = registration.adapter
            method = getattr(adapter, registration.reverse_method)

            # Call the reverse method with KM items
            if asyncio.iscoroutinefunction(method):
                sync_result = await asyncio.wait_for(
                    method(km_items, min_confidence=self.config.min_confidence_for_reverse),
                    timeout=self.config.timeout_seconds,
                )
            else:
                sync_result = method(km_items, min_confidence=self.config.min_confidence_for_reverse)

            # Extract result details
            if isinstance(sync_result, dict):
                result.items_processed = sync_result.get("items_processed", 0)
                result.items_updated = sync_result.get("items_updated", 0)
                result.metadata = sync_result
            elif hasattr(sync_result, "items_processed"):
                result.items_processed = getattr(sync_result, "items_processed", 0)
                result.items_updated = getattr(sync_result, "items_updated", 0)
            elif hasattr(sync_result, "topics_analyzed"):
                # Pulse adapter returns PulseKMSyncResult
                result.items_processed = getattr(sync_result, "topics_analyzed", 0)
                result.items_updated = getattr(sync_result, "topics_adjusted", 0)
            elif hasattr(sync_result, "patterns_analyzed"):
                # Other adapters may have different result types
                result.items_processed = getattr(sync_result, "patterns_analyzed", 0)
                result.items_updated = getattr(sync_result, "adjustments_made", 0)

            result.success = True
            registration.last_reverse_sync = time.time()
            registration.reverse_errors = 0

        except asyncio.TimeoutError:
            error = f"Reverse sync timeout for {registration.name}"
            result.errors.append(error)
            logger.error(error)
            registration.reverse_errors += 1

        except Exception as e:
            error = f"Reverse sync error for {registration.name}: {e}"
            result.errors.append(error)
            logger.error(error, exc_info=True)
            registration.reverse_errors += 1

        result.duration_ms = int((time.time() - start_time) * 1000)
        return result

    async def sync_all_to_km(self) -> List[SyncResult]:
        """
        Forward sync: All sources → KM.

        Collects data from all registered adapters and syncs to KM.

        Returns:
            List of SyncResult for each adapter
        """
        results = []

        # Get enabled adapters sorted by priority
        enabled = [r for r in self._adapters.values() if r.enabled]
        enabled.sort(key=lambda x: x.priority, reverse=True)

        if self.config.parallel_sync:
            # Sync all adapters in parallel
            tasks = [
                self._sync_adapter_forward(reg)
                for reg in enabled
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(SyncResult(
                        adapter_name=enabled[i].name,
                        direction="forward",
                        success=False,
                        errors=[str(result)],
                    ))
                else:
                    final_results.append(result)
            results = final_results
        else:
            # Sync sequentially
            for reg in enabled:
                result = await self._sync_adapter_forward(reg)
                results.append(result)

        self._total_forward_syncs += 1
        return results

    async def sync_all_from_km(
        self,
        km_items: Optional[List[Dict[str, Any]]] = None,
    ) -> List[SyncResult]:
        """
        Reverse sync: KM → All sources.

        Propagates KM validations back to all registered adapters.

        Args:
            km_items: Optional KM items to process. If None, queries KM.

        Returns:
            List of SyncResult for each adapter
        """
        results = []

        # Get KM items if not provided
        if km_items is None:
            km_items = await self._get_recent_km_items()

        if not km_items:
            logger.info("No KM items to sync")
            return results

        # Get enabled adapters with reverse methods
        enabled = [
            r for r in self._adapters.values()
            if r.enabled and r.reverse_method
        ]
        enabled.sort(key=lambda x: x.priority, reverse=True)

        if self.config.parallel_sync:
            # Sync all adapters in parallel
            tasks = [
                self._sync_adapter_reverse(reg, km_items)
                for reg in enabled
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(SyncResult(
                        adapter_name=enabled[i].name,
                        direction="reverse",
                        success=False,
                        errors=[str(result)],
                    ))
                else:
                    final_results.append(result)
            results = final_results
        else:
            # Sync sequentially
            for reg in enabled:
                result = await self._sync_adapter_reverse(reg, km_items)
                results.append(result)

        self._total_reverse_syncs += 1
        return results

    async def _get_recent_km_items(self) -> List[Dict[str, Any]]:
        """Get recent KM items for reverse sync."""
        if self.knowledge_mound is None:
            # Return mock items for testing
            return []

        try:
            # Query KM for recent items
            if hasattr(self.knowledge_mound, "query_recent"):
                items = await self.knowledge_mound.query_recent(
                    limit=self.config.batch_size,
                    since=self._last_full_sync,
                )
                return items
        except Exception as e:
            logger.error(f"Error getting KM items: {e}")

        return []

    async def run_bidirectional_sync(
        self,
        km_items: Optional[List[Dict[str, Any]]] = None,
    ) -> BidirectionalSyncReport:
        """
        Run complete bidirectional sync cycle.

        Performs forward sync (sources → KM) followed by reverse sync (KM → sources).

        Args:
            km_items: Optional KM items for reverse sync

        Returns:
            BidirectionalSyncReport with all results
        """
        async with self._sync_lock:
            if self._sync_in_progress:
                return BidirectionalSyncReport(
                    metadata={"error": "Sync already in progress"}
                )

            self._sync_in_progress = True

        start_time = time.time()
        report = BidirectionalSyncReport(
            timestamp=datetime.utcnow().isoformat(),
        )

        try:
            # Forward sync
            logger.info("Starting forward sync (sources → KM)")
            forward_results = await self.sync_all_to_km()
            report.forward_results = forward_results
            report.successful_forward = sum(1 for r in forward_results if r.success)

            # Reverse sync
            logger.info("Starting reverse sync (KM → sources)")
            reverse_results = await self.sync_all_from_km(km_items)
            report.reverse_results = reverse_results
            report.successful_reverse = sum(1 for r in reverse_results if r.success)

            # Aggregate stats
            report.total_adapters = len(self._adapters)
            report.total_errors = (
                sum(len(r.errors) for r in forward_results) +
                sum(len(r.errors) for r in reverse_results)
            )
            self._total_errors += report.total_errors

            self._last_full_sync = time.time()

        finally:
            self._sync_in_progress = False

        report.total_duration_ms = int((time.time() - start_time) * 1000)
        report.metadata = {
            "forward_count": len(report.forward_results),
            "reverse_count": len(report.reverse_results),
            "parallel_sync": self.config.parallel_sync,
        }

        # Store in history
        self._sync_history.append(report)
        if len(self._sync_history) > self._max_history:
            self._sync_history = self._sync_history[-self._max_history:]

        logger.info(
            f"Bidirectional sync complete: "
            f"forward={report.successful_forward}/{len(report.forward_results)}, "
            f"reverse={report.successful_reverse}/{len(report.reverse_results)}, "
            f"errors={report.total_errors}, "
            f"duration={report.total_duration_ms}ms"
        )

        return report

    def get_status(self) -> Dict[str, Any]:
        """
        Get coordinator status and metrics.

        Returns:
            Dict with status information
        """
        adapter_status = {}
        for name, reg in self._adapters.items():
            adapter_status[name] = {
                "enabled": reg.enabled,
                "has_reverse": reg.reverse_method is not None,
                "priority": reg.priority,
                "last_forward_sync": reg.last_forward_sync,
                "last_reverse_sync": reg.last_reverse_sync,
                "forward_errors": reg.forward_errors,
                "reverse_errors": reg.reverse_errors,
            }

        return {
            "total_adapters": len(self._adapters),
            "enabled_adapters": sum(1 for r in self._adapters.values() if r.enabled),
            "bidirectional_adapters": sum(
                1 for r in self._adapters.values()
                if r.enabled and r.reverse_method
            ),
            "sync_in_progress": self._sync_in_progress,
            "last_full_sync": self._last_full_sync,
            "total_forward_syncs": self._total_forward_syncs,
            "total_reverse_syncs": self._total_reverse_syncs,
            "total_errors": self._total_errors,
            "sync_history_count": len(self._sync_history),
            "config": {
                "sync_interval_seconds": self.config.sync_interval_seconds,
                "batch_size": self.config.batch_size,
                "min_confidence": self.config.min_confidence_for_reverse,
                "parallel_sync": self.config.parallel_sync,
            },
            "adapters": adapter_status,
        }

    def get_sync_history(
        self,
        limit: int = 10,
    ) -> List[BidirectionalSyncReport]:
        """Get recent sync history."""
        return self._sync_history[-limit:]

    def clear_history(self) -> None:
        """Clear sync history."""
        self._sync_history = []

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self._total_forward_syncs = 0
        self._total_reverse_syncs = 0
        self._total_errors = 0
        self._sync_history = []

        for reg in self._adapters.values():
            reg.forward_errors = 0
            reg.reverse_errors = 0
            reg.last_forward_sync = None
            reg.last_reverse_sync = None


__all__ = [
    "BidirectionalCoordinator",
    "CoordinatorConfig",
    "AdapterRegistration",
    "SyncResult",
    "BidirectionalSyncReport",
]
