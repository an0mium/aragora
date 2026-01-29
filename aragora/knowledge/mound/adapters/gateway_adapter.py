"""
Gateway Adapter for Knowledge Mound Integration.

Bridges the LocalGateway with Knowledge Mound to enable:
- Message routing history and patterns
- Channel performance analytics
- Device registration tracking
- Routing rule effectiveness analysis

ID Prefixes:
- gw_message_: Message routing records
- gw_channel_: Channel performance snapshots
- gw_device_: Device registration records
- gw_route_: Routing decision records
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ._base import KnowledgeMoundAdapter

if TYPE_CHECKING:
    from aragora.gateway import LocalGateway

# KnowledgeMound is used dynamically, use Any for typing
KnowledgeMound = Any

logger = logging.getLogger(__name__)

# Type alias for event callbacks
EventCallback = Callable[[str, Dict[str, Any]], None]


@dataclass
class MessageRoutingRecord:
    """Record of a message routing event for KM storage."""

    message_id: str
    channel: str
    sender: str
    agent_id: Optional[str]
    routing_rule: Optional[str]
    success: bool
    latency_ms: float
    priority: str = "normal"
    thread_id: Optional[str] = None
    error_message: Optional[str] = None
    workspace_id: str = "default"


@dataclass
class ChannelPerformanceSnapshot:
    """Snapshot of channel performance for KM storage."""

    channel: str
    messages_received: int
    messages_routed: int
    messages_failed: int
    avg_latency_ms: float
    active_threads: int = 0
    unique_senders: int = 0
    workspace_id: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceRegistrationRecord:
    """Record of device registration for KM storage."""

    device_id: str
    device_name: str
    device_type: str
    status: str  # DeviceStatus value
    capabilities: List[str]
    registered_at: float
    last_seen: float
    workspace_id: str = "default"


@dataclass
class RoutingDecisionRecord:
    """Record of a routing decision for analysis."""

    decision_id: str
    message_id: str
    channel: str
    rule_matched: Optional[str]
    agent_selected: Optional[str]
    fallback_used: bool
    capabilities_required: List[str]
    capabilities_available: List[str]
    timestamp: float
    workspace_id: str = "default"


class GatewayAdapter(KnowledgeMoundAdapter):
    """
    Adapter that bridges LocalGateway to the Knowledge Mound.

    Provides bidirectional sync between Gateway operations and KM:
    - Forward: Routing records, channel stats, device registrations → KM
    - Reverse: Historical patterns → Routing optimization

    Usage:
        from aragora.gateway import LocalGateway
        from aragora.knowledge.mound.adapters import GatewayAdapter

        gateway = LocalGateway()
        adapter = GatewayAdapter(gateway)

        # Store routing record
        await adapter.store_routing_record(routing_record)

        # Get channel recommendations
        recommendations = await adapter.get_routing_recommendations("slack")
    """

    adapter_name = "gateway"

    def __init__(
        self,
        gateway: Optional["LocalGateway"] = None,
        knowledge_mound: Optional["KnowledgeMound"] = None,
        workspace_id: str = "default",
        event_callback: Optional[EventCallback] = None,
        min_confidence_threshold: float = 0.6,
        enable_dual_write: bool = False,
    ):
        """
        Initialize the adapter.

        Args:
            gateway: The LocalGateway instance to wrap
            knowledge_mound: Optional KnowledgeMound for direct storage
            workspace_id: Workspace ID for multi-tenancy
            event_callback: Optional callback for emitting events
            min_confidence_threshold: Minimum confidence to store records
            enable_dual_write: If True, writes to both systems during migration
        """
        super().__init__(
            enable_dual_write=enable_dual_write,
            event_callback=event_callback,
        )
        self._gateway = gateway
        self._knowledge_mound = knowledge_mound
        self._workspace_id = workspace_id
        self._min_confidence_threshold = min_confidence_threshold

        # Caches for reverse flow
        self._channel_performance_cache: Dict[str, List[ChannelPerformanceSnapshot]] = {}
        self._routing_patterns_cache: Dict[str, List[MessageRoutingRecord]] = {}
        self._cache_ttl: float = 300  # 5 minutes
        self._cache_times: Dict[str, float] = {}

        # Statistics
        self._stats = {
            "routing_records_stored": 0,
            "channel_snapshots_stored": 0,
            "device_records_stored": 0,
            "routing_decisions_stored": 0,
            "channel_queries": 0,
            "routing_queries": 0,
        }

    # =========================================================================
    # Forward Sync: Gateway → KM
    # =========================================================================

    async def store_routing_record(
        self,
        record: MessageRoutingRecord,
    ) -> Optional[str]:
        """
        Store a message routing record in the Knowledge Mound.

        Args:
            record: Message routing record to store

        Returns:
            KM item ID if stored, None if below threshold
        """
        if not self._knowledge_mound:
            logger.debug("No KnowledgeMound configured, skipping routing record storage")
            return None

        with self._timed_operation("store_routing_record", message_id=record.message_id):
            try:
                from aragora.knowledge.unified.types import (
                    KnowledgeItem,
                    KnowledgeSource,
                    ConfidenceLevel,
                )

                confidence = ConfidenceLevel.HIGH if record.success else ConfidenceLevel.MEDIUM
                now = datetime.now()

                status = "success" if record.success else "failed"
                agent_info = f" → {record.agent_id}" if record.agent_id else ""
                rule_info = f" (rule: {record.routing_rule})" if record.routing_rule else ""

                item = KnowledgeItem(
                    id=f"gw_message_{record.message_id}",
                    content=(
                        f"Message routed ({status}): {record.channel}{agent_info}{rule_info}, "
                        f"{record.latency_ms:.0f}ms"
                    ),
                    source=KnowledgeSource.CONTINUUM,
                    source_id=record.message_id,
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "gateway_routing_record",
                        "message_id": record.message_id,
                        "channel": record.channel,
                        "sender": record.sender,
                        "agent_id": record.agent_id,
                        "routing_rule": record.routing_rule,
                        "success": record.success,
                        "latency_ms": record.latency_ms,
                        "priority": record.priority,
                        "thread_id": record.thread_id,
                        "error_message": record.error_message,
                        "workspace_id": record.workspace_id,
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["routing_records_stored"] += 1
                self._emit_event(
                    "gw_routing_record_stored",
                    {
                        "message_id": record.message_id,
                        "channel": record.channel,
                        "success": record.success,
                    },
                )

                # Invalidate cache
                if record.channel in self._routing_patterns_cache:
                    del self._routing_patterns_cache[record.channel]

                return item_id

            except Exception as e:
                logger.error(f"Failed to store routing record: {e}")
                return None

    async def store_channel_snapshot(
        self,
        snapshot: ChannelPerformanceSnapshot,
    ) -> Optional[str]:
        """
        Store a channel performance snapshot in the Knowledge Mound.

        Args:
            snapshot: Channel performance snapshot to store

        Returns:
            KM item ID if stored
        """
        if not self._knowledge_mound:
            return None

        with self._timed_operation("store_channel_snapshot", channel=snapshot.channel):
            try:
                from aragora.knowledge.unified.types import (
                    KnowledgeItem,
                    KnowledgeSource,
                    ConfidenceLevel,
                )

                # Calculate success rate
                total = snapshot.messages_routed + snapshot.messages_failed
                success_rate = snapshot.messages_routed / max(1, total)

                confidence = ConfidenceLevel.HIGH if total >= 10 else ConfidenceLevel.MEDIUM
                now = datetime.now()

                item = KnowledgeItem(
                    id=f"gw_channel_{snapshot.channel}_{int(time.time())}",
                    content=(
                        f"Channel '{snapshot.channel}' performance: "
                        f"{snapshot.messages_routed}/{total} routed ({success_rate:.0%}), "
                        f"avg {snapshot.avg_latency_ms:.0f}ms, "
                        f"{snapshot.active_threads} threads, {snapshot.unique_senders} senders"
                    ),
                    source=KnowledgeSource.ELO,  # Performance data
                    source_id=f"channel_{snapshot.channel}",
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "gateway_channel_snapshot",
                        "channel": snapshot.channel,
                        "messages_received": snapshot.messages_received,
                        "messages_routed": snapshot.messages_routed,
                        "messages_failed": snapshot.messages_failed,
                        "success_rate": success_rate,
                        "avg_latency_ms": snapshot.avg_latency_ms,
                        "active_threads": snapshot.active_threads,
                        "unique_senders": snapshot.unique_senders,
                        "workspace_id": snapshot.workspace_id,
                        **snapshot.metadata,
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["channel_snapshots_stored"] += 1
                self._emit_event(
                    "gw_channel_snapshot_stored",
                    {
                        "channel": snapshot.channel,
                        "success_rate": success_rate,
                        "messages_routed": snapshot.messages_routed,
                    },
                )

                # Invalidate cache
                if snapshot.channel in self._channel_performance_cache:
                    del self._channel_performance_cache[snapshot.channel]

                return item_id

            except Exception as e:
                logger.error(f"Failed to store channel snapshot: {e}")
                return None

    async def store_device_registration(
        self,
        record: DeviceRegistrationRecord,
    ) -> Optional[str]:
        """
        Store a device registration record in the Knowledge Mound.

        Args:
            record: Device registration record to store

        Returns:
            KM item ID if stored
        """
        if not self._knowledge_mound:
            return None

        with self._timed_operation("store_device_registration", device_id=record.device_id):
            try:
                from aragora.knowledge.unified.types import (
                    KnowledgeItem,
                    KnowledgeSource,
                    ConfidenceLevel,
                )

                confidence = ConfidenceLevel.HIGH  # Device registrations are authoritative
                now = datetime.now()

                capabilities_str = ", ".join(record.capabilities[:5])
                if len(record.capabilities) > 5:
                    capabilities_str += f" (+{len(record.capabilities) - 5} more)"

                item = KnowledgeItem(
                    id=f"gw_device_{record.device_id}",
                    content=(
                        f"Device '{record.device_name}' ({record.device_type}): "
                        f"status={record.status}, capabilities=[{capabilities_str}]"
                    ),
                    source=KnowledgeSource.CONTINUUM,
                    source_id=record.device_id,
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "gateway_device_registration",
                        "device_id": record.device_id,
                        "device_name": record.device_name,
                        "device_type": record.device_type,
                        "status": record.status,
                        "capabilities": record.capabilities,
                        "registered_at": record.registered_at,
                        "last_seen": record.last_seen,
                        "workspace_id": record.workspace_id,
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["device_records_stored"] += 1
                self._emit_event(
                    "gw_device_registered",
                    {
                        "device_id": record.device_id,
                        "device_type": record.device_type,
                        "status": record.status,
                    },
                )

                return item_id

            except Exception as e:
                logger.error(f"Failed to store device registration: {e}")
                return None

    async def store_routing_decision(
        self,
        record: RoutingDecisionRecord,
    ) -> Optional[str]:
        """
        Store a routing decision record for analysis.

        Args:
            record: Routing decision record to store

        Returns:
            KM item ID if stored
        """
        if not self._knowledge_mound:
            return None

        with self._timed_operation("store_routing_decision", decision_id=record.decision_id):
            try:
                from aragora.knowledge.unified.types import (
                    KnowledgeItem,
                    KnowledgeSource,
                    ConfidenceLevel,
                )

                confidence = ConfidenceLevel.HIGH
                now = datetime.now()

                fallback_str = " (fallback)" if record.fallback_used else ""
                agent_str = record.agent_selected or "no agent"

                item = KnowledgeItem(
                    id=f"gw_route_{record.decision_id}",
                    content=(
                        f"Routing decision: {record.channel} → {agent_str}{fallback_str}, "
                        f"rule={record.rule_matched or 'none'}"
                    ),
                    source=KnowledgeSource.CONTINUUM,
                    source_id=record.decision_id,
                    confidence=confidence,
                    created_at=now,
                    updated_at=now,
                    metadata={
                        "type": "gateway_routing_decision",
                        "decision_id": record.decision_id,
                        "message_id": record.message_id,
                        "channel": record.channel,
                        "rule_matched": record.rule_matched,
                        "agent_selected": record.agent_selected,
                        "fallback_used": record.fallback_used,
                        "capabilities_required": record.capabilities_required,
                        "capabilities_available": record.capabilities_available,
                        "timestamp": record.timestamp,
                        "workspace_id": record.workspace_id,
                    },
                )

                item_id = await self._knowledge_mound.ingest(item)

                self._stats["routing_decisions_stored"] += 1

                return item_id

            except Exception as e:
                logger.error(f"Failed to store routing decision: {e}")
                return None

    # =========================================================================
    # Reverse Flow: KM → Gateway
    # =========================================================================

    async def get_channel_performance_history(
        self,
        channel: str,
        limit: int = 20,
        use_cache: bool = True,
    ) -> List[ChannelPerformanceSnapshot]:
        """
        Get historical channel performance from KM.

        Args:
            channel: Channel to query
            limit: Maximum snapshots to return
            use_cache: Whether to use cached results

        Returns:
            List of ChannelPerformanceSnapshot sorted by recency
        """
        self._stats["channel_queries"] += 1

        # Check cache
        if use_cache:
            if channel in self._channel_performance_cache:
                cache_time = self._cache_times.get(f"channel_{channel}", 0)
                if time.time() - cache_time < self._cache_ttl:
                    return self._channel_performance_cache[channel][:limit]

        if not self._knowledge_mound:
            return []

        with self._timed_operation("get_channel_performance", channel=channel):
            try:
                results = await self._knowledge_mound.query(
                    query=f"gateway channel {channel}",
                    limit=limit * 2,
                    workspace_id=self._workspace_id,
                )

                snapshots = []
                for result in results:
                    metadata = result.get("metadata", {})
                    if metadata.get("type") != "gateway_channel_snapshot":
                        continue
                    if metadata.get("channel") != channel:
                        continue

                    snapshot = ChannelPerformanceSnapshot(
                        channel=metadata.get("channel", ""),
                        messages_received=metadata.get("messages_received", 0),
                        messages_routed=metadata.get("messages_routed", 0),
                        messages_failed=metadata.get("messages_failed", 0),
                        avg_latency_ms=metadata.get("avg_latency_ms", 0.0),
                        active_threads=metadata.get("active_threads", 0),
                        unique_senders=metadata.get("unique_senders", 0),
                        workspace_id=self._workspace_id,
                    )
                    snapshots.append(snapshot)

                # Cache results
                if use_cache:
                    self._channel_performance_cache[channel] = snapshots
                    self._cache_times[f"channel_{channel}"] = time.time()

                return snapshots[:limit]

            except Exception as e:
                logger.error(f"Failed to get channel performance history: {e}")
                return []

    async def get_routing_patterns(
        self,
        channel: str,
        limit: int = 50,
        use_cache: bool = True,
    ) -> List[MessageRoutingRecord]:
        """
        Get historical routing patterns for a channel from KM.

        Args:
            channel: Channel to query
            limit: Maximum records to return
            use_cache: Whether to use cached results

        Returns:
            List of MessageRoutingRecord sorted by recency
        """
        self._stats["routing_queries"] += 1

        # Check cache
        if use_cache:
            if channel in self._routing_patterns_cache:
                cache_time = self._cache_times.get(f"routing_{channel}", 0)
                if time.time() - cache_time < self._cache_ttl:
                    return self._routing_patterns_cache[channel][:limit]

        if not self._knowledge_mound:
            return []

        with self._timed_operation("get_routing_patterns", channel=channel):
            try:
                results = await self._knowledge_mound.query(
                    query=f"gateway routing {channel}",
                    limit=limit * 2,
                    workspace_id=self._workspace_id,
                )

                records = []
                for result in results:
                    metadata = result.get("metadata", {})
                    if metadata.get("type") != "gateway_routing_record":
                        continue
                    if metadata.get("channel") != channel:
                        continue

                    record = MessageRoutingRecord(
                        message_id=metadata.get("message_id", ""),
                        channel=metadata.get("channel", ""),
                        sender=metadata.get("sender", ""),
                        agent_id=metadata.get("agent_id"),
                        routing_rule=metadata.get("routing_rule"),
                        success=metadata.get("success", False),
                        latency_ms=metadata.get("latency_ms", 0.0),
                        priority=metadata.get("priority", "normal"),
                        thread_id=metadata.get("thread_id"),
                        error_message=metadata.get("error_message"),
                        workspace_id=self._workspace_id,
                    )
                    if record.message_id:
                        records.append(record)

                # Cache results
                if use_cache:
                    self._routing_patterns_cache[channel] = records
                    self._cache_times[f"routing_{channel}"] = time.time()

                return records[:limit]

            except Exception as e:
                logger.error(f"Failed to get routing patterns: {e}")
                return []

    async def get_routing_recommendations(
        self,
        channel: str,
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Get routing recommendations for a channel based on historical performance.

        Args:
            channel: Channel to get recommendations for
            top_n: Number of recommendations to return

        Returns:
            List of recommendations with agent and rule scores
        """
        patterns = await self.get_routing_patterns(channel, limit=100, use_cache=True)

        if not patterns:
            return [
                {
                    "recommendation": "No routing history found",
                    "confidence": 0.0,
                }
            ]

        # Analyze agent performance
        agent_stats: Dict[str, Dict[str, Any]] = {}
        rule_stats: Dict[str, Dict[str, int]] = {}

        for record in patterns:
            # Track agent performance
            if record.agent_id:
                if record.agent_id not in agent_stats:
                    agent_stats[record.agent_id] = {
                        "success": 0,
                        "failure": 0,
                        "total_latency": 0.0,
                    }
                stats = agent_stats[record.agent_id]
                if record.success:
                    stats["success"] += 1
                else:
                    stats["failure"] += 1
                stats["total_latency"] += record.latency_ms

            # Track rule effectiveness
            if record.routing_rule:
                if record.routing_rule not in rule_stats:
                    rule_stats[record.routing_rule] = {"success": 0, "failure": 0}
                if record.success:
                    rule_stats[record.routing_rule]["success"] += 1
                else:
                    rule_stats[record.routing_rule]["failure"] += 1

        recommendations = []

        # Best agent recommendation
        if agent_stats:
            best_agent = max(
                agent_stats.items(),
                key=lambda x: x[1]["success"] / max(1, x[1]["success"] + x[1]["failure"]),
            )
            agent_id, stats = best_agent
            total = stats["success"] + stats["failure"]
            success_rate = stats["success"] / max(1, total)
            avg_latency = stats["total_latency"] / max(1, total)

            recommendations.append(
                {
                    "type": "agent",
                    "recommendation": f"Route to agent '{agent_id}' ({success_rate:.0%} success, {avg_latency:.0f}ms avg)",
                    "agent_id": agent_id,
                    "success_rate": success_rate,
                    "avg_latency_ms": avg_latency,
                    "sample_size": total,
                    "confidence": min(1.0, total / 20),
                }
            )

        # Best rule recommendation
        if rule_stats:
            best_rule = max(
                rule_stats.items(),
                key=lambda x: x[1]["success"] / max(1, x[1]["success"] + x[1]["failure"]),
            )
            rule_id, stats = best_rule
            total = stats["success"] + stats["failure"]
            success_rate = stats["success"] / max(1, total)

            recommendations.append(
                {
                    "type": "rule",
                    "recommendation": f"Use rule '{rule_id}' ({success_rate:.0%} success)",
                    "rule_id": rule_id,
                    "success_rate": success_rate,
                    "sample_size": total,
                    "confidence": min(1.0, total / 20),
                }
            )

        # Overall channel health
        total_success = sum(1 for r in patterns if r.success)
        overall_success_rate = total_success / max(1, len(patterns))

        recommendations.append(
            {
                "type": "health",
                "recommendation": f"Channel '{channel}' overall success rate: {overall_success_rate:.0%}",
                "success_rate": overall_success_rate,
                "total_messages": len(patterns),
                "confidence": min(1.0, len(patterns) / 50),
            }
        )

        return recommendations[:top_n]

    async def get_device_capabilities_analysis(
        self,
        capability: str,
    ) -> Dict[str, Any]:
        """
        Analyze device distribution for a specific capability.

        Args:
            capability: Capability to analyze

        Returns:
            Analysis with device counts and availability
        """
        if not self._knowledge_mound:
            return {"analysis_available": False}

        with self._timed_operation("get_device_capabilities_analysis", capability=capability):
            try:
                results = await self._knowledge_mound.query(
                    query=f"gateway device {capability}",
                    limit=100,
                    workspace_id=self._workspace_id,
                )

                devices_with_capability = []
                devices_without = []

                for result in results:
                    metadata = result.get("metadata", {})
                    if metadata.get("type") != "gateway_device_registration":
                        continue

                    device_id = metadata.get("device_id", "")
                    capabilities = metadata.get("capabilities", [])
                    status = metadata.get("status", "")

                    if capability in capabilities:
                        devices_with_capability.append(
                            {
                                "device_id": device_id,
                                "device_type": metadata.get("device_type", ""),
                                "status": status,
                                "online": status in ("online", "active"),
                            }
                        )
                    else:
                        devices_without.append(device_id)

                online_devices = [d for d in devices_with_capability if d["online"]]

                return {
                    "analysis_available": True,
                    "capability": capability,
                    "devices_with_capability": len(devices_with_capability),
                    "devices_online": len(online_devices),
                    "devices_without": len(devices_without),
                    "availability": len(online_devices) / max(1, len(devices_with_capability)),
                    "device_types": list(set(d["device_type"] for d in devices_with_capability)),
                }

            except Exception as e:
                logger.error(f"Failed to get device capabilities analysis: {e}")
                return {"analysis_available": False, "error": str(e)}

    # =========================================================================
    # Sync from Gateway
    # =========================================================================

    async def sync_from_gateway(self) -> Dict[str, Any]:
        """
        Sync current gateway state to Knowledge Mound.

        Returns:
            Dict with counts of synced items
        """
        if not self._gateway:
            logger.warning("No gateway configured for sync")
            return {"error": "No gateway configured"}

        synced = {
            "channels": 0,
            "devices": 0,
        }

        with self._timed_operation("sync_from_gateway"):
            try:
                # Get gateway stats
                stats = await self._gateway.get_stats()

                # Channel performance from stats
                inbox_size = stats.get("inbox_size", 0)
                messages_routed = stats.get("messages_routed", 0)
                messages_failed = stats.get("messages_failed", 0)

                # Create a general channel snapshot
                if messages_routed + messages_failed > 0:
                    snapshot = ChannelPerformanceSnapshot(
                        channel="all",
                        messages_received=inbox_size,
                        messages_routed=messages_routed,
                        messages_failed=messages_failed,
                        avg_latency_ms=0.0,  # Would need actual latency tracking
                        workspace_id=self._workspace_id,
                    )
                    if await self.store_channel_snapshot(snapshot):
                        synced["channels"] += 1

                # Sync device registrations
                devices = await self._gateway.list_devices()  # type: ignore[attr-defined]
                for device in devices:
                    record = DeviceRegistrationRecord(
                        device_id=device.device_id,
                        device_name=device.name,
                        device_type=device.device_type,
                        status=device.status.value,
                        capabilities=device.capabilities,
                        registered_at=device.registered_at,
                        last_seen=device.last_seen,
                        workspace_id=self._workspace_id,
                    )
                    if await self.store_device_registration(record):
                        synced["devices"] += 1

                logger.info(f"Synced from gateway: {synced}")
                return synced

            except Exception as e:
                logger.error(f"Failed to sync from gateway: {e}")
                return {"error": str(e)}

    # =========================================================================
    # Stats and Health
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return {
            **self._stats,
            "workspace_id": self._workspace_id,
            "channel_cache_size": len(self._channel_performance_cache),
            "routing_cache_size": len(self._routing_patterns_cache),
            "has_knowledge_mound": self._knowledge_mound is not None,
            "has_gateway": self._gateway is not None,
        }

    def clear_cache(self) -> int:
        """Clear all caches and return count of cleared items."""
        count = len(self._channel_performance_cache) + len(self._routing_patterns_cache)
        self._channel_performance_cache.clear()
        self._routing_patterns_cache.clear()
        self._cache_times.clear()
        return count


__all__ = [
    "GatewayAdapter",
    "MessageRoutingRecord",
    "ChannelPerformanceSnapshot",
    "DeviceRegistrationRecord",
    "RoutingDecisionRecord",
]
