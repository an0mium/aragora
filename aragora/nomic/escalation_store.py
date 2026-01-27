"""
Escalation Store: Persistent Escalation State.

This module provides persistent storage for escalation state, enabling
escalations to survive restarts and tracking escalation progression
over time.

Key concepts:
- EscalationStore: JSONL-backed persistence for escalation states
- EscalationChain: Track escalation progression (WARN → THROTTLE → SUSPEND)
- EscalationRecovery: Load and resume active escalations on startup
- EscalationPolicy: Rules for automatic escalation progression

Usage:
    from aragora.nomic.escalation_store import EscalationStore, EscalationChain

    store = EscalationStore()
    await store.initialize()

    # Create an escalation chain
    chain = await store.create_chain(
        source="agent_monitor",
        target="agent-001",
        reason="response_latency_exceeded",
    )

    # Escalate to next level
    await chain.escalate()

    # Check current level
    print(chain.current_level)  # EscalationLevel.WARN

    # On restart, recover active escalations
    active = await store.get_active_escalations()
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from aragora.nomic.molecules import EscalationLevel

logger = logging.getLogger(__name__)


class EscalationStatus(str, Enum):
    """Status of an escalation chain."""

    ACTIVE = "active"  # Escalation in progress
    RESOLVED = "resolved"  # Issue resolved, escalation complete
    EXPIRED = "expired"  # Timed out without resolution
    CANCELLED = "cancelled"  # Manually cancelled
    SUPPRESSED = "suppressed"  # Suppressed by policy


@dataclass
class EscalationEvent:
    """
    A single event in an escalation chain.

    Records state transitions and actions taken.
    """

    id: str
    chain_id: str
    level: EscalationLevel
    action: str  # escalate, resolve, suppress, timeout
    timestamp: datetime
    reason: str
    previous_level: Optional[EscalationLevel] = None
    handler_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "chain_id": self.chain_id,
            "level": self.level.value,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "previous_level": self.previous_level.value if self.previous_level else None,
            "handler_result": self.handler_result,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EscalationEvent":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            chain_id=data["chain_id"],
            level=EscalationLevel(data["level"]),
            action=data["action"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            reason=data["reason"],
            previous_level=(
                EscalationLevel(data["previous_level"]) if data.get("previous_level") else None
            ),
            handler_result=data.get("handler_result"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EscalationChainConfig:
    """Configuration for escalation chain behavior."""

    # Level progression
    levels: List[EscalationLevel] = field(
        default_factory=lambda: [
            EscalationLevel.WARN,
            EscalationLevel.THROTTLE,
            EscalationLevel.SUSPEND,
            EscalationLevel.TERMINATE,
        ]
    )

    # Timing
    auto_escalate_minutes: int = 5  # Auto-escalate if not resolved
    cooldown_minutes: int = 30  # Minimum time between de-escalations
    max_duration_hours: int = 24  # Maximum escalation duration

    # Behavior
    allow_skip_levels: bool = False  # Allow jumping directly to higher levels
    auto_resolve_on_success: bool = True  # Auto-resolve when handler succeeds
    suppress_duplicates_minutes: int = 10  # Suppress duplicate escalations


@dataclass
class EscalationChain:
    """
    Tracks an escalation through multiple levels.

    An escalation chain represents a single issue being escalated
    through progressively more severe responses.
    """

    id: str
    source: str  # What triggered the escalation (e.g., "agent_monitor")
    target: str  # What is being escalated (e.g., "agent-001")
    reason: str  # Why escalation was triggered
    status: EscalationStatus
    current_level: EscalationLevel
    config: EscalationChainConfig
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    events: List[EscalationEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    auto_escalate_at: Optional[datetime] = None
    suppress_until: Optional[datetime] = None
    _store: Optional["EscalationStore"] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "reason": self.reason,
            "status": self.status.value,
            "current_level": self.current_level.value,
            "config": {
                "levels": [level.value for level in self.config.levels],
                "auto_escalate_minutes": self.config.auto_escalate_minutes,
                "cooldown_minutes": self.config.cooldown_minutes,
                "max_duration_hours": self.config.max_duration_hours,
                "allow_skip_levels": self.config.allow_skip_levels,
                "auto_resolve_on_success": self.config.auto_resolve_on_success,
                "suppress_duplicates_minutes": self.config.suppress_duplicates_minutes,
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
            "auto_escalate_at": self.auto_escalate_at.isoformat()
            if self.auto_escalate_at
            else None,
            "suppress_until": self.suppress_until.isoformat() if self.suppress_until else None,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], store: Optional["EscalationStore"] = None
    ) -> "EscalationChain":
        """Deserialize from dictionary."""
        config_data = data.get("config", {})
        config = EscalationChainConfig(
            levels=[
                EscalationLevel(lv)
                for lv in config_data.get("levels", ["warn", "throttle", "suspend", "terminate"])
            ],
            auto_escalate_minutes=config_data.get("auto_escalate_minutes", 5),
            cooldown_minutes=config_data.get("cooldown_minutes", 30),
            max_duration_hours=config_data.get("max_duration_hours", 24),
            allow_skip_levels=config_data.get("allow_skip_levels", False),
            auto_resolve_on_success=config_data.get("auto_resolve_on_success", True),
            suppress_duplicates_minutes=config_data.get("suppress_duplicates_minutes", 10),
        )

        return cls(
            id=data["id"],
            source=data["source"],
            target=data["target"],
            reason=data.get("reason", ""),
            status=EscalationStatus(data["status"]),
            current_level=EscalationLevel(data["current_level"]),
            config=config,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            resolved_at=(
                datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None
            ),
            events=[EscalationEvent.from_dict(e) for e in data.get("events", [])],
            metadata=data.get("metadata", {}),
            auto_escalate_at=(
                datetime.fromisoformat(data["auto_escalate_at"])
                if data.get("auto_escalate_at")
                else None
            ),
            suppress_until=(
                datetime.fromisoformat(data["suppress_until"])
                if data.get("suppress_until")
                else None
            ),
            _store=store,
        )

    def _get_level_index(self, level: EscalationLevel) -> int:
        """Get index of a level in the configured levels."""
        try:
            return self.config.levels.index(level)
        except ValueError:
            return -1

    def can_escalate(self) -> bool:
        """Check if escalation to next level is possible."""
        if self.status != EscalationStatus.ACTIVE:
            return False

        current_idx = self._get_level_index(self.current_level)
        return current_idx >= 0 and current_idx < len(self.config.levels) - 1

    def get_next_level(self) -> Optional[EscalationLevel]:
        """Get the next escalation level."""
        if not self.can_escalate():
            return None

        current_idx = self._get_level_index(self.current_level)
        return self.config.levels[current_idx + 1]

    def can_deescalate(self) -> bool:
        """Check if de-escalation is possible."""
        if self.status != EscalationStatus.ACTIVE:
            return False

        current_idx = self._get_level_index(self.current_level)
        return current_idx > 0

    def get_previous_level(self) -> Optional[EscalationLevel]:
        """Get the previous escalation level."""
        if not self.can_deescalate():
            return None

        current_idx = self._get_level_index(self.current_level)
        return self.config.levels[current_idx - 1]

    async def escalate(
        self,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[EscalationEvent]:
        """
        Escalate to the next level.

        Args:
            reason: Optional reason for escalation
            metadata: Optional additional metadata

        Returns:
            EscalationEvent if escalated, None otherwise
        """
        import uuid

        if not self.can_escalate():
            logger.warning(f"Cannot escalate chain {self.id}: already at max level or not active")
            return None

        next_level = self.get_next_level()
        if not next_level:
            return None

        now = datetime.now(timezone.utc)
        previous_level = self.current_level

        event = EscalationEvent(
            id=str(uuid.uuid4()),
            chain_id=self.id,
            level=next_level,
            action="escalate",
            timestamp=now,
            reason=reason or f"Escalating from {previous_level.value} to {next_level.value}",
            previous_level=previous_level,
            metadata=metadata or {},
        )

        self.current_level = next_level
        self.updated_at = now
        self.auto_escalate_at = now + timedelta(minutes=self.config.auto_escalate_minutes)
        self.events.append(event)

        if self._store:
            await self._store._save_chain(self)

        logger.info(f"Escalated chain {self.id}: {previous_level.value} -> {next_level.value}")
        return event

    async def resolve(
        self,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EscalationEvent:
        """
        Resolve the escalation.

        Args:
            reason: Optional reason for resolution
            metadata: Optional additional metadata

        Returns:
            EscalationEvent for the resolution
        """
        import uuid

        now = datetime.now(timezone.utc)

        event = EscalationEvent(
            id=str(uuid.uuid4()),
            chain_id=self.id,
            level=self.current_level,
            action="resolve",
            timestamp=now,
            reason=reason or "Issue resolved",
            metadata=metadata or {},
        )

        self.status = EscalationStatus.RESOLVED
        self.resolved_at = now
        self.updated_at = now
        self.auto_escalate_at = None
        self.events.append(event)

        if self._store:
            await self._store._save_chain(self)

        logger.info(f"Resolved escalation chain {self.id} at level {self.current_level.value}")
        return event

    async def deescalate(
        self,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[EscalationEvent]:
        """
        De-escalate to the previous level.

        Args:
            reason: Optional reason for de-escalation
            metadata: Optional additional metadata

        Returns:
            EscalationEvent if de-escalated, None otherwise
        """
        import uuid

        if not self.can_deescalate():
            logger.warning(
                f"Cannot de-escalate chain {self.id}: already at min level or not active"
            )
            return None

        previous_level = self.get_previous_level()
        if not previous_level:
            return None

        now = datetime.now(timezone.utc)
        old_level = self.current_level

        event = EscalationEvent(
            id=str(uuid.uuid4()),
            chain_id=self.id,
            level=previous_level,
            action="deescalate",
            timestamp=now,
            reason=reason or f"De-escalating from {old_level.value} to {previous_level.value}",
            previous_level=old_level,
            metadata=metadata or {},
        )

        self.current_level = previous_level
        self.updated_at = now
        self.auto_escalate_at = now + timedelta(minutes=self.config.auto_escalate_minutes)
        self.events.append(event)

        if self._store:
            await self._store._save_chain(self)

        logger.info(f"De-escalated chain {self.id}: {old_level.value} -> {previous_level.value}")
        return event

    async def suppress(
        self,
        duration_minutes: int,
        reason: Optional[str] = None,
    ) -> EscalationEvent:
        """
        Suppress the escalation for a duration.

        Args:
            duration_minutes: Duration to suppress
            reason: Optional reason for suppression

        Returns:
            EscalationEvent for the suppression
        """
        import uuid

        now = datetime.now(timezone.utc)

        event = EscalationEvent(
            id=str(uuid.uuid4()),
            chain_id=self.id,
            level=self.current_level,
            action="suppress",
            timestamp=now,
            reason=reason or f"Suppressed for {duration_minutes} minutes",
            metadata={"duration_minutes": duration_minutes},
        )

        self.status = EscalationStatus.SUPPRESSED
        self.suppress_until = now + timedelta(minutes=duration_minutes)
        self.auto_escalate_at = None
        self.updated_at = now
        self.events.append(event)

        if self._store:
            await self._store._save_chain(self)

        logger.info(f"Suppressed escalation chain {self.id} for {duration_minutes} minutes")
        return event

    @property
    def duration(self) -> timedelta:
        """Get the duration of this escalation."""
        end = self.resolved_at or datetime.now(timezone.utc)
        return end - self.created_at

    @property
    def is_expired(self) -> bool:
        """Check if escalation has exceeded max duration."""
        max_duration = timedelta(hours=self.config.max_duration_hours)
        return self.duration > max_duration

    @property
    def needs_auto_escalate(self) -> bool:
        """Check if auto-escalation is due."""
        if self.status != EscalationStatus.ACTIVE:
            return False
        if not self.auto_escalate_at:
            return False
        return datetime.now(timezone.utc) >= self.auto_escalate_at


class EscalationStore:
    """
    Persistent storage for escalation chains.

    Provides JSONL-backed persistence with recovery support.
    """

    def __init__(
        self,
        storage_dir: Optional[Path] = None,
        default_config: Optional[EscalationChainConfig] = None,
    ):
        """
        Initialize the escalation store.

        Args:
            storage_dir: Directory for storage files
            default_config: Default configuration for new chains
        """
        self.storage_dir = storage_dir or Path(".escalations")
        self.default_config = default_config or EscalationChainConfig()

        self._chains: Dict[str, EscalationChain] = {}
        self._by_target: Dict[str, Set[str]] = {}  # target -> chain_ids
        self._by_source: Dict[str, Set[str]] = {}  # source -> chain_ids
        self._lock = asyncio.Lock()
        self._initialized = False
        self._handlers: Dict[EscalationLevel, Callable] = {}

    async def initialize(self) -> None:
        """Initialize the store, loading existing chains."""
        if self._initialized:
            return

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        await self._load_chains()
        self._initialized = True
        logger.info(f"EscalationStore initialized with {len(self._chains)} chains")

    async def _load_chains(self) -> None:
        """Load chains from storage."""
        chains_file = self.storage_dir / "chains.jsonl"
        if not chains_file.exists():
            return

        try:
            with open(chains_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        chain = EscalationChain.from_dict(data, self)
                        self._chains[chain.id] = chain
                        self._index_chain(chain)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.warning(f"Invalid chain data: {e}")
        except Exception as e:
            logger.error(f"Failed to load chains: {e}")

    def _index_chain(self, chain: EscalationChain) -> None:
        """Add chain to lookup indices."""
        if chain.target not in self._by_target:
            self._by_target[chain.target] = set()
        self._by_target[chain.target].add(chain.id)

        if chain.source not in self._by_source:
            self._by_source[chain.source] = set()
        self._by_source[chain.source].add(chain.id)

    async def _save_chain(self, chain: EscalationChain) -> None:
        """Save a single chain update."""
        # For simplicity, rewrite the entire file (could optimize with append-only)
        await self._save_all_chains()

    async def _save_all_chains(self) -> None:
        """Save all chains to storage."""
        chains_file = self.storage_dir / "chains.jsonl"
        temp_file = chains_file.with_suffix(".tmp")

        try:
            with open(temp_file, "w") as f:
                for chain in self._chains.values():
                    f.write(json.dumps(chain.to_dict()) + "\n")
            temp_file.rename(chains_file)
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            logger.error(f"Failed to save chains: {e}")

    def register_handler(
        self,
        level: EscalationLevel,
        handler: Callable,
    ) -> None:
        """
        Register a handler for an escalation level.

        Args:
            level: The escalation level
            handler: Handler callable (sync or async)
        """
        self._handlers[level] = handler

    async def create_chain(
        self,
        source: str,
        target: str,
        reason: str,
        config: Optional[EscalationChainConfig] = None,
        metadata: Optional[Dict[str, Any]] = None,
        initial_level: Optional[EscalationLevel] = None,
    ) -> EscalationChain:
        """
        Create a new escalation chain.

        Args:
            source: Source of the escalation
            target: Target being escalated
            reason: Reason for escalation
            config: Optional chain configuration
            metadata: Optional metadata
            initial_level: Optional starting level (default: first level)

        Returns:
            New EscalationChain
        """
        import uuid

        async with self._lock:
            # Check for duplicate active escalation
            existing = await self.get_active_chain(source, target)
            if existing and not existing.is_expired:
                # Check suppression
                if existing.suppress_until and datetime.now(timezone.utc) < existing.suppress_until:
                    logger.info(
                        f"Escalation for {target} is suppressed until {existing.suppress_until}"
                    )
                    return existing

                # Check duplicate suppression window
                suppress_window = timedelta(
                    minutes=(config or self.default_config).suppress_duplicates_minutes
                )
                if datetime.now(timezone.utc) - existing.updated_at < suppress_window:
                    logger.info(f"Duplicate escalation for {target} suppressed")
                    return existing

            chain_config = config or self.default_config
            now = datetime.now(timezone.utc)
            start_level = initial_level or chain_config.levels[0]

            chain = EscalationChain(
                id=str(uuid.uuid4()),
                source=source,
                target=target,
                reason=reason,
                status=EscalationStatus.ACTIVE,
                current_level=start_level,
                config=chain_config,
                created_at=now,
                updated_at=now,
                auto_escalate_at=now + timedelta(minutes=chain_config.auto_escalate_minutes),
                metadata=metadata or {},
                _store=self,
            )

            # Add initial event
            initial_event = EscalationEvent(
                id=str(uuid.uuid4()),
                chain_id=chain.id,
                level=start_level,
                action="create",
                timestamp=now,
                reason=reason,
            )
            chain.events.append(initial_event)

            self._chains[chain.id] = chain
            self._index_chain(chain)
            await self._save_all_chains()

            logger.info(
                f"Created escalation chain {chain.id}: source={source} target={target} "
                f"level={start_level.value}"
            )

            # Execute handler if registered
            await self._execute_handler(chain, start_level)

            return chain

    async def _execute_handler(
        self,
        chain: EscalationChain,
        level: EscalationLevel,
    ) -> Optional[Any]:
        """Execute the handler for an escalation level."""
        handler = self._handlers.get(level)
        if not handler:
            logger.debug(f"No handler registered for level {level.value}")
            return None

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await handler(chain)
            else:
                result = handler(chain)

            # Update latest event with handler result
            if chain.events:
                chain.events[-1].handler_result = {"result": str(result)}

            return result
        except Exception as e:
            logger.error(f"Handler for level {level.value} failed: {e}")
            if chain.events:
                chain.events[-1].handler_result = {"error": str(e)}
            return None

    async def get_chain(self, chain_id: str) -> Optional[EscalationChain]:
        """Get a chain by ID."""
        return self._chains.get(chain_id)

    async def get_active_chain(self, source: str, target: str) -> Optional[EscalationChain]:
        """Get an active chain for a source-target pair."""
        target_chains = self._by_target.get(target, set())
        source_chains = self._by_source.get(source, set())
        common_chains = target_chains & source_chains

        for chain_id in common_chains:
            chain = self._chains.get(chain_id)
            if chain and chain.status == EscalationStatus.ACTIVE:
                return chain

        return None

    async def get_active_escalations(self) -> List[EscalationChain]:
        """Get all active escalation chains."""
        return [c for c in self._chains.values() if c.status == EscalationStatus.ACTIVE]

    async def get_chains_by_target(self, target: str) -> List[EscalationChain]:
        """Get all chains for a target."""
        chain_ids = self._by_target.get(target, set())
        return [self._chains[cid] for cid in chain_ids if cid in self._chains]

    async def get_chains_by_source(self, source: str) -> List[EscalationChain]:
        """Get all chains from a source."""
        chain_ids = self._by_source.get(source, set())
        return [self._chains[cid] for cid in chain_ids if cid in self._chains]

    async def process_auto_escalations(self) -> List[EscalationEvent]:
        """
        Process all pending auto-escalations.

        Returns:
            List of escalation events created
        """
        async with self._lock:
            events = []
            now = datetime.now(timezone.utc)

            for chain in list(self._chains.values()):
                # Check for expired escalations
                if chain.status == EscalationStatus.ACTIVE and chain.is_expired:
                    event = EscalationEvent(
                        id=str(__import__("uuid").uuid4()),
                        chain_id=chain.id,
                        level=chain.current_level,
                        action="timeout",
                        timestamp=now,
                        reason="Maximum escalation duration exceeded",
                    )
                    chain.status = EscalationStatus.EXPIRED
                    chain.updated_at = now
                    chain.events.append(event)
                    events.append(event)
                    logger.info(f"Escalation chain {chain.id} expired")
                    continue

                # Check for suppression expiry
                if chain.status == EscalationStatus.SUPPRESSED:
                    if chain.suppress_until and now >= chain.suppress_until:
                        chain.status = EscalationStatus.ACTIVE
                        chain.suppress_until = None
                        chain.auto_escalate_at = now + timedelta(
                            minutes=chain.config.auto_escalate_minutes
                        )
                        chain.updated_at = now
                        logger.info(f"Escalation chain {chain.id} unsuppressed")

                # Check for auto-escalation
                if chain.needs_auto_escalate and chain.can_escalate():
                    event = await chain.escalate(reason="Auto-escalation due to timeout")
                    if event:
                        events.append(event)
                        # Execute handler for new level
                        await self._execute_handler(chain, chain.current_level)

            if events:
                await self._save_all_chains()

            return events

    async def resolve_by_target(
        self,
        target: str,
        reason: Optional[str] = None,
    ) -> List[EscalationEvent]:
        """
        Resolve all active escalations for a target.

        Args:
            target: Target to resolve
            reason: Optional reason

        Returns:
            List of resolution events
        """
        events = []
        chains = await self.get_chains_by_target(target)

        for chain in chains:
            if chain.status == EscalationStatus.ACTIVE:
                event = await chain.resolve(reason=reason)
                events.append(event)

        return events

    async def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about escalations."""
        chains = list(self._chains.values())
        by_status: Dict[str, int] = {}
        by_level: Dict[str, int] = {}
        by_source: Dict[str, int] = {}

        for chain in chains:
            status_key = chain.status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1

            if chain.status == EscalationStatus.ACTIVE:
                level_key = chain.current_level.value
                by_level[level_key] = by_level.get(level_key, 0) + 1

            source_key = chain.source
            by_source[source_key] = by_source.get(source_key, 0) + 1

        return {
            "total_chains": len(chains),
            "active_chains": len([c for c in chains if c.status == EscalationStatus.ACTIVE]),
            "by_status": by_status,
            "by_level": by_level,
            "by_source": by_source,
            "total_events": sum(len(c.events) for c in chains),
        }


class EscalationRecovery:
    """
    Handles recovery of escalations on startup.

    Checks for interrupted escalations and resumes them.
    """

    def __init__(self, store: EscalationStore):
        """
        Initialize the recovery handler.

        Args:
            store: The escalation store
        """
        self.store = store

    async def recover(self) -> List[EscalationChain]:
        """
        Recover interrupted escalations.

        Returns:
            List of recovered chains
        """
        await self.store.initialize()
        active = await self.store.get_active_escalations()

        recovered = []
        for chain in active:
            # Check if chain needs attention
            if chain.needs_auto_escalate:
                recovered.append(chain)
                logger.info(f"Recovered escalation chain {chain.id} (needs auto-escalate)")
            elif chain.is_expired:
                recovered.append(chain)
                logger.info(f"Recovered escalation chain {chain.id} (expired)")

        # Process any pending escalations
        if recovered:
            await self.store.process_auto_escalations()

        return recovered


# Singleton instance
_default_store: Optional[EscalationStore] = None


async def get_escalation_store(
    storage_dir: Optional[Path] = None,
) -> EscalationStore:
    """Get the default escalation store instance."""
    global _default_store
    if _default_store is None:
        _default_store = EscalationStore(storage_dir)
        await _default_store.initialize()
    return _default_store


def reset_escalation_store() -> None:
    """Reset the default store (for testing)."""
    global _default_store
    _default_store = None
