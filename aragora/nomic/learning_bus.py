"""Cross-agent learning bus for sharing findings during Nomic Loop execution.

Enables agents to publish discoveries mid-cycle so other agents can benefit,
preventing duplicate work and enabling coordinated fixes.

Persistence: When ``persist=True`` (the default), findings are written through
to the Knowledge Mound via the NomicCycleAdapter so they survive across
sessions.  Historical findings are loaded automatically on first
``get_instance()`` call.  KM failures are silently swallowed so the in-memory
bus always works even when KM is unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time as _time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, ClassVar

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Finding dataclass
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """A discovery published to the learning bus."""

    agent_id: str
    topic: str  # pattern_bug, test_failure, architecture, dependency
    description: str
    affected_files: list[str] = field(default_factory=list)
    severity: str = "info"  # info, warning, critical
    suggested_action: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary suitable for KM metadata."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        """Reconstruct a Finding from a dictionary."""
        data = dict(data)  # shallow copy to avoid mutating caller
        ts = data.get("timestamp")
        if isinstance(ts, str):
            data["timestamp"] = datetime.fromisoformat(ts)
        elif not isinstance(ts, datetime):
            data["timestamp"] = datetime.now(timezone.utc)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Persistence configuration
# ---------------------------------------------------------------------------

@dataclass
class LearningBusConfig:
    """Configuration for LearningBus persistence behaviour."""

    persist: bool = True
    persist_cycles: bool = True
    max_historical: int = 50
    historical_days: int = 7
    workspace_id: str = "nomic"


# ---------------------------------------------------------------------------
# LearningBus
# ---------------------------------------------------------------------------

class LearningBus:
    """Thread-safe singleton bus for cross-agent finding sharing.

    Agents publish findings as they work; subscribers receive callbacks
    for topics they care about. Query methods let agents check what has
    already been discovered before starting work.

    When ``config.persist`` is True (default) findings are also written
    through to the Knowledge Mound for cross-session survival.  On the
    first ``get_instance()`` call, recent historical findings are loaded
    from KM to pre-populate the bus.
    """

    _instance: ClassVar[LearningBus | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config: LearningBusConfig | None = None) -> None:
        self._config = config or LearningBusConfig()
        self._findings: list[Finding] = []
        self._subscribers: dict[str, list[Callable[[Finding], None]]] = {}
        self._bus_lock = threading.Lock()
        self._historical_loaded = False
        self._km_adapter: Any | None = None  # lazy

    # ------------------------------------------------------------------
    # Singleton management
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, config: LearningBusConfig | None = None) -> LearningBus:
        """Return the singleton LearningBus, creating it if needed.

        On first creation when ``config.persist`` is True the bus will
        attempt to load recent historical findings from KM.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = cls(config)
                    cls._instance = instance
                    if instance._config.persist:
                        instance._load_historical_findings()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # KM adapter (lazy)
    # ------------------------------------------------------------------

    def _get_km_adapter(self) -> Any | None:
        """Lazily obtain the NomicCycleAdapter. Returns None on failure."""
        if self._km_adapter is not None:
            return self._km_adapter
        try:
            from aragora.knowledge.mound.adapters.nomic_cycle_adapter import (
                NomicCycleAdapter,
            )
            self._km_adapter = NomicCycleAdapter()
            return self._km_adapter
        except (ImportError, RuntimeError, ValueError, TypeError, OSError, AttributeError):
            logger.debug("KM adapter not available for LearningBus persistence")
            return None

    def _get_km_mound(self) -> Any | None:
        """Get the KM mound instance, returning None on failure."""
        adapter = self._get_km_adapter()
        if adapter is None:
            return None
        try:
            return adapter.mound
        except (RuntimeError, ValueError, TypeError, AttributeError):
            return None

    # ------------------------------------------------------------------
    # Persistence: write-through to KM
    # ------------------------------------------------------------------

    def _persist_finding(self, finding: Finding) -> None:
        """Write a finding to KM (fire-and-forget, sync wrapper)."""
        if not self._config.persist:
            return

        try:
            import asyncio

            loop: asyncio.AbstractEventLoop | None = None
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                pass

            if loop is not None and loop.is_running():
                # We are inside an async context - schedule as a task
                loop.create_task(self._persist_finding_async(finding))
            else:
                # No running loop - run synchronously
                asyncio.run(self._persist_finding_async(finding))
        except (ImportError, RuntimeError, ValueError, TypeError, OSError, AttributeError, KeyError):
            logger.debug("Failed to persist finding to KM")

    async def _persist_finding_async(self, finding: Finding) -> None:
        """Async implementation of finding persistence."""
        try:
            mound = self._get_km_mound()
            if mound is None:
                return

            from aragora.knowledge.mound import IngestionRequest
            from aragora.knowledge.unified.types import KnowledgeSource

            # Build a content string for the finding
            content = (
                f"FINDING [{finding.severity.upper()}]: {finding.description}"
            )
            if finding.suggested_action:
                content += f"\nSuggested action: {finding.suggested_action}"
            if finding.affected_files:
                content += f"\nAffected files: {', '.join(finding.affected_files[:10])}"

            # Generate a stable document ID from content + timestamp
            doc_hash = hashlib.sha256(
                f"{finding.agent_id}:{finding.topic}:{finding.description}:{finding.timestamp.isoformat()}".encode()
            ).hexdigest()[:12]
            document_id = f"finding_{doc_hash}"

            request = IngestionRequest(
                content=content,
                workspace_id=self._config.workspace_id,
                source_type=KnowledgeSource.INSIGHT,
                document_id=document_id,
                confidence=_severity_to_confidence(finding.severity),
                topics=["nomic", "learning_bus", finding.topic],
                metadata={
                    "type": "learning_bus_finding",
                    "agent_id": finding.agent_id,
                    "topic": finding.topic,
                    "severity": finding.severity,
                    "suggested_action": finding.suggested_action,
                    "affected_files": finding.affected_files[:20],
                    "timestamp": finding.timestamp.isoformat(),
                    "finding_data": finding.to_dict(),
                },
            )

            await mound.store(request)
            logger.debug(
                "Persisted finding to KM: topic=%s agent=%s",
                finding.topic,
                finding.agent_id,
            )
        except (ImportError, RuntimeError, ValueError, TypeError, OSError, AttributeError, KeyError):
            logger.debug("Failed to persist finding to KM")

    # ------------------------------------------------------------------
    # Persistence: load historical findings from KM
    # ------------------------------------------------------------------

    def _load_historical_findings(self) -> None:
        """Load recent findings from KM into memory.

        Called once on first ``get_instance()``.  Uses synchronous
        execution since it runs at startup.
        """
        if self._historical_loaded:
            return
        self._historical_loaded = True

        if not self._config.persist:
            return

        try:
            import asyncio
            findings = asyncio.run(self._load_historical_findings_async())
            if findings:
                with self._bus_lock:
                    # Prepend historical findings before any session findings
                    self._findings = findings + self._findings
                logger.info(
                    "Loaded %d historical findings from KM", len(findings)
                )
        except (ImportError, RuntimeError, ValueError, TypeError, OSError, AttributeError, KeyError):
            logger.debug("Could not load historical findings from KM")

    async def _load_historical_findings_async(self) -> list[Finding]:
        """Async implementation of historical finding loading."""
        mound = self._get_km_mound()
        if mound is None:
            return []

        try:
            results = await mound.search(
                query="nomic learning_bus finding",
                workspace_id=self._config.workspace_id,
                limit=self._config.max_historical,
                filters={"type": "learning_bus_finding"},
            )

            cutoff = datetime.now(timezone.utc) - timedelta(
                days=self._config.historical_days
            )

            findings: list[Finding] = []
            for result in results:
                metadata = getattr(result, "metadata", {})
                if metadata.get("type") != "learning_bus_finding":
                    continue

                finding_data = metadata.get("finding_data")
                if finding_data:
                    try:
                        finding = Finding.from_dict(finding_data)
                        if finding.timestamp >= cutoff:
                            findings.append(finding)
                    except (KeyError, TypeError, ValueError):
                        logger.debug("Skipping malformed historical finding")
                        continue

            # Sort by timestamp ascending (oldest first)
            findings.sort(key=lambda f: f.timestamp)
            return findings[: self._config.max_historical]

        except (ImportError, RuntimeError, ValueError, TypeError, OSError, AttributeError, KeyError):
            logger.debug("Failed to load historical findings from KM")
            return []

    def load_historical_findings(self) -> int:
        """Explicitly (re-)load historical findings from KM.

        Returns the number of findings loaded.  Safe to call multiple
        times; idempotent after the first successful load unless
        ``_historical_loaded`` is reset.
        """
        self._historical_loaded = False
        self._load_historical_findings()
        return len(self._findings)

    # ------------------------------------------------------------------
    # Cycle summary persistence
    # ------------------------------------------------------------------

    def save_cycle_summary(
        self,
        cycle_id: str,
        objective: str,
        agent_contributions: dict[str, Any] | None = None,
        surprise_events: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Aggregate current session findings into a NomicCycleRecord and persist.

        Args:
            cycle_id: Unique identifier for this cycle.
            objective: High-level objective of the cycle.
            agent_contributions: Optional dict of agent_name -> contribution data.
            surprise_events: Optional list of surprise event dicts.

        Returns:
            True if the cycle was saved successfully, False otherwise.
        """
        if not self._config.persist_cycles:
            return False

        try:
            from aragora.nomic.cycle_record import (
                AgentContribution,
                NomicCycleRecord,
                SurpriseEvent,
            )
            from aragora.nomic.cycle_store import get_cycle_store

            with self._bus_lock:
                findings = list(self._findings)

            # Build the cycle record from current findings
            record = NomicCycleRecord(
                cycle_id=cycle_id,
                started_at=_time.time(),
            )
            record.topics_debated = list({f.topic for f in findings})

            # Populate agent contributions
            if agent_contributions:
                for name, data in agent_contributions.items():
                    if isinstance(data, dict):
                        record.agent_contributions[name] = AgentContribution(
                            agent_name=name,
                            proposals_made=data.get("proposals_made", 0),
                            proposals_accepted=data.get("proposals_accepted", 0),
                            critiques_given=data.get("critiques_given", 0),
                            critiques_valuable=data.get("critiques_valuable", 0),
                        )

            # Populate surprise events
            if surprise_events:
                for se in surprise_events:
                    record.surprise_events.append(
                        SurpriseEvent(
                            phase=se.get("phase", "unknown"),
                            description=se.get("description", ""),
                            expected=se.get("expected", ""),
                            actual=se.get("actual", ""),
                            impact=se.get("impact", "low"),
                        )
                    )

            # Derive file lists from findings
            all_files: set[str] = set()
            for f in findings:
                all_files.update(f.affected_files)
            record.files_modified = sorted(all_files)

            # Mark as complete
            record.mark_complete(success=True)
            record.triggering_context = objective

            # Persist via CycleLearningStore
            store = get_cycle_store()
            store.save_cycle(record)

            logger.info(
                "Saved cycle summary: cycle_id=%s findings=%d files=%d",
                cycle_id,
                len(findings),
                len(all_files),
            )

            # Also persist to KM via the adapter if available
            self._persist_cycle_to_km(record, objective, findings)

            return True

        except (ImportError, RuntimeError, ValueError, TypeError, OSError, AttributeError, KeyError):
            logger.warning("Failed to save cycle summary for %s", cycle_id)
            return False

    def _persist_cycle_to_km(
        self,
        record: Any,
        objective: str,
        findings: list[Finding],
    ) -> None:
        """Persist the cycle summary to KM via NomicCycleAdapter (best-effort)."""
        try:
            import asyncio
            from aragora.knowledge.mound.adapters.nomic_cycle_adapter import (
                CycleStatus,
                NomicCycleOutcome,
            )

            outcome = NomicCycleOutcome(
                cycle_id=record.cycle_id,
                objective=objective,
                status=CycleStatus.SUCCESS if record.success else CycleStatus.FAILED,
                started_at=datetime.fromtimestamp(record.started_at, tz=timezone.utc),
                completed_at=datetime.fromtimestamp(
                    record.completed_at or _time.time(), tz=timezone.utc
                ),
                goals_attempted=len(findings),
                goals_succeeded=sum(1 for f in findings if f.severity != "critical"),
                goals_failed=sum(1 for f in findings if f.severity == "critical"),
                total_files_changed=len(record.files_modified),
                what_worked=[
                    f.description for f in findings if f.severity == "info"
                ][:5],
                what_failed=[
                    f.description for f in findings if f.severity == "critical"
                ][:5],
                agents_used=list(record.agent_contributions.keys()),
                tracks_affected=record.topics_debated,
            )

            adapter = self._get_km_adapter()
            if adapter is not None:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop is not None and loop.is_running():
                    loop.create_task(adapter.ingest_cycle_outcome(outcome))
                else:
                    asyncio.run(adapter.ingest_cycle_outcome(outcome))

        except (ImportError, RuntimeError, ValueError, TypeError, OSError, AttributeError, KeyError):
            logger.debug("Failed to persist cycle to KM adapter")

    # ------------------------------------------------------------------
    # Core publish / subscribe / query API (unchanged contract)
    # ------------------------------------------------------------------

    def publish(self, finding: Finding) -> None:
        """Add a finding and notify subscribers for its topic.

        When persistence is enabled the finding is also written through
        to the Knowledge Mound.
        """
        with self._bus_lock:
            self._findings.append(finding)
            callbacks = list(self._subscribers.get(finding.topic, []))

        for cb in callbacks:
            try:
                cb(finding)
            except (TypeError, ValueError, RuntimeError, AttributeError, KeyError, OSError):
                logger.warning("Subscriber callback failed for topic %s", finding.topic)

        # Write-through to KM (non-blocking, best-effort)
        self._persist_finding(finding)

    def subscribe(self, topic: str, callback: Callable[[Finding], None]) -> None:
        """Register a callback for findings on a given topic."""
        with self._bus_lock:
            self._subscribers.setdefault(topic, []).append(callback)

    def unsubscribe(self, topic: str, callback: Callable[[Finding], None]) -> None:
        """Remove a callback for a topic."""
        with self._bus_lock:
            cbs = self._subscribers.get(topic, [])
            try:
                cbs.remove(callback)
            except ValueError:
                pass

    def get_findings(
        self,
        topic: str | None = None,
        since: datetime | None = None,
    ) -> list[Finding]:
        """Query findings, optionally filtered by topic and/or time."""
        with self._bus_lock:
            results = list(self._findings)

        if topic is not None:
            results = [f for f in results if f.topic == topic]
        if since is not None:
            results = [f for f in results if f.timestamp >= since]
        return results

    def get_findings_for_files(self, files: list[str]) -> list[Finding]:
        """Return findings that affect any of the given files."""
        file_set = set(files)
        with self._bus_lock:
            return [
                f for f in self._findings
                if file_set & set(f.affected_files)
            ]

    def clear(self) -> None:
        """Reset all findings and subscribers (between cycles)."""
        with self._bus_lock:
            self._findings.clear()
            self._subscribers.clear()

    def summary(self) -> dict[str, Any]:
        """Return counts by topic and severity."""
        with self._bus_lock:
            findings = list(self._findings)

        by_topic: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for f in findings:
            by_topic[f.topic] = by_topic.get(f.topic, 0) + 1
            by_severity[f.severity] = by_severity.get(f.severity, 0) + 1

        return {
            "total": len(findings),
            "by_topic": by_topic,
            "by_severity": by_severity,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _severity_to_confidence(severity: str) -> float:
    """Map severity level to a KM confidence score."""
    return {"critical": 0.95, "warning": 0.85, "info": 0.7}.get(severity, 0.7)
