"""Cross-agent learning bus for sharing findings during Nomic Loop execution.

Enables agents to publish discoveries mid-cycle so other agents can benefit,
preventing duplicate work and enabling coordinated fixes.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, ClassVar

logger = logging.getLogger(__name__)


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


class LearningBus:
    """Thread-safe singleton bus for cross-agent finding sharing.

    Agents publish findings as they work; subscribers receive callbacks
    for topics they care about. Query methods let agents check what has
    already been discovered before starting work.
    """

    _instance: ClassVar[LearningBus | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self) -> None:
        self._findings: list[Finding] = []
        self._subscribers: dict[str, list[Callable[[Finding], None]]] = {}
        self._bus_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> LearningBus:
        """Return the singleton LearningBus, creating it if needed."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def publish(self, finding: Finding) -> None:
        """Add a finding and notify subscribers for its topic."""
        with self._bus_lock:
            self._findings.append(finding)
            callbacks = list(self._subscribers.get(finding.topic, []))

        for cb in callbacks:
            try:
                cb(finding)
            except Exception:
                logger.warning("Subscriber callback failed for topic %s", finding.topic)

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
