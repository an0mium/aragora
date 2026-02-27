"""
Bridge between codebase analysis tools and the event system.

Emits RISK_WARNING events when BugDetector, SecretsScanner, or SASTScanner
find significant issues, enabling real-time security alerting and debate
context enrichment.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class AnalysisEventBridge:
    """Bridges codebase analysis findings to the event system."""

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence
        self.stats: dict[str, int] = {"events_emitted": 0, "findings_processed": 0}

    def emit_bug_findings(self, findings: list[Any]) -> int:
        """Emit RISK_WARNING events for bug detector findings.

        Args:
            findings: List of BugFinding objects from BugDetector

        Returns:
            Number of events emitted
        """
        emitted = 0
        for finding in findings:
            self.stats["findings_processed"] += 1
            confidence = getattr(finding, "confidence", 0.0)
            if confidence < self.min_confidence:
                continue
            severity = getattr(finding, "severity", "medium")
            # BugSeverity is a str enum, so .value gives the string
            if hasattr(severity, "value"):
                severity = severity.value
            bug_type = getattr(finding, "bug_type", "")
            if hasattr(bug_type, "value"):
                bug_type = bug_type.value
            self._emit_risk_warning(
                risk_type="bug_detected",
                severity=severity,
                description=getattr(finding, "description", "")[:500],
                details={
                    "bug_type": str(bug_type),
                    "file": getattr(finding, "file_path", ""),
                    "line": getattr(finding, "line_number", 0),
                    "confidence": round(confidence, 4),
                    "bug_id": getattr(finding, "bug_id", ""),
                },
            )
            emitted += 1
        return emitted

    def emit_secret_findings(self, findings: list[Any]) -> int:
        """Emit RISK_WARNING events for secrets scanner findings.

        Args:
            findings: List of SecretFinding objects from SecretsScanner

        Returns:
            Number of events emitted
        """
        emitted = 0
        for finding in findings:
            self.stats["findings_processed"] += 1
            secret_type = getattr(finding, "secret_type", "unknown")
            if hasattr(secret_type, "value"):
                secret_type = secret_type.value
            self._emit_risk_warning(
                risk_type="secret_detected",
                severity="critical",
                description=f"Potential secret found: {secret_type}",
                details={
                    "secret_type": str(secret_type),
                    "file": getattr(finding, "file_path", ""),
                    "line": getattr(finding, "line_number", 0),
                    # Never include the actual secret value
                },
            )
            emitted += 1
        return emitted

    def emit_sast_findings(self, findings: list[Any]) -> int:
        """Emit RISK_WARNING events for SAST scanner findings.

        Args:
            findings: List of SASTFinding objects from SASTScanner

        Returns:
            Number of events emitted
        """
        emitted = 0
        for finding in findings:
            self.stats["findings_processed"] += 1
            severity = getattr(finding, "severity", "medium")
            if hasattr(severity, "value"):
                severity = severity.value
            if severity in ("info", "low", "warning"):
                continue  # Only emit error+ severity
            self._emit_risk_warning(
                risk_type="sast_finding",
                severity=severity,
                description=getattr(finding, "message", "")[:500],
                details={
                    "rule_id": getattr(finding, "rule_id", ""),
                    "file": getattr(finding, "file_path", ""),
                    "line": getattr(finding, "line_start", 0),
                    "category": getattr(finding, "vulnerability_class", ""),
                },
            )
            emitted += 1
        return emitted

    def _emit_risk_warning(
        self,
        risk_type: str,
        severity: str,
        description: str,
        details: dict[str, Any],
    ) -> None:
        """Emit a single RISK_WARNING event via the event dispatcher."""
        try:
            from aragora.events.dispatcher import dispatch_event

            dispatch_event(
                "risk_warning",
                {
                    "risk_type": risk_type,
                    "severity": severity,
                    "description": description,
                    **details,
                },
            )
            self.stats["events_emitted"] += 1
        except Exception as e:  # noqa: BLE001
            # Broad catch: event emission is optional and must never disrupt analysis.
            logger.debug("Risk warning event emission unavailable: %s", e)


def get_analysis_event_bridge(min_confidence: float = 0.7) -> AnalysisEventBridge:
    """Get an analysis event bridge instance."""
    return AnalysisEventBridge(min_confidence=min_confidence)
