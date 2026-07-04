"""
Security event emission for scan findings.

This module provides functions to emit security events when
vulnerabilities, secrets, or SAST findings are detected.
Events can trigger multi-agent debates for critical findings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aragora.events.security_events import (
    SecurityEvent,
    SecurityEventType,
    SecurityFinding,
    SecuritySeverity,
    get_security_emitter,
)

if TYPE_CHECKING:
    from aragora.analysis.codebase import (
        SASTScanResult,
        ScanResult,
        SecretsScanResult,
    )

logger = logging.getLogger(__name__)


async def emit_scan_events(
    result: ScanResult,
    repo_id: str,
    scan_id: str,
    workspace_id: str | None = None,
) -> None:
    """
    Emit security events for scan findings.

    Automatically triggers multi-agent debates for critical vulnerabilities.
    """
    try:
        emitter = get_security_emitter()

        # Build findings from scan result
        findings = []
        for dep in result.dependencies:
            for vuln in dep.vulnerabilities:
                severity_map = {
                    "critical": SecuritySeverity.CRITICAL,
                    "high": SecuritySeverity.HIGH,
                    "medium": SecuritySeverity.MEDIUM,
                    "low": SecuritySeverity.LOW,
                }
                vuln_severity_str = (
                    vuln.severity.value.lower()
                    if hasattr(vuln.severity, "value")
                    else str(vuln.severity).lower()
                )
                severity = severity_map.get(vuln_severity_str, SecuritySeverity.MEDIUM)

                findings.append(
                    SecurityFinding(
                        id=vuln.id,
                        finding_type="vulnerability",
                        severity=severity,
                        title=vuln.title or vuln.id or "Unknown",
                        description=vuln.description or "",
                        cve_id=vuln.id,  # vuln.id is the CVE ID
                        package_name=dep.name,
                        package_version=dep.version,
                        recommendation=vuln.remediation_guidance,
                        metadata={
                            "ecosystem": dep.ecosystem,
                            "cvss_score": getattr(vuln, "cvss_score", None),
                            "sources": getattr(vuln, "sources", []),
                        },
                    )
                )

        if not findings:
            logger.debug("[Security] No findings to emit for scan %s", scan_id)
            return

        # Determine overall severity
        critical_count = sum(1 for f in findings if f.severity == SecuritySeverity.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == SecuritySeverity.HIGH)

        if critical_count > 0:
            overall_severity = SecuritySeverity.CRITICAL
            event_type = SecurityEventType.CRITICAL_VULNERABILITY
        elif high_count > 0:
            overall_severity = SecuritySeverity.HIGH
            event_type = SecurityEventType.VULNERABILITY_DETECTED
        else:
            overall_severity = SecuritySeverity.MEDIUM
            event_type = SecurityEventType.SCAN_COMPLETED

        # Emit scan completed event with findings
        # The emitter will auto-trigger debate for critical findings
        event = SecurityEvent(
            event_type=event_type,
            severity=overall_severity,
            repository=repo_id,
            scan_id=scan_id,
            workspace_id=workspace_id,
            findings=findings[:20],  # Limit to top 20 findings
        )

        await emitter.emit(event)

        logger.info(
            "[Security] Emitted %s event for scan %s: %s critical, %s high severity findings",
            event_type.value,
            scan_id,
            critical_count,
            high_count,
        )

    except (KeyError, ValueError, TypeError, RuntimeError) as e:
        logger.warning("[Security] Failed to emit scan events: %s", e)


async def emit_secrets_events(
    result: SecretsScanResult,
    repo_id: str,
    scan_id: str,
    workspace_id: str | None = None,
) -> None:
    """
    Emit security events for secrets scan findings.

    Automatically triggers multi-agent debates for critical secrets.
    """
    try:
        emitter = get_security_emitter()

        # Build findings from scan result
        findings = []
        for secret in result.secrets:
            severity_map = {
                "critical": SecuritySeverity.CRITICAL,
                "high": SecuritySeverity.HIGH,
                "medium": SecuritySeverity.MEDIUM,
                "low": SecuritySeverity.LOW,
            }
            secret_severity_str = (
                secret.severity.value.lower()
                if hasattr(secret.severity, "value")
                else str(secret.severity).lower()
            )
            severity = severity_map.get(secret_severity_str, SecuritySeverity.HIGH)

            findings.append(
                SecurityFinding(
                    id=secret.id,
                    finding_type="secret",
                    severity=severity,
                    title=f"Exposed {secret.secret_type.value if hasattr(secret.secret_type, 'value') else secret.secret_type}",
                    description=f"Hardcoded credential detected in {secret.file_path}",
                    file_path=secret.file_path,
                    line_number=secret.line_number,
                    recommendation="Rotate the credential immediately and remove from codebase",
                    metadata={
                        "secret_type": (
                            secret.secret_type.value
                            if hasattr(secret.secret_type, "value")
                            else str(secret.secret_type)
                        ),
                        "confidence": secret.confidence,
                        "is_in_history": getattr(secret, "is_in_history", False),
                    },
                )
            )

        if not findings:
            logger.debug("[Security] No secrets findings to emit for scan %s", scan_id)
            return

        # Determine overall severity
        critical_count = sum(1 for f in findings if f.severity == SecuritySeverity.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == SecuritySeverity.HIGH)

        if critical_count > 0:
            overall_severity = SecuritySeverity.CRITICAL
            event_type = SecurityEventType.CRITICAL_SECRET
        elif high_count > 0:
            overall_severity = SecuritySeverity.HIGH
            event_type = SecurityEventType.SECRET_DETECTED
        else:
            overall_severity = SecuritySeverity.MEDIUM
            event_type = SecurityEventType.SCAN_COMPLETED

        # Emit secrets event with findings
        event = SecurityEvent(
            event_type=event_type,
            severity=overall_severity,
            repository=repo_id,
            scan_id=scan_id,
            workspace_id=workspace_id,
            findings=findings[:20],  # Limit to top 20 findings
        )

        await emitter.emit(event)

        logger.info(
            "[Security] Emitted %s event for secrets scan %s: %s critical, %s high severity findings",
            event_type.value,
            scan_id,
            critical_count,
            high_count,
        )

    except (KeyError, ValueError, TypeError, RuntimeError) as e:
        logger.warning("[Security] Failed to emit secrets scan events: %s", e)


async def emit_sast_events(
    result: SASTScanResult,
    repo_id: str,
    scan_id: str,
    workspace_id: str | None = None,
) -> None:
    """Emit security events for SAST findings."""
    try:
        emitter = get_security_emitter()

        for finding in result.findings:
            if finding.severity.value not in ("critical", "error"):
                continue

            severity = (
                SecuritySeverity.CRITICAL
                if finding.severity.value == "critical"
                else SecuritySeverity.HIGH
            )

            sec_finding = SecurityFinding(
                id=f"{scan_id}:{finding.rule_id}:{finding.line_start}",
                finding_type="sast",
                title=finding.message[:100],
                description=finding.message,
                severity=severity,
                file_path=finding.file_path,
                line_number=finding.line_start,
                cve_id=finding.cwe_ids[0] if finding.cwe_ids else None,
                recommendation=finding.remediation or "Review and fix the security issue",
                metadata={
                    "category": finding.owasp_category.value,
                    "source": "sast_scanner",
                    "confidence": finding.confidence,
                },
            )

            event = SecurityEvent(
                event_type=SecurityEventType.SAST_CRITICAL,
                severity=severity,
                source="sast_scanner",
                findings=[sec_finding],
                scan_id=scan_id,
                repository=repo_id,
                metadata={
                    "rule_id": finding.rule_id,
                    "language": finding.language,
                    "owasp_category": finding.owasp_category.value,
                },
            )

            await emitter.emit(event)

    except (KeyError, ValueError, TypeError, RuntimeError) as e:
        logger.warning("Failed to emit SAST events: %s", e)
