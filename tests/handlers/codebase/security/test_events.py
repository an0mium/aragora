"""
Tests for security event emission functions
(aragora/server/handlers/codebase/security/events.py).

Covers all three async functions:
- emit_scan_events: Emit security events for vulnerability scan findings
- emit_secrets_events: Emit security events for secrets scan findings
- emit_sast_events: Emit security events for SAST findings

Tests include: happy path, no-findings early return, severity mapping,
unknown severity fallback, event type selection (critical/high/medium),
workspace_id propagation, findings truncation to 20, emitter failure
handling, SAST severity filtering, and edge cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.analysis.codebase import (
    DependencyInfo,
    ScanResult,
    SecretFinding,
    SecretsScanResult,
    SecretType,
    VulnerabilityFinding,
    VulnerabilitySeverity,
)
from aragora.analysis.codebase.sast.models import (
    OWASPCategory,
    SASTFinding,
    SASTScanResult,
    SASTSeverity,
)
from aragora.events.security_events import (
    SecurityEvent,
    SecurityEventType,
    SecurityFinding,
    SecuritySeverity,
)
from aragora.server.handlers.codebase.security.events import (
    emit_scan_events,
    emit_secrets_events,
    emit_sast_events,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_vuln(
    vuln_id: str = "CVE-2024-0001",
    title: str = "Test Vulnerability",
    description: str = "A test vulnerability",
    severity: VulnerabilitySeverity = VulnerabilitySeverity.HIGH,
    cvss_score: float | None = 7.5,
    remediation_guidance: str | None = "Upgrade to latest version",
) -> VulnerabilityFinding:
    """Create a VulnerabilityFinding for testing."""
    return VulnerabilityFinding(
        id=vuln_id,
        title=title,
        description=description,
        severity=severity,
        cvss_score=cvss_score,
        remediation_guidance=remediation_guidance,
    )


def _make_dep(
    name: str = "requests",
    version: str = "2.28.0",
    ecosystem: str = "pypi",
    vulnerabilities: list[VulnerabilityFinding] | None = None,
) -> DependencyInfo:
    """Create a DependencyInfo for testing."""
    return DependencyInfo(
        name=name,
        version=version,
        ecosystem=ecosystem,
        vulnerabilities=vulnerabilities or [],
    )


def _make_scan_result(
    dependencies: list[DependencyInfo] | None = None,
) -> ScanResult:
    """Create a ScanResult for testing."""
    return ScanResult(
        scan_id="scan-001",
        repository="test-repo",
        dependencies=dependencies or [],
    )


def _make_secret(
    secret_id: str = "sec-001",
    secret_type: SecretType = SecretType.GITHUB_TOKEN,
    file_path: str = "config.py",
    line_number: int = 42,
    severity: VulnerabilitySeverity = VulnerabilitySeverity.HIGH,
    confidence: float = 0.95,
    is_in_history: bool = False,
) -> SecretFinding:
    """Create a SecretFinding for testing."""
    return SecretFinding(
        id=secret_id,
        secret_type=secret_type,
        file_path=file_path,
        line_number=line_number,
        column_start=0,
        column_end=40,
        matched_text="ghp_****XXXX",
        context_line='TOKEN = "ghp_****XXXX"',
        severity=severity,
        confidence=confidence,
        is_in_history=is_in_history,
    )


def _make_secrets_result(
    secrets: list[SecretFinding] | None = None,
) -> SecretsScanResult:
    """Create a SecretsScanResult for testing."""
    return SecretsScanResult(
        scan_id="sec-scan-001",
        repository="test-repo",
        secrets=secrets or [],
    )


def _make_sast_finding(
    rule_id: str = "python.lang.security.injection.sql-injection",
    file_path: str = "app/db.py",
    line_start: int = 42,
    line_end: int = 42,
    column_start: int = 0,
    column_end: int = 50,
    message: str = "Possible SQL injection via string formatting",
    severity: SASTSeverity = SASTSeverity.ERROR,
    confidence: float = 0.9,
    language: str = "python",
    cwe_ids: list[str] | None = None,
    owasp_category: OWASPCategory = OWASPCategory.A03_INJECTION,
    remediation: str = "Use parameterized queries",
) -> SASTFinding:
    """Create a SASTFinding for testing."""
    if cwe_ids is None:
        cwe_ids = ["CWE-89"]
    return SASTFinding(
        rule_id=rule_id,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        column_start=column_start,
        column_end=column_end,
        message=message,
        severity=severity,
        confidence=confidence,
        language=language,
        cwe_ids=cwe_ids,
        owasp_category=owasp_category,
        remediation=remediation,
    )


def _make_sast_result(
    findings: list[SASTFinding] | None = None,
) -> SASTScanResult:
    """Create a SASTScanResult for testing."""
    return SASTScanResult(
        repository_path="/tmp/test-repo",
        scan_id="sast-scan-001",
        findings=findings or [],
        scanned_files=10,
        skipped_files=0,
        scan_duration_ms=1234.5,
        languages_detected=["python"],
        rules_used=["security"],
    )


# ============================================================================
# emit_scan_events Tests
# ============================================================================


class TestEmitScanEvents:
    """Tests for emit_scan_events()."""

    @pytest.mark.asyncio
    async def test_happy_path_single_high_vuln(self):
        """Emit event with a single high-severity vulnerability."""
        vuln = _make_vuln(severity=VulnerabilitySeverity.HIGH)
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        mock_emitter.emit.assert_called_once()
        event = mock_emitter.emit.call_args[0][0]
        assert isinstance(event, SecurityEvent)
        assert event.event_type == SecurityEventType.VULNERABILITY_DETECTED
        assert event.severity == SecuritySeverity.HIGH
        assert event.repository == "repo-1"
        assert event.scan_id == "scan-1"
        assert len(event.findings) == 1
        assert event.findings[0].finding_type == "vulnerability"
        assert event.findings[0].severity == SecuritySeverity.HIGH

    @pytest.mark.asyncio
    async def test_critical_vulnerability_triggers_critical_event(self):
        """Critical vulnerability sets event type to CRITICAL_VULNERABILITY."""
        vuln = _make_vuln(severity=VulnerabilitySeverity.CRITICAL, cvss_score=9.8)
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.CRITICAL_VULNERABILITY
        assert event.severity == SecuritySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_medium_severity_only_uses_scan_completed(self):
        """Only medium-severity vulns emit SCAN_COMPLETED event type."""
        vuln = _make_vuln(severity=VulnerabilitySeverity.MEDIUM)
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.SCAN_COMPLETED
        assert event.severity == SecuritySeverity.MEDIUM

    @pytest.mark.asyncio
    async def test_low_severity_uses_scan_completed(self):
        """Only low-severity vulns emit SCAN_COMPLETED event type."""
        vuln = _make_vuln(severity=VulnerabilitySeverity.LOW)
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.SCAN_COMPLETED
        assert event.severity == SecuritySeverity.MEDIUM  # Overall stays MEDIUM

    @pytest.mark.asyncio
    async def test_no_findings_early_return(self):
        """No vulnerabilities in deps means no event emitted."""
        dep = _make_dep(vulnerabilities=[])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        mock_emitter.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_dependencies_no_event(self):
        """Empty dependencies list means no event emitted."""
        result = _make_scan_result(dependencies=[])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        mock_emitter.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_workspace_id_propagated(self):
        """workspace_id is passed through to the emitted event."""
        vuln = _make_vuln()
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1", workspace_id="ws-123")

        event = mock_emitter.emit.call_args[0][0]
        assert event.workspace_id == "ws-123"

    @pytest.mark.asyncio
    async def test_workspace_id_none_by_default(self):
        """workspace_id defaults to None when not provided."""
        vuln = _make_vuln()
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.workspace_id is None

    @pytest.mark.asyncio
    async def test_findings_truncated_to_20(self):
        """More than 20 findings are truncated to 20 in the event."""
        vulns = [
            _make_vuln(vuln_id=f"CVE-2024-{i:04d}", severity=VulnerabilitySeverity.MEDIUM)
            for i in range(25)
        ]
        dep = _make_dep(vulnerabilities=vulns)
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert len(event.findings) == 20

    @pytest.mark.asyncio
    async def test_multiple_dependencies_multiple_vulns(self):
        """Multiple dependencies with multiple vulns aggregate findings."""
        vuln1 = _make_vuln(vuln_id="CVE-2024-0001", severity=VulnerabilitySeverity.HIGH)
        vuln2 = _make_vuln(vuln_id="CVE-2024-0002", severity=VulnerabilitySeverity.CRITICAL)
        dep1 = _make_dep(name="requests", vulnerabilities=[vuln1])
        dep2 = _make_dep(name="flask", version="2.0.0", vulnerabilities=[vuln2])
        result = _make_scan_result(dependencies=[dep1, dep2])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert len(event.findings) == 2
        # Critical takes priority
        assert event.event_type == SecurityEventType.CRITICAL_VULNERABILITY
        assert event.severity == SecuritySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_finding_fields_correctly_mapped(self):
        """Verify all SecurityFinding fields are correctly populated."""
        vuln = _make_vuln(
            vuln_id="CVE-2024-9999",
            title="Buffer Overflow",
            description="Stack buffer overflow in parser",
            severity=VulnerabilitySeverity.CRITICAL,
            cvss_score=9.8,
            remediation_guidance="Upgrade to v3.0",
        )
        dep = _make_dep(name="libxml2", version="2.9.10", ecosystem="system", vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert finding.id == "CVE-2024-9999"
        assert finding.finding_type == "vulnerability"
        assert finding.severity == SecuritySeverity.CRITICAL
        assert finding.title == "Buffer Overflow"
        assert finding.description == "Stack buffer overflow in parser"
        assert finding.cve_id == "CVE-2024-9999"
        assert finding.package_name == "libxml2"
        assert finding.package_version == "2.9.10"
        assert finding.recommendation == "Upgrade to v3.0"
        assert finding.metadata["ecosystem"] == "system"
        assert finding.metadata["cvss_score"] == 9.8

    @pytest.mark.asyncio
    async def test_vuln_without_title_uses_id(self):
        """Vulnerability with no title falls back to id."""
        vuln = VulnerabilityFinding(
            id="CVE-2024-0001",
            title="",
            description="desc",
            severity=VulnerabilitySeverity.HIGH,
        )
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        finding = mock_emitter.emit.call_args[0][0].findings[0]
        # title is falsy (""), so vuln.id is used next
        assert finding.title == "CVE-2024-0001"

    @pytest.mark.asyncio
    async def test_vuln_without_title_or_id_uses_unknown(self):
        """Vulnerability with no title and no id falls back to 'Unknown'."""
        vuln = VulnerabilityFinding(
            id="",
            title="",
            description="desc",
            severity=VulnerabilitySeverity.HIGH,
        )
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert finding.title == "Unknown"

    @pytest.mark.asyncio
    async def test_unknown_severity_defaults_to_medium(self):
        """Unknown/unmapped severity value defaults to MEDIUM."""
        vuln = _make_vuln(severity=VulnerabilitySeverity.UNKNOWN)
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert finding.severity == SecuritySeverity.MEDIUM

    @pytest.mark.asyncio
    async def test_severity_string_fallback(self):
        """When severity has no .value attribute, str() fallback is used."""
        vuln = _make_vuln()
        # Replace severity with a plain string mock (no .value attribute)
        vuln.severity = "high"  # type: ignore[assignment]
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert finding.severity == SecuritySeverity.HIGH

    @pytest.mark.asyncio
    async def test_emitter_error_is_caught(self):
        """Exceptions from emitter.emit are caught and logged."""
        vuln = _make_vuln()
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        mock_emitter.emit.side_effect = RuntimeError("Connection failed")
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            # Should not raise
            await emit_scan_events(result, "repo-1", "scan-1")

    @pytest.mark.asyncio
    async def test_get_emitter_error_is_caught(self):
        """Exceptions from get_security_emitter are caught."""
        vuln = _make_vuln()
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            side_effect=RuntimeError("Emitter unavailable"),
        ):
            # Should not raise
            await emit_scan_events(result, "repo-1", "scan-1")

    @pytest.mark.asyncio
    async def test_value_error_is_caught(self):
        """ValueError from emitter is caught."""
        vuln = _make_vuln()
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        mock_emitter.emit.side_effect = ValueError("Bad value")
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

    @pytest.mark.asyncio
    async def test_type_error_is_caught(self):
        """TypeError from emitter is caught."""
        vuln = _make_vuln()
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        mock_emitter.emit.side_effect = TypeError("Bad type")
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

    @pytest.mark.asyncio
    async def test_key_error_is_caught(self):
        """KeyError from emitter is caught."""
        vuln = _make_vuln()
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        mock_emitter.emit.side_effect = KeyError("missing_key")
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

    @pytest.mark.asyncio
    async def test_mixed_severity_critical_takes_priority(self):
        """When both critical and high vulns exist, critical takes priority."""
        vuln_high = _make_vuln(vuln_id="CVE-H", severity=VulnerabilitySeverity.HIGH)
        vuln_crit = _make_vuln(vuln_id="CVE-C", severity=VulnerabilitySeverity.CRITICAL)
        vuln_low = _make_vuln(vuln_id="CVE-L", severity=VulnerabilitySeverity.LOW)
        dep = _make_dep(vulnerabilities=[vuln_high, vuln_crit, vuln_low])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.CRITICAL_VULNERABILITY
        assert event.severity == SecuritySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_high_only_uses_vulnerability_detected(self):
        """Only high vulns (no critical) use VULNERABILITY_DETECTED event type."""
        vuln1 = _make_vuln(vuln_id="CVE-1", severity=VulnerabilitySeverity.HIGH)
        vuln2 = _make_vuln(vuln_id="CVE-2", severity=VulnerabilitySeverity.HIGH)
        dep = _make_dep(vulnerabilities=[vuln1, vuln2])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.VULNERABILITY_DETECTED
        assert event.severity == SecuritySeverity.HIGH

    @pytest.mark.asyncio
    async def test_vuln_no_description_defaults_to_empty(self):
        """Vulnerability with no description defaults to empty string."""
        vuln = VulnerabilityFinding(
            id="CVE-2024-0001",
            title="Test",
            description="",
            severity=VulnerabilitySeverity.HIGH,
        )
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert finding.description == ""

    @pytest.mark.asyncio
    async def test_vuln_no_remediation(self):
        """Vulnerability with no remediation_guidance passes None."""
        vuln = _make_vuln(remediation_guidance=None)
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert finding.recommendation is None

    @pytest.mark.asyncio
    async def test_vuln_without_cvss_score(self):
        """Vulnerability without cvss_score attribute stores None in metadata."""
        vuln = _make_vuln(cvss_score=None)
        dep = _make_dep(vulnerabilities=[vuln])
        result = _make_scan_result(dependencies=[dep])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_scan_events(result, "repo-1", "scan-1")

        finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert finding.metadata["cvss_score"] is None


# ============================================================================
# emit_secrets_events Tests
# ============================================================================


class TestEmitSecretsEvents:
    """Tests for emit_secrets_events()."""

    @pytest.mark.asyncio
    async def test_happy_path_single_high_secret(self):
        """Emit event with a single high-severity secret."""
        secret = _make_secret(severity=VulnerabilitySeverity.HIGH)
        result = _make_secrets_result(secrets=[secret])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

        mock_emitter.emit.assert_called_once()
        event = mock_emitter.emit.call_args[0][0]
        assert isinstance(event, SecurityEvent)
        assert event.event_type == SecurityEventType.SECRET_DETECTED
        assert event.severity == SecuritySeverity.HIGH
        assert len(event.findings) == 1
        assert event.findings[0].finding_type == "secret"

    @pytest.mark.asyncio
    async def test_critical_secret_triggers_critical_event(self):
        """Critical secret sets event type to CRITICAL_SECRET."""
        secret = _make_secret(severity=VulnerabilitySeverity.CRITICAL)
        result = _make_secrets_result(secrets=[secret])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.CRITICAL_SECRET
        assert event.severity == SecuritySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_medium_severity_secret_uses_scan_completed(self):
        """Only medium-severity secrets emit SCAN_COMPLETED event type."""
        secret = _make_secret(severity=VulnerabilitySeverity.MEDIUM)
        result = _make_secrets_result(secrets=[secret])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.SCAN_COMPLETED
        assert event.severity == SecuritySeverity.MEDIUM

    @pytest.mark.asyncio
    async def test_low_severity_secret_uses_scan_completed(self):
        """Low-severity secrets emit SCAN_COMPLETED event type."""
        secret = _make_secret(severity=VulnerabilitySeverity.LOW)
        result = _make_secrets_result(secrets=[secret])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.SCAN_COMPLETED
        assert event.severity == SecuritySeverity.MEDIUM

    @pytest.mark.asyncio
    async def test_no_secrets_early_return(self):
        """No secrets found means no event emitted."""
        result = _make_secrets_result(secrets=[])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

        mock_emitter.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_workspace_id_propagated(self):
        """workspace_id is passed through to the emitted event."""
        secret = _make_secret()
        result = _make_secrets_result(secrets=[secret])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1", workspace_id="ws-456")

        event = mock_emitter.emit.call_args[0][0]
        assert event.workspace_id == "ws-456"

    @pytest.mark.asyncio
    async def test_secrets_truncated_to_20(self):
        """More than 20 secrets are truncated to 20 in the event."""
        secrets = [
            _make_secret(secret_id=f"sec-{i:03d}", severity=VulnerabilitySeverity.MEDIUM)
            for i in range(25)
        ]
        result = _make_secrets_result(secrets=secrets)

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert len(event.findings) == 20

    @pytest.mark.asyncio
    async def test_secret_finding_fields_correctly_mapped(self):
        """Verify all SecurityFinding fields are correctly populated for secrets."""
        secret = _make_secret(
            secret_id="sec-abc",
            secret_type=SecretType.AWS_ACCESS_KEY,
            file_path="deploy/config.py",
            line_number=100,
            severity=VulnerabilitySeverity.CRITICAL,
            confidence=0.99,
            is_in_history=True,
        )
        result = _make_secrets_result(secrets=[secret])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

        finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert finding.id == "sec-abc"
        assert finding.finding_type == "secret"
        assert finding.severity == SecuritySeverity.CRITICAL
        assert "Exposed" in finding.title
        assert "aws_access_key" in finding.title
        assert "deploy/config.py" in finding.description
        assert finding.file_path == "deploy/config.py"
        assert finding.line_number == 100
        assert finding.recommendation == "Rotate the credential immediately and remove from codebase"
        assert finding.metadata["secret_type"] == "aws_access_key"
        assert finding.metadata["confidence"] == 0.99
        assert finding.metadata["is_in_history"] is True

    @pytest.mark.asyncio
    async def test_secret_type_string_fallback(self):
        """When secret_type has no .value attribute, str() fallback is used."""
        secret = _make_secret()
        # Replace secret_type with a plain string (no .value attribute)
        secret.secret_type = "custom_token"  # type: ignore[assignment]
        result = _make_secrets_result(secrets=[secret])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

        finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert "custom_token" in finding.title
        assert finding.metadata["secret_type"] == "custom_token"

    @pytest.mark.asyncio
    async def test_secret_severity_string_fallback(self):
        """When severity has no .value attribute, str() fallback is used."""
        secret = _make_secret()
        secret.severity = "critical"  # type: ignore[assignment]
        result = _make_secrets_result(secrets=[secret])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

        finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert finding.severity == SecuritySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_unknown_severity_defaults_to_high_for_secrets(self):
        """Unknown/unmapped severity value defaults to HIGH for secrets."""
        secret = _make_secret(severity=VulnerabilitySeverity.UNKNOWN)
        result = _make_secrets_result(secrets=[secret])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

        finding = mock_emitter.emit.call_args[0][0].findings[0]
        # Secrets default to HIGH for unknown severity (unlike vulns which default MEDIUM)
        assert finding.severity == SecuritySeverity.HIGH

    @pytest.mark.asyncio
    async def test_mixed_severity_secrets_critical_priority(self):
        """When both critical and high secrets exist, critical takes priority."""
        sec_high = _make_secret(secret_id="s1", severity=VulnerabilitySeverity.HIGH)
        sec_crit = _make_secret(secret_id="s2", severity=VulnerabilitySeverity.CRITICAL)
        result = _make_secrets_result(secrets=[sec_high, sec_crit])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.CRITICAL_SECRET
        assert event.severity == SecuritySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_high_only_secrets_use_secret_detected(self):
        """Only high secrets (no critical) use SECRET_DETECTED event type."""
        sec1 = _make_secret(secret_id="s1", severity=VulnerabilitySeverity.HIGH)
        sec2 = _make_secret(secret_id="s2", severity=VulnerabilitySeverity.HIGH)
        result = _make_secrets_result(secrets=[sec1, sec2])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.event_type == SecurityEventType.SECRET_DETECTED
        assert event.severity == SecuritySeverity.HIGH

    @pytest.mark.asyncio
    async def test_emitter_error_is_caught_secrets(self):
        """Exceptions from emitter.emit are caught for secrets events."""
        secret = _make_secret()
        result = _make_secrets_result(secrets=[secret])

        mock_emitter = AsyncMock()
        mock_emitter.emit.side_effect = RuntimeError("Connection failed")
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

    @pytest.mark.asyncio
    async def test_get_emitter_error_is_caught_secrets(self):
        """Exceptions from get_security_emitter are caught for secrets."""
        secret = _make_secret()
        result = _make_secrets_result(secrets=[secret])

        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            side_effect=ValueError("Bad emitter"),
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

    @pytest.mark.asyncio
    async def test_key_error_is_caught_secrets(self):
        """KeyError from emitter is caught for secrets."""
        secret = _make_secret()
        result = _make_secrets_result(secrets=[secret])

        mock_emitter = AsyncMock()
        mock_emitter.emit.side_effect = KeyError("missing")
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")

    @pytest.mark.asyncio
    async def test_type_error_is_caught_secrets(self):
        """TypeError from emitter is caught for secrets."""
        secret = _make_secret()
        result = _make_secrets_result(secrets=[secret])

        mock_emitter = AsyncMock()
        mock_emitter.emit.side_effect = TypeError("bad type")
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_secrets_events(result, "repo-1", "scan-1")


# ============================================================================
# emit_sast_events Tests
# ============================================================================


class TestEmitSastEvents:
    """Tests for emit_sast_events()."""

    @pytest.mark.asyncio
    async def test_happy_path_error_severity(self):
        """Emit event for SAST finding with error severity."""
        finding = _make_sast_finding(severity=SASTSeverity.ERROR)
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        mock_emitter.emit.assert_called_once()
        event = mock_emitter.emit.call_args[0][0]
        assert isinstance(event, SecurityEvent)
        assert event.event_type == SecurityEventType.SAST_CRITICAL
        assert event.severity == SecuritySeverity.HIGH
        assert event.repository == "repo-1"
        assert event.scan_id == "sast-scan-1"
        assert len(event.findings) == 1

    @pytest.mark.asyncio
    async def test_critical_severity(self):
        """SAST finding with critical severity maps to CRITICAL."""
        finding = _make_sast_finding(severity=SASTSeverity.CRITICAL)
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.severity == SecuritySeverity.CRITICAL
        assert event.findings[0].severity == SecuritySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_warning_severity_filtered_out(self):
        """SAST findings with warning severity are skipped."""
        finding = _make_sast_finding(severity=SASTSeverity.WARNING)
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        mock_emitter.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_info_severity_filtered_out(self):
        """SAST findings with info severity are skipped."""
        finding = _make_sast_finding(severity=SASTSeverity.INFO)
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        mock_emitter.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_findings_no_events(self):
        """Empty findings list means no events emitted."""
        result = _make_sast_result(findings=[])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        mock_emitter.emit.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_severities_only_critical_and_error_emitted(self):
        """Only critical and error findings are emitted; others filtered."""
        f_info = _make_sast_finding(rule_id="info-rule", severity=SASTSeverity.INFO)
        f_warn = _make_sast_finding(rule_id="warn-rule", severity=SASTSeverity.WARNING)
        f_error = _make_sast_finding(rule_id="error-rule", severity=SASTSeverity.ERROR)
        f_crit = _make_sast_finding(rule_id="crit-rule", severity=SASTSeverity.CRITICAL)
        result = _make_sast_result(findings=[f_info, f_warn, f_error, f_crit])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        assert mock_emitter.emit.call_count == 2

    @pytest.mark.asyncio
    async def test_sast_finding_fields_correctly_mapped(self):
        """Verify all SecurityFinding fields are correctly populated for SAST."""
        finding = _make_sast_finding(
            rule_id="python.security.xss",
            file_path="web/views.py",
            line_start=99,
            message="Cross-site scripting via template rendering. " * 3,  # Long message
            severity=SASTSeverity.CRITICAL,
            confidence=0.95,
            language="python",
            cwe_ids=["CWE-79", "CWE-80"],
            owasp_category=OWASPCategory.A03_INJECTION,
            remediation="Use template auto-escaping",
        )
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        event = mock_emitter.emit.call_args[0][0]
        sec_finding = event.findings[0]
        # ID is composed of scan_id:rule_id:line_start
        assert sec_finding.id == "sast-scan-1:python.security.xss:99"
        assert sec_finding.finding_type == "sast"
        assert sec_finding.severity == SecuritySeverity.CRITICAL
        # Title is truncated to 100 chars
        assert len(sec_finding.title) <= 100
        assert sec_finding.description == finding.message
        assert sec_finding.file_path == "web/views.py"
        assert sec_finding.line_number == 99
        assert sec_finding.cve_id == "CWE-79"  # First CWE ID
        assert sec_finding.recommendation == "Use template auto-escaping"
        assert sec_finding.metadata["category"] == OWASPCategory.A03_INJECTION.value
        assert sec_finding.metadata["source"] == "sast_scanner"
        assert sec_finding.metadata["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_sast_no_cwe_ids(self):
        """SAST finding with empty cwe_ids sets cve_id to None."""
        finding = _make_sast_finding(cwe_ids=[])
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        sec_finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert sec_finding.cve_id is None

    @pytest.mark.asyncio
    async def test_sast_no_remediation(self):
        """SAST finding without remediation uses default message."""
        finding = _make_sast_finding(remediation="")
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        sec_finding = mock_emitter.emit.call_args[0][0].findings[0]
        # Empty string is falsy, so the `or` kicks in
        assert sec_finding.recommendation == "Review and fix the security issue"

    @pytest.mark.asyncio
    async def test_sast_with_remediation(self):
        """SAST finding with remediation passes it through."""
        finding = _make_sast_finding(remediation="Apply input validation")
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        sec_finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert sec_finding.recommendation == "Apply input validation"

    @pytest.mark.asyncio
    async def test_sast_event_metadata(self):
        """SAST event metadata includes rule_id, language, and owasp_category."""
        finding = _make_sast_finding(
            rule_id="go.injection.cmd-injection",
            language="go",
            owasp_category=OWASPCategory.A03_INJECTION,
        )
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.metadata["rule_id"] == "go.injection.cmd-injection"
        assert event.metadata["language"] == "go"
        assert event.metadata["owasp_category"] == OWASPCategory.A03_INJECTION.value
        assert event.source == "sast_scanner"

    @pytest.mark.asyncio
    async def test_sast_workspace_id_propagated(self):
        """workspace_id is not explicitly set by emit_sast_events (no param in event)."""
        finding = _make_sast_finding()
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1", workspace_id="ws-789")

        # emit_sast_events does NOT pass workspace_id to the event (unlike the others)
        # The function signature accepts it but the current implementation doesn't use it
        # This test documents the current behavior
        event = mock_emitter.emit.call_args[0][0]
        # The SecurityEvent default for workspace_id is None
        # Check that event was still emitted regardless
        assert event.repository == "repo-1"

    @pytest.mark.asyncio
    async def test_sast_each_finding_emits_separate_event(self):
        """Each qualifying SAST finding emits its own separate event."""
        f1 = _make_sast_finding(rule_id="rule-1", line_start=10, severity=SASTSeverity.ERROR)
        f2 = _make_sast_finding(rule_id="rule-2", line_start=20, severity=SASTSeverity.CRITICAL)
        f3 = _make_sast_finding(rule_id="rule-3", line_start=30, severity=SASTSeverity.ERROR)
        result = _make_sast_result(findings=[f1, f2, f3])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        assert mock_emitter.emit.call_count == 3
        events = [call[0][0] for call in mock_emitter.emit.call_args_list]
        # Each event has exactly 1 finding
        for event in events:
            assert len(event.findings) == 1

    @pytest.mark.asyncio
    async def test_sast_title_truncation(self):
        """SAST finding message longer than 100 chars is truncated for title."""
        long_message = "A" * 200
        finding = _make_sast_finding(message=long_message)
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        sec_finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert len(sec_finding.title) == 100
        # But description has full message
        assert len(sec_finding.description) == 200

    @pytest.mark.asyncio
    async def test_sast_emitter_error_is_caught(self):
        """Exceptions from emitter.emit are caught for SAST events."""
        finding = _make_sast_finding()
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        mock_emitter.emit.side_effect = RuntimeError("Connection failed")
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

    @pytest.mark.asyncio
    async def test_sast_get_emitter_error_is_caught(self):
        """Exceptions from get_security_emitter are caught for SAST."""
        finding = _make_sast_finding()
        result = _make_sast_result(findings=[finding])

        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            side_effect=TypeError("Bad emitter type"),
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

    @pytest.mark.asyncio
    async def test_sast_value_error_is_caught(self):
        """ValueError from emitter is caught for SAST."""
        finding = _make_sast_finding()
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        mock_emitter.emit.side_effect = ValueError("bad value")
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

    @pytest.mark.asyncio
    async def test_sast_key_error_is_caught(self):
        """KeyError from emitter is caught for SAST."""
        finding = _make_sast_finding()
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        mock_emitter.emit.side_effect = KeyError("missing")
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

    @pytest.mark.asyncio
    async def test_sast_error_severity_maps_to_high(self):
        """SAST error severity maps to SecuritySeverity.HIGH."""
        finding = _make_sast_finding(severity=SASTSeverity.ERROR)
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.severity == SecuritySeverity.HIGH
        assert event.findings[0].severity == SecuritySeverity.HIGH

    @pytest.mark.asyncio
    async def test_sast_critical_severity_maps_to_critical(self):
        """SAST critical severity maps to SecuritySeverity.CRITICAL."""
        finding = _make_sast_finding(severity=SASTSeverity.CRITICAL)
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        event = mock_emitter.emit.call_args[0][0]
        assert event.severity == SecuritySeverity.CRITICAL
        assert event.findings[0].severity == SecuritySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_sast_finding_id_composition(self):
        """SAST finding ID is composed as scan_id:rule_id:line_start."""
        finding = _make_sast_finding(
            rule_id="python.security.crypto.weak-hash",
            line_start=77,
        )
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "my-scan-123")

        sec_finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert sec_finding.id == "my-scan-123:python.security.crypto.weak-hash:77"

    @pytest.mark.asyncio
    async def test_sast_multiple_cwe_ids_uses_first(self):
        """When multiple CWE IDs present, only the first is used as cve_id."""
        finding = _make_sast_finding(cwe_ids=["CWE-79", "CWE-80", "CWE-116"])
        result = _make_sast_result(findings=[finding])

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        sec_finding = mock_emitter.emit.call_args[0][0].findings[0]
        assert sec_finding.cve_id == "CWE-79"

    @pytest.mark.asyncio
    async def test_many_critical_findings_each_emits_event(self):
        """Verify that many critical findings each generate an event."""
        findings = [
            _make_sast_finding(
                rule_id=f"rule-{i}",
                line_start=i * 10,
                severity=SASTSeverity.CRITICAL,
            )
            for i in range(10)
        ]
        result = _make_sast_result(findings=findings)

        mock_emitter = AsyncMock()
        with patch(
            "aragora.server.handlers.codebase.security.events.get_security_emitter",
            return_value=mock_emitter,
        ):
            await emit_sast_events(result, "repo-1", "sast-scan-1")

        assert mock_emitter.emit.call_count == 10
