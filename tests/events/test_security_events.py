"""
Tests for security events module.

Covers SecurityEventType, SecuritySeverity, SecurityFinding, SecurityEvent,
SecurityEventEmitter, debate integration, convenience functions, and singleton
management.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.events.security_events import (
    SecurityEvent,
    SecurityEventEmitter,
    SecurityEventType,
    SecurityFinding,
    SecuritySeverity,
    create_scan_completed_event,
    create_secret_event,
    create_vulnerability_event,
    get_security_debate_result,
    get_security_debate_runner,
    get_security_emitter,
    is_secret_finding,
    list_security_debates,
    register_security_debate_runner,
    set_security_emitter,
    _security_debate_results,
    _store_security_debate_result,
)


# =============================================================================
# SecurityEventType enum
# =============================================================================


class TestSecurityEventType:
    """Tests for SecurityEventType enum values and classification."""

    def test_vulnerability_event_types(self):
        """Should define vulnerability-related event types."""
        assert SecurityEventType.VULNERABILITY_DETECTED == "vulnerability_detected"
        assert SecurityEventType.CRITICAL_VULNERABILITY == "critical_vulnerability"
        assert SecurityEventType.VULNERABILITY_RESOLVED == "vulnerability_resolved"

    def test_cve_event_types(self):
        """Should define CVE-specific event types."""
        assert SecurityEventType.CRITICAL_CVE == "critical_cve"

    def test_secret_event_types(self):
        """Should define secret-related event types."""
        assert SecurityEventType.SECRET_DETECTED == "secret_detected"
        assert SecurityEventType.CRITICAL_SECRET == "critical_secret"
        assert SecurityEventType.SECRET_ROTATED == "secret_rotated"

    def test_sast_event_types(self):
        """Should define SAST event types."""
        assert SecurityEventType.SAST_CRITICAL == "sast_critical"

    def test_threat_intel_event_types(self):
        """Should define threat intelligence event types."""
        assert SecurityEventType.THREAT_DETECTED == "threat_detected"

    def test_scan_event_types(self):
        """Should define scan lifecycle event types."""
        assert SecurityEventType.SCAN_STARTED == "scan_started"
        assert SecurityEventType.SCAN_COMPLETED == "scan_completed"
        assert SecurityEventType.SCAN_FAILED == "scan_failed"

    def test_debate_event_types(self):
        """Should define debate-related event types."""
        assert SecurityEventType.SECURITY_DEBATE_REQUESTED == "security_debate_requested"
        assert SecurityEventType.SECURITY_DEBATE_STARTED == "security_debate_started"
        assert SecurityEventType.SECURITY_DEBATE_COMPLETED == "security_debate_completed"

    def test_event_type_is_string_enum(self):
        """Event types should be usable as strings."""
        assert isinstance(SecurityEventType.VULNERABILITY_DETECTED, str)
        assert SecurityEventType.VULNERABILITY_DETECTED == "vulnerability_detected"

    def test_all_event_types_exist(self):
        """All expected event types should be defined."""
        expected = {
            "VULNERABILITY_DETECTED",
            "CRITICAL_VULNERABILITY",
            "VULNERABILITY_RESOLVED",
            "CRITICAL_CVE",
            "SECRET_DETECTED",
            "CRITICAL_SECRET",
            "SECRET_ROTATED",
            "SAST_CRITICAL",
            "THREAT_DETECTED",
            "SCAN_STARTED",
            "SCAN_COMPLETED",
            "SCAN_FAILED",
            "SECURITY_DEBATE_REQUESTED",
            "SECURITY_DEBATE_STARTED",
            "SECURITY_DEBATE_COMPLETED",
        }
        actual = {member.name for member in SecurityEventType}
        assert expected == actual


# =============================================================================
# SecuritySeverity enum
# =============================================================================


class TestSecuritySeverity:
    """Tests for SecuritySeverity levels."""

    def test_severity_levels_exist(self):
        """Should define all standard severity levels."""
        assert SecuritySeverity.CRITICAL == "critical"
        assert SecuritySeverity.HIGH == "high"
        assert SecuritySeverity.MEDIUM == "medium"
        assert SecuritySeverity.LOW == "low"
        assert SecuritySeverity.INFO == "info"

    def test_severity_is_string_enum(self):
        """Severity levels should be usable as strings."""
        assert isinstance(SecuritySeverity.CRITICAL, str)
        assert SecuritySeverity.CRITICAL == "critical"

    def test_severity_count(self):
        """Should have exactly 5 severity levels."""
        assert len(SecuritySeverity) == 5


# =============================================================================
# SecurityFinding dataclass
# =============================================================================


class TestSecurityFinding:
    """Tests for SecurityFinding creation and serialization."""

    def _make_finding(self, **overrides):
        """Helper to create a SecurityFinding with defaults."""
        defaults = {
            "id": "finding-001",
            "finding_type": "vulnerability",
            "severity": SecuritySeverity.HIGH,
            "title": "SQL Injection in login",
            "description": "Unsanitized user input in SQL query",
        }
        defaults.update(overrides)
        return SecurityFinding(**defaults)

    def test_create_finding_with_required_fields(self):
        """Should create a finding with only required fields."""
        finding = self._make_finding()
        assert finding.id == "finding-001"
        assert finding.finding_type == "vulnerability"
        assert finding.severity == SecuritySeverity.HIGH
        assert finding.title == "SQL Injection in login"
        assert finding.description == "Unsanitized user input in SQL query"

    def test_optional_fields_default_to_none(self):
        """Optional fields should default to None."""
        finding = self._make_finding()
        assert finding.file_path is None
        assert finding.line_number is None
        assert finding.cve_id is None
        assert finding.package_name is None
        assert finding.package_version is None
        assert finding.recommendation is None

    def test_metadata_defaults_to_empty_dict(self):
        """Metadata should default to an empty dict."""
        finding = self._make_finding()
        assert finding.metadata == {}

    def test_create_finding_with_all_fields(self):
        """Should create a finding with all fields populated."""
        finding = self._make_finding(
            file_path="src/auth.py",
            line_number=42,
            cve_id="CVE-2024-12345",
            package_name="django",
            package_version="3.2.1",
            recommendation="Upgrade to django>=4.0",
            metadata={"cvss": 9.8, "exploit_available": True},
        )
        assert finding.file_path == "src/auth.py"
        assert finding.line_number == 42
        assert finding.cve_id == "CVE-2024-12345"
        assert finding.package_name == "django"
        assert finding.package_version == "3.2.1"
        assert finding.recommendation == "Upgrade to django>=4.0"
        assert finding.metadata["cvss"] == 9.8

    def test_to_dict_serialization(self):
        """Should serialize all fields to a dictionary."""
        finding = self._make_finding(
            cve_id="CVE-2024-99999",
            package_name="requests",
            package_version="2.28.0",
            file_path="requirements.txt",
            line_number=10,
            recommendation="Upgrade requests",
            metadata={"source": "trivy"},
        )
        d = finding.to_dict()
        assert d["id"] == "finding-001"
        assert d["finding_type"] == "vulnerability"
        assert d["severity"] == "high"
        assert d["title"] == "SQL Injection in login"
        assert d["description"] == "Unsanitized user input in SQL query"
        assert d["file_path"] == "requirements.txt"
        assert d["line_number"] == 10
        assert d["cve_id"] == "CVE-2024-99999"
        assert d["package_name"] == "requests"
        assert d["package_version"] == "2.28.0"
        assert d["recommendation"] == "Upgrade requests"
        assert d["metadata"] == {"source": "trivy"}

    def test_to_dict_severity_is_string_value(self):
        """Severity should be serialized as the enum string value."""
        finding = self._make_finding(severity=SecuritySeverity.CRITICAL)
        d = finding.to_dict()
        assert d["severity"] == "critical"

    def test_to_dict_is_json_serializable(self):
        """The dict representation should be JSON serializable."""
        finding = self._make_finding(metadata={"nested": {"deep": True}})
        d = finding.to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["id"] == "finding-001"
        assert deserialized["metadata"]["nested"]["deep"] is True

    def test_to_dict_redacts_secret_findings(self):
        """Direct finding serialization should not leak secret material."""
        finding = self._make_finding(
            finding_type="secret",
            severity=SecuritySeverity.CRITICAL,
            title="Token literal-secret",
            description="literal-secret",
            metadata={"secret_type": "api_token", "raw_secret": "literal-secret"},
        )

        d = finding.to_dict()
        serialized = json.dumps(d)

        assert d["finding_type"] == "secret"
        assert d["title"] == "Secret finding"
        assert d["description"] == "[redacted secret finding description]"
        assert d["metadata"] == {"secret_type": "api_token"}
        assert "literal-secret" not in serialized


# =============================================================================
# SecurityEvent dataclass
# =============================================================================


class TestSecurityEvent:
    """Tests for SecurityEvent creation, properties, and serialization."""

    def _make_event(self, **overrides):
        """Helper to create a SecurityEvent with defaults."""
        return SecurityEvent(**overrides)

    def _make_finding(self, severity=SecuritySeverity.HIGH, **overrides):
        """Helper to create a SecurityFinding."""
        defaults = {
            "id": str(uuid.uuid4()),
            "finding_type": "vulnerability",
            "severity": severity,
            "title": "Test finding",
            "description": "A test finding",
        }
        defaults.update(overrides)
        return SecurityFinding(**defaults)

    def test_create_event_with_defaults(self):
        """Should create an event with sensible defaults."""
        event = self._make_event()
        assert event.id is not None and len(event.id) > 0
        assert event.event_type == SecurityEventType.VULNERABILITY_DETECTED
        assert isinstance(event.timestamp, datetime)
        assert event.severity == SecuritySeverity.MEDIUM
        assert event.source == "sast"
        assert event.repository is None
        assert event.scan_id is None
        assert event.workspace_id is None
        assert event.findings == []
        assert event.debate_requested is False
        assert event.debate_id is None
        assert event.debate_question is None
        assert event.correlation_id is None
        assert event.metadata == {}

    def test_create_event_with_all_fields(self):
        """Should create an event with all fields specified."""
        ts = datetime.now(timezone.utc)
        finding = self._make_finding()
        event = self._make_event(
            id="evt-123",
            event_type=SecurityEventType.CRITICAL_VULNERABILITY,
            timestamp=ts,
            severity=SecuritySeverity.CRITICAL,
            source="dependency",
            repository="myorg/myrepo",
            scan_id="scan-456",
            workspace_id="ws-789",
            findings=[finding],
            debate_requested=True,
            debate_id="debate-001",
            debate_question="How to fix?",
            correlation_id="corr-100",
            metadata={"extra": "info"},
        )
        assert event.id == "evt-123"
        assert event.event_type == SecurityEventType.CRITICAL_VULNERABILITY
        assert event.timestamp == ts
        assert event.severity == SecuritySeverity.CRITICAL
        assert event.source == "dependency"
        assert event.repository == "myorg/myrepo"
        assert event.scan_id == "scan-456"
        assert event.workspace_id == "ws-789"
        assert len(event.findings) == 1
        assert event.debate_requested is True
        assert event.debate_id == "debate-001"
        assert event.debate_question == "How to fix?"
        assert event.correlation_id == "corr-100"
        assert event.metadata == {"extra": "info"}

    def test_auto_generated_uuid_id(self):
        """Each event should get a unique UUID by default."""
        event1 = self._make_event()
        event2 = self._make_event()
        assert event1.id != event2.id
        # Should be a valid UUID
        uuid.UUID(event1.id)

    def test_auto_generated_timestamp(self):
        """Timestamp should default to current UTC time."""
        before = datetime.now(timezone.utc)
        event = self._make_event()
        after = datetime.now(timezone.utc)
        assert before <= event.timestamp <= after

    # --- Serialization ---

    def test_to_dict_basic(self):
        """Should serialize event to a dictionary."""
        event = self._make_event(
            id="evt-ser",
            event_type=SecurityEventType.SCAN_COMPLETED,
            severity=SecuritySeverity.LOW,
            source="sast",
            repository="org/repo",
            scan_id="scan-1",
        )
        d = event.to_dict()
        assert d["id"] == "evt-ser"
        assert d["event_type"] == "scan_completed"
        assert d["severity"] == "low"
        assert d["source"] == "sast"
        assert d["repository"] == "org/repo"
        assert d["scan_id"] == "scan-1"
        assert d["findings"] == []
        assert d["debate_requested"] is False

    def test_to_dict_with_findings(self):
        """Should serialize findings within the event."""
        finding = self._make_finding(
            severity=SecuritySeverity.CRITICAL,
            title="Critical RCE",
        )
        event = self._make_event(findings=[finding])
        d = event.to_dict()
        assert len(d["findings"]) == 1
        assert d["findings"][0]["title"] == "Critical RCE"
        assert d["findings"][0]["severity"] == "critical"

    def test_to_dict_timestamp_is_isoformat(self):
        """Timestamp should be serialized in ISO 8601 format."""
        event = self._make_event()
        d = event.to_dict()
        # Should be parseable back to datetime
        parsed = datetime.fromisoformat(d["timestamp"])
        assert isinstance(parsed, datetime)

    def test_to_dict_is_json_serializable(self):
        """The dict representation should be JSON serializable."""
        finding = self._make_finding()
        event = self._make_event(
            findings=[finding],
            metadata={"key": "value"},
        )
        d = event.to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["id"] == event.id

    def test_to_dict_roundtrip_fields(self):
        """All fields should survive serialization to dict."""
        event = self._make_event(
            id="evt-rt",
            event_type=SecurityEventType.SECRET_DETECTED,
            severity=SecuritySeverity.HIGH,
            source="secrets",
            repository="org/app",
            scan_id="scan-rt",
            workspace_id="ws-rt",
            debate_requested=True,
            debate_id="debate-rt",
            debate_question="What to do?",
            correlation_id="corr-rt",
            metadata={"round": "trip"},
        )
        d = event.to_dict()
        assert d["event_type"] == "secret_detected"
        assert d["correlation_id"] == "corr-rt"
        assert d["debate_question"] == "What to do?"
        assert d["metadata"]["round"] == "trip"

    # --- Properties ---

    def test_is_critical_with_critical_severity(self):
        """Event should be critical if its severity is CRITICAL."""
        event = self._make_event(severity=SecuritySeverity.CRITICAL)
        assert event.is_critical is True

    def test_is_critical_with_critical_finding(self):
        """Event should be critical if any finding has CRITICAL severity."""
        finding = self._make_finding(severity=SecuritySeverity.CRITICAL)
        event = self._make_event(
            severity=SecuritySeverity.MEDIUM,
            findings=[finding],
        )
        assert event.is_critical is True

    def test_is_not_critical_with_high_severity(self):
        """Event should not be critical if severity is only HIGH."""
        event = self._make_event(severity=SecuritySeverity.HIGH)
        assert event.is_critical is False

    def test_is_not_critical_with_only_high_findings(self):
        """Event should not be critical if all findings are HIGH."""
        findings = [self._make_finding(severity=SecuritySeverity.HIGH) for _ in range(3)]
        event = self._make_event(
            severity=SecuritySeverity.HIGH,
            findings=findings,
        )
        assert event.is_critical is False

    def test_critical_count(self):
        """Should count critical findings correctly."""
        findings = [
            self._make_finding(severity=SecuritySeverity.CRITICAL),
            self._make_finding(severity=SecuritySeverity.HIGH),
            self._make_finding(severity=SecuritySeverity.CRITICAL),
            self._make_finding(severity=SecuritySeverity.MEDIUM),
        ]
        event = self._make_event(findings=findings)
        assert event.critical_count == 2

    def test_critical_count_zero(self):
        """Should return 0 when no findings are critical."""
        findings = [self._make_finding(severity=SecuritySeverity.LOW)]
        event = self._make_event(findings=findings)
        assert event.critical_count == 0

    def test_high_count(self):
        """Should count high severity findings correctly."""
        findings = [
            self._make_finding(severity=SecuritySeverity.CRITICAL),
            self._make_finding(severity=SecuritySeverity.HIGH),
            self._make_finding(severity=SecuritySeverity.HIGH),
            self._make_finding(severity=SecuritySeverity.LOW),
        ]
        event = self._make_event(findings=findings)
        assert event.high_count == 2

    def test_high_count_zero(self):
        """Should return 0 when no findings are high severity."""
        event = self._make_event()
        assert event.high_count == 0


# =============================================================================
# SecurityEventEmitter
# =============================================================================


class TestSecurityEventEmitter:
    """Tests for SecurityEventEmitter subscribe/emit/filtering/debate logic."""

    def _make_emitter(self, **kwargs):
        """Create an emitter with auto-debate disabled by default for simpler testing."""
        defaults = {"enable_auto_debate": False}
        defaults.update(kwargs)
        return SecurityEventEmitter(**defaults)

    def _make_event(self, **overrides):
        """Helper to create a SecurityEvent."""
        return SecurityEvent(**overrides)

    def _make_finding(self, severity=SecuritySeverity.HIGH, **overrides):
        defaults = {
            "id": str(uuid.uuid4()),
            "finding_type": "vulnerability",
            "severity": severity,
            "title": "Test finding",
            "description": "A test finding",
        }
        defaults.update(overrides)
        return SecurityFinding(**defaults)

    # --- Subscribe / Emit ---

    @pytest.mark.asyncio
    async def test_emit_calls_subscribed_handler(self):
        """Emitting an event should call the handler subscribed to that type."""
        emitter = self._make_emitter()
        handler = AsyncMock()
        emitter.subscribe(SecurityEventType.VULNERABILITY_DETECTED, handler)

        event = self._make_event(event_type=SecurityEventType.VULNERABILITY_DETECTED)
        await emitter.emit(event)

        handler.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_emit_does_not_call_unrelated_handler(self):
        """Emitting should not call handlers for other event types."""
        emitter = self._make_emitter()
        handler = AsyncMock()
        emitter.subscribe(SecurityEventType.SECRET_DETECTED, handler)

        event = self._make_event(event_type=SecurityEventType.VULNERABILITY_DETECTED)
        await emitter.emit(event)

        handler.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_emit_calls_multiple_handlers(self):
        """Emitting should call all handlers for the same event type."""
        emitter = self._make_emitter()
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        emitter.subscribe(SecurityEventType.SCAN_COMPLETED, handler1)
        emitter.subscribe(SecurityEventType.SCAN_COMPLETED, handler2)

        event = self._make_event(event_type=SecurityEventType.SCAN_COMPLETED)
        await emitter.emit(event)

        handler1.assert_awaited_once_with(event)
        handler2.assert_awaited_once_with(event)

    @pytest.mark.asyncio
    async def test_subscribe_all_receives_all_events(self):
        """Global handlers should be called for all event types."""
        emitter = self._make_emitter()
        global_handler = AsyncMock()
        emitter.subscribe_all(global_handler)

        for event_type in [
            SecurityEventType.VULNERABILITY_DETECTED,
            SecurityEventType.SECRET_DETECTED,
            SecurityEventType.SCAN_COMPLETED,
        ]:
            event = self._make_event(event_type=event_type)
            await emitter.emit(event)

        assert global_handler.await_count == 3

    @pytest.mark.asyncio
    async def test_emit_continues_on_handler_error(self):
        """Emit should continue calling other handlers if one raises."""
        emitter = self._make_emitter()
        failing_handler = AsyncMock(side_effect=RuntimeError("handler error"))
        success_handler = AsyncMock()
        emitter.subscribe(SecurityEventType.VULNERABILITY_DETECTED, failing_handler)
        emitter.subscribe(SecurityEventType.VULNERABILITY_DETECTED, success_handler)

        event = self._make_event(event_type=SecurityEventType.VULNERABILITY_DETECTED)
        await emitter.emit(event)

        failing_handler.assert_awaited_once()
        success_handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_emit_continues_on_global_handler_error(self):
        """Emit should continue calling other global handlers if one raises."""
        emitter = self._make_emitter()
        failing = AsyncMock(side_effect=ValueError("boom"))
        success = AsyncMock()
        emitter.subscribe_all(failing)
        emitter.subscribe_all(success)

        event = self._make_event()
        await emitter.emit(event)

        failing.assert_awaited_once()
        success.assert_awaited_once()

    # --- Unsubscribe ---

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_handler(self):
        """Unsubscribed handler should no longer be called."""
        emitter = self._make_emitter()
        handler = AsyncMock()
        emitter.subscribe(SecurityEventType.VULNERABILITY_DETECTED, handler)

        removed = emitter.unsubscribe(SecurityEventType.VULNERABILITY_DETECTED, handler)
        assert removed is True

        event = self._make_event(event_type=SecurityEventType.VULNERABILITY_DETECTED)
        await emitter.emit(event)
        handler.assert_not_awaited()

    def test_unsubscribe_returns_false_for_unknown_handler(self):
        """Unsubscribing a handler that was never added should return False."""
        emitter = self._make_emitter()
        handler = AsyncMock()
        result = emitter.unsubscribe(SecurityEventType.VULNERABILITY_DETECTED, handler)
        assert result is False

    def test_unsubscribe_returns_false_for_wrong_event_type(self):
        """Unsubscribing from an event type with no subscribers should return False."""
        emitter = self._make_emitter()
        handler = AsyncMock()
        emitter.subscribe(SecurityEventType.VULNERABILITY_DETECTED, handler)
        result = emitter.unsubscribe(SecurityEventType.SECRET_DETECTED, handler)
        assert result is False

    # --- Event History ---

    @pytest.mark.asyncio
    async def test_emit_stores_event_in_history(self):
        """Emitted events should be stored in history."""
        emitter = self._make_emitter()
        event = self._make_event()
        await emitter.emit(event)

        recent = emitter.get_recent_events()
        assert len(recent) == 1
        assert recent[0].id == event.id

    @pytest.mark.asyncio
    async def test_history_respects_max_limit(self):
        """History should be capped at _max_history entries."""
        emitter = self._make_emitter()
        emitter._max_history = 5

        for i in range(10):
            event = self._make_event(id=f"evt-{i}")
            await emitter.emit(event)

        assert len(emitter._event_history) == 5
        # Should keep the most recent events
        assert emitter._event_history[-1].id == "evt-9"
        assert emitter._event_history[0].id == "evt-5"

    @pytest.mark.asyncio
    async def test_history_newest_first_in_get_recent(self):
        """get_recent_events should return newest events first."""
        emitter = self._make_emitter()
        for i in range(5):
            event = self._make_event(id=f"evt-{i}")
            await emitter.emit(event)

        recent = emitter.get_recent_events()
        assert recent[0].id == "evt-4"
        assert recent[-1].id == "evt-0"

    # --- Workspace assignment ---

    @pytest.mark.asyncio
    async def test_emit_sets_default_workspace_id(self):
        """Should set workspace_id from emitter default if not on event."""
        emitter = self._make_emitter(workspace_id="ws-default")
        event = self._make_event()
        assert event.workspace_id is None

        await emitter.emit(event)
        assert event.workspace_id == "ws-default"

    @pytest.mark.asyncio
    async def test_emit_does_not_override_existing_workspace_id(self):
        """Should not override workspace_id if already set on the event."""
        emitter = self._make_emitter(workspace_id="ws-default")
        event = self._make_event(workspace_id="ws-explicit")

        await emitter.emit(event)
        assert event.workspace_id == "ws-explicit"

    # --- Filtering ---

    @pytest.mark.asyncio
    async def test_filter_by_event_type(self):
        """get_recent_events should filter by event type."""
        emitter = self._make_emitter()
        await emitter.emit(self._make_event(event_type=SecurityEventType.VULNERABILITY_DETECTED))
        await emitter.emit(self._make_event(event_type=SecurityEventType.SECRET_DETECTED))
        await emitter.emit(self._make_event(event_type=SecurityEventType.VULNERABILITY_DETECTED))

        vulns = emitter.get_recent_events(event_type=SecurityEventType.VULNERABILITY_DETECTED)
        assert len(vulns) == 2
        assert all(e.event_type == SecurityEventType.VULNERABILITY_DETECTED for e in vulns)

    @pytest.mark.asyncio
    async def test_filter_by_severity(self):
        """get_recent_events should filter by minimum severity."""
        emitter = self._make_emitter()
        await emitter.emit(self._make_event(severity=SecuritySeverity.INFO))
        await emitter.emit(self._make_event(severity=SecuritySeverity.LOW))
        await emitter.emit(self._make_event(severity=SecuritySeverity.MEDIUM))
        await emitter.emit(self._make_event(severity=SecuritySeverity.HIGH))
        await emitter.emit(self._make_event(severity=SecuritySeverity.CRITICAL))

        # Filter for HIGH and above
        high_and_above = emitter.get_recent_events(severity=SecuritySeverity.HIGH)
        assert len(high_and_above) == 2
        severities = {e.severity for e in high_and_above}
        assert severities == {SecuritySeverity.HIGH, SecuritySeverity.CRITICAL}

    @pytest.mark.asyncio
    async def test_filter_by_severity_critical_only(self):
        """Filtering by CRITICAL should return only critical events."""
        emitter = self._make_emitter()
        await emitter.emit(self._make_event(severity=SecuritySeverity.HIGH))
        await emitter.emit(self._make_event(severity=SecuritySeverity.CRITICAL))
        await emitter.emit(self._make_event(severity=SecuritySeverity.MEDIUM))

        critical = emitter.get_recent_events(severity=SecuritySeverity.CRITICAL)
        assert len(critical) == 1
        assert critical[0].severity == SecuritySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_filter_by_severity_info_returns_all(self):
        """Filtering by INFO should return all events."""
        emitter = self._make_emitter()
        await emitter.emit(self._make_event(severity=SecuritySeverity.INFO))
        await emitter.emit(self._make_event(severity=SecuritySeverity.LOW))
        await emitter.emit(self._make_event(severity=SecuritySeverity.CRITICAL))

        all_events = emitter.get_recent_events(severity=SecuritySeverity.INFO)
        assert len(all_events) == 3

    @pytest.mark.asyncio
    async def test_filter_by_type_and_severity_combined(self):
        """Should filter by both event type and severity simultaneously."""
        emitter = self._make_emitter()
        await emitter.emit(
            self._make_event(
                event_type=SecurityEventType.VULNERABILITY_DETECTED,
                severity=SecuritySeverity.LOW,
            )
        )
        await emitter.emit(
            self._make_event(
                event_type=SecurityEventType.VULNERABILITY_DETECTED,
                severity=SecuritySeverity.CRITICAL,
            )
        )
        await emitter.emit(
            self._make_event(
                event_type=SecurityEventType.SECRET_DETECTED,
                severity=SecuritySeverity.CRITICAL,
            )
        )

        filtered = emitter.get_recent_events(
            event_type=SecurityEventType.VULNERABILITY_DETECTED,
            severity=SecuritySeverity.HIGH,
        )
        assert len(filtered) == 1
        assert filtered[0].event_type == SecurityEventType.VULNERABILITY_DETECTED
        assert filtered[0].severity == SecuritySeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_get_recent_events_respects_limit(self):
        """get_recent_events should respect the limit parameter."""
        emitter = self._make_emitter()
        for i in range(10):
            await emitter.emit(self._make_event(id=f"evt-{i}"))

        limited = emitter.get_recent_events(limit=3)
        assert len(limited) == 3

    # --- Debate trigger logic ---

    def test_should_trigger_debate_for_critical_event(self):
        """Critical events should trigger debate when auto-debate is enabled."""
        emitter = SecurityEventEmitter(enable_auto_debate=True)
        event = self._make_event(severity=SecuritySeverity.CRITICAL)
        assert emitter._should_trigger_debate(event) is True

    def test_should_trigger_debate_for_critical_finding(self):
        """Events with a critical finding should trigger debate."""
        emitter = SecurityEventEmitter(enable_auto_debate=True)
        finding = self._make_finding(severity=SecuritySeverity.CRITICAL)
        event = self._make_event(
            severity=SecuritySeverity.MEDIUM,
            findings=[finding],
        )
        assert emitter._should_trigger_debate(event) is True

    def test_should_trigger_debate_for_multiple_high_findings(self):
        """Three or more HIGH findings should trigger debate."""
        emitter = SecurityEventEmitter(enable_auto_debate=True)
        findings = [self._make_finding(severity=SecuritySeverity.HIGH) for _ in range(3)]
        event = self._make_event(
            severity=SecuritySeverity.HIGH,
            findings=findings,
        )
        assert emitter._should_trigger_debate(event) is True

    def test_should_not_trigger_debate_for_two_high_findings(self):
        """Fewer than three HIGH findings should not trigger debate."""
        emitter = SecurityEventEmitter(enable_auto_debate=True)
        findings = [self._make_finding(severity=SecuritySeverity.HIGH) for _ in range(2)]
        event = self._make_event(
            severity=SecuritySeverity.HIGH,
            findings=findings,
        )
        assert emitter._should_trigger_debate(event) is False

    def test_should_not_trigger_debate_when_disabled(self):
        """Auto-debate disabled should prevent debate triggering."""
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        event = self._make_event(severity=SecuritySeverity.CRITICAL)
        assert emitter._should_trigger_debate(event) is False

    def test_should_not_trigger_debate_if_already_has_debate_id(self):
        """Should not re-trigger debate if event already has a debate_id."""
        emitter = SecurityEventEmitter(enable_auto_debate=True)
        event = self._make_event(
            severity=SecuritySeverity.CRITICAL,
            debate_id="existing-debate-123",
        )
        assert emitter._should_trigger_debate(event) is False

    def test_event_to_dict_redacts_secret_findings(self):
        """Event serialization should not expose raw secret finding material."""
        finding = SecurityFinding(
            id="secret-1",
            finding_type="Credential",
            severity=SecuritySeverity.CRITICAL,
            title="Token literal-secret",
            description="literal-secret",
            file_path="src/config.py",
            line_number=42,
            metadata={"secret_type": "api_token", "raw_secret": "literal-secret"},
        )
        event = self._make_event(severity=SecuritySeverity.CRITICAL, findings=[finding])

        data = event.to_dict()
        serialized = json.dumps(data)

        assert data["findings"][0]["title"] == "Secret finding"
        assert data["findings"][0]["file_path"] == "src/config.py"
        assert data["findings"][0]["line_number"] == 42
        assert data["findings"][0]["metadata"] == {"secret_type": "api_token"}
        assert "literal-secret" not in serialized

    def test_event_to_dict_redacts_sast_secret_findings(self):
        """SAST secret rules should be redacted even when finding_type is generic."""
        finding = SecurityFinding(
            id="sast-secret-1",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="Hardcoded credential",
            description="Matched code: AWS_SECRET_ACCESS_KEY = 'literal-secret'",
            file_path="src/config.py",
            line_number=42,
            metadata={
                "scanner": "semgrep",
                "rule_id": "python.lang.security.audit.hardcoded-credential",
                "snippet": "api_key = 'literal-secret'",
            },
        )
        event = self._make_event(severity=SecuritySeverity.CRITICAL, findings=[finding])

        data = event.to_dict()
        serialized = json.dumps(data)

        assert is_secret_finding(finding) is True
        assert data["findings"][0]["title"] == "Secret finding"
        assert data["findings"][0]["file_path"] == "src/config.py"
        assert data["findings"][0]["line_number"] == 42
        assert data["findings"][0]["metadata"] == {"secret_type": "unknown"}
        assert "literal-secret" not in serialized
        assert "AWS_SECRET_ACCESS_KEY" not in serialized
        assert "api_key" not in serialized

    def test_event_to_dict_redacts_sast_scanner_bridge_secret_text(self):
        """Scanner-emitted generic vulnerabilities should redact hardcoded secrets."""
        finding = SecurityFinding(
            id="sast-secret-bridge-1",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="hardcoded-password",
            description="Matched code: password = 'literal-secret'",
            file_path="src/settings.py",
            line_number=12,
            metadata={
                "scanner": "semgrep",
                "snippet": "password = 'literal-secret'",
                "rule_source": "semgrep",
                "vulnerability_class": "hardcoded-password",
                "confidence": 0.95,
            },
        )
        event = self._make_event(severity=SecuritySeverity.CRITICAL, findings=[finding])

        data = event.to_dict()
        serialized = json.dumps(data)

        assert is_secret_finding(finding) is True
        assert data["findings"][0]["title"] == "Secret finding"
        assert data["findings"][0]["file_path"] == "src/settings.py"
        assert data["findings"][0]["line_number"] == 12
        assert "literal-secret" not in serialized
        assert "hardcoded-password" not in serialized
        assert "password =" not in serialized

    def test_event_to_dict_uses_event_rule_metadata_for_sast_secret_redaction(self):
        """Single-finding SAST events may carry the secret rule signal on the event."""
        finding = SecurityFinding(
            id="sast-event-rule-secret-1",
            finding_type="sast",
            severity=SecuritySeverity.CRITICAL,
            title="Scanner finding",
            description="Matched code: literal-secret",
            file_path="src/settings.py",
            line_number=12,
            metadata={
                "scanner": "semgrep",
                "snippet": "literal-secret",
            },
        )
        event = self._make_event(
            severity=SecuritySeverity.CRITICAL,
            findings=[finding],
            metadata={
                "rule_id": "python.lang.security.audit.hardcoded-credential",
                "snippet": "api_key = 'literal-secret'",
                "raw_secret": "literal-secret",
            },
        )

        data = event.to_dict()
        serialized = json.dumps(data)

        assert is_secret_finding(finding) is False
        assert is_secret_finding(finding, event_metadata=event.metadata) is True
        assert data["findings"][0]["title"] == "Secret finding"
        assert data["metadata"] == {"secret_type": "unknown"}
        assert "literal-secret" not in serialized
        assert "api_key" not in serialized

    def test_event_to_dict_redacts_secret_metadata_keys(self):
        """Secret-bearing metadata key names should redact generic findings."""
        finding = SecurityFinding(
            id="metadata-secret-key-1",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="Config metadata finding",
            description="Scanner reported sensitive metadata.",
            metadata={"secret_key": "literal-secret"},
        )
        event = self._make_event(severity=SecuritySeverity.CRITICAL, findings=[finding])

        data = event.to_dict()
        serialized = json.dumps(data)

        assert is_secret_finding(finding) is True
        assert data["findings"][0]["title"] == "Secret finding"
        assert "literal-secret" not in serialized
        assert "secret_key" not in serialized

    def test_event_to_dict_sanitizes_secret_type_metadata(self):
        """Redacted metadata should not echo scanner-controlled secret_type values."""
        finding = SecurityFinding(
            id="metadata-secret-type-1",
            finding_type="secret",
            severity=SecuritySeverity.CRITICAL,
            title="Token literal-secret",
            description="literal-secret",
            metadata={"secret_type": "literal-secret", "raw_secret": "literal-secret"},
        )
        event = self._make_event(severity=SecuritySeverity.CRITICAL, findings=[finding])

        data = event.to_dict()
        serialized = json.dumps(data)

        assert data["findings"][0]["metadata"] == {"secret_type": "unknown"}
        assert "literal-secret" not in serialized

    def test_event_to_dict_redacts_nested_secret_metadata_keys(self):
        """Nested secret-bearing metadata key names should fail closed."""
        finding = SecurityFinding(
            id="metadata-client-secret-1",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="Config metadata finding",
            description="Scanner reported sensitive nested metadata.",
            metadata={"config": {"client_secret": "literal-secret"}},
        )
        event = self._make_event(severity=SecuritySeverity.CRITICAL, findings=[finding])

        data = event.to_dict()
        serialized = json.dumps(data)

        assert is_secret_finding(finding) is True
        assert data["findings"][0]["title"] == "Secret finding"
        assert "literal-secret" not in serialized
        assert "client_secret" not in serialized

    def test_event_to_dict_keeps_mixed_event_vulnerability_context(self):
        """Event-level secret metadata should not redact unrelated findings."""
        secret_finding = SecurityFinding(
            id="metadata-secret-key-1",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="Config metadata finding",
            description="Scanner reported sensitive metadata.",
            metadata={"secret_key": "literal-secret"},
        )
        cve_finding = SecurityFinding(
            id="cve-credential-redirect-1",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="Credential redirect vulnerability",
            description="Leaking Authorization credentials on cross-host redirect.",
            cve_id="CVE-2024-9999",
            package_name="requests",
        )
        event = self._make_event(
            severity=SecuritySeverity.CRITICAL,
            findings=[secret_finding, cve_finding],
            metadata={
                "rule_id": "python.lang.security.audit.hardcoded-credential",
                "raw_secret": "literal-secret",
            },
        )

        data = event.to_dict()
        serialized = json.dumps(data)

        assert data["metadata"] == {"secret_type": "unknown"}
        assert data["findings"][0]["title"] == "Secret finding"
        assert data["findings"][1]["finding_type"] == "vulnerability"
        assert data["findings"][1]["title"] == "Credential redirect vulnerability"
        assert data["findings"][1]["cve_id"] == "CVE-2024-9999"
        assert data["findings"][1]["package_name"] == "requests"
        assert "literal-secret" not in serialized
        assert "Authorization credentials" in serialized

    def test_event_to_dict_keeps_sast_token_vulnerability_context(self):
        """SAST vulnerability text mentioning tokens should not be redacted as a secret."""
        finding = SecurityFinding(
            id="sast-csrf-1",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="Missing CSRF token validation",
            description="POST handler accepts requests without verifying the CSRF token.",
            file_path="src/views.py",
            line_number=27,
            metadata={
                "scanner": "semgrep",
                "rule_id": "python.django.security.csrf-token-missing",
                "message": "Missing CSRF token validation",
                "snippet": "csrf_token = request.headers.get('X-CSRF-Token')",
            },
        )
        event = self._make_event(severity=SecuritySeverity.CRITICAL, findings=[finding])

        data = event.to_dict()
        serialized = json.dumps(data)

        assert is_secret_finding(finding) is False
        assert data["findings"][0]["title"] == "Missing CSRF token validation"
        assert data["findings"][0]["description"] == (
            "POST handler accepts requests without verifying the CSRF token."
        )
        assert "Secret finding" not in serialized
        assert "csrf_token" in serialized

        password_finding = SecurityFinding(
            id="sast-password-check-1",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="Improper password check bypass",
            description="Password comparison can be bypassed through alternate auth flow.",
            metadata={
                "scanner": "semgrep",
                "rule_id": "python.auth.security.password-check-bypass",
                "message": "Improper password check bypass",
                "snippet": "if password_ok or oauth_bypass: return user",
            },
        )
        assert is_secret_finding(password_finding) is False

    def test_event_to_dict_keeps_credential_vulnerability_context(self):
        """Credential-mentioning vulnerabilities are not automatically secrets."""
        finding = SecurityFinding(
            id="vuln-credential-redirect-1",
            finding_type="vulnerability",
            severity=SecuritySeverity.CRITICAL,
            title="Credential redirect vulnerability",
            description="Leaking Authorization credentials on cross-host redirect.",
            cve_id="CVE-2024-9999",
            package_name="requests",
            package_version="2.31.0",
            metadata={
                "scanner": "dependency-audit",
                "message": "Leaking Authorization credentials on cross-host redirect.",
            },
        )
        event = self._make_event(severity=SecuritySeverity.CRITICAL, findings=[finding])

        data = event.to_dict()
        serialized = json.dumps(data)

        assert is_secret_finding(finding) is False
        assert data["findings"][0]["finding_type"] == "vulnerability"
        assert data["findings"][0]["title"] == "Credential redirect vulnerability"
        assert data["findings"][0]["description"] == (
            "Leaking Authorization credentials on cross-host redirect."
        )
        assert data["findings"][0]["cve_id"] == "CVE-2024-9999"
        assert data["findings"][0]["package_name"] == "requests"
        assert "Secret finding" not in serialized

    def test_should_not_trigger_debate_for_low_severity(self):
        """LOW severity events should not trigger debates."""
        emitter = SecurityEventEmitter(enable_auto_debate=True)
        event = self._make_event(severity=SecuritySeverity.LOW)
        assert emitter._should_trigger_debate(event) is False

    @pytest.mark.asyncio
    async def test_emit_triggers_debate_for_critical(self):
        """Emitting a critical event with auto-debate enabled should trigger debate."""
        emitter = SecurityEventEmitter(
            enable_auto_debate=True,
            debate_confidence_threshold=0.8,
            debate_timeout_seconds=123,
        )
        event = self._make_event(severity=SecuritySeverity.CRITICAL)

        mock_trigger = AsyncMock(return_value="debate-auto-123")
        with patch(
            "aragora.events.security_events.get_security_debate_runner",
            return_value=mock_trigger,
        ):
            await emitter.emit(event)
            mock_trigger.assert_awaited_once_with(
                event,
                confidence_threshold=0.8,
                timeout_seconds=123,
            )
            assert event.debate_requested is True
            assert event.debate_id == "debate-auto-123"

    @pytest.mark.asyncio
    async def test_emit_passes_event_positionally_to_modern_runner(self):
        """Runner kwargs should not force a literal ``event=`` parameter name."""
        emitter = SecurityEventEmitter(
            enable_auto_debate=True,
            debate_confidence_threshold=0.8,
            debate_timeout_seconds=123,
        )
        event = self._make_event(severity=SecuritySeverity.CRITICAL)
        calls: list[tuple[SecurityEvent, float, int]] = []

        async def modern_runner(
            security_event: SecurityEvent,
            *,
            confidence_threshold: float,
            timeout_seconds: int,
        ) -> str:
            calls.append((security_event, confidence_threshold, timeout_seconds))
            return "debate-modern-123"

        with patch(
            "aragora.events.security_events.get_security_debate_runner",
            return_value=modern_runner,
        ):
            await emitter.emit(event)

        assert calls == [(event, 0.8, 123)]
        assert event.debate_requested is True
        assert event.debate_id == "debate-modern-123"

    @pytest.mark.asyncio
    async def test_emit_triggers_legacy_one_arg_debate_runner(self):
        """Custom pre-options runners should remain valid auto-debate hooks."""
        emitter = SecurityEventEmitter(enable_auto_debate=True)
        event = self._make_event(severity=SecuritySeverity.CRITICAL)
        calls: list[SecurityEvent] = []

        async def legacy_runner(event: SecurityEvent) -> str:
            calls.append(event)
            return "debate-legacy-123"

        with patch(
            "aragora.events.security_events.get_security_debate_runner",
            return_value=legacy_runner,
        ):
            await emitter.emit(event)

        assert calls == [event]
        assert event.debate_requested is True
        assert event.debate_id == "debate-legacy-123"

    @pytest.mark.asyncio
    async def test_emit_isolates_runner_type_error(self, caplog):
        """Runner signature failures should not escape critical event delivery."""
        emitter = SecurityEventEmitter(enable_auto_debate=True)
        event = self._make_event(severity=SecuritySeverity.CRITICAL)

        async def broken_runner(event: SecurityEvent, *, confidence_threshold: float) -> str:
            raise TypeError("runner implementation failed")

        with (
            patch(
                "aragora.events.security_events.get_security_debate_runner",
                return_value=broken_runner,
            ),
            caplog.at_level(logging.ERROR, logger="aragora.events.security_events"),
        ):
            await emitter.emit(event)

        assert event.debate_requested is False
        assert event.debate_id is None
        assert "Failed to trigger security debate" in caplog.text

    @pytest.mark.asyncio
    async def test_emit_with_no_runner_is_graceful(self, caplog):
        """emit() must never await a missing runner (`await None(...)`).

        With no registered runner (e.g. a process where no composition root
        has imported aragora.debate.security_response yet -- this module
        never imports aragora.debate itself), the auto-debate path logs a
        warning and skips -- no exception escapes emit(), matching the
        dispatcher's log-and-skip semantics.
        """
        emitter = SecurityEventEmitter(enable_auto_debate=True)
        event = self._make_event(severity=SecuritySeverity.CRITICAL)

        with (
            patch(
                "aragora.events.security_events.get_security_debate_runner",
                return_value=None,
            ),
            caplog.at_level(logging.WARNING, logger="aragora.events.security_events"),
        ):
            await emitter.emit(event)

        assert event.debate_requested is False
        assert event.debate_id is None
        assert "No security debate runner registered" in caplog.text

    @pytest.mark.asyncio
    async def test_debate_started_event_redacts_secret_findings(self):
        """Follow-on debate events should not carry raw secret findings."""
        emitter = SecurityEventEmitter(enable_auto_debate=True)
        started_events: list[SecurityEvent] = []

        async def on_started(event: SecurityEvent) -> None:
            started_events.append(event)

        emitter.subscribe(SecurityEventType.SECURITY_DEBATE_STARTED, on_started)
        event = self._make_event(
            severity=SecuritySeverity.CRITICAL,
            findings=[
                SecurityFinding(
                    id="secret-1",
                    finding_type="Credential",
                    severity=SecuritySeverity.CRITICAL,
                    title="Token literal-secret",
                    description="literal-secret",
                    file_path="src/config.py",
                    line_number=42,
                    metadata={"secret_type": "api_token", "raw_secret": "literal-secret"},
                )
            ],
        )

        mock_trigger = AsyncMock(return_value="debate-auto-123")
        with patch(
            "aragora.events.security_events.get_security_debate_runner",
            return_value=mock_trigger,
        ):
            await emitter.emit(event)

        assert len(started_events) == 1
        started = started_events[0]
        assert started.findings[0].title == "Secret finding"
        assert started.findings[0].file_path == "src/config.py"
        assert started.findings[0].line_number == 42
        assert started.findings[0].metadata == {"secret_type": "api_token"}
        assert "literal-secret" not in json.dumps(started.to_dict())

    @pytest.mark.asyncio
    async def test_auto_debate_context_redacts_sast_secret_findings(self):
        """Serialized debate context should not expose generic SAST secret findings."""
        emitter = SecurityEventEmitter(enable_auto_debate=True)
        captured_payloads: list[str] = []
        event = self._make_event(
            severity=SecuritySeverity.CRITICAL,
            findings=[
                SecurityFinding(
                    id="sast-secret-1",
                    finding_type="vulnerability",
                    severity=SecuritySeverity.CRITICAL,
                    title="Hardcoded credential",
                    description="Matched code: AWS_SECRET_ACCESS_KEY = 'literal-secret'",
                    file_path="src/config.py",
                    line_number=42,
                    metadata={
                        "scanner": "semgrep",
                        "rule_id": "python.lang.security.audit.hardcoded-credential",
                        "snippet": "api_key = 'literal-secret'",
                    },
                )
            ],
        )

        async def runner(security_event: SecurityEvent, **_: object) -> str:
            captured_payloads.append(json.dumps(security_event.to_dict()))
            return "debate-sast-secret-123"

        with patch(
            "aragora.events.security_events.get_security_debate_runner",
            return_value=runner,
        ):
            await emitter.emit(event)

        assert event.debate_requested is True
        assert event.debate_id == "debate-sast-secret-123"
        assert len(captured_payloads) == 1
        assert "literal-secret" not in captured_payloads[0]
        assert "AWS_SECRET_ACCESS_KEY" not in captured_payloads[0]
        assert "api_key" not in captured_payloads[0]

    @pytest.mark.asyncio
    async def test_auto_debate_context_redacts_sast_scanner_bridge_secret_text(self):
        """Auto-debate payloads should scrub scanner-shaped hardcoded secrets."""
        emitter = SecurityEventEmitter(enable_auto_debate=True)
        captured_payloads: list[str] = []
        event = self._make_event(
            severity=SecuritySeverity.CRITICAL,
            findings=[
                SecurityFinding(
                    id="sast-secret-bridge-1",
                    finding_type="vulnerability",
                    severity=SecuritySeverity.CRITICAL,
                    title="hardcoded-password",
                    description="Matched code: password = 'literal-secret'",
                    file_path="src/settings.py",
                    line_number=12,
                    metadata={
                        "scanner": "semgrep",
                        "snippet": "password = 'literal-secret'",
                        "rule_source": "semgrep",
                        "vulnerability_class": "hardcoded-password",
                        "confidence": 0.95,
                    },
                )
            ],
        )

        async def runner(security_event: SecurityEvent, **_: object) -> str:
            captured_payloads.append(json.dumps(security_event.to_dict()))
            return "debate-sast-secret-bridge-123"

        with patch(
            "aragora.events.security_events.get_security_debate_runner",
            return_value=runner,
        ):
            await emitter.emit(event)

        assert event.debate_requested is True
        assert event.debate_id == "debate-sast-secret-bridge-123"
        assert len(captured_payloads) == 1
        assert "literal-secret" not in captured_payloads[0]
        assert "hardcoded-password" not in captured_payloads[0]
        assert "password =" not in captured_payloads[0]

    @pytest.mark.asyncio
    async def test_emit_does_not_trigger_debate_for_low(self):
        """Emitting a low severity event should not trigger debate."""
        emitter = SecurityEventEmitter(enable_auto_debate=True)
        event = self._make_event(severity=SecuritySeverity.LOW)

        mock_trigger = AsyncMock()
        with patch(
            "aragora.events.security_events.get_security_debate_runner",
            return_value=mock_trigger,
        ):
            await emitter.emit(event)
            mock_trigger.assert_not_awaited()

    # --- Pending debates ---

    def test_get_pending_debates_empty(self):
        """Should return empty dict when no debates are pending."""
        emitter = self._make_emitter()
        assert emitter.get_pending_debates() == {}

    def test_get_pending_debates_filters_done_tasks(self):
        """Should filter out completed tasks from pending debates."""
        emitter = self._make_emitter()

        done_task = MagicMock()
        done_task.done.return_value = True

        pending_task = MagicMock()
        pending_task.done.return_value = False

        emitter._pending_debates = {"done-1": done_task, "pending-1": pending_task}
        pending = emitter.get_pending_debates()
        assert "pending-1" in pending
        assert "done-1" not in pending


# =============================================================================
# Event Metadata (timestamps, source, etc.)
# =============================================================================


class TestEventMetadata:
    """Tests for event metadata handling."""

    def test_timestamp_is_utc(self):
        """Default timestamp should be in UTC."""
        event = SecurityEvent()
        assert event.timestamp.tzinfo == timezone.utc

    def test_source_field_categories(self):
        """Source field should accept various scanner categories."""
        for source in ["sast", "secrets", "dependency", "threat_intel"]:
            event = SecurityEvent(source=source)
            assert event.source == source

    def test_metadata_stores_arbitrary_context(self):
        """Metadata dict should store arbitrary context like IP, user agent."""
        event = SecurityEvent(
            metadata={
                "source_ip": "192.168.1.100",
                "user_agent": "Mozilla/5.0",
                "geo_location": {"country": "US", "city": "San Francisco"},
                "threat_score": 85,
            }
        )
        assert event.metadata["source_ip"] == "192.168.1.100"
        assert event.metadata["user_agent"] == "Mozilla/5.0"
        assert event.metadata["geo_location"]["country"] == "US"
        assert event.metadata["threat_score"] == 85

    def test_metadata_serialized_in_to_dict(self):
        """Metadata should be included in to_dict output."""
        event = SecurityEvent(metadata={"enriched": True, "tags": ["critical", "rce"]})
        d = event.to_dict()
        assert d["metadata"]["enriched"] is True
        assert "rce" in d["metadata"]["tags"]

    def test_correlation_id_links_related_events(self):
        """Correlation ID should link related events together."""
        corr_id = "corr-chain-001"
        event1 = SecurityEvent(
            event_type=SecurityEventType.VULNERABILITY_DETECTED,
            correlation_id=corr_id,
        )
        event2 = SecurityEvent(
            event_type=SecurityEventType.SECURITY_DEBATE_STARTED,
            correlation_id=corr_id,
        )
        assert event1.correlation_id == event2.correlation_id == corr_id

    def test_scan_id_links_events_to_scan(self):
        """Scan ID should identify which scan produced the event."""
        event = SecurityEvent(scan_id="scan-abc-123")
        assert event.scan_id == "scan-abc-123"
        assert event.to_dict()["scan_id"] == "scan-abc-123"


# =============================================================================
# Event Validation
# =============================================================================


class TestEventValidation:
    """Tests for event validation and field integrity."""

    def test_severity_must_be_valid_enum(self):
        """Should only accept valid SecuritySeverity values."""
        for sev in SecuritySeverity:
            event = SecurityEvent(severity=sev)
            assert event.severity == sev

    def test_event_type_must_be_valid_enum(self):
        """Should only accept valid SecurityEventType values."""
        for evt_type in SecurityEventType:
            event = SecurityEvent(event_type=evt_type)
            assert event.event_type == evt_type

    def test_finding_severity_serialized_as_string(self):
        """Finding severity should be serialized as its string value."""
        for sev in SecuritySeverity:
            finding = SecurityFinding(
                id="test",
                finding_type="vulnerability",
                severity=sev,
                title="Test",
                description="Test desc",
            )
            assert finding.to_dict()["severity"] == sev.value

    def test_event_type_serialized_as_string(self):
        """Event type should be serialized as its string value."""
        for evt_type in SecurityEventType:
            event = SecurityEvent(event_type=evt_type)
            assert event.to_dict()["event_type"] == evt_type.value


# =============================================================================
# Event Enrichment
# =============================================================================


class TestEventEnrichment:
    """Tests for event enrichment via metadata."""

    def test_enrich_with_geo_location(self):
        """Should support geo-location enrichment via metadata."""
        event = SecurityEvent()
        event.metadata["geo_location"] = {
            "country": "DE",
            "city": "Berlin",
            "latitude": 52.52,
            "longitude": 13.405,
        }
        assert event.metadata["geo_location"]["country"] == "DE"

    def test_enrich_with_threat_intel(self):
        """Should support threat intelligence enrichment via metadata."""
        event = SecurityEvent()
        event.metadata["threat_intel"] = {
            "ioc_match": True,
            "threat_actor": "APT-29",
            "confidence": 0.92,
            "references": ["https://example.com/threat-report"],
        }
        assert event.metadata["threat_intel"]["threat_actor"] == "APT-29"
        assert event.metadata["threat_intel"]["confidence"] == 0.92

    def test_finding_metadata_for_enrichment(self):
        """Finding metadata should support additional context."""
        finding = SecurityFinding(
            id="f-enrich",
            finding_type="vulnerability",
            severity=SecuritySeverity.HIGH,
            title="RCE via deserialization",
            description="Unsafe deserialization",
            metadata={
                "cvss": 9.1,
                "exploit_available": True,
                "epss_score": 0.87,
                "cisa_kev": True,
            },
        )
        assert finding.metadata["cvss"] == 9.1
        assert finding.metadata["exploit_available"] is True
        assert finding.metadata["epss_score"] == 0.87


# =============================================================================
# Event Correlation
# =============================================================================


class TestEventCorrelation:
    """Tests for linking related events via correlation_id."""

    @pytest.mark.asyncio
    async def test_correlated_events_filter(self):
        """Events with same correlation_id can be found together."""
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        corr_id = "corr-xyz-789"

        await emitter.emit(
            SecurityEvent(
                id="evt-a",
                event_type=SecurityEventType.VULNERABILITY_DETECTED,
                correlation_id=corr_id,
            )
        )
        await emitter.emit(
            SecurityEvent(
                id="evt-b",
                event_type=SecurityEventType.SECRET_DETECTED,
                correlation_id="other-corr",
            )
        )
        await emitter.emit(
            SecurityEvent(
                id="evt-c",
                event_type=SecurityEventType.SECURITY_DEBATE_STARTED,
                correlation_id=corr_id,
            )
        )

        all_events = emitter.get_recent_events()
        correlated = [e for e in all_events if e.correlation_id == corr_id]
        assert len(correlated) == 2
        correlated_ids = {e.id for e in correlated}
        assert correlated_ids == {"evt-a", "evt-c"}

    @pytest.mark.asyncio
    async def test_correlation_across_event_types(self):
        """Correlation ID should work across different event types."""
        emitter = SecurityEventEmitter(enable_auto_debate=False)
        corr_id = "incident-42"

        types = [
            SecurityEventType.SCAN_STARTED,
            SecurityEventType.VULNERABILITY_DETECTED,
            SecurityEventType.SCAN_COMPLETED,
        ]
        for et in types:
            await emitter.emit(SecurityEvent(event_type=et, correlation_id=corr_id))

        all_events = emitter.get_recent_events()
        correlated = [e for e in all_events if e.correlation_id == corr_id]
        assert len(correlated) == 3


# =============================================================================
# Event Aggregation
# =============================================================================


class TestEventAggregation:
    """Tests for counting/aggregating similar events."""

    @pytest.mark.asyncio
    async def test_count_events_by_type(self):
        """Should be able to count events grouped by type."""
        emitter = SecurityEventEmitter(enable_auto_debate=False)

        await emitter.emit(SecurityEvent(event_type=SecurityEventType.VULNERABILITY_DETECTED))
        await emitter.emit(SecurityEvent(event_type=SecurityEventType.VULNERABILITY_DETECTED))
        await emitter.emit(SecurityEvent(event_type=SecurityEventType.SECRET_DETECTED))

        vuln_events = emitter.get_recent_events(event_type=SecurityEventType.VULNERABILITY_DETECTED)
        secret_events = emitter.get_recent_events(event_type=SecurityEventType.SECRET_DETECTED)
        assert len(vuln_events) == 2
        assert len(secret_events) == 1

    @pytest.mark.asyncio
    async def test_count_events_by_severity(self):
        """Should be able to count events at each severity level."""
        emitter = SecurityEventEmitter(enable_auto_debate=False)

        await emitter.emit(SecurityEvent(severity=SecuritySeverity.CRITICAL))
        await emitter.emit(SecurityEvent(severity=SecuritySeverity.HIGH))
        await emitter.emit(SecurityEvent(severity=SecuritySeverity.HIGH))
        await emitter.emit(SecurityEvent(severity=SecuritySeverity.LOW))

        critical = emitter.get_recent_events(severity=SecuritySeverity.CRITICAL)
        high_and_above = emitter.get_recent_events(severity=SecuritySeverity.HIGH)
        assert len(critical) == 1
        assert len(high_and_above) == 3  # 1 critical + 2 high

    @pytest.mark.asyncio
    async def test_critical_finding_count_across_events(self):
        """Should be able to aggregate critical finding counts across events."""
        emitter = SecurityEventEmitter(enable_auto_debate=False)

        for i in range(3):
            findings = [
                SecurityFinding(
                    id=f"f-{i}-{j}",
                    finding_type="vulnerability",
                    severity=SecuritySeverity.CRITICAL,
                    title=f"Critical {i}-{j}",
                    description="Desc",
                )
                for j in range(i + 1)  # 1, 2, 3 findings
            ]
            await emitter.emit(SecurityEvent(findings=findings))

        all_events = emitter.get_recent_events()
        total_critical = sum(e.critical_count for e in all_events)
        assert total_critical == 6  # 1 + 2 + 3


# =============================================================================
# Rate Limit Detection (many failed auth events)
# =============================================================================


class TestRateLimitDetection:
    """Tests for detecting patterns like too many failed auth attempts."""

    @pytest.mark.asyncio
    async def test_detect_burst_of_similar_events(self):
        """Should be possible to detect a burst of similar security events."""
        emitter = SecurityEventEmitter(enable_auto_debate=False)

        # Simulate 10 failed auth events in quick succession
        for i in range(10):
            await emitter.emit(
                SecurityEvent(
                    event_type=SecurityEventType.THREAT_DETECTED,
                    severity=SecuritySeverity.MEDIUM,
                    metadata={"reason": "failed_auth", "source_ip": "10.0.0.1"},
                )
            )

        threats = emitter.get_recent_events(event_type=SecurityEventType.THREAT_DETECTED)
        assert len(threats) == 10

        # Pattern detection: many events from same IP
        same_ip = [e for e in threats if e.metadata.get("source_ip") == "10.0.0.1"]
        assert len(same_ip) == 10
        # Threshold check: if more than 5, it's suspicious
        assert len(same_ip) > 5

    @pytest.mark.asyncio
    async def test_high_event_count_triggers_auto_debate_via_findings(self):
        """Multiple high findings should trigger auto-debate (>= 3 HIGH)."""
        emitter = SecurityEventEmitter(enable_auto_debate=True)

        findings = [
            SecurityFinding(
                id=f"f-rate-{i}",
                finding_type="vulnerability",
                severity=SecuritySeverity.HIGH,
                title=f"Failed auth attempt {i}",
                description="Brute force indicator",
            )
            for i in range(5)
        ]
        event = SecurityEvent(
            event_type=SecurityEventType.THREAT_DETECTED,
            severity=SecuritySeverity.HIGH,
            findings=findings,
        )

        mock_trigger = AsyncMock(return_value="debate-rate-001")
        with patch(
            "aragora.events.security_events.get_security_debate_runner",
            return_value=mock_trigger,
        ):
            await emitter.emit(event)
            # Should trigger because high_count >= 3
            mock_trigger.assert_awaited_once()


# =============================================================================
# Convenience functions
# =============================================================================


class TestCreateVulnerabilityEvent:
    """Tests for the create_vulnerability_event convenience function."""

    def test_creates_event_with_correct_type(self):
        """Should create a VULNERABILITY_DETECTED event for non-critical."""
        event = create_vulnerability_event(
            vulnerability={
                "id": "vuln-1",
                "severity": "high",
                "title": "XSS in form",
                "description": "Cross-site scripting",
            },
            repository="org/web",
            scan_id="scan-001",
        )
        assert event.event_type == SecurityEventType.VULNERABILITY_DETECTED
        assert event.severity == SecuritySeverity.HIGH

    def test_creates_critical_event_type_for_critical_severity(self):
        """Should create a CRITICAL_VULNERABILITY event for critical severity."""
        event = create_vulnerability_event(
            vulnerability={
                "id": "vuln-2",
                "severity": "critical",
                "title": "RCE",
                "description": "Remote code execution",
                "cve_id": "CVE-2024-00001",
            },
            repository="org/api",
            scan_id="scan-002",
        )
        assert event.event_type == SecurityEventType.CRITICAL_VULNERABILITY
        assert event.severity == SecuritySeverity.CRITICAL

    def test_finding_has_correct_fields(self):
        """The created finding should have fields from the vulnerability dict."""
        event = create_vulnerability_event(
            vulnerability={
                "id": "vuln-3",
                "severity": "medium",
                "title": "SSRF",
                "description": "Server-side request forgery",
                "cve_id": "CVE-2024-55555",
                "package_name": "requests",
                "package_version": "2.28.0",
                "recommendation": "Upgrade requests",
            },
            repository="org/svc",
            scan_id="scan-003",
        )
        assert len(event.findings) == 1
        f = event.findings[0]
        assert f.id == "vuln-3"
        assert f.finding_type == "vulnerability"
        assert f.severity == SecuritySeverity.MEDIUM
        assert f.title == "SSRF"
        assert f.cve_id == "CVE-2024-55555"
        assert f.package_name == "requests"
        assert f.package_version == "2.28.0"
        assert f.recommendation == "Upgrade requests"

    def test_sets_repository_and_scan_id(self):
        """Should set repository and scan_id on the event."""
        event = create_vulnerability_event(
            vulnerability={"severity": "low", "title": "Info", "description": "Test"},
            repository="org/lib",
            scan_id="scan-004",
        )
        assert event.repository == "org/lib"
        assert event.scan_id == "scan-004"

    def test_sets_workspace_id(self):
        """Should set workspace_id when provided."""
        event = create_vulnerability_event(
            vulnerability={"severity": "low", "title": "Test", "description": "d"},
            repository="org/x",
            scan_id="s-1",
            workspace_id="ws-100",
        )
        assert event.workspace_id == "ws-100"

    def test_defaults_to_medium_for_unknown_severity(self):
        """Should default to MEDIUM when severity is unknown."""
        event = create_vulnerability_event(
            vulnerability={"severity": "unknown", "title": "Test", "description": "d"},
            repository="org/x",
            scan_id="s-2",
        )
        assert event.severity == SecuritySeverity.MEDIUM

    def test_title_fallback_to_cve_id(self):
        """Title should fall back to cve_id if title is missing."""
        event = create_vulnerability_event(
            vulnerability={
                "severity": "high",
                "description": "desc",
                "cve_id": "CVE-2024-11111",
            },
            repository="org/x",
            scan_id="s-3",
        )
        assert event.findings[0].title == "CVE-2024-11111"


class TestCreateSecretEvent:
    """Tests for the create_secret_event convenience function."""

    def test_creates_event_for_regular_secret(self):
        """Should create SECRET_DETECTED for non-critical secrets."""
        event = create_secret_event(
            secret={
                "id": "sec-1",
                "severity": "high",
                "secret_type": "api_key",
                "file_path": "config.py",
                "line_number": 15,
            },
            repository="org/cfg",
            scan_id="scan-s1",
        )
        assert event.event_type == SecurityEventType.SECRET_DETECTED
        assert event.severity == SecuritySeverity.HIGH

    def test_creates_event_for_critical_secret(self):
        """Should create CRITICAL_SECRET for critical severity."""
        event = create_secret_event(
            secret={
                "id": "sec-2",
                "severity": "critical",
                "secret_type": "private_key",
            },
            repository="org/keys",
            scan_id="scan-s2",
        )
        assert event.event_type == SecurityEventType.CRITICAL_SECRET
        assert event.severity == SecuritySeverity.CRITICAL

    def test_finding_has_secret_fields(self):
        """The finding should have secret-specific fields."""
        event = create_secret_event(
            secret={
                "id": "sec-3",
                "severity": "high",
                "secret_type": "aws_secret",
                "file_path": ".env",
                "line_number": 3,
                "description": "AWS secret key exposed",
            },
            repository="org/app",
            scan_id="scan-s3",
        )
        f = event.findings[0]
        assert f.finding_type == "secret"
        assert f.file_path == ".env"
        assert f.line_number == 3
        assert "Exposed aws_secret" in f.title
        assert f.recommendation == "Rotate the credential immediately and remove from codebase"

    def test_defaults_to_high_severity_for_unknown(self):
        """Secrets should default to HIGH severity when unknown."""
        event = create_secret_event(
            secret={"severity": "unknown", "secret_type": "token"},
            repository="org/x",
            scan_id="s-s4",
        )
        assert event.severity == SecuritySeverity.HIGH

    def test_sets_workspace_id(self):
        """Should set workspace_id when provided."""
        event = create_secret_event(
            secret={"severity": "high"},
            repository="org/x",
            scan_id="s-1",
            workspace_id="ws-sec",
        )
        assert event.workspace_id == "ws-sec"


class TestCreateScanCompletedEvent:
    """Tests for the create_scan_completed_event convenience function."""

    def test_creates_scan_completed_event(self):
        """Should create a SCAN_COMPLETED event."""
        event = create_scan_completed_event(
            scan_result={"critical_count": 0, "high_count": 0},
            repository="org/repo",
            scan_id="scan-c1",
        )
        assert event.event_type == SecurityEventType.SCAN_COMPLETED

    def test_critical_count_sets_critical_severity(self):
        """Critical findings in scan result should set CRITICAL severity."""
        event = create_scan_completed_event(
            scan_result={"critical_count": 2, "high_count": 1},
            repository="org/repo",
            scan_id="scan-c2",
        )
        assert event.severity == SecuritySeverity.CRITICAL

    def test_high_count_sets_high_severity(self):
        """High findings without criticals should set HIGH severity."""
        event = create_scan_completed_event(
            scan_result={"critical_count": 0, "high_count": 3},
            repository="org/repo",
            scan_id="scan-c3",
        )
        assert event.severity == SecuritySeverity.HIGH

    def test_no_critical_or_high_sets_medium_severity(self):
        """No critical or high findings should default to MEDIUM."""
        event = create_scan_completed_event(
            scan_result={"critical_count": 0, "high_count": 0},
            repository="org/repo",
            scan_id="scan-c4",
        )
        assert event.severity == SecuritySeverity.MEDIUM

    def test_builds_findings_from_vulnerabilities(self):
        """Should create findings from scan result vulnerabilities."""
        event = create_scan_completed_event(
            scan_result={
                "critical_count": 1,
                "high_count": 0,
                "vulnerabilities": [
                    {
                        "id": "v-1",
                        "severity": "critical",
                        "title": "RCE vuln",
                        "description": "Remote code execution",
                        "cve_id": "CVE-2024-11111",
                        "package_name": "flask",
                        "package_version": "1.0",
                    },
                ],
            },
            repository="org/web",
            scan_id="scan-c5",
        )
        assert len(event.findings) == 1
        f = event.findings[0]
        assert f.id == "v-1"
        assert f.severity == SecuritySeverity.CRITICAL
        assert f.cve_id == "CVE-2024-11111"

    def test_limits_findings_to_ten(self):
        """Should limit findings to at most 10."""
        vulns = [
            {
                "id": f"v-{i}",
                "severity": "medium",
                "title": f"Vuln {i}",
                "description": f"Desc {i}",
            }
            for i in range(20)
        ]
        event = create_scan_completed_event(
            scan_result={"critical_count": 0, "high_count": 0, "vulnerabilities": vulns},
            repository="org/big",
            scan_id="scan-c6",
        )
        assert len(event.findings) <= 10

    def test_sets_workspace_id(self):
        """Should set workspace_id on scan completed event."""
        event = create_scan_completed_event(
            scan_result={"critical_count": 0, "high_count": 0},
            repository="org/repo",
            scan_id="scan-c7",
            workspace_id="ws-scan",
        )
        assert event.workspace_id == "ws-scan"


# =============================================================================
# Debate Result Storage
# =============================================================================


class TestDebateResultStorage:
    """Tests for security debate result storage functions."""

    @pytest.fixture(autouse=True)
    def clear_results(self):
        """Clear stored debate results before each test."""
        _security_debate_results.clear()
        yield
        _security_debate_results.clear()

    @pytest.mark.asyncio
    async def test_store_and_retrieve_result(self):
        """Should store and retrieve a debate result by ID."""
        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.final_answer = "Upgrade the package."

        event = SecurityEvent(
            id="evt-store",
            repository="org/repo",
            findings=[
                SecurityFinding(
                    id="f-1",
                    finding_type="vulnerability",
                    severity=SecuritySeverity.HIGH,
                    title="T",
                    description="D",
                )
            ],
        )

        await _store_security_debate_result("debate-store-1", event, mock_result)

        result = await get_security_debate_result("debate-store-1")
        assert result is not None
        assert result["debate_id"] == "debate-store-1"
        assert result["event_id"] == "evt-store"
        assert result["repository"] == "org/repo"
        assert result["findings_count"] == 1
        assert result["consensus_reached"] is True
        assert result["confidence"] == 0.85
        assert result["final_answer"] == "Upgrade the package."
        assert "completed_at" in result

    @pytest.mark.asyncio
    async def test_get_nonexistent_result_returns_none(self):
        """Should return None for a non-existent debate ID."""
        result = await get_security_debate_result("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_security_debates(self):
        """Should list stored debate results."""
        for i in range(3):
            _security_debate_results[f"debate-{i}"] = {
                "debate_id": f"debate-{i}",
                "repository": "org/repo",
                "completed_at": f"2024-01-0{i + 1}T00:00:00Z",
            }

        results = await list_security_debates()
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_list_security_debates_filter_by_repository(self):
        """Should filter debate results by repository."""
        _security_debate_results["d-1"] = {
            "debate_id": "d-1",
            "repository": "org/repo-a",
            "completed_at": "2024-01-01T00:00:00Z",
        }
        _security_debate_results["d-2"] = {
            "debate_id": "d-2",
            "repository": "org/repo-b",
            "completed_at": "2024-01-02T00:00:00Z",
        }

        results = await list_security_debates(repository="org/repo-a")
        assert len(results) == 1
        assert results[0]["debate_id"] == "d-1"

    @pytest.mark.asyncio
    async def test_list_security_debates_respects_limit(self):
        """Should respect the limit parameter."""
        for i in range(10):
            _security_debate_results[f"d-{i}"] = {
                "debate_id": f"d-{i}",
                "completed_at": f"2024-01-{i + 1:02d}T00:00:00Z",
            }

        results = await list_security_debates(limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_list_security_debates_sorted_by_time_descending(self):
        """Results should be sorted newest first."""
        _security_debate_results["d-old"] = {
            "debate_id": "d-old",
            "completed_at": "2024-01-01T00:00:00Z",
        }
        _security_debate_results["d-new"] = {
            "debate_id": "d-new",
            "completed_at": "2024-06-15T00:00:00Z",
        }

        results = await list_security_debates()
        assert results[0]["debate_id"] == "d-new"
        assert results[1]["debate_id"] == "d-old"


# =============================================================================
# Singleton Management
# =============================================================================


class TestSingletonEmitter:
    """Tests for get_security_emitter / set_security_emitter singleton."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton before and after each test."""
        import aragora.events.security_events as mod

        original = mod._default_emitter
        mod._default_emitter = None
        yield
        mod._default_emitter = original

    def test_get_security_emitter_creates_default(self):
        """Should create a default emitter on first call."""
        emitter = get_security_emitter()
        assert isinstance(emitter, SecurityEventEmitter)

    def test_get_security_emitter_returns_same_instance(self):
        """Subsequent calls should return the same instance."""
        emitter1 = get_security_emitter()
        emitter2 = get_security_emitter()
        assert emitter1 is emitter2

    def test_set_security_emitter_replaces_default(self):
        """Setting a custom emitter should replace the default."""
        custom = SecurityEventEmitter(enable_auto_debate=False)
        set_security_emitter(custom)
        assert get_security_emitter() is custom

    def test_set_security_emitter_overrides_existing(self):
        """Should override a previously created singleton."""
        first = get_security_emitter()
        custom = SecurityEventEmitter(workspace_id="ws-override")
        set_security_emitter(custom)
        assert get_security_emitter() is not first
        assert get_security_emitter() is custom


# =============================================================================
# Security debate runner registry hook
# =============================================================================


class TestSecurityDebateRunnerRegistry:
    """Tests for register_security_debate_runner / get_security_debate_runner."""

    @pytest.fixture(autouse=True)
    def reset_runner(self):
        """Snapshot and restore the module-level runner around each test."""
        import aragora.events.security_events as mod

        original = mod._security_debate_runner
        yield
        mod._security_debate_runner = original

    def test_register_sets_the_runner(self):
        """Registering a runner should make it retrievable via the getter."""

        async def fake_runner(event, **kwargs):
            return "debate-fake"

        register_security_debate_runner(fake_runner)
        assert get_security_debate_runner() is fake_runner

    def test_register_replaces_previous_runner(self):
        """Registering a new runner should replace any previously registered one."""

        async def first_runner(event, **kwargs):
            return "first"

        async def second_runner(event, **kwargs):
            return "second"

        register_security_debate_runner(first_runner)
        register_security_debate_runner(second_runner)
        assert get_security_debate_runner() is second_runner

    def test_register_none_clears_the_runner(self):
        """Registering None should clear any previously registered runner."""

        async def fake_runner(event, **kwargs):
            return "debate-fake"

        register_security_debate_runner(fake_runner)
        register_security_debate_runner(None)
        assert get_security_debate_runner() is None

    def test_registry_stays_unset_without_composition_root(self):
        """Cold consumers (never-set registry) get NO default runner.

        This module never imports aragora.debate itself (P4a
        security-debate-unification removed the lazy-import self-heal
        fallback). Only a composition root that imports
        aragora.debate.security_response (aragora.debate.orchestrator,
        aragora.debate.event_subscribers.bootstrap_debate_event_subscribers,
        or aragora.analysis.codebase.sast.scanner) makes the default runner
        available -- see test_explicit_clear_sticks_against_default_register
        for that registration path.
        """
        import aragora.events.security_events as mod

        mod._security_debate_runner = mod._UNSET_RUNNER

        assert get_security_debate_runner() is None

    def test_explicit_clear_sticks_against_default_register(self):
        """Import-time default registration must not clobber an explicit clear."""
        import aragora.events.security_events as mod

        async def default_runner(event, **kwargs):
            return "debate-default"

        register_security_debate_runner(None)
        mod._register_default_security_debate_runner(default_runner)

        assert get_security_debate_runner() is None
        assert mod._security_debate_runner is None

    def test_reregister_after_clear_restores_runner(self):
        """Registering a runner after an explicit clear re-enables auto-debate."""

        async def fake_runner(event, **kwargs):
            return "debate-after-clear"

        register_security_debate_runner(None)
        register_security_debate_runner(fake_runner)

        assert get_security_debate_runner() is fake_runner

    @pytest.mark.asyncio
    async def test_emit_after_explicit_clear_does_not_fire_auto_debate(self):
        """End to end: after register_security_debate_runner(None), a critical
        emit neither runs a debate nor re-registers the default runner."""
        register_security_debate_runner(None)

        emitter = SecurityEventEmitter(enable_auto_debate=True)
        event = SecurityEvent(
            event_type=SecurityEventType.CRITICAL_VULNERABILITY,
            severity=SecuritySeverity.CRITICAL,
            findings=[
                SecurityFinding(
                    id="f-clear-1",
                    finding_type="vulnerability",
                    severity=SecuritySeverity.CRITICAL,
                    title="Critical CVE",
                    description="test finding",
                )
            ],
        )

        await emitter.emit(event)

        assert event.debate_requested is False
        assert event.debate_id is None
        assert get_security_debate_runner() is None
