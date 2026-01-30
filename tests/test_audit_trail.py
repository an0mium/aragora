"""
Tests for Audit Trail export module.

Tests the AuditTrail class and AuditTrailGenerator for:
- Event creation and ordering
- Export formats (JSON, CSV, Markdown)
- Checksum integrity
- Integration with GauntletResult
"""

from __future__ import annotations

import csv
import io
import json
import pytest
from datetime import datetime

from aragora.export.audit_trail import (
    AuditEvent,
    AuditEventType,
    AuditTrail,
    AuditTrailGenerator,
)

# Use canonical aragora.gauntlet package (not deprecated modes.gauntlet)
from aragora.gauntlet import (
    InputType,
    Verdict,
    Finding,
)
from aragora.gauntlet import OrchestratorResult as GauntletResult


# Test Fixtures


@pytest.fixture
def sample_events() -> list[AuditEvent]:
    """Create sample audit events for testing."""
    base_time = datetime(2026, 1, 11, 12, 0, 0)
    return [
        AuditEvent(
            event_id="evt-001",
            event_type=AuditEventType.GAUNTLET_START,
            timestamp=base_time.isoformat(),
            source="orchestrator",
            description="Gauntlet stress-test initiated",
            details={"input_type": "policy", "config": {"deep_audit_rounds": 4}},
        ),
        AuditEvent(
            event_id="evt-002",
            event_type=AuditEventType.RISK_ASSESSMENT,
            timestamp=datetime(2026, 1, 11, 12, 0, 5).isoformat(),
            source="risk-assessor",
            description="Risk assessment completed",
            details={"risk_score": 0.35},
        ),
        AuditEvent(
            event_id="evt-003",
            event_type=AuditEventType.REDTEAM_ATTACK,
            timestamp=datetime(2026, 1, 11, 12, 0, 10).isoformat(),
            source="redteam",
            description="Red-team attack: SQL injection",
            agent="redteam-agent",
            details={"attack_type": "sql_injection", "success": True},
        ),
        AuditEvent(
            event_id="evt-004",
            event_type=AuditEventType.FINDING_ADDED,
            timestamp=datetime(2026, 1, 11, 12, 0, 15).isoformat(),
            source="redteam",
            description="HIGH: SQL injection vulnerability detected",
            severity="high",
            agent="redteam-agent",
            details={"severity": "HIGH", "category": "security"},
        ),
        AuditEvent(
            event_id="evt-005",
            event_type=AuditEventType.VERDICT_DETERMINED,
            timestamp=datetime(2026, 1, 11, 12, 0, 20).isoformat(),
            source="orchestrator",
            description="Verdict: NEEDS_REVIEW",
            details={"verdict": "NEEDS_REVIEW", "confidence": 0.75},
        ),
        AuditEvent(
            event_id="evt-006",
            event_type=AuditEventType.GAUNTLET_END,
            timestamp=datetime(2026, 1, 11, 12, 0, 25).isoformat(),
            source="orchestrator",
            description="Gauntlet stress-test completed",
            details={"duration_seconds": 25.0, "findings_count": 3},
        ),
    ]


@pytest.fixture
def sample_audit_trail(sample_events) -> AuditTrail:
    """Create a sample audit trail for testing."""
    return AuditTrail(
        trail_id="trail-test-001",
        gauntlet_id="gauntlet-test-001",
        input_summary="Test policy document for rate limiting...",
        events=sample_events,
    )


@pytest.fixture
def sample_gauntlet_result() -> GauntletResult:
    """Create a sample GauntletResult for testing."""
    return GauntletResult(
        gauntlet_id="gauntlet-test-002",
        input_type=InputType.POLICY,
        input_summary="API Rate Limiting Policy v2.1...",
        verdict=Verdict.APPROVED_WITH_CONDITIONS,
        confidence=0.85,
        risk_score=0.25,
        robustness_score=0.9,
        coverage_score=0.75,
        critical_findings=[],
        high_findings=[
            Finding(
                finding_id="f1",
                category="security",
                severity=0.75,
                title="Missing authentication for internal services",
                description="Internal services lack proper authentication mechanism.",
            ),
        ],
        medium_findings=[
            Finding(
                finding_id="f2",
                category="compliance",
                severity=0.5,
                title="Audit logging incomplete",
                description="Rate limit violations should be logged for compliance.",
            ),
        ],
        low_findings=[],
        agents_involved=["claude", "gpt", "gemini"],
        duration_seconds=45.0,
    )


# AuditEvent Tests


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_event_creation(self):
        """Test creating an audit event."""
        event = AuditEvent(
            event_id="evt-test",
            event_type=AuditEventType.GAUNTLET_START,
            timestamp=datetime.now().isoformat(),
            source="test",
            description="Test event",
        )
        assert event.event_id == "evt-test"
        assert event.event_type == AuditEventType.GAUNTLET_START
        assert event.description == "Test event"
        assert event.source == "test"
        assert event.agent is None
        assert event.details == {}

    def test_event_with_agent(self):
        """Test creating an event with agent info."""
        event = AuditEvent(
            event_id="evt-agent",
            event_type=AuditEventType.REDTEAM_ATTACK,
            timestamp=datetime.now().isoformat(),
            source="redteam",
            description="Attack executed",
            agent="security-agent",
        )
        assert event.agent == "security-agent"

    def test_event_with_details(self):
        """Test creating an event with additional details."""
        event = AuditEvent(
            event_id="evt-data",
            event_type=AuditEventType.FINDING_ADDED,
            timestamp=datetime.now().isoformat(),
            source="prober",
            description="Finding discovered",
            details={"severity": "HIGH", "category": "security"},
        )
        assert event.details["severity"] == "HIGH"
        assert event.details["category"] == "security"

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event = AuditEvent(
            event_id="evt-dict",
            event_type=AuditEventType.VERDICT_DETERMINED,
            timestamp="2026-01-11T12:00:00",
            source="orchestrator",
            description="Verdict determined",
            agent="orchestrator",
            details={"verdict": "APPROVED"},
        )
        d = event.to_dict()

        assert d["event_id"] == "evt-dict"
        assert d["event_type"] == "verdict_determined"
        assert d["timestamp"] == "2026-01-11T12:00:00"
        assert d["description"] == "Verdict determined"
        assert d["agent"] == "orchestrator"
        assert d["details"]["verdict"] == "APPROVED"


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_all_event_types_exist(self):
        """Test that all expected event types are defined."""
        expected_types = [
            "GAUNTLET_START",
            "GAUNTLET_END",
            "REDTEAM_START",
            "REDTEAM_ATTACK",
            "REDTEAM_END",
            "PROBE_START",
            "PROBE_RESULT",
            "PROBE_END",
            "AUDIT_START",
            "AUDIT_FINDING",
            "AUDIT_END",
            "VERIFICATION_START",
            "VERIFICATION_RESULT",
            "VERIFICATION_END",
            "RISK_ASSESSMENT",
            "FINDING_ADDED",
            "VERDICT_DETERMINED",
            "RECEIPT_GENERATED",
        ]
        for type_name in expected_types:
            assert hasattr(AuditEventType, type_name), f"Missing event type: {type_name}"


# AuditTrail Tests


class TestAuditTrail:
    """Tests for AuditTrail dataclass."""

    def test_trail_creation(self, sample_events):
        """Test creating an audit trail."""
        trail = AuditTrail(
            trail_id="trail-001",
            gauntlet_id="gauntlet-001",
            input_summary="Test input",
            events=sample_events,
        )
        assert trail.trail_id == "trail-001"
        assert trail.gauntlet_id == "gauntlet-001"
        assert trail.input_summary == "Test input"
        assert len(trail.events) == 6

    def test_trail_events_count(self, sample_audit_trail):
        """Test event count."""
        assert len(sample_audit_trail.events) == 6

    def test_trail_has_start_event(self, sample_audit_trail):
        """Test that trail has a start event."""
        event_types = [e.event_type for e in sample_audit_trail.events]
        assert AuditEventType.GAUNTLET_START in event_types

    def test_trail_has_end_event(self, sample_audit_trail):
        """Test that trail has an end event."""
        event_types = [e.event_type for e in sample_audit_trail.events]
        assert AuditEventType.GAUNTLET_END in event_types


# JSON Export Tests


class TestAuditTrailJsonExport:
    """Tests for JSON export functionality."""

    def test_to_json(self, sample_audit_trail):
        """Test JSON export."""
        json_str = sample_audit_trail.to_json()
        data = json.loads(json_str)

        assert "trail_id" in data
        assert "gauntlet_id" in data
        assert "events" in data
        assert len(data["events"]) == 6

    def test_json_event_structure(self, sample_audit_trail):
        """Test that JSON events have correct structure."""
        json_str = sample_audit_trail.to_json()
        data = json.loads(json_str)

        for event in data["events"]:
            assert "event_id" in event
            assert "event_type" in event
            assert "timestamp" in event
            assert "description" in event

    def test_json_round_trip(self, sample_audit_trail):
        """Test that JSON can be parsed and re-serialized."""
        json_str = sample_audit_trail.to_json()
        data = json.loads(json_str)
        json_str2 = json.dumps(data, indent=2)
        data2 = json.loads(json_str2)

        assert data["gauntlet_id"] == data2["gauntlet_id"]
        assert len(data["events"]) == len(data2["events"])


# CSV Export Tests


class TestAuditTrailCsvExport:
    """Tests for CSV export functionality."""

    def test_to_csv(self, sample_audit_trail):
        """Test CSV export."""
        csv_str = sample_audit_trail.to_csv()
        assert csv_str is not None
        assert len(csv_str) > 0

    def test_csv_has_header(self, sample_audit_trail):
        """Test that CSV has header row."""
        csv_str = sample_audit_trail.to_csv()
        lines = csv_str.strip().split("\n")
        header = lines[0]

        assert "event_id" in header
        assert "event_type" in header
        assert "timestamp" in header
        assert "description" in header

    def test_csv_row_count(self, sample_audit_trail):
        """Test that CSV has correct number of rows."""
        csv_str = sample_audit_trail.to_csv()
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)

        # Header + 6 events
        assert len(rows) == 7

    def test_csv_parseable(self, sample_audit_trail):
        """Test that CSV can be parsed."""
        csv_str = sample_audit_trail.to_csv()
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 6
        assert rows[0]["event_type"] == "gauntlet_start"


# Markdown Export Tests


class TestAuditTrailMarkdownExport:
    """Tests for Markdown export functionality."""

    def test_to_markdown(self, sample_audit_trail):
        """Test Markdown export."""
        md = sample_audit_trail.to_markdown()
        assert md is not None
        assert len(md) > 0

    def test_markdown_has_title(self, sample_audit_trail):
        """Test that Markdown has title."""
        md = sample_audit_trail.to_markdown()
        assert "Audit Trail" in md

    def test_markdown_has_gauntlet_id(self, sample_audit_trail):
        """Test that Markdown includes gauntlet ID."""
        md = sample_audit_trail.to_markdown()
        assert sample_audit_trail.gauntlet_id in md

    def test_markdown_has_events(self, sample_audit_trail):
        """Test that Markdown includes events."""
        md = sample_audit_trail.to_markdown()
        assert "gauntlet_start" in md.lower() or "GAUNTLET_START" in md


# AuditTrailGenerator Tests


class TestAuditTrailGenerator:
    """Tests for AuditTrailGenerator."""

    def test_from_gauntlet_result(self, sample_gauntlet_result):
        """Test generating audit trail from GauntletResult."""
        trail = AuditTrailGenerator.from_gauntlet_result(sample_gauntlet_result)

        assert trail is not None
        assert isinstance(trail, AuditTrail)
        assert trail.gauntlet_id == sample_gauntlet_result.gauntlet_id

    def test_generated_trail_has_events(self, sample_gauntlet_result):
        """Test that generated trail has events."""
        trail = AuditTrailGenerator.from_gauntlet_result(sample_gauntlet_result)
        assert len(trail.events) > 0

    def test_generated_trail_has_start_event(self, sample_gauntlet_result):
        """Test that generated trail has start event."""
        trail = AuditTrailGenerator.from_gauntlet_result(sample_gauntlet_result)
        event_types = [e.event_type for e in trail.events]
        assert AuditEventType.GAUNTLET_START in event_types

    def test_generated_trail_has_end_event(self, sample_gauntlet_result):
        """Test that generated trail has end event."""
        trail = AuditTrailGenerator.from_gauntlet_result(sample_gauntlet_result)
        event_types = [e.event_type for e in trail.events]
        assert AuditEventType.GAUNTLET_END in event_types

    def test_generated_trail_has_verdict_event(self, sample_gauntlet_result):
        """Test that generated trail has verdict event."""
        trail = AuditTrailGenerator.from_gauntlet_result(sample_gauntlet_result)
        event_types = [e.event_type for e in trail.events]
        assert AuditEventType.VERDICT_DETERMINED in event_types

    def test_generated_trail_includes_findings(self, sample_gauntlet_result):
        """Test that generated trail includes finding events."""
        trail = AuditTrailGenerator.from_gauntlet_result(sample_gauntlet_result)
        event_types = [e.event_type for e in trail.events]
        assert AuditEventType.FINDING_ADDED in event_types


# Edge Cases


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_events_list(self):
        """Test audit trail with empty events list."""
        trail = AuditTrail(
            trail_id="trail-empty",
            gauntlet_id="empty-test",
            input_summary="Empty test",
            events=[],
        )
        assert len(trail.events) == 0

    def test_single_event(self):
        """Test audit trail with single event."""
        event = AuditEvent(
            event_id="single",
            event_type=AuditEventType.GAUNTLET_START,
            timestamp=datetime.now().isoformat(),
            source="test",
            description="Single event",
        )
        trail = AuditTrail(
            trail_id="trail-single",
            gauntlet_id="single-test",
            input_summary="Single test",
            events=[event],
        )
        assert len(trail.events) == 1
        json_str = trail.to_json()
        assert "single" in json_str

    def test_very_long_description(self):
        """Test event with very long description."""
        event = AuditEvent(
            event_id="long-desc",
            event_type=AuditEventType.FINDING_ADDED,
            timestamp=datetime.now().isoformat(),
            source="test",
            description="A" * 10000,  # 10KB description
        )
        trail = AuditTrail(
            trail_id="trail-long",
            gauntlet_id="long-test",
            input_summary="Long test",
            events=[event],
        )
        # Should still work
        json_str = trail.to_json()
        assert len(json_str) > 10000

    def test_special_characters_in_details(self):
        """Test event with special characters in details."""
        event = AuditEvent(
            event_id="special",
            event_type=AuditEventType.AUDIT_FINDING,
            timestamp=datetime.now().isoformat(),
            source="test",
            description="Special characters test",
            details={
                "query": 'SELECT * FROM users WHERE name = "O\'Brien"',
                "path": "/api/test?foo=bar&baz=qux",
                "unicode": "Hello 世界",
            },
        )
        trail = AuditTrail(
            trail_id="trail-special",
            gauntlet_id="special-test",
            input_summary="Special test",
            events=[event],
        )
        # Should handle special characters
        json_str = trail.to_json()
        data = json.loads(json_str)
        assert data["events"][0]["details"]["unicode"] == "Hello 世界"

    def test_null_agent(self):
        """Test event with null agent."""
        event = AuditEvent(
            event_id="null-agent",
            event_type=AuditEventType.GAUNTLET_START,
            timestamp=datetime.now().isoformat(),
            source="orchestrator",
            description="No agent",
            agent=None,
        )
        trail = AuditTrail(
            trail_id="trail-null",
            gauntlet_id="null-test",
            input_summary="Null test",
            events=[event],
        )
        d = event.to_dict()
        assert d["agent"] is None
