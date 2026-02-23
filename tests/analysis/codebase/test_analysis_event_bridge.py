"""Tests for the analysis-to-event-system bridge.

Verifies that AnalysisEventBridge correctly translates BugDetector,
SecretsScanner, and SASTScanner findings into RISK_WARNING events.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from unittest.mock import MagicMock, patch

import pytest

from aragora.analysis.codebase.event_bridge import (
    AnalysisEventBridge,
    get_analysis_event_bridge,
)


# ---------------------------------------------------------------------------
# Lightweight stubs matching the real dataclass fields
# ---------------------------------------------------------------------------


class _BugSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class _BugType(str, Enum):
    NULL_POINTER = "null_pointer"
    RESOURCE_LEAK = "resource_leak"


@dataclass
class _BugFinding:
    bug_id: str = "B001"
    bug_type: _BugType = _BugType.NULL_POINTER
    severity: _BugSeverity = _BugSeverity.HIGH
    file_path: str = "src/app.py"
    line_number: int = 42
    description: str = "Possible null dereference"
    confidence: float = 0.9


class _SecretType(str, Enum):
    API_KEY = "api_key"
    PASSWORD = "password"


@dataclass
class _SecretFinding:
    id: str = "S001"
    secret_type: _SecretType = _SecretType.API_KEY
    file_path: str = "config/secrets.yaml"
    line_number: int = 10
    severity: str = "critical"
    confidence: float = 0.95


class _SASTSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class _SASTFinding:
    rule_id: str = "sql-injection"
    file_path: str = "api/views.py"
    line_start: int = 77
    message: str = "SQL injection risk in query builder"
    severity: _SASTSeverity = _SASTSeverity.ERROR
    vulnerability_class: str = "injection"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmitBugFindings:
    """Tests for emit_bug_findings()."""

    def test_emits_for_findings_above_threshold(self) -> None:
        bridge = AnalysisEventBridge(min_confidence=0.7)
        findings = [_BugFinding(confidence=0.9)]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            count = bridge.emit_bug_findings(findings)

        assert count == 1
        mock_dispatch.assert_called_once()
        call_args = mock_dispatch.call_args
        assert call_args[0][0] == "risk_warning"
        data = call_args[0][1]
        assert data["risk_type"] == "bug_detected"
        assert data["severity"] == "high"
        assert data["file"] == "src/app.py"
        assert data["line"] == 42
        assert data["confidence"] == 0.9

    def test_filters_below_confidence_threshold(self) -> None:
        bridge = AnalysisEventBridge(min_confidence=0.8)
        findings = [_BugFinding(confidence=0.5)]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            count = bridge.emit_bug_findings(findings)

        assert count == 0
        mock_dispatch.assert_not_called()

    def test_mixed_confidence_findings(self) -> None:
        bridge = AnalysisEventBridge(min_confidence=0.7)
        findings = [
            _BugFinding(confidence=0.9, bug_id="B001"),
            _BugFinding(confidence=0.5, bug_id="B002"),
            _BugFinding(confidence=0.75, bug_id="B003"),
        ]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            count = bridge.emit_bug_findings(findings)

        assert count == 2
        assert mock_dispatch.call_count == 2

    def test_description_truncated_to_500_chars(self) -> None:
        bridge = AnalysisEventBridge(min_confidence=0.5)
        long_desc = "x" * 1000
        findings = [_BugFinding(description=long_desc, confidence=0.9)]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            bridge.emit_bug_findings(findings)

        data = mock_dispatch.call_args[0][1]
        assert len(data["description"]) == 500

    def test_event_data_contains_bug_type(self) -> None:
        bridge = AnalysisEventBridge(min_confidence=0.5)
        findings = [
            _BugFinding(
                bug_type=_BugType.RESOURCE_LEAK,
                confidence=0.8,
            )
        ]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            bridge.emit_bug_findings(findings)

        data = mock_dispatch.call_args[0][1]
        assert "resource_leak" in data["bug_type"]

    def test_empty_findings_list(self) -> None:
        bridge = AnalysisEventBridge()

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            count = bridge.emit_bug_findings([])

        assert count == 0
        mock_dispatch.assert_not_called()


class TestEmitSecretFindings:
    """Tests for emit_secret_findings()."""

    def test_emits_with_critical_severity(self) -> None:
        bridge = AnalysisEventBridge()
        findings = [_SecretFinding()]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            count = bridge.emit_secret_findings(findings)

        assert count == 1
        data = mock_dispatch.call_args[0][1]
        assert data["severity"] == "critical"
        assert data["risk_type"] == "secret_detected"

    def test_never_includes_secret_value(self) -> None:
        bridge = AnalysisEventBridge()
        finding = _SecretFinding()
        # Add an attribute that looks like a secret value
        finding.matched_text = "sk-XXXXXXXXXXXXXXXX"  # type: ignore[attr-defined]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            bridge.emit_secret_findings([finding])

        data = mock_dispatch.call_args[0][1]
        # The bridge should NOT include matched_text or any raw secret
        assert "matched_text" not in data
        assert "sk-" not in str(data.get("description", ""))

    def test_includes_secret_type_in_description(self) -> None:
        bridge = AnalysisEventBridge()
        findings = [_SecretFinding(secret_type=_SecretType.PASSWORD)]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            bridge.emit_secret_findings(findings)

        data = mock_dispatch.call_args[0][1]
        assert "password" in data["description"]

    def test_includes_file_and_line(self) -> None:
        bridge = AnalysisEventBridge()
        findings = [_SecretFinding(file_path="env/.env", line_number=5)]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            bridge.emit_secret_findings(findings)

        data = mock_dispatch.call_args[0][1]
        assert data["file"] == "env/.env"
        assert data["line"] == 5

    def test_multiple_secrets(self) -> None:
        bridge = AnalysisEventBridge()
        findings = [
            _SecretFinding(id="S001"),
            _SecretFinding(id="S002"),
            _SecretFinding(id="S003"),
        ]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            count = bridge.emit_secret_findings(findings)

        assert count == 3
        assert mock_dispatch.call_count == 3


class TestEmitSASTFindings:
    """Tests for emit_sast_findings()."""

    def test_emits_error_severity(self) -> None:
        bridge = AnalysisEventBridge()
        findings = [_SASTFinding(severity=_SASTSeverity.ERROR)]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            count = bridge.emit_sast_findings(findings)

        assert count == 1
        data = mock_dispatch.call_args[0][1]
        assert data["risk_type"] == "sast_finding"
        assert data["severity"] == "error"

    def test_emits_critical_severity(self) -> None:
        bridge = AnalysisEventBridge()
        findings = [_SASTFinding(severity=_SASTSeverity.CRITICAL)]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            count = bridge.emit_sast_findings(findings)

        assert count == 1
        data = mock_dispatch.call_args[0][1]
        assert data["severity"] == "critical"

    def test_filters_info_severity(self) -> None:
        bridge = AnalysisEventBridge()
        findings = [_SASTFinding(severity=_SASTSeverity.INFO)]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            count = bridge.emit_sast_findings(findings)

        assert count == 0
        mock_dispatch.assert_not_called()

    def test_filters_warning_severity(self) -> None:
        bridge = AnalysisEventBridge()
        findings = [_SASTFinding(severity=_SASTSeverity.WARNING)]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            count = bridge.emit_sast_findings(findings)

        assert count == 0
        mock_dispatch.assert_not_called()

    def test_includes_rule_id_and_category(self) -> None:
        bridge = AnalysisEventBridge()
        findings = [
            _SASTFinding(
                rule_id="xss-reflected",
                vulnerability_class="cross-site-scripting",
                severity=_SASTSeverity.CRITICAL,
            )
        ]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            bridge.emit_sast_findings(findings)

        data = mock_dispatch.call_args[0][1]
        assert data["rule_id"] == "xss-reflected"
        assert data["category"] == "cross-site-scripting"

    def test_message_truncated_to_500_chars(self) -> None:
        bridge = AnalysisEventBridge()
        long_msg = "z" * 800
        findings = [_SASTFinding(message=long_msg, severity=_SASTSeverity.ERROR)]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            bridge.emit_sast_findings(findings)

        data = mock_dispatch.call_args[0][1]
        assert len(data["description"]) == 500

    def test_mixed_severities(self) -> None:
        bridge = AnalysisEventBridge()
        findings = [
            _SASTFinding(severity=_SASTSeverity.INFO, rule_id="R1"),
            _SASTFinding(severity=_SASTSeverity.WARNING, rule_id="R2"),
            _SASTFinding(severity=_SASTSeverity.ERROR, rule_id="R3"),
            _SASTFinding(severity=_SASTSeverity.CRITICAL, rule_id="R4"),
        ]

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            count = bridge.emit_sast_findings(findings)

        # Only error and critical should emit
        assert count == 2
        assert mock_dispatch.call_count == 2


class TestStatsTracking:
    """Tests for statistics tracking."""

    def test_findings_processed_counts_all(self) -> None:
        bridge = AnalysisEventBridge(min_confidence=0.8)
        findings = [
            _BugFinding(confidence=0.9),  # emitted
            _BugFinding(confidence=0.5),  # filtered
        ]

        with patch("aragora.events.dispatcher.dispatch_event"):
            bridge.emit_bug_findings(findings)

        assert bridge.stats["findings_processed"] == 2

    def test_events_emitted_counts_only_emitted(self) -> None:
        bridge = AnalysisEventBridge(min_confidence=0.8)
        findings = [
            _BugFinding(confidence=0.9),
            _BugFinding(confidence=0.5),
        ]

        with patch("aragora.events.dispatcher.dispatch_event"):
            bridge.emit_bug_findings(findings)

        assert bridge.stats["events_emitted"] == 1

    def test_stats_accumulate_across_calls(self) -> None:
        bridge = AnalysisEventBridge(min_confidence=0.5)

        with patch("aragora.events.dispatcher.dispatch_event"):
            bridge.emit_bug_findings([_BugFinding(confidence=0.9)])
            bridge.emit_secret_findings([_SecretFinding()])
            bridge.emit_sast_findings([_SASTFinding(severity=_SASTSeverity.CRITICAL)])

        assert bridge.stats["findings_processed"] == 3
        assert bridge.stats["events_emitted"] == 3

    def test_stats_count_filtered_sast(self) -> None:
        bridge = AnalysisEventBridge()
        findings = [_SASTFinding(severity=_SASTSeverity.INFO)]

        with patch("aragora.events.dispatcher.dispatch_event"):
            bridge.emit_sast_findings(findings)

        assert bridge.stats["findings_processed"] == 1
        assert bridge.stats["events_emitted"] == 0


class TestGracefulDegradation:
    """Tests for graceful degradation when dispatcher is unavailable."""

    def test_import_error_handled(self) -> None:
        bridge = AnalysisEventBridge(min_confidence=0.5)
        findings = [_BugFinding(confidence=0.9)]

        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=ImportError("no module"),
        ):
            count = bridge.emit_bug_findings(findings)

        # The finding was "processed" but the emit failed silently
        assert count == 1
        assert bridge.stats["findings_processed"] == 1
        assert bridge.stats["events_emitted"] == 0

    def test_runtime_error_handled(self) -> None:
        bridge = AnalysisEventBridge(min_confidence=0.5)
        findings = [_SecretFinding()]

        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=RuntimeError("dispatcher not ready"),
        ):
            count = bridge.emit_secret_findings(findings)

        assert count == 1
        assert bridge.stats["events_emitted"] == 0

    def test_attribute_error_handled(self) -> None:
        bridge = AnalysisEventBridge(min_confidence=0.5)
        findings = [_SASTFinding(severity=_SASTSeverity.CRITICAL)]

        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=AttributeError("missing attribute"),
        ):
            count = bridge.emit_sast_findings(findings)

        assert count == 1
        assert bridge.stats["events_emitted"] == 0


class TestFactory:
    """Tests for the get_analysis_event_bridge factory."""

    def test_returns_correct_type(self) -> None:
        bridge = get_analysis_event_bridge()
        assert isinstance(bridge, AnalysisEventBridge)

    def test_default_min_confidence(self) -> None:
        bridge = get_analysis_event_bridge()
        assert bridge.min_confidence == 0.7

    def test_custom_min_confidence(self) -> None:
        bridge = get_analysis_event_bridge(min_confidence=0.9)
        assert bridge.min_confidence == 0.9

    def test_fresh_stats(self) -> None:
        bridge = get_analysis_event_bridge()
        assert bridge.stats == {"events_emitted": 0, "findings_processed": 0}
