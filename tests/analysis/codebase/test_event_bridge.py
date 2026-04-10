"""Comprehensive tests for AnalysisEventBridge event dispatch integration.

Covers initialization, handler invocation, error isolation, and
mock handler verification across all three finding types.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from aragora.analysis.codebase.event_bridge import (
    AnalysisEventBridge,
    get_analysis_event_bridge,
)

DISPATCH = "aragora.events.dispatcher.dispatch_event"


def _bug(confidence: float = 0.9, **kw):
    return SimpleNamespace(
        bug_id=kw.get("bug_id", "B1"),
        bug_type=kw.get("bug_type", "null_deref"),
        severity=kw.get("severity", "high"),
        file_path=kw.get("file_path", "a.py"),
        line_number=kw.get("line_number", 1),
        description=kw.get("description", "bug"),
        confidence=confidence,
    )


def _secret(**kw):
    return SimpleNamespace(
        secret_type=kw.get("secret_type", "api_key"),
        file_path=kw.get("file_path", "s.yaml"),
        line_number=kw.get("line_number", 1),
    )


def _sast(severity: str = "error", **kw):
    return SimpleNamespace(
        rule_id=kw.get("rule_id", "R1"),
        file_path=kw.get("file_path", "v.py"),
        line_start=kw.get("line_start", 1),
        message=kw.get("message", "issue"),
        severity=severity,
        vulnerability_class=kw.get("vulnerability_class", "injection"),
    )


# -- Initialization ----------------------------------------------------------


class TestInitialization:
    def test_default_confidence_threshold(self):
        bridge = AnalysisEventBridge()
        assert bridge.min_confidence == 0.7

    def test_custom_confidence_threshold(self):
        bridge = AnalysisEventBridge(min_confidence=0.5)
        assert bridge.min_confidence == 0.5

    def test_stats_start_at_zero(self):
        bridge = AnalysisEventBridge()
        assert bridge.stats == {"events_emitted": 0, "findings_processed": 0}


# -- Handler registration via dispatch_event --------------------------------


class TestHandlerRegistration:
    """The bridge delegates to dispatch_event; verify it is called correctly."""

    def test_bug_finding_dispatches_risk_warning(self):
        bridge = AnalysisEventBridge(min_confidence=0.5)
        with patch(DISPATCH) as mock:
            bridge.emit_bug_findings([_bug()])
        mock.assert_called_once()
        assert mock.call_args[0][0] == "risk_warning"

    def test_secret_finding_dispatches_risk_warning(self):
        bridge = AnalysisEventBridge()
        with patch(DISPATCH) as mock:
            bridge.emit_secret_findings([_secret()])
        mock.assert_called_once()
        assert mock.call_args[0][0] == "risk_warning"

    def test_sast_finding_dispatches_risk_warning(self):
        bridge = AnalysisEventBridge()
        with patch(DISPATCH) as mock:
            bridge.emit_sast_findings([_sast("critical")])
        mock.assert_called_once()
        assert mock.call_args[0][0] == "risk_warning"


# -- Emission to registered handlers ----------------------------------------


class TestEventEmission:
    def test_bug_payload_fields(self):
        bridge = AnalysisEventBridge(min_confidence=0.5)
        with patch(DISPATCH) as mock:
            bridge.emit_bug_findings([_bug(file_path="x.py", line_number=10)])
        data = mock.call_args[0][1]
        assert data["risk_type"] == "bug_detected"
        assert data["file"] == "x.py"
        assert data["line"] == 10

    def test_secret_payload_always_critical(self):
        bridge = AnalysisEventBridge()
        with patch(DISPATCH) as mock:
            bridge.emit_secret_findings([_secret()])
        assert mock.call_args[0][1]["severity"] == "critical"

    def test_sast_payload_includes_rule_id(self):
        bridge = AnalysisEventBridge()
        with patch(DISPATCH) as mock:
            bridge.emit_sast_findings([_sast(rule_id="XSS-01")])
        assert mock.call_args[0][1]["rule_id"] == "XSS-01"


# -- Unregistered / filtered event types ------------------------------------


class TestUnregisteredEventTypes:
    def test_low_confidence_bugs_not_emitted(self):
        bridge = AnalysisEventBridge(min_confidence=0.8)
        with patch(DISPATCH) as mock:
            count = bridge.emit_bug_findings([_bug(0.5)])
        assert count == 0
        mock.assert_not_called()

    def test_info_sast_not_emitted(self):
        bridge = AnalysisEventBridge()
        with patch(DISPATCH) as mock:
            count = bridge.emit_sast_findings([_sast("info")])
        assert count == 0
        mock.assert_not_called()

    def test_low_sast_not_emitted(self):
        bridge = AnalysisEventBridge()
        with patch(DISPATCH) as mock:
            count = bridge.emit_sast_findings([_sast("low")])
        assert count == 0
        mock.assert_not_called()


# -- Multiple handlers for same event type ----------------------------------


class TestMultipleFindings:
    def test_multiple_bugs_emit_multiple_events(self):
        bridge = AnalysisEventBridge(min_confidence=0.5)
        bugs = [_bug(bug_id=f"B{i}") for i in range(3)]
        with patch(DISPATCH) as mock:
            count = bridge.emit_bug_findings(bugs)
        assert count == 3
        assert mock.call_count == 3

    def test_multiple_secrets_all_emitted(self):
        bridge = AnalysisEventBridge()
        secrets = [_secret(file_path=f"f{i}.env") for i in range(4)]
        with patch(DISPATCH) as mock:
            count = bridge.emit_secret_findings(secrets)
        assert count == 4
        assert mock.call_count == 4

    def test_stats_accumulate_across_finding_types(self):
        bridge = AnalysisEventBridge(min_confidence=0.5)
        with patch(DISPATCH):
            bridge.emit_bug_findings([_bug()])
            bridge.emit_secret_findings([_secret()])
            bridge.emit_sast_findings([_sast("error")])
        assert bridge.stats["findings_processed"] == 3
        assert bridge.stats["events_emitted"] == 3


# -- Error isolation between handlers --------------------------------------


class TestErrorIsolation:
    def test_dispatch_failure_does_not_raise(self):
        bridge = AnalysisEventBridge(min_confidence=0.5)
        with patch(DISPATCH, side_effect=RuntimeError("boom")):
            count = bridge.emit_bug_findings([_bug()])
        assert count == 1  # finding counted even though emit failed

    def test_dispatch_failure_does_not_increment_emitted(self):
        bridge = AnalysisEventBridge(min_confidence=0.5)
        with patch(DISPATCH, side_effect=Exception("fail")):
            bridge.emit_secret_findings([_secret()])
        assert bridge.stats["events_emitted"] == 0
        assert bridge.stats["findings_processed"] == 1

    def test_one_failure_does_not_block_subsequent(self):
        bridge = AnalysisEventBridge(min_confidence=0.5)
        effects = [RuntimeError("fail"), None]
        with patch(DISPATCH, side_effect=effects):
            count = bridge.emit_bug_findings([_bug(bug_id="B1"), _bug(bug_id="B2")])
        assert count == 2
        assert bridge.stats["events_emitted"] == 1  # only second succeeded


# -- Mock handler verification ----------------------------------------------


class TestMockHandlerVerification:
    def test_dispatch_called_with_exact_args(self):
        bridge = AnalysisEventBridge(min_confidence=0.5)
        with patch(DISPATCH) as mock:
            bridge.emit_bug_findings([_bug(confidence=0.85, bug_id="B99")])
        mock.assert_called_once_with(
            "risk_warning",
            {
                "risk_type": "bug_detected",
                "severity": "high",
                "description": "bug",
                "bug_type": "null_deref",
                "file": "a.py",
                "line": 1,
                "confidence": 0.85,
                "bug_id": "B99",
            },
        )

    def test_factory_produces_usable_bridge(self):
        bridge = get_analysis_event_bridge(min_confidence=0.6)
        with patch(DISPATCH) as mock:
            bridge.emit_bug_findings([_bug(0.7)])
        mock.assert_called_once()
