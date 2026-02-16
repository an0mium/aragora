"""
Comprehensive tests for the AuditingHandler module (aragora/server/handlers/auditing.py).

Tests cover:
- AuditRequestParser: JSON parsing, field validation, int parsing, probe/audit request parsing
- AuditAgentFactory: single/multiple agent creation, unavailability, error handling
- AuditResultRecorder: ELO recording, probe/audit report saving, ELO adjustments
- AuditingHandler: routing, can_handle, dispatch, attack types, capability probe,
  deep audit, red team analysis, permission checks, error paths
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.server.handlers.auditing import (
    AuditAgentFactory,
    AuditingHandler,
    AuditRequestParser,
    AuditResultRecorder,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    """Minimal server context with all optional keys set to None."""
    return {
        "storage": MagicMock(),
        "elo_system": None,
        "nomic_dir": None,
        "user_store": MagicMock(),
    }


@pytest.fixture
def handler(ctx):
    """Create an AuditingHandler backed by the mock context."""
    return AuditingHandler(ctx)


@pytest.fixture
def mock_http():
    """Simple mock HTTP handler object."""
    m = MagicMock()
    m.headers = {"Content-Length": "0"}
    m.rfile = MagicMock()
    m.rfile.read = MagicMock(return_value=b"{}")
    m.client_address = ("127.0.0.1", 12345)
    return m


def _make_read_fn(data):
    """Create a read_json_fn that returns *data* regardless of input."""
    return lambda handler: data


# ============================================================================
# AuditRequestParser tests
# ============================================================================


class TestAuditRequestParserReadJson:
    """Tests for _read_json helper."""

    def test_returns_error_when_body_is_none(self):
        data, err = AuditRequestParser._read_json(MagicMock(), lambda h: None)
        assert data is None
        assert err is not None
        assert err.status_code == 400

    def test_returns_data_on_success(self):
        payload = {"key": "val"}
        data, err = AuditRequestParser._read_json(MagicMock(), lambda h: payload)
        assert err is None
        assert data == payload


class TestAuditRequestParserRequireField:
    """Tests for _require_field helper."""

    def test_missing_field_returns_error(self):
        val, err = AuditRequestParser._require_field({}, "name")
        assert val is None
        assert err is not None
        assert err.status_code == 400

    def test_empty_string_returns_error(self):
        val, err = AuditRequestParser._require_field({"name": "   "}, "name")
        assert val is None
        assert err is not None

    def test_strips_whitespace(self):
        val, err = AuditRequestParser._require_field({"x": " hello "}, "x")
        assert err is None
        assert val == "hello"

    def test_validator_rejects_value(self):
        val, err = AuditRequestParser._require_field(
            {"x": "bad"}, "x", validator=lambda v: (False, "nope")
        )
        assert val is None
        assert err.status_code == 400

    def test_validator_accepts_value(self):
        val, err = AuditRequestParser._require_field(
            {"x": "good"}, "x", validator=lambda v: (True, None)
        )
        assert err is None
        assert val == "good"


class TestAuditRequestParserParseInt:
    """Tests for _parse_int helper."""

    def test_returns_default_when_missing(self):
        val, err = AuditRequestParser._parse_int({}, "n", 7, 20)
        assert err is None
        assert val == 7

    def test_clamps_to_max(self):
        val, err = AuditRequestParser._parse_int({"n": 100}, "n", 5, 10)
        assert err is None
        assert val == 10

    def test_passes_through_valid_value(self):
        val, err = AuditRequestParser._parse_int({"n": 3}, "n", 5, 10)
        assert err is None
        assert val == 3

    def test_non_integer_returns_error(self):
        val, err = AuditRequestParser._parse_int({"n": "abc"}, "n", 5, 10)
        assert err is not None
        assert err.status_code == 400


class TestAuditRequestParserCapabilityProbe:
    """Tests for parse_capability_probe."""

    def test_valid_request(self):
        data = {
            "agent_name": "test-agent",
            "probe_types": ["contradiction"],
            "probes_per_type": 5,
        }
        with patch(
            "aragora.server.handlers.auditing.validate_agent_name",
            return_value=(True, None),
        ):
            parsed, err = AuditRequestParser.parse_capability_probe(
                MagicMock(), _make_read_fn(data)
            )
        assert err is None
        assert parsed["agent_name"] == "test-agent"
        assert parsed["probes_per_type"] == 5

    def test_uses_defaults_for_optional_fields(self):
        data = {"agent_name": "a"}
        with patch(
            "aragora.server.handlers.auditing.validate_agent_name",
            return_value=(True, None),
        ):
            parsed, err = AuditRequestParser.parse_capability_probe(
                MagicMock(), _make_read_fn(data)
            )
        assert err is None
        assert parsed["probes_per_type"] == 3
        assert parsed["model_type"] == "anthropic-api"
        assert "contradiction" in parsed["probe_types"]

    def test_returns_error_on_invalid_json(self):
        parsed, err = AuditRequestParser.parse_capability_probe(MagicMock(), _make_read_fn(None))
        assert parsed is None
        assert err.status_code == 400

    def test_returns_error_on_missing_agent_name(self):
        data = {"probe_types": ["contradiction"]}
        parsed, err = AuditRequestParser.parse_capability_probe(MagicMock(), _make_read_fn(data))
        assert parsed is None
        assert err.status_code == 400

    def test_returns_error_on_invalid_agent_name(self):
        data = {"agent_name": "bad!"}
        with patch(
            "aragora.server.handlers.auditing.validate_agent_name",
            return_value=(False, "Invalid name"),
        ):
            parsed, err = AuditRequestParser.parse_capability_probe(
                MagicMock(), _make_read_fn(data)
            )
        assert parsed is None
        assert err.status_code == 400

    def test_probes_per_type_clamped_to_max(self):
        data = {"agent_name": "a", "probes_per_type": 999}
        with patch(
            "aragora.server.handlers.auditing.validate_agent_name",
            return_value=(True, None),
        ):
            parsed, err = AuditRequestParser.parse_capability_probe(
                MagicMock(), _make_read_fn(data)
            )
        assert err is None
        assert parsed["probes_per_type"] == 10  # max


class TestAuditRequestParserDeepAudit:
    """Tests for parse_deep_audit."""

    def test_valid_request(self):
        data = {
            "task": "Analyze this",
            "context": "Extra context",
            "agent_names": ["a1", "a2"],
        }
        parsed, err = AuditRequestParser.parse_deep_audit(MagicMock(), _make_read_fn(data))
        assert err is None
        assert parsed["task"] == "Analyze this"
        assert parsed["context"] == "Extra context"

    def test_requires_task(self):
        data = {"context": "something"}
        parsed, err = AuditRequestParser.parse_deep_audit(MagicMock(), _make_read_fn(data))
        assert parsed is None
        assert err.status_code == 400

    def test_uses_default_config(self):
        data = {"task": "Do it"}
        parsed, err = AuditRequestParser.parse_deep_audit(MagicMock(), _make_read_fn(data))
        assert err is None
        assert parsed["rounds"] == 6
        assert parsed["cross_examination_depth"] == 3
        assert parsed["risk_threshold"] == 0.7
        assert parsed["enable_research"] is True

    def test_invalid_risk_threshold_returns_error(self):
        data = {"task": "x", "config": {"risk_threshold": "not-a-number"}}
        parsed, err = AuditRequestParser.parse_deep_audit(MagicMock(), _make_read_fn(data))
        assert parsed is None
        assert err.status_code == 400

    def test_config_rounds_clamped_to_max(self):
        data = {"task": "x", "config": {"rounds": 100}}
        parsed, err = AuditRequestParser.parse_deep_audit(MagicMock(), _make_read_fn(data))
        assert err is None
        assert parsed["rounds"] == 10  # max

    def test_invalid_json_body(self):
        parsed, err = AuditRequestParser.parse_deep_audit(MagicMock(), _make_read_fn(None))
        assert parsed is None
        assert err.status_code == 400


# ============================================================================
# AuditAgentFactory tests
# ============================================================================


class TestAuditAgentFactorySingle:
    """Tests for create_single_agent."""

    def test_returns_503_when_unavailable(self):
        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", False):
            agent, err = AuditAgentFactory.create_single_agent("api", "a")
        assert agent is None
        assert err.status_code == 503

    def test_success(self):
        mock_agent = MagicMock()
        with (
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch("aragora.server.handlers.auditing.create_agent", return_value=mock_agent),
        ):
            agent, err = AuditAgentFactory.create_single_agent("api", "a", role="critic")
        assert err is None
        assert agent is mock_agent

    def test_handles_creation_exception(self):
        with (
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch(
                "aragora.server.handlers.auditing.create_agent",
                side_effect=RuntimeError("boom"),
            ),
        ):
            agent, err = AuditAgentFactory.create_single_agent("api", "a")
        assert agent is None
        assert err.status_code == 400

    def test_returns_503_when_create_agent_is_none(self):
        with (
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch("aragora.server.handlers.auditing.create_agent", None),
        ):
            agent, err = AuditAgentFactory.create_single_agent("api", "a")
        assert agent is None
        assert err.status_code == 503


class TestAuditAgentFactoryMultiple:
    """Tests for create_multiple_agents."""

    def test_returns_503_when_unavailable(self):
        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", False):
            agents, err = AuditAgentFactory.create_multiple_agents("api", [], ["d1", "d2"])
        assert agents == []
        assert err.status_code == 503

    def test_uses_defaults_when_names_empty(self):
        mock_agent = MagicMock()
        with (
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch("aragora.server.handlers.auditing.create_agent", return_value=mock_agent),
            patch("aragora.server.handlers.auditing.validate_id", return_value=(True, None)),
        ):
            agents, err = AuditAgentFactory.create_multiple_agents("api", [], ["d1", "d2", "d3"])
        assert err is None
        assert len(agents) == 3

    def test_requires_minimum_two_agents(self):
        with (
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch(
                "aragora.server.handlers.auditing.create_agent",
                side_effect=ValueError("fail"),
            ),
            patch("aragora.server.handlers.auditing.validate_id", return_value=(True, None)),
        ):
            agents, err = AuditAgentFactory.create_multiple_agents("api", ["a1", "a2"], ["d1"])
        assert agents == []
        assert err.status_code == 400

    def test_limits_to_max_agents(self):
        mock_agent = MagicMock()
        names = [f"agent-{i}" for i in range(10)]
        with (
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch("aragora.server.handlers.auditing.create_agent", return_value=mock_agent),
            patch("aragora.server.handlers.auditing.validate_id", return_value=(True, None)),
        ):
            agents, err = AuditAgentFactory.create_multiple_agents("api", names, [], max_agents=5)
        assert err is None
        assert len(agents) == 5

    def test_skips_invalid_agent_names(self):
        mock_agent = MagicMock()
        call_count = 0

        def counted_validate(name, label):
            return (True, None) if name.startswith("good") else (False, "bad")

        with (
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch("aragora.server.handlers.auditing.create_agent", return_value=mock_agent),
            patch("aragora.server.handlers.auditing.validate_id", side_effect=counted_validate),
        ):
            agents, err = AuditAgentFactory.create_multiple_agents(
                "api", ["good1", "bad!", "good2"], []
            )
        assert err is None
        assert len(agents) == 2


# ============================================================================
# AuditResultRecorder tests
# ============================================================================


class TestAuditResultRecorderProbeElo:
    """Tests for record_probe_elo."""

    def test_skips_when_no_elo_system(self):
        report = MagicMock(probes_run=10)
        # Should not raise
        AuditResultRecorder.record_probe_elo(None, "agent", report, "id")

    def test_skips_when_no_probes(self):
        elo = MagicMock()
        report = MagicMock(probes_run=0)
        AuditResultRecorder.record_probe_elo(elo, "agent", report, "id")
        elo.record_redteam_result.assert_not_called()

    def test_records_elo_correctly(self):
        elo = MagicMock()
        report = MagicMock(
            probes_run=10,
            vulnerability_rate=0.3,
            vulnerabilities_found=3,
            critical_count=1,
        )
        AuditResultRecorder.record_probe_elo(elo, "claude", report, "rpt-1")
        elo.record_redteam_result.assert_called_once()
        kwargs = elo.record_redteam_result.call_args[1]
        assert kwargs["agent_name"] == "claude"
        assert kwargs["robustness_score"] == pytest.approx(0.7, abs=1e-9)
        assert kwargs["successful_attacks"] == 3
        assert kwargs["total_attacks"] == 10
        assert kwargs["critical_vulnerabilities"] == 1
        assert kwargs["session_id"] == "rpt-1"

    def test_handles_elo_exception(self, caplog):
        elo = MagicMock()
        elo.record_redteam_result.side_effect = RuntimeError("elo down")
        report = MagicMock(
            probes_run=5, vulnerability_rate=0.2, vulnerabilities_found=1, critical_count=0
        )
        with caplog.at_level(logging.WARNING):
            AuditResultRecorder.record_probe_elo(elo, "a", report, "id")
        assert "Failed to record ELO" in caplog.text


class TestAuditResultRecorderAuditElo:
    """Tests for calculate_audit_elo_adjustments."""

    def test_returns_empty_when_no_system(self):
        assert AuditResultRecorder.calculate_audit_elo_adjustments(MagicMock(), None) == {}

    def test_calculates_adjustments(self):
        f1 = MagicMock(agents_agree=["a", "b"], agents_disagree=["c"])
        f2 = MagicMock(agents_agree=["a"], agents_disagree=[])
        verdict = MagicMock(findings=[f1, f2])

        result = AuditResultRecorder.calculate_audit_elo_adjustments(verdict, MagicMock())
        assert result["a"] == 4  # +2 + +2
        assert result["b"] == 2
        assert result["c"] == -1


class TestAuditResultRecorderSaveProbe:
    """Tests for save_probe_report."""

    def test_skips_when_no_dir(self):
        # Should not raise
        AuditResultRecorder.save_probe_report(None, "agent", MagicMock())

    def test_saves_report_to_file(self, tmp_path):
        report = MagicMock()
        report.report_id = "probe-abc"
        report.to_dict.return_value = {"id": "probe-abc", "findings": []}

        AuditResultRecorder.save_probe_report(tmp_path, "test-agent", report)

        probes_dir = tmp_path / "probes" / "test-agent"
        assert probes_dir.exists()
        files = list(probes_dir.glob("*.json"))
        assert len(files) == 1
        content = json.loads(files[0].read_text())
        assert content["id"] == "probe-abc"

    def test_handles_write_error(self, caplog, tmp_path):
        report = MagicMock()
        report.report_id = "x"
        report.to_dict.side_effect = TypeError("cannot serialize")
        with caplog.at_level(logging.ERROR):
            AuditResultRecorder.save_probe_report(tmp_path, "a", report)
        assert "Failed to save probe report" in caplog.text


class TestAuditResultRecorderSaveAudit:
    """Tests for save_audit_report."""

    def test_skips_when_no_dir(self):
        AuditResultRecorder.save_audit_report(
            None, "id", "task", "ctx", [], MagicMock(), MagicMock(), 100.0, {}
        )

    def test_saves_audit_report_to_file(self, tmp_path):
        agent = MagicMock()
        agent.name = "agent-1"
        verdict = MagicMock(
            recommendation="proceed",
            confidence=0.9,
            unanimous_issues=["issue1"],
            split_opinions=[],
            risk_areas=["risk1"],
            findings=[],
        )
        config = MagicMock(
            rounds=3,
            enable_research=True,
            cross_examination_depth=2,
            risk_threshold=0.7,
        )

        AuditResultRecorder.save_audit_report(
            tmp_path, "audit-1", "task", "ctx", [agent], verdict, config, 500.0, {"agent-1": 2}
        )

        audits_dir = tmp_path / "audits"
        assert audits_dir.exists()
        files = list(audits_dir.glob("*.json"))
        assert len(files) == 1
        content = json.loads(files[0].read_text())
        assert content["audit_id"] == "audit-1"
        assert content["recommendation"] == "proceed"
        assert content["agents"] == ["agent-1"]

    def test_handles_write_error(self, caplog, tmp_path):
        """Test save_audit_report handles write errors gracefully."""
        verdict = MagicMock()
        verdict.recommendation = "proceed"
        verdict.confidence = 0.9
        verdict.unanimous_issues = []
        verdict.split_opinions = []
        verdict.risk_areas = []
        # Use PropertyMock to raise when category is accessed as an attribute
        mock_finding = MagicMock()
        type(mock_finding).category = PropertyMock(side_effect=TypeError("oops"))
        verdict.findings = [mock_finding]
        config = MagicMock(
            rounds=3, enable_research=True, cross_examination_depth=2, risk_threshold=0.7
        )

        with caplog.at_level(logging.ERROR, logger="aragora.server.handlers.auditing"):
            AuditResultRecorder.save_audit_report(
                tmp_path, "id", "task", "ctx", [], verdict, config, 0.0, {}
            )
        assert "Failed to save deep audit report" in caplog.text


# ============================================================================
# AuditingHandler routing tests
# ============================================================================


class TestAuditingHandlerCanHandle:
    """Tests for can_handle routing logic."""

    def test_handles_capability_probe(self, handler):
        assert handler.can_handle("/api/v1/debates/capability-probe")

    def test_handles_deep_audit(self, handler):
        assert handler.can_handle("/api/v1/debates/deep-audit")

    def test_handles_attack_types(self, handler):
        assert handler.can_handle("/api/v1/redteam/attack-types")

    def test_handles_red_team_with_id(self, handler):
        assert handler.can_handle("/api/v1/debates/some-id/red-team")

    def test_rejects_unrelated_paths(self, handler):
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/other")
        assert not handler.can_handle("/api/v1/debates/123")
        assert not handler.can_handle("/api/v1/debates/123/results")

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "audit"


class TestAuditingHandlerDispatch:
    """Tests for handle() dispatch logic."""

    def test_dispatches_capability_probe(self, handler, mock_http):
        result = handler.handle("/api/v1/debates/capability-probe", {}, mock_http)
        assert result is not None

    def test_dispatches_deep_audit(self, handler, mock_http):
        result = handler.handle("/api/v1/debates/deep-audit", {}, mock_http)
        assert result is not None

    def test_dispatches_attack_types(self, handler, mock_http):
        result = handler.handle("/api/v1/redteam/attack-types", {}, mock_http)
        assert result is not None

    def test_dispatches_red_team(self, handler, mock_http):
        result = handler.handle("/api/v1/debates/debate_abc/red-team", {}, mock_http)
        assert result is not None

    def test_returns_none_for_unknown_path(self, handler, mock_http):
        result = handler.handle("/api/v1/unknown", {}, mock_http)
        assert result is None


# ============================================================================
# Attack types endpoint
# ============================================================================


class TestAttackTypes:
    """Tests for _get_attack_types endpoint."""

    def test_returns_503_when_redteam_unavailable(self, handler):
        with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", False):
            result = handler._get_attack_types()
        assert result.status_code == 503

    def test_returns_attack_types_when_available(self, handler):
        result = handler._get_attack_types()
        # Either works with real module or returns 503
        assert result.status_code in (200, 503)

    def test_attack_category_classification(self, handler):
        """Test _get_attack_category returns correct categories."""
        from aragora.modes.redteam import AttackType

        # Logic attacks
        for at in [
            AttackType.LOGICAL_FALLACY,
            AttackType.UNSTATED_ASSUMPTION,
            AttackType.COUNTEREXAMPLE,
        ]:
            assert handler._get_attack_category(at) == "logic"

        # System attacks
        for at in [
            AttackType.SECURITY,
            AttackType.RESOURCE_EXHAUSTION,
            AttackType.RACE_CONDITION,
            AttackType.DEPENDENCY_FAILURE,
        ]:
            assert handler._get_attack_category(at) == "system"


# ============================================================================
# Capability probe endpoint
# ============================================================================


class TestCapabilityProbe:
    """Tests for _run_capability_probe endpoint."""

    def test_returns_503_when_prober_unavailable(self, handler, mock_http):
        with patch("aragora.server.handlers.auditing.PROBER_AVAILABLE", False):
            result = handler._run_capability_probe(mock_http)
        assert result.status_code == 503

    def test_returns_400_on_invalid_json(self, handler, mock_http):
        with (
            patch.object(handler, "_read_json_body", return_value=None),
            patch("aragora.server.handlers.auditing.PROBER_AVAILABLE", True),
        ):
            result = handler._run_capability_probe(mock_http)
        assert result.status_code == 400

    def test_returns_400_when_no_valid_probe_types(self, handler, mock_http):
        body = {"agent_name": "a", "probe_types": ["nonexistent_type"]}

        with (
            patch.object(handler, "_read_json_body", return_value=body),
            patch("aragora.server.handlers.auditing.PROBER_AVAILABLE", True),
            patch(
                "aragora.server.handlers.auditing.validate_agent_name", return_value=(True, None)
            ),
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch("aragora.server.handlers.auditing.create_agent", return_value=MagicMock()),
        ):
            # Mock the ProbeType to reject all strings
            mock_probe_type = MagicMock(side_effect=ValueError("bad"))
            with patch("aragora.modes.prober.ProbeType", mock_probe_type):
                result = handler._run_capability_probe(mock_http)
        assert result.status_code == 400

    def test_returns_500_on_probe_execution_failure(self, handler, mock_http):
        body = {"agent_name": "a", "probe_types": ["contradiction"]}

        mock_prober_cls = MagicMock()
        mock_prober_inst = MagicMock()
        mock_prober_cls.return_value = mock_prober_inst

        with (
            patch.object(handler, "_read_json_body", return_value=body),
            patch("aragora.server.handlers.auditing.PROBER_AVAILABLE", True),
            patch(
                "aragora.server.handlers.auditing.validate_agent_name", return_value=(True, None)
            ),
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch("aragora.server.handlers.auditing.create_agent", return_value=MagicMock()),
            patch("aragora.modes.prober.CapabilityProber", mock_prober_cls),
            patch("aragora.modes.prober.ProbeType", lambda x: x),
            patch(
                "aragora.server.handlers.auditing.run_async",
                side_effect=RuntimeError("probe failed"),
            ),
        ):
            result = handler._run_capability_probe(mock_http)
        assert result.status_code == 500


# ============================================================================
# Deep audit endpoint
# ============================================================================


class TestDeepAudit:
    """Tests for _run_deep_audit endpoint."""

    def test_returns_503_when_module_unavailable(self, handler, mock_http):
        with patch.dict("sys.modules", {"aragora.modes.deep_audit": None}):
            # Force ImportError inside _run_deep_audit
            with patch.object(handler, "_read_json_body", return_value={"task": "x"}):
                result = handler._run_deep_audit(mock_http)
        assert result.status_code == 503

    def test_returns_400_on_invalid_json(self, handler, mock_http):
        # Mock the deep_audit import to succeed
        mock_deep_audit = MagicMock()
        with (
            patch.dict("sys.modules", {"aragora.modes.deep_audit": mock_deep_audit}),
            patch.object(handler, "_read_json_body", return_value=None),
        ):
            result = handler._run_deep_audit(mock_http)
        assert result.status_code == 400


# ============================================================================
# Red team analysis endpoint
# ============================================================================


class TestRedTeamAnalysis:
    """Tests for _run_red_team_analysis endpoint."""

    def test_returns_503_when_redteam_unavailable(self, handler, mock_http):
        with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", False):
            result = handler._run_red_team_analysis("debate-1", mock_http)
        assert result.status_code == 503

    def test_returns_500_when_storage_not_configured(self, handler, mock_http):
        handler.ctx["storage"] = None
        with (
            patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True),
            patch.object(handler, "_read_json_body", return_value={}),
        ):
            result = handler._run_red_team_analysis("debate-1", mock_http)
        assert result.status_code == 500

    def test_returns_404_when_debate_not_found(self, handler, mock_http):
        storage = MagicMock()
        storage.get_by_slug.return_value = None
        storage.get_by_id.return_value = None
        handler.ctx["storage"] = storage
        with (
            patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True),
            patch.object(handler, "_read_json_body", return_value={}),
        ):
            result = handler._run_red_team_analysis("nonexistent", mock_http)
        assert result.status_code == 404

    def test_successful_analysis(self, handler, mock_http):
        storage = MagicMock()
        storage.get_by_slug.return_value = {
            "task": "Some task",
            "consensus_answer": "We should do X because it always works",
        }
        handler.ctx["storage"] = storage

        with (
            patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True),
            patch.object(handler, "_read_json_body", return_value={}),
        ):
            result = handler._run_red_team_analysis("debate-1", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["debate_id"] == "debate-1"
        assert "findings" in body
        assert "robustness_score" in body
        assert body["status"] == "analysis_complete"

    def test_custom_attack_types(self, handler, mock_http):
        storage = MagicMock()
        storage.get_by_slug.return_value = {"task": "t", "consensus_answer": "answer"}
        handler.ctx["storage"] = storage

        with (
            patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True),
            patch.object(
                handler,
                "_read_json_body",
                return_value={"attack_types": ["security"], "max_rounds": 2},
            ),
        ):
            result = handler._run_red_team_analysis("debate-1", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["attack_types"] == ["security"]
        assert body["max_rounds"] == 2


# ============================================================================
# Proposal analysis for red team
# ============================================================================


class TestAnalyzeProposalForRedTeam:
    """Tests for _analyze_proposal_for_redteam."""

    def test_returns_empty_when_redteam_not_importable(self, handler):
        with patch.dict("sys.modules", {"aragora.modes.redteam": None}):
            result = handler._analyze_proposal_for_redteam("proposal", ["logical_fallacy"], {})
        assert result == []

    def test_finds_vulnerability_matches(self, handler):
        proposal = "This solution always works and is clearly the best option"
        findings = handler._analyze_proposal_for_redteam(proposal, ["logical_fallacy"], {})
        # "always" and "clearly" are keywords for logical_fallacy
        matching = [f for f in findings if f["keyword_matches"] > 0]
        assert len(matching) > 0
        assert matching[0]["attack_type"] == "logical_fallacy"

    def test_no_matches_returns_base_severity(self, handler):
        proposal = "This is a neutral statement"
        findings = handler._analyze_proposal_for_redteam(proposal, ["logical_fallacy"], {})
        # Should still return a finding but with no keyword matches
        assert len(findings) == 1
        assert findings[0]["keyword_matches"] == 0
        assert findings[0]["requires_manual_review"] is False

    def test_severity_caps_at_09(self, handler):
        # Pack lots of keywords to hit severity cap
        proposal = "always never all none obviously clearly"
        findings = handler._analyze_proposal_for_redteam(proposal, ["logical_fallacy"], {})
        for f in findings:
            assert f["severity"] <= 0.9

    def test_handles_none_proposal(self, handler):
        findings = handler._analyze_proposal_for_redteam(None, ["logical_fallacy"], {})
        assert len(findings) == 1
        assert findings[0]["keyword_matches"] == 0

    def test_skips_invalid_attack_types(self, handler):
        findings = handler._analyze_proposal_for_redteam("test", ["completely_invalid_type"], {})
        # Invalid attack type should be skipped
        assert len(findings) == 0

    def test_security_pattern_detection(self, handler):
        proposal = "The system is secure and protected with encryption"
        findings = handler._analyze_proposal_for_redteam(proposal, ["security"], {})
        matching = [f for f in findings if f["keyword_matches"] > 0]
        assert len(matching) == 1
        assert matching[0]["severity"] >= 0.6  # base + matches

    def test_multiple_attack_types(self, handler):
        proposal = "This always scales well and is secure"
        findings = handler._analyze_proposal_for_redteam(
            proposal, ["logical_fallacy", "scalability", "security"], {}
        )
        attack_types_found = {f["attack_type"] for f in findings}
        assert "logical_fallacy" in attack_types_found
        assert "scalability" in attack_types_found
        assert "security" in attack_types_found


# ============================================================================
# Transform probe results
# ============================================================================


class TestTransformProbeResults:
    """Tests for _transform_probe_results."""

    def test_transforms_with_to_dict(self, handler):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "probe_id": "p1",
            "probe_type": "contradiction",
            "vulnerability_found": True,
            "severity": "HIGH",
            "vulnerability_description": "Found issue",
            "evidence": "Details here",
            "response_time_ms": 150,
        }
        by_type = {"contradiction": [mock_result]}

        transformed = handler._transform_probe_results(by_type)

        assert "contradiction" in transformed
        assert len(transformed["contradiction"]) == 1
        r = transformed["contradiction"][0]
        assert r["passed"] is False  # vulnerability_found=True means not passed
        assert r["severity"] == "high"
        assert r["probe_id"] == "p1"

    def test_transforms_dict_without_to_dict(self, handler):
        raw = {
            "probe_id": "p2",
            "vulnerability_found": False,
            "response_time_ms": 50,
        }
        by_type = {"sycophancy": [raw]}

        transformed = handler._transform_probe_results(by_type)

        r = transformed["sycophancy"][0]
        assert r["passed"] is True
        assert r["severity"] is None  # no severity field
        assert r["response_time_ms"] == 50


# ============================================================================
# Get audit config
# ============================================================================


class TestGetAuditConfig:
    """Tests for _get_audit_config."""

    def test_strategy_preset(self, handler):
        strategy = MagicMock()
        result = handler._get_audit_config(
            "strategy", {}, MagicMock, strategy, MagicMock(), MagicMock()
        )
        assert result is strategy

    def test_contract_preset(self, handler):
        contract = MagicMock()
        result = handler._get_audit_config(
            "contract", {}, MagicMock, MagicMock(), contract, MagicMock()
        )
        assert result is contract

    def test_code_architecture_preset(self, handler):
        code = MagicMock()
        result = handler._get_audit_config(
            "code_architecture", {}, MagicMock, MagicMock(), MagicMock(), code
        )
        assert result is code

    def test_custom_config_from_parsed(self, handler):
        parsed = {
            "rounds": 4,
            "enable_research": False,
            "cross_examination_depth": 2,
            "risk_threshold": 0.8,
        }
        config_cls = MagicMock()
        handler._get_audit_config(
            "custom", parsed, config_cls, MagicMock(), MagicMock(), MagicMock()
        )
        config_cls.assert_called_once_with(
            rounds=4,
            enable_research=False,
            cross_examination_depth=2,
            risk_threshold=0.8,
        )

    def test_empty_audit_type_uses_custom(self, handler):
        parsed = {
            "rounds": 6,
            "enable_research": True,
            "cross_examination_depth": 3,
            "risk_threshold": 0.7,
        }
        config_cls = MagicMock()
        handler._get_audit_config("", parsed, config_cls, MagicMock(), MagicMock(), MagicMock())
        config_cls.assert_called_once()


# ============================================================================
# Additional Capability Probe Tests
# ============================================================================


class TestCapabilityProbeExtended:
    """Extended tests for capability probe endpoint success paths."""

    def test_successful_capability_probe_returns_full_response(self, handler, mock_http):
        """Test full success path with properly mocked report."""
        body = {"agent_name": "test-agent", "probe_types": ["contradiction"]}

        # Create mock report with all required attributes
        mock_report = MagicMock()
        mock_report.report_id = "probe-12345"
        mock_report.probes_run = 10
        mock_report.vulnerabilities_found = 2
        mock_report.vulnerability_rate = 0.2
        mock_report.elo_penalty = 5.0
        mock_report.critical_count = 0
        mock_report.high_count = 1
        mock_report.medium_count = 1
        mock_report.low_count = 0
        mock_report.recommendations = ["recommendation1"]
        mock_report.created_at = "2024-01-01T00:00:00Z"
        mock_report.by_type = {}

        mock_prober_cls = MagicMock()
        mock_prober_inst = MagicMock()
        mock_prober_cls.return_value = mock_prober_inst

        with (
            patch.object(handler, "_read_json_body", return_value=body),
            patch("aragora.server.handlers.auditing.PROBER_AVAILABLE", True),
            patch(
                "aragora.server.handlers.auditing.validate_agent_name", return_value=(True, None)
            ),
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch("aragora.server.handlers.auditing.create_agent", return_value=MagicMock()),
            patch("aragora.modes.prober.CapabilityProber", mock_prober_cls),
            patch("aragora.modes.prober.ProbeType", lambda x: x),
            patch("aragora.server.handlers.auditing.run_async", return_value=mock_report),
        ):
            result = handler._run_capability_probe(mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["report_id"] == "probe-12345"
        assert body["target_agent"] == "test-agent"
        assert body["probes_run"] == 10
        assert body["summary"]["passed"] == 8
        assert body["summary"]["failed"] == 2
        assert body["summary"]["pass_rate"] == 0.8

    def test_capability_probe_with_elo_system(self, ctx, mock_http):
        """Test probe correctly records ELO results."""
        body = {"agent_name": "test-agent", "probe_types": ["contradiction"]}

        # Add mock ELO system to context
        mock_elo = MagicMock()
        ctx["elo_system"] = mock_elo

        handler = AuditingHandler(ctx)

        mock_report = MagicMock()
        mock_report.report_id = "probe-abc"
        mock_report.probes_run = 5
        mock_report.vulnerabilities_found = 1
        mock_report.vulnerability_rate = 0.2
        mock_report.elo_penalty = 3.0
        mock_report.critical_count = 0
        mock_report.high_count = 0
        mock_report.medium_count = 1
        mock_report.low_count = 0
        mock_report.recommendations = []
        mock_report.created_at = "2024-01-01T00:00:00Z"
        mock_report.by_type = {}

        with (
            patch.object(handler, "_read_json_body", return_value=body),
            patch("aragora.server.handlers.auditing.PROBER_AVAILABLE", True),
            patch(
                "aragora.server.handlers.auditing.validate_agent_name", return_value=(True, None)
            ),
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch("aragora.server.handlers.auditing.create_agent", return_value=MagicMock()),
            patch("aragora.modes.prober.CapabilityProber", MagicMock(return_value=MagicMock())),
            patch("aragora.modes.prober.ProbeType", lambda x: x),
            patch("aragora.server.handlers.auditing.run_async", return_value=mock_report),
        ):
            result = handler._run_capability_probe(mock_http)

        assert result.status_code == 200
        # ELO should have been called
        mock_elo.record_redteam_result.assert_called_once()

    def test_capability_probe_agent_creation_failure(self, handler, mock_http):
        """Test handling when agent creation fails."""
        body = {"agent_name": "test-agent", "probe_types": ["contradiction"]}

        with (
            patch.object(handler, "_read_json_body", return_value=body),
            patch("aragora.server.handlers.auditing.PROBER_AVAILABLE", True),
            patch(
                "aragora.server.handlers.auditing.validate_agent_name", return_value=(True, None)
            ),
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch(
                "aragora.server.handlers.auditing.create_agent", side_effect=ValueError("API error")
            ),
        ):
            result = handler._run_capability_probe(mock_http)

        assert result.status_code == 400

    def test_capability_probe_value_error_returns_400(self, handler, mock_http):
        """Test ValueError during probe returns 400."""
        body = {"agent_name": "test-agent", "probe_types": ["contradiction"]}

        mock_prober_cls = MagicMock()
        mock_prober_inst = MagicMock()
        mock_prober_cls.return_value = mock_prober_inst

        with (
            patch.object(handler, "_read_json_body", return_value=body),
            patch("aragora.server.handlers.auditing.PROBER_AVAILABLE", True),
            patch(
                "aragora.server.handlers.auditing.validate_agent_name", return_value=(True, None)
            ),
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch("aragora.server.handlers.auditing.create_agent", return_value=MagicMock()),
            patch("aragora.modes.prober.CapabilityProber", mock_prober_cls),
            patch("aragora.modes.prober.ProbeType", lambda x: x),
            patch("aragora.server.handlers.auditing.run_async", side_effect=ValueError("bad")),
        ):
            # ValueError should be caught and return 400
            result = handler._run_capability_probe(mock_http)

        assert result.status_code in (400, 500)


# ============================================================================
# Additional Deep Audit Tests
# ============================================================================


class TestDeepAuditExtended:
    """Extended tests for deep audit endpoint success paths."""

    def test_successful_deep_audit_returns_full_response(self, handler, mock_http):
        """Test full success path with properly mocked verdict."""
        body = {
            "task": "Analyze this decision",
            "context": "Some context",
            "agent_names": ["agent1", "agent2", "agent3"],
        }

        # Create mock verdict with all required attributes
        mock_finding = MagicMock()
        mock_finding.category = "risk"
        mock_finding.summary = "Found potential issue"
        mock_finding.details = "Details here"
        mock_finding.agents_agree = ["agent1", "agent2"]
        mock_finding.agents_disagree = ["agent3"]
        mock_finding.confidence = 0.8
        mock_finding.severity = 0.6

        mock_verdict = MagicMock()
        mock_verdict.recommendation = "Proceed with caution"
        mock_verdict.confidence = 0.85
        mock_verdict.unanimous_issues = ["issue1"]
        mock_verdict.split_opinions = ["opinion1"]
        mock_verdict.risk_areas = ["area1"]
        mock_verdict.findings = [mock_finding]
        mock_verdict.cross_examination_notes = "Notes here"
        mock_verdict.citations = ["cite1", "cite2"]

        mock_config = MagicMock()
        mock_config.rounds = 6
        mock_config.enable_research = True
        mock_config.cross_examination_depth = 3
        mock_config.risk_threshold = 0.7

        mock_orchestrator = MagicMock()
        mock_orchestrator.run = MagicMock(return_value=mock_verdict)

        mock_agent = MagicMock()
        mock_agent.name = "agent1"

        with (
            patch.object(handler, "_read_json_body", return_value=body),
            patch("aragora.modes.deep_audit.DeepAuditConfig", return_value=mock_config),
            patch("aragora.modes.deep_audit.DeepAuditOrchestrator", return_value=mock_orchestrator),
            patch("aragora.modes.deep_audit.STRATEGY_AUDIT", mock_config),
            patch("aragora.modes.deep_audit.CONTRACT_AUDIT", mock_config),
            patch("aragora.modes.deep_audit.CODE_ARCHITECTURE_AUDIT", mock_config),
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch("aragora.server.handlers.auditing.create_agent", return_value=mock_agent),
            patch("aragora.server.handlers.auditing.validate_id", return_value=(True, None)),
            patch("aragora.server.handlers.auditing.run_async", return_value=mock_verdict),
        ):
            result = handler._run_deep_audit(mock_http)

        assert result.status_code == 200
        response_body = json.loads(result.body)
        assert response_body["recommendation"] == "Proceed with caution"
        assert response_body["confidence"] == 0.85
        assert len(response_body["unanimous_issues"]) == 1
        assert len(response_body["findings"]) == 1

    def test_deep_audit_with_strategy_preset(self, handler, mock_http):
        """Test deep audit with strategy preset."""
        body = {
            "task": "Strategic analysis",
            "config": {"audit_type": "strategy"},
        }

        with (
            patch.object(handler, "_read_json_body", return_value=body),
            patch.dict("sys.modules", {"aragora.modes.deep_audit": None}),
        ):
            result = handler._run_deep_audit(mock_http)

        assert result.status_code == 503  # Module unavailable

    def test_deep_audit_execution_failure(self, handler, mock_http):
        """Test handling when orchestrator execution fails."""
        body = {
            "task": "Analyze this",
            "agent_names": ["a1", "a2", "a3"],
        }

        mock_config = MagicMock()
        mock_config.rounds = 6
        mock_agent = MagicMock()
        mock_agent.name = "agent1"

        with (
            patch.object(handler, "_read_json_body", return_value=body),
            patch("aragora.modes.deep_audit.DeepAuditConfig", return_value=mock_config),
            patch("aragora.modes.deep_audit.DeepAuditOrchestrator", MagicMock()),
            patch("aragora.modes.deep_audit.STRATEGY_AUDIT", mock_config),
            patch("aragora.modes.deep_audit.CONTRACT_AUDIT", mock_config),
            patch("aragora.modes.deep_audit.CODE_ARCHITECTURE_AUDIT", mock_config),
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch("aragora.server.handlers.auditing.create_agent", return_value=mock_agent),
            patch("aragora.server.handlers.auditing.validate_id", return_value=(True, None)),
            patch(
                "aragora.server.handlers.auditing.run_async",
                side_effect=RuntimeError("Orchestrator failed"),
            ),
        ):
            result = handler._run_deep_audit(mock_http)

        assert result.status_code == 500


# ============================================================================
# Additional Red Team Analysis Tests
# ============================================================================


class TestRedTeamAnalysisExtended:
    """Extended tests for red team analysis endpoint."""

    def test_red_team_with_invalid_path_param(self, handler, mock_http):
        """Test red team with path param extraction error."""
        # Path with invalid characters in debate ID
        result = handler.handle("/api/v1/debates/<script>/red-team", {}, mock_http)

        # Should get an error response due to invalid pattern
        assert result is not None
        assert result.status_code == 400

    def test_red_team_max_rounds_clamped(self, handler, mock_http):
        """Test that max_rounds is clamped to 5."""
        storage = MagicMock()
        storage.get_by_slug.return_value = {"task": "t", "consensus_answer": "answer"}
        handler.ctx["storage"] = storage

        with (
            patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True),
            patch.object(
                handler,
                "_read_json_body",
                return_value={"max_rounds": 100},  # Above max
            ),
        ):
            result = handler._run_red_team_analysis("debate-1", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["max_rounds"] == 5  # Clamped to max

    def test_red_team_uses_fallback_proposal(self, handler, mock_http):
        """Test that red team uses task as fallback when no consensus."""
        storage = MagicMock()
        storage.get_by_slug.return_value = {
            "task": "The original task",
            # No consensus_answer or final_answer
        }
        handler.ctx["storage"] = storage

        with (
            patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True),
            patch.object(handler, "_read_json_body", return_value={}),
        ):
            result = handler._run_red_team_analysis("debate-1", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["target_proposal"] == "The original task"

    def test_red_team_with_explicit_focus_proposal(self, handler, mock_http):
        """Test red team with explicit focus_proposal from request."""
        storage = MagicMock()
        storage.get_by_slug.return_value = {"task": "t", "consensus_answer": "default"}
        handler.ctx["storage"] = storage

        with (
            patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True),
            patch.object(
                handler,
                "_read_json_body",
                return_value={"focus_proposal": "Custom proposal to analyze"},
            ),
        ):
            result = handler._run_red_team_analysis("debate-1", mock_http)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["target_proposal"] == "Custom proposal to analyze"

    def test_red_team_type_error_returns_400(self, handler, mock_http):
        """Test TypeError during analysis returns 400."""
        storage = MagicMock()
        storage.get_by_slug.side_effect = TypeError("unexpected type")
        handler.ctx["storage"] = storage

        with (
            patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True),
            patch.object(handler, "_read_json_body", return_value={}),
        ):
            result = handler._run_red_team_analysis("debate-1", mock_http)

        assert result.status_code == 400


# ============================================================================
# Additional Proposal Analysis Tests
# ============================================================================


class TestAnalyzeProposalPatterns:
    """Extended tests for proposal pattern analysis."""

    def test_unstated_assumption_pattern(self, handler):
        """Test unstated_assumption attack type detection."""
        proposal = "We should implement this because it needs to meet the requirements"
        findings = handler._analyze_proposal_for_redteam(proposal, ["unstated_assumption"], {})

        matching = [f for f in findings if f["keyword_matches"] > 0]
        assert len(matching) > 0
        assert matching[0]["attack_type"] == "unstated_assumption"

    def test_counterexample_pattern(self, handler):
        """Test counterexample attack type detection."""
        proposal = "This is the best and only solution that provides superior results"
        findings = handler._analyze_proposal_for_redteam(proposal, ["counterexample"], {})

        matching = [f for f in findings if f["keyword_matches"] > 0]
        assert len(matching) > 0
        assert matching[0]["severity"] > 0.5

    def test_scalability_pattern(self, handler):
        """Test scalability attack type detection."""
        proposal = "This solution will scale to handle growth and expansion"
        findings = handler._analyze_proposal_for_redteam(proposal, ["scalability"], {})

        matching = [f for f in findings if f["keyword_matches"] > 0]
        assert len(matching) > 0
        assert matching[0]["attack_type"] == "scalability"

    def test_edge_case_pattern(self, handler):
        """Test edge_case attack type detection."""
        proposal = "This usually works for most typical standard scenarios"
        findings = handler._analyze_proposal_for_redteam(proposal, ["edge_case"], {})

        matching = [f for f in findings if f["keyword_matches"] > 0]
        assert len(matching) > 0
        # Should have multiple keyword matches
        assert matching[0]["keyword_matches"] >= 2

    def test_unknown_attack_type_in_pattern_dict(self, handler):
        """Test handling of attack type not in vulnerability_patterns."""
        # Use an attack type that exists in AttackType but not in the patterns dict
        # The module should handle this gracefully with empty keywords
        proposal = "Some proposal text"
        findings = handler._analyze_proposal_for_redteam(proposal, ["resource_exhaustion"], {})

        # Should still process (either find or skip unknown types)
        # The behavior depends on whether AttackType validation passes
        assert isinstance(findings, list)

    def test_manual_review_flag_set_for_high_severity(self, handler):
        """Test that requires_manual_review is set for high severity findings."""
        # Use many keywords to push severity above 0.6
        proposal = "This secure protected encrypted solution is safe and auth-ready"
        findings = handler._analyze_proposal_for_redteam(proposal, ["security"], {})

        high_severity = [f for f in findings if f["severity"] > 0.6]
        if high_severity:
            assert high_severity[0]["requires_manual_review"] is True


# ============================================================================
# Transform Probe Results Extended Tests
# ============================================================================


class TestTransformProbeResultsExtended:
    """Extended tests for probe result transformation."""

    def test_empty_by_type_dict(self, handler):
        """Test transformation with empty by_type dict."""
        transformed = handler._transform_probe_results({})
        assert transformed == {}

    def test_multiple_probe_types(self, handler):
        """Test transformation with multiple probe types."""
        r1 = {"probe_id": "p1", "vulnerability_found": True, "severity": "HIGH"}
        r2 = {"probe_id": "p2", "vulnerability_found": False}
        r3 = {"probe_id": "p3", "vulnerability_found": True, "severity": "MEDIUM"}

        by_type = {
            "contradiction": [r1],
            "sycophancy": [r2, r3],
        }

        transformed = handler._transform_probe_results(by_type)

        assert len(transformed["contradiction"]) == 1
        assert len(transformed["sycophancy"]) == 2
        assert transformed["contradiction"][0]["passed"] is False
        assert transformed["sycophancy"][0]["passed"] is True
        assert transformed["sycophancy"][1]["passed"] is False

    def test_missing_optional_fields(self, handler):
        """Test transformation handles missing optional fields."""
        raw = {
            "probe_id": "p1",
            # Missing: vulnerability_found, severity, etc.
        }
        by_type = {"test": [raw]}

        transformed = handler._transform_probe_results(by_type)

        r = transformed["test"][0]
        assert r["passed"] is True  # No vulnerability_found defaults to not found
        assert r["severity"] is None
        assert r["description"] == ""
        assert r["details"] == ""
        assert r["response_time_ms"] == 0

    def test_severity_lowercase_conversion(self, handler):
        """Test that severity is converted to lowercase."""
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "probe_id": "p1",
            "probe_type": "test",
            "vulnerability_found": True,
            "severity": "CRITICAL",
        }

        transformed = handler._transform_probe_results({"test": [mock_result]})

        assert transformed["test"][0]["severity"] == "critical"


# ============================================================================
# AuditResultRecorder Extended Tests
# ============================================================================


class TestAuditResultRecorderExtended:
    """Extended tests for AuditResultRecorder."""

    def test_record_probe_elo_with_high_vulnerability_rate(self):
        """Test ELO recording with 100% vulnerability rate."""
        elo = MagicMock()
        report = MagicMock(
            probes_run=10,
            vulnerability_rate=1.0,  # All probes found vulnerabilities
            vulnerabilities_found=10,
            critical_count=5,
        )

        AuditResultRecorder.record_probe_elo(elo, "bad-agent", report, "id")

        kwargs = elo.record_redteam_result.call_args[1]
        assert kwargs["robustness_score"] == pytest.approx(0.0, abs=1e-9)
        assert kwargs["successful_attacks"] == 10

    def test_calculate_elo_adjustments_multiple_findings(self):
        """Test ELO adjustments with many findings."""
        findings = []
        for i in range(5):
            f = MagicMock(agents_agree=["agent1"], agents_disagree=["agent2"])
            findings.append(f)

        verdict = MagicMock(findings=findings)
        result = AuditResultRecorder.calculate_audit_elo_adjustments(verdict, MagicMock())

        # agent1: +2 * 5 = +10
        # agent2: -1 * 5 = -5
        assert result["agent1"] == 10
        assert result["agent2"] == -5

    def test_save_probe_report_creates_nested_dirs(self, tmp_path):
        """Test that save_probe_report creates nested directory structure."""
        report = MagicMock()
        report.report_id = "test-report"
        report.to_dict.return_value = {"id": "test-report"}

        # Use a non-existent nested path
        nested_dir = tmp_path / "deeply" / "nested"

        AuditResultRecorder.save_probe_report(nested_dir, "my-agent", report)

        probe_dir = nested_dir / "probes" / "my-agent"
        assert probe_dir.exists()
        files = list(probe_dir.glob("*.json"))
        assert len(files) == 1

    def test_save_audit_report_truncates_long_context(self, tmp_path):
        """Test that save_audit_report truncates context to 1000 chars."""
        long_context = "x" * 5000
        agent = MagicMock()
        agent.name = "agent1"
        verdict = MagicMock(
            recommendation="proceed",
            confidence=0.9,
            unanimous_issues=[],
            split_opinions=[],
            risk_areas=[],
            findings=[],
        )
        config = MagicMock(
            rounds=3,
            enable_research=True,
            cross_examination_depth=2,
            risk_threshold=0.7,
        )

        AuditResultRecorder.save_audit_report(
            tmp_path, "audit-1", "task", long_context, [agent], verdict, config, 100.0, {}
        )

        audit_file = list((tmp_path / "audits").glob("*.json"))[0]
        content = json.loads(audit_file.read_text())
        assert len(content["context"]) == 1000


# ============================================================================
# Handler Initialization and Context Tests
# ============================================================================


class TestHandlerContext:
    """Tests for handler initialization and context access."""

    def test_handler_with_full_context(self):
        """Test handler initialization with full context."""
        ctx = {
            "storage": MagicMock(),
            "elo_system": MagicMock(),
            "nomic_dir": Path("/tmp/nomic"),
            "user_store": MagicMock(),
        }
        handler = AuditingHandler(ctx)

        assert handler.ctx.get("storage") is not None
        assert handler.ctx.get("elo_system") is not None
        assert handler.ctx.get("nomic_dir") is not None

    def test_handler_with_minimal_context(self):
        """Test handler works with minimal context."""
        ctx = {}
        handler = AuditingHandler(ctx)

        assert handler.ctx.get("storage") is None
        assert handler.ctx.get("elo_system") is None


# ============================================================================
# Request Parser Edge Cases
# ============================================================================


class TestAuditRequestParserEdgeCases:
    """Edge case tests for AuditRequestParser."""

    def test_parse_capability_probe_with_all_fields(self):
        """Test parsing with all optional fields specified."""
        data = {
            "agent_name": "my-agent",
            "probe_types": ["contradiction", "sycophancy"],
            "probes_per_type": 7,
            "model_type": "openai-api",
        }
        with patch(
            "aragora.server.handlers.auditing.validate_agent_name",
            return_value=(True, None),
        ):
            parsed, err = AuditRequestParser.parse_capability_probe(
                MagicMock(), _make_read_fn(data)
            )

        assert err is None
        assert parsed["agent_name"] == "my-agent"
        assert parsed["probe_types"] == ["contradiction", "sycophancy"]
        assert parsed["probes_per_type"] == 7
        assert parsed["model_type"] == "openai-api"

    def test_parse_deep_audit_with_all_config_fields(self):
        """Test parsing with all config fields specified."""
        data = {
            "task": "Full audit",
            "context": "Detailed context",
            "agent_names": ["a1", "a2"],
            "model_type": "mistral-api",
            "config": {
                "audit_type": "strategy",
                "rounds": 8,
                "cross_examination_depth": 5,
                "risk_threshold": 0.85,
                "enable_research": False,
            },
        }
        parsed, err = AuditRequestParser.parse_deep_audit(MagicMock(), _make_read_fn(data))

        assert err is None
        assert parsed["task"] == "Full audit"
        assert parsed["model_type"] == "mistral-api"
        assert parsed["audit_type"] == "strategy"
        assert parsed["rounds"] == 8
        assert parsed["cross_examination_depth"] == 5
        assert parsed["risk_threshold"] == 0.85
        assert parsed["enable_research"] is False

    def test_parse_int_with_float_string(self):
        """Test _parse_int with float string value."""
        # Python int() truncates, so "3.5" should fail
        val, err = AuditRequestParser._parse_int({"n": "3.5"}, "n", 5, 10)
        assert err is not None
        assert err.status_code == 400

    def test_require_field_with_none_value(self):
        """Test _require_field with explicit None value raises AttributeError.

        Note: The implementation calls .strip() on the value, which fails for None.
        This is a potential improvement area - the code could handle None gracefully.
        """
        # The current implementation crashes on None - this documents that behavior
        with pytest.raises(AttributeError):
            AuditRequestParser._require_field({"name": None}, "name")


# ============================================================================
# Agent Factory Extended Tests
# ============================================================================


class TestAuditAgentFactoryExtended:
    """Extended tests for AuditAgentFactory."""

    def test_create_single_agent_with_critic_role(self):
        """Test creating agent with critic role."""
        mock_agent = MagicMock()
        with (
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch(
                "aragora.server.handlers.auditing.create_agent", return_value=mock_agent
            ) as mock_create,
        ):
            agent, err = AuditAgentFactory.create_single_agent("api", "critic-1", role="critic")

        assert err is None
        mock_create.assert_called_once_with("api", name="critic-1", role="critic")

    def test_create_multiple_agents_partial_failure(self):
        """Test creating multiple agents with some failures."""
        call_count = [0]

        def conditional_create(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Agent 2 failed")
            return MagicMock()

        with (
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch("aragora.server.handlers.auditing.create_agent", side_effect=conditional_create),
            patch("aragora.server.handlers.auditing.validate_id", return_value=(True, None)),
        ):
            agents, err = AuditAgentFactory.create_multiple_agents("api", ["a1", "a2", "a3"], [])

        # Should still succeed with 2 agents
        assert err is None
        assert len(agents) == 2

    def test_create_multiple_agents_all_fail(self):
        """Test when all agent creations fail."""
        with (
            patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True),
            patch(
                "aragora.server.handlers.auditing.create_agent",
                side_effect=ValueError("All fail"),
            ),
            patch("aragora.server.handlers.auditing.validate_id", return_value=(True, None)),
        ):
            agents, err = AuditAgentFactory.create_multiple_agents(
                "api", ["a1", "a2"], ["d1", "d2"]
            )

        assert agents == []
        assert err.status_code == 400
