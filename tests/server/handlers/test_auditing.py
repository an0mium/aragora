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
                side_effect=Exception("fail"),
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
        verdict = MagicMock()
        verdict.recommendation = "proceed"
        verdict.confidence = 0.9
        verdict.unanimous_issues = []
        verdict.split_opinions = []
        verdict.risk_areas = []
        verdict.findings = [MagicMock()]
        # Make findings iteration fail
        verdict.findings[0].category = MagicMock(side_effect=TypeError("oops"))
        config = MagicMock(
            rounds=3, enable_research=True, cross_examination_depth=2, risk_threshold=0.7
        )

        with caplog.at_level(logging.ERROR):
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
        try:
            from aragora.modes.redteam import AttackType
        except ImportError:
            pytest.skip("redteam module not available")

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
