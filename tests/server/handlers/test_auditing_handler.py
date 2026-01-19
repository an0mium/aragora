"""
Tests for the AuditingHandler module.

Tests cover:
- Handler routing for all auditing endpoints
- AuditRequestParser utility class
- AuditAgentFactory utility class
- AuditResultRecorder utility class
- can_handle method
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from aragora.server.handlers.auditing import (
    AuditingHandler,
    AuditRequestParser,
    AuditAgentFactory,
    AuditResultRecorder,
)


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


class TestAuditingHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return AuditingHandler(mock_server_context)

    def test_can_handle_capability_probe(self, handler):
        """Handler can handle capability-probe endpoint."""
        assert handler.can_handle("/api/debates/capability-probe")

    def test_can_handle_deep_audit(self, handler):
        """Handler can handle deep-audit endpoint."""
        assert handler.can_handle("/api/debates/deep-audit")

    def test_can_handle_attack_types(self, handler):
        """Handler can handle attack-types endpoint."""
        assert handler.can_handle("/api/redteam/attack-types")

    def test_can_handle_red_team(self, handler):
        """Handler can handle red-team endpoint with debate ID."""
        assert handler.can_handle("/api/debates/debate_123/red-team")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/other")
        assert not handler.can_handle("/api/debates/123")


class TestAuditingHandlerRoutesAttribute:
    """Tests for ROUTES class attribute."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return AuditingHandler(mock_server_context)

    def test_routes_contains_capability_probe(self, handler):
        """ROUTES contains capability-probe."""
        assert "/api/debates/capability-probe" in handler.ROUTES

    def test_routes_contains_deep_audit(self, handler):
        """ROUTES contains deep-audit."""
        assert "/api/debates/deep-audit" in handler.ROUTES

    def test_routes_contains_attack_types(self, handler):
        """ROUTES contains attack-types."""
        assert "/api/redteam/attack-types" in handler.ROUTES


class TestAuditingHandlerRouteDispatch:
    """Tests for route dispatch logic."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return AuditingHandler(mock_server_context)

    def test_handle_dispatches_capability_probe(self, handler):
        """Handle dispatches capability-probe to correct method."""
        mock_http = MagicMock()

        result = handler.handle("/api/debates/capability-probe", {}, mock_http)

        # Should return a result (permission error expected)
        assert result is not None

    def test_handle_dispatches_deep_audit(self, handler):
        """Handle dispatches deep-audit to correct method."""
        mock_http = MagicMock()

        result = handler.handle("/api/debates/deep-audit", {}, mock_http)

        assert result is not None

    def test_handle_dispatches_attack_types(self, handler):
        """Handle dispatches attack-types to correct method."""
        mock_http = MagicMock()

        result = handler.handle("/api/redteam/attack-types", {}, mock_http)

        assert result is not None

    def test_handle_dispatches_red_team(self, handler):
        """Handle dispatches red-team to correct method."""
        mock_http = MagicMock()

        result = handler.handle("/api/debates/debate_123/red-team", {}, mock_http)

        assert result is not None

    def test_handle_returns_none_for_unknown(self, handler):
        """Handle returns None for unknown paths."""
        mock_http = MagicMock()

        result = handler.handle("/api/other", {}, mock_http)

        assert result is None


class TestAuditRequestParser:
    """Tests for AuditRequestParser utility class."""

    def test_read_json_returns_error_for_none(self):
        """_read_json returns error when read_json_fn returns None."""
        mock_handler = MagicMock()
        mock_read_fn = MagicMock(return_value=None)

        data, err = AuditRequestParser._read_json(mock_handler, mock_read_fn)

        assert data is None
        assert err is not None
        assert err.status_code == 400

    def test_read_json_returns_data(self):
        """_read_json returns data when read_json_fn succeeds."""
        mock_handler = MagicMock()
        mock_read_fn = MagicMock(return_value={"key": "value"})

        data, err = AuditRequestParser._read_json(mock_handler, mock_read_fn)

        assert data == {"key": "value"}
        assert err is None

    def test_require_field_returns_error_for_missing(self):
        """_require_field returns error for missing field."""
        data = {"other_field": "value"}

        value, err = AuditRequestParser._require_field(data, "required_field")

        assert value is None
        assert err is not None
        assert err.status_code == 400

    def test_require_field_returns_value(self):
        """_require_field returns value when present."""
        data = {"name": "test_value"}

        value, err = AuditRequestParser._require_field(data, "name")

        assert value == "test_value"
        assert err is None

    def test_require_field_strips_whitespace(self):
        """_require_field strips whitespace from value."""
        data = {"name": "  test_value  "}

        value, err = AuditRequestParser._require_field(data, "name")

        assert value == "test_value"

    def test_require_field_validates_with_validator(self):
        """_require_field uses validator when provided."""
        data = {"name": "invalid!"}

        def validator(v):
            return False, "Invalid name"

        value, err = AuditRequestParser._require_field(data, "name", validator)

        assert value is None
        assert err is not None
        assert err.status_code == 400

    def test_parse_int_returns_default(self):
        """_parse_int returns default when field missing."""
        data = {}

        value, err = AuditRequestParser._parse_int(data, "count", 5, 10)

        assert value == 5
        assert err is None

    def test_parse_int_clamps_to_max(self):
        """_parse_int clamps value to max."""
        data = {"count": 100}

        value, err = AuditRequestParser._parse_int(data, "count", 5, 10)

        assert value == 10
        assert err is None

    def test_parse_int_returns_error_for_invalid(self):
        """_parse_int returns error for non-integer."""
        data = {"count": "not_a_number"}

        value, err = AuditRequestParser._parse_int(data, "count", 5, 10)

        assert err is not None
        assert err.status_code == 400

    def test_parse_capability_probe_valid(self):
        """parse_capability_probe parses valid request."""
        mock_handler = MagicMock()

        def mock_read(h):
            return {
                "agent_name": "test_agent",
                "probe_types": ["contradiction"],
                "probes_per_type": 5,
            }

        with patch(
            "aragora.server.handlers.auditing.validate_agent_name",
            return_value=(True, None),
        ):
            parsed, err = AuditRequestParser.parse_capability_probe(mock_handler, mock_read)

        assert err is None
        assert parsed["agent_name"] == "test_agent"
        assert parsed["probes_per_type"] == 5

    def test_parse_capability_probe_uses_defaults(self):
        """parse_capability_probe uses defaults when not specified."""
        mock_handler = MagicMock()

        def mock_read(h):
            return {"agent_name": "test_agent"}

        with patch(
            "aragora.server.handlers.auditing.validate_agent_name",
            return_value=(True, None),
        ):
            parsed, err = AuditRequestParser.parse_capability_probe(mock_handler, mock_read)

        assert err is None
        assert parsed["probes_per_type"] == 3
        assert "contradiction" in parsed["probe_types"]
        assert parsed["model_type"] == "anthropic-api"

    def test_parse_deep_audit_valid(self):
        """parse_deep_audit parses valid request."""
        mock_handler = MagicMock()

        def mock_read(h):
            return {
                "task": "Analyze this decision",
                "context": "Additional context",
                "agent_names": ["agent1", "agent2"],
            }

        parsed, err = AuditRequestParser.parse_deep_audit(mock_handler, mock_read)

        assert err is None
        assert parsed["task"] == "Analyze this decision"
        assert parsed["context"] == "Additional context"

    def test_parse_deep_audit_requires_task(self):
        """parse_deep_audit requires task field."""
        mock_handler = MagicMock()

        def mock_read(h):
            return {"context": "Some context"}

        parsed, err = AuditRequestParser.parse_deep_audit(mock_handler, mock_read)

        assert parsed is None
        assert err is not None
        assert err.status_code == 400


class TestAuditAgentFactory:
    """Tests for AuditAgentFactory utility class."""

    def test_create_single_agent_unavailable(self):
        """create_single_agent returns error when debate unavailable."""
        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", False):
            agent, err = AuditAgentFactory.create_single_agent("anthropic-api", "test_agent")

        assert agent is None
        assert err is not None
        assert err.status_code == 503

    def test_create_single_agent_success(self):
        """create_single_agent creates agent successfully."""
        mock_agent = MagicMock()
        mock_create = MagicMock(return_value=mock_agent)

        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True):
            with patch("aragora.server.handlers.auditing.create_agent", mock_create):
                agent, err = AuditAgentFactory.create_single_agent("anthropic-api", "test_agent")

        assert err is None
        assert agent is mock_agent
        mock_create.assert_called_once_with("anthropic-api", name="test_agent", role="proposer")

    def test_create_single_agent_handles_exception(self):
        """create_single_agent handles creation exception."""
        mock_create = MagicMock(side_effect=Exception("Creation failed"))

        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True):
            with patch("aragora.server.handlers.auditing.create_agent", mock_create):
                agent, err = AuditAgentFactory.create_single_agent("anthropic-api", "test_agent")

        assert agent is None
        assert err is not None
        assert err.status_code == 400

    def test_create_multiple_agents_unavailable(self):
        """create_multiple_agents returns error when debate unavailable."""
        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", False):
            agents, err = AuditAgentFactory.create_multiple_agents(
                "anthropic-api", ["a1", "a2"], ["default1", "default2"]
            )

        assert agents == []
        assert err is not None
        assert err.status_code == 503

    def test_create_multiple_agents_uses_defaults(self):
        """create_multiple_agents uses defaults when agent_names empty."""
        mock_agent = MagicMock()
        mock_create = MagicMock(return_value=mock_agent)

        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True):
            with patch("aragora.server.handlers.auditing.create_agent", mock_create):
                with patch(
                    "aragora.server.handlers.auditing.validate_id",
                    return_value=(True, None),
                ):
                    agents, err = AuditAgentFactory.create_multiple_agents(
                        "anthropic-api", [], ["default1", "default2"]
                    )

        assert err is None
        assert len(agents) == 2

    def test_create_multiple_agents_requires_minimum(self):
        """create_multiple_agents requires at least 2 agents."""
        mock_create = MagicMock(side_effect=Exception("Fail"))

        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True):
            with patch("aragora.server.handlers.auditing.create_agent", mock_create):
                with patch(
                    "aragora.server.handlers.auditing.validate_id",
                    return_value=(True, None),
                ):
                    agents, err = AuditAgentFactory.create_multiple_agents(
                        "anthropic-api", ["a1"], ["default1"]
                    )

        assert agents == []
        assert err is not None
        assert err.status_code == 400


class TestAuditResultRecorder:
    """Tests for AuditResultRecorder utility class."""

    def test_record_probe_elo_skips_when_no_system(self):
        """record_probe_elo does nothing when elo_system is None."""
        mock_report = MagicMock()
        mock_report.probes_run = 10

        # Should not raise
        AuditResultRecorder.record_probe_elo(None, "test_agent", mock_report, "report_123")

    def test_record_probe_elo_skips_when_no_probes(self):
        """record_probe_elo does nothing when no probes run."""
        mock_elo = MagicMock()
        mock_report = MagicMock()
        mock_report.probes_run = 0

        AuditResultRecorder.record_probe_elo(mock_elo, "test_agent", mock_report, "report_123")

        mock_elo.record_redteam_result.assert_not_called()

    def test_record_probe_elo_records_result(self):
        """record_probe_elo records result to ELO system."""
        mock_elo = MagicMock()
        mock_report = MagicMock()
        mock_report.probes_run = 10
        mock_report.vulnerability_rate = 0.2
        mock_report.vulnerabilities_found = 2
        mock_report.critical_count = 1

        AuditResultRecorder.record_probe_elo(mock_elo, "test_agent", mock_report, "report_123")

        mock_elo.record_redteam_result.assert_called_once()
        call_kwargs = mock_elo.record_redteam_result.call_args[1]
        assert call_kwargs["agent_name"] == "test_agent"
        assert call_kwargs["robustness_score"] == 0.8  # 1.0 - 0.2
        assert call_kwargs["successful_attacks"] == 2
        assert call_kwargs["total_attacks"] == 10

    def test_calculate_audit_elo_adjustments_empty_when_no_system(self):
        """calculate_audit_elo_adjustments returns empty when no elo_system."""
        mock_verdict = MagicMock()

        result = AuditResultRecorder.calculate_audit_elo_adjustments(mock_verdict, None)

        assert result == {}

    def test_calculate_audit_elo_adjustments_processes_findings(self):
        """calculate_audit_elo_adjustments calculates adjustments from findings."""
        mock_finding1 = MagicMock()
        mock_finding1.agents_agree = ["agent1", "agent2"]
        mock_finding1.agents_disagree = ["agent3"]

        mock_finding2 = MagicMock()
        mock_finding2.agents_agree = ["agent1"]
        mock_finding2.agents_disagree = []

        mock_verdict = MagicMock()
        mock_verdict.findings = [mock_finding1, mock_finding2]

        mock_elo = MagicMock()

        result = AuditResultRecorder.calculate_audit_elo_adjustments(mock_verdict, mock_elo)

        assert result["agent1"] == 4  # +2 +2
        assert result["agent2"] == 2  # +2
        assert result["agent3"] == -1  # -1

    def test_save_probe_report_skips_when_no_dir(self):
        """save_probe_report does nothing when nomic_dir is None."""
        mock_report = MagicMock()

        # Should not raise
        AuditResultRecorder.save_probe_report(None, "test_agent", mock_report)


class TestAuditingHandlerAttackTypes:
    """Tests for attack types endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return AuditingHandler(mock_server_context)

    def test_get_attack_types_unavailable(self, handler):
        """Attack types returns 503 when redteam unavailable."""
        with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", False):
            result = handler._get_attack_types()

        assert result is not None
        assert result.status_code == 503

    def test_get_attack_types_returns_result(self, handler):
        """Attack types returns some result when available."""
        # This test verifies the endpoint works without mocking the complex enum
        # The actual response depends on whether redteam module is available
        result = handler._get_attack_types()

        # Should return a valid response (either success or 503)
        assert result is not None
        assert result.status_code in (200, 503)
