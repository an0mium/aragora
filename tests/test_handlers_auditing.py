"""
Tests for the auditing and security analysis endpoint handlers.

Covers:
- AuditRequestParser: JSON body parsing and validation
- AuditAgentFactory: Agent creation and validation
- AuditResultRecorder: ELO recording and report saving
- AuditingHandler: Main handler routing and endpoints
"""

from __future__ import annotations

import io
import json
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.server.handlers.auditing import (
    AuditRequestParser,
    AuditAgentFactory,
    AuditResultRecorder,
    AuditingHandler,
)
from aragora.server.handlers.base import json_response, error_response


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def server_context():
    """Create a basic server context."""
    return {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "nomic_dir": Path(tempfile.mkdtemp()),
    }


@pytest.fixture
def mock_handler():
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.headers = {"Content-Type": "application/json"}
    return handler


@pytest.fixture
def mock_handler_with_body():
    """Create a mock handler with JSON body."""

    def create(body: dict):
        handler = MagicMock()
        body_bytes = json.dumps(body).encode()
        handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        handler.rfile = io.BytesIO(body_bytes)
        return handler

    return create


@pytest.fixture
def auditing_handler(server_context):
    """Create an AuditingHandler instance."""
    return AuditingHandler(server_context)


@pytest.fixture
def mock_admin_user():
    """Create a mock authenticated admin user with admin:audit permission."""
    from dataclasses import dataclass

    @dataclass
    class MockUserContext:
        user_id: str = "user-123"
        org_id: str = "org-456"
        is_authenticated: bool = True
        role: str = "admin"
        permissions: list = None

        def __post_init__(self):
            if self.permissions is None:
                self.permissions = ["admin:audit", "debates:read", "debates:write"]

        def has_permission(self, perm: str) -> bool:
            """Check if user has a permission."""
            if self.role == "admin":
                return True  # Admins have all permissions
            return perm in self.permissions

    return MockUserContext()


@pytest.fixture
def auth_bypass():
    """Fixture to bypass authentication for testing protected endpoints."""

    @dataclass
    class MockUserContext:
        user_id: str = "admin-user"
        org_id: str = "org-123"
        is_authenticated: bool = True
        role: str = "admin"

        def has_permission(self, perm: str) -> bool:
            return True

    def bypass_extractor(handler, user_store=None):
        return MockUserContext()

    # Patch at the location where it's imported
    return patch("aragora.billing.jwt_auth.extract_user_from_request", bypass_extractor)


@dataclass
class MockProbeReport:
    """Mock probe report for testing."""

    report_id: str = "probe-report-abc123"
    probes_run: int = 10
    vulnerabilities_found: int = 2
    vulnerability_rate: float = 0.2
    elo_penalty: float = 5.0
    critical_count: int = 0
    high_count: int = 1
    medium_count: int = 1
    low_count: int = 0
    recommendations: list = None
    created_at: str = "2026-01-12T00:00:00"
    by_type: dict = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = ["Improve input validation"]
        if self.by_type is None:
            self.by_type = {}

    def to_dict(self):
        return {
            "report_id": self.report_id,
            "probes_run": self.probes_run,
            "vulnerabilities_found": self.vulnerabilities_found,
        }


@dataclass
class MockFinding:
    """Mock audit finding."""

    category: str = "security"
    summary: str = "Test finding"
    details: str = "Detailed description"
    agents_agree: list = None
    agents_disagree: list = None
    confidence: float = 0.8
    severity: float = 0.6
    citations: list = None

    def __post_init__(self):
        if self.agents_agree is None:
            self.agents_agree = ["agent-1"]
        if self.agents_disagree is None:
            self.agents_disagree = []
        if self.citations is None:
            self.citations = []


@dataclass
class MockVerdict:
    """Mock audit verdict."""

    recommendation: str = "Proceed with caution"
    confidence: float = 0.75
    unanimous_issues: list = None
    split_opinions: list = None
    risk_areas: list = None
    findings: list = None
    cross_examination_notes: str = "Notes"
    citations: list = None

    def __post_init__(self):
        if self.unanimous_issues is None:
            self.unanimous_issues = ["Issue 1"]
        if self.split_opinions is None:
            self.split_opinions = []
        if self.risk_areas is None:
            self.risk_areas = ["Area 1"]
        if self.findings is None:
            self.findings = [MockFinding()]
        if self.citations is None:
            self.citations = []


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str = "test-agent"
    role: str = "proposer"

    def generate(self, prompt: str) -> str:
        return f"Response to: {prompt}"


@dataclass
class MockUserContext:
    """Mock user authentication context."""

    is_authenticated: bool = True
    user_id: str = "user_123"
    org_id: str = "org_456"
    email: str = "admin@example.com"
    role: str = "admin"


# =============================================================================
# AuditRequestParser Tests
# =============================================================================


class TestAuditRequestParser:
    """Tests for AuditRequestParser utility class."""

    def test_read_json_success(self):
        """Test successful JSON reading."""

        def read_json_fn(h):
            return {"key": "value"}

        data, err = AuditRequestParser._read_json(None, read_json_fn)
        assert data == {"key": "value"}
        assert err is None

    def test_read_json_failure(self):
        """Test JSON reading failure."""

        def read_json_fn(h):
            return None

        data, err = AuditRequestParser._read_json(None, read_json_fn)
        assert data is None
        assert err.status_code == 400

    def test_require_field_success(self):
        """Test required field extraction."""
        data = {"name": "test-value"}
        value, err = AuditRequestParser._require_field(data, "name")
        assert value == "test-value"
        assert err is None

    def test_require_field_missing(self):
        """Test missing required field."""
        data = {}
        value, err = AuditRequestParser._require_field(data, "name")
        assert value is None
        assert err.status_code == 400
        assert "Missing required field" in json.loads(err.body)["error"]

    def test_require_field_empty(self):
        """Test empty required field."""
        data = {"name": "   "}
        value, err = AuditRequestParser._require_field(data, "name")
        assert value is None
        assert err.status_code == 400

    def test_require_field_with_validator(self):
        """Test required field with validation."""

        def validator(val):
            if len(val) < 3:
                return False, "Too short"
            return True, None

        data = {"name": "ab"}
        value, err = AuditRequestParser._require_field(data, "name", validator)
        assert value is None
        assert err.status_code == 400

    def test_parse_int_success(self):
        """Test integer parsing."""
        data = {"count": "5"}
        value, err = AuditRequestParser._parse_int(data, "count", 3, 10)
        assert value == 5
        assert err is None

    def test_parse_int_default(self):
        """Test integer parsing with default."""
        data = {}
        value, err = AuditRequestParser._parse_int(data, "count", 3, 10)
        assert value == 3
        assert err is None

    def test_parse_int_clamped(self):
        """Test integer parsing clamped to max."""
        data = {"count": "100"}
        value, err = AuditRequestParser._parse_int(data, "count", 3, 10)
        assert value == 10
        assert err is None

    def test_parse_int_invalid(self):
        """Test invalid integer parsing."""
        data = {"count": "not-a-number"}
        value, err = AuditRequestParser._parse_int(data, "count", 3, 10)
        assert err.status_code == 400
        assert "must be an integer" in json.loads(err.body)["error"]

    def test_parse_capability_probe_valid(self):
        """Test parsing valid capability probe request."""

        def read_json_fn(h):
            return {
                "agent_name": "claude-test",
                "probe_types": ["contradiction", "hallucination"],
                "probes_per_type": 5,
            }

        result, err = AuditRequestParser.parse_capability_probe(None, read_json_fn)
        assert err is None
        assert result["agent_name"] == "claude-test"
        assert result["probe_types"] == ["contradiction", "hallucination"]
        assert result["probes_per_type"] == 5
        assert result["model_type"] == "anthropic-api"

    def test_parse_capability_probe_missing_agent(self):
        """Test parsing probe request without agent name."""

        def read_json_fn(h):
            return {"probe_types": ["contradiction"]}

        result, err = AuditRequestParser.parse_capability_probe(None, read_json_fn)
        assert result is None
        assert err.status_code == 400

    def test_parse_capability_probe_defaults(self):
        """Test parsing probe request with defaults."""

        def read_json_fn(h):
            return {"agent_name": "test-agent"}

        result, err = AuditRequestParser.parse_capability_probe(None, read_json_fn)
        assert err is None
        assert result["probes_per_type"] == 3
        assert "contradiction" in result["probe_types"]

    def test_parse_deep_audit_valid(self):
        """Test parsing valid deep audit request."""

        def read_json_fn(h):
            return {
                "task": "Review this code for security issues",
                "context": "def process(input): return eval(input)",
                "config": {
                    "rounds": 4,
                    "risk_threshold": 0.8,
                },
            }

        result, err = AuditRequestParser.parse_deep_audit(None, read_json_fn)
        assert err is None
        assert result["task"] == "Review this code for security issues"
        assert result["rounds"] == 4
        assert result["risk_threshold"] == 0.8

    def test_parse_deep_audit_missing_task(self):
        """Test parsing audit request without task."""

        def read_json_fn(h):
            return {"context": "some context"}

        result, err = AuditRequestParser.parse_deep_audit(None, read_json_fn)
        assert result is None
        assert err.status_code == 400

    def test_parse_deep_audit_invalid_risk_threshold(self):
        """Test parsing audit with invalid risk threshold."""

        def read_json_fn(h):
            return {"task": "Test task", "config": {"risk_threshold": "invalid"}}

        result, err = AuditRequestParser.parse_deep_audit(None, read_json_fn)
        assert result is None
        assert err.status_code == 400
        assert "risk_threshold" in json.loads(err.body)["error"]


# =============================================================================
# AuditAgentFactory Tests
# =============================================================================


class TestAuditAgentFactory:
    """Tests for AuditAgentFactory utility class."""

    def test_create_single_agent_not_available(self):
        """Test agent creation when debate module unavailable."""
        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", False):
            agent, err = AuditAgentFactory.create_single_agent("anthropic-api", "test-agent")
        assert agent is None
        assert err.status_code == 503

    def test_create_single_agent_success(self):
        """Test successful single agent creation."""
        mock_agent = MockAgent()
        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True):
            with patch("aragora.server.handlers.auditing.create_agent") as mock_create:
                mock_create.return_value = mock_agent
                agent, err = AuditAgentFactory.create_single_agent("anthropic-api", "test-agent")

        assert err is None
        assert agent == mock_agent
        mock_create.assert_called_once_with("anthropic-api", name="test-agent", role="proposer")

    def test_create_single_agent_failure(self):
        """Test agent creation failure."""
        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True):
            with patch("aragora.server.handlers.auditing.create_agent") as mock_create:
                mock_create.side_effect = ValueError("Invalid agent")
                agent, err = AuditAgentFactory.create_single_agent("anthropic-api", "test-agent")

        assert agent is None
        assert err.status_code == 400

    def test_create_multiple_agents_not_available(self):
        """Test multiple agent creation when unavailable."""
        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", False):
            agents, err = AuditAgentFactory.create_multiple_agents(
                "anthropic-api", ["agent1", "agent2"], ["default1", "default2"]
            )
        assert agents == []
        assert err.status_code == 503

    def test_create_multiple_agents_success(self):
        """Test successful multiple agent creation."""
        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True):
            with patch("aragora.server.handlers.auditing.create_agent") as mock_create:
                mock_create.side_effect = [
                    MockAgent(name="agent1"),
                    MockAgent(name="agent2"),
                ]
                agents, err = AuditAgentFactory.create_multiple_agents(
                    "anthropic-api", ["agent1", "agent2"], ["default1"]
                )

        assert err is None
        assert len(agents) == 2

    def test_create_multiple_agents_uses_defaults(self):
        """Test multiple agents uses defaults when no names provided."""
        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True):
            with patch("aragora.server.handlers.auditing.create_agent") as mock_create:
                mock_create.side_effect = [
                    MockAgent(name="default1"),
                    MockAgent(name="default2"),
                ]
                agents, err = AuditAgentFactory.create_multiple_agents(
                    "anthropic-api", [], ["default1", "default2"]
                )

        assert err is None
        assert len(agents) == 2

    def test_create_multiple_agents_minimum_two(self):
        """Test requires at least 2 agents."""
        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True):
            with patch("aragora.server.handlers.auditing.create_agent") as mock_create:
                mock_create.side_effect = [MockAgent()]
                # Only one valid agent creation succeeds
                agents, err = AuditAgentFactory.create_multiple_agents(
                    "anthropic-api", ["agent1"], ["default1"]
                )

        assert agents == []
        assert err.status_code == 400
        assert "at least 2 agents" in json.loads(err.body)["error"]

    def test_create_multiple_agents_max_limit(self):
        """Test agents limited to max count."""
        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", True):
            with patch("aragora.server.handlers.auditing.create_agent") as mock_create:
                mock_create.side_effect = [MockAgent(name=f"agent{i}") for i in range(10)]
                agents, err = AuditAgentFactory.create_multiple_agents(
                    "anthropic-api", [f"agent{i}" for i in range(10)], [], max_agents=5
                )

        assert err is None
        assert len(agents) == 5


# =============================================================================
# AuditResultRecorder Tests
# =============================================================================


class TestAuditResultRecorder:
    """Tests for AuditResultRecorder utility class."""

    def test_record_probe_elo_success(self):
        """Test recording probe ELO results."""
        elo_system = MagicMock()
        report = MockProbeReport()

        AuditResultRecorder.record_probe_elo(elo_system, "test-agent", report, "report-123")

        elo_system.record_redteam_result.assert_called_once()
        call_args = elo_system.record_redteam_result.call_args
        assert call_args.kwargs["agent_name"] == "test-agent"
        assert call_args.kwargs["robustness_score"] == 0.8  # 1.0 - 0.2

    def test_record_probe_elo_no_system(self):
        """Test recording with no ELO system."""
        report = MockProbeReport()
        # Should not raise
        AuditResultRecorder.record_probe_elo(None, "test-agent", report, "report-123")

    def test_record_probe_elo_no_probes(self):
        """Test recording with no probes run."""
        elo_system = MagicMock()
        report = MockProbeReport(probes_run=0)

        AuditResultRecorder.record_probe_elo(elo_system, "test-agent", report, "report-123")

        elo_system.record_redteam_result.assert_not_called()

    def test_calculate_audit_elo_adjustments(self):
        """Test calculating ELO adjustments from findings."""
        elo_system = MagicMock()
        verdict = MockVerdict(
            findings=[
                MockFinding(agents_agree=["agent1", "agent2"], agents_disagree=["agent3"]),
                MockFinding(agents_agree=["agent1"], agents_disagree=["agent2"]),
            ]
        )

        adjustments = AuditResultRecorder.calculate_audit_elo_adjustments(verdict, elo_system)

        assert adjustments["agent1"] == 4  # +2 + 2
        assert adjustments["agent2"] == 1  # +2 - 1
        assert adjustments["agent3"] == -1

    def test_calculate_audit_elo_no_system(self):
        """Test calculation with no ELO system."""
        verdict = MockVerdict()
        adjustments = AuditResultRecorder.calculate_audit_elo_adjustments(verdict, None)
        assert adjustments == {}

    def test_save_probe_report(self):
        """Test saving probe report to storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            report = MockProbeReport()

            AuditResultRecorder.save_probe_report(nomic_dir, "test-agent", report)

            probes_dir = nomic_dir / "probes" / "test-agent"
            assert probes_dir.exists()
            files = list(probes_dir.glob("*.json"))
            assert len(files) == 1

    def test_save_probe_report_no_dir(self):
        """Test saving with no nomic dir."""
        report = MockProbeReport()
        # Should not raise
        AuditResultRecorder.save_probe_report(None, "test-agent", report)

    def test_save_audit_report(self):
        """Test saving audit report to storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nomic_dir = Path(tmpdir)
            verdict = MockVerdict()
            config = MagicMock()
            config.rounds = 6
            config.enable_research = True
            config.cross_examination_depth = 3
            config.risk_threshold = 0.7
            agents = [MockAgent(name="agent1"), MockAgent(name="agent2")]

            AuditResultRecorder.save_audit_report(
                nomic_dir,
                "audit-123",
                "Test task",
                "Test context",
                agents,
                verdict,
                config,
                1500.0,
                {"agent1": 2},
            )

            audits_dir = nomic_dir / "audits"
            assert audits_dir.exists()
            files = list(audits_dir.glob("*.json"))
            assert len(files) == 1


# =============================================================================
# AuditingHandler Tests
# =============================================================================


class TestAuditingHandler:
    """Tests for AuditingHandler class."""

    # === Route Handling ===

    def test_can_handle_capability_probe(self, auditing_handler):
        """Test can_handle for capability probe."""
        assert auditing_handler.can_handle("/api/debates/capability-probe") is True

    def test_can_handle_deep_audit(self, auditing_handler):
        """Test can_handle for deep audit."""
        assert auditing_handler.can_handle("/api/debates/deep-audit") is True

    def test_can_handle_attack_types(self, auditing_handler):
        """Test can_handle for attack types."""
        assert auditing_handler.can_handle("/api/redteam/attack-types") is True

    def test_can_handle_red_team(self, auditing_handler):
        """Test can_handle for red team analysis."""
        assert auditing_handler.can_handle("/api/debates/abc123/red-team") is True

    def test_can_handle_unrelated_path(self, auditing_handler):
        """Test can_handle returns False for unrelated paths."""
        assert auditing_handler.can_handle("/api/debates") is False
        assert auditing_handler.can_handle("/api/other") is False

    def test_handle_routes_capability_probe(self, auditing_handler, mock_handler):
        """Test handle routes to capability probe."""
        with patch.object(auditing_handler, "_run_capability_probe") as mock_method:
            mock_method.return_value = json_response({"success": True})
            result = auditing_handler.handle("/api/debates/capability-probe", {}, mock_handler)

        mock_method.assert_called_once_with(mock_handler)

    def test_handle_routes_deep_audit(self, auditing_handler, mock_handler):
        """Test handle routes to deep audit."""
        with patch.object(auditing_handler, "_run_deep_audit") as mock_method:
            mock_method.return_value = json_response({"success": True})
            result = auditing_handler.handle("/api/debates/deep-audit", {}, mock_handler)

        mock_method.assert_called_once_with(mock_handler)

    def test_handle_routes_attack_types(self, auditing_handler, mock_handler):
        """Test handle routes to attack types."""
        with patch.object(auditing_handler, "_get_attack_types") as mock_method:
            mock_method.return_value = json_response({"attack_types": []})
            result = auditing_handler.handle("/api/redteam/attack-types", {}, mock_handler)

        mock_method.assert_called_once()

    def test_handle_routes_red_team(self, auditing_handler, mock_handler):
        """Test handle routes to red team analysis."""
        with patch.object(auditing_handler, "_run_red_team_analysis") as mock_method:
            mock_method.return_value = json_response({"success": True})
            result = auditing_handler.handle("/api/debates/debate123/red-team", {}, mock_handler)

        mock_method.assert_called_once_with("debate123", mock_handler)

    def test_handle_invalid_debate_id_red_team(self, auditing_handler, mock_handler):
        """Test handle rejects invalid debate ID for red team."""
        result = auditing_handler.handle("/api/debates/../etc/passwd/red-team", {}, mock_handler)
        assert result.status_code == 400

    def test_handle_unhandled_path(self, auditing_handler, mock_handler):
        """Test handle returns None for unhandled paths."""
        result = auditing_handler.handle("/api/other", {}, mock_handler)
        assert result is None

    # === Attack Types Endpoint ===

    def test_get_attack_types_not_available(self, auditing_handler):
        """Test attack types when module unavailable."""
        with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", False):
            result = auditing_handler._get_attack_types()

        assert result.status_code == 503

    def test_get_attack_types_success(self, auditing_handler):
        """Test getting attack types successfully."""
        from enum import Enum

        # Include all attack types that _get_attack_category checks for
        class MockAttackType(Enum):
            LOGICAL_FALLACY = "logical_fallacy"
            UNSTATED_ASSUMPTION = "unstated_assumption"
            COUNTEREXAMPLE = "counterexample"
            SECURITY = "security"
            RESOURCE_EXHAUSTION = "resource_exhaustion"
            RACE_CONDITION = "race_condition"
            DEPENDENCY_FAILURE = "dependency_failure"
            EDGE_CASE = "edge_case"

        mock_module = MagicMock()
        mock_module.AttackType = MockAttackType

        with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True):
            with patch.dict("sys.modules", {"aragora.modes.redteam": mock_module}):
                result = auditing_handler._get_attack_types()

        assert result.status_code == 200
        parsed = json.loads(result.body)
        assert "attack_types" in parsed
        assert parsed["count"] == 8

    # === Capability Probe Endpoint ===

    def test_run_capability_probe_not_available(self, auditing_handler, mock_handler, auth_bypass):
        """Test capability probe when prober unavailable."""
        with auth_bypass:
            with patch("aragora.server.handlers.auditing.PROBER_AVAILABLE", False):
                result = auditing_handler._run_capability_probe(mock_handler)

        assert result.status_code == 503

    def test_run_capability_probe_invalid_json(self, auditing_handler, mock_handler, auth_bypass):
        """Test capability probe with invalid JSON."""
        mock_handler.rfile = io.BytesIO(b"not json")
        mock_handler.headers = {"Content-Length": "8"}

        with auth_bypass:
            with patch("aragora.server.handlers.auditing.PROBER_AVAILABLE", True):
                result = auditing_handler._run_capability_probe(mock_handler)

        assert result.status_code == 400

    # === Deep Audit Endpoint ===

    def test_run_deep_audit_module_not_available(self, auditing_handler, mock_handler, auth_bypass):
        """Test deep audit when module unavailable."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "aragora.modes.deep_audit":
                raise ImportError("No module named 'aragora.modes.deep_audit'")
            return original_import(name, *args, **kwargs)

        with auth_bypass:
            with patch.object(builtins, "__import__", side_effect=mock_import):
                result = auditing_handler._run_deep_audit(mock_handler)

        # Should be 503 when module unavailable
        assert result.status_code == 503

    # === Red Team Analysis Endpoint ===

    def test_run_red_team_not_available(self, auditing_handler, mock_handler, auth_bypass):
        """Test red team when module unavailable."""
        with auth_bypass:
            with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", False):
                result = auditing_handler._run_red_team_analysis("debate123", mock_handler)

        assert result.status_code == 503

    def test_run_red_team_no_storage(self, server_context, mock_handler, auth_bypass):
        """Test red team with no storage."""
        server_context["storage"] = None
        handler = AuditingHandler(server_context)

        with auth_bypass:
            with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True):
                result = handler._run_red_team_analysis("debate123", mock_handler)

        assert result.status_code == 500

    def test_run_red_team_debate_not_found(self, auditing_handler, mock_handler, auth_bypass):
        """Test red team with non-existent debate."""
        auditing_handler.ctx["storage"].get_by_slug.return_value = None
        auditing_handler.ctx["storage"].get_by_id.return_value = None

        with auth_bypass:
            with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True):
                result = auditing_handler._run_red_team_analysis("nonexistent", mock_handler)

        assert result.status_code == 404

    def test_run_red_team_success(self, auditing_handler, mock_handler_with_body, auth_bypass):
        """Test successful red team analysis."""
        mock_handler = mock_handler_with_body(
            {
                "attack_types": ["logical_fallacy"],
                "max_rounds": 2,
            }
        )

        from enum import Enum

        class MockAttackType(Enum):
            LOGICAL_FALLACY = "logical_fallacy"

        auditing_handler.ctx["storage"].get_by_slug.return_value = {
            "id": "debate123",
            "task": "Test task",
            "consensus_answer": "The solution is X",
        }

        mock_redteam_module = MagicMock()
        mock_redteam_module.AttackType = MockAttackType

        with auth_bypass:
            with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True):
                with patch.dict("sys.modules", {"aragora.modes.redteam": mock_redteam_module}):
                    result = auditing_handler._run_red_team_analysis("debate123", mock_handler)

        assert result.status_code == 200
        parsed = json.loads(result.body)
        assert "session_id" in parsed
        assert "findings" in parsed
        assert parsed["status"] == "analysis_complete"

    # === Proposal Analysis ===

    def test_analyze_proposal_for_redteam(self, auditing_handler):
        """Test proposal vulnerability analysis."""
        from enum import Enum

        class MockAttackType(Enum):
            LOGICAL_FALLACY = "logical_fallacy"
            SECURITY = "security"

        proposal = "This solution is always the best and most secure option"

        with patch("aragora.modes.redteam.AttackType", MockAttackType, create=True):
            findings = auditing_handler._analyze_proposal_for_redteam(
                proposal, ["logical_fallacy", "security"], {}
            )

        assert len(findings) == 2
        # Should find matches for "always" and "best" (logical_fallacy)
        # and "secure" (security)
        fallacy_finding = next((f for f in findings if f["attack_type"] == "logical_fallacy"), None)
        assert fallacy_finding is not None
        assert fallacy_finding["keyword_matches"] > 0

    def test_analyze_proposal_no_matches(self, auditing_handler):
        """Test analysis with no keyword matches."""
        from enum import Enum

        class MockAttackType(Enum):
            LOGICAL_FALLACY = "logical_fallacy"

        proposal = "Simple text without any special keywords"

        with patch("aragora.modes.redteam.AttackType", MockAttackType, create=True):
            findings = auditing_handler._analyze_proposal_for_redteam(
                proposal, ["logical_fallacy"], {}
            )

        assert len(findings) == 1
        assert findings[0]["keyword_matches"] == 0
        assert findings[0]["requires_manual_review"] is False

    def test_analyze_proposal_invalid_attack_type(self, auditing_handler):
        """Test analysis skips invalid attack types."""
        from enum import Enum

        class MockAttackType(Enum):
            LOGICAL_FALLACY = "logical_fallacy"

        with patch("aragora.modes.redteam.AttackType", MockAttackType, create=True):
            findings = auditing_handler._analyze_proposal_for_redteam("test", ["invalid_type"], {})

        assert len(findings) == 0

    # === Audit Config ===

    def test_get_audit_config_strategy(self, auditing_handler):
        """Test getting strategy preset config."""
        strategy_preset = MagicMock()
        result = auditing_handler._get_audit_config(
            "strategy", {}, MagicMock, strategy_preset, None, None
        )
        assert result == strategy_preset

    def test_get_audit_config_contract(self, auditing_handler):
        """Test getting contract preset config."""
        contract_preset = MagicMock()
        result = auditing_handler._get_audit_config(
            "contract", {}, MagicMock, None, contract_preset, None
        )
        assert result == contract_preset

    def test_get_audit_config_code_architecture(self, auditing_handler):
        """Test getting code architecture preset config."""
        code_preset = MagicMock()
        result = auditing_handler._get_audit_config(
            "code_architecture", {}, MagicMock, None, None, code_preset
        )
        assert result == code_preset

    def test_get_audit_config_custom(self, auditing_handler):
        """Test getting custom config from parsed values."""
        parsed = {
            "rounds": 8,
            "enable_research": False,
            "cross_examination_depth": 5,
            "risk_threshold": 0.9,
        }

        config_class = MagicMock()
        result = auditing_handler._get_audit_config(
            "custom", parsed, config_class, None, None, None
        )

        config_class.assert_called_once_with(
            rounds=8,
            enable_research=False,
            cross_examination_depth=5,
            risk_threshold=0.9,
        )

    # === Transform Probe Results ===

    def test_transform_probe_results(self, auditing_handler):
        """Test transforming probe results for API response."""
        by_type = {
            "contradiction": [
                {
                    "probe_id": "p1",
                    "probe_type": "contradiction",
                    "vulnerability_found": True,
                    "severity": "HIGH",
                    "vulnerability_description": "Found issue",
                    "evidence": "Details here",
                    "response_time_ms": 150,
                }
            ]
        }

        result = auditing_handler._transform_probe_results(by_type)

        assert "contradiction" in result
        assert len(result["contradiction"]) == 1
        probe = result["contradiction"][0]
        assert probe["passed"] is False
        assert probe["severity"] == "high"
        assert probe["response_time_ms"] == 150

    def test_transform_probe_results_passed(self, auditing_handler):
        """Test transforming passed probe result."""
        by_type = {
            "hallucination": [
                {
                    "probe_id": "p2",
                    "vulnerability_found": False,
                }
            ]
        }

        result = auditing_handler._transform_probe_results(by_type)
        assert result["hallucination"][0]["passed"] is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestAuditingHandlerIntegration:
    """Integration tests for AuditingHandler."""

    def test_full_red_team_flow(self, server_context, mock_handler_with_body, auth_bypass):
        """Test complete red team analysis flow."""
        from enum import Enum

        class MockAttackType(Enum):
            EDGE_CASE = "edge_case"
            SECURITY = "security"

        handler = AuditingHandler(server_context)
        mock_http = mock_handler_with_body(
            {
                "attack_types": ["edge_case", "security"],
            }
        )

        server_context["storage"].get_by_slug.return_value = {
            "id": "test-debate",
            "task": "Design a user authentication system",
            "consensus_answer": "Use JWT tokens with secure encryption",
        }

        mock_redteam_module = MagicMock()
        mock_redteam_module.AttackType = MockAttackType

        with auth_bypass:
            with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True):
                with patch.dict("sys.modules", {"aragora.modes.redteam": mock_redteam_module}):
                    result = handler.handle("/api/debates/test-debate/red-team", {}, mock_http)

        assert result.status_code == 200
        parsed = json.loads(result.body)

        # Verify response structure
        assert "session_id" in parsed
        assert parsed["debate_id"] == "test-debate"
        assert "findings" in parsed
        assert "robustness_score" in parsed
        assert 0 <= parsed["robustness_score"] <= 1

        # Should find security-related keywords
        security_finding = next(
            (f for f in parsed["findings"] if f["attack_type"] == "security"), None
        )
        assert security_finding is not None
        # "secure" and "encrypt" should be matched
        assert security_finding["keyword_matches"] >= 1
