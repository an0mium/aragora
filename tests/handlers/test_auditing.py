"""Tests for auditing handler endpoints.

Tests the auditing API endpoints including:
- Attack types listing
- Capability probing
- Deep audit
- Red team analysis
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.auditing import (
    AuditAgentFactory,
    AuditingHandler,
    AuditRequestParser,
    AuditResultRecorder,
)


def parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


class MockUser:
    """Mock user object for auditing tests."""

    def __init__(
        self,
        id: str,
        email: str,
        name: str = "Test User",
        role: str = "admin",
        org_id: Optional[str] = "org_1",
    ):
        self.id = id
        self.user_id = id
        self.email = email
        self.name = name
        self.role = role
        self.org_id = org_id


class MockAuthContext:
    """Mock authentication context."""

    def __init__(
        self,
        user_id: str,
        is_authenticated: bool = True,
        org_id: Optional[str] = None,
        role: str = "admin",
        error_reason: Optional[str] = None,
    ):
        self.user_id = user_id
        self.is_authenticated = is_authenticated
        self.org_id = org_id
        self.role = role
        self.error_reason = error_reason


class MockHandler:
    """Mock HTTP handler."""

    def __init__(
        self,
        body: Optional[dict] = None,
        command: str = "POST",
        user_store=None,
    ):
        self.command = command
        self.headers = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)
        self.path = ""
        self.user_store = user_store

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str, role: str = "proposer"):
        self.name = name
        self.role = role

    async def generate(self, prompt: str) -> str:
        return f"Response from {self.name}"


class MockProbeResult:
    """Mock probe result."""

    def __init__(self, probe_id: str, passed: bool = True, severity: str = "low"):
        self.probe_id = probe_id
        self.vulnerability_found = not passed
        self.severity = severity

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "vulnerability_found": self.vulnerability_found,
            "severity": self.severity,
        }


class MockProbeReport:
    """Mock probe report."""

    def __init__(
        self,
        probes_run: int = 10,
        vulnerabilities_found: int = 2,
    ):
        self.report_id = "test-report-123"
        self.probes_run = probes_run
        self.vulnerabilities_found = vulnerabilities_found
        self.vulnerability_rate = vulnerabilities_found / probes_run if probes_run > 0 else 0
        self.elo_penalty = self.vulnerability_rate * 10
        self.critical_count = 0
        self.high_count = 1
        self.medium_count = 1
        self.low_count = 0
        self.recommendations = ["Review high severity findings"]
        self.created_at = datetime.now().isoformat()
        self.by_type = {
            "contradiction": [MockProbeResult("p1", True), MockProbeResult("p2", False)],
        }

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "probes_run": self.probes_run,
            "vulnerabilities_found": self.vulnerabilities_found,
        }


class MockAuditFinding:
    """Mock audit finding."""

    def __init__(
        self,
        category: str = "security",
        summary: str = "Test finding",
        severity: float = 0.5,
    ):
        self.category = category
        self.summary = summary
        self.details = "Detailed description of finding"
        self.agents_agree = ["Claude-Analyst"]
        self.agents_disagree = ["Claude-Skeptic"]
        self.confidence = 0.8
        self.severity = severity
        self.citations = []


class MockAuditVerdict:
    """Mock audit verdict."""

    def __init__(self):
        self.recommendation = "Proceed with caution"
        self.confidence = 0.75
        self.unanimous_issues = ["Issue 1"]
        self.split_opinions = ["Opinion 1"]
        self.risk_areas = ["Risk 1"]
        self.findings = [MockAuditFinding()]
        self.cross_examination_notes = "Cross examination notes"
        self.citations = ["Citation 1"]


class MockStorage:
    """Mock storage for debate data."""

    def __init__(self):
        self._debates: Dict[str, dict] = {}

    def add_debate(self, debate_id: str, data: dict):
        self._debates[debate_id] = data

    def get_by_slug(self, slug: str) -> Optional[dict]:
        return self._debates.get(slug)

    def get_by_id(self, id: str) -> Optional[dict]:
        return self._debates.get(id)


class MockEloSystem:
    """Mock ELO system for testing."""

    def __init__(self):
        self.recorded_results = []

    def record_redteam_result(self, **kwargs):
        self.recorded_results.append(kwargs)


@pytest.fixture
def storage():
    """Create mock storage with test data."""
    store = MockStorage()
    store.add_debate(
        "test-debate-1",
        {
            "id": "test-debate-1",
            "task": "Test debate task",
            "consensus_answer": "Test consensus",
            "final_answer": "Test answer",
        },
    )
    return store


@pytest.fixture
def auditing_handler(storage):
    """Create auditing handler with mock context."""
    ctx = {
        "storage": storage,
        "elo_system": MockEloSystem(),
    }
    return AuditingHandler(ctx)


class TestAuditingHandlerRouting:
    """Tests for auditing handler routing."""

    def test_can_handle_capability_probe(self, auditing_handler):
        """Test can_handle for capability probe endpoint."""
        assert auditing_handler.can_handle("/api/v1/debates/capability-probe")

    def test_can_handle_deep_audit(self, auditing_handler):
        """Test can_handle for deep audit endpoint."""
        assert auditing_handler.can_handle("/api/v1/debates/deep-audit")

    def test_can_handle_attack_types(self, auditing_handler):
        """Test can_handle for attack types endpoint."""
        assert auditing_handler.can_handle("/api/v1/redteam/attack-types")

    def test_can_handle_red_team(self, auditing_handler):
        """Test can_handle for red team analysis endpoint."""
        assert auditing_handler.can_handle("/api/v1/debates/test-id/red-team")
        assert auditing_handler.can_handle("/api/v1/debates/abc-123/red-team")

    def test_cannot_handle_non_audit_paths(self, auditing_handler):
        """Test can_handle rejects non-auditing paths."""
        assert not auditing_handler.can_handle("/api/v1/debates")
        assert not auditing_handler.can_handle("/api/v1/users")
        assert not auditing_handler.can_handle("/api/v1/billing/plans")


class TestAuditRequestParser:
    """Tests for audit request parsing."""

    def test_parse_capability_probe_success(self):
        """Test successful capability probe parsing."""
        mock_handler = MockHandler(
            body={
                "agent_name": "test-agent",
                "probe_types": ["contradiction", "hallucination"],
                "probes_per_type": 5,
                "model_type": "anthropic-api",
            }
        )

        def read_json(handler):
            data = handler.rfile.read()
            return json.loads(data)

        result, error = AuditRequestParser.parse_capability_probe(mock_handler, read_json)

        assert error is None
        assert result["agent_name"] == "test-agent"
        assert result["probes_per_type"] == 5
        assert "contradiction" in result["probe_types"]

    def test_parse_capability_probe_missing_agent_name(self):
        """Test capability probe parsing fails without agent name."""
        mock_handler = MockHandler(body={"probe_types": ["contradiction"]})

        def read_json(handler):
            data = handler.rfile.read()
            return json.loads(data)

        result, error = AuditRequestParser.parse_capability_probe(mock_handler, read_json)

        assert result is None
        assert error is not None
        assert error.status_code == 400

    def test_parse_capability_probe_clamps_probes_per_type(self):
        """Test probes_per_type is clamped to max value."""
        mock_handler = MockHandler(
            body={
                "agent_name": "test-agent",
                "probes_per_type": 100,  # Above max
            }
        )

        def read_json(handler):
            data = handler.rfile.read()
            return json.loads(data)

        result, error = AuditRequestParser.parse_capability_probe(mock_handler, read_json)

        assert error is None
        assert result["probes_per_type"] == 10  # Clamped to max

    def test_parse_deep_audit_success(self):
        """Test successful deep audit parsing."""
        mock_handler = MockHandler(
            body={
                "task": "Analyze this security policy",
                "context": "Additional context here",
                "agent_names": ["Agent1", "Agent2"],
                "config": {
                    "rounds": 5,
                    "risk_threshold": 0.8,
                },
            }
        )

        def read_json(handler):
            data = handler.rfile.read()
            return json.loads(data)

        result, error = AuditRequestParser.parse_deep_audit(mock_handler, read_json)

        assert error is None
        assert result["task"] == "Analyze this security policy"
        assert result["rounds"] == 5
        assert result["risk_threshold"] == 0.8

    def test_parse_deep_audit_missing_task(self):
        """Test deep audit parsing fails without task."""
        mock_handler = MockHandler(body={"context": "Some context"})

        def read_json(handler):
            data = handler.rfile.read()
            return json.loads(data)

        result, error = AuditRequestParser.parse_deep_audit(mock_handler, read_json)

        assert result is None
        assert error is not None
        assert error.status_code == 400

    def test_parse_deep_audit_invalid_risk_threshold(self):
        """Test deep audit parsing fails with invalid risk_threshold."""
        mock_handler = MockHandler(
            body={
                "task": "Test task",
                "config": {"risk_threshold": "invalid"},
            }
        )

        def read_json(handler):
            data = handler.rfile.read()
            return json.loads(data)

        result, error = AuditRequestParser.parse_deep_audit(mock_handler, read_json)

        assert result is None
        assert error is not None
        assert error.status_code == 400


class TestAuditAgentFactory:
    """Tests for audit agent factory."""

    def test_create_single_agent_unavailable(self):
        """Test agent creation when debate module unavailable."""
        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", False):
            agent, error = AuditAgentFactory.create_single_agent("anthropic-api", "test-agent")

            assert agent is None
            assert error is not None
            assert error.status_code == 503

    def test_create_multiple_agents_unavailable(self):
        """Test multiple agent creation when debate module unavailable."""
        with patch("aragora.server.handlers.auditing.DEBATE_AVAILABLE", False):
            agents, error = AuditAgentFactory.create_multiple_agents(
                "anthropic-api", ["Agent1", "Agent2"], ["Default1", "Default2"]
            )

            assert agents == []
            assert error is not None
            assert error.status_code == 503


class TestAuditResultRecorder:
    """Tests for audit result recording."""

    def test_record_probe_elo_with_valid_report(self):
        """Test ELO recording with valid probe report."""
        elo_system = MockEloSystem()
        report = MockProbeReport(probes_run=10, vulnerabilities_found=2)

        AuditResultRecorder.record_probe_elo(elo_system, "test-agent", report, "report-123")

        assert len(elo_system.recorded_results) == 1
        result = elo_system.recorded_results[0]
        assert result["agent_name"] == "test-agent"
        assert result["robustness_score"] == 0.8  # 1 - 0.2 vulnerability rate

    def test_record_probe_elo_no_probes_run(self):
        """Test ELO recording is skipped when no probes run."""
        elo_system = MockEloSystem()
        report = MockProbeReport(probes_run=0, vulnerabilities_found=0)

        AuditResultRecorder.record_probe_elo(elo_system, "test-agent", report, "report-123")

        assert len(elo_system.recorded_results) == 0

    def test_record_probe_elo_no_elo_system(self):
        """Test ELO recording handles missing ELO system."""
        report = MockProbeReport(probes_run=10, vulnerabilities_found=2)

        # Should not raise
        AuditResultRecorder.record_probe_elo(None, "test-agent", report, "report-123")

    def test_calculate_audit_elo_adjustments(self):
        """Test ELO adjustment calculation from findings."""
        elo_system = MockEloSystem()
        verdict = MockAuditVerdict()

        adjustments = AuditResultRecorder.calculate_audit_elo_adjustments(verdict, elo_system)

        assert "Claude-Analyst" in adjustments
        assert adjustments["Claude-Analyst"] == 2  # Agreeing agents get +2
        assert "Claude-Skeptic" in adjustments
        assert adjustments["Claude-Skeptic"] == -1  # Disagreeing agents get -1


class TestGetAttackTypes:
    """Tests for attack types endpoint."""

    def test_get_attack_types_success(self, auditing_handler):
        """Test successful attack types retrieval."""
        mock_handler = MockHandler(command="GET")

        with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True):
            from aragora.modes.redteam import AttackType

            result = auditing_handler.handle("/api/redteam/attack-types", {}, mock_handler)
            body = parse_body(result)

            assert result.status_code == 200
            assert "attack_types" in body
            assert "count" in body
            assert body["count"] > 0

    def test_get_attack_types_unavailable(self, auditing_handler):
        """Test attack types returns 503 when module unavailable."""
        mock_handler = MockHandler(command="GET")

        with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", False):
            result = auditing_handler.handle("/api/redteam/attack-types", {}, mock_handler)

            assert result.status_code == 503


class TestCapabilityProbe:
    """Tests for capability probe endpoint."""

    def make_auth_context(self, user_id: str = "admin_1", role: str = "admin"):
        """Create an auth context."""
        return MockAuthContext(
            user_id=user_id,
            is_authenticated=True,
            org_id="org_1",
            role=role,
        )

    def test_capability_probe_requires_permission(self, auditing_handler):
        """Test capability probe requires admin:audit permission."""
        mock_handler = MockHandler(body={"agent_name": "test-agent"})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            # Member role lacks admin:audit permission
            mock_extract.return_value = self.make_auth_context("user_1", "member")

            result = auditing_handler.handle("/api/debates/capability-probe", {}, mock_handler)

            assert result.status_code == 403

    def test_capability_probe_prober_unavailable(self, auditing_handler):
        """Test capability probe returns 503 when prober unavailable."""
        mock_handler = MockHandler(body={"agent_name": "test-agent"})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.auditing.PROBER_AVAILABLE", False):
                mock_extract.return_value = self.make_auth_context()

                result = auditing_handler.handle("/api/debates/capability-probe", {}, mock_handler)

                assert result.status_code == 503
                assert "not available" in parse_body(result)["error"]

    def test_capability_probe_invalid_json(self, auditing_handler):
        """Test capability probe rejects invalid JSON."""
        mock_handler = MockHandler()
        mock_handler.rfile.read.return_value = b"not valid json"

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.auditing.PROBER_AVAILABLE", True):
                mock_extract.return_value = self.make_auth_context()

                result = auditing_handler.handle("/api/debates/capability-probe", {}, mock_handler)

                assert result.status_code == 400

    def test_capability_probe_success(self, auditing_handler):
        """Test successful capability probe execution."""
        mock_handler = MockHandler(
            body={
                "agent_name": "test-agent",
                "probe_types": ["contradiction"],
                "probes_per_type": 2,
            }
        )

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.auditing.PROBER_AVAILABLE", True):
                with patch(
                    "aragora.server.handlers.auditing.AuditAgentFactory.create_single_agent"
                ) as mock_create:
                    with patch("aragora.server.handlers.auditing.run_async") as mock_run:
                        from aragora.modes.prober import ProbeType

                        mock_extract.return_value = self.make_auth_context()
                        mock_create.return_value = (MockAgent("test-agent"), None)
                        mock_run.return_value = MockProbeReport()

                        result = auditing_handler.handle(
                            "/api/debates/capability-probe", {}, mock_handler
                        )
                        body = parse_body(result)

                        assert result.status_code == 200
                        assert "report_id" in body
                        assert "target_agent" in body
                        assert "probes_run" in body
                        assert "summary" in body


class TestDeepAudit:
    """Tests for deep audit endpoint."""

    def make_auth_context(self, user_id: str = "admin_1", role: str = "admin"):
        """Create an auth context."""
        return MockAuthContext(
            user_id=user_id,
            is_authenticated=True,
            org_id="org_1",
            role=role,
        )

    def test_deep_audit_requires_permission(self, auditing_handler):
        """Test deep audit requires admin:audit permission."""
        mock_handler = MockHandler(body={"task": "Test task"})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = self.make_auth_context("user_1", "member")

            result = auditing_handler.handle("/api/debates/deep-audit", {}, mock_handler)

            assert result.status_code == 403

    def test_deep_audit_module_unavailable(self, auditing_handler):
        """Test deep audit returns 503 when module unavailable."""
        mock_handler = MockHandler(body={"task": "Test task"})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            with patch.dict("sys.modules", {"aragora.modes.deep_audit": None}):
                mock_extract.return_value = self.make_auth_context()

                # This will trigger ImportError
                result = auditing_handler.handle("/api/debates/deep-audit", {}, mock_handler)

                assert result.status_code == 503

    def test_deep_audit_missing_task(self, auditing_handler):
        """Test deep audit rejects missing task."""
        mock_handler = MockHandler(body={"context": "Some context"})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = self.make_auth_context()

            # Mock the module import
            with patch(
                "aragora.server.handlers.auditing.AuditingHandler._run_deep_audit"
            ) as mock_run:
                # Set up the mock to simulate the validation failure
                from aragora.server.handlers.base import error_response

                mock_run.return_value = error_response("Missing required field: task", 400)

                result = auditing_handler.handle("/api/debates/deep-audit", {}, mock_handler)

                assert result.status_code == 400


class TestRedTeamAnalysis:
    """Tests for red team analysis endpoint."""

    def make_auth_context(self, user_id: str = "admin_1", role: str = "admin"):
        """Create an auth context."""
        return MockAuthContext(
            user_id=user_id,
            is_authenticated=True,
            org_id="org_1",
            role=role,
        )

    def test_red_team_requires_permission(self, auditing_handler):
        """Test red team analysis requires admin:audit permission."""
        mock_handler = MockHandler(body={})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = self.make_auth_context("user_1", "member")

            result = auditing_handler.handle(
                "/api/debates/test-debate-1/red-team", {}, mock_handler
            )

            assert result.status_code == 403

    def test_red_team_module_unavailable(self, auditing_handler):
        """Test red team returns 503 when module unavailable."""
        mock_handler = MockHandler(body={})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", False):
                mock_extract.return_value = self.make_auth_context()

                result = auditing_handler.handle(
                    "/api/debates/test-debate-1/red-team", {}, mock_handler
                )

                assert result.status_code == 503

    def test_red_team_debate_not_found(self, auditing_handler):
        """Test red team returns 404 for non-existent debate."""
        mock_handler = MockHandler(body={})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True):
                mock_extract.return_value = self.make_auth_context()

                result = auditing_handler.handle(
                    "/api/debates/nonexistent-debate/red-team", {}, mock_handler
                )

                assert result.status_code == 404

    def test_red_team_no_storage(self, auditing_handler):
        """Test red team returns 500 when storage not configured."""
        # Create handler without storage
        handler_no_storage = AuditingHandler({})
        mock_handler = MockHandler(body={})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True):
                mock_extract.return_value = self.make_auth_context()

                result = handler_no_storage.handle(
                    "/api/debates/test-debate-1/red-team", {}, mock_handler
                )

                assert result.status_code == 500
                assert "Storage" in parse_body(result)["error"]

    def test_red_team_success(self, auditing_handler):
        """Test successful red team analysis."""
        mock_handler = MockHandler(
            body={
                "attack_types": ["logical_fallacy", "security"],
                "max_rounds": 3,
            }
        )

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True):
                mock_extract.return_value = self.make_auth_context()

                result = auditing_handler.handle(
                    "/api/debates/test-debate-1/red-team", {}, mock_handler
                )
                body = parse_body(result)

                assert result.status_code == 200
                assert "session_id" in body
                assert "debate_id" in body
                assert "findings" in body
                assert "robustness_score" in body
                assert body["debate_id"] == "test-debate-1"

    def test_red_team_clamps_max_rounds(self, auditing_handler):
        """Test max_rounds is clamped to maximum value."""
        mock_handler = MockHandler(
            body={
                "max_rounds": 100,  # Above max of 5
            }
        )

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True):
                mock_extract.return_value = self.make_auth_context()

                result = auditing_handler.handle(
                    "/api/debates/test-debate-1/red-team", {}, mock_handler
                )
                body = parse_body(result)

                assert result.status_code == 200
                assert body["max_rounds"] == 5  # Clamped to max

    def test_red_team_default_attack_types(self, auditing_handler):
        """Test red team uses default attack types when not specified."""
        mock_handler = MockHandler(body={})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True):
                mock_extract.return_value = self.make_auth_context()

                result = auditing_handler.handle(
                    "/api/debates/test-debate-1/red-team", {}, mock_handler
                )
                body = parse_body(result)

                assert result.status_code == 200
                assert "attack_types" in body
                assert len(body["attack_types"]) > 0


class TestInvalidDebateId:
    """Tests for invalid debate ID handling."""

    def make_auth_context(self, user_id: str = "admin_1", role: str = "admin"):
        """Create an auth context."""
        return MockAuthContext(
            user_id=user_id,
            is_authenticated=True,
            org_id="org_1",
            role=role,
        )

    def test_red_team_invalid_debate_id(self, auditing_handler):
        """Test red team rejects invalid debate ID patterns."""
        mock_handler = MockHandler(body={})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True):
                mock_extract.return_value = self.make_auth_context()

                # Use invalid characters in debate ID
                result = auditing_handler.handle(
                    "/api/debates/<script>alert(1)</script>/red-team",
                    {},
                    mock_handler,
                )

                assert result.status_code == 400


class TestUnauthorized:
    """Tests for unauthenticated access."""

    def test_capability_probe_unauthenticated(self, auditing_handler):
        """Test capability probe rejects unauthenticated requests."""
        mock_handler = MockHandler(body={"agent_name": "test"})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = MockAuthContext("", is_authenticated=False)

            with patch("aragora.server.handlers.auditing.PROBER_AVAILABLE", True):
                result = auditing_handler.handle("/api/debates/capability-probe", {}, mock_handler)

                assert result.status_code == 401

    def test_deep_audit_unauthenticated(self, auditing_handler):
        """Test deep audit rejects unauthenticated requests."""
        mock_handler = MockHandler(body={"task": "Test"})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = MockAuthContext("", is_authenticated=False)

            result = auditing_handler.handle("/api/debates/deep-audit", {}, mock_handler)

            assert result.status_code == 401

    def test_red_team_unauthenticated(self, auditing_handler):
        """Test red team rejects unauthenticated requests."""
        mock_handler = MockHandler(body={})

        with patch("aragora.billing.jwt_auth.extract_user_from_request") as mock_extract:
            mock_extract.return_value = MockAuthContext("", is_authenticated=False)

            with patch("aragora.server.handlers.auditing.REDTEAM_AVAILABLE", True):
                result = auditing_handler.handle(
                    "/api/debates/test-debate-1/red-team", {}, mock_handler
                )

                assert result.status_code == 401
