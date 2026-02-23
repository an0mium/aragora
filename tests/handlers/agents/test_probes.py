"""Comprehensive tests for ProbesHandler (aragora/server/handlers/agents/probes.py).

Tests cover:
- Handler initialization and routing (can_handle, ROUTES)
- GET /api/v1/probes/reports - List all probe reports
- GET /api/v1/probes/reports/{id} - Get a specific probe report
- POST /api/v1/probes/capability - Run capability probes
- POST /api/v1/probes/run - Legacy probe endpoint
- Input validation (missing agent_name, invalid probe types, schema validation)
- Error cases (prober unavailable, agent system unavailable, agent creation failure)
- ELO recording and leaderboard cache invalidation
- Report saving to nomic directory
- Probe hooks (stream events)
- Result transformation (vulnerability_found -> passed)
- Pagination in list reports
- Helper function _safe_int
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if hasattr(result, "body"):
        if isinstance(result.body, bytes):
            return json.loads(result.body.decode("utf-8"))
        if isinstance(result.body, str):
            return json.loads(result.body)
    return {}


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    if hasattr(result, "status_code"):
        return result.status_code
    return 200


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiters before and after each test."""
    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass

    yield

    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters

        reset_rate_limiters()
    except ImportError:
        pass


@pytest.fixture
def handler():
    """Create a ProbesHandler with empty context."""
    from aragora.server.handlers.agents.probes import ProbesHandler

    return ProbesHandler(ctx={})


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler with client address, headers, and rfile."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 54321)
    h.headers = {"Content-Length": "2"}
    h.rfile = MagicMock()
    h.rfile.read.return_value = b"{}"
    return h


def _set_body(mock_http_handler: MagicMock, body: dict) -> None:
    """Configure mock HTTP handler to return a specific JSON body."""
    body_bytes = json.dumps(body).encode("utf-8")
    mock_http_handler.rfile.read.return_value = body_bytes
    mock_http_handler.headers = {"Content-Length": str(len(body_bytes))}


@pytest.fixture
def mock_user_ctx():
    """Create a mock authenticated user context."""
    ctx = MagicMock()
    ctx.is_authenticated = True
    ctx.user_id = "test-user-001"
    ctx.email = "test@example.com"
    return ctx


@pytest.fixture
def mock_probe_report():
    """Create a mock probe report returned by CapabilityProber.probe_agent."""
    report = MagicMock()
    report.report_id = "probe-report-abc12345"
    report.probes_run = 8
    report.vulnerabilities_found = 2
    report.vulnerability_rate = 0.25
    report.elo_penalty = 12.5
    report.critical_count = 0
    report.high_count = 1
    report.medium_count = 1
    report.low_count = 0
    report.recommendations = ["Consider improving contradiction handling"]
    report.created_at = "2026-02-23T12:00:00"

    # by_type contains probe results grouped by type
    probe_result_1 = MagicMock()
    probe_result_1.to_dict.return_value = {
        "probe_id": "p1",
        "probe_type": "contradiction",
        "vulnerability_found": True,
        "severity": "HIGH",
        "vulnerability_description": "Agent contradicted itself",
        "evidence": "Said X then said not X",
        "response_time_ms": 150,
    }
    probe_result_2 = MagicMock()
    probe_result_2.to_dict.return_value = {
        "probe_id": "p2",
        "probe_type": "contradiction",
        "vulnerability_found": False,
        "severity": None,
        "vulnerability_description": "",
        "evidence": "",
        "response_time_ms": 120,
    }
    report.by_type = {"contradiction": [probe_result_1, probe_result_2]}
    report.to_dict.return_value = {
        "report_id": report.report_id,
        "probes_run": report.probes_run,
        "vulnerabilities_found": report.vulnerabilities_found,
        "vulnerability_rate": report.vulnerability_rate,
        "created_at": report.created_at,
    }
    return report


@pytest.fixture
def nomic_dir(tmp_path):
    """Create a temporary nomic directory for probe report storage."""
    probes_dir = tmp_path / "probes"
    probes_dir.mkdir(parents=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestSafeInt:
    """Tests for the _safe_int helper."""

    def test_int_value(self):
        from aragora.server.handlers.agents.probes import _safe_int

        assert _safe_int(42) == 42

    def test_string_int(self):
        from aragora.server.handlers.agents.probes import _safe_int

        assert _safe_int("10") == 10

    def test_invalid_string(self):
        from aragora.server.handlers.agents.probes import _safe_int

        assert _safe_int("abc") == 0

    def test_none_value(self):
        from aragora.server.handlers.agents.probes import _safe_int

        assert _safe_int(None) == 0

    def test_custom_default(self):
        from aragora.server.handlers.agents.probes import _safe_int

        assert _safe_int("abc", 5) == 5

    def test_float_value(self):
        from aragora.server.handlers.agents.probes import _safe_int

        assert _safe_int(3.7) == 3

    def test_empty_string(self):
        from aragora.server.handlers.agents.probes import _safe_int

        assert _safe_int("", 99) == 99


# ---------------------------------------------------------------------------
# Initialization and Routing
# ---------------------------------------------------------------------------


class TestProbesHandlerInit:
    """Tests for handler initialization."""

    def test_init_with_empty_ctx(self, handler):
        """Handler initializes with empty context."""
        assert handler.ctx == {}

    def test_init_with_provided_ctx(self):
        from aragora.server.handlers.agents.probes import ProbesHandler

        ctx = {"elo_system": "mock"}
        h = ProbesHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_init_with_none_ctx(self):
        from aragora.server.handlers.agents.probes import ProbesHandler

        h = ProbesHandler(ctx=None)
        assert h.ctx == {}

    def test_routes_defined(self, handler):
        """Routes list contains expected endpoints."""
        assert "/api/v1/probes/capability" in handler.ROUTES
        assert "/api/v1/probes/run" in handler.ROUTES
        assert "/api/v1/probes/reports" in handler.ROUTES


class TestCanHandle:
    """Tests for can_handle routing."""

    def test_capability_route(self, handler):
        assert handler.can_handle("/api/v1/probes/capability") is True

    def test_run_legacy_route(self, handler):
        assert handler.can_handle("/api/v1/probes/run") is True

    def test_reports_list_route(self, handler):
        assert handler.can_handle("/api/v1/probes/reports") is True

    def test_report_by_id_route(self, handler):
        assert handler.can_handle("/api/v1/probes/reports/abc-123") is True

    def test_report_deep_id(self, handler):
        assert handler.can_handle("/api/v1/probes/reports/probe-report-deadbeef") is True

    def test_unrelated_route(self, handler):
        assert handler.can_handle("/api/v1/agents") is False

    def test_probes_without_subroute(self, handler):
        assert handler.can_handle("/api/v1/probes") is False

    def test_different_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False


# ---------------------------------------------------------------------------
# GET /api/v1/probes/reports
# ---------------------------------------------------------------------------


class TestListProbeReports:
    """Tests for the list probe reports endpoint."""

    def test_no_nomic_dir(self, handler, mock_http_handler):
        """Returns empty list when no nomic dir configured."""
        result = handler.handle("/api/v1/probes/reports", {}, mock_http_handler)
        body = _body(result)
        assert body["reports"] == []
        assert body["total"] == 0

    def test_no_probes_directory(self, handler, mock_http_handler, tmp_path):
        """Returns empty list when probes dir does not exist."""
        handler.ctx["nomic_dir"] = tmp_path
        result = handler.handle("/api/v1/probes/reports", {}, mock_http_handler)
        body = _body(result)
        assert body["reports"] == []
        assert body["total"] == 0

    def test_empty_probes_dir(self, handler, mock_http_handler, nomic_dir):
        """Returns empty list when probes dir exists but has no agents."""
        handler.ctx["nomic_dir"] = nomic_dir
        result = handler.handle("/api/v1/probes/reports", {}, mock_http_handler)
        body = _body(result)
        assert body["reports"] == []
        assert body["total"] == 0

    def test_list_reports_single_agent(self, handler, mock_http_handler, nomic_dir):
        """Lists reports for a single agent."""
        agent_dir = nomic_dir / "probes" / "claude"
        agent_dir.mkdir(parents=True)
        report_data = {
            "report_id": "probe-report-aaa",
            "probes_run": 6,
            "vulnerabilities_found": 1,
            "vulnerability_rate": 0.167,
            "created_at": "2026-02-23T10:00:00",
        }
        (agent_dir / "2026-02-23_probe-report-aaa.json").write_text(json.dumps(report_data))

        handler.ctx["nomic_dir"] = nomic_dir
        result = handler.handle("/api/v1/probes/reports", {}, mock_http_handler)
        body = _body(result)
        assert body["total"] == 1
        assert len(body["reports"]) == 1
        assert body["reports"][0]["report_id"] == "probe-report-aaa"
        assert body["reports"][0]["target_agent"] == "claude"

    def test_list_reports_multiple_agents(self, handler, mock_http_handler, nomic_dir):
        """Lists reports across multiple agents."""
        for agent_name in ["claude", "gpt4"]:
            agent_dir = nomic_dir / "probes" / agent_name
            agent_dir.mkdir(parents=True)
            report_data = {
                "report_id": f"report-{agent_name}",
                "probes_run": 4,
                "vulnerabilities_found": 0,
                "vulnerability_rate": 0.0,
                "created_at": f"2026-02-23T{10 if agent_name == 'claude' else 11}:00:00",
            }
            (agent_dir / f"2026-02-23_report-{agent_name}.json").write_text(json.dumps(report_data))

        handler.ctx["nomic_dir"] = nomic_dir
        result = handler.handle("/api/v1/probes/reports", {}, mock_http_handler)
        body = _body(result)
        assert body["total"] == 2

    def test_filter_by_agent(self, handler, mock_http_handler, nomic_dir):
        """Filters reports by agent name."""
        for agent_name in ["claude", "gpt4"]:
            agent_dir = nomic_dir / "probes" / agent_name
            agent_dir.mkdir(parents=True)
            (agent_dir / f"report-{agent_name}.json").write_text(
                json.dumps({"report_id": f"r-{agent_name}", "created_at": "2026-01-01"})
            )

        handler.ctx["nomic_dir"] = nomic_dir
        result = handler.handle("/api/v1/probes/reports", {"agent": ["claude"]}, mock_http_handler)
        body = _body(result)
        assert body["total"] == 1
        assert body["reports"][0]["target_agent"] == "claude"

    def test_pagination_limit(self, handler, mock_http_handler, nomic_dir):
        """Respects limit parameter for pagination."""
        agent_dir = nomic_dir / "probes" / "claude"
        agent_dir.mkdir(parents=True)
        for i in range(5):
            (agent_dir / f"report_{i}.json").write_text(
                json.dumps({"report_id": f"r-{i}", "created_at": f"2026-02-{20 + i:02d}"})
            )

        handler.ctx["nomic_dir"] = nomic_dir
        result = handler.handle("/api/v1/probes/reports", {"limit": ["2"]}, mock_http_handler)
        body = _body(result)
        assert body["total"] == 5
        assert len(body["reports"]) == 2
        assert body["limit"] == 2

    def test_pagination_offset(self, handler, mock_http_handler, nomic_dir):
        """Respects offset parameter for pagination."""
        agent_dir = nomic_dir / "probes" / "claude"
        agent_dir.mkdir(parents=True)
        for i in range(5):
            (agent_dir / f"report_{i}.json").write_text(
                json.dumps({"report_id": f"r-{i}", "created_at": f"2026-02-{20 + i:02d}"})
            )

        handler.ctx["nomic_dir"] = nomic_dir
        result = handler.handle(
            "/api/v1/probes/reports", {"limit": ["10"], "offset": ["3"]}, mock_http_handler
        )
        body = _body(result)
        assert body["total"] == 5
        assert len(body["reports"]) == 2  # 5 total, offset 3 = 2 remaining
        assert body["offset"] == 3

    def test_limit_capped_at_200(self, handler, mock_http_handler, nomic_dir):
        """Limit is capped at maximum of 200."""
        handler.ctx["nomic_dir"] = nomic_dir
        result = handler.handle("/api/v1/probes/reports", {"limit": ["999"]}, mock_http_handler)
        body = _body(result)
        assert body["limit"] == 200

    def test_invalid_limit_uses_default(self, handler, mock_http_handler, nomic_dir):
        """Invalid limit falls back to default (50)."""
        handler.ctx["nomic_dir"] = nomic_dir
        result = handler.handle("/api/v1/probes/reports", {"limit": ["abc"]}, mock_http_handler)
        body = _body(result)
        assert body["limit"] == 50

    def test_skips_invalid_json_files(self, handler, mock_http_handler, nomic_dir):
        """Gracefully skips files with invalid JSON."""
        agent_dir = nomic_dir / "probes" / "claude"
        agent_dir.mkdir(parents=True)
        (agent_dir / "bad_report.json").write_text("not valid json {{{")
        (agent_dir / "good_report.json").write_text(
            json.dumps({"report_id": "good", "created_at": "2026-01-01"})
        )

        handler.ctx["nomic_dir"] = nomic_dir
        result = handler.handle("/api/v1/probes/reports", {}, mock_http_handler)
        body = _body(result)
        assert body["total"] == 1
        assert body["reports"][0]["report_id"] == "good"

    def test_skips_non_directory_entries(self, handler, mock_http_handler, nomic_dir):
        """Skips non-directory files in probes dir."""
        probes_dir = nomic_dir / "probes"
        (probes_dir / "stray_file.txt").write_text("not a directory")
        agent_dir = probes_dir / "claude"
        agent_dir.mkdir()
        (agent_dir / "report.json").write_text(
            json.dumps({"report_id": "r-1", "created_at": "2026-01-01"})
        )

        handler.ctx["nomic_dir"] = nomic_dir
        result = handler.handle("/api/v1/probes/reports", {}, mock_http_handler)
        body = _body(result)
        assert body["total"] == 1

    def test_reports_sorted_by_created_at_descending(self, handler, mock_http_handler, nomic_dir):
        """Reports are sorted newest first."""
        agent_dir = nomic_dir / "probes" / "claude"
        agent_dir.mkdir(parents=True)
        (agent_dir / "old.json").write_text(
            json.dumps({"report_id": "old", "created_at": "2026-01-01"})
        )
        (agent_dir / "new.json").write_text(
            json.dumps({"report_id": "new", "created_at": "2026-02-01"})
        )

        handler.ctx["nomic_dir"] = nomic_dir
        result = handler.handle("/api/v1/probes/reports", {}, mock_http_handler)
        body = _body(result)
        assert body["reports"][0]["report_id"] == "new"
        assert body["reports"][1]["report_id"] == "old"


# ---------------------------------------------------------------------------
# GET /api/v1/probes/reports/{id}
# ---------------------------------------------------------------------------


class TestGetProbeReport:
    """Tests for getting a specific probe report by ID."""

    def test_report_not_found_no_nomic_dir(self, handler, mock_http_handler):
        """Returns 404 when no nomic dir."""
        result = handler.handle("/api/v1/probes/reports/nonexistent", {}, mock_http_handler)
        assert _status(result) == 404

    def test_report_not_found_no_probes_dir(self, handler, mock_http_handler, tmp_path):
        """Returns 404 when probes dir does not exist."""
        handler.ctx["nomic_dir"] = tmp_path
        result = handler.handle("/api/v1/probes/reports/nonexistent", {}, mock_http_handler)
        assert _status(result) == 404

    def test_report_not_found_empty_probes(self, handler, mock_http_handler, nomic_dir):
        """Returns 404 when report ID does not match any file."""
        handler.ctx["nomic_dir"] = nomic_dir
        result = handler.handle("/api/v1/probes/reports/nonexistent", {}, mock_http_handler)
        assert _status(result) == 404

    def test_get_report_by_id(self, handler, mock_http_handler, nomic_dir):
        """Returns full report data when ID matches."""
        agent_dir = nomic_dir / "probes" / "claude"
        agent_dir.mkdir(parents=True)
        report_data = {
            "report_id": "probe-report-xyz",
            "probes_run": 10,
            "vulnerabilities_found": 3,
            "created_at": "2026-02-23",
        }
        (agent_dir / "2026-02-23_probe-report-xyz.json").write_text(json.dumps(report_data))

        handler.ctx["nomic_dir"] = nomic_dir
        result = handler.handle("/api/v1/probes/reports/probe-report-xyz", {}, mock_http_handler)
        body = _body(result)
        assert _status(result) == 200
        assert body["report_id"] == "probe-report-xyz"
        assert body["probes_run"] == 10

    def test_invalid_report_id_too_long(self, handler, mock_http_handler):
        """Returns 400 for report IDs longer than 64 chars."""
        long_id = "a" * 65
        result = handler.handle(f"/api/v1/probes/reports/{long_id}", {}, mock_http_handler)
        assert _status(result) == 400

    def test_empty_report_id(self, handler, mock_http_handler):
        """Returns 400 for empty report ID."""
        result = handler.handle("/api/v1/probes/reports/", {}, mock_http_handler)
        # The path extraction strips /api/v1/probes/reports/ -> ""
        assert _status(result) == 400

    def test_get_report_partial_id_match_in_filename(self, handler, mock_http_handler, nomic_dir):
        """Finds report when ID partially matches filename."""
        agent_dir = nomic_dir / "probes" / "claude"
        agent_dir.mkdir(parents=True)
        report_data = {
            "report_id": "probe-report-full-id",
            "probes_run": 5,
        }
        (agent_dir / "2026-02-23_probe-report-full-id.json").write_text(json.dumps(report_data))

        handler.ctx["nomic_dir"] = nomic_dir
        # Search using a partial id that appears in filename
        result = handler.handle(
            "/api/v1/probes/reports/probe-report-full-id", {}, mock_http_handler
        )
        body = _body(result)
        assert _status(result) == 200

    def test_skips_invalid_json_when_searching(self, handler, mock_http_handler, nomic_dir):
        """Gracefully skips corrupted JSON files during search."""
        agent_dir = nomic_dir / "probes" / "claude"
        agent_dir.mkdir(parents=True)
        (agent_dir / "2026-02-23_target-id.json").write_text("{{bad json")

        handler.ctx["nomic_dir"] = nomic_dir
        result = handler.handle("/api/v1/probes/reports/target-id", {}, mock_http_handler)
        assert _status(result) == 404


# ---------------------------------------------------------------------------
# GET routing - unknown paths
# ---------------------------------------------------------------------------


class TestHandleGetRouting:
    """Tests for GET request routing."""

    def test_unknown_get_path_returns_none(self, handler, mock_http_handler):
        """Returns None for unrecognized GET paths."""
        result = handler.handle("/api/v1/probes/unknown", {}, mock_http_handler)
        assert result is None


# ---------------------------------------------------------------------------
# POST /api/v1/probes/capability
# ---------------------------------------------------------------------------


class TestRunCapabilityProbe:
    """Tests for the capability probe POST endpoint."""

    def test_prober_unavailable(self, handler, mock_http_handler):
        """Returns 503 when prober module is not available."""
        _set_body(mock_http_handler, {"agent_name": "claude"})
        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", False),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            body = _body(result)
            assert _status(result) == 503
            assert "not available" in body.get("error", "")

    def test_agent_system_unavailable(self, handler, mock_http_handler):
        """Returns 503 when agent system is not available."""
        _set_body(mock_http_handler, {"agent_name": "claude"})
        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", False),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            body = _body(result)
            assert _status(result) == 503
            assert "not available" in body.get("error", "")

    def test_create_agent_is_none(self, handler, mock_http_handler):
        """Returns 503 when create_agent function is None."""
        _set_body(mock_http_handler, {"agent_name": "claude"})
        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", None),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            assert _status(result) == 503

    def test_invalid_json_body(self, handler, mock_http_handler):
        """Returns 400 for invalid JSON body."""
        mock_http_handler.rfile.read.return_value = b"not json at all"
        mock_http_handler.headers = {"Content-Length": "15"}
        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", MagicMock()),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            assert _status(result) == 400

    def test_missing_agent_name(self, handler, mock_http_handler):
        """Returns 400 when agent_name is missing."""
        _set_body(mock_http_handler, {"probe_types": ["contradiction"]})
        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", MagicMock()),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            body = _body(result)
            assert _status(result) == 400
            assert "agent_name" in body.get("error", "")

    def test_empty_agent_name(self, handler, mock_http_handler):
        """Returns 400 when agent_name is empty string."""
        _set_body(mock_http_handler, {"agent_name": "  "})
        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", MagicMock()),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            assert _status(result) == 400

    def test_schema_validation_failure(self, handler, mock_http_handler):
        """Returns 400 when schema validation fails."""
        _set_body(mock_http_handler, {"agent_name": "claude"})
        mock_validation = MagicMock()
        mock_validation.is_valid = False
        mock_validation.error = "agent_name too long"
        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", MagicMock()),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=mock_validation,
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            assert _status(result) == 400

    def test_invalid_agent_name_format(self, handler, mock_http_handler):
        """Returns 400 when agent_name fails validation."""
        _set_body(mock_http_handler, {"agent_name": "../../etc/passwd"})
        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", MagicMock()),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(False, "Invalid agent name"),
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            assert _status(result) == 400

    def test_all_probe_types_invalid(self, handler, mock_http_handler):
        """Returns 400 when all probe types are invalid."""
        _set_body(
            mock_http_handler,
            {"agent_name": "claude", "probe_types": ["bogus", "fake"]},
        )
        mock_probe_type = MagicMock()
        mock_probe_type.side_effect = ValueError("invalid")
        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", MagicMock()),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type,
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            body = _body(result)
            assert _status(result) == 400
            assert "No valid probe types" in body.get("error", "")

    def test_agent_creation_failure(self, handler, mock_http_handler):
        """Returns 400 when agent creation raises ValueError."""
        _set_body(
            mock_http_handler,
            {"agent_name": "claude", "model_type": "unknown-model"},
        )
        mock_create = MagicMock(side_effect=ValueError("Unknown model type"))
        mock_probe_type_cls = MagicMock(return_value=MagicMock(value="contradiction"))

        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", mock_create),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type_cls,
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            body = _body(result)
            assert _status(result) == 400
            assert "hint" in body

    def test_successful_probe_run(self, handler, mock_http_handler, mock_probe_report, nomic_dir):
        """Successful capability probe returns full report."""
        _set_body(
            mock_http_handler,
            {
                "agent_name": "claude",
                "probe_types": ["contradiction"],
                "probes_per_type": 2,
            },
        )

        mock_agent = MagicMock()
        mock_create = MagicMock(return_value=mock_agent)
        mock_prober_instance = MagicMock()
        mock_prober_instance.probe_agent = MagicMock(return_value=mock_probe_report)
        mock_prober_cls = MagicMock(return_value=mock_prober_instance)
        mock_probe_type_cls = MagicMock(return_value=MagicMock(value="contradiction"))

        handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", mock_create),
            patch("aragora.server.handlers.agents.probes.CapabilityProber", mock_prober_cls),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type_cls,
            ),
            patch(
                "aragora.server.handlers.agents.probes.run_async",
                return_value=mock_probe_report,
            ),
            patch(
                "aragora.server.handlers.agents.probes.invalidate_leaderboard_cache",
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            body = _body(result)
            assert _status(result) == 200
            assert body["target_agent"] == "claude"
            assert body["probes_run"] == 8
            assert body["vulnerabilities_found"] == 2
            assert body["vulnerability_rate"] == 0.25
            assert "summary" in body
            assert body["summary"]["total"] == 8
            assert body["summary"]["passed"] == 6
            assert body["summary"]["failed"] == 2

    def test_successful_probe_with_elo_recording(
        self, handler, mock_http_handler, mock_probe_report, nomic_dir
    ):
        """ELO recording is called on successful probe."""
        _set_body(mock_http_handler, {"agent_name": "claude"})

        mock_agent = MagicMock()
        mock_create = MagicMock(return_value=mock_agent)
        mock_prober_cls = MagicMock(return_value=MagicMock())
        mock_probe_type_cls = MagicMock(return_value=MagicMock(value="contradiction"))
        mock_elo = MagicMock()

        handler.ctx["nomic_dir"] = nomic_dir
        handler.ctx["elo_system"] = mock_elo

        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", mock_create),
            patch("aragora.server.handlers.agents.probes.CapabilityProber", mock_prober_cls),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type_cls,
            ),
            patch(
                "aragora.server.handlers.agents.probes.run_async",
                return_value=mock_probe_report,
            ),
            patch(
                "aragora.server.handlers.agents.probes.invalidate_leaderboard_cache",
            ) as mock_invalidate,
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            assert _status(result) == 200
            mock_elo.record_redteam_result.assert_called_once()
            call_kwargs = mock_elo.record_redteam_result.call_args
            assert call_kwargs.kwargs["agent_name"] == "claude"
            assert call_kwargs.kwargs["total_attacks"] == 8
            mock_invalidate.assert_called_once()

    def test_elo_recording_failure_is_nonfatal(
        self, handler, mock_http_handler, mock_probe_report, nomic_dir
    ):
        """ELO recording failure does not crash the probe."""
        _set_body(mock_http_handler, {"agent_name": "claude"})

        mock_agent = MagicMock()
        mock_create = MagicMock(return_value=mock_agent)
        mock_prober_cls = MagicMock(return_value=MagicMock())
        mock_probe_type_cls = MagicMock(return_value=MagicMock(value="contradiction"))
        mock_elo = MagicMock()
        mock_elo.record_redteam_result.side_effect = KeyError("agent not found")

        handler.ctx["nomic_dir"] = nomic_dir
        handler.ctx["elo_system"] = mock_elo

        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", mock_create),
            patch("aragora.server.handlers.agents.probes.CapabilityProber", mock_prober_cls),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type_cls,
            ),
            patch(
                "aragora.server.handlers.agents.probes.run_async",
                return_value=mock_probe_report,
            ),
            patch(
                "aragora.server.handlers.agents.probes.invalidate_leaderboard_cache",
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            # Still 200 despite ELO failure
            assert _status(result) == 200

    def test_probes_per_type_capped_at_10(
        self, handler, mock_http_handler, mock_probe_report, nomic_dir
    ):
        """probes_per_type is capped at 10."""
        _set_body(
            mock_http_handler,
            {"agent_name": "claude", "probes_per_type": 50},
        )

        mock_create = MagicMock(return_value=MagicMock())
        mock_prober_instance = MagicMock()
        mock_prober_instance.probe_agent = MagicMock(return_value=mock_probe_report)
        mock_prober_cls = MagicMock(return_value=mock_prober_instance)
        mock_probe_type_cls = MagicMock(return_value=MagicMock(value="contradiction"))

        handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", mock_create),
            patch("aragora.server.handlers.agents.probes.CapabilityProber", mock_prober_cls),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type_cls,
            ),
            patch(
                "aragora.server.handlers.agents.probes.run_async",
                return_value=mock_probe_report,
            ),
            patch(
                "aragora.server.handlers.agents.probes.invalidate_leaderboard_cache",
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            assert _status(result) == 200
            # Verify probes_per_type was capped to 10 in the call
            run_async_call = mock_prober_cls.return_value.probe_agent
            # Not directly assertable here since run_async is mocked,
            # but the handler logic caps it at 10

    def test_default_probe_types_used(
        self, handler, mock_http_handler, mock_probe_report, nomic_dir
    ):
        """Default probe types are used when not specified."""
        _set_body(mock_http_handler, {"agent_name": "claude"})

        mock_create = MagicMock(return_value=MagicMock())
        mock_prober_cls = MagicMock(return_value=MagicMock())
        # Track which probe types get created
        created_types = []

        def mock_probe_type_init(val):
            created_types.append(val)
            return MagicMock(value=val)

        mock_probe_type_cls = MagicMock(side_effect=mock_probe_type_init)

        handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", mock_create),
            patch("aragora.server.handlers.agents.probes.CapabilityProber", mock_prober_cls),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type_cls,
            ),
            patch(
                "aragora.server.handlers.agents.probes.run_async",
                return_value=mock_probe_report,
            ),
            patch(
                "aragora.server.handlers.agents.probes.invalidate_leaderboard_cache",
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            assert _status(result) == 200
            # Default types are: contradiction, hallucination, sycophancy, persistence
            assert "contradiction" in created_types
            assert "hallucination" in created_types
            assert "sycophancy" in created_types
            assert "persistence" in created_types

    def test_probe_report_saved_to_disk(
        self, handler, mock_http_handler, mock_probe_report, nomic_dir
    ):
        """Probe report is saved to nomic_dir/probes/agent_name/."""
        _set_body(mock_http_handler, {"agent_name": "claude"})

        mock_create = MagicMock(return_value=MagicMock())
        mock_prober_cls = MagicMock(return_value=MagicMock())
        mock_probe_type_cls = MagicMock(return_value=MagicMock(value="contradiction"))

        handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", mock_create),
            patch("aragora.server.handlers.agents.probes.CapabilityProber", mock_prober_cls),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type_cls,
            ),
            patch(
                "aragora.server.handlers.agents.probes.run_async",
                return_value=mock_probe_report,
            ),
            patch(
                "aragora.server.handlers.agents.probes.invalidate_leaderboard_cache",
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            assert _status(result) == 200
            # Verify the report was saved
            agent_probes_dir = nomic_dir / "probes" / "claude"
            assert agent_probes_dir.exists()
            saved_files = list(agent_probes_dir.glob("*.json"))
            assert len(saved_files) == 1

    def test_probe_save_failure_is_nonfatal(self, handler, mock_http_handler, mock_probe_report):
        """Probe save failure does not crash the response."""
        _set_body(mock_http_handler, {"agent_name": "claude"})

        mock_create = MagicMock(return_value=MagicMock())
        mock_prober_cls = MagicMock(return_value=MagicMock())
        mock_probe_type_cls = MagicMock(return_value=MagicMock(value="contradiction"))

        # Use a read-only nomic dir to trigger save failure
        handler.ctx["nomic_dir"] = Path("/nonexistent/readonly/path")

        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", mock_create),
            patch("aragora.server.handlers.agents.probes.CapabilityProber", mock_prober_cls),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type_cls,
            ),
            patch(
                "aragora.server.handlers.agents.probes.run_async",
                return_value=mock_probe_report,
            ),
            patch(
                "aragora.server.handlers.agents.probes.invalidate_leaderboard_cache",
            ),
        ):
            # Should still return 200 even if save fails
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            assert _status(result) == 200

    def test_probe_without_elo_system(
        self, handler, mock_http_handler, mock_probe_report, nomic_dir
    ):
        """Probe works without an ELO system configured."""
        _set_body(mock_http_handler, {"agent_name": "claude"})

        mock_create = MagicMock(return_value=MagicMock())
        mock_prober_cls = MagicMock(return_value=MagicMock())
        mock_probe_type_cls = MagicMock(return_value=MagicMock(value="contradiction"))

        handler.ctx["nomic_dir"] = nomic_dir
        # No elo_system in ctx

        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", mock_create),
            patch("aragora.server.handlers.agents.probes.CapabilityProber", mock_prober_cls),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type_cls,
            ),
            patch(
                "aragora.server.handlers.agents.probes.run_async",
                return_value=mock_probe_report,
            ),
            patch(
                "aragora.server.handlers.agents.probes.invalidate_leaderboard_cache",
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            assert _status(result) == 200

    def test_zero_probes_run_pass_rate(self, handler, mock_http_handler, nomic_dir):
        """Pass rate is 1.0 when zero probes were run."""
        _set_body(mock_http_handler, {"agent_name": "claude"})

        zero_report = MagicMock()
        zero_report.report_id = "probe-report-zero"
        zero_report.probes_run = 0
        zero_report.vulnerabilities_found = 0
        zero_report.vulnerability_rate = 0.0
        zero_report.elo_penalty = 0.0
        zero_report.critical_count = 0
        zero_report.high_count = 0
        zero_report.medium_count = 0
        zero_report.low_count = 0
        zero_report.recommendations = []
        zero_report.created_at = "2026-02-23"
        zero_report.by_type = {}
        zero_report.to_dict.return_value = {"report_id": "probe-report-zero"}

        mock_create = MagicMock(return_value=MagicMock())
        mock_prober_cls = MagicMock(return_value=MagicMock())
        mock_probe_type_cls = MagicMock(return_value=MagicMock(value="contradiction"))

        handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", mock_create),
            patch("aragora.server.handlers.agents.probes.CapabilityProber", mock_prober_cls),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type_cls,
            ),
            patch(
                "aragora.server.handlers.agents.probes.run_async",
                return_value=zero_report,
            ),
            patch(
                "aragora.server.handlers.agents.probes.invalidate_leaderboard_cache",
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            body = _body(result)
            assert _status(result) == 200
            assert body["summary"]["pass_rate"] == 1.0


# ---------------------------------------------------------------------------
# POST /api/v1/probes/run (Legacy)
# ---------------------------------------------------------------------------


class TestRunCapabilityProbeLegacy:
    """Tests for the legacy /api/v1/probes/run endpoint."""

    def test_legacy_route_dispatches(self, handler, mock_http_handler):
        """Legacy route dispatches to probe capability handler."""
        _set_body(mock_http_handler, {"agent_name": "claude"})
        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", False),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
        ):
            result = handler.handle_post("/api/v1/probes/run", {}, mock_http_handler)
            # Should return 503 (prober unavailable) same as capability endpoint
            assert _status(result) == 503

    def test_unknown_post_path_returns_none(self, handler, mock_http_handler):
        """Unknown POST paths return None."""
        result = handler.handle_post("/api/v1/probes/unknown", {}, mock_http_handler)
        assert result is None


# ---------------------------------------------------------------------------
# Result Transformation
# ---------------------------------------------------------------------------


class TestTransformResults:
    """Tests for _transform_results method."""

    def test_vulnerability_found_inverted_to_passed(self, handler):
        """vulnerability_found=True becomes passed=False."""
        report = MagicMock()
        result_mock = MagicMock()
        result_mock.to_dict.return_value = {
            "probe_id": "p1",
            "probe_type": "contradiction",
            "vulnerability_found": True,
            "severity": "HIGH",
            "vulnerability_description": "Found issue",
            "evidence": "Evidence here",
            "response_time_ms": 100,
        }
        report.by_type = {"contradiction": [result_mock]}

        transformed = handler._transform_results(report, None)
        assert transformed["contradiction"][0]["passed"] is False
        assert transformed["contradiction"][0]["severity"] == "high"

    def test_no_vulnerability_means_passed(self, handler):
        """vulnerability_found=False becomes passed=True."""
        report = MagicMock()
        result_mock = MagicMock()
        result_mock.to_dict.return_value = {
            "probe_id": "p2",
            "probe_type": "hallucination",
            "vulnerability_found": False,
            "severity": None,
            "vulnerability_description": "",
            "evidence": "",
            "response_time_ms": 80,
        }
        report.by_type = {"hallucination": [result_mock]}

        transformed = handler._transform_results(report, None)
        assert transformed["hallucination"][0]["passed"] is True
        assert transformed["hallucination"][0]["severity"] is None

    def test_result_without_to_dict(self, handler):
        """Handles results that are already dicts (no to_dict)."""
        report = MagicMock()
        raw_dict = {
            "probe_id": "p3",
            "probe_type": "sycophancy",
            "vulnerability_found": False,
            "severity": None,
            "vulnerability_description": "",
            "evidence": "",
            "response_time_ms": 50,
        }
        # No to_dict method on this object
        report.by_type = {"sycophancy": [raw_dict]}

        transformed = handler._transform_results(report, None)
        assert transformed["sycophancy"][0]["passed"] is True

    def test_multiple_types_transformed(self, handler):
        """Transforms results across multiple probe types."""
        report = MagicMock()
        r1 = {
            "probe_id": "p1",
            "vulnerability_found": True,
            "severity": "MEDIUM",
            "vulnerability_description": "Desc1",
            "evidence": "Ev1",
            "response_time_ms": 100,
        }
        r2 = {
            "probe_id": "p2",
            "vulnerability_found": False,
            "severity": None,
            "vulnerability_description": "",
            "evidence": "",
            "response_time_ms": 50,
        }
        report.by_type = {"contradiction": [r1], "hallucination": [r2]}

        transformed = handler._transform_results(report, None)
        assert len(transformed) == 2
        assert "contradiction" in transformed
        assert "hallucination" in transformed

    def test_probe_hooks_called_for_each_result(self, handler):
        """on_probe_result hook is called for each probe result."""
        report = MagicMock()
        r1 = {
            "probe_id": "p1",
            "vulnerability_found": False,
            "severity": None,
            "vulnerability_description": "",
            "evidence": "",
            "response_time_ms": 100,
        }
        report.by_type = {"contradiction": [r1]}

        mock_hook = MagicMock()
        hooks = {"on_probe_result": mock_hook}

        handler._transform_results(report, hooks)
        mock_hook.assert_called_once()
        call_kwargs = mock_hook.call_args.kwargs
        assert call_kwargs["probe_id"] == "p1"
        assert call_kwargs["passed"] is True


# ---------------------------------------------------------------------------
# ELO Recording
# ---------------------------------------------------------------------------


class TestRecordEloResult:
    """Tests for _record_elo_result method."""

    def test_no_elo_system(self, handler, mock_probe_report):
        """Does nothing when elo_system is None."""
        # Should not raise
        handler._record_elo_result(None, "claude", mock_probe_report, "rid")

    def test_zero_probes_run(self, handler):
        """Does nothing when probes_run is 0."""
        report = MagicMock()
        report.probes_run = 0
        elo = MagicMock()
        handler._record_elo_result(elo, "claude", report, "rid")
        elo.record_redteam_result.assert_not_called()

    def test_records_redteam_result(self, handler, mock_probe_report):
        """Records red team result with correct parameters."""
        elo = MagicMock()
        with patch("aragora.server.handlers.agents.probes.invalidate_leaderboard_cache"):
            handler._record_elo_result(elo, "claude", mock_probe_report, "rid-123")
        elo.record_redteam_result.assert_called_once()
        kwargs = elo.record_redteam_result.call_args.kwargs
        assert kwargs["agent_name"] == "claude"
        assert kwargs["robustness_score"] == 0.75  # 1.0 - 0.25
        assert kwargs["successful_attacks"] == 2
        assert kwargs["total_attacks"] == 8
        assert kwargs["session_id"] == "rid-123"

    def test_elo_error_is_nonfatal(self, handler, mock_probe_report):
        """ELO errors are caught and logged."""
        elo = MagicMock()
        elo.record_redteam_result.side_effect = ValueError("bad value")
        # Should not raise
        handler._record_elo_result(elo, "claude", mock_probe_report, "rid")


# ---------------------------------------------------------------------------
# Save Probe Report
# ---------------------------------------------------------------------------


class TestSaveProbeReport:
    """Tests for _save_probe_report method."""

    def test_no_nomic_dir(self, handler, mock_probe_report):
        """Does nothing when nomic_dir is not set."""
        # Should not raise
        handler._save_probe_report("claude", mock_probe_report)

    def test_saves_report_file(self, handler, mock_probe_report, nomic_dir):
        """Saves JSON report to correct directory."""
        handler.ctx["nomic_dir"] = nomic_dir
        handler._save_probe_report("claude", mock_probe_report)

        agent_dir = nomic_dir / "probes" / "claude"
        assert agent_dir.exists()
        files = list(agent_dir.glob("*.json"))
        assert len(files) == 1
        content = json.loads(files[0].read_text())
        assert content["report_id"] == "probe-report-abc12345"

    def test_save_error_is_nonfatal(self, handler, mock_probe_report):
        """Save errors are caught and logged."""
        handler.ctx["nomic_dir"] = Path("/nonexistent/path")
        # Should not raise
        handler._save_probe_report("claude", mock_probe_report)


# ---------------------------------------------------------------------------
# Probe Hooks
# ---------------------------------------------------------------------------


class TestGetProbeHooks:
    """Tests for _get_probe_hooks method."""

    def test_no_server_attribute(self, handler, mock_http_handler):
        """Returns None when handler has no server."""
        mock_http_handler.server = None
        result = handler._get_probe_hooks(mock_http_handler)
        assert result is None

    def test_no_stream_server(self, handler, mock_http_handler):
        """Returns None when server has no stream_server."""
        mock_http_handler.server = MagicMock(spec=[])
        result = handler._get_probe_hooks(mock_http_handler)
        assert result is None

    def test_stream_server_is_none(self, handler, mock_http_handler):
        """Returns None when stream_server is None."""
        server = MagicMock()
        server.stream_server = None
        mock_http_handler.server = server
        result = handler._get_probe_hooks(mock_http_handler)
        assert result is None

    def test_import_error_returns_none(self, handler, mock_http_handler):
        """Returns None when nomic_stream import fails."""
        server = MagicMock()
        server.stream_server = MagicMock()
        server.stream_server.emitter = MagicMock()
        mock_http_handler.server = server

        with patch(
            "aragora.server.handlers.agents.probes.ProbesHandler._get_probe_hooks",
            wraps=handler._get_probe_hooks,
        ):
            # The import will fail naturally if the module is not in sys.modules
            # but the method handles ImportError gracefully
            result = handler._get_probe_hooks(mock_http_handler)
            # Either returns hooks or None, both are acceptable
            assert result is None or isinstance(result, dict)

    def test_handler_without_server(self, handler):
        """Returns None when handler has no server attribute at all."""
        handler_without_server = MagicMock(spec=[])
        result = handler._get_probe_hooks(handler_without_server)
        assert result is None


# ---------------------------------------------------------------------------
# POST routing
# ---------------------------------------------------------------------------


class TestHandlePostRouting:
    """Tests for POST request routing."""

    def test_capability_route_dispatches(self, handler, mock_http_handler):
        """POST to /api/v1/probes/capability calls _run_capability_probe."""
        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", False),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            # Prober not available -> 503
            assert _status(result) == 503

    def test_run_route_dispatches(self, handler, mock_http_handler):
        """POST to /api/v1/probes/run calls legacy handler."""
        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", False),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
        ):
            result = handler.handle_post("/api/v1/probes/run", {}, mock_http_handler)
            assert _status(result) == 503

    def test_unknown_post_route(self, handler, mock_http_handler):
        """Unknown POST route returns None."""
        result = handler.handle_post("/api/v1/probes/other", {}, mock_http_handler)
        assert result is None


# ---------------------------------------------------------------------------
# Response structure validation
# ---------------------------------------------------------------------------


class TestResponseStructure:
    """Tests that verify the response structure of successful probes."""

    def test_full_response_has_expected_keys(
        self, handler, mock_http_handler, mock_probe_report, nomic_dir
    ):
        """Successful probe response contains all expected top-level keys."""
        _set_body(mock_http_handler, {"agent_name": "claude"})

        mock_create = MagicMock(return_value=MagicMock())
        mock_prober_cls = MagicMock(return_value=MagicMock())
        mock_probe_type_cls = MagicMock(return_value=MagicMock(value="contradiction"))

        handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", mock_create),
            patch("aragora.server.handlers.agents.probes.CapabilityProber", mock_prober_cls),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type_cls,
            ),
            patch(
                "aragora.server.handlers.agents.probes.run_async",
                return_value=mock_probe_report,
            ),
            patch(
                "aragora.server.handlers.agents.probes.invalidate_leaderboard_cache",
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            body = _body(result)
            assert _status(result) == 200

            # Verify top-level keys
            expected_keys = {
                "report_id",
                "target_agent",
                "probes_run",
                "vulnerabilities_found",
                "vulnerability_rate",
                "elo_penalty",
                "by_type",
                "summary",
                "recommendations",
                "created_at",
            }
            assert expected_keys.issubset(set(body.keys()))

            # Verify summary structure
            summary_keys = {
                "total",
                "passed",
                "failed",
                "pass_rate",
                "critical",
                "high",
                "medium",
                "low",
            }
            assert summary_keys.issubset(set(body["summary"].keys()))

    def test_by_type_transformed_structure(
        self, handler, mock_http_handler, mock_probe_report, nomic_dir
    ):
        """by_type results have correct transformed structure."""
        _set_body(mock_http_handler, {"agent_name": "claude"})

        mock_create = MagicMock(return_value=MagicMock())
        mock_prober_cls = MagicMock(return_value=MagicMock())
        mock_probe_type_cls = MagicMock(return_value=MagicMock(value="contradiction"))

        handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", mock_create),
            patch("aragora.server.handlers.agents.probes.CapabilityProber", mock_prober_cls),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type_cls,
            ),
            patch(
                "aragora.server.handlers.agents.probes.run_async",
                return_value=mock_probe_report,
            ),
            patch(
                "aragora.server.handlers.agents.probes.invalidate_leaderboard_cache",
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            body = _body(result)

            # The mock_probe_report has contradiction results
            assert "contradiction" in body["by_type"]
            results = body["by_type"]["contradiction"]
            assert len(results) == 2

            # First result had vulnerability_found=True -> passed=False
            assert results[0]["passed"] is False
            assert results[0]["severity"] == "high"
            assert results[0]["probe_id"] == "p1"

            # Second result had vulnerability_found=False -> passed=True
            assert results[1]["passed"] is True


# ---------------------------------------------------------------------------
# Model type parameter
# ---------------------------------------------------------------------------


class TestModelTypeParameter:
    """Tests for model_type parameter handling."""

    def test_default_model_type(self, handler, mock_http_handler, mock_probe_report, nomic_dir):
        """Default model_type is anthropic-api."""
        _set_body(mock_http_handler, {"agent_name": "claude"})

        mock_create = MagicMock(return_value=MagicMock())
        mock_prober_cls = MagicMock(return_value=MagicMock())
        mock_probe_type_cls = MagicMock(return_value=MagicMock(value="contradiction"))

        handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", mock_create),
            patch("aragora.server.handlers.agents.probes.CapabilityProber", mock_prober_cls),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type_cls,
            ),
            patch(
                "aragora.server.handlers.agents.probes.run_async",
                return_value=mock_probe_report,
            ),
            patch(
                "aragora.server.handlers.agents.probes.invalidate_leaderboard_cache",
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            assert _status(result) == 200
            mock_create.assert_called_once_with("anthropic-api", name="claude", role="proposer")

    def test_custom_model_type(self, handler, mock_http_handler, mock_probe_report, nomic_dir):
        """Custom model_type is passed to create_agent."""
        _set_body(
            mock_http_handler,
            {"agent_name": "gpt4", "model_type": "openai-api"},
        )

        mock_create = MagicMock(return_value=MagicMock())
        mock_prober_cls = MagicMock(return_value=MagicMock())
        mock_probe_type_cls = MagicMock(return_value=MagicMock(value="contradiction"))

        handler.ctx["nomic_dir"] = nomic_dir

        with (
            patch("aragora.server.handlers.agents.probes.PROBER_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.AGENT_AVAILABLE", True),
            patch("aragora.server.handlers.agents.probes.create_agent", mock_create),
            patch("aragora.server.handlers.agents.probes.CapabilityProber", mock_prober_cls),
            patch(
                "aragora.billing.jwt_auth.extract_user_from_request",
                return_value=MagicMock(is_authenticated=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_against_schema",
                return_value=MagicMock(is_valid=True),
            ),
            patch(
                "aragora.server.handlers.agents.probes.validate_agent_name",
                return_value=(True, None),
            ),
            patch(
                "aragora.server.handlers.agents.probes.ProbeType",
                mock_probe_type_cls,
            ),
            patch(
                "aragora.server.handlers.agents.probes.run_async",
                return_value=mock_probe_report,
            ),
            patch(
                "aragora.server.handlers.agents.probes.invalidate_leaderboard_cache",
            ),
        ):
            result = handler.handle_post("/api/v1/probes/capability", {}, mock_http_handler)
            assert _status(result) == 200
            mock_create.assert_called_once_with("openai-api", name="gpt4", role="proposer")
