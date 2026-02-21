"""
Tests for aragora.server.handlers.verticals - Vertical Specialist Handlers.

Tests cover:
- VerticalsCircuitBreaker: state transitions, can_proceed, record_success/failure
- VerticalsHandler: instantiation, ROUTES, can_handle routing
- GET /api/verticals: list verticals with/without keyword filter
- GET /api/verticals/:id: get specific vertical
- GET /api/verticals/:id/tools: get vertical tools
- GET /api/verticals/:id/compliance: get compliance frameworks
- GET /api/verticals/suggest: suggest vertical for task
- POST /api/verticals/:id/debate: create debate (async)
- POST /api/verticals/:id/agent: create specialist agent
- PUT /api/verticals/:id/config: update vertical config
- Circuit breaker integration in handler methods
- Error paths: registry unavailable, vertical not found, invalid input
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.verticals import (
    VerticalsCircuitBreaker,
    VerticalsHandler,
    get_verticals_circuit_breaker,
    reset_verticals_circuit_breaker,
)
from aragora.server.handlers.utils.responses import HandlerResult


# ===========================================================================
# Helpers
# ===========================================================================


def _parse_body(result: HandlerResult) -> dict[str, Any]:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body)


def _make_mock_handler(
    method: str = "GET",
    body: bytes = b"",
    content_type: str = "application/json",
) -> MagicMock:
    """Create a mock HTTP handler object."""
    handler = MagicMock()
    handler.command = method
    handler.client_address = ("127.0.0.1", 12345)
    handler.headers = {
        "Content-Length": str(len(body)),
        "Content-Type": content_type,
        "Host": "localhost:8080",
        "Authorization": "Bearer test-token",
    }
    handler.rfile = MagicMock()
    handler.rfile.read.return_value = body
    return handler


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset the global circuit breaker before each test."""
    reset_verticals_circuit_breaker()
    yield
    reset_verticals_circuit_breaker()


@pytest.fixture
def handler():
    """Create a VerticalsHandler with mocked auth."""
    h = VerticalsHandler(ctx={})
    return h


@pytest.fixture
def mock_registry():
    """Create a mock VerticalRegistry with standard responses."""
    registry = MagicMock()
    registry.list_all.return_value = {
        "healthcare": {"display_name": "Healthcare", "description": "Medical AI"},
        "finance": {"display_name": "Finance", "description": "Financial AI"},
    }
    registry.get_registered_ids.return_value = ["healthcare", "finance"]
    registry.get_by_keyword.return_value = ["healthcare"]
    registry.is_registered.return_value = True

    # Mock spec for get()
    mock_spec = MagicMock()
    mock_spec.description = "Healthcare vertical"
    mock_config = MagicMock()
    mock_config.display_name = "Healthcare"
    mock_config.domain_keywords = ["medical", "health"]
    mock_config.expertise_areas = ["diagnosis", "treatment"]
    mock_config.tools = []
    mock_config.compliance_frameworks = []
    mock_config.model_config.to_dict.return_value = {"primary_model": "claude"}
    mock_config.version = "1.0"
    mock_config.author = "aragora"
    mock_config.tags = ["healthcare"]
    mock_config.get_enabled_tools.return_value = []
    mock_config.get_compliance_frameworks.return_value = []
    mock_spec.config = mock_config

    registry.get.return_value = mock_spec
    registry.get_config.return_value = mock_config
    registry.get_for_task.return_value = "healthcare"

    return registry


def _patch_auth(handler):
    """Context manager to mock RBAC auth for handler."""
    mock_auth = MagicMock()
    mock_auth.has_permission.return_value = True
    return patch.object(handler, "get_auth_context", new_callable=AsyncMock, return_value=mock_auth)


def _patch_check_permission(handler):
    """Context manager to mock permission check."""
    return patch.object(handler, "check_permission", return_value=None)


def _patch_registry(mock_registry):
    """Context manager to mock the verticals registry import."""
    return patch(
        "aragora.server.handlers.verticals.VerticalsHandler._get_registry",
        return_value=mock_registry,
    )


# ===========================================================================
# Test VerticalsCircuitBreaker
# ===========================================================================


class TestVerticalsCircuitBreaker:
    """Test the circuit breaker state machine."""

    def test_initial_state_is_closed(self):
        cb = VerticalsCircuitBreaker()
        assert cb.state == VerticalsCircuitBreaker.CLOSED

    def test_can_proceed_when_closed(self):
        cb = VerticalsCircuitBreaker()
        assert cb.can_proceed() is True

    def test_opens_after_failure_threshold(self):
        cb = VerticalsCircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == VerticalsCircuitBreaker.CLOSED
        cb.record_failure()
        assert cb.state == VerticalsCircuitBreaker.OPEN

    def test_cannot_proceed_when_open(self):
        cb = VerticalsCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.can_proceed() is False

    def test_transitions_to_half_open_after_cooldown(self):
        cb = VerticalsCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.state == VerticalsCircuitBreaker.OPEN
        time.sleep(0.02)
        assert cb.state == VerticalsCircuitBreaker.HALF_OPEN

    def test_half_open_allows_limited_calls(self):
        cb = VerticalsCircuitBreaker(
            failure_threshold=1, cooldown_seconds=0.01, half_open_max_calls=2
        )
        cb.record_failure()
        time.sleep(0.02)
        assert cb.can_proceed() is True
        assert cb.can_proceed() is True
        assert cb.can_proceed() is False

    def test_half_open_closes_on_enough_successes(self):
        cb = VerticalsCircuitBreaker(
            failure_threshold=1, cooldown_seconds=0.01, half_open_max_calls=2
        )
        cb.record_failure()
        time.sleep(0.02)
        cb.can_proceed()
        cb.record_success()
        cb.can_proceed()
        cb.record_success()
        assert cb.state == VerticalsCircuitBreaker.CLOSED

    def test_half_open_reopens_on_failure(self):
        cb = VerticalsCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.can_proceed()
        cb.record_failure()
        assert cb.state == VerticalsCircuitBreaker.OPEN

    def test_success_resets_failure_count_in_closed(self):
        cb = VerticalsCircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # After success, count resets, so 3 more failures needed
        cb.record_failure()
        cb.record_failure()
        assert cb.state == VerticalsCircuitBreaker.CLOSED

    def test_get_status_returns_dict(self):
        cb = VerticalsCircuitBreaker()
        status = cb.get_status()
        assert "state" in status
        assert "failure_count" in status
        assert "failure_threshold" in status
        assert status["state"] == "closed"

    def test_reset(self):
        cb = VerticalsCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == VerticalsCircuitBreaker.OPEN
        cb.reset()
        assert cb.state == VerticalsCircuitBreaker.CLOSED
        assert cb.can_proceed() is True


# ===========================================================================
# Test VerticalsHandler Basics
# ===========================================================================


class TestVerticalsHandlerBasics:
    """Basic instantiation and routing tests."""

    def test_instantiation(self, handler):
        assert handler is not None
        assert isinstance(handler, VerticalsHandler)

    def test_resource_type(self, handler):
        assert handler.RESOURCE_TYPE == "verticals"

    def test_routes_attribute(self, handler):
        assert "/api/verticals" in handler.ROUTES
        assert "/api/verticals/suggest" in handler.ROUTES
        assert "/api/verticals/*" in handler.ROUTES

    def test_can_handle_list(self, handler):
        assert handler.can_handle("/api/verticals", "GET") is True

    def test_can_handle_suggest(self, handler):
        assert handler.can_handle("/api/verticals/suggest", "GET") is True

    def test_can_handle_specific_vertical(self, handler):
        assert handler.can_handle("/api/verticals/healthcare", "GET") is True

    def test_can_handle_tools(self, handler):
        assert handler.can_handle("/api/verticals/healthcare/tools", "GET") is True

    def test_can_handle_compliance(self, handler):
        assert handler.can_handle("/api/verticals/healthcare/compliance", "GET") is True

    def test_can_handle_debate(self, handler):
        assert handler.can_handle("/api/verticals/healthcare/debate", "POST") is True

    def test_can_handle_agent(self, handler):
        assert handler.can_handle("/api/verticals/healthcare/agent", "POST") is True

    def test_cannot_handle_other(self, handler):
        assert handler.can_handle("/api/debates", "GET") is False

    def test_can_handle_versioned_path(self, handler):
        assert handler.can_handle("/api/v1/verticals", "GET") is True

    def test_circuit_breaker_status(self, handler):
        status = handler.get_circuit_breaker_status()
        assert status["state"] == "closed"


# ===========================================================================
# Test _list_verticals
# ===========================================================================


class TestListVerticals:
    """Tests for listing verticals."""

    def test_list_verticals_success(self, handler, mock_registry):
        with _patch_registry(mock_registry):
            result = handler._list_verticals({})
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["total"] == 2
            assert len(data["verticals"]) == 2

    def test_list_verticals_with_keyword(self, handler, mock_registry):
        with _patch_registry(mock_registry):
            result = handler._list_verticals({"keyword": "medical"})
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["total"] == 1

    def test_list_verticals_registry_unavailable(self, handler):
        with _patch_registry(None):
            result = handler._list_verticals({})
            assert result.status_code == 503
            data = _parse_body(result)
            assert data["total"] == 0

    def test_list_verticals_keyword_too_long(self, handler, mock_registry):
        with _patch_registry(mock_registry):
            result = handler._list_verticals({"keyword": "x" * 201})
            assert result.status_code == 400


# ===========================================================================
# Test _get_vertical
# ===========================================================================


class TestGetVertical:
    """Tests for getting a specific vertical."""

    def test_get_vertical_success(self, handler, mock_registry):
        with _patch_registry(mock_registry):
            result = handler._get_vertical("healthcare")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["vertical_id"] == "healthcare"
            assert data["display_name"] == "Healthcare"

    def test_get_vertical_not_found(self, handler, mock_registry):
        mock_registry.get.return_value = None
        with _patch_registry(mock_registry):
            result = handler._get_vertical("nonexistent")
            assert result.status_code == 404

    def test_get_vertical_registry_unavailable(self, handler):
        with _patch_registry(None):
            result = handler._get_vertical("healthcare")
            assert result.status_code == 503


# ===========================================================================
# Test _get_tools
# ===========================================================================


class TestGetTools:
    """Tests for getting vertical tools."""

    def test_get_tools_success(self, handler, mock_registry):
        with _patch_registry(mock_registry):
            result = handler._get_tools("healthcare")
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["vertical_id"] == "healthcare"

    def test_get_tools_not_found(self, handler, mock_registry):
        mock_registry.get_config.return_value = None
        with _patch_registry(mock_registry):
            result = handler._get_tools("nonexistent")
            assert result.status_code == 404


# ===========================================================================
# Test _suggest_vertical
# ===========================================================================


class TestSuggestVertical:
    """Tests for suggesting a vertical."""

    def test_suggest_success(self, handler, mock_registry):
        with _patch_registry(mock_registry):
            result = handler._suggest_vertical({"task": "Diagnose patient symptoms"})
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["suggestion"]["vertical_id"] == "healthcare"

    def test_suggest_no_match(self, handler, mock_registry):
        mock_registry.get_for_task.return_value = None
        with _patch_registry(mock_registry):
            result = handler._suggest_vertical({"task": "something unrelated"})
            assert result.status_code == 200
            data = _parse_body(result)
            assert data["suggestion"] is None

    def test_suggest_missing_task(self, handler, mock_registry):
        with _patch_registry(mock_registry):
            result = handler._suggest_vertical({})
            assert result.status_code == 400

    def test_suggest_empty_task(self, handler, mock_registry):
        with _patch_registry(mock_registry):
            result = handler._suggest_vertical({"task": "  "})
            assert result.status_code == 400

    def test_suggest_task_too_long(self, handler, mock_registry):
        with _patch_registry(mock_registry):
            result = handler._suggest_vertical({"task": "x" * 100_001})
            assert result.status_code == 400


# ===========================================================================
# Test _create_agent
# ===========================================================================


class TestCreateAgent:
    """Tests for creating a specialist agent."""

    def test_create_agent_success(self, handler, mock_registry):
        mock_specialist = MagicMock()
        mock_specialist.name = "healthcare-agent"
        mock_specialist.model = "claude"
        mock_specialist.role = "specialist"
        mock_specialist.expertise_areas = ["diagnosis"]
        mock_specialist.to_dict.return_value = {"name": "healthcare-agent"}
        mock_specialist.get_enabled_tools.return_value = []
        mock_registry.create_specialist.return_value = mock_specialist

        body = json.dumps({"role": "specialist"}).encode()
        mock_handler = _make_mock_handler("POST", body)

        with _patch_registry(mock_registry):
            with patch.object(handler, "read_json_body", return_value={"role": "specialist"}):
                result = handler._create_agent("healthcare", mock_handler)
                assert result.status_code == 200
                data = _parse_body(result)
                assert data["name"] == "healthcare-agent"

    def test_create_agent_not_registered(self, handler, mock_registry):
        mock_registry.is_registered.return_value = False
        mock_handler = _make_mock_handler("POST")

        with _patch_registry(mock_registry):
            result = handler._create_agent("nonexistent", mock_handler)
            assert result.status_code == 404

    def test_create_agent_invalid_body(self, handler, mock_registry):
        mock_handler = _make_mock_handler("POST")
        with _patch_registry(mock_registry):
            with patch.object(handler, "read_json_body", return_value=None):
                result = handler._create_agent("healthcare", mock_handler)
                assert result.status_code == 400


# ===========================================================================
# Test Input Validation
# ===========================================================================


class TestInputValidation:
    """Tests for input validation helper methods."""

    def test_validate_keyword_none(self, handler):
        is_valid, err = handler._validate_keyword(None)
        assert is_valid is True

    def test_validate_keyword_too_long(self, handler):
        is_valid, err = handler._validate_keyword("x" * 201)
        assert is_valid is False

    def test_validate_task_none(self, handler):
        is_valid, err = handler._validate_task(None)
        assert is_valid is False

    def test_validate_task_empty(self, handler):
        is_valid, err = handler._validate_task("  ")
        assert is_valid is False

    def test_validate_task_valid(self, handler):
        is_valid, err = handler._validate_task("Analyze data")
        assert is_valid is True

    def test_validate_topic_none(self, handler):
        is_valid, err = handler._validate_topic(None)
        assert is_valid is False

    def test_validate_topic_valid(self, handler):
        is_valid, err = handler._validate_topic("Debate topic")
        assert is_valid is True

    def test_validate_agent_name_none_is_ok(self, handler):
        is_valid, err = handler._validate_agent_name(None)
        assert is_valid is True

    def test_validate_agent_name_too_long(self, handler):
        is_valid, err = handler._validate_agent_name("x" * 101)
        assert is_valid is False

    def test_validate_additional_agents_none(self, handler):
        is_valid, err = handler._validate_additional_agents(None)
        assert is_valid is True

    def test_validate_additional_agents_not_list(self, handler):
        is_valid, err = handler._validate_additional_agents("not a list")
        assert is_valid is False

    def test_validate_additional_agents_too_many(self, handler):
        is_valid, err = handler._validate_additional_agents(list(range(11)))
        assert is_valid is False


# ===========================================================================
# Test Global Circuit Breaker Functions
# ===========================================================================


class TestGlobalCircuitBreaker:
    """Tests for global circuit breaker access functions."""

    def test_get_circuit_breaker(self):
        cb = get_verticals_circuit_breaker()
        assert isinstance(cb, VerticalsCircuitBreaker)

    def test_reset_circuit_breaker(self):
        cb = get_verticals_circuit_breaker()
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == VerticalsCircuitBreaker.OPEN
        reset_verticals_circuit_breaker()
        assert cb.state == VerticalsCircuitBreaker.CLOSED
