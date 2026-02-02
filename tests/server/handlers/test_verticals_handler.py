"""
Tests for VerticalsHandler - Industry vertical specialist endpoints.

Stability: STABLE (graduated from EXPERIMENTAL)

Covers:
- Vertical listing and filtering
- Vertical configuration retrieval
- Tools and compliance framework access
- Vertical suggestion for tasks
- Specialist agent creation
- Vertical-specific debate creation
- Configuration updates
- Route matching (can_handle)
- RBAC permission checks
- Error handling
- Circuit breaker functionality
- Input validation
- Rate limiting integration
"""

from __future__ import annotations

import time
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.verticals import (
    VerticalsHandler,
    VerticalsCircuitBreaker,
    get_verticals_circuit_breaker,
    reset_verticals_circuit_breaker,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def server_context():
    """Create a mock server context."""
    return {"config": {"debug": True}}


@pytest.fixture
def verticals_handler(server_context):
    """Create a VerticalsHandler instance."""
    reset_verticals_circuit_breaker()
    return VerticalsHandler(server_context)


@pytest.fixture
def mock_handler():
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.headers = {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json",
    }
    handler.command = "GET"
    return handler


@pytest.fixture
def mock_handler_post():
    """Create a mock HTTP request handler for POST."""
    handler = MagicMock()
    handler.headers = {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json",
    }
    handler.command = "POST"
    handler.rfile = MagicMock()
    return handler


@pytest.fixture
def mock_handler_put():
    """Create a mock HTTP request handler for PUT."""
    handler = MagicMock()
    handler.headers = {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json",
    }
    handler.command = "PUT"
    handler.rfile = MagicMock()
    return handler


@pytest.fixture
def mock_vertical_config():
    """Create a mock vertical configuration."""
    config = MagicMock()
    config.display_name = "Healthcare"
    config.domain_keywords = ["medical", "healthcare", "clinical"]
    config.expertise_areas = ["diagnostics", "treatment", "compliance"]
    config.version = "1.0.0"
    config.author = "Aragora"
    config.tags = ["healthcare", "enterprise"]

    # Tools
    tool1 = MagicMock()
    tool1.name = "patient_lookup"
    tool1.to_dict = MagicMock(
        return_value={
            "name": "patient_lookup",
            "description": "Look up patient records",
            "enabled": True,
        }
    )

    tool2 = MagicMock()
    tool2.name = "medication_checker"
    tool2.to_dict = MagicMock(
        return_value={
            "name": "medication_checker",
            "description": "Check medication interactions",
            "enabled": True,
        }
    )

    config.tools = [tool1, tool2]
    config.get_enabled_tools = MagicMock(return_value=[tool1, tool2])

    # Compliance frameworks
    compliance1 = MagicMock()
    compliance1.name = "HIPAA"
    compliance1.to_dict = MagicMock(
        return_value={
            "name": "HIPAA",
            "level": "strict",
            "requirements": ["data_encryption", "audit_logging"],
        }
    )

    config.compliance_frameworks = [compliance1]
    config.get_compliance_frameworks = MagicMock(return_value=[compliance1])

    # Model config
    model_config = MagicMock()
    model_config.to_dict = MagicMock(
        return_value={
            "preferred_model": "claude-opus-4",
            "temperature": 0.7,
            "max_tokens": 4096,
        }
    )
    model_config.temperature = 0.7
    model_config.max_tokens = 4096
    model_config.primary_model = "claude-opus-4"
    model_config.primary_provider = "anthropic"
    config.model_config = model_config

    return config


@pytest.fixture
def mock_vertical_spec(mock_vertical_config):
    """Create a mock vertical specification."""
    spec = MagicMock()
    spec.config = mock_vertical_config
    spec.description = "Healthcare vertical for medical applications"
    return spec


@pytest.fixture
def mock_registry(mock_vertical_spec, mock_vertical_config):
    """Create a mock VerticalRegistry."""
    registry = MagicMock()

    # List all verticals
    registry.list_all = MagicMock(
        return_value={
            "healthcare": {
                "display_name": "Healthcare",
                "description": "Healthcare vertical",
            },
            "finance": {
                "display_name": "Finance",
                "description": "Finance vertical",
            },
            "legal": {
                "display_name": "Legal",
                "description": "Legal vertical",
            },
        }
    )

    # Get by keyword
    registry.get_by_keyword = MagicMock(return_value=["healthcare"])

    # Get specific vertical
    registry.get = MagicMock(return_value=mock_vertical_spec)
    registry.get_config = MagicMock(return_value=mock_vertical_config)

    # Registration checks
    registry.is_registered = MagicMock(return_value=True)
    registry.get_registered_ids = MagicMock(return_value=["healthcare", "finance", "legal"])

    # Task suggestion
    registry.get_for_task = MagicMock(return_value="healthcare")

    # Agent creation
    mock_specialist = MagicMock()
    mock_specialist.name = "healthcare-specialist"
    mock_specialist.model = "claude-opus-4"
    mock_specialist.role = "specialist"
    mock_specialist.expertise_areas = ["diagnostics", "treatment"]
    mock_specialist.to_dict = MagicMock(
        return_value={
            "name": "healthcare-specialist",
            "model": "claude-opus-4",
        }
    )
    mock_specialist.get_enabled_tools = MagicMock(return_value=[])

    registry.create_specialist = MagicMock(return_value=mock_specialist)

    return registry


@pytest.fixture
def mock_auth_context():
    """Create a mock authorization context."""
    ctx = MagicMock()
    ctx.user_id = "user-123"
    ctx.roles = ["admin"]
    return ctx


# -----------------------------------------------------------------------------
# Circuit Breaker Tests
# -----------------------------------------------------------------------------


class TestVerticalsCircuitBreaker:
    """Tests for VerticalsCircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Circuit breaker starts in closed state."""
        cb = VerticalsCircuitBreaker()
        assert cb.state == "closed"

    def test_can_proceed_when_closed(self):
        """Requests are allowed when circuit is closed."""
        cb = VerticalsCircuitBreaker()
        assert cb.can_proceed() is True

    def test_failure_increments_count(self):
        """Recording a failure increments the failure count."""
        cb = VerticalsCircuitBreaker(failure_threshold=3)
        cb.record_failure()
        status = cb.get_status()
        assert status["failure_count"] == 1

    def test_success_resets_failure_count_in_closed_state(self):
        """Recording success resets failure count in closed state."""
        cb = VerticalsCircuitBreaker()
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        status = cb.get_status()
        assert status["failure_count"] == 0

    def test_circuit_opens_after_threshold_failures(self):
        """Circuit opens after reaching failure threshold."""
        cb = VerticalsCircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "open"

    def test_cannot_proceed_when_open(self):
        """Requests are blocked when circuit is open."""
        cb = VerticalsCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == "open"
        assert cb.can_proceed() is False

    def test_circuit_transitions_to_half_open_after_cooldown(self):
        """Circuit transitions to half-open after cooldown period."""
        cb = VerticalsCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.02)
        assert cb.state == "half_open"

    def test_can_proceed_limited_times_in_half_open(self):
        """Limited number of test requests allowed in half-open state."""
        cb = VerticalsCircuitBreaker(
            failure_threshold=1, cooldown_seconds=0.01, half_open_max_calls=2
        )
        cb.record_failure()
        time.sleep(0.02)

        # First call should be allowed
        assert cb.can_proceed() is True
        # Second call should be allowed
        assert cb.can_proceed() is True
        # Third call should be blocked
        assert cb.can_proceed() is False

    def test_circuit_closes_after_successful_recovery(self):
        """Circuit closes after enough successes in half-open state."""
        cb = VerticalsCircuitBreaker(
            failure_threshold=1, cooldown_seconds=0.01, half_open_max_calls=2
        )
        cb.record_failure()
        time.sleep(0.02)

        # Consume the test calls and record successes
        cb.can_proceed()
        cb.record_success()
        cb.can_proceed()
        cb.record_success()

        assert cb.state == "closed"

    def test_circuit_reopens_on_failure_in_half_open(self):
        """Circuit reopens if failure occurs in half-open state."""
        cb = VerticalsCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)

        cb.can_proceed()
        cb.record_failure()

        assert cb.state == "open"

    def test_reset_returns_to_closed_state(self):
        """Reset method returns circuit to closed state."""
        cb = VerticalsCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == "open"

        cb.reset()

        assert cb.state == "closed"
        status = cb.get_status()
        assert status["failure_count"] == 0

    def test_get_status_returns_all_fields(self):
        """get_status returns complete status information."""
        cb = VerticalsCircuitBreaker(failure_threshold=5, cooldown_seconds=60.0)
        status = cb.get_status()

        assert "state" in status
        assert "failure_count" in status
        assert "success_count" in status
        assert "failure_threshold" in status
        assert "cooldown_seconds" in status
        assert "last_failure_time" in status
        assert status["failure_threshold"] == 5
        assert status["cooldown_seconds"] == 60.0


class TestGlobalCircuitBreaker:
    """Tests for global circuit breaker functions."""

    def test_get_verticals_circuit_breaker_returns_instance(self):
        """get_verticals_circuit_breaker returns a circuit breaker."""
        cb = get_verticals_circuit_breaker()
        assert isinstance(cb, VerticalsCircuitBreaker)

    def test_reset_verticals_circuit_breaker_resets_global(self):
        """reset_verticals_circuit_breaker resets the global instance."""
        cb = get_verticals_circuit_breaker()
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()

        reset_verticals_circuit_breaker()

        assert cb.state == "closed"


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration with handler."""

    def test_handler_has_circuit_breaker(self, verticals_handler):
        """Handler has a circuit breaker instance."""
        assert hasattr(verticals_handler, "_circuit_breaker")
        assert isinstance(verticals_handler._circuit_breaker, VerticalsCircuitBreaker)

    def test_get_circuit_breaker_status(self, verticals_handler):
        """Handler provides circuit breaker status."""
        status = verticals_handler.get_circuit_breaker_status()
        assert "state" in status
        assert status["state"] == "closed"

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_requests_when_open(self, verticals_handler, mock_handler):
        """Handler returns 503 when circuit breaker is open."""
        # Open the circuit breaker
        for _ in range(3):
            verticals_handler._circuit_breaker.record_failure()

        with (
            patch.object(
                verticals_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(verticals_handler, "check_permission"),
        ):
            mock_auth.return_value = MagicMock()

            result = await verticals_handler.handle("/api/verticals", {}, mock_handler)

        assert result.status_code == 503
        assert "temporarily unavailable" in result.body.decode().lower()


# -----------------------------------------------------------------------------
# Input Validation Tests
# -----------------------------------------------------------------------------


class TestInputValidation:
    """Tests for input validation methods."""

    def test_validate_keyword_accepts_valid_keyword(self, verticals_handler):
        """Valid keyword passes validation."""
        is_valid, err = verticals_handler._validate_keyword("healthcare")
        assert is_valid is True
        assert err == ""

    def test_validate_keyword_accepts_none(self, verticals_handler):
        """None keyword is valid (optional parameter)."""
        is_valid, err = verticals_handler._validate_keyword(None)
        assert is_valid is True

    def test_validate_keyword_rejects_too_long(self, verticals_handler):
        """Keyword exceeding max length is rejected."""
        long_keyword = "a" * (VerticalsHandler.MAX_KEYWORD_LENGTH + 1)
        is_valid, err = verticals_handler._validate_keyword(long_keyword)
        assert is_valid is False
        assert "exceeds maximum length" in err

    def test_validate_task_accepts_valid_task(self, verticals_handler):
        """Valid task passes validation."""
        is_valid, err = verticals_handler._validate_task("Diagnose a patient")
        assert is_valid is True
        assert err == ""

    def test_validate_task_rejects_none(self, verticals_handler):
        """Missing task is rejected."""
        is_valid, err = verticals_handler._validate_task(None)
        assert is_valid is False
        assert "Missing required parameter" in err

    def test_validate_task_rejects_empty(self, verticals_handler):
        """Empty task is rejected."""
        is_valid, err = verticals_handler._validate_task("   ")
        assert is_valid is False
        assert "cannot be empty" in err

    def test_validate_task_rejects_too_long(self, verticals_handler):
        """Task exceeding max length is rejected."""
        long_task = "a" * (VerticalsHandler.MAX_TASK_LENGTH + 1)
        is_valid, err = verticals_handler._validate_task(long_task)
        assert is_valid is False
        assert "exceeds maximum length" in err

    def test_validate_topic_accepts_valid_topic(self, verticals_handler):
        """Valid topic passes validation."""
        is_valid, err = verticals_handler._validate_topic("Medical diagnosis debate")
        assert is_valid is True
        assert err == ""

    def test_validate_topic_rejects_none(self, verticals_handler):
        """Missing topic is rejected."""
        is_valid, err = verticals_handler._validate_topic(None)
        assert is_valid is False
        assert "Missing required field" in err

    def test_validate_topic_rejects_empty(self, verticals_handler):
        """Empty topic is rejected."""
        is_valid, err = verticals_handler._validate_topic("")
        assert is_valid is False
        assert "cannot be empty" in err

    def test_validate_topic_rejects_too_long(self, verticals_handler):
        """Topic exceeding max length is rejected."""
        long_topic = "a" * (VerticalsHandler.MAX_TOPIC_LENGTH + 1)
        is_valid, err = verticals_handler._validate_topic(long_topic)
        assert is_valid is False
        assert "exceeds maximum length" in err

    def test_validate_agent_name_accepts_valid_name(self, verticals_handler):
        """Valid agent name passes validation."""
        is_valid, err = verticals_handler._validate_agent_name("myagent123")
        assert is_valid is True
        assert err in ("", None)

    def test_validate_agent_name_accepts_none(self, verticals_handler):
        """None agent name is valid (optional parameter)."""
        is_valid, err = verticals_handler._validate_agent_name(None)
        assert is_valid is True

    def test_validate_agent_name_rejects_too_long(self, verticals_handler):
        """Agent name exceeding max length is rejected."""
        long_name = "a" * (VerticalsHandler.MAX_AGENT_NAME_LENGTH + 1)
        is_valid, err = verticals_handler._validate_agent_name(long_name)
        assert is_valid is False
        assert "exceeds maximum length" in err

    def test_validate_additional_agents_accepts_valid_list(self, verticals_handler):
        """Valid agents list passes validation."""
        is_valid, err = verticals_handler._validate_additional_agents(["agent1", "agent2"])
        assert is_valid is True
        assert err == ""

    def test_validate_additional_agents_accepts_none(self, verticals_handler):
        """None agents list is valid (optional parameter)."""
        is_valid, err = verticals_handler._validate_additional_agents(None)
        assert is_valid is True

    def test_validate_additional_agents_rejects_non_list(self, verticals_handler):
        """Non-list agents is rejected."""
        is_valid, err = verticals_handler._validate_additional_agents("not a list")
        assert is_valid is False
        assert "must be a list" in err

    def test_validate_additional_agents_rejects_too_many(self, verticals_handler):
        """Too many additional agents is rejected."""
        too_many = ["agent"] * (VerticalsHandler.MAX_ADDITIONAL_AGENTS + 1)
        is_valid, err = verticals_handler._validate_additional_agents(too_many)
        assert is_valid is False
        assert "exceeds maximum count" in err


# -----------------------------------------------------------------------------
# Route Matching Tests (can_handle)
# -----------------------------------------------------------------------------


class TestVerticalsHandlerRouteMatching:
    """Tests for VerticalsHandler.can_handle() method."""

    def test_can_handle_list_verticals(self, verticals_handler):
        """Handler matches /api/verticals."""
        assert verticals_handler.can_handle("/api/verticals") is True

    def test_can_handle_suggest(self, verticals_handler):
        """Handler matches /api/verticals/suggest."""
        assert verticals_handler.can_handle("/api/verticals/suggest") is True

    def test_can_handle_specific_vertical(self, verticals_handler):
        """Handler matches /api/verticals/:id."""
        assert verticals_handler.can_handle("/api/verticals/healthcare") is True
        assert verticals_handler.can_handle("/api/verticals/finance") is True

    def test_can_handle_vertical_tools(self, verticals_handler):
        """Handler matches /api/verticals/:id/tools."""
        assert verticals_handler.can_handle("/api/verticals/healthcare/tools") is True

    def test_can_handle_vertical_compliance(self, verticals_handler):
        """Handler matches /api/verticals/:id/compliance."""
        assert verticals_handler.can_handle("/api/verticals/healthcare/compliance") is True

    def test_can_handle_vertical_debate(self, verticals_handler):
        """Handler matches /api/verticals/:id/debate."""
        assert verticals_handler.can_handle("/api/verticals/healthcare/debate") is True

    def test_can_handle_vertical_agent(self, verticals_handler):
        """Handler matches /api/verticals/:id/agent."""
        assert verticals_handler.can_handle("/api/verticals/healthcare/agent") is True

    def test_can_handle_vertical_config(self, verticals_handler):
        """Handler matches /api/verticals/:id/config."""
        assert verticals_handler.can_handle("/api/verticals/healthcare/config") is True

    def test_cannot_handle_unrelated(self, verticals_handler):
        """Handler does not match unrelated paths."""
        assert verticals_handler.can_handle("/api/debates") is False
        assert verticals_handler.can_handle("/api/agents") is False
        assert verticals_handler.can_handle("/other/verticals") is False

    def test_can_handle_with_version_prefix(self, verticals_handler):
        """Handler handles version prefix properly."""
        # The handler strips version prefix internally
        assert verticals_handler.can_handle("/api/v1/verticals") is True

    def test_can_handle_with_different_methods(self, verticals_handler):
        """Handler can_handle is method-agnostic."""
        assert verticals_handler.can_handle("/api/verticals", "GET") is True
        assert verticals_handler.can_handle("/api/verticals", "POST") is True
        assert verticals_handler.can_handle("/api/verticals", "PUT") is True


# -----------------------------------------------------------------------------
# List Verticals Tests
# -----------------------------------------------------------------------------


class TestVerticalsListEndpoint:
    """Tests for listing verticals."""

    def test_list_all_verticals(self, verticals_handler, mock_registry, mock_handler):
        """Test listing all verticals."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(
                verticals_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(verticals_handler, "check_permission"),
        ):
            mock_auth.return_value = MagicMock()

            result = verticals_handler._list_verticals({})

        assert result.status_code == 200

    def test_list_verticals_with_keyword_filter(
        self, verticals_handler, mock_registry, mock_handler
    ):
        """Test filtering verticals by keyword."""
        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = verticals_handler._list_verticals({"keyword": "medical"})

        assert result.status_code == 200

    def test_list_verticals_registry_unavailable(self, verticals_handler, mock_handler):
        """Test listing when registry is unavailable."""
        with patch.object(verticals_handler, "_get_registry", return_value=None):
            result = verticals_handler._list_verticals({})

        assert result.status_code == 503

    def test_list_verticals_with_keyword_too_long(self, verticals_handler, mock_registry):
        """Test listing with keyword exceeding max length."""
        long_keyword = "a" * (VerticalsHandler.MAX_KEYWORD_LENGTH + 1)
        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = verticals_handler._list_verticals({"keyword": long_keyword})

        assert result.status_code == 400
        assert "exceeds maximum length" in result.body.decode()

    def test_list_verticals_empty_result(self, verticals_handler, mock_registry):
        """Test listing when no verticals match."""
        mock_registry.list_all = MagicMock(return_value={})
        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = verticals_handler._list_verticals({})

        assert result.status_code == 200


# -----------------------------------------------------------------------------
# Get Specific Vertical Tests
# -----------------------------------------------------------------------------


class TestGetVerticalEndpoint:
    """Tests for getting a specific vertical."""

    def test_get_vertical_success(self, verticals_handler, mock_registry, mock_vertical_spec):
        """Test getting a specific vertical."""
        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = verticals_handler._get_vertical("healthcare")

        assert result.status_code == 200

    def test_get_vertical_not_found(self, verticals_handler, mock_registry):
        """Test getting a non-existent vertical."""
        mock_registry.get = MagicMock(return_value=None)

        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = verticals_handler._get_vertical("nonexistent")

        assert result.status_code == 404

    def test_get_vertical_registry_unavailable(self, verticals_handler):
        """Test getting vertical when registry unavailable."""
        with patch.object(verticals_handler, "_get_registry", return_value=None):
            result = verticals_handler._get_vertical("healthcare")

        assert result.status_code == 503

    def test_get_vertical_records_success_on_circuit_breaker(
        self, verticals_handler, mock_registry
    ):
        """Test that successful get records success on circuit breaker."""
        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            verticals_handler._get_vertical("healthcare")

        status = verticals_handler.get_circuit_breaker_status()
        assert status["failure_count"] == 0


# -----------------------------------------------------------------------------
# Vertical Tools Tests
# -----------------------------------------------------------------------------


class TestVerticalToolsEndpoint:
    """Tests for vertical tools endpoint."""

    def test_get_tools_success(self, verticals_handler, mock_registry, mock_vertical_config):
        """Test getting tools for a vertical."""
        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = verticals_handler._get_tools("healthcare")

        assert result.status_code == 200

    def test_get_tools_not_found(self, verticals_handler, mock_registry):
        """Test getting tools for non-existent vertical."""
        mock_registry.get_config = MagicMock(return_value=None)

        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = verticals_handler._get_tools("nonexistent")

        assert result.status_code == 404

    def test_get_tools_registry_unavailable(self, verticals_handler):
        """Test getting tools when registry unavailable."""
        with patch.object(verticals_handler, "_get_registry", return_value=None):
            result = verticals_handler._get_tools("healthcare")

        assert result.status_code == 503


# -----------------------------------------------------------------------------
# Vertical Compliance Tests
# -----------------------------------------------------------------------------


class TestVerticalComplianceEndpoint:
    """Tests for vertical compliance endpoint."""

    def test_get_compliance_success(self, verticals_handler, mock_registry, mock_vertical_config):
        """Test getting compliance frameworks."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch(
                "aragora.server.handlers.verticals.get_string_param",
                return_value=None,
            ),
        ):
            result = verticals_handler._get_compliance("healthcare", {})

        assert result.status_code == 200

    def test_get_compliance_with_level_filter(
        self, verticals_handler, mock_registry, mock_vertical_config
    ):
        """Test filtering compliance by level."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch("aragora.verticals.config.ComplianceLevel") as mock_level,
        ):
            mock_level.return_value = MagicMock()

            result = verticals_handler._get_compliance("healthcare", {"level": "strict"})

        # Should attempt to filter
        assert result is not None

    def test_get_compliance_not_found(self, verticals_handler, mock_registry):
        """Test getting compliance for non-existent vertical."""
        mock_registry.get_config = MagicMock(return_value=None)

        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = verticals_handler._get_compliance("nonexistent", {})

        assert result.status_code == 404


# -----------------------------------------------------------------------------
# Suggest Vertical Tests
# -----------------------------------------------------------------------------


class TestSuggestVerticalEndpoint:
    """Tests for vertical suggestion endpoint."""

    def test_suggest_vertical_success(self, verticals_handler, mock_registry):
        """Test suggesting a vertical for a task."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch(
                "aragora.server.handlers.verticals.get_string_param",
                return_value="I need to diagnose a patient",
            ),
        ):
            result = verticals_handler._suggest_vertical({"task": "I need to diagnose a patient"})

        assert result.status_code == 200

    def test_suggest_vertical_missing_task(self, verticals_handler, mock_registry):
        """Test suggestion without task parameter."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch(
                "aragora.server.handlers.verticals.get_string_param",
                return_value=None,
            ),
        ):
            result = verticals_handler._suggest_vertical({})

        assert result.status_code == 400

    def test_suggest_vertical_empty_task(self, verticals_handler, mock_registry):
        """Test suggestion with empty task."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch(
                "aragora.server.handlers.verticals.get_string_param",
                return_value="   ",
            ),
        ):
            result = verticals_handler._suggest_vertical({})

        assert result.status_code == 400

    def test_suggest_vertical_task_too_long(self, verticals_handler, mock_registry):
        """Test suggestion with task too long."""
        long_task = "a" * (VerticalsHandler.MAX_TASK_LENGTH + 1)
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch(
                "aragora.server.handlers.verticals.get_string_param",
                return_value=long_task,
            ),
        ):
            result = verticals_handler._suggest_vertical({})

        assert result.status_code == 400

    def test_suggest_vertical_no_match(self, verticals_handler, mock_registry):
        """Test suggestion when no vertical matches."""
        mock_registry.get_for_task = MagicMock(return_value=None)

        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch(
                "aragora.server.handlers.verticals.get_string_param",
                return_value="random task",
            ),
        ):
            result = verticals_handler._suggest_vertical({"task": "random task"})

        assert result.status_code == 200


# -----------------------------------------------------------------------------
# Create Agent Tests
# -----------------------------------------------------------------------------


class TestCreateAgentEndpoint:
    """Tests for creating specialist agents."""

    def test_create_agent_success(self, verticals_handler, mock_registry, mock_handler):
        """Test creating a specialist agent."""
        mock_handler.rfile.read = MagicMock(
            return_value=b'{"name": "test-agent", "model": "claude-opus-4"}'
        )
        mock_handler.headers = {
            "Content-Length": "50",
            "Content-Type": "application/json",
        }

        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(verticals_handler, "read_json_body", return_value={"name": "test-agent"}),
        ):
            result = verticals_handler._create_agent("healthcare", mock_handler)

        assert result.status_code == 200

    def test_create_agent_vertical_not_found(self, verticals_handler, mock_registry, mock_handler):
        """Test creating agent for non-existent vertical."""
        mock_registry.is_registered = MagicMock(return_value=False)

        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = verticals_handler._create_agent("nonexistent", mock_handler)

        assert result.status_code == 404

    def test_create_agent_invalid_body(self, verticals_handler, mock_registry, mock_handler):
        """Test creating agent with invalid request body."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(verticals_handler, "read_json_body", return_value=None),
        ):
            result = verticals_handler._create_agent("healthcare", mock_handler)

        assert result.status_code == 400

    def test_create_agent_name_too_long(self, verticals_handler, mock_registry, mock_handler):
        """Test creating agent with name too long."""
        long_name = "a" * (VerticalsHandler.MAX_AGENT_NAME_LENGTH + 1)
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(verticals_handler, "read_json_body", return_value={"name": long_name}),
        ):
            result = verticals_handler._create_agent("healthcare", mock_handler)

        assert result.status_code == 400

    def test_create_agent_default_name(self, verticals_handler, mock_registry, mock_handler):
        """Test creating agent with default name."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(verticals_handler, "read_json_body", return_value={}),
        ):
            result = verticals_handler._create_agent("healthcare", mock_handler)

        assert result.status_code == 200


# -----------------------------------------------------------------------------
# Create Debate Tests
# -----------------------------------------------------------------------------


class TestCreateDebateEndpoint:
    """Tests for creating vertical-specific debates."""

    @pytest.mark.asyncio
    async def test_create_debate_success(self, verticals_handler, mock_registry, mock_handler):
        """Test creating a vertical-specific debate."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(
                verticals_handler,
                "read_json_body",
                return_value={"topic": "Medical diagnosis discussion"},
            ),
            patch("aragora.debate.orchestrator.Arena") as mock_arena,
            patch("aragora.core.DebateProtocol"),
            patch("aragora.core.Environment"),
        ):
            mock_result = MagicMock()
            mock_result.debate_id = "debate-123"
            mock_result.consensus_reached = True
            mock_result.final_answer = "Diagnosis complete"
            mock_result.confidence = 0.85

            mock_arena_instance = MagicMock()
            mock_arena_instance.run = AsyncMock(return_value=mock_result)
            mock_arena.return_value = mock_arena_instance

            result = await verticals_handler._create_debate("healthcare", mock_handler)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_create_debate_missing_topic(
        self, verticals_handler, mock_registry, mock_handler
    ):
        """Test creating debate without topic."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(verticals_handler, "read_json_body", return_value={}),
        ):
            result = await verticals_handler._create_debate("healthcare", mock_handler)

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_debate_topic_too_long(
        self, verticals_handler, mock_registry, mock_handler
    ):
        """Test creating debate with topic too long."""
        long_topic = "a" * (VerticalsHandler.MAX_TOPIC_LENGTH + 1)
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(verticals_handler, "read_json_body", return_value={"topic": long_topic}),
        ):
            result = await verticals_handler._create_debate("healthcare", mock_handler)

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_debate_vertical_not_found(
        self, verticals_handler, mock_registry, mock_handler
    ):
        """Test creating debate for non-existent vertical."""
        mock_registry.is_registered = MagicMock(return_value=False)

        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = await verticals_handler._create_debate("nonexistent", mock_handler)

        assert result.status_code == 404

    @pytest.mark.asyncio
    async def test_create_debate_invalid_rounds(
        self, verticals_handler, mock_registry, mock_handler
    ):
        """Test creating debate with invalid rounds."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(
                verticals_handler,
                "read_json_body",
                return_value={"topic": "Test topic", "rounds": 100},
            ),
        ):
            result = await verticals_handler._create_debate("healthcare", mock_handler)

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_debate_rounds_too_low(
        self, verticals_handler, mock_registry, mock_handler
    ):
        """Test creating debate with rounds less than 1."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(
                verticals_handler,
                "read_json_body",
                return_value={"topic": "Test topic", "rounds": 0},
            ),
        ):
            result = await verticals_handler._create_debate("healthcare", mock_handler)

        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_create_debate_too_many_additional_agents(
        self, verticals_handler, mock_registry, mock_handler
    ):
        """Test creating debate with too many additional agents."""
        many_agents = ["agent"] * (VerticalsHandler.MAX_ADDITIONAL_AGENTS + 1)
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(
                verticals_handler,
                "read_json_body",
                return_value={"topic": "Test topic", "additional_agents": many_agents},
            ),
        ):
            result = await verticals_handler._create_debate("healthcare", mock_handler)

        assert result.status_code == 400


# -----------------------------------------------------------------------------
# Update Config Tests
# -----------------------------------------------------------------------------


class TestUpdateConfigEndpoint:
    """Tests for updating vertical configuration."""

    def test_update_config_tools(self, verticals_handler, mock_registry, mock_handler):
        """Test updating tools configuration."""
        update_data = {
            "tools": [
                {
                    "name": "new_tool",
                    "description": "A new tool",
                    "enabled": True,
                }
            ]
        }

        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(verticals_handler, "read_json_body", return_value=update_data),
            patch("aragora.verticals.config.ToolConfig") as mock_tool,
        ):
            mock_tool.return_value = MagicMock()

            result = verticals_handler._update_config("healthcare", mock_handler)

        assert result.status_code == 200

    def test_update_config_no_valid_fields(self, verticals_handler, mock_registry, mock_handler):
        """Test update with no valid fields."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(verticals_handler, "read_json_body", return_value={}),
        ):
            result = verticals_handler._update_config("healthcare", mock_handler)

        assert result.status_code == 400

    def test_update_config_vertical_not_found(self, verticals_handler, mock_registry, mock_handler):
        """Test update for non-existent vertical."""
        mock_registry.is_registered = MagicMock(return_value=False)

        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = verticals_handler._update_config("nonexistent", mock_handler)

        assert result.status_code == 404

    def test_update_config_tools_not_list(self, verticals_handler, mock_registry, mock_handler):
        """Test update with tools not being a list."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(verticals_handler, "read_json_body", return_value={"tools": "not a list"}),
        ):
            result = verticals_handler._update_config("healthcare", mock_handler)

        assert result.status_code == 400

    def test_update_config_too_many_tools(self, verticals_handler, mock_registry, mock_handler):
        """Test update with too many tools."""
        many_tools = [{"name": f"tool{i}"} for i in range(VerticalsHandler.MAX_TOOLS_COUNT + 1)]
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(verticals_handler, "read_json_body", return_value={"tools": many_tools}),
        ):
            result = verticals_handler._update_config("healthcare", mock_handler)

        assert result.status_code == 400

    def test_update_config_tool_name_too_long(self, verticals_handler, mock_registry, mock_handler):
        """Test update with tool name too long."""
        long_name = "a" * (VerticalsHandler.MAX_AGENT_NAME_LENGTH + 1)
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(
                verticals_handler,
                "read_json_body",
                return_value={"tools": [{"name": long_name}]},
            ),
        ):
            result = verticals_handler._update_config("healthcare", mock_handler)

        assert result.status_code == 400

    def test_update_config_compliance_frameworks(
        self, verticals_handler, mock_registry, mock_handler
    ):
        """Test updating compliance frameworks."""
        update_data = {
            "compliance_frameworks": [
                {
                    "framework": "SOC2",
                    "version": "2.0",
                    "level": "warning",
                }
            ]
        }

        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(verticals_handler, "read_json_body", return_value=update_data),
        ):
            result = verticals_handler._update_config("healthcare", mock_handler)

        assert result.status_code == 200

    def test_update_config_too_many_frameworks(
        self, verticals_handler, mock_registry, mock_handler
    ):
        """Test update with too many compliance frameworks."""
        many_frameworks = [
            {"framework": f"fw{i}"} for i in range(VerticalsHandler.MAX_FRAMEWORKS_COUNT + 1)
        ]
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(
                verticals_handler,
                "read_json_body",
                return_value={"compliance_frameworks": many_frameworks},
            ),
        ):
            result = verticals_handler._update_config("healthcare", mock_handler)

        assert result.status_code == 400

    def test_update_config_model_config(self, verticals_handler, mock_registry, mock_handler):
        """Test updating model configuration."""
        update_data = {
            "model_config": {
                "temperature": 0.5,
                "max_tokens": 8192,
            }
        }

        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(verticals_handler, "read_json_body", return_value=update_data),
        ):
            result = verticals_handler._update_config("healthcare", mock_handler)

        assert result.status_code == 200

    def test_update_config_invalid_temperature(
        self, verticals_handler, mock_registry, mock_handler
    ):
        """Test update with invalid temperature."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(
                verticals_handler,
                "read_json_body",
                return_value={"model_config": {"temperature": 5.0}},
            ),
        ):
            result = verticals_handler._update_config("healthcare", mock_handler)

        assert result.status_code == 400

    def test_update_config_invalid_max_tokens(self, verticals_handler, mock_registry, mock_handler):
        """Test update with invalid max_tokens."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(
                verticals_handler,
                "read_json_body",
                return_value={"model_config": {"max_tokens": -1}},
            ),
        ):
            result = verticals_handler._update_config("healthcare", mock_handler)

        assert result.status_code == 400

    def test_update_config_max_tokens_too_high(
        self, verticals_handler, mock_registry, mock_handler
    ):
        """Test update with max_tokens too high."""
        with (
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch.object(
                verticals_handler,
                "read_json_body",
                return_value={"model_config": {"max_tokens": 500000}},
            ),
        ):
            result = verticals_handler._update_config("healthcare", mock_handler)

        assert result.status_code == 400


# -----------------------------------------------------------------------------
# Error Handling Tests
# -----------------------------------------------------------------------------


class TestVerticalsErrorHandling:
    """Tests for error handling in verticals handler."""

    def test_handles_registry_import_error(self, verticals_handler):
        """Test handling of registry import error."""
        with patch.object(verticals_handler, "_get_registry", return_value=None):
            result = verticals_handler._list_verticals({})

        assert result.status_code == 503
        assert "not available" in result.body.decode().lower()

    def test_handles_data_error_in_list(self, verticals_handler, mock_registry):
        """Test handling of data errors in list."""
        mock_registry.list_all = MagicMock(side_effect=KeyError("missing key"))

        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = verticals_handler._list_verticals({})

        assert result.status_code == 400

    def test_handles_unexpected_error(self, verticals_handler, mock_registry):
        """Test handling of unexpected errors."""
        mock_registry.list_all = MagicMock(side_effect=RuntimeError("Unexpected error"))

        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = verticals_handler._list_verticals({})

        assert result.status_code == 500

    def test_error_records_failure_on_circuit_breaker(self, verticals_handler, mock_registry):
        """Test that errors record failure on circuit breaker."""
        mock_registry.list_all = MagicMock(side_effect=RuntimeError("Error"))

        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            verticals_handler._list_verticals({})

        status = verticals_handler.get_circuit_breaker_status()
        assert status["failure_count"] > 0


# -----------------------------------------------------------------------------
# Path Validation Tests
# -----------------------------------------------------------------------------


class TestVerticalsPathValidation:
    """Tests for path segment validation."""

    @pytest.mark.asyncio
    async def test_rejects_invalid_vertical_id(self, verticals_handler, mock_handler):
        """Test rejection of invalid vertical ID."""
        with (
            patch.object(
                verticals_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(verticals_handler, "check_permission"),
        ):
            mock_auth.return_value = MagicMock()

            # Attempt with malicious ID
            result = await verticals_handler.handle(
                "/api/verticals/../../../etc/passwd/tools",
                {},
                mock_handler,
            )

        # Should either return None (not matched) or 400 (bad request)
        if result is not None:
            assert result.status_code in [400, 404]

    @pytest.mark.asyncio
    async def test_rejects_special_characters_in_id(self, verticals_handler, mock_handler):
        """Test rejection of special characters in vertical ID."""
        with (
            patch.object(
                verticals_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(verticals_handler, "check_permission"),
        ):
            mock_auth.return_value = MagicMock()

            result = await verticals_handler.handle(
                "/api/verticals/<script>alert(1)</script>/tools",
                {},
                mock_handler,
            )

        if result is not None:
            assert result.status_code == 400


# -----------------------------------------------------------------------------
# RBAC Tests
# -----------------------------------------------------------------------------


class TestVerticalsRBAC:
    """Tests for RBAC permission checks."""

    @pytest.mark.asyncio
    async def test_read_requires_read_permission(self, verticals_handler, mock_handler):
        """Test that read operations require read permission."""
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch.object(
            verticals_handler,
            "get_auth_context",
            new_callable=AsyncMock,
            side_effect=UnauthorizedError("Not authenticated"),
        ):
            result = await verticals_handler.handle("/api/verticals", {}, mock_handler)

        assert result is not None
        assert result.status_code == 401

    @pytest.mark.asyncio
    async def test_write_requires_update_permission(self, verticals_handler, mock_handler_post):
        """Test that write operations require update permission."""
        from aragora.server.handlers.utils.auth import ForbiddenError

        with (
            patch.object(
                verticals_handler,
                "get_auth_context",
                new_callable=AsyncMock,
            ) as mock_auth,
            patch.object(
                verticals_handler,
                "check_permission",
                side_effect=ForbiddenError("Permission denied"),
            ),
        ):
            mock_auth.return_value = MagicMock()

            result = await verticals_handler.handle(
                "/api/verticals/healthcare/config",
                {},
                mock_handler_post,
            )

        assert result is not None
        assert result.status_code == 403

    @pytest.mark.asyncio
    async def test_put_requires_update_permission(self, verticals_handler, mock_handler_put):
        """Test that PUT operations require update permission."""
        from aragora.server.handlers.utils.auth import ForbiddenError

        with (
            patch.object(
                verticals_handler,
                "get_auth_context",
                new_callable=AsyncMock,
            ) as mock_auth,
            patch.object(
                verticals_handler,
                "check_permission",
                side_effect=ForbiddenError("Permission denied"),
            ),
        ):
            mock_auth.return_value = MagicMock()

            result = await verticals_handler.handle(
                "/api/verticals/healthcare/config",
                {},
                mock_handler_put,
            )

        assert result is not None
        assert result.status_code == 403


# -----------------------------------------------------------------------------
# Handler Constants Tests
# -----------------------------------------------------------------------------


class TestHandlerConstants:
    """Tests for handler constants and configuration."""

    def test_max_topic_length_is_reasonable(self):
        """MAX_TOPIC_LENGTH should be reasonable."""
        assert VerticalsHandler.MAX_TOPIC_LENGTH > 0
        assert VerticalsHandler.MAX_TOPIC_LENGTH <= 100000

    def test_max_agent_name_length_is_reasonable(self):
        """MAX_AGENT_NAME_LENGTH should be reasonable."""
        assert VerticalsHandler.MAX_AGENT_NAME_LENGTH > 0
        assert VerticalsHandler.MAX_AGENT_NAME_LENGTH <= 500

    def test_max_additional_agents_is_reasonable(self):
        """MAX_ADDITIONAL_AGENTS should be reasonable."""
        assert VerticalsHandler.MAX_ADDITIONAL_AGENTS > 0
        assert VerticalsHandler.MAX_ADDITIONAL_AGENTS <= 100

    def test_max_tools_count_is_reasonable(self):
        """MAX_TOOLS_COUNT should be reasonable."""
        assert VerticalsHandler.MAX_TOOLS_COUNT > 0
        assert VerticalsHandler.MAX_TOOLS_COUNT <= 500

    def test_max_frameworks_count_is_reasonable(self):
        """MAX_FRAMEWORKS_COUNT should be reasonable."""
        assert VerticalsHandler.MAX_FRAMEWORKS_COUNT > 0
        assert VerticalsHandler.MAX_FRAMEWORKS_COUNT <= 100

    def test_routes_are_defined(self, verticals_handler):
        """ROUTES should be defined and non-empty."""
        assert hasattr(verticals_handler, "ROUTES")
        assert len(VerticalsHandler.ROUTES) > 0

    def test_resource_type_is_defined(self, verticals_handler):
        """RESOURCE_TYPE should be defined."""
        assert hasattr(verticals_handler, "RESOURCE_TYPE")
        assert VerticalsHandler.RESOURCE_TYPE == "verticals"


# -----------------------------------------------------------------------------
# Full Handle Method Integration Tests
# -----------------------------------------------------------------------------


class TestHandleMethodIntegration:
    """Integration tests for the full handle method."""

    @pytest.mark.asyncio
    async def test_handle_list_verticals(self, verticals_handler, mock_handler, mock_registry):
        """Test handle method for listing verticals."""
        with (
            patch.object(
                verticals_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(verticals_handler, "check_permission"),
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
        ):
            mock_auth.return_value = MagicMock()

            result = await verticals_handler.handle("/api/verticals", {}, mock_handler)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_get_vertical(self, verticals_handler, mock_handler, mock_registry):
        """Test handle method for getting a specific vertical."""
        with (
            patch.object(
                verticals_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(verticals_handler, "check_permission"),
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
        ):
            mock_auth.return_value = MagicMock()

            result = await verticals_handler.handle("/api/verticals/healthcare", {}, mock_handler)

        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unmatched_path(self, verticals_handler, mock_handler):
        """Test handle returns None for paths it doesn't handle."""
        with (
            patch.object(
                verticals_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(verticals_handler, "check_permission"),
        ):
            mock_auth.return_value = MagicMock()

            # Path that doesn't match any route pattern
            result = await verticals_handler.handle(
                "/api/verticals/healthcare/unknown/endpoint", {}, mock_handler
            )

        assert result is None

    @pytest.mark.asyncio
    async def test_handle_suggest_vertical(self, verticals_handler, mock_handler, mock_registry):
        """Test handle method for suggesting verticals."""
        with (
            patch.object(
                verticals_handler, "get_auth_context", new_callable=AsyncMock
            ) as mock_auth,
            patch.object(verticals_handler, "check_permission"),
            patch.object(verticals_handler, "_get_registry", return_value=mock_registry),
            patch(
                "aragora.server.handlers.verticals.get_string_param",
                return_value="test task",
            ),
        ):
            mock_auth.return_value = MagicMock()

            result = await verticals_handler.handle(
                "/api/verticals/suggest", {"task": "test task"}, mock_handler
            )

        assert result.status_code == 200
