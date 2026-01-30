"""
Tests for VerticalsHandler - Industry vertical specialist endpoints.

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
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


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
    from aragora.server.handlers.verticals import VerticalsHandler

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
            patch("aragora.server.handlers.verticals.ComplianceLevel") as mock_level,
        ):
            mock_level.return_value = MagicMock()

            result = verticals_handler._get_compliance("healthcare", {"level": "strict"})

        # Should attempt to filter
        assert result is not None


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
        mock_handler.headers = {"Content-Length": "50", "Content-Type": "application/json"}

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
            patch("aragora.server.handlers.verticals.Arena") as mock_arena,
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
    async def test_create_debate_vertical_not_found(
        self, verticals_handler, mock_registry, mock_handler
    ):
        """Test creating debate for non-existent vertical."""
        mock_registry.is_registered = MagicMock(return_value=False)

        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = await verticals_handler._create_debate("nonexistent", mock_handler)

        assert result.status_code == 404


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
            patch("aragora.server.handlers.verticals.VerticalTool") as mock_tool,
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
        mock_registry.list_all = MagicMock(side_effect=Exception("Unexpected error"))

        with patch.object(verticals_handler, "_get_registry", return_value=mock_registry):
            result = verticals_handler._list_verticals({})

        assert result.status_code == 500


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
