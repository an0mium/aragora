"""
Integration tests for API endpoints not used by frontend.

These tests ensure backend endpoints work correctly even when
the frontend doesn't currently call them. Coverage priorities:
1. Authentication (/api/auth/*)
2. Batch operations (/api/debates/batch)
3. Evidence collection (/api/evidence/*)
4. Plugins (/api/plugins/*)
5. Training exports (/api/training/export/*)
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json
import io


# Test fixtures
@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler for testing."""
    handler = MagicMock()
    handler.headers = {"Content-Type": "application/json"}
    handler.wfile = io.BytesIO()
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler._add_cors_headers = MagicMock()
    handler._add_security_headers = MagicMock()
    return handler


@pytest.fixture
def handler_ctx():
    """Create a mock handler context."""
    return {
        "storage": MagicMock(),
        "elo_system": MagicMock(),
        "nomic_dir": None,
        "debate_embeddings": MagicMock(),
        "critique_store": MagicMock(),
        "document_store": MagicMock(),
        "persona_manager": MagicMock(),
        "position_ledger": MagicMock(),
        "user_store": MagicMock(),
    }


class TestAuthEndpoints:
    """Tests for authentication endpoints."""

    def test_auth_handler_exists(self):
        """Verify AuthHandler can be imported."""
        from aragora.server.handlers.auth import AuthHandler

        assert AuthHandler is not None

    def test_auth_handler_routes(self, handler_ctx):
        """Verify AuthHandler has expected routes."""
        from aragora.server.handlers.auth import AuthHandler

        handler = AuthHandler(handler_ctx)

        expected_routes = [
            "/api/auth/login",
            "/api/auth/register",
            "/api/auth/logout",
            "/api/auth/me",
        ]

        for route in expected_routes:
            assert handler.can_handle(route), f"AuthHandler should handle {route}"

    def test_auth_login_requires_credentials(self, handler_ctx, mock_http_handler):
        """Test login endpoint requires email and password."""
        from aragora.server.handlers.auth import AuthHandler

        handler = AuthHandler(handler_ctx)

        # Mock reading empty body
        mock_http_handler.rfile = io.BytesIO(b"{}")
        mock_http_handler.headers = {"Content-Length": "2"}

        result = handler.handle_post("/api/auth/login", {}, mock_http_handler)

        # Should return error for missing credentials
        if result:
            assert result.status_code in [400, 401, 422]

    def test_auth_register_validates_email(self, handler_ctx, mock_http_handler):
        """Test register endpoint validates email format."""
        from aragora.server.handlers.auth import AuthHandler

        handler = AuthHandler(handler_ctx)

        body = json.dumps({"email": "invalid-email", "password": "test123"})
        mock_http_handler.rfile = io.BytesIO(body.encode())
        mock_http_handler.headers = {"Content-Length": str(len(body))}

        result = handler.handle_post("/api/auth/register", {}, mock_http_handler)

        # Should validate email format
        if result:
            assert result.status_code in [400, 422]


class TestDebateBatchEndpoints:
    """Tests for batch debate submission endpoints."""

    def test_debates_batch_mixin_exists(self):
        """Verify debates batch functionality exists as a mixin."""
        from aragora.server.handlers.debates_batch import BatchOperationsMixin

        assert BatchOperationsMixin is not None

    def test_debates_handler_has_batch_methods(self, handler_ctx):
        """Verify DebatesHandler has batch operations via mixin."""
        from aragora.server.handlers.debates import DebatesHandler

        handler = DebatesHandler(handler_ctx)

        # Batch methods are mixed in from BatchOperationsMixin
        assert hasattr(handler, "_submit_batch")
        assert hasattr(handler, "_get_batch_status")

    def test_debates_handler_handles_batch_route(self, handler_ctx):
        """Test DebatesHandler can handle batch routes."""
        from aragora.server.handlers.debates import DebatesHandler

        handler = DebatesHandler(handler_ctx)

        # DebatesHandler handles all /api/debates/* routes including batch
        assert handler.can_handle("/api/debates/batch")


class TestEvidenceEndpoints:
    """Tests for evidence collection endpoints."""

    def test_evidence_handler_exists(self):
        """Verify EvidenceHandler can be imported."""
        from aragora.server.handlers.evidence import EvidenceHandler

        assert EvidenceHandler is not None

    def test_evidence_routes(self, handler_ctx):
        """Verify evidence routes are handled."""
        from aragora.server.handlers.evidence import EvidenceHandler

        handler = EvidenceHandler(handler_ctx)

        expected_routes = [
            "/api/evidence",
            "/api/evidence/collect",
            "/api/evidence/search",
        ]

        for route in expected_routes:
            assert handler.can_handle(route), f"EvidenceHandler should handle {route}"

    def test_evidence_search_is_post_endpoint(self, handler_ctx):
        """Test evidence search is available as POST endpoint."""
        from aragora.server.handlers.evidence import EvidenceHandler

        handler = EvidenceHandler(handler_ctx)

        # Verify the POST search endpoint is defined in routes
        assert "POST /api/evidence/search" in handler.routes


class TestPluginsEndpoints:
    """Tests for plugins management endpoints."""

    def test_plugins_handler_exists(self):
        """Verify PluginsHandler can be imported."""
        from aragora.server.handlers.plugins import PluginsHandler

        assert PluginsHandler is not None

    def test_plugins_routes(self, handler_ctx):
        """Verify plugins routes are handled."""
        from aragora.server.handlers.plugins import PluginsHandler

        handler = PluginsHandler(handler_ctx)

        expected_routes = [
            "/api/plugins",
            "/api/plugins/installed",
        ]

        for route in expected_routes:
            assert handler.can_handle(route), f"PluginsHandler should handle {route}"

    def test_plugins_list_returns_array(self, handler_ctx):
        """Test plugins list endpoint returns array."""
        from aragora.server.handlers.plugins import PluginsHandler

        handler = PluginsHandler(handler_ctx)

        result = handler.handle("/api/plugins", {}, MagicMock())

        if result and result.status_code == 200:
            body = json.loads(result.body.decode())
            assert "plugins" in body or isinstance(body, list)


class TestTrainingExportEndpoints:
    """Tests for training data export endpoints."""

    def test_training_handler_exists(self):
        """Verify TrainingHandler can be imported."""
        from aragora.server.handlers.training import TrainingHandler

        assert TrainingHandler is not None

    def test_training_routes(self, handler_ctx):
        """Verify training export routes are handled."""
        from aragora.server.handlers.training import TrainingHandler

        handler = TrainingHandler(handler_ctx)

        expected_routes = [
            "/api/training/export/sft",
            "/api/training/export/dpo",
            "/api/training/formats",
        ]

        for route in expected_routes:
            assert handler.can_handle(route), f"TrainingHandler should handle {route}"

    def test_training_formats_returns_list(self, handler_ctx):
        """Test training formats endpoint returns supported formats."""
        from aragora.server.handlers.training import TrainingHandler

        handler = TrainingHandler(handler_ctx)

        result = handler.handle("/api/training/formats", {}, MagicMock())

        if result and result.status_code == 200:
            body = json.loads(result.body.decode())
            assert "formats" in body or isinstance(body, list)


class TestGauntletEndpoints:
    """Tests for Gauntlet stress-testing endpoints."""

    def test_gauntlet_handler_exists(self):
        """Verify GauntletHandler can be imported."""
        from aragora.server.handlers.gauntlet import GauntletHandler

        assert GauntletHandler is not None

    def test_gauntlet_routes(self, handler_ctx):
        """Verify gauntlet routes are handled."""
        from aragora.server.handlers.gauntlet import GauntletHandler

        handler = GauntletHandler(handler_ctx)

        expected_routes = [
            "/api/gauntlet/run",
            "/api/gauntlet/results",
            "/api/gauntlet/personas",
        ]

        for route in expected_routes:
            assert handler.can_handle(route), f"GauntletHandler should handle {route}"


class TestGraphDebatesEndpoints:
    """Tests for graph/matrix debate endpoints."""

    def test_graph_debates_handler_exists(self):
        """Verify GraphDebatesHandler can be imported."""
        from aragora.server.handlers.graph_debates import GraphDebatesHandler

        assert GraphDebatesHandler is not None

    def test_matrix_debates_handler_exists(self):
        """Verify MatrixDebatesHandler can be imported."""
        from aragora.server.handlers.matrix_debates import MatrixDebatesHandler

        assert MatrixDebatesHandler is not None

    def test_graph_routes(self, handler_ctx):
        """Verify graph debate routes are handled."""
        from aragora.server.handlers.graph_debates import GraphDebatesHandler

        handler = GraphDebatesHandler(handler_ctx)

        assert handler.can_handle("/api/debates/graph")

    def test_matrix_routes(self, handler_ctx):
        """Verify matrix debate routes are handled."""
        from aragora.server.handlers.matrix_debates import MatrixDebatesHandler

        handler = MatrixDebatesHandler(handler_ctx)

        assert handler.can_handle("/api/debates/matrix")


class TestMetricsEndpoints:
    """Tests for metrics/monitoring endpoints."""

    def test_metrics_handler_exists(self):
        """Verify MetricsHandler can be imported."""
        from aragora.server.handlers.metrics import MetricsHandler

        assert MetricsHandler is not None

    def test_metrics_routes(self, handler_ctx):
        """Verify metrics routes are handled."""
        from aragora.server.handlers.metrics import MetricsHandler

        handler = MetricsHandler(handler_ctx)

        expected_routes = [
            "/api/metrics",
        ]

        for route in expected_routes:
            assert handler.can_handle(route), f"MetricsHandler should handle {route}"


class TestIntrospectionEndpoints:
    """Tests for agent introspection endpoints."""

    def test_introspection_handler_exists(self):
        """Verify IntrospectionHandler can be imported."""
        from aragora.server.handlers.introspection import IntrospectionHandler

        assert IntrospectionHandler is not None

    def test_introspection_routes(self, handler_ctx):
        """Verify introspection routes are handled."""
        from aragora.server.handlers.introspection import IntrospectionHandler

        handler = IntrospectionHandler(handler_ctx)

        expected_routes = [
            "/api/introspection/agents",
            "/api/introspection/all",
        ]

        for route in expected_routes:
            assert handler.can_handle(route), f"IntrospectionHandler should handle {route}"


class TestLearningEndpoints:
    """Tests for learning/analytics endpoints."""

    def test_learning_handler_exists(self):
        """Verify LearningHandler can be imported."""
        from aragora.server.handlers.learning import LearningHandler

        assert LearningHandler is not None

    def test_learning_routes(self, handler_ctx):
        """Verify learning routes are handled."""
        from aragora.server.handlers.learning import LearningHandler

        handler = LearningHandler(handler_ctx)

        expected_routes = [
            "/api/learning/patterns",
            "/api/learning/insights",
        ]

        for route in expected_routes:
            assert handler.can_handle(route), f"LearningHandler should handle {route}"


class TestGenesisEndpoints:
    """Tests for genesis/genome evolution endpoints."""

    def test_genesis_handler_exists(self):
        """Verify GenesisHandler can be imported."""
        from aragora.server.handlers.genesis import GenesisHandler

        assert GenesisHandler is not None

    def test_genesis_routes(self, handler_ctx):
        """Verify genesis routes are handled."""
        from aragora.server.handlers.genesis import GenesisHandler

        handler = GenesisHandler(handler_ctx)

        expected_routes = [
            "/api/genesis/stats",
            "/api/genesis/events",
            "/api/genesis/genomes",
        ]

        for route in expected_routes:
            assert handler.can_handle(route), f"GenesisHandler should handle {route}"


class TestCalibrationEndpoints:
    """Tests for agent calibration endpoints."""

    def test_calibration_handler_exists(self):
        """Verify CalibrationHandler can be imported."""
        from aragora.server.handlers.calibration import CalibrationHandler

        assert CalibrationHandler is not None

    def test_calibration_routes(self, handler_ctx):
        """Verify calibration routes are handled."""
        from aragora.server.handlers.calibration import CalibrationHandler

        handler = CalibrationHandler(handler_ctx)

        expected_routes = [
            "/api/calibration/leaderboard",
            "/api/calibration/visualization",
        ]

        for route in expected_routes:
            assert handler.can_handle(route), f"CalibrationHandler should handle {route}"

    def test_calibration_agent_routes(self, handler_ctx):
        """Verify agent-specific calibration routes are handled."""
        from aragora.server.handlers.calibration import CalibrationHandler

        handler = CalibrationHandler(handler_ctx)

        # Agent-specific routes use pattern matching
        assert handler.can_handle("/api/agent/claude/calibration-curve")
        assert handler.can_handle("/api/agent/gpt4/calibration-summary")


class TestVerificationEndpoints:
    """Tests for formal verification endpoints."""

    def test_verification_handler_exists(self):
        """Verify VerificationHandler can be imported."""
        from aragora.server.handlers.verification import VerificationHandler

        assert VerificationHandler is not None

    def test_formal_verification_handler_exists(self):
        """Verify FormalVerificationHandler can be imported."""
        from aragora.server.handlers.formal_verification import FormalVerificationHandler

        assert FormalVerificationHandler is not None

    def test_verification_routes(self, handler_ctx):
        """Verify verification routes are handled."""
        from aragora.server.handlers.verification import VerificationHandler

        handler = VerificationHandler(handler_ctx)

        assert handler.can_handle("/api/verification/status")


# Run quick smoke tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
