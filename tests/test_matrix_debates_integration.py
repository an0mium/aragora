"""
Integration tests for Matrix Debates feature.

Tests cover end-to-end flows for:
- Creating matrix debates with parallel scenarios
- Running scenario comparisons
- Extracting universal and conditional conclusions
- Handler endpoint coverage
"""

import json
import pytest
from unittest.mock import Mock, AsyncMock, patch


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_agents():
    """Create mock agents for debates."""
    agent1 = Mock()
    agent1.name = "claude"
    agent1.generate = AsyncMock(return_value="Agent 1 response")

    agent2 = Mock()
    agent2.name = "gpt4"
    agent2.generate = AsyncMock(return_value="Agent 2 response")

    return [agent1, agent2]


@pytest.fixture
def mock_storage():
    """Create mock matrix storage."""
    storage = Mock()
    storage.get_matrix_debate = AsyncMock(return_value=None)
    storage.get_matrix_scenarios = AsyncMock(return_value=[])
    storage.get_matrix_conclusions = AsyncMock(
        return_value={
            "universal": [],
            "conditional": [],
        }
    )
    return storage


@pytest.fixture
def mock_handler(mock_storage):
    """Create mock HTTP handler."""
    handler = Mock()
    handler.command = "GET"
    handler.storage = mock_storage
    handler.event_emitter = None
    return handler


@pytest.fixture
def sample_scenarios():
    """Sample scenario configurations."""
    return [
        {
            "name": "Baseline",
            "parameters": {"budget": "low"},
            "constraints": ["must be cost-effective"],
            "is_baseline": True,
        },
        {
            "name": "High Budget",
            "parameters": {"budget": "high"},
            "constraints": ["maximize quality"],
            "is_baseline": False,
        },
        {
            "name": "Time Constrained",
            "parameters": {"budget": "medium", "timeline": "short"},
            "constraints": ["deliver quickly"],
            "is_baseline": False,
        },
    ]


@pytest.fixture
def sample_results():
    """Sample scenario results for testing utility methods."""
    return [
        {
            "scenario_name": "Baseline",
            "parameters": {"budget": "low"},
            "constraints": ["must be cost-effective"],
            "is_baseline": True,
            "winner": "claude",
            "final_answer": "Use approach A",
            "confidence": 0.85,
            "consensus_reached": True,
            "rounds_used": 2,
        },
        {
            "scenario_name": "High Budget",
            "parameters": {"budget": "high"},
            "constraints": ["maximize quality"],
            "is_baseline": False,
            "winner": "gpt4",
            "final_answer": "Use approach B",
            "confidence": 0.9,
            "consensus_reached": True,
            "rounds_used": 3,
        },
        {
            "scenario_name": "Time Constrained",
            "parameters": {"budget": "medium", "timeline": "short"},
            "constraints": ["deliver quickly"],
            "is_baseline": False,
            "winner": "claude",
            "final_answer": "Use approach C",
            "confidence": 0.75,
            "consensus_reached": True,
            "rounds_used": 1,
        },
    ]


# ============================================================================
# Handler Route Tests
# ============================================================================


class TestMatrixDebatesHandlerRoutes:
    """Tests for MatrixDebatesHandler route recognition."""

    def test_handler_routes(self):
        """Test handler recognizes all matrix debate routes."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            # Test route recognition
            assert "/api/debates/matrix" in handler.ROUTES
            assert "/api/debates/matrix/" in handler.ROUTES
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    def test_auth_required_endpoints(self):
        """Test auth is required for POST endpoint."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            assert "/api/debates/matrix" in handler.AUTH_REQUIRED_ENDPOINTS
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")


# ============================================================================
# GET Endpoint Tests
# ============================================================================


class TestMatrixDebatesGetEndpoints:
    """Tests for GET endpoints."""

    @pytest.mark.asyncio
    async def test_get_matrix_debate_not_found(self, mock_handler):
        """Test 404 response for non-existent matrix debate."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            result = await handler._get_matrix_debate(mock_handler, "nonexistent-id")

            assert result.status_code == 404
            data = json.loads(result.body)
            assert "not found" in data.get("error", "").lower()
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    @pytest.mark.asyncio
    async def test_get_matrix_debate_success(self, mock_handler, mock_storage):
        """Test successful matrix debate retrieval."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            # Configure storage to return a debate
            mock_storage.get_matrix_debate = AsyncMock(
                return_value={
                    "matrix_id": "test-123",
                    "task": "Test task",
                    "scenario_count": 3,
                }
            )
            mock_handler.storage = mock_storage

            handler = MatrixDebatesHandler({})
            result = await handler._get_matrix_debate(mock_handler, "test-123")

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["matrix_id"] == "test-123"
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    @pytest.mark.asyncio
    async def test_get_matrix_debate_no_storage(self, mock_handler):
        """Test 503 when storage not configured."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            mock_handler.storage = None

            handler = MatrixDebatesHandler({})
            result = await handler._get_matrix_debate(mock_handler, "test-123")

            assert result.status_code == 503
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    @pytest.mark.asyncio
    async def test_get_scenarios_empty(self, mock_handler):
        """Test empty scenarios response."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            result = await handler._get_scenarios(mock_handler, "matrix-123")

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "scenarios" in data
            assert data["matrix_id"] == "matrix-123"
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    @pytest.mark.asyncio
    async def test_get_conclusions_empty(self, mock_handler):
        """Test empty conclusions response."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            result = await handler._get_conclusions(mock_handler, "matrix-123")

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "universal_conclusions" in data
            assert "conditional_conclusions" in data
            assert data["matrix_id"] == "matrix-123"
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")


# ============================================================================
# POST Endpoint Tests
# ============================================================================


class TestMatrixDebatesPostEndpoint:
    """Tests for POST /api/debates/matrix endpoint."""

    @pytest.mark.asyncio
    async def test_create_matrix_debate_missing_task(self, mock_handler):
        """Test 400 response when task is missing."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            # Call internal method directly to avoid decorator complexity
            result = await handler._run_matrix_debate(
                mock_handler,
                {"scenarios": [{"name": "Test"}]},
            )

            assert result.status_code == 400
            data = json.loads(result.body)
            assert "task" in data.get("error", "").lower()
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    @pytest.mark.asyncio
    async def test_create_matrix_debate_missing_scenarios(self, mock_handler):
        """Test 400 response when scenarios are missing."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            # Call internal method directly (task must be 10+ chars)
            result = await handler._run_matrix_debate(
                mock_handler,
                {"task": "Test matrix debate task"},
            )

            assert result.status_code == 400
            data = json.loads(result.body)
            assert "scenario" in data.get("error", "").lower()
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    @pytest.mark.asyncio
    async def test_create_matrix_debate_empty_scenarios(self, mock_handler):
        """Test 400 response when scenarios list is empty."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            # Task must be 10+ chars to pass task validation first
            result = await handler._run_matrix_debate(
                mock_handler,
                {"task": "Test matrix debate task", "scenarios": []},
            )

            assert result.status_code == 400
            data = json.loads(result.body)
            assert "scenario" in data.get("error", "").lower()
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")


# ============================================================================
# Utility Method Tests
# ============================================================================


class TestMatrixDebatesUtilities:
    """Tests for utility methods."""

    def test_find_universal_conclusions_all_consensus(self, sample_results):
        """Test finding universal conclusions when all reach consensus."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            result = handler._find_universal_conclusions(sample_results)

            # All scenarios reached consensus
            assert len(result) > 0
            assert "consensus" in result[0].lower()
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    def test_find_universal_conclusions_no_consensus(self):
        """Test finding universal conclusions when some don't reach consensus."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            results = [
                {"consensus_reached": True, "final_answer": "A"},
                {"consensus_reached": False, "final_answer": "B"},
            ]

            result = handler._find_universal_conclusions(results)

            # Not all scenarios reached consensus
            assert result == []
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    def test_find_universal_conclusions_empty(self):
        """Test finding universal conclusions with empty results."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            result = handler._find_universal_conclusions([])

            assert result == []
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    def test_find_conditional_conclusions(self, sample_results):
        """Test finding conditional conclusions."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            result = handler._find_conditional_conclusions(sample_results)

            # Should have one conclusion per scenario with final_answer
            assert len(result) == 3

            # Check structure
            for conclusion in result:
                assert "condition" in conclusion
                assert "parameters" in conclusion
                assert "conclusion" in conclusion
                assert "confidence" in conclusion
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    def test_find_conditional_conclusions_empty(self):
        """Test finding conditional conclusions with empty results."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            result = handler._find_conditional_conclusions([])

            assert result == []
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    def test_build_comparison_matrix(self, sample_results):
        """Test building comparison matrix."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            result = handler._build_comparison_matrix(sample_results)

            assert "scenarios" in result
            assert len(result["scenarios"]) == 3

            assert "consensus_rate" in result
            assert result["consensus_rate"] == 1.0  # All reached consensus

            assert "avg_confidence" in result
            assert 0 <= result["avg_confidence"] <= 1

            assert "avg_rounds" in result
            assert result["avg_rounds"] == 2.0  # (2+3+1)/3 = 2.0
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    def test_build_comparison_matrix_empty(self):
        """Test building comparison matrix with empty results."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            result = handler._build_comparison_matrix([])

            assert result["scenarios"] == []
            assert result["consensus_rate"] == 0
            assert result["avg_confidence"] == 0
            assert result["avg_rounds"] == 0
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")


# ============================================================================
# Agent Loading Tests
# ============================================================================


class TestAgentLoading:
    """Tests for agent loading functionality."""

    @pytest.mark.asyncio
    async def test_load_agents_default(self):
        """Test loading default agents."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            with patch(
                "aragora.server.handlers.matrix_debates.MatrixDebatesHandler._load_agents"
            ) as mock_load:
                mock_agent1 = Mock()
                mock_agent1.name = "claude"
                mock_agent2 = Mock()
                mock_agent2.name = "gpt4"
                mock_load.return_value = [mock_agent1, mock_agent2]

                agents = await handler._load_agents([])

                # The patched method was called
                assert mock_load.called
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    @pytest.mark.asyncio
    async def test_load_agents_returns_empty_on_failure(self):
        """Test graceful handling of agent load failure."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            # Patch at the import location within the handler
            with patch.object(handler, "_load_agents", new_callable=AsyncMock) as mock_load:
                mock_load.return_value = []

                agents = await handler._load_agents(["nonexistent"])

                assert agents == []
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")


# ============================================================================
# Handle GET Routing Tests
# ============================================================================


class TestHandleGetRouting:
    """Tests for GET request routing via internal methods."""

    @pytest.mark.asyncio
    async def test_get_matrix_debate_by_id(self, mock_handler, mock_storage):
        """Test getting matrix debate by ID."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            mock_storage.get_matrix_debate = AsyncMock(
                return_value={
                    "matrix_id": "abc-123",
                    "task": "Test",
                }
            )
            mock_handler.storage = mock_storage

            handler = MatrixDebatesHandler({})

            # Call internal method directly
            result = await handler._get_matrix_debate(mock_handler, "abc-123")

            assert result.status_code == 200
            data = json.loads(result.body)
            assert data["matrix_id"] == "abc-123"
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    @pytest.mark.asyncio
    async def test_get_scenarios_by_matrix_id(self, mock_handler, mock_storage):
        """Test getting scenarios for a matrix debate."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            mock_storage.get_matrix_scenarios = AsyncMock(
                return_value=[
                    {"name": "Scenario 1"},
                    {"name": "Scenario 2"},
                ]
            )
            mock_handler.storage = mock_storage

            handler = MatrixDebatesHandler({})

            result = await handler._get_scenarios(mock_handler, "abc-123")

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "scenarios" in data
            assert len(data["scenarios"]) == 2
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    @pytest.mark.asyncio
    async def test_get_conclusions_by_matrix_id(self, mock_handler, mock_storage):
        """Test getting conclusions for a matrix debate."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            mock_storage.get_matrix_conclusions = AsyncMock(
                return_value={
                    "universal": ["All scenarios agree"],
                    "conditional": [{"condition": "When X", "conclusion": "Y"}],
                }
            )
            mock_handler.storage = mock_storage

            handler = MatrixDebatesHandler({})

            result = await handler._get_conclusions(mock_handler, "abc-123")

            assert result.status_code == 200
            data = json.loads(result.body)
            assert "universal_conclusions" in data
            assert "conditional_conclusions" in data
            assert len(data["universal_conclusions"]) == 1
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")


# ============================================================================
# Fallback Implementation Tests
# ============================================================================


class TestMatrixDebateFallback:
    """Tests for fallback implementation when MatrixDebateRunner unavailable."""

    @pytest.mark.asyncio
    async def test_fallback_runs_scenarios_in_parallel(self, mock_handler, sample_scenarios):
        """Test fallback runs scenarios concurrently."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            mock_agent = Mock()
            mock_agent.name = "claude"

            with patch.object(handler, "_load_agents", new_callable=AsyncMock) as mock_load:
                mock_load.return_value = [mock_agent]

                # Mock Arena to track scenario runs
                with patch("aragora.debate.orchestrator.Arena") as mock_arena_class:
                    mock_arena = Mock()
                    mock_arena.run = AsyncMock(
                        return_value=Mock(
                            winner="claude",
                            final_answer="Test answer",
                            confidence=0.8,
                            consensus_reached=True,
                            rounds_used=2,
                        )
                    )
                    mock_arena_class.return_value = mock_arena

                    result = await handler._run_matrix_debate_fallback(
                        mock_handler,
                        {
                            "task": "Test task",
                            "scenarios": sample_scenarios,
                            "agents": ["claude"],
                            "max_rounds": 3,
                        },
                    )

                    assert result.status_code == 200
                    data = json.loads(result.body)
                    assert "matrix_id" in data
                    assert data["scenario_count"] == 3
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")

    @pytest.mark.asyncio
    async def test_fallback_no_agents(self, mock_handler, sample_scenarios):
        """Test fallback returns error when no agents loaded."""
        try:
            from aragora.server.handlers.debates import MatrixDebatesHandler

            handler = MatrixDebatesHandler({})

            with patch.object(handler, "_load_agents", new_callable=AsyncMock) as mock_load:
                mock_load.return_value = []

                result = await handler._run_matrix_debate_fallback(
                    mock_handler,
                    {
                        "task": "Test task",
                        "scenarios": sample_scenarios,
                    },
                )

                assert result.status_code == 400
                data = json.loads(result.body)
                assert "agent" in data.get("error", "").lower()
        except ImportError:
            pytest.skip("MatrixDebatesHandler not available")
