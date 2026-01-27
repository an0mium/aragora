"""
Tests for Agent CV (Curriculum Vitae) system.

Tests the unified capability profile that aggregates data from
ELO, calibration, and performance systems.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import tempfile
from pathlib import Path

from aragora.agents.cv import (
    AgentCV,
    CVBuilder,
    DomainPerformance,
    ReliabilityMetrics,
    get_cv_builder,
)
from aragora.agents.cv_store import CVStore, get_cv_store


# =============================================================================
# ReliabilityMetrics Tests
# =============================================================================


class TestReliabilityMetrics:
    """Tests for ReliabilityMetrics dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        metrics = ReliabilityMetrics()
        assert metrics.success_rate == 0.0
        assert metrics.failure_rate == 0.0
        assert metrics.timeout_rate == 0.0
        assert metrics.total_calls == 0

    def test_is_reliable_with_high_success(self):
        """Test reliability check with high success rate."""
        metrics = ReliabilityMetrics(
            success_rate=0.95,
            total_calls=10,
        )
        assert metrics.is_reliable is True

    def test_is_reliable_with_low_success(self):
        """Test reliability check with low success rate."""
        metrics = ReliabilityMetrics(
            success_rate=0.75,
            total_calls=10,
        )
        assert metrics.is_reliable is False

    def test_is_reliable_with_insufficient_calls(self):
        """Test reliability check with too few calls."""
        metrics = ReliabilityMetrics(
            success_rate=0.95,
            total_calls=3,
        )
        assert metrics.is_reliable is False

    def test_to_dict(self):
        """Test dictionary conversion."""
        metrics = ReliabilityMetrics(
            success_rate=0.9,
            failure_rate=0.1,
            timeout_rate=0.05,
            total_calls=100,
            avg_latency_ms=150.0,
        )
        data = metrics.to_dict()
        assert data["success_rate"] == 0.9
        assert data["total_calls"] == 100
        assert "is_reliable" in data


# =============================================================================
# DomainPerformance Tests
# =============================================================================


class TestDomainPerformance:
    """Tests for DomainPerformance dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        perf = DomainPerformance(domain="code")
        assert perf.domain == "code"
        assert perf.elo == 1000.0
        assert perf.debates_count == 0

    def test_has_meaningful_data(self):
        """Test meaningful data threshold."""
        perf_low = DomainPerformance(domain="code", debates_count=2)
        perf_high = DomainPerformance(domain="code", debates_count=5)

        assert perf_low.has_meaningful_data is False
        assert perf_high.has_meaningful_data is True

    def test_composite_score_with_data(self):
        """Test composite score calculation."""
        perf = DomainPerformance(
            domain="code",
            elo=1100.0,
            win_rate=0.7,
            debates_count=10,
            brier_score=0.2,
        )
        score = perf.composite_score
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be a good score

    def test_composite_score_without_data(self):
        """Test composite score with insufficient data."""
        perf = DomainPerformance(domain="code", debates_count=1)
        assert perf.composite_score == 0.0

    def test_to_dict(self):
        """Test dictionary conversion."""
        perf = DomainPerformance(
            domain="code",
            elo=1100.0,
            win_rate=0.7,
            debates_count=10,
        )
        data = perf.to_dict()
        assert data["domain"] == "code"
        assert data["elo"] == 1100.0
        assert "composite_score" in data


# =============================================================================
# AgentCV Tests
# =============================================================================


class TestAgentCV:
    """Tests for AgentCV dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        cv = AgentCV(agent_id="claude-opus")
        assert cv.agent_id == "claude-opus"
        assert cv.overall_elo == 1000.0
        assert cv.total_debates == 0
        assert cv.calibration_bias == "unknown"

    def test_has_meaningful_data_with_debates(self):
        """Test meaningful data based on debates."""
        cv_low = AgentCV(agent_id="test", total_debates=3)
        cv_high = AgentCV(agent_id="test", total_debates=10)

        assert cv_low.has_meaningful_data is False
        assert cv_high.has_meaningful_data is True

    def test_has_meaningful_data_with_calls(self):
        """Test meaningful data based on reliability calls."""
        cv = AgentCV(
            agent_id="test",
            total_debates=2,
            reliability=ReliabilityMetrics(total_calls=15),
        )
        assert cv.has_meaningful_data is True

    def test_best_domains(self):
        """Test best domains extraction."""
        cv = AgentCV(
            agent_id="test",
            domain_performance={
                "code": DomainPerformance(
                    domain="code", elo=1200, debates_count=10, brier_score=0.1
                ),
                "research": DomainPerformance(
                    domain="research", elo=1100, debates_count=8, brier_score=0.2
                ),
                "creative": DomainPerformance(
                    domain="creative", elo=1000, debates_count=5, brier_score=0.3
                ),
                "new": DomainPerformance(domain="new", debates_count=1),  # Not enough data
            },
        )
        best = cv.best_domains
        assert "code" in best
        assert "research" in best
        assert "new" not in best  # Insufficient data

    def test_is_well_calibrated(self):
        """Test calibration quality check."""
        cv_good = AgentCV(agent_id="test", expected_calibration_error=0.1)
        cv_bad = AgentCV(agent_id="test", expected_calibration_error=0.3)

        assert cv_good.is_well_calibrated is True
        assert cv_bad.is_well_calibrated is False

    def test_get_domain_score(self):
        """Test domain score retrieval."""
        cv = AgentCV(
            agent_id="test",
            domain_performance={
                "code": DomainPerformance(
                    domain="code", elo=1200, debates_count=10, brier_score=0.1
                ),
            },
        )
        assert cv.get_domain_score("code") > 0.0
        assert cv.get_domain_score("nonexistent") == 0.0

    def test_compute_selection_score(self):
        """Test selection score computation."""
        cv = AgentCV(
            agent_id="test",
            overall_elo=1100.0,
            overall_win_rate=0.7,
            brier_score=0.2,
            reliability=ReliabilityMetrics(success_rate=0.95),
            domain_performance={
                "code": DomainPerformance(
                    domain="code", elo=1200, debates_count=10, win_rate=0.8, brier_score=0.1
                ),
            },
        )

        # Without domain
        score_no_domain = cv.compute_selection_score()
        assert 0.0 <= score_no_domain <= 2.0  # Can exceed 1.0 due to weights

        # With matching domain (strong domain performance should help)
        score_with_domain = cv.compute_selection_score(domain="code")
        assert 0.0 <= score_with_domain <= 2.0

        # Domain score should be calculated using domain-specific composite
        domain_component = cv.get_domain_score("code")
        assert domain_component > 0.0  # Should have domain expertise

    def test_to_dict_and_from_dict(self):
        """Test round-trip serialization."""
        original = AgentCV(
            agent_id="claude-opus",
            model_name="claude-3-opus",
            overall_elo=1150.0,
            overall_win_rate=0.72,
            total_debates=25,
            calibration_accuracy=0.85,
            brier_score=0.15,
            calibration_bias="well-calibrated",
            reliability=ReliabilityMetrics(
                success_rate=0.95,
                total_calls=100,
                avg_latency_ms=200.0,
            ),
            domain_performance={
                "code": DomainPerformance(domain="code", elo=1200, debates_count=15),
            },
            learning_category="steady",
            model_capabilities=["reasoning", "code"],
            learned_strengths=["technical analysis", "clear explanations"],
        )

        data = original.to_dict()
        restored = AgentCV.from_dict(data)

        assert restored.agent_id == original.agent_id
        assert restored.overall_elo == original.overall_elo
        assert restored.calibration_bias == original.calibration_bias
        assert "code" in restored.domain_performance
        assert restored.reliability.success_rate == 0.95


# =============================================================================
# CVBuilder Tests
# =============================================================================


class TestCVBuilder:
    """Tests for CVBuilder class."""

    def test_build_cv_without_sources(self):
        """Test building CV with no data sources."""
        builder = CVBuilder()
        cv = builder.build_cv("test-agent")

        assert cv.agent_id == "test-agent"
        assert cv.data_sources == []

    def test_build_cv_with_elo_system(self):
        """Test building CV with ELO system."""
        # Mock ELO system
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.elo = 1100.0
        mock_rating.win_rate = 0.65
        mock_rating.debates_count = 20
        mock_rating.domain_elos = {"code": 1150.0}
        mock_elo.get_rating.return_value = mock_rating
        mock_elo.get_learning_efficiency.return_value = {
            "learning_category": "steady",
            "elo_gain_rate": 2.5,
        }

        builder = CVBuilder(elo_system=mock_elo)
        cv = builder.build_cv("claude-opus")

        assert cv.overall_elo == 1100.0
        assert cv.overall_win_rate == 0.65
        assert cv.total_debates == 20
        assert "code" in cv.domain_performance
        assert cv.learning_category == "steady"
        assert "elo" in cv.data_sources

    def test_build_cv_with_calibration_tracker(self):
        """Test building CV with calibration tracker."""
        # Mock calibration tracker
        mock_calibration = MagicMock()
        mock_summary = MagicMock()
        mock_summary.accuracy = 0.82
        mock_summary.brier_score = 0.18
        mock_summary.ece = 0.12
        mock_summary.bias_direction = "well-calibrated"
        mock_calibration.get_calibration_summary.return_value = mock_summary
        mock_calibration.get_domain_breakdown.return_value = {}

        builder = CVBuilder(calibration_tracker=mock_calibration)
        cv = builder.build_cv("claude-opus")

        assert cv.calibration_accuracy == 0.82
        assert cv.brier_score == 0.18
        assert cv.expected_calibration_error == 0.12
        assert cv.calibration_bias == "well-calibrated"
        assert "calibration" in cv.data_sources

    def test_build_cv_with_performance_monitor(self):
        """Test building CV with performance monitor."""
        # Mock performance monitor
        mock_monitor = MagicMock()
        mock_stats = MagicMock()
        mock_stats.success_rate = 95.0  # Percentage
        mock_stats.timeout_rate = 2.0
        mock_stats.total_calls = 50
        mock_stats.avg_duration_ms = 180.0
        mock_stats.max_duration_ms = 500.0
        mock_monitor.agent_stats = {"claude-opus": mock_stats}

        builder = CVBuilder(performance_monitor=mock_monitor)
        cv = builder.build_cv("claude-opus")

        assert cv.reliability.success_rate == 0.95
        assert cv.reliability.total_calls == 50
        assert "performance" in cv.data_sources

    def test_build_cvs_batch(self):
        """Test batch CV building."""
        # Mock ELO system with batch support
        mock_elo = MagicMock()
        mock_elo.get_ratings_batch.return_value = {
            "agent1": MagicMock(elo=1100, win_rate=0.6, debates_count=10, domain_elos={}),
            "agent2": MagicMock(elo=1050, win_rate=0.5, debates_count=8, domain_elos={}),
        }

        builder = CVBuilder(elo_system=mock_elo)
        cvs = builder.build_cvs_batch(["agent1", "agent2"])

        assert "agent1" in cvs
        assert "agent2" in cvs
        assert cvs["agent1"].overall_elo == 1100


# =============================================================================
# CVStore Tests
# =============================================================================


class TestCVStore:
    """Tests for CVStore persistence."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_cv.db"

    @pytest.fixture
    def store(self, temp_db_path):
        """Create a CVStore with mocked builder."""
        mock_builder = MagicMock()
        mock_builder.build_cv.return_value = AgentCV(
            agent_id="test-agent",
            overall_elo=1100.0,
            total_debates=15,
        )
        mock_builder.build_cvs_batch.return_value = {
            "agent1": AgentCV(agent_id="agent1", overall_elo=1100.0),
            "agent2": AgentCV(agent_id="agent2", overall_elo=1050.0),
        }
        return CVStore(db_path=temp_db_path, cv_builder=mock_builder)

    @pytest.mark.asyncio
    async def test_save_and_get_cv(self, store):
        """Test saving and retrieving a CV."""
        cv = AgentCV(
            agent_id="claude-opus",
            overall_elo=1150.0,
            calibration_bias="well-calibrated",
        )

        await store.save_cv(cv)
        retrieved = await store.get_cv("claude-opus", auto_build=False)

        assert retrieved is not None
        assert retrieved.agent_id == "claude-opus"
        assert retrieved.overall_elo == 1150.0

    @pytest.mark.asyncio
    async def test_get_cv_uses_cache(self, store):
        """Test that get_cv uses cache."""
        cv = AgentCV(agent_id="cached-agent", overall_elo=1100.0)
        await store.save_cv(cv)

        # First call should populate cache
        cv1 = await store.get_cv("cached-agent")

        # Second call should use cache (builder not called again)
        cv2 = await store.get_cv("cached-agent")

        assert cv1 is cv2  # Same object from cache

    @pytest.mark.asyncio
    async def test_get_cv_auto_builds(self, store):
        """Test that get_cv auto-builds missing CVs."""
        # Request CV that doesn't exist
        cv = await store.get_cv("new-agent", auto_build=True)

        assert cv is not None
        assert cv.agent_id == "test-agent"  # From mock builder
        store.cv_builder.build_cv.assert_called_once_with("new-agent")

    @pytest.mark.asyncio
    async def test_refresh_cv(self, store):
        """Test force refresh of CV."""
        # Save initial CV
        cv_initial = AgentCV(agent_id="refresh-agent", overall_elo=1000.0)
        await store.save_cv(cv_initial)

        # Refresh should call builder
        cv_refreshed = await store.refresh_cv("refresh-agent")

        store.cv_builder.build_cv.assert_called_with("refresh-agent")
        assert cv_refreshed.overall_elo == 1100.0  # From mock builder

    @pytest.mark.asyncio
    async def test_get_cvs_batch(self, store):
        """Test batch CV retrieval."""
        cvs = await store.get_cvs_batch(["agent1", "agent2"])

        assert "agent1" in cvs
        assert "agent2" in cvs
        store.cv_builder.build_cvs_batch.assert_called()

    @pytest.mark.asyncio
    async def test_delete_cv(self, store):
        """Test CV deletion."""
        cv = AgentCV(agent_id="delete-me", overall_elo=1100.0)
        await store.save_cv(cv)

        deleted = await store.delete_cv("delete-me")
        assert deleted is True

        # Should not exist anymore
        retrieved = await store.get_cv("delete-me", auto_build=False)
        assert retrieved is None

    def test_invalidate_cache(self, store):
        """Test cache invalidation."""
        # Manually add to cache
        store._cache["agent1"] = (datetime.now(), AgentCV(agent_id="agent1"))
        store._cache["agent2"] = (datetime.now(), AgentCV(agent_id="agent2"))

        # Invalidate one
        count = store.invalidate_cache("agent1")
        assert count == 1
        assert "agent1" not in store._cache
        assert "agent2" in store._cache

        # Invalidate all
        count = store.invalidate_cache()
        assert count == 1
        assert len(store._cache) == 0

    def test_get_cv_sync(self, store):
        """Test synchronous CV retrieval."""
        cv = store.get_cv_sync("test-agent", auto_build=True)

        assert cv is not None
        assert cv.agent_id == "test-agent"

    @pytest.mark.asyncio
    async def test_get_all_cvs(self, store):
        """Test retrieving all CVs."""
        # Save some CVs
        for i in range(5):
            cv = AgentCV(agent_id=f"agent-{i}", overall_elo=1000 + i * 50)
            await store.save_cv(cv)

        all_cvs = await store.get_all_cvs(limit=10)

        assert len(all_cvs) == 5

    @pytest.mark.asyncio
    async def test_get_top_agents_for_domain(self, store):
        """Test retrieving top agents for a domain."""
        # Save CVs with domain performance
        for i in range(3):
            cv = AgentCV(
                agent_id=f"domain-agent-{i}",
                domain_performance={
                    "code": DomainPerformance(
                        domain="code",
                        elo=1000 + i * 100,
                        debates_count=10,
                        brier_score=0.3 - i * 0.1,
                    ),
                },
            )
            await store.save_cv(cv)

        top = await store.get_top_agents_for_domain("code", limit=2)

        assert len(top) == 2
        # Should be sorted by composite score
        assert (
            top[0].domain_performance["code"].composite_score
            >= top[1].domain_performance["code"].composite_score
        )


# =============================================================================
# Integration Tests
# =============================================================================


class TestCVIntegration:
    """Integration tests for the CV system."""

    def test_cv_selection_workflow(self):
        """Test using CV for agent selection."""
        # Create CVs for a team
        cvs = [
            AgentCV(
                agent_id="claude",
                overall_elo=1150.0,
                brier_score=0.15,
                reliability=ReliabilityMetrics(success_rate=0.95),
                domain_performance={
                    "code": DomainPerformance(
                        domain="code", elo=1200, debates_count=20, brier_score=0.1
                    ),
                },
            ),
            AgentCV(
                agent_id="gpt-4",
                overall_elo=1100.0,
                brier_score=0.2,
                reliability=ReliabilityMetrics(success_rate=0.92),
                domain_performance={
                    "code": DomainPerformance(
                        domain="code", elo=1080, debates_count=15, brier_score=0.18
                    ),
                },
            ),
            AgentCV(
                agent_id="gemini",
                overall_elo=1050.0,
                brier_score=0.25,
                reliability=ReliabilityMetrics(success_rate=0.88),
                domain_performance={
                    "research": DomainPerformance(
                        domain="research", elo=1150, debates_count=12, brier_score=0.12
                    ),
                },
            ),
        ]

        # Rank for code domain
        code_scores = [(cv.agent_id, cv.compute_selection_score(domain="code")) for cv in cvs]
        code_scores.sort(key=lambda x: x[1], reverse=True)

        # Claude should rank highest for code
        assert code_scores[0][0] == "claude"

        # Rank for research domain
        research_scores = [
            (cv.agent_id, cv.compute_selection_score(domain="research")) for cv in cvs
        ]
        research_scores.sort(key=lambda x: x[1], reverse=True)

        # Gemini should rank higher for research (its specialty)
        gemini_rank = next(i for i, (name, _) in enumerate(research_scores) if name == "gemini")
        assert gemini_rank <= 1  # Top 2

    def test_cv_reliability_filtering(self):
        """Test filtering unreliable agents."""
        cvs = [
            AgentCV(
                agent_id="reliable",
                reliability=ReliabilityMetrics(success_rate=0.95, total_calls=100),
            ),
            AgentCV(
                agent_id="unreliable",
                reliability=ReliabilityMetrics(success_rate=0.6, total_calls=50),
            ),
            AgentCV(
                agent_id="untested",
                reliability=ReliabilityMetrics(success_rate=1.0, total_calls=2),
            ),
        ]

        reliable_agents = [cv for cv in cvs if cv.reliability.is_reliable]

        assert len(reliable_agents) == 1
        assert reliable_agents[0].agent_id == "reliable"

    def test_cv_learning_trajectory(self):
        """Test using learning trajectory for selection."""
        cvs = [
            AgentCV(agent_id="rapid-learner", learning_category="rapid", elo_gain_rate=5.0),
            AgentCV(agent_id="steady-learner", learning_category="steady", elo_gain_rate=2.0),
            AgentCV(agent_id="declining", learning_category="declining", elo_gain_rate=-1.0),
        ]

        # Filter to agents that are improving
        improving = [cv for cv in cvs if cv.learning_category in ("rapid", "steady")]

        assert len(improving) == 2
        assert "declining" not in [cv.agent_id for cv in improving]


# =============================================================================
# Singleton Tests
# =============================================================================


class TestSingletons:
    """Tests for singleton accessors."""

    def test_get_cv_builder(self):
        """Test global CV builder singleton."""
        # Reset singleton for testing
        import aragora.agents.cv as cv_module

        cv_module._cv_builder = None

        builder = get_cv_builder()
        assert builder is not None

        # Second call returns same instance
        builder2 = get_cv_builder()
        assert builder is builder2
