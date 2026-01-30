"""
Tests for Cost Optimization Recommendations.

Tests cover:
- Recommendation types, priorities, and statuses
- Model alternatives and caching/batching opportunities
- Implementation steps
- OptimizationRecommendation dataclass functionality
- RecommendationSummary aggregation
- Serialization (to_dict/from_dict)
- Edge cases and error handling
"""

from decimal import Decimal
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from aragora.billing.recommendations import (
    RecommendationType,
    RecommendationPriority,
    RecommendationStatus,
    ModelAlternative,
    CachingOpportunity,
    BatchingOpportunity,
    ImplementationStep,
    OptimizationRecommendation,
    RecommendationSummary,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_model_alternative():
    """Sample ModelAlternative for testing."""
    return ModelAlternative(
        provider="openai",
        model="gpt-4o-mini",
        cost_per_1k_input=Decimal("0.00015"),
        cost_per_1k_output=Decimal("0.0006"),
        quality_score=0.85,
        latency_multiplier=0.8,
        suitable_for=["simple_queries", "classification"],
    )


@pytest.fixture
def sample_caching_opportunity():
    """Sample CachingOpportunity for testing."""
    return CachingOpportunity(
        pattern="system_prompt",
        estimated_hit_rate=0.75,
        unique_queries=50,
        repeat_count=200,
        cache_strategy="semantic",
    )


@pytest.fixture
def sample_batching_opportunity():
    """Sample BatchingOpportunity for testing."""
    return BatchingOpportunity(
        operation_type="embedding",
        current_batch_size=1,
        optimal_batch_size=10,
        requests_per_hour=500,
        latency_impact_ms=50.0,
    )


@pytest.fixture
def sample_implementation_step():
    """Sample ImplementationStep for testing."""
    return ImplementationStep(
        order=1,
        description="Configure the caching middleware",
        code_snippet="cache = SemanticCache(ttl=3600)",
        config_change={"cache_enabled": True, "cache_ttl": 3600},
        estimated_effort="medium",
    )


@pytest.fixture
def sample_recommendation(
    sample_model_alternative,
    sample_caching_opportunity,
    sample_implementation_step,
):
    """Sample OptimizationRecommendation for testing."""
    return OptimizationRecommendation(
        id="rec-001",
        type=RecommendationType.MODEL_DOWNGRADE,
        priority=RecommendationPriority.HIGH,
        status=RecommendationStatus.PENDING,
        current_cost_usd=Decimal("100.00"),
        projected_cost_usd=Decimal("40.00"),
        confidence_score=0.85,
        workspace_id="ws-123",
        org_id="org-456",
        affected_agents=["claude", "gemini"],
        affected_operations=["debate_round", "summarization"],
        title="Switch to GPT-4o-mini for simple queries",
        description="Use a cheaper model for classification and simple Q&A",
        rationale="Analysis shows 60% of queries are simple classifications",
        model_alternative=sample_model_alternative,
        caching_opportunity=sample_caching_opportunity,
        implementation_steps=[sample_implementation_step],
        auto_apply_available=True,
        requires_approval=True,
        quality_impact="Slight reduction in nuanced reasoning",
        quality_impact_score=0.15,
        risk_level="low",
        metadata={"source": "auto_analysis"},
    )


# =============================================================================
# RecommendationType Tests
# =============================================================================


class TestRecommendationType:
    """Tests for RecommendationType enum."""

    def test_all_types_exist(self):
        """Test all recommendation types are defined."""
        assert RecommendationType.MODEL_DOWNGRADE.value == "model_downgrade"
        assert RecommendationType.CACHING.value == "caching"
        assert RecommendationType.BATCHING.value == "batching"
        assert RecommendationType.RATE_LIMITING.value == "rate_limiting"
        assert RecommendationType.PROMPT_OPTIMIZATION.value == "prompt_optimization"
        assert RecommendationType.PROVIDER_SWITCH.value == "provider_switch"
        assert RecommendationType.TIME_SHIFTING.value == "time_shifting"
        assert RecommendationType.QUOTA_ADJUSTMENT.value == "quota_adjustment"

    def test_type_is_string_enum(self):
        """Test types are string enums."""
        for rec_type in RecommendationType:
            assert isinstance(rec_type.value, str)
            assert str(rec_type) == f"RecommendationType.{rec_type.name}"

    def test_type_count(self):
        """Test total number of recommendation types."""
        assert len(RecommendationType) == 8


# =============================================================================
# RecommendationPriority Tests
# =============================================================================


class TestRecommendationPriority:
    """Tests for RecommendationPriority enum."""

    def test_all_priorities_exist(self):
        """Test all priority levels are defined."""
        assert RecommendationPriority.CRITICAL.value == "critical"
        assert RecommendationPriority.HIGH.value == "high"
        assert RecommendationPriority.MEDIUM.value == "medium"
        assert RecommendationPriority.LOW.value == "low"

    def test_priority_count(self):
        """Test total number of priority levels."""
        assert len(RecommendationPriority) == 4


# =============================================================================
# RecommendationStatus Tests
# =============================================================================


class TestRecommendationStatus:
    """Tests for RecommendationStatus enum."""

    def test_all_statuses_exist(self):
        """Test all status values are defined."""
        assert RecommendationStatus.PENDING.value == "pending"
        assert RecommendationStatus.APPLIED.value == "applied"
        assert RecommendationStatus.DISMISSED.value == "dismissed"
        assert RecommendationStatus.EXPIRED.value == "expired"
        assert RecommendationStatus.PARTIAL.value == "partial"

    def test_status_count(self):
        """Test total number of status values."""
        assert len(RecommendationStatus) == 5


# =============================================================================
# ModelAlternative Tests
# =============================================================================


class TestModelAlternative:
    """Tests for ModelAlternative dataclass."""

    def test_create_model_alternative(self, sample_model_alternative):
        """Test creating a model alternative."""
        assert sample_model_alternative.provider == "openai"
        assert sample_model_alternative.model == "gpt-4o-mini"
        assert sample_model_alternative.cost_per_1k_input == Decimal("0.00015")
        assert sample_model_alternative.cost_per_1k_output == Decimal("0.0006")
        assert sample_model_alternative.quality_score == 0.85
        assert sample_model_alternative.latency_multiplier == 0.8

    def test_model_alternative_suitable_for(self, sample_model_alternative):
        """Test suitable_for list."""
        assert "simple_queries" in sample_model_alternative.suitable_for
        assert "classification" in sample_model_alternative.suitable_for
        assert len(sample_model_alternative.suitable_for) == 2

    def test_model_alternative_default_suitable_for(self):
        """Test default suitable_for is empty list."""
        alt = ModelAlternative(
            provider="mistral",
            model="mistral-small",
            cost_per_1k_input=Decimal("0.0001"),
            cost_per_1k_output=Decimal("0.0003"),
            quality_score=0.7,
            latency_multiplier=1.0,
        )
        assert alt.suitable_for == []

    def test_model_alternative_quality_score_bounds(self):
        """Test quality score edge values."""
        alt_zero = ModelAlternative(
            provider="test",
            model="test",
            cost_per_1k_input=Decimal("0"),
            cost_per_1k_output=Decimal("0"),
            quality_score=0.0,
            latency_multiplier=1.0,
        )
        assert alt_zero.quality_score == 0.0

        alt_one = ModelAlternative(
            provider="test",
            model="test",
            cost_per_1k_input=Decimal("0"),
            cost_per_1k_output=Decimal("0"),
            quality_score=1.0,
            latency_multiplier=1.0,
        )
        assert alt_one.quality_score == 1.0


# =============================================================================
# CachingOpportunity Tests
# =============================================================================


class TestCachingOpportunity:
    """Tests for CachingOpportunity dataclass."""

    def test_create_caching_opportunity(self, sample_caching_opportunity):
        """Test creating a caching opportunity."""
        assert sample_caching_opportunity.pattern == "system_prompt"
        assert sample_caching_opportunity.estimated_hit_rate == 0.75
        assert sample_caching_opportunity.unique_queries == 50
        assert sample_caching_opportunity.repeat_count == 200
        assert sample_caching_opportunity.cache_strategy == "semantic"

    def test_different_cache_strategies(self):
        """Test different cache strategies."""
        for strategy in ["exact", "semantic", "prefix"]:
            opportunity = CachingOpportunity(
                pattern="test",
                estimated_hit_rate=0.5,
                unique_queries=10,
                repeat_count=100,
                cache_strategy=strategy,
            )
            assert opportunity.cache_strategy == strategy

    def test_hit_rate_bounds(self):
        """Test hit rate edge values."""
        zero_hit = CachingOpportunity(
            pattern="none",
            estimated_hit_rate=0.0,
            unique_queries=100,
            repeat_count=0,
            cache_strategy="exact",
        )
        assert zero_hit.estimated_hit_rate == 0.0

        full_hit = CachingOpportunity(
            pattern="all",
            estimated_hit_rate=1.0,
            unique_queries=1,
            repeat_count=1000,
            cache_strategy="exact",
        )
        assert full_hit.estimated_hit_rate == 1.0


# =============================================================================
# BatchingOpportunity Tests
# =============================================================================


class TestBatchingOpportunity:
    """Tests for BatchingOpportunity dataclass."""

    def test_create_batching_opportunity(self, sample_batching_opportunity):
        """Test creating a batching opportunity."""
        assert sample_batching_opportunity.operation_type == "embedding"
        assert sample_batching_opportunity.current_batch_size == 1
        assert sample_batching_opportunity.optimal_batch_size == 10
        assert sample_batching_opportunity.requests_per_hour == 500
        assert sample_batching_opportunity.latency_impact_ms == 50.0

    def test_batch_size_improvement_ratio(self, sample_batching_opportunity):
        """Test batch size improvement calculation."""
        improvement = (
            sample_batching_opportunity.optimal_batch_size
            / sample_batching_opportunity.current_batch_size
        )
        assert improvement == 10.0

    def test_zero_latency_impact(self):
        """Test batching with no latency impact."""
        opportunity = BatchingOpportunity(
            operation_type="inference",
            current_batch_size=5,
            optimal_batch_size=10,
            requests_per_hour=100,
            latency_impact_ms=0.0,
        )
        assert opportunity.latency_impact_ms == 0.0


# =============================================================================
# ImplementationStep Tests
# =============================================================================


class TestImplementationStep:
    """Tests for ImplementationStep dataclass."""

    def test_create_implementation_step(self, sample_implementation_step):
        """Test creating an implementation step."""
        assert sample_implementation_step.order == 1
        assert sample_implementation_step.description == "Configure the caching middleware"
        assert sample_implementation_step.code_snippet == "cache = SemanticCache(ttl=3600)"
        assert sample_implementation_step.config_change == {
            "cache_enabled": True,
            "cache_ttl": 3600,
        }
        assert sample_implementation_step.estimated_effort == "medium"

    def test_step_with_no_code_snippet(self):
        """Test step without code snippet."""
        step = ImplementationStep(
            order=1,
            description="Review configuration",
            code_snippet=None,
            estimated_effort="low",
        )
        assert step.code_snippet is None

    def test_step_with_no_config_change(self):
        """Test step without config change."""
        step = ImplementationStep(
            order=2,
            description="Update code",
            config_change=None,
        )
        assert step.config_change is None

    def test_default_estimated_effort(self):
        """Test default estimated effort is low."""
        step = ImplementationStep(
            order=1,
            description="Simple change",
        )
        assert step.estimated_effort == "low"

    def test_effort_levels(self):
        """Test different effort levels."""
        for effort in ["low", "medium", "high"]:
            step = ImplementationStep(
                order=1,
                description="Test",
                estimated_effort=effort,
            )
            assert step.estimated_effort == effort


# =============================================================================
# OptimizationRecommendation Tests
# =============================================================================


class TestOptimizationRecommendation:
    """Tests for OptimizationRecommendation dataclass."""

    def test_create_recommendation(self, sample_recommendation):
        """Test creating a recommendation."""
        assert sample_recommendation.id == "rec-001"
        assert sample_recommendation.type == RecommendationType.MODEL_DOWNGRADE
        assert sample_recommendation.priority == RecommendationPriority.HIGH
        assert sample_recommendation.status == RecommendationStatus.PENDING
        assert sample_recommendation.workspace_id == "ws-123"
        assert sample_recommendation.org_id == "org-456"

    def test_calculated_savings(self, sample_recommendation):
        """Test savings are calculated in __post_init__."""
        assert sample_recommendation.estimated_savings_usd == Decimal("60.00")
        assert sample_recommendation.savings_percentage == 60.0

    def test_calculated_savings_zero_current(self):
        """Test savings calculation with zero current cost."""
        rec = OptimizationRecommendation(
            current_cost_usd=Decimal("0"),
            projected_cost_usd=Decimal("0"),
        )
        assert rec.estimated_savings_usd == Decimal("0")
        assert rec.savings_percentage == 0.0

    def test_auto_generated_id(self):
        """Test auto-generated UUID for id."""
        rec = OptimizationRecommendation()
        # Should be a valid UUID string
        UUID(rec.id)

    def test_default_values(self):
        """Test default field values."""
        rec = OptimizationRecommendation()
        assert rec.type == RecommendationType.MODEL_DOWNGRADE
        assert rec.priority == RecommendationPriority.MEDIUM
        assert rec.status == RecommendationStatus.PENDING
        assert rec.current_cost_usd == Decimal("0")
        assert rec.projected_cost_usd == Decimal("0")
        assert rec.confidence_score == 0.0
        assert rec.workspace_id == ""
        assert rec.org_id is None
        assert rec.affected_agents == []
        assert rec.affected_operations == []
        assert rec.title == ""
        assert rec.description == ""
        assert rec.model_alternative is None
        assert rec.caching_opportunity is None
        assert rec.batching_opportunity is None
        assert rec.implementation_steps == []
        assert rec.auto_apply_available is False
        assert rec.requires_approval is True
        assert rec.quality_impact_score == 0.0
        assert rec.risk_level == "low"
        assert rec.expires_at is None
        assert rec.applied_at is None
        assert rec.applied_by is None
        assert rec.metadata == {}

    def test_created_at_timestamp(self):
        """Test created_at is set automatically."""
        before = datetime.now(timezone.utc)
        rec = OptimizationRecommendation()
        after = datetime.now(timezone.utc)

        assert before <= rec.created_at <= after

    def test_affected_agents_list(self, sample_recommendation):
        """Test affected agents list."""
        assert "claude" in sample_recommendation.affected_agents
        assert "gemini" in sample_recommendation.affected_agents
        assert len(sample_recommendation.affected_agents) == 2

    def test_affected_operations_list(self, sample_recommendation):
        """Test affected operations list."""
        assert "debate_round" in sample_recommendation.affected_operations
        assert "summarization" in sample_recommendation.affected_operations


class TestOptimizationRecommendationApply:
    """Tests for apply method."""

    def test_apply_recommendation(self, sample_recommendation):
        """Test applying a recommendation."""
        before = datetime.now(timezone.utc)
        sample_recommendation.apply("user-001")
        after = datetime.now(timezone.utc)

        assert sample_recommendation.status == RecommendationStatus.APPLIED
        assert sample_recommendation.applied_by == "user-001"
        assert before <= sample_recommendation.applied_at <= after

    def test_apply_updates_status(self):
        """Test apply changes status from pending."""
        rec = OptimizationRecommendation(status=RecommendationStatus.PENDING)
        rec.apply("admin")
        assert rec.status == RecommendationStatus.APPLIED


class TestOptimizationRecommendationDismiss:
    """Tests for dismiss method."""

    def test_dismiss_recommendation(self, sample_recommendation):
        """Test dismissing a recommendation."""
        sample_recommendation.dismiss()
        assert sample_recommendation.status == RecommendationStatus.DISMISSED

    def test_dismiss_from_pending(self):
        """Test dismiss changes status from pending."""
        rec = OptimizationRecommendation(status=RecommendationStatus.PENDING)
        rec.dismiss()
        assert rec.status == RecommendationStatus.DISMISSED


class TestOptimizationRecommendationToDict:
    """Tests for to_dict method."""

    def test_to_dict_basic_fields(self, sample_recommendation):
        """Test to_dict includes basic fields."""
        data = sample_recommendation.to_dict()

        assert data["id"] == "rec-001"
        assert data["type"] == "model_downgrade"
        assert data["priority"] == "high"
        assert data["status"] == "pending"
        assert data["workspace_id"] == "ws-123"
        assert data["org_id"] == "org-456"

    def test_to_dict_cost_fields(self, sample_recommendation):
        """Test to_dict includes cost fields as strings."""
        data = sample_recommendation.to_dict()

        assert data["current_cost_usd"] == "100.00"
        assert data["projected_cost_usd"] == "40.00"
        assert data["estimated_savings_usd"] == "60.00"
        assert data["savings_percentage"] == 60.0
        assert data["confidence_score"] == 0.85

    def test_to_dict_model_alternative(self, sample_recommendation):
        """Test to_dict serializes model alternative."""
        data = sample_recommendation.to_dict()

        assert data["model_alternative"] is not None
        assert data["model_alternative"]["provider"] == "openai"
        assert data["model_alternative"]["model"] == "gpt-4o-mini"
        assert data["model_alternative"]["cost_per_1k_input"] == "0.00015"
        assert data["model_alternative"]["quality_score"] == 0.85

    def test_to_dict_caching_opportunity(self, sample_recommendation):
        """Test to_dict serializes caching opportunity."""
        data = sample_recommendation.to_dict()

        assert data["caching_opportunity"] is not None
        assert data["caching_opportunity"]["pattern"] == "system_prompt"
        assert data["caching_opportunity"]["estimated_hit_rate"] == 0.75
        assert data["caching_opportunity"]["cache_strategy"] == "semantic"

    def test_to_dict_batching_opportunity(self, sample_batching_opportunity):
        """Test to_dict serializes batching opportunity."""
        rec = OptimizationRecommendation(
            batching_opportunity=sample_batching_opportunity,
        )
        data = rec.to_dict()

        assert data["batching_opportunity"] is not None
        assert data["batching_opportunity"]["operation_type"] == "embedding"
        assert data["batching_opportunity"]["current_batch_size"] == 1
        assert data["batching_opportunity"]["optimal_batch_size"] == 10

    def test_to_dict_implementation_steps(self, sample_recommendation):
        """Test to_dict serializes implementation steps."""
        data = sample_recommendation.to_dict()

        assert len(data["implementation_steps"]) == 1
        step = data["implementation_steps"][0]
        assert step["order"] == 1
        assert step["description"] == "Configure the caching middleware"
        assert step["estimated_effort"] == "medium"

    def test_to_dict_null_opportunities(self):
        """Test to_dict with null opportunities."""
        rec = OptimizationRecommendation()
        data = rec.to_dict()

        assert data["model_alternative"] is None
        assert data["caching_opportunity"] is None
        assert data["batching_opportunity"] is None

    def test_to_dict_timestamps(self, sample_recommendation):
        """Test to_dict formats timestamps as ISO strings."""
        data = sample_recommendation.to_dict()

        assert "created_at" in data
        # Parse the timestamp to verify format
        datetime.fromisoformat(data["created_at"])

        assert data["expires_at"] is None
        assert data["applied_at"] is None

    def test_to_dict_with_applied_timestamp(self, sample_recommendation):
        """Test to_dict includes applied timestamp after apply."""
        sample_recommendation.apply("user-001")
        data = sample_recommendation.to_dict()

        assert data["applied_at"] is not None
        assert data["applied_by"] == "user-001"
        datetime.fromisoformat(data["applied_at"])

    def test_to_dict_with_expires_at(self):
        """Test to_dict includes expires_at timestamp."""
        expires = datetime.now(timezone.utc) + timedelta(days=7)
        rec = OptimizationRecommendation(expires_at=expires)
        data = rec.to_dict()

        assert data["expires_at"] is not None
        datetime.fromisoformat(data["expires_at"])


class TestOptimizationRecommendationFromDict:
    """Tests for from_dict class method."""

    def test_from_dict_basic_fields(self):
        """Test from_dict parses basic fields."""
        data = {
            "id": "rec-002",
            "type": "caching",
            "priority": "critical",
            "status": "applied",
            "workspace_id": "ws-abc",
            "org_id": "org-xyz",
        }

        rec = OptimizationRecommendation.from_dict(data)

        assert rec.id == "rec-002"
        assert rec.type == RecommendationType.CACHING
        assert rec.priority == RecommendationPriority.CRITICAL
        assert rec.status == RecommendationStatus.APPLIED
        assert rec.workspace_id == "ws-abc"
        assert rec.org_id == "org-xyz"

    def test_from_dict_cost_fields(self):
        """Test from_dict parses cost fields."""
        data = {
            "current_cost_usd": "150.50",
            "projected_cost_usd": "75.25",
            "confidence_score": 0.92,
        }

        rec = OptimizationRecommendation.from_dict(data)

        assert rec.current_cost_usd == Decimal("150.50")
        assert rec.projected_cost_usd == Decimal("75.25")
        assert rec.confidence_score == 0.92
        # Savings should be calculated
        assert rec.estimated_savings_usd == Decimal("75.25")

    def test_from_dict_lists(self):
        """Test from_dict parses list fields."""
        data = {
            "affected_agents": ["claude", "gpt-4", "gemini"],
            "affected_operations": ["summarize", "translate"],
        }

        rec = OptimizationRecommendation.from_dict(data)

        assert len(rec.affected_agents) == 3
        assert "claude" in rec.affected_agents
        assert len(rec.affected_operations) == 2

    def test_from_dict_text_fields(self):
        """Test from_dict parses text fields."""
        data = {
            "title": "Enable caching",
            "description": "Cache repeated prompts",
            "rationale": "High repetition detected",
            "quality_impact": "No impact",
        }

        rec = OptimizationRecommendation.from_dict(data)

        assert rec.title == "Enable caching"
        assert rec.description == "Cache repeated prompts"
        assert rec.rationale == "High repetition detected"
        assert rec.quality_impact == "No impact"

    def test_from_dict_flags(self):
        """Test from_dict parses boolean flags."""
        data = {
            "auto_apply_available": True,
            "requires_approval": False,
        }

        rec = OptimizationRecommendation.from_dict(data)

        assert rec.auto_apply_available is True
        assert rec.requires_approval is False

    def test_from_dict_quality_and_risk(self):
        """Test from_dict parses quality and risk fields."""
        data = {
            "quality_impact_score": 0.25,
            "risk_level": "high",
        }

        rec = OptimizationRecommendation.from_dict(data)

        assert rec.quality_impact_score == 0.25
        assert rec.risk_level == "high"

    def test_from_dict_metadata(self):
        """Test from_dict parses metadata."""
        data = {
            "metadata": {"source": "manual", "version": 2},
        }

        rec = OptimizationRecommendation.from_dict(data)

        assert rec.metadata["source"] == "manual"
        assert rec.metadata["version"] == 2

    def test_from_dict_default_values(self):
        """Test from_dict uses defaults for missing fields."""
        data = {}

        rec = OptimizationRecommendation.from_dict(data)

        assert rec.type == RecommendationType.MODEL_DOWNGRADE
        assert rec.priority == RecommendationPriority.MEDIUM
        assert rec.status == RecommendationStatus.PENDING
        assert rec.current_cost_usd == Decimal("0")
        assert rec.workspace_id == ""
        assert rec.affected_agents == []

    def test_from_dict_auto_generates_id(self):
        """Test from_dict generates id if not provided."""
        data = {}
        rec = OptimizationRecommendation.from_dict(data)
        # Should be a valid UUID
        UUID(rec.id)

    def test_roundtrip_to_dict_from_dict(self, sample_recommendation):
        """Test to_dict/from_dict roundtrip preserves key data."""
        data = sample_recommendation.to_dict()
        restored = OptimizationRecommendation.from_dict(data)

        assert restored.id == sample_recommendation.id
        assert restored.type == sample_recommendation.type
        assert restored.priority == sample_recommendation.priority
        assert restored.current_cost_usd == sample_recommendation.current_cost_usd
        assert restored.workspace_id == sample_recommendation.workspace_id
        assert restored.title == sample_recommendation.title


# =============================================================================
# RecommendationSummary Tests
# =============================================================================


class TestRecommendationSummary:
    """Tests for RecommendationSummary dataclass."""

    def test_create_summary(self):
        """Test creating a recommendation summary."""
        summary = RecommendationSummary(
            workspace_id="ws-123",
            total_recommendations=10,
            pending_count=5,
            applied_count=3,
            dismissed_count=2,
            total_potential_savings=Decimal("500.00"),
            realized_savings=Decimal("150.00"),
            critical_count=1,
            high_count=3,
            medium_count=4,
            low_count=2,
            by_type={"model_downgrade": 4, "caching": 3, "batching": 3},
        )

        assert summary.workspace_id == "ws-123"
        assert summary.total_recommendations == 10
        assert summary.pending_count == 5
        assert summary.applied_count == 3
        assert summary.dismissed_count == 2

    def test_summary_default_values(self):
        """Test summary default values."""
        summary = RecommendationSummary(workspace_id="ws-456")

        assert summary.total_recommendations == 0
        assert summary.pending_count == 0
        assert summary.applied_count == 0
        assert summary.dismissed_count == 0
        assert summary.total_potential_savings == Decimal("0")
        assert summary.realized_savings == Decimal("0")
        assert summary.critical_count == 0
        assert summary.high_count == 0
        assert summary.medium_count == 0
        assert summary.low_count == 0
        assert summary.by_type == {}

    def test_summary_savings(self):
        """Test summary savings tracking."""
        summary = RecommendationSummary(
            workspace_id="ws-123",
            total_potential_savings=Decimal("1000.00"),
            realized_savings=Decimal("300.00"),
        )

        assert summary.total_potential_savings == Decimal("1000.00")
        assert summary.realized_savings == Decimal("300.00")

    def test_summary_priority_counts(self):
        """Test summary priority count totals."""
        summary = RecommendationSummary(
            workspace_id="ws-123",
            total_recommendations=20,
            critical_count=2,
            high_count=5,
            medium_count=8,
            low_count=5,
        )

        total = (
            summary.critical_count + summary.high_count + summary.medium_count + summary.low_count
        )
        assert total == 20


class TestRecommendationSummaryToDict:
    """Tests for RecommendationSummary to_dict method."""

    def test_to_dict_basic(self):
        """Test to_dict serialization."""
        summary = RecommendationSummary(
            workspace_id="ws-123",
            total_recommendations=5,
            pending_count=3,
            applied_count=1,
            dismissed_count=1,
        )

        data = summary.to_dict()

        assert data["workspace_id"] == "ws-123"
        assert data["total_recommendations"] == 5
        assert data["pending_count"] == 3
        assert data["applied_count"] == 1
        assert data["dismissed_count"] == 1

    def test_to_dict_savings_as_strings(self):
        """Test to_dict converts savings to strings."""
        summary = RecommendationSummary(
            workspace_id="ws-123",
            total_potential_savings=Decimal("1234.56"),
            realized_savings=Decimal("789.01"),
        )

        data = summary.to_dict()

        assert data["total_potential_savings_usd"] == "1234.56"
        assert data["realized_savings_usd"] == "789.01"

    def test_to_dict_by_priority(self):
        """Test to_dict includes priority breakdown."""
        summary = RecommendationSummary(
            workspace_id="ws-123",
            critical_count=1,
            high_count=2,
            medium_count=3,
            low_count=4,
        )

        data = summary.to_dict()

        assert data["by_priority"]["critical"] == 1
        assert data["by_priority"]["high"] == 2
        assert data["by_priority"]["medium"] == 3
        assert data["by_priority"]["low"] == 4

    def test_to_dict_by_type(self):
        """Test to_dict includes type breakdown."""
        summary = RecommendationSummary(
            workspace_id="ws-123",
            by_type={
                "model_downgrade": 5,
                "caching": 3,
                "batching": 2,
            },
        )

        data = summary.to_dict()

        assert data["by_type"]["model_downgrade"] == 5
        assert data["by_type"]["caching"] == 3
        assert data["by_type"]["batching"] == 2


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_recommendation_with_empty_lists(self):
        """Test recommendation with empty list fields."""
        rec = OptimizationRecommendation(
            affected_agents=[],
            affected_operations=[],
            implementation_steps=[],
        )

        assert rec.affected_agents == []
        assert rec.affected_operations == []
        assert rec.implementation_steps == []

        data = rec.to_dict()
        assert data["affected_agents"] == []
        assert data["implementation_steps"] == []

    def test_recommendation_with_zero_costs(self):
        """Test recommendation with zero cost values."""
        rec = OptimizationRecommendation(
            current_cost_usd=Decimal("0"),
            projected_cost_usd=Decimal("0"),
        )

        assert rec.estimated_savings_usd == Decimal("0")
        assert rec.savings_percentage == 0.0

    def test_recommendation_negative_savings(self):
        """Test recommendation where projected exceeds current (cost increase)."""
        rec = OptimizationRecommendation(
            current_cost_usd=Decimal("50.00"),
            projected_cost_usd=Decimal("75.00"),
        )

        # Savings should be negative (cost increase)
        assert rec.estimated_savings_usd == Decimal("-25.00")
        assert rec.savings_percentage == -50.0

    def test_recommendation_large_values(self):
        """Test recommendation with large cost values."""
        rec = OptimizationRecommendation(
            current_cost_usd=Decimal("1000000.00"),
            projected_cost_usd=Decimal("100000.00"),
        )

        assert rec.estimated_savings_usd == Decimal("900000.00")
        assert rec.savings_percentage == 90.0

    def test_recommendation_small_decimal_values(self):
        """Test recommendation with very small decimal values."""
        rec = OptimizationRecommendation(
            current_cost_usd=Decimal("0.0001"),
            projected_cost_usd=Decimal("0.00005"),
        )

        assert rec.estimated_savings_usd == Decimal("0.00005")

    def test_from_dict_invalid_type(self):
        """Test from_dict with invalid recommendation type."""
        data = {"type": "invalid_type"}

        with pytest.raises(ValueError):
            OptimizationRecommendation.from_dict(data)

    def test_from_dict_invalid_priority(self):
        """Test from_dict with invalid priority."""
        data = {"priority": "super_high"}

        with pytest.raises(ValueError):
            OptimizationRecommendation.from_dict(data)

    def test_from_dict_invalid_status(self):
        """Test from_dict with invalid status."""
        data = {"status": "in_progress"}

        with pytest.raises(ValueError):
            OptimizationRecommendation.from_dict(data)

    def test_model_alternative_extreme_latency(self):
        """Test model alternative with extreme latency multiplier."""
        alt = ModelAlternative(
            provider="slow",
            model="slow-model",
            cost_per_1k_input=Decimal("0.001"),
            cost_per_1k_output=Decimal("0.002"),
            quality_score=0.95,
            latency_multiplier=10.0,  # 10x slower
        )

        assert alt.latency_multiplier == 10.0

    def test_caching_opportunity_zero_repeat(self):
        """Test caching opportunity with zero repeats."""
        opportunity = CachingOpportunity(
            pattern="unique",
            estimated_hit_rate=0.0,
            unique_queries=1000,
            repeat_count=0,
            cache_strategy="exact",
        )

        assert opportunity.repeat_count == 0
        assert opportunity.estimated_hit_rate == 0.0

    def test_batching_opportunity_same_batch_sizes(self):
        """Test batching where current equals optimal (no improvement)."""
        opportunity = BatchingOpportunity(
            operation_type="already_optimal",
            current_batch_size=10,
            optimal_batch_size=10,
            requests_per_hour=100,
            latency_impact_ms=0.0,
        )

        assert opportunity.current_batch_size == opportunity.optimal_batch_size

    def test_implementation_step_empty_description(self):
        """Test implementation step with empty description."""
        step = ImplementationStep(
            order=1,
            description="",
        )

        assert step.description == ""

    def test_summary_all_dismissed(self):
        """Test summary where all recommendations are dismissed."""
        summary = RecommendationSummary(
            workspace_id="ws-123",
            total_recommendations=10,
            pending_count=0,
            applied_count=0,
            dismissed_count=10,
            total_potential_savings=Decimal("0"),
            realized_savings=Decimal("0"),
        )

        assert summary.dismissed_count == summary.total_recommendations
        assert summary.realized_savings == Decimal("0")

    def test_recommendation_metadata_complex(self):
        """Test recommendation with complex metadata."""
        rec = OptimizationRecommendation(
            metadata={
                "analysis_id": "ana-123",
                "confidence_breakdown": {
                    "model": 0.9,
                    "cost": 0.85,
                    "quality": 0.8,
                },
                "tags": ["auto", "verified", "high-impact"],
                "nested": {
                    "level1": {
                        "level2": "deep_value",
                    },
                },
            },
        )

        assert rec.metadata["analysis_id"] == "ana-123"
        assert rec.metadata["confidence_breakdown"]["model"] == 0.9
        assert "auto" in rec.metadata["tags"]
        assert rec.metadata["nested"]["level1"]["level2"] == "deep_value"


class TestNullInputs:
    """Tests for handling null/None inputs."""

    def test_recommendation_none_org_id(self):
        """Test recommendation with None org_id."""
        rec = OptimizationRecommendation(
            workspace_id="ws-123",
            org_id=None,
        )

        assert rec.org_id is None
        data = rec.to_dict()
        assert data["org_id"] is None

    def test_recommendation_none_model_alternative(self):
        """Test recommendation with None model_alternative."""
        rec = OptimizationRecommendation(
            model_alternative=None,
        )

        assert rec.model_alternative is None
        data = rec.to_dict()
        assert data["model_alternative"] is None

    def test_recommendation_none_expires_at(self):
        """Test recommendation with None expires_at."""
        rec = OptimizationRecommendation(
            expires_at=None,
        )

        assert rec.expires_at is None
        data = rec.to_dict()
        assert data["expires_at"] is None

    def test_from_dict_none_org_id(self):
        """Test from_dict with None org_id."""
        data = {"org_id": None}
        rec = OptimizationRecommendation.from_dict(data)
        assert rec.org_id is None


# =============================================================================
# All Exports Tests
# =============================================================================


class TestExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test __all__ exports are accessible."""
        from aragora.billing import recommendations

        for name in recommendations.__all__:
            assert hasattr(recommendations, name)

    def test_all_exports_count(self):
        """Test expected number of exports."""
        from aragora.billing.recommendations import __all__

        assert len(__all__) == 9
        assert "RecommendationType" in __all__
        assert "RecommendationPriority" in __all__
        assert "RecommendationStatus" in __all__
        assert "ModelAlternative" in __all__
        assert "CachingOpportunity" in __all__
        assert "BatchingOpportunity" in __all__
        assert "ImplementationStep" in __all__
        assert "OptimizationRecommendation" in __all__
        assert "RecommendationSummary" in __all__
