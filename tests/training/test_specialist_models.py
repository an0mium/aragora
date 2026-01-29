"""Tests for specialist models.

Tests the specialist model system including:
- Vertical: enterprise vertical enum
- TrainingStatus: training job status enum
- SpecialistModelConfig: configuration dataclass
- SpecialistModel: specialist model dataclass
- SpecialistModelRegistry: registry for specialist models
- SpecialistTrainingPipeline: training orchestration
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.training.specialist_models import (
    SpecialistModel,
    SpecialistModelConfig,
    SpecialistModelRegistry,
    SpecialistTrainingPipeline,
    TrainingStatus,
    Vertical,
    VERTICAL_DEFAULTS,
    VERTICAL_KEYWORDS,
    get_specialist_registry,
    get_vertical_config,
)
from aragora.training.model_registry import ModelMetadata, ModelRegistry


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def base_registry():
    """Create a mock base model registry."""
    registry = MagicMock(spec=ModelRegistry)
    registry._models = {}
    return registry


@pytest.fixture
def specialist_registry(base_registry):
    """Create a specialist model registry."""
    return SpecialistModelRegistry(base_registry)


@pytest.fixture
def sample_config():
    """Create a sample specialist model config."""
    return SpecialistModelConfig(
        vertical=Vertical.SECURITY,
        base_model="llama-3.3-70b",
        adapter_name="security-adapter",
        min_confidence=0.8,
    )


@pytest.fixture
def sample_model():
    """Create a sample specialist model."""
    return SpecialistModel(
        id="sm_security_abc123",
        base_model="llama-3.3-70b",
        adapter_name="security-expert",
        vertical=Vertical.SECURITY,
        org_id=None,
        status=TrainingStatus.READY,
        elo_rating=1400,
    )


@pytest.fixture
def populated_registry(specialist_registry):
    """Create a registry with multiple models."""
    models = [
        SpecialistModel(
            id="sm_security_001",
            base_model="llama-3.3-70b",
            adapter_name="security-v1",
            vertical=Vertical.SECURITY,
            org_id=None,
            status=TrainingStatus.READY,
            elo_rating=1400,
        ),
        SpecialistModel(
            id="sm_security_002",
            base_model="llama-3.3-70b",
            adapter_name="security-v2",
            vertical=Vertical.SECURITY,
            org_id="org-123",
            status=TrainingStatus.READY,
            elo_rating=1500,
        ),
        SpecialistModel(
            id="sm_legal_001",
            base_model="llama-3.3-70b",
            adapter_name="legal-v1",
            vertical=Vertical.LEGAL,
            org_id=None,
            status=TrainingStatus.READY,
            elo_rating=1350,
        ),
        SpecialistModel(
            id="sm_healthcare_001",
            base_model="qwen-2.5-72b",
            adapter_name="healthcare-v1",
            vertical=Vertical.HEALTHCARE,
            org_id=None,
            status=TrainingStatus.TRAINING,
            elo_rating=None,
        ),
    ]

    for model in models:
        specialist_registry.register(model)

    return specialist_registry


# =============================================================================
# Vertical Enum Tests
# =============================================================================


class TestVertical:
    """Test Vertical enum."""

    def test_vertical_values(self):
        """Test vertical enum values."""
        assert Vertical.LEGAL.value == "legal"
        assert Vertical.HEALTHCARE.value == "healthcare"
        assert Vertical.SECURITY.value == "security"
        assert Vertical.ACCOUNTING.value == "accounting"
        assert Vertical.REGULATORY.value == "regulatory"
        assert Vertical.ACADEMIC.value == "academic"
        assert Vertical.SOFTWARE.value == "software"
        assert Vertical.GENERAL.value == "general"

    def test_vertical_is_string(self):
        """Test vertical is string subclass."""
        assert isinstance(Vertical.LEGAL, str)
        assert Vertical.LEGAL == "legal"

    def test_all_verticals_have_keywords(self):
        """Test all verticals (except GENERAL) have keywords defined."""
        for vertical in Vertical:
            if vertical != Vertical.GENERAL:
                assert vertical.value in VERTICAL_KEYWORDS, f"Missing keywords for {vertical}"


# =============================================================================
# TrainingStatus Enum Tests
# =============================================================================


class TestTrainingStatus:
    """Test TrainingStatus enum."""

    def test_status_values(self):
        """Test training status enum values."""
        assert TrainingStatus.PENDING.value == "pending"
        assert TrainingStatus.EXPORTING_DATA.value == "exporting_data"
        assert TrainingStatus.TRAINING.value == "training"
        assert TrainingStatus.EVALUATING.value == "evaluating"
        assert TrainingStatus.READY.value == "ready"
        assert TrainingStatus.FAILED.value == "failed"
        assert TrainingStatus.DEPRECATED.value == "deprecated"

    def test_status_is_string(self):
        """Test status is string subclass."""
        assert isinstance(TrainingStatus.READY, str)
        assert TrainingStatus.READY == "ready"


# =============================================================================
# SpecialistModelConfig Tests
# =============================================================================


class TestSpecialistModelConfig:
    """Test SpecialistModelConfig dataclass."""

    def test_create_minimal(self):
        """Test creating config with minimal fields."""
        config = SpecialistModelConfig(vertical=Vertical.SECURITY)

        assert config.vertical == Vertical.SECURITY
        assert config.base_model == "llama-3.3-70b"
        assert config.adapter_name is None
        assert config.org_id is None

    def test_create_with_training_params(self):
        """Test creating config with training parameters."""
        config = SpecialistModelConfig(
            vertical=Vertical.LEGAL,
            lora_rank=32,
            learning_rate=5e-5,
            max_steps=2000,
            batch_size=8,
        )

        assert config.lora_rank == 32
        assert config.learning_rate == 5e-5
        assert config.max_steps == 2000
        assert config.batch_size == 8

    def test_create_with_data_filtering(self):
        """Test creating config with data filtering options."""
        config = SpecialistModelConfig(
            vertical=Vertical.HEALTHCARE,
            min_confidence=0.9,
            min_debates=50,
            include_critiques=False,
            include_consensus=False,
        )

        assert config.min_confidence == 0.9
        assert config.min_debates == 50
        assert config.include_critiques is False
        assert config.include_consensus is False

    def test_create_with_org_scope(self):
        """Test creating config with organization scope."""
        config = SpecialistModelConfig(
            vertical=Vertical.ACCOUNTING,
            org_id="org-456",
            workspace_ids=["ws-1", "ws-2"],
        )

        assert config.org_id == "org-456"
        assert config.workspace_ids == ["ws-1", "ws-2"]

    def test_to_dict(self, sample_config):
        """Test serialization to dictionary."""
        data = sample_config.to_dict()

        assert data["vertical"] == "security"
        assert data["base_model"] == "llama-3.3-70b"
        assert data["adapter_name"] == "security-adapter"
        assert data["min_confidence"] == 0.8

    def test_default_values(self):
        """Test default configuration values."""
        config = SpecialistModelConfig(vertical=Vertical.SOFTWARE)

        assert config.lora_rank == 16
        assert config.learning_rate == 1e-4
        assert config.max_steps == 1000
        assert config.batch_size == 4
        assert config.min_confidence == 0.7
        assert config.min_debates == 10
        assert config.eval_split == 0.1
        assert config.run_gauntlet is True


# =============================================================================
# SpecialistModel Tests
# =============================================================================


class TestSpecialistModel:
    """Test SpecialistModel dataclass."""

    def test_create_minimal(self):
        """Test creating model with minimal fields."""
        model = SpecialistModel(
            id="sm_test_001",
            base_model="llama",
            adapter_name="adapter",
            vertical=Vertical.LEGAL,
            org_id=None,
        )

        assert model.id == "sm_test_001"
        assert model.vertical == Vertical.LEGAL
        assert model.status == TrainingStatus.PENDING
        assert model.version == 1

    def test_create_with_training_info(self):
        """Test creating model with training information."""
        model = SpecialistModel(
            id="sm_test_002",
            base_model="llama",
            adapter_name="adapter",
            vertical=Vertical.SECURITY,
            org_id=None,
            training_job_id="job-123",
            training_data_debates=500,
            training_data_examples=5000,
            final_loss=0.05,
        )

        assert model.training_job_id == "job-123"
        assert model.training_data_debates == 500
        assert model.training_data_examples == 5000
        assert model.final_loss == 0.05

    def test_create_with_performance_metrics(self):
        """Test creating model with performance metrics."""
        model = SpecialistModel(
            id="sm_test_003",
            base_model="llama",
            adapter_name="adapter",
            vertical=Vertical.HEALTHCARE,
            org_id=None,
            elo_rating=1400,
            win_rate=0.65,
            calibration_score=0.85,
            vertical_accuracy=0.92,
        )

        assert model.elo_rating == 1400
        assert model.win_rate == 0.65
        assert model.calibration_score == 0.85
        assert model.vertical_accuracy == 0.92

    def test_to_dict(self, sample_model):
        """Test serialization to dictionary."""
        data = sample_model.to_dict()

        assert data["id"] == "sm_security_abc123"
        assert data["vertical"] == "security"
        assert data["status"] == "ready"
        assert data["elo_rating"] == 1400
        assert "created_at" in data
        assert "updated_at" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "id": "sm_legal_001",
            "base_model": "llama",
            "adapter_name": "legal-adapter",
            "vertical": "legal",
            "org_id": "org-123",
            "status": "ready",
            "elo_rating": 1300,
            "tags": ["production"],
        }

        model = SpecialistModel.from_dict(data)

        assert model.id == "sm_legal_001"
        assert model.vertical == Vertical.LEGAL
        assert model.org_id == "org-123"
        assert model.status == TrainingStatus.READY
        assert model.elo_rating == 1300
        assert "production" in model.tags

    def test_to_model_metadata(self, sample_model):
        """Test conversion to base ModelMetadata."""
        metadata = sample_model.to_model_metadata()

        assert isinstance(metadata, ModelMetadata)
        assert metadata.model_id == sample_model.id
        assert metadata.base_model == sample_model.base_model
        assert metadata.training_type == "specialist"
        assert metadata.primary_domain == "security"
        assert f"vertical:{sample_model.vertical.value}" in metadata.tags

    def test_roundtrip_serialization(self, sample_model):
        """Test roundtrip through to_dict and from_dict."""
        sample_model.tags = ["test", "roundtrip"]
        sample_model.notes = "Test notes"

        data = sample_model.to_dict()
        restored = SpecialistModel.from_dict(data)

        assert restored.id == sample_model.id
        assert restored.vertical == sample_model.vertical
        assert restored.tags == sample_model.tags
        assert restored.notes == sample_model.notes


# =============================================================================
# SpecialistModelRegistry Tests
# =============================================================================


class TestSpecialistModelRegistry:
    """Test SpecialistModelRegistry."""

    def test_create_empty_registry(self, base_registry):
        """Test creating empty registry."""
        registry = SpecialistModelRegistry(base_registry)

        assert len(registry._specialists) == 0

    def test_register_model(self, specialist_registry, sample_model):
        """Test registering a model."""
        specialist_registry.register(sample_model)

        assert sample_model.id in specialist_registry._specialists
        assert specialist_registry.get(sample_model.id) == sample_model

    def test_register_syncs_to_base(self, base_registry, sample_model):
        """Test that registration syncs to base registry."""
        registry = SpecialistModelRegistry(base_registry)
        registry.register(sample_model)

        base_registry.register.assert_called_once()

    def test_get_existing_model(self, populated_registry):
        """Test getting existing model."""
        model = populated_registry.get("sm_security_001")

        assert model is not None
        assert model.id == "sm_security_001"

    def test_get_nonexistent_model(self, specialist_registry):
        """Test getting nonexistent model returns None."""
        model = specialist_registry.get("nonexistent")

        assert model is None


# =============================================================================
# SpecialistModelRegistry - Vertical Selection Tests
# =============================================================================


class TestVerticalSelection:
    """Test vertical-based model selection."""

    def test_get_for_vertical_global(self, populated_registry):
        """Test getting global model for vertical."""
        model = populated_registry.get_for_vertical(Vertical.SECURITY)

        assert model is not None
        assert model.vertical == Vertical.SECURITY

    def test_get_for_vertical_org_specific(self, populated_registry):
        """Test getting org-specific model."""
        model = populated_registry.get_for_vertical(
            Vertical.SECURITY,
            org_id="org-123",
        )

        # Should prefer org-specific model
        assert model is not None
        assert model.org_id == "org-123"

    def test_get_for_vertical_only_ready(self, populated_registry):
        """Test only returns ready models."""
        # Healthcare model is in TRAINING status
        model = populated_registry.get_for_vertical(Vertical.HEALTHCARE)

        assert model is None

    def test_get_for_vertical_no_global(self, populated_registry):
        """Test with include_global=False."""
        model = populated_registry.get_for_vertical(
            Vertical.SECURITY,
            org_id="org-456",  # Different org
            include_global=False,
        )

        # No org-456 specific model, and global excluded
        assert model is None

    def test_get_for_vertical_fallback_to_global(self, populated_registry):
        """Test falls back to global when no org-specific."""
        model = populated_registry.get_for_vertical(
            Vertical.LEGAL,
            org_id="org-456",
            include_global=True,
        )

        # Should return global legal model
        assert model is not None
        assert model.vertical == Vertical.LEGAL
        assert model.org_id is None


# =============================================================================
# SpecialistModelRegistry - Listing Tests
# =============================================================================


class TestRegistryListing:
    """Test registry listing functionality."""

    def test_list_for_vertical(self, populated_registry):
        """Test listing models for vertical."""
        models = populated_registry.list_for_vertical(Vertical.SECURITY)

        assert len(models) == 2
        assert all(m.vertical == Vertical.SECURITY for m in models)

    def test_list_for_vertical_with_org(self, populated_registry):
        """Test listing models for vertical with org filter."""
        models = populated_registry.list_for_vertical(
            Vertical.SECURITY,
            org_id="org-123",
        )

        assert len(models) == 1
        assert models[0].org_id == "org-123"

    def test_list_for_vertical_with_status(self, populated_registry):
        """Test listing models with status filter."""
        models = populated_registry.list_for_vertical(
            Vertical.HEALTHCARE,
            status=TrainingStatus.TRAINING,
        )

        assert len(models) == 1
        assert models[0].status == TrainingStatus.TRAINING

    def test_list_for_org(self, populated_registry):
        """Test listing models for organization."""
        models = populated_registry.list_for_org("org-123")

        assert len(models) == 1
        assert models[0].org_id == "org-123"

    def test_list_for_org_with_vertical(self, populated_registry):
        """Test listing models for org with vertical filter."""
        models = populated_registry.list_for_org(
            "org-123",
            vertical=Vertical.SECURITY,
        )

        assert len(models) == 1

    def test_list_for_nonexistent_org(self, populated_registry):
        """Test listing models for nonexistent org."""
        models = populated_registry.list_for_org("nonexistent")

        assert len(models) == 0


# =============================================================================
# SpecialistModelRegistry - Status Update Tests
# =============================================================================


class TestStatusUpdate:
    """Test status update functionality."""

    def test_update_status(self, populated_registry):
        """Test updating model status."""
        result = populated_registry.update_status(
            "sm_healthcare_001",
            TrainingStatus.READY,
        )

        assert result is True
        model = populated_registry.get("sm_healthcare_001")
        assert model.status == TrainingStatus.READY

    def test_update_status_with_metrics(self, populated_registry):
        """Test updating status with metrics."""
        result = populated_registry.update_status(
            "sm_healthcare_001",
            TrainingStatus.READY,
            elo_rating=1400,
            win_rate=0.65,
        )

        assert result is True
        model = populated_registry.get("sm_healthcare_001")
        assert model.elo_rating == 1400
        assert model.win_rate == 0.65

    def test_update_status_nonexistent(self, specialist_registry):
        """Test updating nonexistent model."""
        result = specialist_registry.update_status(
            "nonexistent",
            TrainingStatus.READY,
        )

        assert result is False

    def test_deprecate_model(self, populated_registry):
        """Test deprecating a model."""
        result = populated_registry.deprecate(
            "sm_security_001",
            notes="Replaced by v2",
        )

        assert result is True
        model = populated_registry.get("sm_security_001")
        assert model.status == TrainingStatus.DEPRECATED


# =============================================================================
# SpecialistModelRegistry - Statistics Tests
# =============================================================================


class TestRegistryStats:
    """Test registry statistics."""

    def test_get_stats(self, populated_registry):
        """Test getting statistics."""
        stats = populated_registry.get_stats()

        assert stats["total_models"] == 4
        assert "by_vertical" in stats
        assert "by_status" in stats
        assert "by_org" in stats
        assert "ready_count" in stats
        assert "average_elo" in stats

    def test_stats_by_vertical(self, populated_registry):
        """Test statistics by vertical."""
        stats = populated_registry.get_stats()

        assert stats["by_vertical"]["security"] == 2
        assert stats["by_vertical"]["legal"] == 1
        assert stats["by_vertical"]["healthcare"] == 1

    def test_stats_by_status(self, populated_registry):
        """Test statistics by status."""
        stats = populated_registry.get_stats()

        assert stats["by_status"]["ready"] == 3
        assert stats["by_status"]["training"] == 1

    def test_stats_ready_count(self, populated_registry):
        """Test ready model count."""
        stats = populated_registry.get_stats()

        assert stats["ready_count"] == 3

    def test_stats_empty_registry(self, specialist_registry):
        """Test statistics for empty registry."""
        stats = specialist_registry.get_stats()

        assert stats["total_models"] == 0
        assert stats["ready_count"] == 0
        assert stats["average_elo"] == 0


# =============================================================================
# SpecialistTrainingPipeline Tests
# =============================================================================


class TestSpecialistTrainingPipeline:
    """Test SpecialistTrainingPipeline."""

    @pytest.fixture
    def pipeline(self, specialist_registry):
        """Create a training pipeline."""
        mock_tinker = AsyncMock()
        return SpecialistTrainingPipeline(specialist_registry, mock_tinker)

    @pytest.mark.asyncio
    async def test_create_training_job(self, pipeline, sample_config):
        """Test creating a training job."""
        model = await pipeline.create_training_job(
            sample_config,
            created_by="test-user",
        )

        assert model is not None
        assert model.vertical == Vertical.SECURITY
        assert model.status == TrainingStatus.PENDING
        assert model.created_by == "test-user"

    @pytest.mark.asyncio
    async def test_create_job_generates_id(self, pipeline, sample_config):
        """Test that job creation generates unique ID."""
        model = await pipeline.create_training_job(sample_config, "user")

        assert model.id.startswith("sm_security_")

    @pytest.mark.asyncio
    async def test_create_job_sets_adapter_name(self, pipeline):
        """Test adapter name is auto-generated if not provided."""
        config = SpecialistModelConfig(vertical=Vertical.LEGAL)

        model = await pipeline.create_training_job(config, "user")

        assert model.adapter_name == "aragora-legal-v1"

    @pytest.mark.asyncio
    async def test_create_job_uses_provided_adapter(self, pipeline, sample_config):
        """Test provided adapter name is used."""
        model = await pipeline.create_training_job(sample_config, "user")

        assert model.adapter_name == "security-adapter"

    @pytest.mark.asyncio
    async def test_get_training_status(self, pipeline, specialist_registry, sample_model):
        """Test getting training status."""
        specialist_registry.register(sample_model)

        status = await pipeline.get_training_status(sample_model.id)

        assert status["model_id"] == sample_model.id
        assert status["vertical"] == "security"
        assert status["status"] == "ready"

    @pytest.mark.asyncio
    async def test_get_training_status_not_found(self, pipeline):
        """Test getting status for nonexistent model."""
        with pytest.raises(ValueError, match="Model not found"):
            await pipeline.get_training_status("nonexistent")


# =============================================================================
# Vertical Defaults and Keywords Tests
# =============================================================================


class TestVerticalDefaults:
    """Test vertical defaults and keywords."""

    def test_all_verticals_have_defaults(self):
        """Test all verticals (except GENERAL) have defaults defined."""
        for vertical in Vertical:
            if vertical != Vertical.GENERAL:
                assert vertical in VERTICAL_DEFAULTS, f"Missing defaults for {vertical}"

    def test_vertical_defaults_structure(self):
        """Test vertical defaults have expected structure."""
        for vertical, defaults in VERTICAL_DEFAULTS.items():
            assert "base_model" in defaults
            assert "lora_rank" in defaults
            assert "max_steps" in defaults

    def test_get_vertical_config(self):
        """Test get_vertical_config function."""
        config = get_vertical_config(Vertical.SECURITY)

        assert config.vertical == Vertical.SECURITY
        assert config.base_model == VERTICAL_DEFAULTS[Vertical.SECURITY]["base_model"]
        assert config.lora_rank == VERTICAL_DEFAULTS[Vertical.SECURITY]["lora_rank"]

    def test_get_vertical_config_with_org(self):
        """Test get_vertical_config with organization."""
        config = get_vertical_config(Vertical.LEGAL, org_id="org-123")

        assert config.org_id == "org-123"

    def test_get_vertical_config_with_overrides(self):
        """Test get_vertical_config with overrides."""
        config = get_vertical_config(
            Vertical.HEALTHCARE,
            base_model="custom-model",
            max_steps=5000,
        )

        assert config.base_model == "custom-model"
        assert config.max_steps == 5000


# =============================================================================
# Vertical Keywords Tests
# =============================================================================


class TestVerticalKeywords:
    """Test vertical keywords."""

    def test_keywords_coverage(self):
        """Test each vertical has meaningful keywords."""
        for vertical, keywords in VERTICAL_KEYWORDS.items():
            assert len(keywords) >= 5, f"{vertical} should have at least 5 keywords"

    def test_legal_keywords(self):
        """Test legal vertical keywords."""
        keywords = VERTICAL_KEYWORDS["legal"]

        assert "contract" in keywords
        assert "legal" in keywords
        assert "compliance" in keywords

    def test_healthcare_keywords(self):
        """Test healthcare vertical keywords."""
        keywords = VERTICAL_KEYWORDS["healthcare"]

        assert "medical" in keywords
        assert "patient" in keywords
        assert "hipaa" in keywords

    def test_security_keywords(self):
        """Test security vertical keywords."""
        keywords = VERTICAL_KEYWORDS["security"]

        assert "security" in keywords
        assert "vulnerability" in keywords
        assert "encryption" in keywords


# =============================================================================
# Global Registry Function Tests
# =============================================================================


class TestGlobalSpecialistRegistry:
    """Test global specialist registry function."""

    def test_get_specialist_registry_creates_instance(self):
        """Test get_specialist_registry creates instance."""
        registry = get_specialist_registry()

        assert registry is not None
        assert isinstance(registry, SpecialistModelRegistry)

    def test_get_specialist_registry_with_base(self, base_registry):
        """Test get_specialist_registry with base registry."""
        # Note: This will return the global instance, so we just verify it works
        registry = get_specialist_registry(base_registry)

        assert registry is not None
