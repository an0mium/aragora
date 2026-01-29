"""Tests for model registry.

Tests the model registry including:
- ModelMetadata: dataclass validation and serialization
- ModelRegistry: CRUD operations, filtering, persistence
- Domain-based model selection
- Metrics updates and status management
"""

import json
import tempfile
from pathlib import Path

import pytest

from aragora.training.model_registry import (
    ModelMetadata,
    ModelRegistry,
    get_registry,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def registry_path(temp_dir):
    """Create a path for the registry file."""
    return temp_dir / "test_registry.json"


@pytest.fixture
def registry(registry_path):
    """Create a fresh model registry."""
    return ModelRegistry(registry_path)


@pytest.fixture
def sample_metadata():
    """Create sample model metadata."""
    return ModelMetadata(
        model_id="test-model-001",
        base_model="llama-3.3-70b",
        adapter_name="test-adapter",
        training_type="sft",
    )


@pytest.fixture
def populated_registry(registry):
    """Create a registry with multiple models."""
    models = [
        ModelMetadata(
            model_id="model-sft-1",
            base_model="llama-3.3-70b",
            adapter_name="sft-adapter-1",
            training_type="sft",
            elo_rating=1200,
            status="active",
            primary_domain="security",
        ),
        ModelMetadata(
            model_id="model-dpo-1",
            base_model="llama-3.3-70b",
            adapter_name="dpo-adapter-1",
            training_type="dpo",
            elo_rating=1300,
            status="active",
            tags=["production"],
        ),
        ModelMetadata(
            model_id="model-deprecated",
            base_model="qwen-2.5-72b",
            adapter_name="old-adapter",
            training_type="sft",
            elo_rating=1100,
            status="deprecated",
        ),
    ]

    for model in models:
        registry.register(model)

    return registry


# =============================================================================
# ModelMetadata Tests
# =============================================================================


class TestModelMetadata:
    """Test ModelMetadata dataclass."""

    def test_create_minimal(self):
        """Test creating metadata with minimal fields."""
        meta = ModelMetadata(
            model_id="test-001",
            base_model="llama",
            adapter_name="adapter",
            training_type="sft",
        )

        assert meta.model_id == "test-001"
        assert meta.base_model == "llama"
        assert meta.adapter_name == "adapter"
        assert meta.training_type == "sft"
        assert meta.status == "active"

    def test_create_with_metrics(self):
        """Test creating metadata with performance metrics."""
        meta = ModelMetadata(
            model_id="test-002",
            base_model="llama",
            adapter_name="adapter",
            training_type="dpo",
            elo_rating=1500,
            win_rate=0.65,
            calibration_score=0.85,
        )

        assert meta.elo_rating == 1500
        assert meta.win_rate == 0.65
        assert meta.calibration_score == 0.85

    def test_create_with_domain_scores(self):
        """Test creating metadata with domain-specific scores."""
        meta = ModelMetadata(
            model_id="test-003",
            base_model="llama",
            adapter_name="adapter",
            training_type="sft",
            primary_domain="security",
            domain_scores={"security": 1550, "architecture": 1480},
        )

        assert meta.primary_domain == "security"
        assert meta.domain_scores["security"] == 1550
        assert meta.domain_scores["architecture"] == 1480

    def test_create_with_training_info(self):
        """Test creating metadata with training information."""
        meta = ModelMetadata(
            model_id="test-004",
            base_model="llama",
            adapter_name="adapter",
            training_type="sft",
            training_job_id="job-123",
            checkpoint_path="/path/to/checkpoint",
            final_loss=0.05,
            training_steps=5000,
            training_time_seconds=3600,
            training_data_size=10000,
            training_data_source="debates-v1",
        )

        assert meta.training_job_id == "job-123"
        assert meta.checkpoint_path == "/path/to/checkpoint"
        assert meta.final_loss == 0.05
        assert meta.training_steps == 5000

    def test_to_dict(self, sample_metadata):
        """Test serialization to dictionary."""
        data = sample_metadata.to_dict()

        assert data["model_id"] == "test-model-001"
        assert data["base_model"] == "llama-3.3-70b"
        assert data["adapter_name"] == "test-adapter"
        assert data["training_type"] == "sft"
        assert data["status"] == "active"
        assert "created_at" in data

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "model_id": "test-from-dict",
            "base_model": "qwen",
            "adapter_name": "from-dict-adapter",
            "training_type": "dpo",
            "elo_rating": 1400,
            "tags": ["test", "sample"],
            "domain_scores": {"legal": 1420},
        }

        meta = ModelMetadata.from_dict(data)

        assert meta.model_id == "test-from-dict"
        assert meta.base_model == "qwen"
        assert meta.elo_rating == 1400
        assert "test" in meta.tags
        assert meta.domain_scores["legal"] == 1420

    def test_from_dict_with_defaults(self):
        """Test deserialization uses defaults for missing fields."""
        data = {
            "model_id": "minimal",
            "base_model": "llama",
            "adapter_name": "adapter",
            "training_type": "sft",
        }

        meta = ModelMetadata.from_dict(data)

        assert meta.status == "active"
        assert meta.tags == []
        assert meta.domain_scores == {}
        assert meta.training_steps == 0

    def test_roundtrip_serialization(self, sample_metadata):
        """Test roundtrip through to_dict and from_dict."""
        sample_metadata.elo_rating = 1350
        sample_metadata.tags = ["test", "roundtrip"]

        data = sample_metadata.to_dict()
        restored = ModelMetadata.from_dict(data)

        assert restored.model_id == sample_metadata.model_id
        assert restored.elo_rating == sample_metadata.elo_rating
        assert restored.tags == sample_metadata.tags


# =============================================================================
# ModelRegistry Initialization Tests
# =============================================================================


class TestModelRegistryInit:
    """Test ModelRegistry initialization."""

    def test_create_empty_registry(self, registry_path):
        """Test creating new empty registry."""
        registry = ModelRegistry(registry_path)

        assert len(registry._models) == 0

    def test_load_existing_registry(self, temp_dir):
        """Test loading existing registry file."""
        registry_path = temp_dir / "existing.json"

        # Create a registry file
        data = {
            "models": [
                {
                    "model_id": "existing-model",
                    "base_model": "llama",
                    "adapter_name": "adapter",
                    "training_type": "sft",
                }
            ],
            "updated_at": "2024-01-01T00:00:00",
        }
        with open(registry_path, "w") as f:
            json.dump(data, f)

        # Load it
        registry = ModelRegistry(registry_path)

        assert "existing-model" in registry._models

    def test_handle_corrupt_registry(self, temp_dir):
        """Test handling corrupt registry file."""
        registry_path = temp_dir / "corrupt.json"

        # Create invalid JSON
        with open(registry_path, "w") as f:
            f.write("invalid json {{{")

        # Should handle gracefully
        registry = ModelRegistry(registry_path)

        assert len(registry._models) == 0


# =============================================================================
# ModelRegistry CRUD Tests
# =============================================================================


class TestModelRegistryCRUD:
    """Test CRUD operations on ModelRegistry."""

    def test_register_new_model(self, registry, sample_metadata):
        """Test registering a new model."""
        registry.register(sample_metadata)

        assert sample_metadata.model_id in registry._models
        assert registry.get(sample_metadata.model_id) == sample_metadata

    def test_register_updates_existing(self, registry, sample_metadata):
        """Test registering updates existing model."""
        registry.register(sample_metadata)

        # Update and re-register
        sample_metadata.elo_rating = 1500
        registry.register(sample_metadata)

        model = registry.get(sample_metadata.model_id)
        assert model.elo_rating == 1500

    def test_get_existing_model(self, populated_registry):
        """Test getting existing model."""
        model = populated_registry.get("model-sft-1")

        assert model is not None
        assert model.model_id == "model-sft-1"

    def test_get_nonexistent_model(self, registry):
        """Test getting nonexistent model returns None."""
        model = registry.get("nonexistent")

        assert model is None

    def test_delete_model(self, populated_registry):
        """Test deleting a model."""
        result = populated_registry.delete("model-sft-1")

        assert result is True
        assert populated_registry.get("model-sft-1") is None

    def test_delete_nonexistent(self, registry):
        """Test deleting nonexistent model returns False."""
        result = registry.delete("nonexistent")

        assert result is False


# =============================================================================
# ModelRegistry Listing and Filtering Tests
# =============================================================================


class TestModelRegistryListing:
    """Test listing and filtering models."""

    def test_list_all_models(self, populated_registry):
        """Test listing all models."""
        models = populated_registry.list_models()

        assert len(models) == 3

    def test_list_by_status(self, populated_registry):
        """Test filtering by status."""
        active = populated_registry.list_models(status="active")
        deprecated = populated_registry.list_models(status="deprecated")

        assert len(active) == 2
        assert len(deprecated) == 1
        assert all(m.status == "active" for m in active)

    def test_list_by_training_type(self, populated_registry):
        """Test filtering by training type."""
        sft_models = populated_registry.list_models(training_type="sft")
        dpo_models = populated_registry.list_models(training_type="dpo")

        assert len(sft_models) == 2
        assert len(dpo_models) == 1

    def test_list_by_base_model(self, populated_registry):
        """Test filtering by base model."""
        llama_models = populated_registry.list_models(base_model="llama-3.3-70b")
        qwen_models = populated_registry.list_models(base_model="qwen-2.5-72b")

        assert len(llama_models) == 2
        assert len(qwen_models) == 1

    def test_list_by_tag(self, populated_registry):
        """Test filtering by tag."""
        production = populated_registry.list_models(tag="production")

        assert len(production) == 1
        assert production[0].model_id == "model-dpo-1"

    def test_list_with_limit(self, populated_registry):
        """Test listing with limit."""
        limited = populated_registry.list_models(limit=1)

        assert len(limited) == 1

    def test_list_sorted_by_elo(self, populated_registry):
        """Test models are sorted by ELO rating."""
        models = populated_registry.list_models(status="active")

        # Should be sorted by ELO (highest first)
        elo_ratings = [m.elo_rating or 0 for m in models]
        assert elo_ratings == sorted(elo_ratings, reverse=True)


# =============================================================================
# ModelRegistry Domain Selection Tests
# =============================================================================


class TestDomainSelection:
    """Test domain-based model selection."""

    def test_get_best_for_domain_primary(self, registry):
        """Test getting best model for primary domain."""
        model = ModelMetadata(
            model_id="security-expert",
            base_model="llama",
            adapter_name="security-adapter",
            training_type="sft",
            primary_domain="security",
            elo_rating=1400,
        )
        registry.register(model)

        best = registry.get_best_for_domain("security")

        assert best is not None
        assert best.model_id == "security-expert"

    def test_get_best_for_domain_with_scores(self, registry):
        """Test getting best model using domain scores."""
        model1 = ModelMetadata(
            model_id="model-a",
            base_model="llama",
            adapter_name="adapter-a",
            training_type="sft",
            elo_rating=1300,
            domain_scores={"legal": 1500},
        )
        model2 = ModelMetadata(
            model_id="model-b",
            base_model="llama",
            adapter_name="adapter-b",
            training_type="sft",
            elo_rating=1400,
            domain_scores={"legal": 1200},
        )

        registry.register(model1)
        registry.register(model2)

        best = registry.get_best_for_domain("legal")

        # model-a has higher domain score for legal
        assert best.model_id == "model-a"

    def test_get_best_for_domain_no_models(self, registry):
        """Test returns None when no models exist."""
        best = registry.get_best_for_domain("nonexistent")

        assert best is None

    def test_get_best_for_domain_only_active(self, registry):
        """Test only considers active models."""
        model = ModelMetadata(
            model_id="deprecated-model",
            base_model="llama",
            adapter_name="adapter",
            training_type="sft",
            primary_domain="security",
            elo_rating=1600,
            status="deprecated",
        )
        registry.register(model)

        best = registry.get_best_for_domain("security")

        # Should return None since only model is deprecated
        assert best is None

    def test_get_latest(self, populated_registry):
        """Test getting most recently created model."""
        latest = populated_registry.get_latest()

        assert latest is not None

    def test_get_latest_by_type(self, populated_registry):
        """Test getting latest by training type."""
        latest_dpo = populated_registry.get_latest(training_type="dpo")

        assert latest_dpo is not None
        assert latest_dpo.training_type == "dpo"


# =============================================================================
# ModelRegistry Metrics Update Tests
# =============================================================================


class TestMetricsUpdate:
    """Test metrics update functionality."""

    def test_update_elo_rating(self, registry, sample_metadata):
        """Test updating ELO rating."""
        registry.register(sample_metadata)

        result = registry.update_metrics(sample_metadata.model_id, elo_rating=1500)

        assert result is True
        assert registry.get(sample_metadata.model_id).elo_rating == 1500

    def test_update_win_rate(self, registry, sample_metadata):
        """Test updating win rate."""
        registry.register(sample_metadata)

        result = registry.update_metrics(sample_metadata.model_id, win_rate=0.7)

        assert result is True
        assert registry.get(sample_metadata.model_id).win_rate == 0.7

    def test_update_calibration_score(self, registry, sample_metadata):
        """Test updating calibration score."""
        registry.register(sample_metadata)

        result = registry.update_metrics(sample_metadata.model_id, calibration_score=0.9)

        assert result is True
        assert registry.get(sample_metadata.model_id).calibration_score == 0.9

    def test_update_domain_scores(self, registry, sample_metadata):
        """Test updating domain scores."""
        registry.register(sample_metadata)

        result = registry.update_metrics(
            sample_metadata.model_id,
            domain_scores={"security": 1450, "legal": 1300},
        )

        assert result is True
        model = registry.get(sample_metadata.model_id)
        assert model.domain_scores["security"] == 1450
        assert model.domain_scores["legal"] == 1300

    def test_update_multiple_metrics(self, registry, sample_metadata):
        """Test updating multiple metrics at once."""
        registry.register(sample_metadata)

        result = registry.update_metrics(
            sample_metadata.model_id,
            elo_rating=1500,
            win_rate=0.65,
            calibration_score=0.85,
        )

        assert result is True
        model = registry.get(sample_metadata.model_id)
        assert model.elo_rating == 1500
        assert model.win_rate == 0.65
        assert model.calibration_score == 0.85

    def test_update_nonexistent_model(self, registry):
        """Test updating nonexistent model returns False."""
        result = registry.update_metrics("nonexistent", elo_rating=1500)

        assert result is False


# =============================================================================
# ModelRegistry Status Management Tests
# =============================================================================


class TestStatusManagement:
    """Test model status management."""

    def test_deprecate_model(self, registry, sample_metadata):
        """Test deprecating a model."""
        registry.register(sample_metadata)

        result = registry.deprecate(sample_metadata.model_id, notes="Replaced by v2")

        assert result is True
        model = registry.get(sample_metadata.model_id)
        assert model.status == "deprecated"
        assert model.notes == "Replaced by v2"

    def test_deprecate_nonexistent(self, registry):
        """Test deprecating nonexistent model."""
        result = registry.deprecate("nonexistent")

        assert result is False

    def test_archive_model(self, registry, sample_metadata):
        """Test archiving a model."""
        registry.register(sample_metadata)

        result = registry.archive(sample_metadata.model_id)

        assert result is True
        assert registry.get(sample_metadata.model_id).status == "archived"

    def test_archive_nonexistent(self, registry):
        """Test archiving nonexistent model."""
        result = registry.archive("nonexistent")

        assert result is False


# =============================================================================
# ModelRegistry Tag Management Tests
# =============================================================================


class TestTagManagement:
    """Test tag management functionality."""

    def test_add_tag(self, registry, sample_metadata):
        """Test adding a tag."""
        registry.register(sample_metadata)

        result = registry.add_tag(sample_metadata.model_id, "production")

        assert result is True
        assert "production" in registry.get(sample_metadata.model_id).tags

    def test_add_duplicate_tag(self, registry, sample_metadata):
        """Test adding duplicate tag is idempotent."""
        registry.register(sample_metadata)
        registry.add_tag(sample_metadata.model_id, "production")
        registry.add_tag(sample_metadata.model_id, "production")

        tags = registry.get(sample_metadata.model_id).tags
        assert tags.count("production") == 1

    def test_remove_tag(self, registry, sample_metadata):
        """Test removing a tag."""
        sample_metadata.tags = ["production", "tested"]
        registry.register(sample_metadata)

        result = registry.remove_tag(sample_metadata.model_id, "production")

        assert result is True
        assert "production" not in registry.get(sample_metadata.model_id).tags
        assert "tested" in registry.get(sample_metadata.model_id).tags

    def test_remove_nonexistent_tag(self, registry, sample_metadata):
        """Test removing nonexistent tag."""
        registry.register(sample_metadata)

        result = registry.remove_tag(sample_metadata.model_id, "nonexistent")

        assert result is True  # Still returns True (idempotent)


# =============================================================================
# ModelRegistry Statistics Tests
# =============================================================================


class TestRegistryStatistics:
    """Test registry statistics."""

    def test_get_stats(self, populated_registry):
        """Test getting registry statistics."""
        stats = populated_registry.get_stats()

        assert stats["total_models"] == 3
        assert "by_status" in stats
        assert "by_training_type" in stats
        assert "by_base_model" in stats
        assert "average_elo" in stats

    def test_stats_by_status(self, populated_registry):
        """Test statistics breakdown by status."""
        stats = populated_registry.get_stats()

        assert stats["by_status"]["active"] == 2
        assert stats["by_status"]["deprecated"] == 1

    def test_stats_by_training_type(self, populated_registry):
        """Test statistics breakdown by training type."""
        stats = populated_registry.get_stats()

        assert stats["by_training_type"]["sft"] == 2
        assert stats["by_training_type"]["dpo"] == 1

    def test_stats_empty_registry(self, registry):
        """Test statistics for empty registry."""
        stats = registry.get_stats()

        assert stats["total_models"] == 0
        assert stats["average_elo"] == 0


# =============================================================================
# ModelRegistry Persistence Tests
# =============================================================================


class TestRegistryPersistence:
    """Test registry persistence."""

    def test_save_on_register(self, registry, sample_metadata, registry_path):
        """Test that register saves to file."""
        registry.register(sample_metadata)

        assert registry_path.exists()

        with open(registry_path) as f:
            data = json.load(f)

        assert len(data["models"]) == 1
        assert data["models"][0]["model_id"] == sample_metadata.model_id

    def test_save_on_update(self, registry, sample_metadata, registry_path):
        """Test that updates save to file."""
        registry.register(sample_metadata)
        registry.update_metrics(sample_metadata.model_id, elo_rating=1500)

        with open(registry_path) as f:
            data = json.load(f)

        assert data["models"][0]["elo_rating"] == 1500

    def test_persistence_roundtrip(self, temp_dir, sample_metadata):
        """Test full persistence roundtrip."""
        path = temp_dir / "roundtrip.json"

        # Create and populate registry
        registry1 = ModelRegistry(path)
        sample_metadata.elo_rating = 1500
        sample_metadata.tags = ["test"]
        registry1.register(sample_metadata)

        # Load in new registry
        registry2 = ModelRegistry(path)

        model = registry2.get(sample_metadata.model_id)
        assert model is not None
        assert model.elo_rating == 1500
        assert "test" in model.tags


# =============================================================================
# Global Registry Function Tests
# =============================================================================


class TestGlobalRegistry:
    """Test global registry function."""

    def test_get_registry_creates_instance(self, temp_dir):
        """Test get_registry creates instance."""
        path = temp_dir / "global_test.json"

        registry = get_registry(path)

        assert registry is not None
        assert isinstance(registry, ModelRegistry)

    def test_get_registry_with_custom_path(self, temp_dir):
        """Test get_registry with custom path."""
        path = temp_dir / "custom.json"

        registry = get_registry(path)

        # Register a model to create the file
        registry.register(
            ModelMetadata(
                model_id="custom-test",
                base_model="llama",
                adapter_name="adapter",
                training_type="sft",
            )
        )

        assert path.exists()
