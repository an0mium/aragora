"""
Model registry for managing fine-tuned adapters.

Tracks trained models, their metadata, performance metrics,
and provides adapter hot-swapping capabilities.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata for a trained model/adapter."""

    model_id: str
    base_model: str
    adapter_name: str
    training_type: str  # sft, dpo, combined
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    training_job_id: str | None = None
    checkpoint_path: str | None = None
    # Training metrics
    final_loss: float | None = None
    training_steps: int = 0
    training_time_seconds: float = 0
    # Training data info
    training_data_size: int = 0
    training_data_source: str = ""
    # Performance metrics (from evaluation)
    elo_rating: float | None = None
    win_rate: float | None = None
    calibration_score: float | None = None
    # Domain specialization
    primary_domain: str | None = None
    domain_scores: dict[str, float] = field(default_factory=dict)
    # Status
    status: str = "active"  # active, deprecated, archived
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "base_model": self.base_model,
            "adapter_name": self.adapter_name,
            "training_type": self.training_type,
            "created_at": self.created_at,
            "training_job_id": self.training_job_id,
            "checkpoint_path": self.checkpoint_path,
            "final_loss": self.final_loss,
            "training_steps": self.training_steps,
            "training_time_seconds": self.training_time_seconds,
            "training_data_size": self.training_data_size,
            "training_data_source": self.training_data_source,
            "elo_rating": self.elo_rating,
            "win_rate": self.win_rate,
            "calibration_score": self.calibration_score,
            "primary_domain": self.primary_domain,
            "domain_scores": self.domain_scores,
            "status": self.status,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(
            model_id=data["model_id"],
            base_model=data["base_model"],
            adapter_name=data["adapter_name"],
            training_type=data["training_type"],
            created_at=data.get("created_at", datetime.now().isoformat()),
            training_job_id=data.get("training_job_id"),
            checkpoint_path=data.get("checkpoint_path"),
            final_loss=data.get("final_loss"),
            training_steps=data.get("training_steps", 0),
            training_time_seconds=data.get("training_time_seconds", 0),
            training_data_size=data.get("training_data_size", 0),
            training_data_source=data.get("training_data_source", ""),
            elo_rating=data.get("elo_rating"),
            win_rate=data.get("win_rate"),
            calibration_score=data.get("calibration_score"),
            primary_domain=data.get("primary_domain"),
            domain_scores=data.get("domain_scores", {}),
            status=data.get("status", "active"),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )


class ModelRegistry:
    """
    Registry for managing fine-tuned models and adapters.

    Provides:
    - Model registration and metadata tracking
    - Performance metric updates
    - Adapter hot-swapping
    - Model versioning and deprecation

    Example:
        registry = ModelRegistry()

        # Register a new model
        registry.register(ModelMetadata(
            model_id="aragora-security-v1",
            base_model="llama-3.3-70b",
            adapter_name="security-expert",
            training_type="sft",
        ))

        # Get best model for a domain
        model = registry.get_best_for_domain("security")

        # Update metrics after evaluation
        registry.update_metrics("aragora-security-v1", elo_rating=1250, win_rate=0.65)
    """

    def __init__(self, registry_path: Path | str = "model_registry.json"):
        self.registry_path = Path(registry_path)
        self._models: dict[str, ModelMetadata] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from file."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                for model_data in data.get("models", []):
                    model = ModelMetadata.from_dict(model_data)
                    self._models[model.model_id] = model
                logger.info("Loaded %d models from registry", len(self._models))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load model registry: %s", e)

    def _save(self) -> None:
        """Save registry to file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "models": [m.to_dict() for m in self._models.values()],
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def register(self, model: ModelMetadata) -> None:
        """
        Register a new model.

        Args:
            model: Model metadata to register
        """
        if model.model_id in self._models:
            logger.warning("Model %s already registered, updating", model.model_id)

        self._models[model.model_id] = model
        self._save()
        logger.info("Registered model: %s", model.model_id)

    def get(self, model_id: str) -> ModelMetadata | None:
        """Get model by ID."""
        return self._models.get(model_id)

    def list_models(
        self,
        status: str | None = None,
        training_type: str | None = None,
        base_model: str | None = None,
        tag: str | None = None,
        limit: int = 100,
    ) -> list[ModelMetadata]:
        """
        List models with optional filtering.

        Args:
            status: Filter by status (active, deprecated, archived)
            training_type: Filter by training type (sft, dpo, combined)
            base_model: Filter by base model
            tag: Filter by tag
            limit: Maximum number of results

        Returns:
            List of matching models
        """
        models = list(self._models.values())

        if status:
            models = [m for m in models if m.status == status]
        if training_type:
            models = [m for m in models if m.training_type == training_type]
        if base_model:
            models = [m for m in models if m.base_model == base_model]
        if tag:
            models = [m for m in models if tag in m.tags]

        # Sort by ELO rating (highest first), then by creation date
        models.sort(
            key=lambda m: (m.elo_rating or 0, m.created_at),
            reverse=True,
        )

        return models[:limit]

    def get_best_for_domain(self, domain: str) -> ModelMetadata | None:
        """
        Get the best performing model for a domain.

        Args:
            domain: Domain name (security, performance, etc.)

        Returns:
            Best model for the domain, or None if no models
        """
        active_models = [m for m in self._models.values() if m.status == "active"]

        if not active_models:
            return None

        # Score models by domain performance
        def domain_score(model: ModelMetadata) -> float:
            # Check domain-specific score
            if domain in model.domain_scores:
                return model.domain_scores[domain]
            # Check if primary domain matches
            if model.primary_domain == domain:
                return model.elo_rating or 1000
            # Fall back to general ELO
            return (model.elo_rating or 1000) * 0.8

        return max(active_models, key=domain_score)

    def get_latest(self, training_type: str | None = None) -> ModelMetadata | None:
        """Get the most recently created model."""
        models = self.list_models(status="active", training_type=training_type)
        return models[0] if models else None

    def update_metrics(
        self,
        model_id: str,
        elo_rating: float | None = None,
        win_rate: float | None = None,
        calibration_score: float | None = None,
        domain_scores: dict[str, float] | None = None,
    ) -> bool:
        """
        Update performance metrics for a model.

        Args:
            model_id: Model to update
            elo_rating: New ELO rating
            win_rate: New win rate
            calibration_score: New calibration score
            domain_scores: Domain-specific scores to update

        Returns:
            True if model was updated
        """
        model = self._models.get(model_id)
        if not model:
            return False

        if elo_rating is not None:
            model.elo_rating = elo_rating
        if win_rate is not None:
            model.win_rate = win_rate
        if calibration_score is not None:
            model.calibration_score = calibration_score
        if domain_scores:
            model.domain_scores.update(domain_scores)

        self._save()
        return True

    def deprecate(self, model_id: str, notes: str = "") -> bool:
        """
        Mark a model as deprecated.

        Args:
            model_id: Model to deprecate
            notes: Optional deprecation notes

        Returns:
            True if model was deprecated
        """
        model = self._models.get(model_id)
        if not model:
            return False

        model.status = "deprecated"
        if notes:
            model.notes = notes

        self._save()
        logger.info("Deprecated model: %s", model_id)
        return True

    def archive(self, model_id: str) -> bool:
        """
        Archive a model (remove from active use).

        Args:
            model_id: Model to archive

        Returns:
            True if model was archived
        """
        model = self._models.get(model_id)
        if not model:
            return False

        model.status = "archived"
        self._save()
        logger.info("Archived model: %s", model_id)
        return True

    def delete(self, model_id: str) -> bool:
        """
        Delete a model from the registry.

        Args:
            model_id: Model to delete

        Returns:
            True if model was deleted
        """
        if model_id not in self._models:
            return False

        del self._models[model_id]
        self._save()
        logger.info("Deleted model from registry: %s", model_id)
        return True

    def add_tag(self, model_id: str, tag: str) -> bool:
        """Add a tag to a model."""
        model = self._models.get(model_id)
        if not model:
            return False

        if tag not in model.tags:
            model.tags.append(tag)
            self._save()
        return True

    def remove_tag(self, model_id: str, tag: str) -> bool:
        """Remove a tag from a model."""
        model = self._models.get(model_id)
        if not model:
            return False

        if tag in model.tags:
            model.tags.remove(tag)
            self._save()
        return True

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics."""
        models = list(self._models.values())

        by_status: dict[str, int] = {}
        by_type: dict[str, int] = {}
        by_base: dict[str, int] = {}

        for m in models:
            by_status[m.status] = by_status.get(m.status, 0) + 1
            by_type[m.training_type] = by_type.get(m.training_type, 0) + 1
            by_base[m.base_model] = by_base.get(m.base_model, 0) + 1

        avg_elo = sum(m.elo_rating or 1000 for m in models) / len(models) if models else 0

        return {
            "total_models": len(models),
            "by_status": by_status,
            "by_training_type": by_type,
            "by_base_model": by_base,
            "average_elo": avg_elo,
        }


# Global registry instance
_default_registry: ModelRegistry | None = None


def get_registry(path: Path | str | None = None) -> ModelRegistry:
    """Get or create the default model registry."""
    global _default_registry
    if _default_registry is None or path is not None:
        registry_path = path or os.getenv(
            "TINKER_MODEL_REGISTRY",
            "model_registry.json",
        )
        _default_registry = ModelRegistry(registry_path)
    return _default_registry
