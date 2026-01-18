"""
Specialist Models for Enterprise Verticals.

Extends the model registry with domain-specific fine-tuned models
for enterprise use cases: legal, healthcare, security, accounting, academic.

The specialist model pipeline:
1. Export domain-specific training data from debates
2. Train models using Tinker API or local PEFT
3. Register in specialist model registry
4. Create agents using specialist models via SpecialistAgentFactory
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from aragora.training.model_registry import ModelMetadata, ModelRegistry

if TYPE_CHECKING:
    from aragora.training.tinker_client import TinkerClient

logger = logging.getLogger(__name__)


class Vertical(str, Enum):
    """Enterprise verticals for specialist models."""

    LEGAL = "legal"
    HEALTHCARE = "healthcare"
    SECURITY = "security"
    ACCOUNTING = "accounting"
    REGULATORY = "regulatory"
    ACADEMIC = "academic"
    SOFTWARE = "software"
    GENERAL = "general"


class TrainingStatus(str, Enum):
    """Status of a specialist model training job."""

    PENDING = "pending"
    EXPORTING_DATA = "exporting_data"
    TRAINING = "training"
    EVALUATING = "evaluating"
    READY = "ready"
    FAILED = "failed"
    DEPRECATED = "deprecated"


@dataclass
class SpecialistModelConfig:
    """Configuration for training a specialist model."""

    vertical: Vertical
    base_model: str = "llama-3.3-70b"
    adapter_name: Optional[str] = None  # Auto-generated if not provided
    org_id: Optional[str] = None  # None for global models
    workspace_ids: List[str] = field(default_factory=list)  # Training data sources

    # Training parameters
    lora_rank: int = 16
    learning_rate: float = 1e-4
    max_steps: int = 1000
    batch_size: int = 4

    # Data filtering
    min_confidence: float = 0.7
    min_debates: int = 10
    include_critiques: bool = True
    include_consensus: bool = True

    # Evaluation
    eval_split: float = 0.1
    run_gauntlet: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "vertical": self.vertical.value,
            "base_model": self.base_model,
            "adapter_name": self.adapter_name,
            "org_id": self.org_id,
            "workspace_ids": self.workspace_ids,
            "lora_rank": self.lora_rank,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "batch_size": self.batch_size,
            "min_confidence": self.min_confidence,
            "min_debates": self.min_debates,
            "include_critiques": self.include_critiques,
            "include_consensus": self.include_consensus,
            "eval_split": self.eval_split,
            "run_gauntlet": self.run_gauntlet,
        }


@dataclass
class SpecialistModel:
    """
    A specialist model fine-tuned for a specific enterprise vertical.

    Extends ModelMetadata with vertical-specific information and
    organization scoping.
    """

    id: str
    base_model: str
    adapter_name: str
    vertical: Vertical
    org_id: Optional[str]  # None for global models
    status: TrainingStatus = TrainingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""

    # Training info
    training_job_id: Optional[str] = None
    training_config: Optional[SpecialistModelConfig] = None
    training_data_debates: int = 0
    training_data_examples: int = 0

    # Performance metrics
    final_loss: Optional[float] = None
    elo_rating: Optional[float] = None
    win_rate: Optional[float] = None
    calibration_score: Optional[float] = None

    # Vertical-specific metrics
    vertical_accuracy: Optional[float] = None
    domain_coverage: Dict[str, float] = field(default_factory=dict)

    # Model artifacts
    checkpoint_path: Optional[str] = None
    hf_model_id: Optional[str] = None

    # Metadata
    version: int = 1
    supersedes: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "base_model": self.base_model,
            "adapter_name": self.adapter_name,
            "vertical": self.vertical.value,
            "org_id": self.org_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "training_job_id": self.training_job_id,
            "training_config": self.training_config.to_dict() if self.training_config else None,
            "training_data_debates": self.training_data_debates,
            "training_data_examples": self.training_data_examples,
            "final_loss": self.final_loss,
            "elo_rating": self.elo_rating,
            "win_rate": self.win_rate,
            "calibration_score": self.calibration_score,
            "vertical_accuracy": self.vertical_accuracy,
            "domain_coverage": self.domain_coverage,
            "checkpoint_path": self.checkpoint_path,
            "hf_model_id": self.hf_model_id,
            "version": self.version,
            "supersedes": self.supersedes,
            "tags": self.tags,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpecialistModel":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            base_model=data["base_model"],
            adapter_name=data["adapter_name"],
            vertical=Vertical(data["vertical"]),
            org_id=data.get("org_id"),
            status=TrainingStatus(data.get("status", "pending")),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else datetime.now(),
            created_by=data.get("created_by", ""),
            training_job_id=data.get("training_job_id"),
            training_data_debates=data.get("training_data_debates", 0),
            training_data_examples=data.get("training_data_examples", 0),
            final_loss=data.get("final_loss"),
            elo_rating=data.get("elo_rating"),
            win_rate=data.get("win_rate"),
            calibration_score=data.get("calibration_score"),
            vertical_accuracy=data.get("vertical_accuracy"),
            domain_coverage=data.get("domain_coverage", {}),
            checkpoint_path=data.get("checkpoint_path"),
            hf_model_id=data.get("hf_model_id"),
            version=data.get("version", 1),
            supersedes=data.get("supersedes"),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )

    def to_model_metadata(self) -> ModelMetadata:
        """Convert to base ModelMetadata for registry compatibility."""
        return ModelMetadata(
            model_id=self.id,
            base_model=self.base_model,
            adapter_name=self.adapter_name,
            training_type="specialist",
            created_at=self.created_at.isoformat(),
            training_job_id=self.training_job_id,
            checkpoint_path=self.checkpoint_path,
            final_loss=self.final_loss,
            training_data_size=self.training_data_examples,
            training_data_source=f"vertical:{self.vertical.value}",
            elo_rating=self.elo_rating,
            win_rate=self.win_rate,
            calibration_score=self.calibration_score,
            primary_domain=self.vertical.value,
            domain_scores=self.domain_coverage,
            status="active" if self.status == TrainingStatus.READY else "pending",
            tags=self.tags + [f"vertical:{self.vertical.value}"],
            notes=self.notes,
        )


class SpecialistModelRegistry:
    """
    Registry for managing specialist models by vertical and organization.

    Extends ModelRegistry with vertical-specific queries and
    organization scoping.
    """

    def __init__(self, base_registry: Optional[ModelRegistry] = None):
        """
        Initialize specialist model registry.

        Args:
            base_registry: Base model registry to sync with
        """
        self._base_registry = base_registry
        self._specialists: Dict[str, SpecialistModel] = {}

    def register(self, model: SpecialistModel) -> None:
        """
        Register a specialist model.

        Args:
            model: Specialist model to register
        """
        self._specialists[model.id] = model

        # Also register in base registry
        if self._base_registry:
            self._base_registry.register(model.to_model_metadata())

        logger.info(f"Registered specialist model: {model.id} ({model.vertical.value})")

    def get(self, model_id: str) -> Optional[SpecialistModel]:
        """Get a specialist model by ID."""
        return self._specialists.get(model_id)

    def get_for_vertical(
        self,
        vertical: Vertical,
        org_id: Optional[str] = None,
        include_global: bool = True,
    ) -> Optional[SpecialistModel]:
        """
        Get the best specialist model for a vertical.

        Args:
            vertical: Vertical to get model for
            org_id: Organization ID (for org-specific models)
            include_global: Include global models in search

        Returns:
            Best available specialist model for the vertical
        """
        candidates = []

        for model in self._specialists.values():
            if model.vertical != vertical:
                continue
            if model.status != TrainingStatus.READY:
                continue

            # Check org scope
            if model.org_id:
                if model.org_id == org_id:
                    candidates.append((model, 2))  # Org-specific, higher priority
            elif include_global:
                candidates.append((model, 1))  # Global model

        if not candidates:
            return None

        # Sort by priority, then by ELO
        candidates.sort(
            key=lambda x: (x[1], x[0].elo_rating or 0),
            reverse=True,
        )

        return candidates[0][0]

    def list_for_vertical(
        self,
        vertical: Vertical,
        org_id: Optional[str] = None,
        status: Optional[TrainingStatus] = None,
    ) -> List[SpecialistModel]:
        """
        List all specialist models for a vertical.

        Args:
            vertical: Vertical to list models for
            org_id: Filter by organization (None for global only)
            status: Filter by training status

        Returns:
            List of matching specialist models
        """
        models = []

        for model in self._specialists.values():
            if model.vertical != vertical:
                continue
            if org_id and model.org_id != org_id:
                continue
            if status and model.status != status:
                continue

            models.append(model)

        # Sort by version (newest first), then by ELO
        models.sort(
            key=lambda m: (m.version, m.elo_rating or 0),
            reverse=True,
        )

        return models

    def list_for_org(
        self,
        org_id: str,
        vertical: Optional[Vertical] = None,
        status: Optional[TrainingStatus] = None,
    ) -> List[SpecialistModel]:
        """
        List all specialist models for an organization.

        Args:
            org_id: Organization ID
            vertical: Filter by vertical
            status: Filter by training status

        Returns:
            List of matching specialist models
        """
        models = []

        for model in self._specialists.values():
            if model.org_id != org_id:
                continue
            if vertical and model.vertical != vertical:
                continue
            if status and model.status != status:
                continue

            models.append(model)

        return models

    def update_status(
        self,
        model_id: str,
        status: TrainingStatus,
        **metrics: Any,
    ) -> bool:
        """
        Update specialist model status and metrics.

        Args:
            model_id: Model to update
            status: New status
            **metrics: Additional metrics to update

        Returns:
            True if model was updated
        """
        model = self._specialists.get(model_id)
        if not model:
            return False

        model.status = status
        model.updated_at = datetime.now()

        # Update metrics
        for key, value in metrics.items():
            if hasattr(model, key):
                setattr(model, key, value)

        # Sync to base registry
        if self._base_registry and status == TrainingStatus.READY:
            self._base_registry.register(model.to_model_metadata())

        logger.info(f"Updated specialist model {model_id} status to {status.value}")

        return True

    def deprecate(self, model_id: str, notes: str = "") -> bool:
        """Deprecate a specialist model."""
        return self.update_status(
            model_id,
            TrainingStatus.DEPRECATED,
            notes=notes,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        by_vertical: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        by_org: Dict[str, int] = {}

        for model in self._specialists.values():
            by_vertical[model.vertical.value] = by_vertical.get(model.vertical.value, 0) + 1
            by_status[model.status.value] = by_status.get(model.status.value, 0) + 1
            org = model.org_id or "global"
            by_org[org] = by_org.get(org, 0) + 1

        ready_models = [m for m in self._specialists.values() if m.status == TrainingStatus.READY]
        avg_elo = sum(m.elo_rating or 1000 for m in ready_models) / len(ready_models) if ready_models else 0

        return {
            "total_models": len(self._specialists),
            "by_vertical": by_vertical,
            "by_status": by_status,
            "by_org": by_org,
            "ready_count": len(ready_models),
            "average_elo": avg_elo,
        }


class SpecialistTrainingPipeline:
    """
    Pipeline for training specialist models.

    Orchestrates:
    1. Export training data from debates
    2. Train model using Tinker API
    3. Evaluate on gauntlet
    4. Register in specialist registry
    """

    def __init__(
        self,
        registry: SpecialistModelRegistry,
        tinker_client: Optional["TinkerClient"] = None,
    ):
        """
        Initialize training pipeline.

        Args:
            registry: Specialist model registry
            tinker_client: Tinker API client for training
        """
        self._registry = registry
        self._tinker = tinker_client

    async def create_training_job(
        self,
        config: SpecialistModelConfig,
        created_by: str,
    ) -> SpecialistModel:
        """
        Create a new specialist model training job.

        Args:
            config: Training configuration
            created_by: User creating the job

        Returns:
            Created specialist model (pending status)
        """
        model_id = f"sm_{config.vertical.value}_{uuid.uuid4().hex[:8]}"
        adapter_name = config.adapter_name or f"aragora-{config.vertical.value}-v1"

        model = SpecialistModel(
            id=model_id,
            base_model=config.base_model,
            adapter_name=adapter_name,
            vertical=config.vertical,
            org_id=config.org_id,
            status=TrainingStatus.PENDING,
            created_by=created_by,
            training_config=config,
        )

        self._registry.register(model)

        logger.info(f"Created training job for specialist model: {model_id}")

        return model

    async def export_training_data(
        self,
        model_id: str,
    ) -> int:
        """
        Export training data for a specialist model.

        Args:
            model_id: Model to export data for

        Returns:
            Number of training examples exported
        """
        model = self._registry.get(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        self._registry.update_status(model_id, TrainingStatus.EXPORTING_DATA)

        # TODO: Implement actual data export from debates
        # This would use the vertical exporter to:
        # 1. Query debates by vertical keywords
        # 2. Filter by confidence and workspace
        # 3. Format as SFT/DPO training examples

        example_count = 0  # Placeholder

        self._registry.update_status(
            model_id,
            TrainingStatus.PENDING,
            training_data_examples=example_count,
        )

        return example_count

    async def start_training(
        self,
        model_id: str,
    ) -> str:
        """
        Start training a specialist model.

        Args:
            model_id: Model to train

        Returns:
            Training job ID
        """
        model = self._registry.get(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        if not self._tinker:
            raise ValueError("Tinker client not configured")

        self._registry.update_status(model_id, TrainingStatus.TRAINING)

        # TODO: Start actual training job with Tinker API
        training_job_id = f"tj_{uuid.uuid4().hex[:12]}"

        self._registry.update_status(
            model_id,
            TrainingStatus.TRAINING,
            training_job_id=training_job_id,
        )

        logger.info(f"Started training job {training_job_id} for model {model_id}")

        return training_job_id

    async def complete_training(
        self,
        model_id: str,
        final_loss: float,
        checkpoint_path: str,
    ) -> None:
        """
        Mark training as complete and start evaluation.

        Args:
            model_id: Model that completed training
            final_loss: Final training loss
            checkpoint_path: Path to saved checkpoint
        """
        self._registry.update_status(
            model_id,
            TrainingStatus.EVALUATING,
            final_loss=final_loss,
            checkpoint_path=checkpoint_path,
        )

        # TODO: Run gauntlet evaluation
        # This would:
        # 1. Load the trained model
        # 2. Run evaluation debates
        # 3. Compute ELO and vertical accuracy

        # For now, mark as ready
        self._registry.update_status(model_id, TrainingStatus.READY)

        logger.info(f"Specialist model {model_id} is ready")

    async def get_training_status(
        self,
        model_id: str,
    ) -> Dict[str, Any]:
        """
        Get training status for a model.

        Args:
            model_id: Model to check

        Returns:
            Training status and metrics
        """
        model = self._registry.get(model_id)
        if not model:
            raise ValueError(f"Model not found: {model_id}")

        return {
            "model_id": model.id,
            "vertical": model.vertical.value,
            "status": model.status.value,
            "training_job_id": model.training_job_id,
            "training_data_examples": model.training_data_examples,
            "final_loss": model.final_loss,
            "elo_rating": model.elo_rating,
            "checkpoint_path": model.checkpoint_path,
        }


# Vertical-specific training configurations
VERTICAL_DEFAULTS: Dict[Vertical, Dict[str, Any]] = {
    Vertical.LEGAL: {
        "base_model": "llama-3.3-70b",
        "lora_rank": 32,
        "max_steps": 2000,
        "keywords": ["contract", "legal", "compliance", "regulation", "clause"],
    },
    Vertical.HEALTHCARE: {
        "base_model": "llama-3.3-70b",
        "lora_rank": 32,
        "max_steps": 2000,
        "keywords": ["medical", "clinical", "patient", "diagnosis", "treatment", "hipaa"],
    },
    Vertical.SECURITY: {
        "base_model": "qwen-2.5-72b",
        "lora_rank": 16,
        "max_steps": 1500,
        "keywords": ["security", "vulnerability", "attack", "auth", "encryption"],
    },
    Vertical.ACCOUNTING: {
        "base_model": "llama-3.3-70b",
        "lora_rank": 16,
        "max_steps": 1500,
        "keywords": ["financial", "accounting", "audit", "tax", "revenue"],
    },
    Vertical.REGULATORY: {
        "base_model": "llama-3.3-70b",
        "lora_rank": 32,
        "max_steps": 2000,
        "keywords": ["compliance", "regulation", "policy", "sox", "gdpr"],
    },
    Vertical.ACADEMIC: {
        "base_model": "qwen-2.5-72b",
        "lora_rank": 16,
        "max_steps": 1500,
        "keywords": ["research", "citation", "academic", "paper", "methodology"],
    },
    Vertical.SOFTWARE: {
        "base_model": "qwen-2.5-72b",
        "lora_rank": 16,
        "max_steps": 1500,
        "keywords": ["code", "architecture", "api", "bug", "refactor"],
    },
}


def get_vertical_config(
    vertical: Vertical,
    org_id: Optional[str] = None,
    **overrides: Any,
) -> SpecialistModelConfig:
    """
    Get default configuration for a vertical.

    Args:
        vertical: Vertical to configure
        org_id: Optional organization ID
        **overrides: Configuration overrides

    Returns:
        SpecialistModelConfig with vertical defaults
    """
    defaults = VERTICAL_DEFAULTS.get(vertical, {})

    return SpecialistModelConfig(
        vertical=vertical,
        org_id=org_id,
        base_model=overrides.get("base_model", defaults.get("base_model", "llama-3.3-70b")),
        lora_rank=overrides.get("lora_rank", defaults.get("lora_rank", 16)),
        max_steps=overrides.get("max_steps", defaults.get("max_steps", 1000)),
        **{k: v for k, v in overrides.items() if k not in ["base_model", "lora_rank", "max_steps"]},
    )


# Global registry instance
_specialist_registry: Optional[SpecialistModelRegistry] = None


def get_specialist_registry(
    base_registry: Optional[ModelRegistry] = None,
) -> SpecialistModelRegistry:
    """Get or create the global specialist model registry."""
    global _specialist_registry
    if _specialist_registry is None:
        _specialist_registry = SpecialistModelRegistry(base_registry)
    return _specialist_registry


__all__ = [
    "Vertical",
    "TrainingStatus",
    "SpecialistModelConfig",
    "SpecialistModel",
    "SpecialistModelRegistry",
    "SpecialistTrainingPipeline",
    "VERTICAL_DEFAULTS",
    "get_vertical_config",
    "get_specialist_registry",
]
