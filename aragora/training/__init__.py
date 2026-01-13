"""
Tinker integration for fine-tuning open-source LLMs.

This module provides infrastructure for exporting Aragora debate data
and training models using the Tinker API (thinkingmachines.ai).

Training Paradigms:
- SFT (Supervised Fine-Tuning): Train on winning debate patterns
- DPO (Direct Preference Optimization): Train on win/loss preference pairs
- Adversarial: Train against Gauntlet vulnerability patterns

Example:
    from aragora.training import TinkerClient, SFTExporter

    # Export training data
    exporter = SFTExporter()
    data = exporter.export(min_confidence=0.8, limit=1000)

    # Train model
    client = TinkerClient()
    await client.train_sft(data, model="llama-3.3-70b")
"""

from aragora.training.debate_exporter import (
    DebateTrainingConfig,
    DebateTrainingExporter,
    export_debate_to_training,
)
from aragora.training.evaluator import ABTestResult, EvaluationMetrics, TinkerEvaluator
from aragora.training.exporters import (
    BaseExporter,
    DPOExporter,
    GauntletExporter,
    SFTExporter,
)
from aragora.training.model_registry import ModelMetadata, ModelRegistry, get_registry
from aragora.training.tinker_client import TinkerClient, TinkerConfig, TinkerModel
from aragora.training.training_scheduler import TrainingJob, TrainingScheduler

__all__ = [
    # Client
    "TinkerClient",
    "TinkerConfig",
    "TinkerModel",
    # Scheduler
    "TrainingScheduler",
    "TrainingJob",
    # Evaluator
    "TinkerEvaluator",
    "ABTestResult",
    "EvaluationMetrics",
    # Model Registry
    "ModelRegistry",
    "ModelMetadata",
    "get_registry",
    # Exporters
    "SFTExporter",
    "DPOExporter",
    "GauntletExporter",
    "BaseExporter",
    # Debate Exporter (Tinker integration)
    "DebateTrainingExporter",
    "DebateTrainingConfig",
    "export_debate_to_training",
]
