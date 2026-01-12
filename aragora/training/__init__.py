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

from aragora.training.tinker_client import TinkerClient, TinkerConfig
from aragora.training.training_scheduler import TrainingScheduler, TrainingJob
from aragora.training.exporters import (
    SFTExporter,
    DPOExporter,
    GauntletExporter,
    BaseExporter,
)

__all__ = [
    "TinkerClient",
    "TinkerConfig",
    "TrainingScheduler",
    "TrainingJob",
    "SFTExporter",
    "DPOExporter",
    "GauntletExporter",
    "BaseExporter",
]
