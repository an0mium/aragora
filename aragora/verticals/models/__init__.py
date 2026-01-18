"""
Vertical Specialist Model Loaders.

Provides loaders for domain-specific HuggingFace models with
support for quantization and LoRA adapters.
"""

from aragora.verticals.models.huggingface_loader import (
    HuggingFaceSpecialistLoader,
    SpecialistModel,
    ModelLoadError,
    RECOMMENDED_MODELS,
)

from aragora.verticals.models.finetuning import (
    VerticalFineTuningPipeline,
    FinetuningConfig,
    TrainingExample,
)

__all__ = [
    # Model loading
    "HuggingFaceSpecialistLoader",
    "SpecialistModel",
    "ModelLoadError",
    "RECOMMENDED_MODELS",
    # Fine-tuning
    "VerticalFineTuningPipeline",
    "FinetuningConfig",
    "TrainingExample",
]
