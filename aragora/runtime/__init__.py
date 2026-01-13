"""
Runtime module for aragora - Execution optimization and control.

Provides:
- Autotuner: Budget-aware round selection and early-stop
- Metadata: Run configuration and reproducibility info
- Metrics: Quality and cost tracking
"""

from aragora.runtime.autotune import AutotuneConfig, Autotuner, RunMetrics
from aragora.runtime.metadata import DebateMetadata, ModelConfig

__all__ = [
    "Autotuner",
    "AutotuneConfig",
    "RunMetrics",
    "DebateMetadata",
    "ModelConfig",
]
