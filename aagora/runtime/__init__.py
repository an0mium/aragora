"""
Runtime module for aagora - Execution optimization and control.

Provides:
- Autotuner: Budget-aware round selection and early-stop
- Metadata: Run configuration and reproducibility info
- Metrics: Quality and cost tracking
"""

from aagora.runtime.autotune import Autotuner, AutotuneConfig, RunMetrics
from aagora.runtime.metadata import DebateMetadata, ModelConfig

__all__ = [
    "Autotuner",
    "AutotuneConfig",
    "RunMetrics",
    "DebateMetadata",
    "ModelConfig",
]
