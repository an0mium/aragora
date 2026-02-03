"""
Sampling strategies for inference-time reasoning improvement.

This module implements advanced sampling strategies including:
- Power Sampling: Power-law weighted sampling for diverse reasoning paths
- Temperature annealing: Adaptive temperature control
- Best-of-N with diversity: Sample selection with diversity constraints
"""

from aragora.reasoning.sampling.power_sampling import (
    PowerSampler,
    PowerSamplingConfig,
    SamplingResult,
    sample_with_power_law,
)

__all__ = [
    "PowerSampler",
    "PowerSamplingConfig",
    "SamplingResult",
    "sample_with_power_law",
]
