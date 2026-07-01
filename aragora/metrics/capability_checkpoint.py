"""Deprecated re-export shim for capability checkpoints.

The originals now live in :mod:`aragora.evaluation.capability_checkpoint`.
This module re-exports them so the legacy
``aragora.metrics.capability_checkpoint`` import path keeps working for one
release.
"""

import warnings

from aragora.evaluation.capability_checkpoint import (
    CapabilityCheckpoint,
    CheckpointCode,
    CheckpointRecord,
    CheckpointRegistry,
    CheckpointRegistryError,
    CheckpointStatus,
    build_default_registry,
    capability_checkpoints_enabled,
)

warnings.warn(
    "aragora.metrics.capability_checkpoint is deprecated; import from "
    "aragora.evaluation.capability_checkpoint instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "CapabilityCheckpoint",
    "CheckpointCode",
    "CheckpointRecord",
    "CheckpointRegistry",
    "CheckpointRegistryError",
    "CheckpointStatus",
    "build_default_registry",
    "capability_checkpoints_enabled",
]
