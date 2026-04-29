"""Public Proof-Carrying Code Unit namespace (DIC-19 / #6030).

The pure schema/model lives in :mod:`aragora.epistemic.proof_unit_model`.
Filesystem scanning and flag handling live in
:mod:`aragora.epistemic.proof_unit_scanner`.  The cross-reference index over
multiple units lives in :mod:`aragora.epistemic.constraint_graph`.  This
facade preserves the original public import path while keeping dataclasses
separate from I/O and graph logic.
"""

from __future__ import annotations

from .constraint_graph import ProofUnitConstraintGraph
from .proof_unit_model import (
    DecayPolicy,
    FallbackPolicy,
    ProofCarryingCodeUnit,
    load_proof_unit,
)
from .proof_unit_scanner import (
    InvalidProofUnitError,
    MalformedProofUnitError,
    ProofUnitLoadError,
    enable_proof_unit_scan,
    load_proof_unit_from_yaml,
    load_proof_units_from_dir,
    proof_unit_scan_enabled,
    reset_proof_unit_scan,
)

__all__ = [
    "DecayPolicy",
    "FallbackPolicy",
    "InvalidProofUnitError",
    "MalformedProofUnitError",
    "ProofCarryingCodeUnit",
    "ProofUnitConstraintGraph",
    "ProofUnitLoadError",
    "enable_proof_unit_scan",
    "load_proof_unit",
    "load_proof_unit_from_yaml",
    "load_proof_units_from_dir",
    "proof_unit_scan_enabled",
    "reset_proof_unit_scan",
]
