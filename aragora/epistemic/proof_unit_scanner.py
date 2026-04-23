"""Flag-gated filesystem scanner for Proof-Carrying Code Units."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from .proof_unit_model import ProofCarryingCodeUnit, load_proof_unit

logger = logging.getLogger(__name__)

_SCAN_FLAG = "ARAGORA_PROOF_UNIT_SCAN_ENABLED"


class ProofUnitLoadError(ValueError):
    """Base error for proof-unit YAML loading failures."""


class InvalidProofUnitError(ProofUnitLoadError):
    """A proof-unit manifest parsed but failed schema validation."""


class MalformedProofUnitError(ProofUnitLoadError):
    """A proof-unit manifest could not be parsed into the schema."""


# Module-level override avoids os.environ mutation (per the pattern in #6454).
# External callers may still set the env var; the override takes priority.
_scan_enabled_override: bool | None = None


def proof_unit_scan_enabled() -> bool:
    """Return True when the directory scanner should load proof-unit manifests.

    Checks the module-level override first, then
    ``ARAGORA_PROOF_UNIT_SCAN_ENABLED`` in the process environment.
    Default is *False*; dataclass construction is always safe regardless.
    """
    if _scan_enabled_override is not None:
        return _scan_enabled_override
    raw = str(os.environ.get(_SCAN_FLAG) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def enable_proof_unit_scan() -> None:
    """Enable the proof-unit directory scanner for the current process.

    Sets a module-level override rather than mutating ``os.environ``.
    Call :func:`reset_proof_unit_scan` to restore the default env-var-driven
    behaviour (useful in test teardown).
    """
    global _scan_enabled_override
    _scan_enabled_override = True


def reset_proof_unit_scan() -> None:
    """Clear the module-level override, reverting to env-var-driven behaviour."""
    global _scan_enabled_override
    _scan_enabled_override = None


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        with path.open() as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise MalformedProofUnitError(f"Malformed proof-unit YAML at {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise MalformedProofUnitError(f"Malformed proof unit at {path}: expected mapping")

    return data


def load_proof_unit_from_yaml(path: Path) -> ProofCarryingCodeUnit:
    """Load and validate a :class:`ProofCarryingCodeUnit` from YAML.

    Raises :class:`InvalidProofUnitError` if the manifest parses but fails
    validation, and :class:`MalformedProofUnitError` if it cannot be converted
    into the schema. Both subclasses inherit :class:`ValueError` for backwards
    compatibility with existing callers.
    """
    data = _read_yaml_mapping(path)
    try:
        unit = load_proof_unit(data)
    except (TypeError, ValueError) as exc:
        raise MalformedProofUnitError(f"Malformed proof unit at {path}: {exc}") from exc

    errors = unit.validate()
    if errors:
        raise InvalidProofUnitError(f"Invalid proof unit at {path}: {errors}")
    return unit


def load_proof_units_from_dir(base: Path) -> list[ProofCarryingCodeUnit]:
    """Load all valid ``*.yaml`` proof-unit manifests under *base*.

    Returns an empty list when :func:`proof_unit_scan_enabled` is *False*
    (default), so callers never need to guard this call themselves.
    Expected validation errors are logged concisely and skipped. Unexpected
    schema/parsing errors are logged with full traceback so operators can
    diagnose malformed files.
    """
    if not proof_unit_scan_enabled():
        return []
    units: list[ProofCarryingCodeUnit] = []
    for path in sorted(base.glob("*.yaml")):
        try:
            units.append(load_proof_unit_from_yaml(path))
        except InvalidProofUnitError as exc:
            logger.warning("skipping invalid proof unit %s: %s", path, exc)
        except MalformedProofUnitError:
            logger.warning("skipping malformed proof unit %s", path, exc_info=True)
    return units
