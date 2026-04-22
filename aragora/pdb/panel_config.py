"""PDB-specific wrapper around :mod:`aragora.brief_engine.panel_config`.

The generic panel config schema moved to
:mod:`aragora.brief_engine.panel_config` in the Phase 1 brief-engine
extraction. This module re-exports the generic dataclasses under their
legacy ``PDB*`` names and keeps the Mode 3 default-config-path wiring
here (the yaml itself — ``aragora/config/pdb_panel.yaml`` — remains
PDB-specific).

Do not add new schema types here — put them in
:mod:`aragora.brief_engine.panel_config` instead.
"""

from __future__ import annotations

from pathlib import Path

from aragora.brief_engine.panel_config import (
    BriefBudgetConfig,
    BriefPanelConfig,
    BriefPanelConfigError,
    BriefPanelDefinition,
    BriefPanelSlot,
    BriefPromptSet,
    panel_slots,
    provider_slot_definitions,
    validate_panel_config,
)
from aragora.brief_engine.panel_config import (
    load_panel_config as _load_panel_config_generic,
)

# Legacy PDB-named aliases. New code should import the ``Brief*`` names
# from :mod:`aragora.brief_engine.panel_config` directly.
PDBBudgetConfig = BriefBudgetConfig
PDBPanelConfig = BriefPanelConfig
PDBPanelConfigError = BriefPanelConfigError
PDBPanelDefinition = BriefPanelDefinition
PDBPanelSlot = BriefPanelSlot
PDBPromptSet = BriefPromptSet

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "PDBBudgetConfig",
    "PDBPanelConfig",
    "PDBPanelConfigError",
    "PDBPanelDefinition",
    "PDBPanelSlot",
    "PDBPromptSet",
    "load_panel_config",
    "panel_slots",
    "provider_slot_definitions",
    "validate_panel_config",
]


DEFAULT_CONFIG_PATH: Path = Path(__file__).resolve().parent.parent / "config" / "pdb_panel.yaml"
"""Filesystem path to the committed Mode 3 PDB panel config."""


def load_panel_config(path: Path | None = None) -> PDBPanelConfig:
    """Load and validate the Mode 3 PDB panel config.

    ``path`` defaults to :data:`DEFAULT_CONFIG_PATH`. This is a thin
    wrapper around :func:`aragora.brief_engine.panel_config.load_panel_config`
    that supplies the PDB-specific default.
    """
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    return _load_panel_config_generic(target)
