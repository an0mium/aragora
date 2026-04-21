"""Panel configuration loader and validator for PDB Mode 3 Protocol B.

This module owns the YAML-driven panel roster described in
``docs/plans/2026-04-21-pdb-mode3-pr2-spec.md``. It is a pure
schema + validation layer: no HTTP, no worker queues, no artifact
writing. The executor in :mod:`aragora.pdb.protocol` is the only
in-process consumer.

Public surface
--------------

Dataclasses:

- :class:`PDBBudgetConfig` — per-brief / per-day USD caps
- :class:`PDBPanelSlot` — one slot's metadata (review role, lens,
  family, candidate provider ids, ``required`` flag)
- :class:`PDBPanelDefinition` — which slots participate in findings,
  critique, and which one acts as synthesizer
- :class:`PDBPromptSet` — which prompt templates this panel uses
- :class:`PDBPanelConfig` — the validated top-level config

Functions:

- :func:`load_panel_config` — load and validate the committed default
  or a caller-supplied path
- :func:`validate_panel_config` — validate a raw ``Mapping`` (typically
  the result of :func:`yaml.safe_load`) with exact field-path errors
- :func:`provider_slot_definitions` — project panel slots into the
  landed :class:`aragora.review.provider_slots.ProviderSlotDefinition`
  sequence for the specified panel
- :func:`panel_slots` — return the ordered :class:`PDBPanelSlot` objects
  for a specific panel (preserves the panel-level ordering required by
  the executor)

Validation posture
------------------

The loader raises :class:`PDBPanelConfigError` with an exact
``field_path`` attribute so callers can surface the failing key path
to operators without string-matching. Every rule in the spec's
"Validation rules" section is enforced:

- ``version == 1``
- ``default_panel`` exists in ``panels``
- ``synthesizer_slot`` exists in ``slots``
- every slot defines ``review_role``, ``lens``, ``family``,
  ``candidates`` (and optionally ``required``)
- ``findings_slots`` includes BOTH required core slots
- the selected panel contains at least one non-core lens
- ``same_as_findings`` is allowed only for ``critique_slots``

The module is intentionally I/O-free aside from one explicit filesystem
read in :func:`load_panel_config`. Mocked configs in tests call
:func:`validate_panel_config` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from aragora.review.provider_slots import ProviderSlotDefinition

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
"""Filesystem path to the committed default config."""

SUPPORTED_VERSION = 1
SAME_AS_FINDINGS_SENTINEL = "same_as_findings"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PDBPanelConfigError(ValueError):
    """Raised when a panel config fails validation.

    Inherits from :class:`ValueError` so callers that want to swallow
    "bad input" errors without importing this package still catch it.
    The ``field_path`` attribute is the dotted path inside the YAML
    document so operators can locate the failing key without string
    matching on the message.
    """

    def __init__(self, field_path: str, message: str) -> None:
        self.field_path = field_path
        super().__init__(f"{field_path}: {message}")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PDBBudgetConfig:
    """Per-brief and per-day USD caps plus a manual-escalation reserve.

    ``reserve_for_manual_escalation_usd`` is retained out of the
    rolling daily pool so escalated attention flows (e.g., a deep human
    review triggered by the brief) still have spend available even
    after automated runs have saturated ``per_day_usd``. The executor
    in :mod:`aragora.pdb.protocol` does not dip into the reserve; it
    is surfaced to the caller for downstream policy layers.
    """

    per_brief_usd: float
    per_day_usd: float
    reserve_for_manual_escalation_usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_brief_usd": self.per_brief_usd,
            "per_day_usd": self.per_day_usd,
            "reserve_for_manual_escalation_usd": self.reserve_for_manual_escalation_usd,
        }


@dataclass(frozen=True, slots=True)
class PDBPanelSlot:
    """One configured slot in the panel roster.

    ``required=True`` means Protocol B MUST fail closed if this slot
    cannot be resolved or cannot be funded. ``required=False`` means
    the executor may degrade by dropping the slot while preserving the
    minimum safe roster (both core slots plus synthesizer).
    """

    slot_id: str
    review_role: str
    lens: str
    family: str
    candidates: tuple[str, ...]
    required: bool = False

    def to_provider_slot_definition(self) -> ProviderSlotDefinition:
        """Project this slot into the landed resolver input shape."""
        return ProviderSlotDefinition(
            slot_id=self.slot_id,
            review_role=self.review_role,
            lens=self.lens,
            family=self.family,
            candidates=self.candidates,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "review_role": self.review_role,
            "lens": self.lens,
            "family": self.family,
            "candidates": list(self.candidates),
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class PDBPanelDefinition:
    """Which slots participate in each phase of Protocol B.

    ``critique_slots == findings_slots`` is the default (the YAML
    sentinel ``same_as_findings`` expands to a copy of
    ``findings_slots`` during validation). ``synthesizer_slot`` MUST
    also appear in ``findings_slots`` so the synthesizer has first-hand
    findings context.
    """

    panel_id: str
    findings_slots: tuple[str, ...]
    critique_slots: tuple[str, ...]
    synthesizer_slot: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "findings_slots": list(self.findings_slots),
            "critique_slots": list(self.critique_slots),
            "synthesizer_slot": self.synthesizer_slot,
        }


@dataclass(frozen=True, slots=True)
class PDBPromptSet:
    """Named prompt-template identifiers per phase.

    Resolved to actual prompt text at invocation time via
    :mod:`aragora.pdb.prompts`. This dataclass only carries the template
    identifiers so the panel config remains behavior-free.
    """

    prompt_set_id: str
    findings_prompt: str
    critique_prompt: str
    synthesis_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_set_id": self.prompt_set_id,
            "findings_prompt": self.findings_prompt,
            "critique_prompt": self.critique_prompt,
            "synthesis_prompt": self.synthesis_prompt,
        }


@dataclass(frozen=True, slots=True)
class PDBPanelConfig:
    """Top-level validated PDB panel configuration."""

    version: int
    default_panel: str
    default_prompt_set: str
    budget: PDBBudgetConfig
    slots: Mapping[str, PDBPanelSlot]
    panels: Mapping[str, PDBPanelDefinition]
    prompt_sets: Mapping[str, PDBPromptSet]
    source_path: Path | None = field(default=None, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "default_panel": self.default_panel,
            "default_prompt_set": self.default_prompt_set,
            "budget": self.budget.to_dict(),
            "slots": {sid: slot.to_dict() for sid, slot in self.slots.items()},
            "panels": {pid: panel.to_dict() for pid, panel in self.panels.items()},
            "prompt_sets": {pid: ps.to_dict() for pid, ps in self.prompt_sets.items()},
        }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_panel_config(path: Path | None = None) -> PDBPanelConfig:
    """Load and validate a panel config from disk.

    ``path`` defaults to :data:`DEFAULT_CONFIG_PATH`. YAML is parsed via
    :func:`yaml.safe_load` (so no code execution, no custom tags).
    """
    target = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        raw_text = target.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PDBPanelConfigError(
            "<path>",
            f"panel config file not found at {target}",
        ) from exc

    raw = yaml.safe_load(raw_text)
    if raw is None:
        raise PDBPanelConfigError("<root>", "panel config file is empty")
    if not isinstance(raw, Mapping):
        raise PDBPanelConfigError(
            "<root>",
            f"panel config top level must be a mapping; got {type(raw).__name__}",
        )
    config = validate_panel_config(raw)
    return _with_source_path(config, target)


def _with_source_path(config: PDBPanelConfig, path: Path) -> PDBPanelConfig:
    """Return a copy of ``config`` with ``source_path`` set.

    The dataclass is frozen, so we materialize a new instance rather
    than mutating.
    """
    return PDBPanelConfig(
        version=config.version,
        default_panel=config.default_panel,
        default_prompt_set=config.default_prompt_set,
        budget=config.budget,
        slots=config.slots,
        panels=config.panels,
        prompt_sets=config.prompt_sets,
        source_path=path,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_panel_config(raw: Mapping[str, Any]) -> PDBPanelConfig:
    """Validate a parsed panel config ``Mapping`` into a typed dataclass.

    Every spec rule is enforced with an exact ``field_path``. No data
    outside ``raw`` is read; this function is a pure transform.
    """
    version = _require(raw, "version", int, path="version")
    if version != SUPPORTED_VERSION:
        raise PDBPanelConfigError(
            "version",
            f"unsupported panel config version: {version} (expected {SUPPORTED_VERSION})",
        )

    default_panel = _require(raw, "default_panel", str, path="default_panel")
    default_prompt_set = _require(raw, "default_prompt_set", str, path="default_prompt_set")

    budget_raw = _require(raw, "budgets", Mapping, path="budgets")
    budget = _parse_budget(budget_raw)

    slots_raw = _require(raw, "slots", Mapping, path="slots")
    if not slots_raw:
        raise PDBPanelConfigError("slots", "slots map must not be empty")
    slots = {sid: _parse_slot(sid, cfg) for sid, cfg in slots_raw.items()}

    panels_raw = _require(raw, "panels", Mapping, path="panels")
    if not panels_raw:
        raise PDBPanelConfigError("panels", "panels map must not be empty")
    panels = {pid: _parse_panel(pid, cfg, slots) for pid, cfg in panels_raw.items()}

    prompt_sets_raw = _require(raw, "prompt_sets", Mapping, path="prompt_sets")
    if not prompt_sets_raw:
        raise PDBPanelConfigError("prompt_sets", "prompt_sets map must not be empty")
    prompt_sets = {pid: _parse_prompt_set(pid, cfg) for pid, cfg in prompt_sets_raw.items()}

    if default_panel not in panels:
        raise PDBPanelConfigError(
            "default_panel",
            f"default_panel {default_panel!r} is not defined in panels",
        )
    if default_prompt_set not in prompt_sets:
        raise PDBPanelConfigError(
            "default_prompt_set",
            f"default_prompt_set {default_prompt_set!r} is not defined in prompt_sets",
        )

    for panel_id, panel in panels.items():
        _validate_panel_against_slots(panel_id, panel, slots)

    return PDBPanelConfig(
        version=version,
        default_panel=default_panel,
        default_prompt_set=default_prompt_set,
        budget=budget,
        slots=slots,
        panels=panels,
        prompt_sets=prompt_sets,
    )


def _parse_budget(raw: Mapping[str, Any]) -> PDBBudgetConfig:
    per_brief = _require_number(raw, "per_brief_usd", path="budgets.per_brief_usd")
    per_day = _require_number(raw, "per_day_usd", path="budgets.per_day_usd")
    reserve = _require_number(
        raw,
        "reserve_for_manual_escalation_usd",
        path="budgets.reserve_for_manual_escalation_usd",
    )
    if per_brief <= 0:
        raise PDBPanelConfigError(
            "budgets.per_brief_usd",
            f"per_brief_usd must be > 0; got {per_brief}",
        )
    if per_day <= 0:
        raise PDBPanelConfigError(
            "budgets.per_day_usd",
            f"per_day_usd must be > 0; got {per_day}",
        )
    if reserve < 0:
        raise PDBPanelConfigError(
            "budgets.reserve_for_manual_escalation_usd",
            f"reserve_for_manual_escalation_usd must be >= 0; got {reserve}",
        )
    if per_brief > per_day:
        raise PDBPanelConfigError(
            "budgets.per_brief_usd",
            f"per_brief_usd ({per_brief}) exceeds per_day_usd ({per_day}); "
            "a single brief cannot be allowed to exceed the daily pool",
        )
    return PDBBudgetConfig(
        per_brief_usd=float(per_brief),
        per_day_usd=float(per_day),
        reserve_for_manual_escalation_usd=float(reserve),
    )


def _parse_slot(slot_id: str, raw: Any) -> PDBPanelSlot:
    path = f"slots.{slot_id}"
    if not isinstance(raw, Mapping):
        raise PDBPanelConfigError(
            path, f"slot definition must be a mapping; got {type(raw).__name__}"
        )
    review_role = _require(raw, "review_role", str, path=f"{path}.review_role")
    lens = _require(raw, "lens", str, path=f"{path}.lens")
    family = _require(raw, "family", str, path=f"{path}.family")
    candidates_raw = _require(raw, "candidates", list, path=f"{path}.candidates")
    if not candidates_raw:
        raise PDBPanelConfigError(
            f"{path}.candidates",
            "candidates list must not be empty",
        )
    candidates: list[str] = []
    for idx, cand in enumerate(candidates_raw):
        if not isinstance(cand, str) or not cand:
            raise PDBPanelConfigError(
                f"{path}.candidates[{idx}]",
                "candidate provider must be a non-empty string",
            )
        candidates.append(cand)
    required_raw = raw.get("required", False)
    if not isinstance(required_raw, bool):
        raise PDBPanelConfigError(
            f"{path}.required",
            f"required must be a boolean; got {type(required_raw).__name__}",
        )
    return PDBPanelSlot(
        slot_id=slot_id,
        review_role=review_role,
        lens=lens,
        family=family,
        candidates=tuple(candidates),
        required=required_raw,
    )


def _parse_panel(
    panel_id: str,
    raw: Any,
    slots: Mapping[str, PDBPanelSlot],
) -> PDBPanelDefinition:
    path = f"panels.{panel_id}"
    if not isinstance(raw, Mapping):
        raise PDBPanelConfigError(
            path,
            f"panel definition must be a mapping; got {type(raw).__name__}",
        )
    findings_slots_raw = _require(raw, "findings_slots", list, path=f"{path}.findings_slots")
    if not findings_slots_raw:
        raise PDBPanelConfigError(
            f"{path}.findings_slots",
            "findings_slots must not be empty",
        )
    findings_slots: list[str] = []
    seen_findings: set[str] = set()
    for idx, slot_id in enumerate(findings_slots_raw):
        if not isinstance(slot_id, str):
            raise PDBPanelConfigError(
                f"{path}.findings_slots[{idx}]",
                "slot id must be a string",
            )
        if slot_id in seen_findings:
            raise PDBPanelConfigError(
                f"{path}.findings_slots[{idx}]",
                f"duplicate slot id {slot_id!r}",
            )
        seen_findings.add(slot_id)
        findings_slots.append(slot_id)

    critique_raw = raw.get("critique_slots", SAME_AS_FINDINGS_SENTINEL)
    if critique_raw == SAME_AS_FINDINGS_SENTINEL:
        critique_slots = tuple(findings_slots)
    elif isinstance(critique_raw, list):
        critique_slots_list: list[str] = []
        seen_critique: set[str] = set()
        for idx, slot_id in enumerate(critique_raw):
            if not isinstance(slot_id, str):
                raise PDBPanelConfigError(
                    f"{path}.critique_slots[{idx}]",
                    "slot id must be a string",
                )
            if slot_id in seen_critique:
                raise PDBPanelConfigError(
                    f"{path}.critique_slots[{idx}]",
                    f"duplicate slot id {slot_id!r}",
                )
            seen_critique.add(slot_id)
            critique_slots_list.append(slot_id)
        critique_slots = tuple(critique_slots_list)
    else:
        raise PDBPanelConfigError(
            f"{path}.critique_slots",
            "critique_slots must be either 'same_as_findings' or a list of slot ids",
        )

    synthesizer_slot = _require(raw, "synthesizer_slot", str, path=f"{path}.synthesizer_slot")

    return PDBPanelDefinition(
        panel_id=panel_id,
        findings_slots=tuple(findings_slots),
        critique_slots=critique_slots,
        synthesizer_slot=synthesizer_slot,
    )


def _parse_prompt_set(prompt_set_id: str, raw: Any) -> PDBPromptSet:
    path = f"prompt_sets.{prompt_set_id}"
    if not isinstance(raw, Mapping):
        raise PDBPanelConfigError(
            path,
            f"prompt set definition must be a mapping; got {type(raw).__name__}",
        )
    findings_prompt = _require(raw, "findings_prompt", str, path=f"{path}.findings_prompt")
    critique_prompt = _require(raw, "critique_prompt", str, path=f"{path}.critique_prompt")
    synthesis_prompt = _require(raw, "synthesis_prompt", str, path=f"{path}.synthesis_prompt")
    return PDBPromptSet(
        prompt_set_id=prompt_set_id,
        findings_prompt=findings_prompt,
        critique_prompt=critique_prompt,
        synthesis_prompt=synthesis_prompt,
    )


def _validate_panel_against_slots(
    panel_id: str,
    panel: PDBPanelDefinition,
    slots: Mapping[str, PDBPanelSlot],
) -> None:
    path = f"panels.{panel_id}"

    for idx, slot_id in enumerate(panel.findings_slots):
        if slot_id not in slots:
            raise PDBPanelConfigError(
                f"{path}.findings_slots[{idx}]",
                f"slot {slot_id!r} is not defined in slots",
            )
    for idx, slot_id in enumerate(panel.critique_slots):
        if slot_id not in slots:
            raise PDBPanelConfigError(
                f"{path}.critique_slots[{idx}]",
                f"slot {slot_id!r} is not defined in slots",
            )
    if panel.synthesizer_slot not in slots:
        raise PDBPanelConfigError(
            f"{path}.synthesizer_slot",
            f"synthesizer_slot {panel.synthesizer_slot!r} is not defined in slots",
        )

    findings_lookup = {sid: slots[sid] for sid in panel.findings_slots}
    required_missing = sorted(
        slot.slot_id
        for slot in slots.values()
        if slot.required and slot.slot_id not in findings_lookup
    )
    if required_missing:
        raise PDBPanelConfigError(
            f"{path}.findings_slots",
            "findings_slots must include every required=true slot; missing: "
            + ", ".join(required_missing),
        )

    core_slots = [slot for slot in findings_lookup.values() if slot.lens == "core"]
    if len(core_slots) < 2:
        raise PDBPanelConfigError(
            f"{path}.findings_slots",
            f"findings_slots must include at least two core-lens slots; got {len(core_slots)}",
        )

    non_core = [slot for slot in findings_lookup.values() if slot.lens != "core"]
    if not non_core:
        raise PDBPanelConfigError(
            f"{path}.findings_slots",
            "findings_slots must include at least one non-core lens; "
            "panel would otherwise collapse to core-only heterogeneity",
        )

    if panel.synthesizer_slot not in findings_lookup:
        raise PDBPanelConfigError(
            f"{path}.synthesizer_slot",
            f"synthesizer_slot {panel.synthesizer_slot!r} must also appear in findings_slots",
        )


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def panel_slots(config: PDBPanelConfig, panel_id: str) -> tuple[PDBPanelSlot, ...]:
    """Return the ordered slot objects participating in the given panel.

    Order mirrors ``findings_slots`` so deterministic downstream code
    can iterate the roster without re-sorting.
    """
    if panel_id not in config.panels:
        raise PDBPanelConfigError(
            "panels",
            f"panel {panel_id!r} is not defined in config.panels",
        )
    panel = config.panels[panel_id]
    return tuple(config.slots[sid] for sid in panel.findings_slots)


def provider_slot_definitions(
    config: PDBPanelConfig,
    panel_id: str,
) -> tuple[ProviderSlotDefinition, ...]:
    """Project the panel's slots into the landed provider-slot resolver shape.

    The resolver in :mod:`aragora.review.provider_slots` is the single
    source of truth for candidate availability; PDB does not fork it.
    """
    return tuple(slot.to_provider_slot_definition() for slot in panel_slots(config, panel_id))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require(raw: Mapping[str, Any], key: str, kind: type, *, path: str) -> Any:
    """Assert that ``raw[key]`` exists and is an instance of ``kind``."""
    if key not in raw:
        raise PDBPanelConfigError(path, f"required key {key!r} is missing")
    value = raw[key]
    if not isinstance(value, kind):
        raise PDBPanelConfigError(
            path,
            f"expected {kind.__name__}; got {type(value).__name__}",
        )
    return value


def _require_number(raw: Mapping[str, Any], key: str, *, path: str) -> float:
    """Require a non-bool numeric value. YAML scalars may be ``int`` or ``float``."""
    if key not in raw:
        raise PDBPanelConfigError(path, f"required key {key!r} is missing")
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PDBPanelConfigError(
            path,
            f"expected numeric value; got {type(value).__name__}",
        )
    return float(value)
