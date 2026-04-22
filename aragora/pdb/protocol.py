"""PDB-specific wrapper around :mod:`aragora.brief_engine.protocol`.

The generic Protocol B executor moved to
:mod:`aragora.brief_engine.protocol` in the Phase 1 brief-engine
extraction. This module re-exports the generic types under their
legacy ``PDB*`` names and supplies the Mode-3-specific wiring — the
PDB panel config default and the PDB prompt renderer — so existing
callers of :func:`run_protocol_b` continue to work unchanged.

Do not add new executor logic here — put it in
:mod:`aragora.brief_engine.protocol` instead.
"""

from __future__ import annotations

from aragora.brief_engine.budget import (
    BriefBudgetLedger,
    SlotCostEstimator,
)
from aragora.brief_engine.panel_config import BriefPanelConfig
from aragora.brief_engine.protocol import (
    STATUS_BUDGET_EXCEEDED,
    STATUS_FAILED_CLOSED,
    STATUS_PANEL_DEGRADED,
    STATUS_PANEL_EXECUTED,
    BriefExecutionInput,
    BriefExecutionResult,
    BriefExecutionStatus,
    BriefPromptRenderer,
    ProviderInvoker,
    SlotCritiqueResponse,
    SlotFindingsResponse,
    SynthesisResponse,
    run_brief_protocol_b,
)
from aragora.review.provider_slots import ProviderSlotResolver

# Legacy PDB-named aliases.
PDBExecutionInput = BriefExecutionInput
PDBExecutionResult = BriefExecutionResult
PDBExecutionStatus = BriefExecutionStatus

__all__ = [
    "PDBExecutionInput",
    "PDBExecutionResult",
    "PDBExecutionStatus",
    "ProviderInvoker",
    "SlotCritiqueResponse",
    "SlotFindingsResponse",
    "SynthesisResponse",
    "run_protocol_b",
    "STATUS_PANEL_EXECUTED",
    "STATUS_PANEL_DEGRADED",
    "STATUS_BUDGET_EXCEEDED",
    "STATUS_FAILED_CLOSED",
]


def _pdb_prompt_renderer() -> BriefPromptRenderer:
    """Return the PDB prompt renderer.

    Imports :mod:`aragora.pdb.prompts` lazily so the generic brief-engine
    layer never needs to know about PDB-specific templates.
    """
    from aragora.pdb.prompts import critique_prompt, findings_prompt, synthesis_prompt

    return BriefPromptRenderer(
        findings_prompt=findings_prompt,
        critique_prompt=critique_prompt,
        synthesis_prompt=synthesis_prompt,
    )


def run_protocol_b(
    input: PDBExecutionInput,
    *,
    invoker: ProviderInvoker,
    config: BriefPanelConfig | None = None,
    ledger: BriefBudgetLedger | None = None,
    resolver: ProviderSlotResolver | None = None,
    cost_estimator: SlotCostEstimator | None = None,
    clock: "_Clock | None" = None,
) -> PDBExecutionResult:
    """Run Mode 3 Protocol B and return the structured outcome.

    Thin wrapper around
    :func:`aragora.brief_engine.protocol.run_brief_protocol_b` that
    injects the Mode 3 PDB prompt renderer and falls back to the PDB
    default panel config (``aragora/config/pdb_panel.yaml``) when the
    caller omits ``config``.
    """
    if config is None:
        # Imported lazily so the generic brief-engine layer stays free
        # of PDB-specific defaults.
        from aragora.pdb.panel_config import load_panel_config

        config = load_panel_config()
    return run_brief_protocol_b(
        input,
        invoker=invoker,
        config=config,
        prompts=_pdb_prompt_renderer(),
        ledger=ledger,
        resolver=resolver,
        cost_estimator=cost_estimator,
        clock=clock,
    )


# Re-export the clock protocol for callers that reference
# ``aragora.pdb.protocol._Clock`` via type annotations.
from aragora.brief_engine.protocol import _Clock as _Clock  # noqa: E402,F401
