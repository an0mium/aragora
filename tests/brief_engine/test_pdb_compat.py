"""Back-compat contract for :mod:`aragora.pdb` after the brief-engine extraction.

The Phase 1 refactor lifts the generic Protocol B primitives out of
:mod:`aragora.pdb` into :mod:`aragora.brief_engine`. Mode 3 PDB now
re-exports the engine's public surface under the legacy ``PDB*`` names
so existing callers — the ``review_queue_brief`` handler, the
``generate_one_brief.py`` CLI, and any in-tree tests — continue to
work unchanged.

These assertions pin the back-compat contract: if a future change
drops a PDB alias, the import will fail here and catch the regression
before the handler does. Each assertion also pins the alias to the
corresponding ``Brief*`` symbol so the shim stays a pure re-export
rather than a divergent copy.
"""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Submodule presence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [
        "aragora.pdb",
        "aragora.pdb.brief_state",
        "aragora.pdb.budget",
        "aragora.pdb.input_loader",
        "aragora.pdb.panel_config",
        "aragora.pdb.prompts",
        "aragora.pdb.protocol",
        "aragora.pdb.storage",
        "aragora.pdb.worker",
    ],
)
def test_pdb_submodule_imports_cleanly(module_name: str) -> None:
    """Every PDB-facing submodule must still import without error."""
    module = importlib.import_module(module_name)
    assert module is not None


# ---------------------------------------------------------------------------
# Lifecycle state machine
# ---------------------------------------------------------------------------


def test_brief_state_lifecycle_aliases_match_brief_engine() -> None:
    from aragora.brief_engine.lifecycle import (
        BriefLifecycleState as EngineState,
        StateTransitionError as EngineError,
        LEGAL_TRANSITIONS as ENGINE_TRANSITIONS,
        validate_transition as engine_validate,
    )
    from aragora.pdb import (
        BriefLifecycleState,
        LEGAL_TRANSITIONS,
        StateTransitionError,
        validate_transition,
    )
    from aragora.pdb.brief_state import (
        BriefLifecycleState as ShimState,
        LEGAL_TRANSITIONS as ShimTransitions,
        StateTransitionError as ShimError,
        validate_transition as shim_validate,
    )

    assert BriefLifecycleState is EngineState is ShimState
    assert StateTransitionError is EngineError is ShimError
    assert LEGAL_TRANSITIONS is ENGINE_TRANSITIONS is ShimTransitions
    assert validate_transition is engine_validate is shim_validate


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def test_storage_shim_reexports_engine_surface() -> None:
    from aragora.brief_engine import storage as engine_storage
    from aragora.pdb import storage as pdb_storage

    for name in (
        "briefs_root",
        "QUEUED_SUBDIR",
        "RUNNING_SUBDIR",
        "FAILED_SUBDIR",
        "INVALIDATED_SUBDIR",
        "INDEX_FILENAME",
        "get_state",
        "load_ready_brief",
        "load_latest_ready_brief",
        "find_ready_briefs_for_pr",
        "queue_generation",
        "mark_running",
        "write_running_phase",
        "mark_ready",
        "mark_failed",
        "invalidate_if_head_changed",
        "cancel_generation",
        "append_index_event",
    ):
        assert getattr(pdb_storage, name) is getattr(engine_storage, name), name

    # ``os`` is re-exported on the shim so legacy tests that patch
    # ``storage.os.replace`` continue to resolve a real module.
    assert pdb_storage.os.__name__ == "os"


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


def test_budget_pdb_aliases_are_brief_engine_types() -> None:
    from aragora.brief_engine.budget import (
        BriefBudgetDecision,
        BriefBudgetLedger,
        BriefBudgetReservation,
        BriefBudgetStatus,
    )
    from aragora.pdb.budget import (
        PDBBudgetDecision,
        PDBBudgetLedger,
        PDBBudgetReservation,
        PDBBudgetStatus,
    )

    assert PDBBudgetDecision is BriefBudgetDecision
    assert PDBBudgetLedger is BriefBudgetLedger
    assert PDBBudgetReservation is BriefBudgetReservation
    assert PDBBudgetStatus is BriefBudgetStatus


def test_budget_functions_are_reexported() -> None:
    from aragora.brief_engine import budget as engine_budget
    from aragora.pdb import budget as pdb_budget

    for name in (
        "DEFAULT_SLOT_COST_USD",
        "DEFAULT_SYNTHESIS_COST_USD",
        "SlotCostEstimator",
        "estimate_slot_costs",
        "evaluate_budget",
        "per_brief_cap_usd",
        "reserve",
    ):
        assert getattr(pdb_budget, name) is getattr(engine_budget, name), name


# ---------------------------------------------------------------------------
# Panel config
# ---------------------------------------------------------------------------


def test_panel_config_pdb_aliases_are_brief_engine_types() -> None:
    from aragora.brief_engine.panel_config import (
        BriefBudgetConfig,
        BriefPanelConfig,
        BriefPanelConfigError,
        BriefPanelDefinition,
        BriefPanelSlot,
        BriefPromptSet,
    )
    from aragora.pdb.panel_config import (
        PDBBudgetConfig,
        PDBPanelConfig,
        PDBPanelConfigError,
        PDBPanelDefinition,
        PDBPanelSlot,
        PDBPromptSet,
    )

    assert PDBBudgetConfig is BriefBudgetConfig
    assert PDBPanelConfig is BriefPanelConfig
    assert PDBPanelConfigError is BriefPanelConfigError
    assert PDBPanelDefinition is BriefPanelDefinition
    assert PDBPanelSlot is BriefPanelSlot
    assert PDBPromptSet is BriefPromptSet


def test_panel_config_default_path_is_pdb_yaml() -> None:
    """The Mode 3 default config stays at aragora/config/pdb_panel.yaml."""
    from aragora.pdb.panel_config import DEFAULT_CONFIG_PATH, load_panel_config

    assert DEFAULT_CONFIG_PATH.name == "pdb_panel.yaml"
    assert DEFAULT_CONFIG_PATH.parent.name == "config"
    # Load smoke-tests that the yaml still validates cleanly.
    config = load_panel_config()
    assert config.version == 1
    assert config.default_panel in config.panels


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_protocol_pdb_aliases_are_brief_engine_types() -> None:
    from aragora.brief_engine.protocol import (
        BriefExecutionInput,
        BriefExecutionResult,
        BriefExecutionStatus,
    )
    from aragora.pdb.protocol import (
        PDBExecutionInput,
        PDBExecutionResult,
        PDBExecutionStatus,
    )

    assert PDBExecutionInput is BriefExecutionInput
    assert PDBExecutionResult is BriefExecutionResult
    assert PDBExecutionStatus is BriefExecutionStatus


def test_run_protocol_b_is_callable() -> None:
    """``run_protocol_b`` stays importable from the PDB namespace."""
    from aragora.pdb.protocol import run_protocol_b

    assert callable(run_protocol_b)


def test_status_constants_reexported() -> None:
    from aragora.brief_engine import protocol as engine_protocol
    from aragora.pdb import protocol as pdb_protocol

    for name in (
        "STATUS_PANEL_EXECUTED",
        "STATUS_PANEL_DEGRADED",
        "STATUS_BUDGET_EXCEEDED",
        "STATUS_FAILED_CLOSED",
    ):
        assert getattr(pdb_protocol, name) == getattr(engine_protocol, name), name


def test_invoker_protocol_reexported() -> None:
    from aragora.brief_engine.protocol import ProviderInvoker as EngineInvoker
    from aragora.pdb.protocol import ProviderInvoker

    assert ProviderInvoker is EngineInvoker


def test_response_dataclasses_reexported() -> None:
    from aragora.brief_engine.protocol import (
        SlotCritiqueResponse as EngineCritique,
        SlotFindingsResponse as EngineFindings,
        SynthesisResponse as EngineSynthesis,
    )
    from aragora.pdb.protocol import (
        SlotCritiqueResponse,
        SlotFindingsResponse,
        SynthesisResponse,
    )

    assert SlotCritiqueResponse is EngineCritique
    assert SlotFindingsResponse is EngineFindings
    assert SynthesisResponse is EngineSynthesis


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def test_worker_types_reexported() -> None:
    from aragora.brief_engine.worker import (
        AlreadyRunningError as EngineAlreadyRunningError,
        BriefGenerationWorker as EngineWorker,
        JobKey as EngineJobKey,
        JobRequest as EngineJobRequest,
        get_worker as engine_get_worker,
        reset_worker as engine_reset_worker,
        set_worker as engine_set_worker,
    )
    from aragora.pdb.worker import (
        AlreadyRunningError,
        BriefGenerationWorker,
        JobKey,
        JobRequest,
        get_worker,
        reset_worker,
        set_worker,
    )

    assert AlreadyRunningError is EngineAlreadyRunningError
    assert BriefGenerationWorker is EngineWorker
    assert JobKey is EngineJobKey
    assert JobRequest is EngineJobRequest
    assert get_worker is engine_get_worker
    assert reset_worker is engine_reset_worker
    assert set_worker is engine_set_worker


def test_worker_exposes_run_protocol_b_for_monkeypatch() -> None:
    """Legacy tests patch ``aragora.pdb.worker.run_protocol_b``.

    The worker's default runner must resolve that attribute at call
    time so the override is observed even after the move to the
    brief-engine layer.
    """
    from aragora.pdb import worker as pdb_worker

    assert hasattr(pdb_worker, "run_protocol_b")
    assert callable(pdb_worker.run_protocol_b)


# ---------------------------------------------------------------------------
# Input loader stays PDB-specific
# ---------------------------------------------------------------------------


def test_input_loader_stays_under_pdb() -> None:
    """The PR-review input loader is PDB-specific and was NOT moved."""
    from aragora.pdb import input_loader
    from aragora.pdb.input_loader import (
        InputLoaderError,
        InputLoaderErrorReason,
        LoadedExecutionInput,
        load_execution_input,
    )

    assert input_loader.__name__ == "aragora.pdb.input_loader"
    assert InputLoaderError is not None
    assert InputLoaderErrorReason is not None
    assert LoadedExecutionInput is not None
    assert callable(load_execution_input)


def test_prompts_stay_under_pdb() -> None:
    """The PR-review prompt templates are PDB-specific and were NOT moved."""
    from aragora.pdb import prompts
    from aragora.pdb.prompts import (
        critique_prompt,
        findings_prompt,
        synthesis_prompt,
    )

    assert prompts.__name__ == "aragora.pdb.prompts"
    assert callable(findings_prompt)
    assert callable(critique_prompt)
    assert callable(synthesis_prompt)


# ---------------------------------------------------------------------------
# Direct handler-level import patterns from review_queue_brief.py
# ---------------------------------------------------------------------------


def test_review_queue_brief_handler_import_pattern() -> None:
    """The ``review_queue_brief`` handler uses these exact imports.

    Keeping this smoke-test green guarantees the handler keeps working
    after the refactor without code changes.
    """
    from aragora.pdb import storage  # noqa: F401
    from aragora.pdb.brief_state import BriefLifecycleState  # noqa: F401
    from aragora.pdb.input_loader import (  # noqa: F401
        InputLoaderError,
        InputLoaderErrorReason,
        LoadedExecutionInput,
        load_execution_input,
    )
    from aragora.pdb.protocol import ProviderInvoker  # noqa: F401
    from aragora.pdb.worker import (  # noqa: F401
        AlreadyRunningError,
        BriefGenerationWorker,
        JobKey,
        JobRequest,
    )
