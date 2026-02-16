"""
Hybrid multi-model implementation system.

Updated routing based on empirical performance data (Dec 2025):
- Gemini: Chief Architect - generates implementation plans (1M context window)
- Claude: Primary Implementer - ALL implementation tasks (37% faster, best quality)
- Codex: Code Reviewer - optional QA phase after implementation (high quality, latency-tolerant)

Research findings:
- Claude completed projects in 1h17m vs alternatives' 2h+ (fastest)
- Codex has severe latency issues per GitHub #5149, #1811, #6990 (5-20min for simple tasks)
- Gemini's 1M token context excels at understanding full codebases for planning
- Codex produces excellent review quality when latency isn't critical

Key features:
- Plan-first workflow with machine-readable task decomposition
- Checkpoint persistence for crash recovery
- Transactional safety with git stash rollback
- Optional Codex code review (enable with ARAGORA_CODEX_REVIEW=1)
"""

from .checkpoint import clear_progress, load_progress, save_progress
from .executor import HybridExecutor
from .planner import create_single_task_plan, generate_implement_plan
from .types import ImplementPlan, ImplementProgress, ImplementTask, TaskResult

# Fabric integration is loaded lazily to break a circular import:
#   implement/__init__ → fabric_integration → pipeline → decision_integrity → implement
_LAZY_IMPORTS = {
    "FabricImplementationConfig": ".fabric_integration",
    "FabricImplementationRunner": ".fabric_integration",
    "register_implementation_executor": ".fabric_integration",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Types
    "ImplementTask",
    "ImplementPlan",
    "TaskResult",
    "ImplementProgress",
    # Planner
    "generate_implement_plan",
    "create_single_task_plan",
    # Executor
    "HybridExecutor",
    "FabricImplementationConfig",
    "FabricImplementationRunner",
    "register_implementation_executor",
    # Checkpoint
    "save_progress",
    "load_progress",
    "clear_progress",
]
