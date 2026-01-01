"""
Hybrid multi-model implementation system.

This module implements the consensus architecture from the 3-provider debate:
- Gemini: Chief Architect - generates implementation plans
- Claude: Primary Implementer - handles complex multi-file changes
- Codex: Specialist - simple isolated tasks

Key features:
- Plan-first workflow with machine-readable task decomposition
- Checkpoint persistence for crash recovery
- Transactional safety with git stash rollback
- Complexity-based model routing
"""

from .checkpoint import clear_progress, load_progress, save_progress
from .executor import HybridExecutor
from .planner import create_single_task_plan, generate_implement_plan
from .types import ImplementPlan, ImplementProgress, ImplementTask, TaskResult


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
    # Checkpoint
    "save_progress",
    "load_progress",
    "clear_progress",
]
