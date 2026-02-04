"""
Task Execution Module for Aragora.

Bridges debate decisions to actionable task execution by routing
task types to appropriate workflow definitions and execution paths.
"""

from __future__ import annotations

from aragora.tasks.router import TaskRoute, TaskRouter

__all__ = [
    "TaskRoute",
    "TaskRouter",
]
