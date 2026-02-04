"""
Task Execution Handler Package.

Exposes the TaskExecutionHandler for registration with the server's
handler registry. This handler bridges debate decisions to actionable
task execution via the workflow engine and control plane scheduler.
"""

from __future__ import annotations

from aragora.server.handlers.tasks.execution import TaskExecutionHandler

__all__ = [
    "TaskExecutionHandler",
]
