"""Control plane integration for TestFixer tasks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aragora.control_plane.scheduler import TaskPriority
from aragora.nomic.testfixer.http_api import TestFixerRunConfig, run_fix_loop

if TYPE_CHECKING:
    from aragora.control_plane.integration import ControlPlaneIntegration

TESTFIXER_TASK_TYPE = "testfixer"


@dataclass
class TestFixerTaskPayload:
    repo_path: str
    test_command: str
    agents: list[str]
    max_iterations: int = 10
    min_confidence: float = 0.5
    timeout_seconds: float = 300.0
    attempt_store_path: str | None = None
    artifacts_dir: str | None = None
    enable_diagnostics: bool = True


class TestFixerControlPlane:
    """Submit and execute TestFixer tasks via the control plane."""

    def __init__(self, integration: ControlPlaneIntegration):
        self._integration = integration

    async def submit(
        self,
        payload: TestFixerTaskPayload,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> str:
        task_id = await self._integration.submit_task(
            task_type=TESTFIXER_TASK_TYPE,
            payload=payload.__dict__,
            required_capabilities=["testfixer"],
            priority=priority,
            timeout_seconds=payload.timeout_seconds,
            metadata={"task_kind": "testfixer"},
        )
        return task_id

    async def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = TestFixerRunConfig(
            repo_path=Path(payload["repo_path"]),
            test_command=payload["test_command"],
            agents=payload.get("agents", ["codex", "claude"]),
            max_iterations=payload.get("max_iterations", 10),
            min_confidence=payload.get("min_confidence", 0.5),
            timeout_seconds=payload.get("timeout_seconds", 300.0),
            attempt_store_path=Path(payload["attempt_store_path"])
            if payload.get("attempt_store_path")
            else None,
            artifacts_dir=Path(payload["artifacts_dir"]) if payload.get("artifacts_dir") else None,
            enable_diagnostics=payload.get("enable_diagnostics", True),
        )
        result = await run_fix_loop(config)
        return result.to_dict()
