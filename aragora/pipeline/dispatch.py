"""OpenClaw dispatch: debate result → safe code execution → receipt.

Provides the default factory binding and public dispatch API that
connects debate results to the ClaudeCodeHarness via CodeImplementationTask,
closing the gap identified in the PMF integration sprint.

Usage:
    from aragora.pipeline.dispatch import dispatch_implementation

    result = await dispatch_implementation(
        debate_result=debate_result,
        repo_path="/path/to/repo",
    )
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


async def create_code_task(
    *,
    implementation_prompt: str,
    files_to_modify: list[str] | None = None,
    repo_path: str = ".",
    timeout_seconds: int = 600,
    harness_type: str = "claude-code",
) -> dict[str, Any]:
    """Default code_task_factory for UnifiedOrchestrator.

    Wires CodeImplementationTask → ClaudeCodeHarness, providing the
    missing binding that the orchestrator requires for OpenClaw execution.

    Args:
        implementation_prompt: Focused instructions for the harness.
        files_to_modify: Optional file scope hint.
        repo_path: Repository path to modify.
        timeout_seconds: Execution timeout.
        harness_type: Harness backend ("claude-code" or "codex").

    Returns:
        Structured result dict with exit_code, files_changed, stdout, etc.
    """
    try:
        from aragora.workflow.nodes.code_implementation import CodeImplementationTask
        from aragora.workflow.step import WorkflowContext
    except ImportError as exc:
        logger.warning("CodeImplementationTask not available: %s", exc)
        return {"success": False, "exit_code": 1, "error": str(exc)}

    task = CodeImplementationTask(
        name="openclaw-dispatch",
        config={
            "repo_path": repo_path,
            "implementation_prompt": implementation_prompt,
            "files_to_modify": files_to_modify or [],
            "timeout_seconds": timeout_seconds,
            "harness_type": harness_type,
        },
    )

    ctx = WorkflowContext(workflow_id="openclaw-dispatch", definition_id="openclaw")
    return await task.execute(ctx)


async def dispatch_implementation(
    debate_result: Any,
    *,
    repo_path: str = ".",
    timeout_seconds: int = 600,
    harness_type: str = "claude-code",
) -> dict[str, Any]:
    """Public dispatch API: debate result → spec extraction → code execution → receipt.

    End-to-end pipeline that:
    1. Extracts an implementation spec from the debate result
    2. Executes it via ClaudeCodeHarness
    3. Creates a ComputerUseActionBundle receipt
    4. Returns the full result with receipt

    Args:
        debate_result: A DebateResult or dict with final_answer.
        repo_path: Repository path to modify.
        timeout_seconds: Execution timeout.
        harness_type: Harness backend.

    Returns:
        Dict with keys: success, spec, execution_result, action_bundle.
    """
    start = time.monotonic()
    result: dict[str, Any] = {
        "success": False,
        "spec": None,
        "execution_result": None,
        "action_bundle": None,
        "duration_seconds": 0.0,
    }

    # Step 1: Extract implementation spec from debate result
    try:
        from aragora.pipeline.spec_extractor import extract_implementation_spec

        spec = extract_implementation_spec(debate_result)
        result["spec"] = {
            "implementation_prompt": getattr(spec, "implementation_prompt", str(spec)),
            "files_to_modify": getattr(spec, "files_to_modify", []),
            "rollback_plan": getattr(spec, "rollback_plan", ""),
        }

        if not getattr(spec, "implementation_prompt", None):
            result["error"] = "No actionable implementation spec extracted from debate result"
            result["duration_seconds"] = time.monotonic() - start
            return result

    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.warning("Spec extraction failed: %s", exc)
        result["error"] = f"Spec extraction failed: {exc}"
        result["duration_seconds"] = time.monotonic() - start
        return result

    # Step 2: Execute via code task factory
    try:
        exec_result = await create_code_task(
            implementation_prompt=spec.implementation_prompt,
            files_to_modify=getattr(spec, "files_to_modify", []),
            repo_path=repo_path,
            timeout_seconds=timeout_seconds,
            harness_type=harness_type,
        )
        result["execution_result"] = exec_result
        result["success"] = exec_result.get("success", False)
    except (RuntimeError, OSError, TimeoutError) as exc:
        logger.warning("Code execution failed: %s", exc)
        result["error"] = f"Execution failed: {exc}"
        result["duration_seconds"] = time.monotonic() - start
        return result

    # Step 3: Create ComputerUseActionBundle receipt
    try:
        from aragora.pipeline.backbone_contracts import ComputerUseActionBundle

        bundle = ComputerUseActionBundle.from_execution_result(
            exec_result,
            harness_name=harness_type,
            action_type="implementation",
        )
        result["action_bundle"] = {
            "harness_name": bundle.harness_name,
            "action_type": bundle.action_type,
            "exit_code": bundle.exit_code,
            "execution_time_seconds": bundle.execution_time_seconds,
            "stdout_summary": (bundle.stdout_summary or "")[:500],
            "policy_violations": bundle.policy_violations,
        }
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("Action bundle creation failed (non-fatal): %s", exc)

    result["duration_seconds"] = time.monotonic() - start
    return result


def get_default_code_task_factory(
    repo_path: str = ".",
    timeout_seconds: int = 600,
    harness_type: str = "claude-code",
) -> Any:
    """Return a code_task_factory callable for injection into UnifiedOrchestrator.

    Usage:
        from aragora.pipeline.dispatch import get_default_code_task_factory

        orchestrator = UnifiedOrchestrator(
            code_task_factory=get_default_code_task_factory(repo_path="/my/repo"),
        )
    """

    async def factory(
        implementation_prompt: str,
        files_to_modify: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await create_code_task(
            implementation_prompt=implementation_prompt,
            files_to_modify=files_to_modify,
            repo_path=repo_path,
            timeout_seconds=timeout_seconds,
            harness_type=harness_type,
        )

    return factory


__all__ = [
    "create_code_task",
    "dispatch_implementation",
    "get_default_code_task_factory",
]
