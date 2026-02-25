"""Sandbox verification for debate code proposals.

Extracts code blocks from debate proposals and executes them
in an isolated sandbox to verify correctness before final consensus.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Pattern to extract fenced code blocks from proposals
_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


@dataclass
class SandboxVerificationResult:
    """Result of sandbox verification of a code proposal."""

    passed: bool
    stdout: str = ""
    stderr: str = ""
    execution_id: str = ""
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "stdout": self.stdout[:500],
            "stderr": self.stderr[:500],
            "execution_id": self.execution_id,
            "error_message": self.error_message,
        }


def extract_code_blocks(text: str) -> list[str]:
    """Extract fenced code blocks from proposal text."""
    return _CODE_BLOCK_RE.findall(text)


async def verify_code_proposal(
    proposal_text: str,
    sandbox_config: dict[str, Any] | None = None,
) -> SandboxVerificationResult:
    """Run code from a proposal in the sandbox and return results.

    Args:
        proposal_text: The proposal text potentially containing code blocks.
        sandbox_config: Optional sandbox configuration overrides.

    Returns:
        SandboxVerificationResult with pass/fail and output.
    """
    code_blocks = extract_code_blocks(proposal_text)
    if not code_blocks:
        return SandboxVerificationResult(
            passed=True,
            error_message="No code blocks found in proposal",
        )

    combined_code = "\n\n".join(code_blocks)

    try:
        from aragora.sandbox.executor import SandboxConfig, SandboxExecutor, ExecutionStatus

        config = sandbox_config or {}
        sb_config = SandboxConfig()
        sb_config.policy.resource_limits.max_execution_seconds = config.get("timeout_seconds", 30)
        sb_config.policy.resource_limits.max_memory_mb = config.get("max_memory_mb", 256)

        executor = SandboxExecutor(config=sb_config)
        result = await executor.execute(combined_code, language="python")

        passed = result.status == ExecutionStatus.COMPLETED and result.exit_code == 0

        return SandboxVerificationResult(
            passed=passed,
            stdout=result.stdout,
            stderr=result.stderr,
            execution_id=result.execution_id,
        )
    except ImportError:
        logger.debug("SandboxExecutor not available; skipping verification")
        return SandboxVerificationResult(
            passed=True,
            error_message="Sandbox not available",
        )
    except (OSError, RuntimeError, ValueError, TimeoutError) as e:
        logger.warning("Sandbox verification failed: %s", e)
        return SandboxVerificationResult(
            passed=False,
            error_message="Sandbox execution failed",
        )
