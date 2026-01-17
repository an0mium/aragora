"""
External Harness Integration Module.

Provides integration with external code analysis tools:
- Claude Code CLI for code review and analysis
- Codex for code generation and understanding
- Kilo Code for specialized analysis

Usage:
    from aragora.harnesses import ClaudeCodeHarness, HarnessResult

    harness = ClaudeCodeHarness()
    result = await harness.analyze_repository(
        repo_path=Path("/path/to/repo"),
        analysis_type="security",
    )
"""

from aragora.harnesses.base import (
    CodeAnalysisHarness,
    HarnessConfig,
    HarnessResult,
    HarnessError,
    AnalysisType,
)
from aragora.harnesses.claude_code import (
    ClaudeCodeHarness,
    ClaudeCodeConfig,
)
from aragora.harnesses.adapter import (
    HarnessResultAdapter,
    adapt_to_audit_findings,
)

__all__ = [
    # Base
    "CodeAnalysisHarness",
    "HarnessConfig",
    "HarnessResult",
    "HarnessError",
    "AnalysisType",
    # Claude Code
    "ClaudeCodeHarness",
    "ClaudeCodeConfig",
    # Adapter
    "HarnessResultAdapter",
    "adapt_to_audit_findings",
]
