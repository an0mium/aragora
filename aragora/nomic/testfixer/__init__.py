"""
TestFixer - Automated test failure diagnosis and repair.

This module implements an autonomous loop that:
1. Runs tests and captures failures
2. Uses AI agents to analyze failures and propose fixes
3. Uses Hegelian debate to cross-check proposed fixes
4. Applies fixes and retests
5. Learns from successful fixes via SDPO

The loop continues until:
- All tests pass
- A maximum iteration count is reached
- The system determines the failures require human intervention

Integration with Aragora:
- Uses debate system for fix proposal cross-checking
- Creates decision receipts for each fix applied
- Learns from successful/failed fixes via SDPO
- Preserves negative space (doesn't paper over real issues)

Usage:
    from aragora.nomic.testfixer import TestFixerOrchestrator

    fixer = TestFixerOrchestrator(
        repo_path="/path/to/repo",
        test_command="pytest tests/ -q --maxfail=1",
    )

    result = await fixer.run_fix_loop(max_iterations=10)
    print(f"Fixed {result.fixes_applied} issues")
"""

from aragora.nomic.testfixer.orchestrator import (
    TestFixerOrchestrator,
    FixLoopResult,
    FixLoopConfig,
)
from aragora.nomic.testfixer.runner import (
    TestRunner,
    TestResult,
    TestFailure,
    RunDiagnostics,
)
from aragora.nomic.testfixer.analyzer import (
    FailureAnalyzer,
    FailureAnalysis,
    FailureCategory,
)
from aragora.nomic.testfixer.proposer import (
    PatchProposer,
    PatchProposal,
    ProposalDebate,
)
from aragora.nomic.testfixer.store import TestFixerAttemptStore
from aragora.nomic.testfixer.generators import AgentCodeGenerator, AgentGeneratorConfig

__all__ = [
    "TestFixerOrchestrator",
    "FixLoopResult",
    "FixLoopConfig",
    "TestRunner",
    "TestResult",
    "TestFailure",
    "RunDiagnostics",
    "FailureAnalyzer",
    "FailureAnalysis",
    "FailureCategory",
    "PatchProposer",
    "PatchProposal",
    "ProposalDebate",
    "TestFixerAttemptStore",
    "AgentCodeGenerator",
    "AgentGeneratorConfig",
]
