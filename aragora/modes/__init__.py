"""
Aragora Mode System.

Provides two complementary mode systems:

1. **Operational Modes** (Kilocode-inspired)
   - Architect, Coder, Reviewer, Debugger, Orchestrator
   - Granular tool access control via ToolGroups
   - Custom mode creation via YAML

2. **Debate Modes**
   - Adversarial red-teaming
   - Specialized debate protocols
"""

# Operational Mode System
from aragora.modes import custom  # Expose submodule for patching
from aragora.modes.base import Mode, ModeRegistry

# Built-in operational modes (auto-registered on import)
from aragora.modes.builtin import (
    ArchitectMode,
    CoderMode,
    DebuggerMode,
    OrchestratorMode,
    ReviewerMode,
    register_all_builtins,
)
from aragora.modes.custom import CustomMode, CustomModeLoader

# Deep Audit Mode (Heavy3-inspired intensive debate protocol)
from aragora.modes.deep_audit import (
    CODE_ARCHITECTURE_AUDIT,
    CONTRACT_AUDIT,
    # Pre-configured protocols
    STRATEGY_AUDIT,
    AuditFinding,
    DeepAuditConfig,
    DeepAuditOrchestrator,
    DeepAuditVerdict,
    run_deep_audit,
)
from aragora.modes.handoff import HandoffContext, ModeHandoff
from aragora.modes.prober import (
    CapabilityProber,
    ContradictionTrap,
    HallucinationBait,
    PersistenceChallenge,
    ProbeBeforePromote,
    ProbeResult,
    ProbeStrategy,
    ProbeType,
    SycophancyTest,
    VulnerabilityReport,
    VulnerabilitySeverity,
    generate_probe_report_markdown,
)

# Debate Modes (existing)
from aragora.modes.redteam import (
    Attack,
    AttackType,
    Defense,
    RedTeamMode,
    RedTeamProtocol,
    RedTeamResult,
    RedTeamRound,
    redteam_code_review,
    redteam_policy,
)
from aragora.modes.tool_groups import ToolGroup, can_use_tool, get_required_group

__all__ = [
    # Operational Mode System
    "ToolGroup",
    "can_use_tool",
    "get_required_group",
    "Mode",
    "ModeRegistry",
    "HandoffContext",
    "ModeHandoff",
    "CustomMode",
    "CustomModeLoader",
    # Built-in Modes
    "ArchitectMode",
    "CoderMode",
    "ReviewerMode",
    "DebuggerMode",
    "OrchestratorMode",
    "register_all_builtins",
    # Debate Modes
    "RedTeamMode",
    "RedTeamProtocol",
    "RedTeamResult",
    "RedTeamRound",
    "Attack",
    "Defense",
    "AttackType",
    "redteam_code_review",
    "redteam_policy",
    # Capability Probing
    "CapabilityProber",
    "VulnerabilityReport",
    "ProbeResult",
    "ProbeType",
    "ProbeStrategy",
    "ContradictionTrap",
    "HallucinationBait",
    "SycophancyTest",
    "PersistenceChallenge",
    "VulnerabilitySeverity",
    "ProbeBeforePromote",
    "generate_probe_report_markdown",
    # Deep Audit Mode
    "DeepAuditOrchestrator",
    "DeepAuditConfig",
    "DeepAuditVerdict",
    "AuditFinding",
    "run_deep_audit",
    "STRATEGY_AUDIT",
    "CONTRACT_AUDIT",
    "CODE_ARCHITECTURE_AUDIT",
]
