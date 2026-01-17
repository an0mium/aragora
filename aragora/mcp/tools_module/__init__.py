"""
MCP Tool Definitions for Aragora.

This package organizes MCP tools into logical modules:
- debate: Core debate operations
- gauntlet: Document stress-testing
- agent: Agent management and breeding
- memory: Memory tier operations
- checkpoint: Debate checkpoint management
- verification: Consensus verification and proofs
- evidence: Evidence collection and citation
- trending: Trending topic analysis
"""

from aragora.mcp.tools_module.debate import (
    run_debate_tool,
    get_debate_tool,
    search_debates_tool,
    fork_debate_tool,
    get_forks_tool,
)
from aragora.mcp.tools_module.gauntlet import run_gauntlet_tool
from aragora.mcp.tools_module.agent import (
    list_agents_tool,
    get_agent_history_tool,
    get_agent_lineage_tool,
    breed_agents_tool,
)
from aragora.mcp.tools_module.memory import (
    query_memory_tool,
    store_memory_tool,
    get_memory_pressure_tool,
)
from aragora.mcp.tools_module.checkpoint import (
    create_checkpoint_tool,
    list_checkpoints_tool,
    resume_checkpoint_tool,
    delete_checkpoint_tool,
)
from aragora.mcp.tools_module.verification import (
    get_consensus_proofs_tool,
    verify_consensus_tool,
    generate_proof_tool,
)
from aragora.mcp.tools_module.evidence import (
    search_evidence_tool,
    cite_evidence_tool,
    verify_citation_tool,
)
from aragora.mcp.tools_module.trending import list_trending_topics_tool
from aragora.mcp.tools_module.audit import (
    list_audit_presets_tool,
    list_audit_types_tool,
    get_audit_preset_tool,
    create_audit_session_tool,
    run_audit_tool,
    get_audit_status_tool,
    get_audit_findings_tool,
    update_finding_status_tool,
    run_quick_audit_tool,
)

__all__ = [
    # Debate tools
    "run_debate_tool",
    "get_debate_tool",
    "search_debates_tool",
    "fork_debate_tool",
    "get_forks_tool",
    # Gauntlet tools
    "run_gauntlet_tool",
    # Agent tools
    "list_agents_tool",
    "get_agent_history_tool",
    "get_agent_lineage_tool",
    "breed_agents_tool",
    # Memory tools
    "query_memory_tool",
    "store_memory_tool",
    "get_memory_pressure_tool",
    # Checkpoint tools
    "create_checkpoint_tool",
    "list_checkpoints_tool",
    "resume_checkpoint_tool",
    "delete_checkpoint_tool",
    # Verification tools
    "get_consensus_proofs_tool",
    "verify_consensus_tool",
    "generate_proof_tool",
    # Evidence tools
    "search_evidence_tool",
    "cite_evidence_tool",
    "verify_citation_tool",
    # Trending tools
    "list_trending_topics_tool",
    # Audit tools
    "list_audit_presets_tool",
    "list_audit_types_tool",
    "get_audit_preset_tool",
    "create_audit_session_tool",
    "run_audit_tool",
    "get_audit_status_tool",
    "get_audit_findings_tool",
    "update_finding_status_tool",
    "run_quick_audit_tool",
]
