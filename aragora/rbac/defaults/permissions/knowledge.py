"""
RBAC Permissions for Knowledge, Memory, Reasoning, and Provenance resources.

Contains permissions related to:
- Knowledge mound and curation
- Memory systems
- Reasoning and belief networks
- Provenance tracking
- Evidence and training data
- Analytics and introspection
- RLM (Recursive Language Models)
- Codebase analysis
- Evolution
"""

from __future__ import annotations

from aragora.rbac.models import Action, Permission, ResourceType

from ._helpers import _permission

# ============================================================================
# KNOWLEDGE PERMISSIONS
# ============================================================================

PERM_KNOWLEDGE_READ = _permission(
    ResourceType.KNOWLEDGE,
    Action.READ,
    "View Knowledge",
    "View knowledge base and mound content",
)
PERM_KNOWLEDGE_UPDATE = _permission(
    ResourceType.KNOWLEDGE,
    Action.UPDATE,
    "Update Knowledge",
    "Modify knowledge base curation and settings",
)
PERM_KNOWLEDGE_WRITE = _permission(
    ResourceType.KNOWLEDGE, Action.WRITE, "Write Knowledge", "Create and update knowledge entries"
)
PERM_KNOWLEDGE_DELETE = _permission(
    ResourceType.KNOWLEDGE, Action.DELETE, "Delete Knowledge", "Delete knowledge entries"
)
PERM_KNOWLEDGE_SHARE = _permission(
    ResourceType.KNOWLEDGE, Action.SHARE, "Share Knowledge", "Share knowledge with others"
)
PERM_CULTURE_READ = _permission(
    ResourceType.KNOWLEDGE,
    Action.READ,
    "View Culture",
    "View organizational culture patterns",
)
PERM_CULTURE_WRITE = _permission(
    ResourceType.KNOWLEDGE,
    Action.UPDATE,
    "Update Culture",
    "Modify culture patterns and promote to organization",
)

# ============================================================================
# MEMORY PERMISSIONS
# ============================================================================

PERM_MEMORY_READ = _permission(
    ResourceType.MEMORY, Action.READ, "View Memory", "View memory contents and analytics"
)
PERM_MEMORY_UPDATE = _permission(
    ResourceType.MEMORY, Action.UPDATE, "Update Memory", "Modify memory contents"
)
PERM_MEMORY_DELETE = _permission(
    ResourceType.MEMORY, Action.DELETE, "Delete Memory", "Clear memory contents"
)
PERM_MEMORY_WRITE = _permission(
    ResourceType.MEMORY, Action.WRITE, "Write Memory", "Full write access to memory"
)

# ============================================================================
# REASONING PERMISSIONS
# ============================================================================

PERM_REASONING_READ = _permission(
    ResourceType.REASONING,
    Action.READ,
    "View Reasoning",
    "Access belief networks and reasoning analysis",
)
PERM_REASONING_UPDATE = _permission(
    ResourceType.REASONING,
    Action.UPDATE,
    "Update Reasoning",
    "Modify belief networks and propagation",
)

# ============================================================================
# PROVENANCE PERMISSIONS
# ============================================================================

PERM_PROVENANCE_READ = _permission(
    ResourceType.PROVENANCE,
    Action.READ,
    "View Provenance",
    "View decision provenance and audit trails",
)
PERM_PROVENANCE_VERIFY = _permission(
    ResourceType.PROVENANCE,
    Action.VERIFY,
    "Verify Provenance",
    "Verify integrity of provenance chains",
)
PERM_PROVENANCE_EXPORT = _permission(
    ResourceType.PROVENANCE,
    Action.EXPORT_DATA,
    "Export Provenance",
    "Export provenance reports for compliance",
)

# ============================================================================
# TRAINING & EVIDENCE PERMISSIONS
# ============================================================================

PERM_TRAINING_READ = _permission(
    ResourceType.TRAINING, Action.READ, "View Training Data", "Access training data exports"
)
PERM_TRAINING_CREATE = _permission(
    ResourceType.TRAINING,
    Action.CREATE,
    "Create Training Exports",
    "Generate training data exports",
)
PERM_EVIDENCE_READ = _permission(
    ResourceType.EVIDENCE, Action.READ, "View Evidence", "Access evidence and citations"
)
PERM_EVIDENCE_CREATE = _permission(
    ResourceType.EVIDENCE, Action.CREATE, "Add Evidence", "Add new evidence sources"
)
PERM_EVIDENCE_DELETE = _permission(
    ResourceType.EVIDENCE,
    Action.DELETE,
    "Delete Evidence",
    "Permanently remove evidence records",
)

# ============================================================================
# ANALYTICS PERMISSIONS
# ============================================================================

PERM_ANALYTICS_READ = _permission(
    ResourceType.ANALYTICS, Action.READ, "View Analytics", "Access analytics dashboards"
)
PERM_ANALYTICS_EXPORT = _permission(
    ResourceType.ANALYTICS, Action.EXPORT_DATA, "Export Analytics", "Export analytics data"
)
PERM_PERFORMANCE_READ = _permission(
    ResourceType.ANALYTICS,
    Action.READ,
    "View Performance",
    "View agent performance metrics and rankings",
)
PERM_PERFORMANCE_WRITE = _permission(
    ResourceType.ANALYTICS,
    Action.UPDATE,
    "Update Performance",
    "Modify agent performance data and ELO adjustments",
)

# ============================================================================
# INTROSPECTION & HISTORY PERMISSIONS
# ============================================================================

PERM_INTROSPECTION_READ = _permission(
    ResourceType.INTROSPECTION,
    Action.READ,
    "View Introspection",
    "Access system introspection and agent status",
)
PERM_HISTORY_READ = _permission(
    ResourceType.INTROSPECTION,
    Action.EXPORT_HISTORY,
    "View History",
    "Access debate and system history data",
)

# ============================================================================
# RLM PERMISSIONS
# ============================================================================

PERM_RLM_READ = _permission(
    ResourceType.RLM, Action.READ, "View RLM", "View recursive language model state"
)
PERM_RLM_CREATE = _permission(ResourceType.RLM, Action.CREATE, "Create RLM", "Create RLM sessions")

# ============================================================================
# CODEBASE PERMISSIONS
# ============================================================================

PERM_CODEBASE_READ = _permission(
    ResourceType.CODEBASE, Action.READ, "View Codebase Analysis", "View codebase analysis results"
)
PERM_CODEBASE_ANALYZE = _permission(
    ResourceType.CODEBASE, Action.RUN, "Analyze Codebase", "Run codebase analysis operations"
)
PERM_CODEBASE_WRITE = _permission(
    ResourceType.CODEBASE, Action.WRITE, "Modify Codebase", "Modify codebase analysis settings"
)

# ============================================================================
# EVOLUTION PERMISSIONS
# ============================================================================

PERM_EVOLUTION_READ = _permission(
    ResourceType.EVOLUTION,
    Action.READ,
    "View Evolution",
    "View prompt evolution history, patterns, and summaries",
)

# ============================================================================
# PULSE PERMISSIONS
# ============================================================================

PERM_PULSE_READ = _permission(
    ResourceType.PULSE, Action.READ, "View Pulse", "View trending topics and pulse data"
)
PERM_PULSE_CREATE = _permission(
    ResourceType.PULSE, Action.CREATE, "Create Pulse", "Create pulse monitoring"
)
PERM_PULSE_UPDATE = _permission(
    ResourceType.PULSE, Action.UPDATE, "Update Pulse", "Modify pulse settings"
)

# All knowledge-related permission exports
__all__ = [
    # Knowledge
    "PERM_KNOWLEDGE_READ",
    "PERM_KNOWLEDGE_UPDATE",
    "PERM_KNOWLEDGE_WRITE",
    "PERM_KNOWLEDGE_DELETE",
    "PERM_KNOWLEDGE_SHARE",
    "PERM_CULTURE_READ",
    "PERM_CULTURE_WRITE",
    # Memory
    "PERM_MEMORY_READ",
    "PERM_MEMORY_UPDATE",
    "PERM_MEMORY_DELETE",
    "PERM_MEMORY_WRITE",
    # Reasoning
    "PERM_REASONING_READ",
    "PERM_REASONING_UPDATE",
    # Provenance
    "PERM_PROVENANCE_READ",
    "PERM_PROVENANCE_VERIFY",
    "PERM_PROVENANCE_EXPORT",
    # Training & Evidence
    "PERM_TRAINING_READ",
    "PERM_TRAINING_CREATE",
    "PERM_EVIDENCE_READ",
    "PERM_EVIDENCE_CREATE",
    "PERM_EVIDENCE_DELETE",
    # Analytics
    "PERM_ANALYTICS_READ",
    "PERM_ANALYTICS_EXPORT",
    "PERM_PERFORMANCE_READ",
    "PERM_PERFORMANCE_WRITE",
    # Introspection & History
    "PERM_INTROSPECTION_READ",
    "PERM_HISTORY_READ",
    # RLM
    "PERM_RLM_READ",
    "PERM_RLM_CREATE",
    # Codebase
    "PERM_CODEBASE_READ",
    "PERM_CODEBASE_ANALYZE",
    "PERM_CODEBASE_WRITE",
    # Evolution
    "PERM_EVOLUTION_READ",
    # Pulse
    "PERM_PULSE_READ",
    "PERM_PULSE_CREATE",
    "PERM_PULSE_UPDATE",
]
