"""
RBAC Permissions for middleware route resources.

Contains permissions referenced by DEFAULT_ROUTE_PERMISSIONS in middleware.py
that don't belong to other domain-specific permission modules.
"""

from __future__ import annotations

from aragora.rbac.models import Action, ResourceType

from ._helpers import _permission

# ============================================================================
# AUDITING PERMISSIONS (Red team operations)
# ============================================================================

PERM_AUDITING_READ = _permission(
    ResourceType.AUDITING, Action.READ, "View Auditing", "View red team results"
)
PERM_AUDITING_CREATE = _permission(
    ResourceType.AUDITING, Action.CREATE, "Create Auditing", "Start red team operations"
)

# ============================================================================
# BELIEF NETWORK PERMISSIONS
# ============================================================================

PERM_BELIEF_READ = _permission(
    ResourceType.BELIEF, Action.READ, "View Belief Network", "View belief network data"
)
PERM_BELIEF_WRITE = _permission(
    ResourceType.BELIEF, Action.WRITE, "Update Belief Network", "Modify belief network data"
)

# ============================================================================
# BILLING - additional write permission
# ============================================================================

PERM_BILLING_WRITE = _permission(
    ResourceType.BILLING, Action.WRITE, "Update Billing", "Modify billing settings"
)

# ============================================================================
# CONNECTOR - singular form aliases for middleware routes
# ============================================================================

PERM_CONNECTOR_CREATE_ALIAS = _permission(
    ResourceType.CONNECTOR, Action.CREATE, "Create Connector", "Create new connectors"
)
PERM_CONNECTOR_DELETE_ALIAS = _permission(
    ResourceType.CONNECTOR, Action.DELETE, "Delete Connector", "Remove connectors"
)

# ============================================================================
# CONSENSUS PERMISSIONS
# ============================================================================

PERM_CONSENSUS_READ = _permission(
    ResourceType.CONSENSUS, Action.READ, "View Consensus", "View consensus data"
)
PERM_CONSENSUS_CREATE = _permission(
    ResourceType.CONSENSUS, Action.CREATE, "Create Consensus", "Submit consensus data"
)

# ============================================================================
# DOCUMENTS - additional write permission
# ============================================================================

PERM_DOCUMENTS_WRITE = _permission(
    ResourceType.DOCUMENTS, Action.WRITE, "Write Documents", "Upload and modify documents"
)

# ============================================================================
# GALLERY PERMISSIONS
# ============================================================================

PERM_GALLERY_READ = _permission(
    ResourceType.GALLERY, Action.READ, "View Gallery", "View gallery items"
)
PERM_GALLERY_WRITE = _permission(
    ResourceType.GALLERY, Action.WRITE, "Update Gallery", "Modify gallery items"
)

# ============================================================================
# GENESIS PERMISSIONS
# ============================================================================

PERM_GENESIS_READ = _permission(
    ResourceType.GENESIS, Action.READ, "View Genesis", "View genesis data"
)
PERM_GENESIS_CREATE = _permission(
    ResourceType.GENESIS, Action.CREATE, "Create Genesis", "Start genesis operations"
)

# ============================================================================
# INSIGHTS PERMISSIONS
# ============================================================================

PERM_INSIGHTS_READ = _permission(
    ResourceType.INSIGHTS, Action.READ, "View Insights", "View system insights"
)

# ============================================================================
# LABORATORY PERMISSIONS
# ============================================================================

PERM_LABORATORY_READ = _permission(
    ResourceType.LABORATORY, Action.READ, "View Laboratory", "View laboratory experiments"
)
PERM_LABORATORY_WRITE = _permission(
    ResourceType.LABORATORY, Action.WRITE, "Update Laboratory", "Modify laboratory experiments"
)

# ============================================================================
# LEARNING PERMISSIONS
# ============================================================================

PERM_LEARNING_READ = _permission(
    ResourceType.LEARNING, Action.READ, "View Learning", "View learning data"
)
PERM_LEARNING_WRITE = _permission(
    ResourceType.LEARNING, Action.WRITE, "Update Learning", "Modify learning data"
)

# ============================================================================
# MOMENTS PERMISSIONS
# ============================================================================

PERM_MOMENTS_READ = _permission(
    ResourceType.MOMENTS, Action.READ, "View Moments", "View moments data"
)
PERM_MOMENTS_WRITE = _permission(
    ResourceType.MOMENTS, Action.WRITE, "Update Moments", "Modify moments data"
)

# ============================================================================
# PERSONAS PERMISSIONS
# ============================================================================

PERM_PERSONAS_READ = _permission(
    ResourceType.PERSONAS, Action.READ, "View Personas", "View persona profiles"
)
PERM_PERSONAS_WRITE = _permission(
    ResourceType.PERSONAS, Action.WRITE, "Update Personas", "Modify persona profiles"
)

# ============================================================================
# PLUGINS - additional read permission
# ============================================================================

PERM_PLUGINS_READ = _permission(
    ResourceType.PLUGINS, Action.READ, "View Plugins", "View installed plugins"
)

# ============================================================================
# PODCAST PERMISSIONS
# ============================================================================

PERM_PODCAST_READ = _permission(
    ResourceType.PODCAST, Action.READ, "View Podcast", "View podcast episodes"
)
PERM_PODCAST_CREATE = _permission(
    ResourceType.PODCAST, Action.CREATE, "Create Podcast", "Create podcast content"
)

# ============================================================================
# PULSE - additional write permission
# ============================================================================

PERM_PULSE_WRITE = _permission(
    ResourceType.PULSE, Action.WRITE, "Write Pulse", "Submit pulse data"
)

# ============================================================================
# RELATIONSHIPS PERMISSIONS
# ============================================================================

PERM_RELATIONSHIPS_READ = _permission(
    ResourceType.RELATIONSHIPS, Action.READ, "View Relationships", "View relationship data"
)
PERM_RELATIONSHIPS_WRITE = _permission(
    ResourceType.RELATIONSHIPS, Action.WRITE, "Update Relationships", "Modify relationship data"
)

# ============================================================================
# REPLAYS - additional create permission
# ============================================================================

PERM_REPLAYS_CREATE = _permission(
    ResourceType.REPLAY, Action.CREATE, "Create Replays", "Create debate replay recordings"
)

# ============================================================================
# TOURNAMENTS PERMISSIONS
# ============================================================================

PERM_TOURNAMENTS_READ = _permission(
    ResourceType.TOURNAMENTS, Action.READ, "View Tournaments", "View tournament data"
)
PERM_TOURNAMENTS_CREATE = _permission(
    ResourceType.TOURNAMENTS, Action.CREATE, "Create Tournaments", "Create tournament events"
)

# ============================================================================
# EVOLUTION - additional write permission
# ============================================================================

PERM_EVOLUTION_WRITE = _permission(
    ResourceType.EVOLUTION, Action.WRITE, "Write Evolution", "Modify evolution data"
)

__all__ = [
    "PERM_AUDITING_READ",
    "PERM_AUDITING_CREATE",
    "PERM_BELIEF_READ",
    "PERM_BELIEF_WRITE",
    "PERM_BILLING_WRITE",
    "PERM_CONSENSUS_READ",
    "PERM_CONSENSUS_CREATE",
    "PERM_DOCUMENTS_WRITE",
    "PERM_GALLERY_READ",
    "PERM_GALLERY_WRITE",
    "PERM_GENESIS_READ",
    "PERM_GENESIS_CREATE",
    "PERM_INSIGHTS_READ",
    "PERM_LABORATORY_READ",
    "PERM_LABORATORY_WRITE",
    "PERM_LEARNING_READ",
    "PERM_LEARNING_WRITE",
    "PERM_MOMENTS_READ",
    "PERM_MOMENTS_WRITE",
    "PERM_PERSONAS_READ",
    "PERM_PERSONAS_WRITE",
    "PERM_PLUGINS_READ",
    "PERM_PODCAST_READ",
    "PERM_PODCAST_CREATE",
    "PERM_PULSE_WRITE",
    "PERM_RELATIONSHIPS_READ",
    "PERM_RELATIONSHIPS_WRITE",
    "PERM_REPLAYS_CREATE",
    "PERM_TOURNAMENTS_READ",
    "PERM_TOURNAMENTS_CREATE",
    "PERM_EVOLUTION_WRITE",
]
