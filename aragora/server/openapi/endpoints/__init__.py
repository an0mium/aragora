"""
OpenAPI Endpoint Definitions.

Each submodule contains endpoint specifications for a specific domain.
"""

from aragora.server.openapi.endpoints.system import SYSTEM_ENDPOINTS
from aragora.server.openapi.endpoints.agents import AGENT_ENDPOINTS
from aragora.server.openapi.endpoints.debates import DEBATE_ENDPOINTS
from aragora.server.openapi.endpoints.analytics import ANALYTICS_ENDPOINTS
from aragora.server.openapi.endpoints.consensus import CONSENSUS_ENDPOINTS
from aragora.server.openapi.endpoints.relationships import RELATIONSHIP_ENDPOINTS
from aragora.server.openapi.endpoints.memory import MEMORY_ENDPOINTS
from aragora.server.openapi.endpoints.belief import BELIEF_ENDPOINTS
from aragora.server.openapi.endpoints.pulse import PULSE_ENDPOINTS
from aragora.server.openapi.endpoints.metrics import METRICS_ENDPOINTS
from aragora.server.openapi.endpoints.verification import VERIFICATION_ENDPOINTS
from aragora.server.openapi.endpoints.documents import DOCUMENT_ENDPOINTS
from aragora.server.openapi.endpoints.plugins import PLUGIN_ENDPOINTS
from aragora.server.openapi.endpoints.additional import ADDITIONAL_ENDPOINTS
from aragora.server.openapi.endpoints.oauth import OAUTH_ENDPOINTS
from aragora.server.openapi.endpoints.workspace import WORKSPACE_ENDPOINTS
from aragora.server.openapi.endpoints.workflows import WORKFLOW_ENDPOINTS
from aragora.server.openapi.endpoints.cross_pollination import CROSS_POLLINATION_ENDPOINTS
from aragora.server.openapi.endpoints.gauntlet import GAUNTLET_ENDPOINTS
from aragora.server.openapi.endpoints.patterns import PATTERN_ENDPOINTS
from aragora.server.openapi.endpoints.checkpoints import CHECKPOINT_ENDPOINTS
from aragora.server.openapi.endpoints.explainability import EXPLAINABILITY_ENDPOINTS
from aragora.server.openapi.endpoints.workflow_templates import WORKFLOW_TEMPLATES_ENDPOINTS
from aragora.server.openapi.endpoints.control_plane import CONTROL_PLANE_ENDPOINTS
from aragora.server.openapi.endpoints.decisions import DECISION_ENDPOINTS
from aragora.server.openapi.endpoints.codebase_security import CODEBASE_SECURITY_ENDPOINTS
from aragora.server.openapi.endpoints.codebase_metrics import CODEBASE_METRICS_ENDPOINTS

# Combined endpoints dictionary
ALL_ENDPOINTS = {
    **SYSTEM_ENDPOINTS,
    **AGENT_ENDPOINTS,
    **DEBATE_ENDPOINTS,
    **ANALYTICS_ENDPOINTS,
    **CONSENSUS_ENDPOINTS,
    **RELATIONSHIP_ENDPOINTS,
    **MEMORY_ENDPOINTS,
    **BELIEF_ENDPOINTS,
    **PULSE_ENDPOINTS,
    **METRICS_ENDPOINTS,
    **VERIFICATION_ENDPOINTS,
    **DOCUMENT_ENDPOINTS,
    **PLUGIN_ENDPOINTS,
    **ADDITIONAL_ENDPOINTS,
    **OAUTH_ENDPOINTS,
    **WORKSPACE_ENDPOINTS,
    **WORKFLOW_ENDPOINTS,
    **CROSS_POLLINATION_ENDPOINTS,
    **GAUNTLET_ENDPOINTS,
    **PATTERN_ENDPOINTS,
    **CHECKPOINT_ENDPOINTS,
    **EXPLAINABILITY_ENDPOINTS,
    **WORKFLOW_TEMPLATES_ENDPOINTS,
    **CONTROL_PLANE_ENDPOINTS,
    **DECISION_ENDPOINTS,
    **CODEBASE_SECURITY_ENDPOINTS,
    **CODEBASE_METRICS_ENDPOINTS,
}

__all__ = [
    "SYSTEM_ENDPOINTS",
    "AGENT_ENDPOINTS",
    "DEBATE_ENDPOINTS",
    "ANALYTICS_ENDPOINTS",
    "CONSENSUS_ENDPOINTS",
    "RELATIONSHIP_ENDPOINTS",
    "MEMORY_ENDPOINTS",
    "BELIEF_ENDPOINTS",
    "PULSE_ENDPOINTS",
    "METRICS_ENDPOINTS",
    "VERIFICATION_ENDPOINTS",
    "DOCUMENT_ENDPOINTS",
    "PLUGIN_ENDPOINTS",
    "ADDITIONAL_ENDPOINTS",
    "OAUTH_ENDPOINTS",
    "WORKSPACE_ENDPOINTS",
    "WORKFLOW_ENDPOINTS",
    "CROSS_POLLINATION_ENDPOINTS",
    "GAUNTLET_ENDPOINTS",
    "PATTERN_ENDPOINTS",
    "CHECKPOINT_ENDPOINTS",
    "EXPLAINABILITY_ENDPOINTS",
    "WORKFLOW_TEMPLATES_ENDPOINTS",
    "CONTROL_PLANE_ENDPOINTS",
    "DECISION_ENDPOINTS",
    "CODEBASE_SECURITY_ENDPOINTS",
    "CODEBASE_METRICS_ENDPOINTS",
    "ALL_ENDPOINTS",
]
