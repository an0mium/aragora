"""
RBAC Permission Definitions.

Contains all system-wide permissions organized by resource type.
"""

from __future__ import annotations

from aragora.rbac.models import Action, Permission, ResourceType


def _permission(
    resource: ResourceType,
    action: Action,
    name: str = "",
    description: str = "",
) -> Permission:
    """Helper to create a permission with auto-generated ID."""
    key = f"{resource.value}.{action.value}"
    return Permission(
        id=key,  # Use key as ID for simplicity
        name=name or key.replace(".", " ").replace("_", " ").title(),
        resource=resource,
        action=action,
        description=description,
    )


# ============================================================================
# DEBATE PERMISSIONS
# ============================================================================

PERM_DEBATE_CREATE = _permission(
    ResourceType.DEBATE, Action.CREATE, "Create Debates", "Create new debates"
)
PERM_DEBATE_READ = _permission(
    ResourceType.DEBATE, Action.READ, "View Debates", "View debate details and history"
)
PERM_DEBATE_UPDATE = _permission(
    ResourceType.DEBATE, Action.UPDATE, "Update Debates", "Modify debate settings"
)
PERM_DEBATE_DELETE = _permission(
    ResourceType.DEBATE, Action.DELETE, "Delete Debates", "Delete debates permanently"
)
PERM_DEBATE_RUN = _permission(
    ResourceType.DEBATE, Action.RUN, "Run Debates", "Start and execute debates"
)
PERM_DEBATE_STOP = _permission(
    ResourceType.DEBATE, Action.STOP, "Stop Debates", "Stop running debates"
)
PERM_DEBATE_FORK = _permission(
    ResourceType.DEBATE, Action.FORK, "Fork Debates", "Create branches from existing debates"
)

# ============================================================================
# AGENT PERMISSIONS
# ============================================================================

PERM_AGENT_CREATE = _permission(
    ResourceType.AGENT, Action.CREATE, "Create Agents", "Create custom agent configurations"
)
PERM_AGENT_READ = _permission(
    ResourceType.AGENT, Action.READ, "View Agents", "View agent details and statistics"
)
PERM_AGENT_UPDATE = _permission(
    ResourceType.AGENT, Action.UPDATE, "Update Agents", "Modify agent configurations"
)
PERM_AGENT_DELETE = _permission(
    ResourceType.AGENT, Action.DELETE, "Delete Agents", "Remove agent configurations"
)
PERM_AGENT_DEPLOY = _permission(
    ResourceType.AGENT, Action.DEPLOY, "Deploy Agents", "Deploy agents to production"
)

# ============================================================================
# USER MANAGEMENT PERMISSIONS
# ============================================================================

PERM_USER_READ = _permission(
    ResourceType.USER, Action.READ, "View Users", "View user profiles in organization"
)
PERM_USER_INVITE = _permission(
    ResourceType.USER, Action.INVITE, "Invite Users", "Invite new users to organization"
)
PERM_USER_REMOVE = _permission(
    ResourceType.USER, Action.REMOVE, "Remove Users", "Remove users from organization"
)
PERM_USER_CHANGE_ROLE = _permission(
    ResourceType.USER, Action.CHANGE_ROLE, "Change User Roles", "Modify user role assignments"
)
PERM_USER_IMPERSONATE = _permission(
    ResourceType.USER, Action.IMPERSONATE, "Impersonate Users", "Act on behalf of other users"
)

# ============================================================================
# ORGANIZATION PERMISSIONS
# ============================================================================

PERM_ORG_READ = _permission(
    ResourceType.ORGANIZATION, Action.READ, "View Organization", "View organization settings"
)
PERM_ORG_UPDATE = _permission(
    ResourceType.ORGANIZATION, Action.UPDATE, "Update Organization", "Modify organization settings"
)
PERM_ORG_BILLING = _permission(
    ResourceType.ORGANIZATION,
    Action.MANAGE_BILLING,
    "Manage Billing",
    "Manage organization billing and subscriptions",
)
PERM_ORG_AUDIT = _permission(
    ResourceType.ORGANIZATION,
    Action.VIEW_AUDIT,
    "View Audit Logs",
    "Access organization audit trail",
)
PERM_ORG_EXPORT = _permission(
    ResourceType.ORGANIZATION, Action.EXPORT_DATA, "Export Data", "Export organization data"
)
PERM_ORG_INVITE = _permission(
    ResourceType.ORGANIZATION,
    Action.INVITE,
    "Invite to Organization",
    "Invite new users to the organization",
)
PERM_ORG_DELETE = _permission(
    ResourceType.ORGANIZATION,
    Action.DELETE,
    "Delete Organization",
    "Permanently delete an organization (irreversible)",
)

# ============================================================================
# API PERMISSIONS
# ============================================================================

PERM_API_GENERATE_KEY = _permission(
    ResourceType.API, Action.GENERATE_KEY, "Generate API Keys", "Create new API keys"
)
PERM_API_REVOKE_KEY = _permission(
    ResourceType.API, Action.REVOKE_KEY, "Revoke API Keys", "Revoke existing API keys"
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

# ============================================================================
# WORKFLOW PERMISSIONS
# ============================================================================

PERM_WORKFLOW_CREATE = _permission(
    ResourceType.WORKFLOW, Action.CREATE, "Create Workflows", "Create new workflows"
)
PERM_WORKFLOW_READ = _permission(
    ResourceType.WORKFLOW, Action.READ, "View Workflows", "View workflow definitions and executions"
)
PERM_WORKFLOW_RUN = _permission(
    ResourceType.WORKFLOW, Action.RUN, "Run Workflows", "Execute workflows"
)
PERM_WORKFLOW_DELETE = _permission(
    ResourceType.WORKFLOW, Action.DELETE, "Delete Workflows", "Delete workflow definitions"
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
# INBOX PERMISSIONS
# ============================================================================

PERM_INBOX_READ = _permission(
    ResourceType.INBOX,
    Action.READ,
    "View Inbox",
    "View action items and meetings",
)
PERM_INBOX_UPDATE = _permission(
    ResourceType.INBOX,
    Action.UPDATE,
    "Manage Inbox",
    "Create and manage action items",
)

# ============================================================================
# SKILLS PERMISSIONS
# ============================================================================

PERM_SKILLS_READ = _permission(
    ResourceType.SKILLS,
    Action.READ,
    "View Skills",
    "Browse skill marketplace and view details",
)
PERM_SKILLS_INSTALL = _permission(
    ResourceType.SKILLS,
    Action.UPDATE,
    "Install Skills",
    "Install and uninstall skills",
)
PERM_SKILLS_PUBLISH = _permission(
    ResourceType.SKILLS,
    Action.CREATE,
    "Publish Skills",
    "Publish skills to marketplace",
)
PERM_SKILLS_RATE = _permission(
    ResourceType.SKILLS,
    Action.UPDATE,
    "Rate Skills",
    "Rate and review skills",
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
# DOCUMENT PERMISSIONS
# ============================================================================

PERM_DOCUMENTS_READ = _permission(
    ResourceType.DOCUMENTS, Action.READ, "View Documents", "Access document metadata and queries"
)
PERM_DOCUMENTS_CREATE = _permission(
    ResourceType.DOCUMENTS, Action.CREATE, "Upload Documents", "Upload and process documents"
)
PERM_UPLOAD_CREATE = _permission(
    ResourceType.UPLOAD, Action.CREATE, "Create Uploads", "Create document folder uploads"
)
PERM_SPEECH_CREATE = _permission(
    ResourceType.SPEECH, Action.CREATE, "Create Speech Jobs", "Generate speech transcripts"
)

# ============================================================================
# CONNECTOR PERMISSIONS
# ============================================================================

PERM_CONNECTOR_READ = _permission(
    ResourceType.CONNECTOR, Action.READ, "View Connectors", "View connector configurations"
)
PERM_CONNECTOR_CREATE = _permission(
    ResourceType.CONNECTOR, Action.CREATE, "Create Connectors", "Configure new data connectors"
)
PERM_CONNECTOR_DELETE = _permission(
    ResourceType.CONNECTOR, Action.DELETE, "Delete Connectors", "Remove connector configurations"
)
PERM_CONNECTOR_AUTHORIZE = _permission(
    ResourceType.CONNECTOR,
    Action.AUTHORIZE,
    "Authorize Connectors",
    "Grant OAuth/API credentials for connectors",
)
PERM_CONNECTOR_ROTATE = _permission(
    ResourceType.CONNECTOR,
    Action.ROTATE,
    "Rotate Connector Credentials",
    "Rotate API keys and secrets",
)
PERM_CONNECTOR_TEST = _permission(
    ResourceType.CONNECTOR,
    Action.TEST,
    "Test Connectors",
    "Verify connector health and connectivity",
)
PERM_CONNECTOR_UPDATE = _permission(
    ResourceType.CONNECTOR,
    Action.UPDATE,
    "Update Connectors",
    "Modify connector configurations",
)
PERM_CONNECTOR_ROLLBACK = _permission(
    ResourceType.CONNECTOR,
    Action.ROLLBACK,
    "Rollback Connectors",
    "Revert failed connector operations",
)

# ============================================================================
# BOT & FEEDBACK PERMISSIONS
# ============================================================================

PERM_BOTS_READ = _permission(
    ResourceType.BOT, Action.READ, "View Bot Status", "View bot integration status"
)
PERM_FEEDBACK_READ = _permission(
    ResourceType.FEEDBACK, Action.READ, "View Feedback", "View user feedback and NPS data"
)
PERM_FEEDBACK_WRITE = _permission(
    ResourceType.FEEDBACK, Action.WRITE, "Submit Feedback", "Submit user feedback"
)
PERM_FEEDBACK_ALL = _permission(
    ResourceType.FEEDBACK,
    Action.UPDATE,
    "Feedback Admin",
    "Full feedback administration including summaries",
)

# ============================================================================
# DEVICE PERMISSIONS
# ============================================================================

PERM_DEVICE_READ = _permission(
    ResourceType.DEVICE, Action.READ, "View Devices", "View registered devices"
)
PERM_DEVICE_WRITE = _permission(
    ResourceType.DEVICE, Action.WRITE, "Manage Devices", "Register or remove devices"
)
PERM_DEVICE_NOTIFY = _permission(
    ResourceType.DEVICE, Action.NOTIFY, "Notify Devices", "Send notifications to devices"
)

# ============================================================================
# REPOSITORY PERMISSIONS
# ============================================================================

PERM_REPOSITORY_READ = _permission(
    ResourceType.REPOSITORY, Action.READ, "View Repositories", "View repository indexing status"
)
PERM_REPOSITORY_CREATE = _permission(
    ResourceType.REPOSITORY, Action.CREATE, "Index Repositories", "Start repository indexing"
)
PERM_REPOSITORY_UPDATE = _permission(
    ResourceType.REPOSITORY,
    Action.UPDATE,
    "Update Repositories",
    "Incrementally update repositories",
)
PERM_REPOSITORY_DELETE = _permission(
    ResourceType.REPOSITORY, Action.DELETE, "Remove Repositories", "Remove indexed repositories"
)

# ============================================================================
# MESSAGE BINDINGS PERMISSIONS
# ============================================================================

PERM_BINDINGS_READ = _permission(
    ResourceType.BINDINGS, Action.READ, "View Bindings", "View message bindings"
)
PERM_BINDINGS_CREATE = _permission(
    ResourceType.BINDINGS, Action.CREATE, "Create Bindings", "Create new message bindings"
)
PERM_BINDINGS_UPDATE = _permission(
    ResourceType.BINDINGS, Action.UPDATE, "Update Bindings", "Update message bindings"
)
PERM_BINDINGS_DELETE = _permission(
    ResourceType.BINDINGS, Action.DELETE, "Delete Bindings", "Remove message bindings"
)

# ============================================================================
# ADMIN PERMISSIONS
# ============================================================================

PERM_ADMIN_CONFIG = _permission(
    ResourceType.ADMIN, Action.SYSTEM_CONFIG, "System Configuration", "Modify system-wide settings"
)
PERM_ADMIN_METRICS = _permission(
    ResourceType.ADMIN,
    Action.VIEW_METRICS,
    "View System Metrics",
    "Access system performance metrics",
)
PERM_ADMIN_FEATURES = _permission(
    ResourceType.ADMIN, Action.MANAGE_FEATURES, "Manage Feature Flags", "Enable/disable features"
)
PERM_ADMIN_ALL = _permission(
    ResourceType.ADMIN, Action.ALL, "Full Admin Access", "All administrative capabilities"
)
PERM_ADMIN_AUDIT = _permission(
    ResourceType.ADMIN,
    Action.AUDIT,
    "Admin Audit",
    "Access administrative audit functions",
)
PERM_ADMIN_SECURITY = _permission(
    ResourceType.ADMIN,
    Action.SECURITY,
    "Admin Security",
    "Manage security configurations and policies",
)
PERM_ADMIN_SYSTEM = _permission(
    ResourceType.ADMIN,
    Action.SYSTEM,
    "Admin System",
    "System-wide administrative operations",
)

# ============================================================================
# WEBHOOK PERMISSIONS
# ============================================================================

PERM_WEBHOOK_READ = _permission(
    ResourceType.WEBHOOK, Action.READ, "View Webhooks", "View webhook configurations"
)
PERM_WEBHOOK_CREATE = _permission(
    ResourceType.WEBHOOK, Action.CREATE, "Create Webhooks", "Create new webhooks"
)
PERM_WEBHOOK_DELETE = _permission(
    ResourceType.WEBHOOK, Action.DELETE, "Delete Webhooks", "Remove webhook configurations"
)
PERM_WEBHOOK_ADMIN = _permission(
    ResourceType.WEBHOOK, Action.ALL, "Webhook Admin", "Admin webhook operations (DLQ management)"
)

# ============================================================================
# CHECKPOINT & REPLAY PERMISSIONS
# ============================================================================

PERM_CHECKPOINT_READ = _permission(
    ResourceType.CHECKPOINT, Action.READ, "View Checkpoints", "View saved checkpoints"
)
PERM_CHECKPOINT_CREATE = _permission(
    ResourceType.CHECKPOINT, Action.CREATE, "Create Checkpoints", "Save debate checkpoints"
)
PERM_CHECKPOINT_DELETE = _permission(
    ResourceType.CHECKPOINT, Action.DELETE, "Delete Checkpoints", "Remove saved checkpoints"
)
PERM_REPLAYS_READ = _permission(
    ResourceType.REPLAY, Action.READ, "View Replays", "View debate replay recordings"
)

# ============================================================================
# GAUNTLET PERMISSIONS
# ============================================================================

PERM_GAUNTLET_RUN = _permission(
    ResourceType.GAUNTLET, Action.RUN, "Run Gauntlet", "Execute adversarial stress-tests"
)
PERM_GAUNTLET_READ = _permission(
    ResourceType.GAUNTLET,
    Action.READ,
    "View Gauntlet Results",
    "View gauntlet results and receipts",
)
PERM_GAUNTLET_DELETE = _permission(
    ResourceType.GAUNTLET, Action.DELETE, "Delete Gauntlet Results", "Delete gauntlet runs"
)
PERM_GAUNTLET_SIGN = _permission(
    ResourceType.GAUNTLET, Action.SIGN, "Sign Receipts", "Cryptographically sign decision receipts"
)
PERM_GAUNTLET_COMPARE = _permission(
    ResourceType.GAUNTLET, Action.COMPARE, "Compare Gauntlets", "Compare gauntlet run results"
)
PERM_GAUNTLET_EXPORT = _permission(
    ResourceType.GAUNTLET, Action.EXPORT_DATA, "Export Gauntlet Data", "Export gauntlet reports"
)

# ============================================================================
# MARKETPLACE PERMISSIONS
# ============================================================================

PERM_MARKETPLACE_READ = _permission(
    ResourceType.MARKETPLACE,
    Action.READ,
    "Browse Marketplace",
    "Browse and search marketplace templates",
)
PERM_MARKETPLACE_PUBLISH = _permission(
    ResourceType.MARKETPLACE,
    Action.PUBLISH,
    "Publish Templates",
    "Publish templates to marketplace",
)
PERM_MARKETPLACE_IMPORT = _permission(
    ResourceType.MARKETPLACE, Action.IMPORT, "Import Templates", "Import templates from marketplace"
)
PERM_MARKETPLACE_RATE = _permission(
    ResourceType.MARKETPLACE, Action.RATE, "Rate Templates", "Rate marketplace templates"
)
PERM_MARKETPLACE_REVIEW = _permission(
    ResourceType.MARKETPLACE, Action.REVIEW, "Review Templates", "Write reviews for templates"
)
PERM_MARKETPLACE_DELETE = _permission(
    ResourceType.MARKETPLACE, Action.DELETE, "Delete Templates", "Remove templates from marketplace"
)

# ============================================================================
# EXPLAINABILITY PERMISSIONS
# ============================================================================

PERM_EXPLAINABILITY_READ = _permission(
    ResourceType.EXPLAINABILITY, Action.READ, "View Explanations", "View decision explanations"
)
PERM_EXPLAINABILITY_BATCH = _permission(
    ResourceType.EXPLAINABILITY,
    Action.BATCH,
    "Batch Explanations",
    "Process batch explanation jobs",
)

# ============================================================================
# FINDINGS PERMISSIONS
# ============================================================================

PERM_FINDINGS_READ = _permission(
    ResourceType.FINDINGS, Action.READ, "View Findings", "View audit findings and history"
)
PERM_FINDINGS_UPDATE = _permission(
    ResourceType.FINDINGS, Action.UPDATE, "Update Findings", "Modify finding status and properties"
)
PERM_FINDINGS_ASSIGN = _permission(
    ResourceType.FINDINGS, Action.ASSIGN, "Assign Findings", "Assign findings to users"
)
PERM_FINDINGS_BULK = _permission(
    ResourceType.FINDINGS,
    Action.BULK,
    "Bulk Finding Operations",
    "Perform bulk actions on findings",
)

# ============================================================================
# DECISION PERMISSIONS
# ============================================================================

PERM_DECISION_CREATE = _permission(
    ResourceType.DECISION, Action.CREATE, "Create Decisions", "Submit decisions via unified router"
)
PERM_DECISION_READ = _permission(
    ResourceType.DECISION, Action.READ, "View Decisions", "View decision results and status"
)
PERM_DECISION_UPDATE = _permission(
    ResourceType.DECISION,
    Action.UPDATE,
    "Update Decisions",
    "Cancel or retry decision requests",
)

# ============================================================================
# POLICY PERMISSIONS
# ============================================================================

PERM_POLICY_READ = _permission(
    ResourceType.POLICY, Action.READ, "View Policies", "View governance policies"
)
PERM_POLICY_CREATE = _permission(
    ResourceType.POLICY, Action.CREATE, "Create Policies", "Create new governance policies"
)
PERM_POLICY_UPDATE = _permission(
    ResourceType.POLICY, Action.UPDATE, "Update Policies", "Modify governance policies"
)
PERM_POLICY_DELETE = _permission(
    ResourceType.POLICY, Action.DELETE, "Delete Policies", "Remove governance policies"
)

# ============================================================================
# COMPLIANCE PERMISSIONS
# ============================================================================

PERM_COMPLIANCE_READ = _permission(
    ResourceType.COMPLIANCE, Action.READ, "View Compliance", "View compliance status and violations"
)
PERM_COMPLIANCE_UPDATE = _permission(
    ResourceType.COMPLIANCE, Action.UPDATE, "Update Compliance", "Update violation status"
)
PERM_COMPLIANCE_CHECK = _permission(
    ResourceType.COMPLIANCE, Action.CHECK, "Run Compliance Checks", "Execute compliance validation"
)
PERM_COMPLIANCE_GDPR = _permission(
    ResourceType.COMPLIANCE,
    Action.GDPR,
    "GDPR Operations",
    "Perform GDPR compliance operations (data export, deletion)",
)
PERM_COMPLIANCE_SOC2 = _permission(
    ResourceType.COMPLIANCE,
    Action.SOC2,
    "SOC2 Operations",
    "Access SOC2 compliance reports and controls",
)
PERM_COMPLIANCE_LEGAL = _permission(
    ResourceType.COMPLIANCE,
    Action.LEGAL,
    "Legal Operations",
    "Manage legal holds and compliance requirements",
)
PERM_COMPLIANCE_AUDIT = _permission(
    ResourceType.COMPLIANCE,
    Action.AUDIT,
    "Audit Operations",
    "Perform compliance audit verification",
)

# ============================================================================
# CONTROL PLANE PERMISSIONS
# ============================================================================

PERM_CONTROL_PLANE_READ = _permission(
    ResourceType.CONTROL_PLANE, Action.READ, "View Control Plane", "View tasks, agents, and status"
)
PERM_CONTROL_PLANE_SUBMIT = _permission(
    ResourceType.CONTROL_PLANE, Action.SUBMIT, "Submit Tasks", "Submit tasks to the control plane"
)
PERM_CONTROL_PLANE_CANCEL = _permission(
    ResourceType.CONTROL_PLANE, Action.CANCEL, "Cancel Tasks", "Cancel pending control plane tasks"
)
PERM_CONTROL_PLANE_DELIBERATE = _permission(
    ResourceType.CONTROL_PLANE,
    Action.DELIBERATE,
    "Start Deliberations",
    "Start multi-agent deliberation processes",
)
PERM_CONTROL_PLANE_AGENTS = _permission(
    ResourceType.CONTROL_PLANE,
    Action.ADMIN_OP,
    "Manage Agents",
    "Full control over agent registry",
)
PERM_CONTROL_PLANE_AGENTS_READ = _permission(
    ResourceType.CONTROL_PLANE,
    Action.AGENTS_READ,
    "View Agents",
    "Read agent registry information",
)
PERM_CONTROL_PLANE_AGENTS_REGISTER = _permission(
    ResourceType.CONTROL_PLANE,
    Action.AGENTS_REGISTER,
    "Register Agents",
    "Register new agents in the control plane",
)
PERM_CONTROL_PLANE_AGENTS_UNREGISTER = _permission(
    ResourceType.CONTROL_PLANE,
    Action.AGENTS_UNREGISTER,
    "Unregister Agents",
    "Remove agents from the control plane",
)
PERM_CONTROL_PLANE_TASKS = _permission(
    ResourceType.CONTROL_PLANE,
    Action.MANAGE,
    "Manage Tasks",
    "Full control over task queue",
)
PERM_CONTROL_PLANE_TASKS_READ = _permission(
    ResourceType.CONTROL_PLANE,
    Action.TASKS_READ,
    "View Tasks",
    "Read task queue information",
)
PERM_CONTROL_PLANE_TASKS_SUBMIT = _permission(
    ResourceType.CONTROL_PLANE,
    Action.TASKS_SUBMIT,
    "Submit Tasks",
    "Submit new tasks to the control plane",
)
PERM_CONTROL_PLANE_TASKS_CLAIM = _permission(
    ResourceType.CONTROL_PLANE,
    Action.TASKS_CLAIM,
    "Claim Tasks",
    "Claim tasks for processing",
)
PERM_CONTROL_PLANE_TASKS_COMPLETE = _permission(
    ResourceType.CONTROL_PLANE,
    Action.TASKS_COMPLETE,
    "Complete Tasks",
    "Mark tasks as completed",
)
PERM_CONTROL_PLANE_HEALTH_READ = _permission(
    ResourceType.CONTROL_PLANE,
    Action.HEALTH_READ,
    "View Health",
    "Read control plane health status",
)

# ============================================================================
# FINANCE PERMISSIONS
# ============================================================================

PERM_FINANCE_READ = _permission(
    ResourceType.FINANCE,
    Action.READ,
    "View Finance",
    "View financial data, invoices, and transactions",
)
PERM_FINANCE_WRITE = _permission(
    ResourceType.FINANCE,
    Action.WRITE,
    "Manage Finance",
    "Create and modify financial records",
)
PERM_FINANCE_APPROVE = _permission(
    ResourceType.FINANCE,
    Action.APPROVE,
    "Approve Finance",
    "Approve financial transactions and invoices",
)

# ============================================================================
# RECEIPT PERMISSIONS
# ============================================================================

PERM_RECEIPT_READ = _permission(
    ResourceType.RECEIPT,
    Action.READ,
    "View Receipts",
    "View decision receipts and audit trails",
)
PERM_RECEIPT_VERIFY = _permission(
    ResourceType.RECEIPT,
    Action.VERIFY,
    "Verify Receipts",
    "Verify integrity of decision receipts",
)
PERM_RECEIPT_EXPORT = _permission(
    ResourceType.RECEIPT,
    Action.EXPORT_DATA,
    "Export Receipts",
    "Export receipts for compliance reporting",
)
PERM_RECEIPT_SEND = _permission(
    ResourceType.RECEIPT,
    Action.SEND,
    "Send Receipts",
    "Send receipts to stakeholders",
)

# ============================================================================
# SCHEDULER PERMISSIONS
# ============================================================================

PERM_SCHEDULER_READ = _permission(
    ResourceType.SCHEDULER,
    Action.READ,
    "View Scheduler",
    "View scheduled jobs and their status",
)
PERM_SCHEDULER_CREATE = _permission(
    ResourceType.SCHEDULER,
    Action.CREATE,
    "Create Schedules",
    "Create new scheduled jobs",
)
PERM_SCHEDULER_EXECUTE = _permission(
    ResourceType.SCHEDULER,
    Action.EXECUTE,
    "Execute Schedules",
    "Manually trigger scheduled jobs",
)

# ============================================================================
# COST PERMISSIONS
# ============================================================================

PERM_COST_READ = _permission(
    ResourceType.COST,
    Action.READ,
    "View Costs",
    "View cost dashboards and optimization recommendations",
)
PERM_COST_WRITE = _permission(
    ResourceType.COST,
    Action.WRITE,
    "Manage Costs",
    "Modify cost allocations and budgets",
)

# ============================================================================
# QUEUE PERMISSIONS
# ============================================================================

PERM_QUEUE_READ = _permission(
    ResourceType.QUEUE, Action.READ, "View Queue", "View job queue status and messages"
)
PERM_QUEUE_MANAGE = _permission(
    ResourceType.QUEUE, Action.MANAGE, "Manage Queue", "Submit, retry, and cancel queue jobs"
)
PERM_QUEUE_ADMIN = _permission(
    ResourceType.QUEUE,
    Action.ADMIN_OP,
    "Administer Queue",
    "Full queue administration including DLQ",
)

# ============================================================================
# NOMIC PERMISSIONS
# ============================================================================

PERM_NOMIC_READ = _permission(
    ResourceType.NOMIC, Action.READ, "View Nomic", "View Nomic loop progress and results"
)
PERM_NOMIC_ADMIN = _permission(
    ResourceType.NOMIC,
    Action.ADMIN_OP,
    "Administer Nomic",
    "Control Nomic self-improvement operations",
)

# ============================================================================
# ORCHESTRATION PERMISSIONS
# ============================================================================

PERM_ORCHESTRATION_READ = _permission(
    ResourceType.ORCHESTRATION,
    Action.READ,
    "View Orchestration",
    "View orchestration templates and status",
)
PERM_ORCHESTRATION_EXECUTE = _permission(
    ResourceType.ORCHESTRATION,
    Action.EXECUTE,
    "Execute Orchestration",
    "Run multi-agent deliberations",
)

# ============================================================================
# SYSTEM PERMISSIONS
# ============================================================================

PERM_SYSTEM_HEALTH_READ = _permission(
    ResourceType.SYSTEM, Action.READ, "View System Health", "View system health and diagnostics"
)

# ============================================================================
# VERTICALS PERMISSIONS
# ============================================================================

PERM_VERTICALS_READ = _permission(
    ResourceType.VERTICALS, Action.READ, "View Verticals", "View domain specialists"
)
PERM_VERTICALS_WRITE = _permission(
    ResourceType.VERTICALS, Action.UPDATE, "Update Verticals", "Configure domain specialists"
)

# ============================================================================
# CANVAS PERMISSIONS
# ============================================================================

PERM_CANVAS_READ = _permission(
    ResourceType.CANVAS, Action.READ, "View Canvas", "View canvas documents and state"
)
PERM_CANVAS_CREATE = _permission(
    ResourceType.CANVAS, Action.CREATE, "Create Canvas", "Create new canvas documents"
)
PERM_CANVAS_UPDATE = _permission(
    ResourceType.CANVAS, Action.UPDATE, "Update Canvas", "Modify canvas content and state"
)
PERM_CANVAS_DELETE = _permission(
    ResourceType.CANVAS, Action.DELETE, "Delete Canvas", "Delete canvas documents"
)
PERM_CANVAS_RUN = _permission(
    ResourceType.CANVAS, Action.RUN, "Run Canvas", "Execute canvas operations"
)

# ============================================================================
# VERIFICATION PERMISSIONS
# ============================================================================

PERM_VERIFICATION_READ = _permission(
    ResourceType.VERIFICATION, Action.READ, "View Verification", "View verification results"
)
PERM_VERIFICATION_CREATE = _permission(
    ResourceType.VERIFICATION, Action.CREATE, "Create Verification", "Run formal verification"
)

# ============================================================================
# CODEBASE PERMISSIONS
# ============================================================================

PERM_CODEBASE_READ = _permission(
    ResourceType.CODEBASE, Action.READ, "View Codebase Analysis", "View codebase analysis results"
)
PERM_CODEBASE_ANALYZE = _permission(
    ResourceType.CODEBASE, Action.RUN, "Analyze Codebase", "Run codebase analysis operations"
)

# ============================================================================
# DATA GOVERNANCE PERMISSIONS
# ============================================================================

PERM_DATA_CLASSIFICATION_READ = _permission(
    ResourceType.DATA_CLASSIFICATION,
    Action.READ,
    "View Data Classifications",
    "View data sensitivity classifications",
)
PERM_DATA_CLASSIFICATION_CLASSIFY = _permission(
    ResourceType.DATA_CLASSIFICATION,
    Action.CLASSIFY,
    "Classify Data",
    "Mark data as confidential/public/internal",
)
PERM_DATA_CLASSIFICATION_UPDATE = _permission(
    ResourceType.DATA_CLASSIFICATION,
    Action.UPDATE,
    "Update Classifications",
    "Modify existing data classifications",
)
PERM_DATA_RETENTION_READ = _permission(
    ResourceType.DATA_RETENTION,
    Action.READ,
    "View Retention Policies",
    "View data retention policies",
)
PERM_DATA_RETENTION_UPDATE = _permission(
    ResourceType.DATA_RETENTION,
    Action.UPDATE,
    "Configure Retention",
    "Set and enforce retention policies",
)
PERM_DATA_LINEAGE_READ = _permission(
    ResourceType.DATA_LINEAGE,
    Action.READ,
    "View Data Lineage",
    "Track data provenance and transformations",
)
PERM_PII_READ = _permission(
    ResourceType.PII, Action.READ, "View PII", "View personally identifiable information"
)
PERM_PII_REDACT = _permission(
    ResourceType.PII, Action.REDACT, "Redact PII", "Redact personally identifiable information"
)
PERM_PII_MASK = _permission(
    ResourceType.PII, Action.MASK, "Mask PII", "Configure PII masking rules"
)

# ============================================================================
# COMPUTER-USE PERMISSIONS
# ============================================================================

PERM_COMPUTER_USE_READ = _permission(
    ResourceType.COMPUTER_USE,
    Action.READ,
    "View Computer-Use Sessions",
    "View computer-use task status and history",
)
PERM_COMPUTER_USE_EXECUTE = _permission(
    ResourceType.COMPUTER_USE,
    Action.EXECUTE,
    "Execute Computer-Use Tasks",
    "Run computer-use automation tasks",
)
PERM_COMPUTER_USE_BROWSER = _permission(
    ResourceType.COMPUTER_USE,
    Action.BROWSER,
    "Browser Automation",
    "Control browser (navigate, click, type)",
)
PERM_COMPUTER_USE_SHELL = _permission(
    ResourceType.COMPUTER_USE,
    Action.SHELL,
    "Shell Execution",
    "Execute shell commands (bash, powershell)",
)
PERM_COMPUTER_USE_FILE_READ = _permission(
    ResourceType.COMPUTER_USE,
    Action.FILE_READ,
    "Read Files",
    "Read files from the filesystem",
)
PERM_COMPUTER_USE_FILE_WRITE = _permission(
    ResourceType.COMPUTER_USE,
    Action.FILE_WRITE,
    "Write Files",
    "Write files to the filesystem",
)
PERM_COMPUTER_USE_SCREENSHOT = _permission(
    ResourceType.COMPUTER_USE,
    Action.SCREENSHOT,
    "Take Screenshots",
    "Capture screen contents",
)
PERM_COMPUTER_USE_NETWORK = _permission(
    ResourceType.COMPUTER_USE,
    Action.NETWORK,
    "Network Access",
    "Make network requests (HTTP, etc.)",
)
PERM_COMPUTER_USE_ADMIN = _permission(
    ResourceType.COMPUTER_USE,
    Action.ADMIN_OP,
    "Computer-Use Admin",
    "Full computer-use administration (policy management, override limits)",
)

# ============================================================================
# COMPLIANCE POLICY PERMISSIONS
# ============================================================================

PERM_COMPLIANCE_POLICY_READ = _permission(
    ResourceType.COMPLIANCE_POLICY,
    Action.READ,
    "View Compliance Policies",
    "Access compliance rules (SOC2, GDPR, HIPAA)",
)
PERM_COMPLIANCE_POLICY_UPDATE = _permission(
    ResourceType.COMPLIANCE_POLICY,
    Action.UPDATE,
    "Update Compliance Policies",
    "Modify compliance rules",
)
PERM_COMPLIANCE_POLICY_ENFORCE = _permission(
    ResourceType.COMPLIANCE_POLICY,
    Action.ENFORCE,
    "Enforce Compliance",
    "Force resolution of compliance findings",
)

# ============================================================================
# AUDIT LOG PERMISSIONS
# ============================================================================

PERM_AUDIT_LOG_READ = _permission(
    ResourceType.AUDIT_LOG, Action.READ, "View Audit Logs", "Access audit trail"
)
PERM_AUDIT_LOG_EXPORT = _permission(
    ResourceType.AUDIT_LOG,
    Action.EXPORT_DATA,
    "Export Audit Logs",
    "Export audit logs for compliance",
)
PERM_AUDIT_LOG_SEARCH = _permission(
    ResourceType.AUDIT_LOG,
    Action.SEARCH,
    "Search Audit Logs",
    "Advanced search in audit logs",
)
PERM_AUDIT_LOG_STREAM = _permission(
    ResourceType.AUDIT_LOG,
    Action.STREAM,
    "Stream Audit Logs",
    "Stream logs to external SIEM",
)
PERM_AUDIT_LOG_CONFIGURE = _permission(
    ResourceType.AUDIT_LOG,
    Action.UPDATE,
    "Configure Audit Retention",
    "Set audit log retention policies",
)
PERM_AUDIT_LOG_DELETE = _permission(
    ResourceType.AUDIT_LOG,
    Action.DELETE,
    "Delete Audit Logs",
    "Permanently delete audit trail records (compliance-critical)",
)

# ============================================================================
# VENDOR PERMISSIONS
# ============================================================================

PERM_VENDOR_READ = _permission(
    ResourceType.VENDOR, Action.READ, "View Vendors", "View third-party vendor list"
)
PERM_VENDOR_APPROVE = _permission(
    ResourceType.VENDOR, Action.GRANT, "Approve Vendors", "Approve third-party integrations"
)

# ============================================================================
# TEAM PERMISSIONS
# ============================================================================

PERM_TEAM_CREATE = _permission(ResourceType.TEAM, Action.CREATE, "Create Teams", "Create new teams")
PERM_TEAM_READ = _permission(
    ResourceType.TEAM, Action.READ, "View Teams", "View team membership and details"
)
PERM_TEAM_UPDATE = _permission(
    ResourceType.TEAM, Action.UPDATE, "Update Teams", "Modify team settings"
)
PERM_TEAM_DELETE = _permission(ResourceType.TEAM, Action.DELETE, "Delete Teams", "Remove teams")
PERM_TEAM_ADD_MEMBER = _permission(
    ResourceType.TEAM, Action.ADD_MEMBER, "Add Team Members", "Invite users to teams"
)
PERM_TEAM_REMOVE_MEMBER = _permission(
    ResourceType.TEAM,
    Action.REMOVE_MEMBER,
    "Remove Team Members",
    "Remove users from teams",
)
PERM_TEAM_SHARE = _permission(
    ResourceType.TEAM, Action.SHARE, "Share with Team", "Grant team access to resources"
)
PERM_TEAM_DISSOLVE = _permission(
    ResourceType.TEAM,
    Action.DISSOLVE,
    "Dissolve Teams",
    "Dissolve teams with resource reallocation",
)

# ============================================================================
# WORKSPACE PERMISSIONS
# ============================================================================

PERM_WORKSPACE_CREATE = _permission(
    ResourceType.WORKSPACE, Action.CREATE, "Create Workspaces", "Create new workspaces"
)
PERM_WORKSPACE_READ = _permission(
    ResourceType.WORKSPACE, Action.READ, "View Workspaces", "View workspace details and settings"
)
PERM_WORKSPACE_UPDATE = _permission(
    ResourceType.WORKSPACE, Action.UPDATE, "Update Workspaces", "Modify workspace settings"
)
PERM_WORKSPACE_DELETE = _permission(
    ResourceType.WORKSPACE, Action.DELETE, "Delete Workspaces", "Remove workspaces"
)
PERM_WORKSPACE_MEMBER_ADD = _permission(
    ResourceType.WORKSPACE_MEMBER,
    Action.ADD_MEMBER,
    "Add Workspace Members",
    "Invite users to workspaces",
)
PERM_WORKSPACE_MEMBER_REMOVE = _permission(
    ResourceType.WORKSPACE_MEMBER,
    Action.REMOVE_MEMBER,
    "Remove Workspace Members",
    "Remove users from workspaces",
)
PERM_WORKSPACE_MEMBER_CHANGE_ROLE = _permission(
    ResourceType.WORKSPACE_MEMBER,
    Action.CHANGE_ROLE,
    "Change Member Roles",
    "Modify member roles within workspaces",
)
PERM_WORKSPACE_SHARE = _permission(
    ResourceType.WORKSPACE,
    Action.SHARE,
    "Share with Workspace",
    "Grant workspace access to resources",
)

# ============================================================================
# QUOTA & BUDGET PERMISSIONS
# ============================================================================

PERM_QUOTA_READ = _permission(
    ResourceType.QUOTA, Action.READ, "View Quotas", "View rate limits and quotas"
)
PERM_QUOTA_UPDATE = _permission(
    ResourceType.QUOTA, Action.SET_LIMIT, "Set Quotas", "Configure rate limits per user/org"
)
PERM_QUOTA_OVERRIDE = _permission(
    ResourceType.QUOTA,
    Action.OVERRIDE,
    "Override Quotas",
    "Emergency quota overrides",
)
PERM_COST_CENTER_READ = _permission(
    ResourceType.COST_CENTER, Action.READ, "View Cost Centers", "View cost center assignments"
)
PERM_COST_CENTER_UPDATE = _permission(
    ResourceType.COST_CENTER,
    Action.CHARGEBACK,
    "Manage Cost Centers",
    "Link resources to cost centers for chargeback",
)
PERM_BUDGET_READ = _permission(
    ResourceType.BUDGET, Action.READ, "View Budgets", "View budget limits and usage"
)
PERM_BUDGET_UPDATE = _permission(
    ResourceType.BUDGET, Action.SET_LIMIT, "Set Budgets", "Configure spending limits and alerts"
)
PERM_BUDGET_OVERRIDE = _permission(
    ResourceType.BUDGET,
    Action.OVERRIDE,
    "Override Budget",
    "Emergency budget increases",
)

# ============================================================================
# BILLING PERMISSIONS
# ============================================================================

PERM_BILLING_READ = _permission(
    ResourceType.BILLING,
    Action.READ,
    "View Cost Data",
    "View cost analytics and efficiency metrics",
)
PERM_BILLING_RECOMMENDATIONS_READ = _permission(
    ResourceType.BILLING,
    Action.READ,
    "View Cost Recommendations",
    "View cost optimization recommendations",
)
PERM_BILLING_RECOMMENDATIONS_APPLY = _permission(
    ResourceType.BILLING,
    Action.UPDATE,
    "Apply Cost Recommendations",
    "Apply and dismiss cost optimization recommendations",
)
PERM_BILLING_FORECAST_READ = _permission(
    ResourceType.BILLING,
    Action.READ,
    "View Cost Forecasts",
    "View cost forecasts and projections",
)
PERM_BILLING_FORECAST_SIMULATE = _permission(
    ResourceType.BILLING,
    Action.UPDATE,
    "Simulate Cost Scenarios",
    "Run what-if cost simulations",
)
PERM_BILLING_EXPORT_HISTORY = _permission(
    ResourceType.BILLING,
    Action.EXPORT_HISTORY,
    "Export Billing History",
    "Export historical billing data",
)

# ============================================================================
# SESSION & AUTH PERMISSIONS
# ============================================================================

PERM_SESSION_READ = _permission(
    ResourceType.SESSION, Action.LIST_ACTIVE, "View Active Sessions", "List active user sessions"
)
PERM_SESSION_REVOKE = _permission(
    ResourceType.SESSION, Action.REVOKE, "Revoke Sessions", "Force logout of user sessions"
)
PERM_SESSION_CREATE = _permission(
    ResourceType.SESSION,
    Action.CREATE,
    "Create Sessions",
    "Create sessions on behalf of users",
)
PERM_AUTH_RESET_PASSWORD = _permission(
    ResourceType.AUTHENTICATION,
    Action.RESET_PASSWORD,
    "Reset Passwords",
    "Reset user passwords",
)
PERM_AUTH_REQUIRE_MFA = _permission(
    ResourceType.AUTHENTICATION,
    Action.REQUIRE_MFA,
    "Require MFA",
    "Enforce MFA for users or operations",
)
PERM_AUTH_READ = _permission(
    ResourceType.AUTHENTICATION,
    Action.READ,
    "View Auth Info",
    "View own authentication info and settings",
)
PERM_AUTH_CREATE = _permission(
    ResourceType.AUTHENTICATION,
    Action.CREATE,
    "Setup Auth",
    "Setup authentication methods (MFA)",
)
PERM_AUTH_UPDATE = _permission(
    ResourceType.AUTHENTICATION,
    Action.UPDATE,
    "Update Auth",
    "Update authentication settings (enable/disable MFA, link accounts)",
)
PERM_AUTH_REVOKE = _permission(
    ResourceType.AUTHENTICATION,
    Action.REVOKE,
    "Revoke Auth",
    "Revoke sessions and tokens (logout)",
)
PERM_API_KEY_CREATE = _permission(
    ResourceType.API_KEY,
    Action.CREATE,
    "Create API Key",
    "Generate personal API keys",
)
PERM_API_KEY_REVOKE = _permission(
    ResourceType.API_KEY,
    Action.REVOKE,
    "Revoke API Key",
    "Revoke personal API keys",
)
PERM_API_KEY_LIST_ALL = _permission(
    ResourceType.API_KEY,
    Action.LIST_ALL,
    "List All API Keys",
    "View all API keys in organization",
)
PERM_API_KEY_EXPORT_SECRET = _permission(
    ResourceType.API_KEY,
    Action.EXPORT_SECRET,
    "Export API Key Secrets",
    "Export API key secrets after creation",
)

# ============================================================================
# APPROVAL PERMISSIONS
# ============================================================================

PERM_APPROVAL_REQUEST = _permission(
    ResourceType.APPROVAL, Action.REQUEST, "Request Access", "Request elevated access or approvals"
)
PERM_APPROVAL_GRANT = _permission(
    ResourceType.APPROVAL,
    Action.GRANT,
    "Grant Approvals",
    "Approve access requests",
)
PERM_APPROVAL_READ = _permission(
    ResourceType.APPROVAL,
    Action.READ,
    "View Approval History",
    "View past approval decisions",
)

# ============================================================================
# TEMPLATE PERMISSIONS
# ============================================================================

PERM_TEMPLATE_CREATE = _permission(
    ResourceType.TEMPLATE,
    Action.CREATE,
    "Create Templates",
    "Create workflow templates",
)
PERM_TEMPLATE_READ = _permission(
    ResourceType.TEMPLATE,
    Action.READ,
    "Read Templates",
    "View workflow templates",
)
PERM_TEMPLATE_UPDATE = _permission(
    ResourceType.TEMPLATE,
    Action.UPDATE,
    "Update Templates",
    "Modify workflow templates",
)
PERM_TEMPLATE_DELETE = _permission(
    ResourceType.TEMPLATE,
    Action.DELETE,
    "Delete Templates",
    "Permanently delete workflow templates",
)

# ============================================================================
# ROLE PERMISSIONS
# ============================================================================

PERM_ROLE_CREATE = _permission(
    ResourceType.ROLE,
    Action.CREATE,
    "Create Roles",
    "Create custom roles",
)
PERM_ROLE_READ = _permission(
    ResourceType.ROLE,
    Action.READ,
    "Read Roles",
    "View custom role definitions",
)
PERM_ROLE_UPDATE = _permission(
    ResourceType.ROLE,
    Action.UPDATE,
    "Update Roles",
    "Modify custom role permissions",
)
PERM_ROLE_DELETE = _permission(
    ResourceType.ROLE,
    Action.DELETE,
    "Delete Roles",
    "Permanently delete custom roles",
)

# ============================================================================
# BACKUP & DR PERMISSIONS
# ============================================================================

PERM_BACKUP_CREATE = _permission(
    ResourceType.BACKUP,
    Action.CREATE,
    "Create Backups",
    "Create system backups",
)
PERM_BACKUP_READ = _permission(
    ResourceType.BACKUP,
    Action.READ,
    "Read Backups",
    "View backup status and metadata",
)
PERM_BACKUP_RESTORE = _permission(
    ResourceType.BACKUP,
    Action.RESTORE,
    "Restore Backups",
    "Restore system from backup (irreversible)",
)
PERM_BACKUP_DELETE = _permission(
    ResourceType.BACKUP,
    Action.DELETE,
    "Delete Backups",
    "Remove backup archives",
)
PERM_DR_READ = _permission(
    ResourceType.DISASTER_RECOVERY,
    Action.READ,
    "Read DR Status",
    "View disaster recovery configuration and status",
)
PERM_DR_EXECUTE = _permission(
    ResourceType.DISASTER_RECOVERY,
    Action.EXECUTE,
    "Execute DR Procedures",
    "Execute disaster recovery procedures",
)
