"""
RBAC Default Roles and Permissions.

Defines the system-wide roles and permissions available in Aragora.
Organizations can create custom roles based on these templates.
"""

from __future__ import annotations


from .models import Action, Permission, ResourceType, Role

# ============================================================================
# SYSTEM PERMISSIONS
# ============================================================================


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


# Debate permissions
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

# Agent permissions
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

# User management permissions
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

# Organization permissions
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

# API permissions
PERM_API_GENERATE_KEY = _permission(
    ResourceType.API, Action.GENERATE_KEY, "Generate API Keys", "Create new API keys"
)
PERM_API_REVOKE_KEY = _permission(
    ResourceType.API, Action.REVOKE_KEY, "Revoke API Keys", "Revoke existing API keys"
)

# Memory permissions
PERM_MEMORY_READ = _permission(
    ResourceType.MEMORY, Action.READ, "View Memory", "View memory contents and analytics"
)
PERM_MEMORY_UPDATE = _permission(
    ResourceType.MEMORY, Action.UPDATE, "Update Memory", "Modify memory contents"
)
PERM_MEMORY_DELETE = _permission(
    ResourceType.MEMORY, Action.DELETE, "Delete Memory", "Clear memory contents"
)

# Workflow permissions
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

# Analytics permissions
PERM_ANALYTICS_READ = _permission(
    ResourceType.ANALYTICS, Action.READ, "View Analytics", "Access analytics dashboards"
)
PERM_ANALYTICS_EXPORT = _permission(
    ResourceType.ANALYTICS, Action.EXPORT_DATA, "Export Analytics", "Export analytics data"
)

# Introspection permissions
PERM_INTROSPECTION_READ = _permission(
    ResourceType.INTROSPECTION,
    Action.READ,
    "View Introspection",
    "Access system introspection and agent status",
)

# Reasoning permissions
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

# Knowledge permissions
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

# Inbox permissions
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

# Training permissions
PERM_TRAINING_READ = _permission(
    ResourceType.TRAINING, Action.READ, "View Training Data", "Access training data exports"
)
PERM_TRAINING_CREATE = _permission(
    ResourceType.TRAINING,
    Action.CREATE,
    "Create Training Exports",
    "Generate training data exports",
)

# Evidence permissions
PERM_EVIDENCE_READ = _permission(
    ResourceType.EVIDENCE, Action.READ, "View Evidence", "Access evidence and citations"
)
PERM_EVIDENCE_CREATE = _permission(
    ResourceType.EVIDENCE, Action.CREATE, "Add Evidence", "Add new evidence sources"
)

# Document permissions
PERM_DOCUMENTS_READ = _permission(
    ResourceType.DOCUMENTS, Action.READ, "View Documents", "Access document metadata and queries"
)
PERM_DOCUMENTS_CREATE = _permission(
    ResourceType.DOCUMENTS, Action.CREATE, "Upload Documents", "Upload and process documents"
)

# Upload permissions
PERM_UPLOAD_CREATE = _permission(
    ResourceType.UPLOAD, Action.CREATE, "Create Uploads", "Create document folder uploads"
)

# Speech permissions
PERM_SPEECH_CREATE = _permission(
    ResourceType.SPEECH, Action.CREATE, "Create Speech Jobs", "Generate speech transcripts"
)

# Connector permissions
PERM_CONNECTOR_READ = _permission(
    ResourceType.CONNECTOR, Action.READ, "View Connectors", "View connector configurations"
)
PERM_CONNECTOR_CREATE = _permission(
    ResourceType.CONNECTOR, Action.CREATE, "Create Connectors", "Configure new data connectors"
)
PERM_CONNECTOR_DELETE = _permission(
    ResourceType.CONNECTOR, Action.DELETE, "Delete Connectors", "Remove connector configurations"
)

# Repository permissions
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

# Message bindings permissions
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

# Admin permissions
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

# Webhook permissions
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

# Checkpoint permissions
PERM_CHECKPOINT_READ = _permission(
    ResourceType.CHECKPOINT, Action.READ, "View Checkpoints", "View saved checkpoints"
)
PERM_CHECKPOINT_CREATE = _permission(
    ResourceType.CHECKPOINT, Action.CREATE, "Create Checkpoints", "Save debate checkpoints"
)
PERM_CHECKPOINT_DELETE = _permission(
    ResourceType.CHECKPOINT, Action.DELETE, "Delete Checkpoints", "Remove saved checkpoints"
)

# Replay permissions
PERM_REPLAYS_READ = _permission(
    ResourceType.REPLAY, Action.READ, "View Replays", "View debate replay recordings"
)

# Gauntlet permissions (adversarial stress-testing)
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

# Marketplace permissions
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

# Explainability permissions
PERM_EXPLAINABILITY_READ = _permission(
    ResourceType.EXPLAINABILITY, Action.READ, "View Explanations", "View decision explanations"
)
PERM_EXPLAINABILITY_BATCH = _permission(
    ResourceType.EXPLAINABILITY,
    Action.BATCH,
    "Batch Explanations",
    "Process batch explanation jobs",
)

# Findings permissions (audit findings management)
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

# Decision permissions (unified decision routing)
PERM_DECISION_CREATE = _permission(
    ResourceType.DECISION, Action.CREATE, "Create Decisions", "Submit decisions via unified router"
)
PERM_DECISION_READ = _permission(
    ResourceType.DECISION, Action.READ, "View Decisions", "View decision results and status"
)

# Policy permissions (governance)
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

# Compliance permissions
PERM_COMPLIANCE_READ = _permission(
    ResourceType.COMPLIANCE, Action.READ, "View Compliance", "View compliance status and violations"
)
PERM_COMPLIANCE_UPDATE = _permission(
    ResourceType.COMPLIANCE, Action.UPDATE, "Update Compliance", "Update violation status"
)
PERM_COMPLIANCE_CHECK = _permission(
    ResourceType.COMPLIANCE, Action.CHECK, "Run Compliance Checks", "Execute compliance validation"
)

# Control plane permissions
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


# ============================================================================
# SYSTEM OPERATIONS PERMISSIONS
# ============================================================================

# Queue management permissions
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

# Nomic self-improvement loop permissions
PERM_NOMIC_READ = _permission(
    ResourceType.NOMIC, Action.READ, "View Nomic", "View Nomic loop progress and results"
)
PERM_NOMIC_ADMIN = _permission(
    ResourceType.NOMIC,
    Action.ADMIN_OP,
    "Administer Nomic",
    "Control Nomic self-improvement operations",
)

# Orchestration permissions
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

# System health permissions
PERM_SYSTEM_HEALTH_READ = _permission(
    ResourceType.SYSTEM, Action.READ, "View System Health", "View system health and diagnostics"
)

# Verticals permissions (domain specialists)
PERM_VERTICALS_READ = _permission(
    ResourceType.VERTICALS, Action.READ, "View Verticals", "View domain specialists"
)
PERM_VERTICALS_WRITE = _permission(
    ResourceType.VERTICALS, Action.UPDATE, "Update Verticals", "Configure domain specialists"
)

# Canvas permissions (visual canvas operations)
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

# Verification permissions (formal verification operations)
PERM_VERIFICATION_READ = _permission(
    ResourceType.VERIFICATION, Action.READ, "View Verification", "View verification results"
)
PERM_VERIFICATION_CREATE = _permission(
    ResourceType.VERIFICATION, Action.CREATE, "Create Verification", "Run formal verification"
)

# Codebase analysis permissions
PERM_CODEBASE_READ = _permission(
    ResourceType.CODEBASE, Action.READ, "View Codebase Analysis", "View codebase analysis results"
)
PERM_CODEBASE_ANALYZE = _permission(
    ResourceType.CODEBASE, Action.RUN, "Analyze Codebase", "Run codebase analysis operations"
)


# ============================================================================
# ENTERPRISE PERMISSIONS - Data Governance
# ============================================================================

# Data classification permissions
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

# Data retention permissions
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

# Data lineage permissions
PERM_DATA_LINEAGE_READ = _permission(
    ResourceType.DATA_LINEAGE,
    Action.READ,
    "View Data Lineage",
    "Track data provenance and transformations",
)

# PII permissions
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
# ENTERPRISE PERMISSIONS - Compliance & Regulatory
# ============================================================================

# Compliance policy permissions
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

# Audit log permissions
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

# Vendor management permissions
PERM_VENDOR_READ = _permission(
    ResourceType.VENDOR, Action.READ, "View Vendors", "View third-party vendor list"
)
PERM_VENDOR_APPROVE = _permission(
    ResourceType.VENDOR, Action.GRANT, "Approve Vendors", "Approve third-party integrations"
)

# ============================================================================
# ENTERPRISE PERMISSIONS - Team Management
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

# ============================================================================
# WORKSPACE PERMISSIONS - SME RBAC-lite
# ============================================================================

# Workspace CRUD operations
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

# Workspace member management
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

# Workspace resource access
PERM_WORKSPACE_SHARE = _permission(
    ResourceType.WORKSPACE,
    Action.SHARE,
    "Share with Workspace",
    "Grant workspace access to resources",
)

# ============================================================================
# ENTERPRISE PERMISSIONS - Cost & Quota Management
# ============================================================================

# Quota permissions
PERM_QUOTA_READ = _permission(
    ResourceType.QUOTA, Action.READ, "View Quotas", "View rate limits and quotas"
)
PERM_QUOTA_UPDATE = _permission(
    ResourceType.QUOTA, Action.SET_LIMIT, "Set Quotas", "Configure rate limits per user/org"
)

# Cost center permissions
PERM_COST_CENTER_READ = _permission(
    ResourceType.COST_CENTER, Action.READ, "View Cost Centers", "View cost center assignments"
)
PERM_COST_CENTER_UPDATE = _permission(
    ResourceType.COST_CENTER,
    Action.CHARGEBACK,
    "Manage Cost Centers",
    "Link resources to cost centers for chargeback",
)

# Budget permissions
PERM_BUDGET_READ = _permission(
    ResourceType.BUDGET, Action.READ, "View Budgets", "View budget limits and usage"
)
PERM_BUDGET_UPDATE = _permission(
    ResourceType.BUDGET, Action.SET_LIMIT, "Set Budgets", "Configure spending limits and alerts"
)

# Cost optimization permissions (billing resource)
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

# ============================================================================
# ENTERPRISE PERMISSIONS - Session & Authentication
# ============================================================================

PERM_SESSION_READ = _permission(
    ResourceType.SESSION, Action.LIST_ACTIVE, "View Active Sessions", "List active user sessions"
)
PERM_SESSION_REVOKE = _permission(
    ResourceType.SESSION, Action.REVOKE, "Revoke Sessions", "Force logout of user sessions"
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

# User self-service authentication permissions
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

# API Key self-service permissions
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

# Organization invite permission
PERM_ORG_INVITE = _permission(
    ResourceType.ORGANIZATION,
    Action.INVITE,
    "Invite to Organization",
    "Invite new users to the organization",
)

# ============================================================================
# ENTERPRISE PERMISSIONS - Approval Workflows
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
# ENTERPRISE PERMISSIONS - Enhanced Connector Lifecycle
# ============================================================================

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

# Enterprise sensitive operations - Audit
PERM_AUDIT_LOG_DELETE = _permission(
    ResourceType.AUDIT_LOG,
    Action.DELETE,
    "Delete Audit Logs",
    "Permanently delete audit trail records (compliance-critical)",
)

# Enterprise sensitive operations - Evidence
PERM_EVIDENCE_DELETE = _permission(
    ResourceType.EVIDENCE,
    Action.DELETE,
    "Delete Evidence",
    "Permanently remove evidence records",
)

# Enterprise sensitive operations - Template
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

# Enterprise sensitive operations - Quota
PERM_QUOTA_OVERRIDE = _permission(
    ResourceType.QUOTA,
    Action.OVERRIDE,
    "Override Quotas",
    "Emergency quota overrides",
)

# Enterprise sensitive operations - Budget
PERM_BUDGET_OVERRIDE = _permission(
    ResourceType.BUDGET,
    Action.OVERRIDE,
    "Override Budget",
    "Emergency budget increases",
)

# Enterprise sensitive operations - Session
PERM_SESSION_CREATE = _permission(
    ResourceType.SESSION,
    Action.CREATE,
    "Create Sessions",
    "Create sessions on behalf of users",
)

# Enterprise sensitive operations - Organization
PERM_ORG_DELETE = _permission(
    ResourceType.ORGANIZATION,
    Action.DELETE,
    "Delete Organization",
    "Permanently delete an organization (irreversible)",
)

# Enterprise sensitive operations - Team
PERM_TEAM_DISSOLVE = _permission(
    ResourceType.TEAM,
    Action.DISSOLVE,
    "Dissolve Teams",
    "Dissolve teams with resource reallocation",
)

# Enterprise sensitive operations - Role
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

# Enterprise sensitive operations - API Key
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

# Enterprise sensitive operations - Billing
PERM_BILLING_EXPORT_HISTORY = _permission(
    ResourceType.BILLING,
    Action.EXPORT_HISTORY,
    "Export Billing History",
    "Export historical billing data",
)

# Enterprise sensitive operations - Backup
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

# Enterprise sensitive operations - Disaster Recovery
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


# All permissions as a dictionary for easy lookup
SYSTEM_PERMISSIONS: dict[str, Permission] = {
    p.key: p
    for p in [
        # Debates
        PERM_DEBATE_CREATE,
        PERM_DEBATE_READ,
        PERM_DEBATE_UPDATE,
        PERM_DEBATE_DELETE,
        PERM_DEBATE_RUN,
        PERM_DEBATE_STOP,
        PERM_DEBATE_FORK,
        # Agents
        PERM_AGENT_CREATE,
        PERM_AGENT_READ,
        PERM_AGENT_UPDATE,
        PERM_AGENT_DELETE,
        PERM_AGENT_DEPLOY,
        # Users
        PERM_USER_READ,
        PERM_USER_INVITE,
        PERM_USER_REMOVE,
        PERM_USER_CHANGE_ROLE,
        PERM_USER_IMPERSONATE,
        # Organization
        PERM_ORG_READ,
        PERM_ORG_UPDATE,
        PERM_ORG_BILLING,
        PERM_ORG_AUDIT,
        PERM_ORG_EXPORT,
        # API
        PERM_API_GENERATE_KEY,
        PERM_API_REVOKE_KEY,
        # Memory
        PERM_MEMORY_READ,
        PERM_MEMORY_UPDATE,
        PERM_MEMORY_DELETE,
        # Workflows
        PERM_WORKFLOW_CREATE,
        PERM_WORKFLOW_READ,
        PERM_WORKFLOW_RUN,
        PERM_WORKFLOW_DELETE,
        # Analytics
        PERM_ANALYTICS_READ,
        PERM_ANALYTICS_EXPORT,
        # Introspection
        PERM_INTROSPECTION_READ,
        # Reasoning
        PERM_REASONING_READ,
        PERM_REASONING_UPDATE,
        # Knowledge
        PERM_KNOWLEDGE_READ,
        PERM_KNOWLEDGE_UPDATE,
        # Inbox
        PERM_INBOX_READ,
        PERM_INBOX_UPDATE,
        # Training
        PERM_TRAINING_READ,
        PERM_TRAINING_CREATE,
        # Evidence
        PERM_EVIDENCE_READ,
        PERM_EVIDENCE_CREATE,
        # Documents
        PERM_DOCUMENTS_READ,
        PERM_DOCUMENTS_CREATE,
        PERM_UPLOAD_CREATE,
        PERM_SPEECH_CREATE,
        # Connectors
        PERM_CONNECTOR_READ,
        PERM_CONNECTOR_CREATE,
        PERM_CONNECTOR_DELETE,
        # Repositories
        PERM_REPOSITORY_READ,
        PERM_REPOSITORY_CREATE,
        PERM_REPOSITORY_UPDATE,
        PERM_REPOSITORY_DELETE,
        # Message bindings
        PERM_BINDINGS_READ,
        PERM_BINDINGS_CREATE,
        PERM_BINDINGS_UPDATE,
        PERM_BINDINGS_DELETE,
        # Admin
        PERM_ADMIN_CONFIG,
        PERM_ADMIN_METRICS,
        PERM_ADMIN_FEATURES,
        PERM_ADMIN_ALL,
        # Webhooks
        PERM_WEBHOOK_READ,
        PERM_WEBHOOK_CREATE,
        PERM_WEBHOOK_DELETE,
        PERM_WEBHOOK_ADMIN,
        # Checkpoints
        PERM_CHECKPOINT_READ,
        PERM_CHECKPOINT_CREATE,
        PERM_CHECKPOINT_DELETE,
        # Replays
        PERM_REPLAYS_READ,
        # Gauntlet (adversarial stress-testing)
        PERM_GAUNTLET_RUN,
        PERM_GAUNTLET_READ,
        PERM_GAUNTLET_DELETE,
        PERM_GAUNTLET_SIGN,
        PERM_GAUNTLET_COMPARE,
        PERM_GAUNTLET_EXPORT,
        # Marketplace
        PERM_MARKETPLACE_READ,
        PERM_MARKETPLACE_PUBLISH,
        PERM_MARKETPLACE_IMPORT,
        PERM_MARKETPLACE_RATE,
        PERM_MARKETPLACE_REVIEW,
        PERM_MARKETPLACE_DELETE,
        # Explainability
        PERM_EXPLAINABILITY_READ,
        PERM_EXPLAINABILITY_BATCH,
        # Findings (audit)
        PERM_FINDINGS_READ,
        PERM_FINDINGS_UPDATE,
        PERM_FINDINGS_ASSIGN,
        PERM_FINDINGS_BULK,
        # Decisions
        PERM_DECISION_CREATE,
        PERM_DECISION_READ,
        # Policies (governance)
        PERM_POLICY_READ,
        PERM_POLICY_CREATE,
        PERM_POLICY_UPDATE,
        PERM_POLICY_DELETE,
        # Compliance
        PERM_COMPLIANCE_READ,
        PERM_COMPLIANCE_UPDATE,
        PERM_COMPLIANCE_CHECK,
        # Control Plane
        PERM_CONTROL_PLANE_READ,
        PERM_CONTROL_PLANE_SUBMIT,
        PERM_CONTROL_PLANE_CANCEL,
        PERM_CONTROL_PLANE_DELIBERATE,
        # Enterprise - Data Governance
        PERM_DATA_CLASSIFICATION_READ,
        PERM_DATA_CLASSIFICATION_CLASSIFY,
        PERM_DATA_CLASSIFICATION_UPDATE,
        PERM_DATA_RETENTION_READ,
        PERM_DATA_RETENTION_UPDATE,
        PERM_DATA_LINEAGE_READ,
        PERM_PII_READ,
        PERM_PII_REDACT,
        PERM_PII_MASK,
        # Enterprise - Compliance
        PERM_COMPLIANCE_POLICY_READ,
        PERM_COMPLIANCE_POLICY_UPDATE,
        PERM_COMPLIANCE_POLICY_ENFORCE,
        PERM_AUDIT_LOG_READ,
        PERM_AUDIT_LOG_EXPORT,
        PERM_AUDIT_LOG_SEARCH,
        PERM_AUDIT_LOG_STREAM,
        PERM_AUDIT_LOG_CONFIGURE,
        PERM_VENDOR_READ,
        PERM_VENDOR_APPROVE,
        # Enterprise - Teams
        PERM_TEAM_CREATE,
        PERM_TEAM_READ,
        PERM_TEAM_UPDATE,
        PERM_TEAM_DELETE,
        PERM_TEAM_ADD_MEMBER,
        PERM_TEAM_REMOVE_MEMBER,
        PERM_TEAM_SHARE,
        # Enterprise - Cost & Quota
        PERM_QUOTA_READ,
        PERM_QUOTA_UPDATE,
        PERM_COST_CENTER_READ,
        PERM_COST_CENTER_UPDATE,
        PERM_BUDGET_READ,
        PERM_BUDGET_UPDATE,
        # Cost optimization
        PERM_BILLING_READ,
        PERM_BILLING_RECOMMENDATIONS_READ,
        PERM_BILLING_RECOMMENDATIONS_APPLY,
        PERM_BILLING_FORECAST_READ,
        PERM_BILLING_FORECAST_SIMULATE,
        # Enterprise - Session & Auth
        PERM_SESSION_READ,
        PERM_SESSION_REVOKE,
        PERM_AUTH_RESET_PASSWORD,
        PERM_AUTH_REQUIRE_MFA,
        PERM_AUTH_READ,
        PERM_AUTH_CREATE,
        PERM_AUTH_UPDATE,
        PERM_AUTH_REVOKE,
        PERM_API_KEY_CREATE,
        PERM_API_KEY_REVOKE,
        PERM_ORG_INVITE,
        # Enterprise - Approvals
        PERM_APPROVAL_REQUEST,
        PERM_APPROVAL_GRANT,
        PERM_APPROVAL_READ,
        # Enterprise - Connector Lifecycle
        PERM_CONNECTOR_AUTHORIZE,
        PERM_CONNECTOR_ROTATE,
        PERM_CONNECTOR_TEST,
        PERM_CONNECTOR_UPDATE,
        PERM_CONNECTOR_ROLLBACK,
        # Enterprise - Sensitive Operations
        PERM_AUDIT_LOG_DELETE,
        PERM_EVIDENCE_DELETE,
        PERM_TEMPLATE_CREATE,
        PERM_TEMPLATE_READ,
        PERM_TEMPLATE_UPDATE,
        PERM_TEMPLATE_DELETE,
        PERM_QUOTA_OVERRIDE,
        PERM_BUDGET_OVERRIDE,
        PERM_SESSION_CREATE,
        PERM_ORG_DELETE,
        PERM_TEAM_DISSOLVE,
        PERM_ROLE_CREATE,
        PERM_ROLE_READ,
        PERM_ROLE_UPDATE,
        PERM_ROLE_DELETE,
        PERM_API_KEY_LIST_ALL,
        PERM_API_KEY_EXPORT_SECRET,
        PERM_BILLING_EXPORT_HISTORY,
        # Enterprise - Backup & DR
        PERM_BACKUP_CREATE,
        PERM_BACKUP_READ,
        PERM_BACKUP_RESTORE,
        PERM_BACKUP_DELETE,
        PERM_DR_READ,
        PERM_DR_EXECUTE,
        # Workspaces (SME RBAC-lite)
        PERM_WORKSPACE_CREATE,
        PERM_WORKSPACE_READ,
        PERM_WORKSPACE_UPDATE,
        PERM_WORKSPACE_DELETE,
        PERM_WORKSPACE_MEMBER_ADD,
        PERM_WORKSPACE_MEMBER_REMOVE,
        PERM_WORKSPACE_MEMBER_CHANGE_ROLE,
        PERM_WORKSPACE_SHARE,
        # System operations
        PERM_QUEUE_READ,
        PERM_QUEUE_MANAGE,
        PERM_QUEUE_ADMIN,
        PERM_NOMIC_READ,
        PERM_NOMIC_ADMIN,
        PERM_ORCHESTRATION_READ,
        PERM_ORCHESTRATION_EXECUTE,
        PERM_SYSTEM_HEALTH_READ,
        # Verticals
        PERM_VERTICALS_READ,
        PERM_VERTICALS_WRITE,
        # Canvas
        PERM_CANVAS_READ,
        PERM_CANVAS_CREATE,
        PERM_CANVAS_UPDATE,
        PERM_CANVAS_DELETE,
        PERM_CANVAS_RUN,
        # Verification
        PERM_VERIFICATION_READ,
        PERM_VERIFICATION_CREATE,
        # Codebase analysis
        PERM_CODEBASE_READ,
        PERM_CODEBASE_ANALYZE,
    ]
}


# ============================================================================
# SYSTEM ROLES
# ============================================================================

# Owner - Full control over organization
ROLE_OWNER = Role(
    id="owner",
    name="owner",
    display_name="Owner",
    description="Full control over the organization. Can manage billing, users, and all resources.",
    permissions={p.key for p in SYSTEM_PERMISSIONS.values()},
    priority=100,
    is_system=True,
)

# Admin - Administrative access without billing
ROLE_ADMIN = Role(
    id="admin",
    name="admin",
    display_name="Administrator",
    description="Manage users and resources. Cannot manage billing.",
    permissions={
        # All debate operations
        PERM_DEBATE_CREATE.key,
        PERM_DEBATE_READ.key,
        PERM_DEBATE_UPDATE.key,
        PERM_DEBATE_DELETE.key,
        PERM_DEBATE_RUN.key,
        PERM_DEBATE_STOP.key,
        PERM_DEBATE_FORK.key,
        # All agent operations
        PERM_AGENT_CREATE.key,
        PERM_AGENT_READ.key,
        PERM_AGENT_UPDATE.key,
        PERM_AGENT_DELETE.key,
        PERM_AGENT_DEPLOY.key,
        # User management
        PERM_USER_READ.key,
        PERM_USER_INVITE.key,
        PERM_USER_REMOVE.key,
        PERM_USER_CHANGE_ROLE.key,
        # Organization (no billing)
        PERM_ORG_READ.key,
        PERM_ORG_UPDATE.key,
        PERM_ORG_AUDIT.key,
        PERM_ORG_EXPORT.key,
        PERM_ORG_INVITE.key,
        # API keys
        PERM_API_GENERATE_KEY.key,
        PERM_API_REVOKE_KEY.key,
        PERM_API_KEY_CREATE.key,
        PERM_API_KEY_REVOKE.key,
        # Authentication management
        PERM_AUTH_READ.key,
        PERM_AUTH_CREATE.key,
        PERM_AUTH_UPDATE.key,
        PERM_AUTH_REVOKE.key,
        PERM_AUTH_RESET_PASSWORD.key,
        PERM_AUTH_REQUIRE_MFA.key,
        # Session management
        PERM_SESSION_READ.key,
        PERM_SESSION_REVOKE.key,
        # All memory
        PERM_MEMORY_READ.key,
        PERM_MEMORY_UPDATE.key,
        PERM_MEMORY_DELETE.key,
        # All workflows
        PERM_WORKFLOW_CREATE.key,
        PERM_WORKFLOW_READ.key,
        PERM_WORKFLOW_RUN.key,
        PERM_WORKFLOW_DELETE.key,
        # Analytics
        PERM_ANALYTICS_READ.key,
        PERM_ANALYTICS_EXPORT.key,
        # Training
        PERM_TRAINING_READ.key,
        PERM_TRAINING_CREATE.key,
        # Evidence
        PERM_EVIDENCE_READ.key,
        PERM_EVIDENCE_CREATE.key,
        # Documents
        PERM_DOCUMENTS_READ.key,
        PERM_DOCUMENTS_CREATE.key,
        PERM_UPLOAD_CREATE.key,
        PERM_SPEECH_CREATE.key,
        # Documents
        PERM_DOCUMENTS_READ.key,
        PERM_DOCUMENTS_CREATE.key,
        PERM_UPLOAD_CREATE.key,
        PERM_SPEECH_CREATE.key,
        # Connectors
        PERM_CONNECTOR_READ.key,
        PERM_CONNECTOR_CREATE.key,
        PERM_CONNECTOR_DELETE.key,
        # Repositories
        PERM_REPOSITORY_READ.key,
        PERM_REPOSITORY_CREATE.key,
        PERM_REPOSITORY_UPDATE.key,
        PERM_REPOSITORY_DELETE.key,
        # Message bindings
        PERM_BINDINGS_READ.key,
        PERM_BINDINGS_CREATE.key,
        PERM_BINDINGS_UPDATE.key,
        PERM_BINDINGS_DELETE.key,
        # Webhooks
        PERM_WEBHOOK_READ.key,
        PERM_WEBHOOK_CREATE.key,
        PERM_WEBHOOK_DELETE.key,
        PERM_WEBHOOK_ADMIN.key,
        # Checkpoints
        PERM_CHECKPOINT_READ.key,
        PERM_CHECKPOINT_CREATE.key,
        PERM_CHECKPOINT_DELETE.key,
        # Gauntlet (all operations)
        PERM_GAUNTLET_RUN.key,
        PERM_GAUNTLET_READ.key,
        PERM_GAUNTLET_DELETE.key,
        PERM_GAUNTLET_SIGN.key,
        PERM_GAUNTLET_COMPARE.key,
        PERM_GAUNTLET_EXPORT.key,
        # Marketplace (all operations)
        PERM_MARKETPLACE_READ.key,
        PERM_MARKETPLACE_PUBLISH.key,
        PERM_MARKETPLACE_IMPORT.key,
        PERM_MARKETPLACE_RATE.key,
        PERM_MARKETPLACE_REVIEW.key,
        PERM_MARKETPLACE_DELETE.key,
        # Explainability (all operations)
        PERM_EXPLAINABILITY_READ.key,
        PERM_EXPLAINABILITY_BATCH.key,
        # Findings (all operations)
        PERM_FINDINGS_READ.key,
        PERM_FINDINGS_UPDATE.key,
        PERM_FINDINGS_ASSIGN.key,
        PERM_FINDINGS_BULK.key,
        # Admin (limited)
        PERM_ADMIN_METRICS.key,
        # Decisions (all)
        PERM_DECISION_CREATE.key,
        PERM_DECISION_READ.key,
        # Workspaces (all)
        PERM_WORKSPACE_CREATE.key,
        PERM_WORKSPACE_READ.key,
        PERM_WORKSPACE_UPDATE.key,
        PERM_WORKSPACE_DELETE.key,
        PERM_WORKSPACE_MEMBER_ADD.key,
        PERM_WORKSPACE_MEMBER_REMOVE.key,
        PERM_WORKSPACE_MEMBER_CHANGE_ROLE.key,
        PERM_WORKSPACE_SHARE.key,
        # System operations (all admin access)
        PERM_QUEUE_READ.key,
        PERM_QUEUE_MANAGE.key,
        PERM_QUEUE_ADMIN.key,
        PERM_NOMIC_READ.key,
        PERM_NOMIC_ADMIN.key,
        PERM_ORCHESTRATION_READ.key,
        PERM_ORCHESTRATION_EXECUTE.key,
        PERM_SYSTEM_HEALTH_READ.key,
        # Verticals
        PERM_VERTICALS_READ.key,
        PERM_VERTICALS_WRITE.key,
        # Canvas (all operations)
        PERM_CANVAS_READ.key,
        PERM_CANVAS_CREATE.key,
        PERM_CANVAS_UPDATE.key,
        PERM_CANVAS_DELETE.key,
        PERM_CANVAS_RUN.key,
        # Verification (all operations)
        PERM_VERIFICATION_READ.key,
        PERM_VERIFICATION_CREATE.key,
        # Codebase analysis (all operations)
        PERM_CODEBASE_READ.key,
        PERM_CODEBASE_ANALYZE.key,
    },
    parent_roles=[],
    priority=80,
    is_system=True,
)

# Debate Creator - Can create and manage debates
ROLE_DEBATE_CREATOR = Role(
    id="debate_creator",
    name="debate_creator",
    display_name="Debate Creator",
    description="Create, run, and manage debates. Cannot manage users or billing.",
    permissions={
        # Debate operations
        PERM_DEBATE_CREATE.key,
        PERM_DEBATE_READ.key,
        PERM_DEBATE_UPDATE.key,
        PERM_DEBATE_RUN.key,
        PERM_DEBATE_STOP.key,
        PERM_DEBATE_FORK.key,
        # Agent (read only + configure)
        PERM_AGENT_READ.key,
        # Memory
        PERM_MEMORY_READ.key,
        PERM_MEMORY_UPDATE.key,
        # Workflows
        PERM_WORKFLOW_CREATE.key,
        PERM_WORKFLOW_READ.key,
        PERM_WORKFLOW_RUN.key,
        # Evidence
        PERM_EVIDENCE_READ.key,
        PERM_EVIDENCE_CREATE.key,
        # Analytics (read)
        PERM_ANALYTICS_READ.key,
        # Checkpoints
        PERM_CHECKPOINT_READ.key,
        PERM_CHECKPOINT_CREATE.key,
        # Gauntlet (run and read)
        PERM_GAUNTLET_RUN.key,
        PERM_GAUNTLET_READ.key,
        PERM_GAUNTLET_COMPARE.key,
        PERM_GAUNTLET_EXPORT.key,
        # Marketplace (create and read)
        PERM_MARKETPLACE_READ.key,
        PERM_MARKETPLACE_PUBLISH.key,
        PERM_MARKETPLACE_IMPORT.key,
        PERM_MARKETPLACE_RATE.key,
        PERM_MARKETPLACE_REVIEW.key,
        # Explainability (read and batch)
        PERM_EXPLAINABILITY_READ.key,
        PERM_EXPLAINABILITY_BATCH.key,
        # Findings (read, update, assign - no bulk)
        PERM_FINDINGS_READ.key,
        PERM_FINDINGS_UPDATE.key,
        PERM_FINDINGS_ASSIGN.key,
        # User (self only - enforced at resource level)
        PERM_USER_READ.key,
        # Org (read only)
        PERM_ORG_READ.key,
        # API keys for self
        PERM_API_GENERATE_KEY.key,
        # Decisions
        PERM_DECISION_CREATE.key,
        PERM_DECISION_READ.key,
        # Verticals
        PERM_VERTICALS_READ.key,
        # Canvas (create and manage)
        PERM_CANVAS_READ.key,
        PERM_CANVAS_CREATE.key,
        PERM_CANVAS_UPDATE.key,
        PERM_CANVAS_RUN.key,
        # Verification (read only)
        PERM_VERIFICATION_READ.key,
        # Codebase analysis (read only)
        PERM_CODEBASE_READ.key,
    },
    priority=50,
    is_system=True,
)

# Analyst - Read access to data and analytics
ROLE_ANALYST = Role(
    id="analyst",
    name="analyst",
    display_name="Analyst",
    description="View debates, analytics, and reports. Cannot create or modify resources.",
    permissions={
        # Read-only debate access
        PERM_DEBATE_READ.key,
        # Agent read
        PERM_AGENT_READ.key,
        # Memory read
        PERM_MEMORY_READ.key,
        # Workflow read
        PERM_WORKFLOW_READ.key,
        # Analytics (full)
        PERM_ANALYTICS_READ.key,
        PERM_ANALYTICS_EXPORT.key,
        # Training read
        PERM_TRAINING_READ.key,
        # Evidence read
        PERM_EVIDENCE_READ.key,
        # Documents read
        PERM_DOCUMENTS_READ.key,
        # Checkpoint read
        PERM_CHECKPOINT_READ.key,
        # Gauntlet (read only)
        PERM_GAUNTLET_READ.key,
        # Marketplace (read only)
        PERM_MARKETPLACE_READ.key,
        # Explainability (read only)
        PERM_EXPLAINABILITY_READ.key,
        # Findings (read only)
        PERM_FINDINGS_READ.key,
        # User/Org read
        PERM_USER_READ.key,
        PERM_ORG_READ.key,
    },
    priority=30,
    is_system=True,
)

# Viewer - Minimal read-only access
ROLE_VIEWER = Role(
    id="viewer",
    name="viewer",
    display_name="Viewer",
    description="View debates and basic information. No modification rights.",
    permissions={
        PERM_DEBATE_READ.key,
        PERM_AGENT_READ.key,
        PERM_ORG_READ.key,
        PERM_FINDINGS_READ.key,
    },
    priority=10,
    is_system=True,
)

# Member - Default role for organization members (backward compatibility)
ROLE_MEMBER = Role(
    id="member",
    name="member",
    display_name="Member",
    description="Default organization member with standard access.",
    permissions={
        # Debate (create + run)
        PERM_DEBATE_CREATE.key,
        PERM_DEBATE_READ.key,
        PERM_DEBATE_RUN.key,
        PERM_DEBATE_STOP.key,
        PERM_DEBATE_FORK.key,
        # Agent read
        PERM_AGENT_READ.key,
        # Memory
        PERM_MEMORY_READ.key,
        # Workflow
        PERM_WORKFLOW_CREATE.key,
        PERM_WORKFLOW_READ.key,
        PERM_WORKFLOW_RUN.key,
        # Evidence
        PERM_EVIDENCE_READ.key,
        PERM_EVIDENCE_CREATE.key,
        # Analytics read
        PERM_ANALYTICS_READ.key,
        # Introspection
        PERM_INTROSPECTION_READ.key,
        # Reasoning
        PERM_REASONING_READ.key,
        # Knowledge
        PERM_KNOWLEDGE_READ.key,
        # Inbox
        PERM_INBOX_READ.key,
        PERM_INBOX_UPDATE.key,
        # Checkpoints
        PERM_CHECKPOINT_READ.key,
        PERM_CHECKPOINT_CREATE.key,
        # Replays (read)
        PERM_REPLAYS_READ.key,
        # Gauntlet (run and read)
        PERM_GAUNTLET_RUN.key,
        PERM_GAUNTLET_READ.key,
        # Marketplace (read and interact)
        PERM_MARKETPLACE_READ.key,
        PERM_MARKETPLACE_IMPORT.key,
        PERM_MARKETPLACE_RATE.key,
        PERM_MARKETPLACE_REVIEW.key,
        # Explainability (read only)
        PERM_EXPLAINABILITY_READ.key,
        # Findings (read and update own assignments)
        PERM_FINDINGS_READ.key,
        PERM_FINDINGS_UPDATE.key,
        # Basic access
        PERM_USER_READ.key,
        PERM_ORG_READ.key,
        # API key for self
        PERM_API_GENERATE_KEY.key,
        PERM_API_KEY_CREATE.key,
        PERM_API_KEY_REVOKE.key,
        # Authentication (self-service)
        PERM_AUTH_READ.key,
        PERM_AUTH_CREATE.key,
        PERM_AUTH_UPDATE.key,
        PERM_AUTH_REVOKE.key,
        # Session management (self)
        PERM_SESSION_READ.key,
        PERM_SESSION_REVOKE.key,
        # Decisions
        PERM_DECISION_CREATE.key,
        PERM_DECISION_READ.key,
        # Workspaces (read and share)
        PERM_WORKSPACE_READ.key,
        PERM_WORKSPACE_SHARE.key,
        # Verticals
        PERM_VERTICALS_READ.key,
        # Canvas (basic access)
        PERM_CANVAS_READ.key,
        PERM_CANVAS_CREATE.key,
        PERM_CANVAS_RUN.key,
        # Nomic (read-only access to self-improvement loop)
        PERM_NOMIC_READ.key,
    },
    parent_roles=[],
    priority=40,
    is_system=True,
)

# Compliance Officer - Enterprise compliance and data governance
ROLE_COMPLIANCE_OFFICER = Role(
    id="compliance_officer",
    name="compliance_officer",
    display_name="Compliance Officer",
    description="Manage compliance policies, data governance, and audit trails. Cannot modify debates or agents.",
    permissions={
        # Data governance (full control)
        PERM_DATA_CLASSIFICATION_READ.key,
        PERM_DATA_CLASSIFICATION_CLASSIFY.key,
        PERM_DATA_CLASSIFICATION_UPDATE.key,
        PERM_DATA_RETENTION_READ.key,
        PERM_DATA_RETENTION_UPDATE.key,
        PERM_DATA_LINEAGE_READ.key,
        PERM_PII_READ.key,
        PERM_PII_REDACT.key,
        PERM_PII_MASK.key,
        # Compliance (full control)
        PERM_COMPLIANCE_POLICY_READ.key,
        PERM_COMPLIANCE_POLICY_UPDATE.key,
        PERM_COMPLIANCE_POLICY_ENFORCE.key,
        # Audit logs (full access)
        PERM_AUDIT_LOG_READ.key,
        PERM_AUDIT_LOG_EXPORT.key,
        PERM_AUDIT_LOG_SEARCH.key,
        PERM_AUDIT_LOG_STREAM.key,
        PERM_AUDIT_LOG_CONFIGURE.key,
        # Vendor management
        PERM_VENDOR_READ.key,
        PERM_VENDOR_APPROVE.key,
        # Approvals (can grant)
        PERM_APPROVAL_GRANT.key,
        PERM_APPROVAL_READ.key,
        # Read access to resources for auditing
        PERM_DEBATE_READ.key,
        PERM_AGENT_READ.key,
        PERM_USER_READ.key,
        PERM_ORG_READ.key,
        PERM_ORG_AUDIT.key,
        PERM_FINDINGS_READ.key,
        PERM_FINDINGS_UPDATE.key,
        PERM_GAUNTLET_READ.key,
        # Session management for security
        PERM_SESSION_READ.key,
        PERM_SESSION_REVOKE.key,
        PERM_AUTH_REQUIRE_MFA.key,
    },
    priority=75,  # Between admin and debate_creator
    is_system=True,
)

# Team Lead - Manage team members and resources
ROLE_TEAM_LEAD = Role(
    id="team_lead",
    name="team_lead",
    display_name="Team Lead",
    description="Manage team membership and share resources with team. Inherits member permissions.",
    permissions={
        # Team management
        PERM_TEAM_READ.key,
        PERM_TEAM_UPDATE.key,
        PERM_TEAM_ADD_MEMBER.key,
        PERM_TEAM_REMOVE_MEMBER.key,
        PERM_TEAM_SHARE.key,
        # Quotas (read only)
        PERM_QUOTA_READ.key,
        # Cost center (read only)
        PERM_COST_CENTER_READ.key,
        # Approvals (can request and view)
        PERM_APPROVAL_REQUEST.key,
        PERM_APPROVAL_READ.key,
        # All debate operations (like member+)
        PERM_DEBATE_CREATE.key,
        PERM_DEBATE_READ.key,
        PERM_DEBATE_UPDATE.key,
        PERM_DEBATE_RUN.key,
        PERM_DEBATE_STOP.key,
        PERM_DEBATE_FORK.key,
        # Agent read
        PERM_AGENT_READ.key,
        # Memory
        PERM_MEMORY_READ.key,
        PERM_MEMORY_UPDATE.key,
        # Workflow
        PERM_WORKFLOW_CREATE.key,
        PERM_WORKFLOW_READ.key,
        PERM_WORKFLOW_RUN.key,
        # Evidence
        PERM_EVIDENCE_READ.key,
        PERM_EVIDENCE_CREATE.key,
        # Analytics
        PERM_ANALYTICS_READ.key,
        # Checkpoints
        PERM_CHECKPOINT_READ.key,
        PERM_CHECKPOINT_CREATE.key,
        # Gauntlet
        PERM_GAUNTLET_RUN.key,
        PERM_GAUNTLET_READ.key,
        # Marketplace
        PERM_MARKETPLACE_READ.key,
        PERM_MARKETPLACE_IMPORT.key,
        # Explainability
        PERM_EXPLAINABILITY_READ.key,
        # Findings
        PERM_FINDINGS_READ.key,
        PERM_FINDINGS_UPDATE.key,
        PERM_FINDINGS_ASSIGN.key,
        # Basic access
        PERM_USER_READ.key,
        PERM_ORG_READ.key,
        PERM_API_GENERATE_KEY.key,
        # Decisions
        PERM_DECISION_CREATE.key,
        PERM_DECISION_READ.key,
    },
    parent_roles=["member"],
    priority=55,  # Between debate_creator and member
    is_system=True,
)

# All system roles
SYSTEM_ROLES: dict[str, Role] = {
    r.name: r
    for r in [
        ROLE_OWNER,
        ROLE_ADMIN,
        ROLE_COMPLIANCE_OFFICER,
        ROLE_DEBATE_CREATOR,
        ROLE_TEAM_LEAD,
        ROLE_ANALYST,
        ROLE_VIEWER,
        ROLE_MEMBER,
    ]
}

# Role hierarchy (for inheritance resolution)
ROLE_HIERARCHY: dict[str, list[str]] = {
    "owner": ["admin"],
    "admin": ["compliance_officer", "debate_creator", "analyst"],
    "compliance_officer": ["analyst"],
    "debate_creator": ["team_lead"],
    "team_lead": ["member"],
    "analyst": ["viewer"],
    "member": ["viewer"],
    "viewer": [],
}


def get_permission(key: str) -> Permission | None:
    """Get a permission by its key."""
    return SYSTEM_PERMISSIONS.get(key)


def get_role(name: str) -> Role | None:
    """Get a role by its name."""
    return SYSTEM_ROLES.get(name)


def get_role_permissions(role_name: str, include_inherited: bool = True) -> set[str]:
    """
    Get all permissions for a role, optionally including inherited permissions.

    Args:
        role_name: Name of the role
        include_inherited: Whether to include permissions from parent roles

    Returns:
        Set of permission keys
    """
    role = get_role(role_name)
    if not role:
        return set()

    permissions = set(role.permissions)

    if include_inherited:
        for parent_name in ROLE_HIERARCHY.get(role_name, []):
            permissions |= get_role_permissions(parent_name, include_inherited=True)

    return permissions


def create_custom_role(
    name: str,
    display_name: str,
    description: str,
    permission_keys: set[str],
    org_id: str,
    base_role: str | None = None,
) -> Role:
    """
    Create a custom role for an organization.

    Args:
        name: Role name (must be unique within org)
        display_name: Human-readable name
        description: Role description
        permission_keys: Set of permission keys to grant
        org_id: Organization ID
        base_role: Optional base role to inherit from

    Returns:
        New custom Role instance
    """
    # Start with base role permissions if specified
    permissions = set()
    parent_roles = []

    if base_role:
        base = get_role(base_role)
        if base:
            permissions = get_role_permissions(base_role)
            parent_roles = [base_role]

    # Add specified permissions
    permissions |= permission_keys

    # Validate all permissions exist
    for key in permissions:
        if key not in SYSTEM_PERMISSIONS and not key.endswith(".*"):
            raise ValueError(f"Unknown permission: {key}")

    return Role(
        id=f"{org_id}:{name}",
        name=name,
        display_name=display_name,
        description=description,
        permissions=permissions,
        parent_roles=parent_roles,
        is_system=False,
        is_custom=True,
        org_id=org_id,
        priority=45,  # Between member and debate_creator
    )


# Predefined role templates for quick setup
ROLE_TEMPLATES = {
    "engineering": {
        "base": "debate_creator",
        "add": {PERM_AGENT_CREATE.key, PERM_AGENT_UPDATE.key, PERM_CONNECTOR_CREATE.key},
        "description": "Engineering team with agent management",
    },
    "research": {
        "base": "analyst",
        "add": {PERM_TRAINING_CREATE.key, PERM_DEBATE_CREATE.key, PERM_DEBATE_RUN.key},
        "description": "Research team with training data access",
    },
    "support": {
        "base": "viewer",
        "add": {PERM_USER_READ.key, PERM_ORG_AUDIT.key},
        "description": "Support team with user visibility",
    },
    "external": {
        "base": "viewer",
        "add": set(),
        "description": "External collaborators with minimal access",
    },
}
