"""
RBAC Permissions for Miscellaneous Resources.

Contains permissions related to:
- Documents and uploads
- Feedback
- Device management
- Message bindings
- Marketplace
- Skills
- Templates
- Inbox
- Computer-use
- Verticals
- Email
- Transcription
- Evaluation
- Reviews
- A2A (Agent-to-Agent)
"""

from __future__ import annotations

from aragora.rbac.models import Action, ResourceType

from ._helpers import _permission

# ============================================================================
# DOCUMENT PERMISSIONS
# ============================================================================

PERM_DOCUMENTS_READ = _permission(
    ResourceType.DOCUMENTS, Action.READ, "View Documents", "Access document metadata and queries"
)
PERM_DOCUMENTS_CREATE = _permission(
    ResourceType.DOCUMENTS, Action.CREATE, "Upload Documents", "Upload and process documents"
)
PERM_DOCUMENTS_DELETE = _permission(
    ResourceType.DOCUMENTS, Action.DELETE, "Delete Documents", "Permanently delete documents"
)
PERM_UPLOAD_CREATE = _permission(
    ResourceType.UPLOAD, Action.CREATE, "Create Uploads", "Create document folder uploads"
)
PERM_SPEECH_CREATE = _permission(
    ResourceType.SPEECH, Action.CREATE, "Create Speech Jobs", "Generate speech transcripts"
)

# ============================================================================
# FEEDBACK PERMISSIONS
# ============================================================================

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
PERM_SKILLS_INVOKE = _permission(
    ResourceType.SKILLS, Action.INVOKE, "Invoke Skills", "Execute skill operations"
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
PERM_INBOX_CREATE = _permission(
    ResourceType.INBOX, Action.CREATE, "Create Inbox Items", "Create action items and meetings"
)
PERM_INBOX_WRITE = _permission(
    ResourceType.INBOX, Action.WRITE, "Write Inbox", "Full write access to inbox"
)
PERM_INBOX_DELETE = _permission(
    ResourceType.INBOX, Action.DELETE, "Delete Inbox Items", "Delete items from inbox"
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
# VERTICALS PERMISSIONS
# ============================================================================

PERM_VERTICALS_READ = _permission(
    ResourceType.VERTICALS, Action.READ, "View Verticals", "View domain specialists"
)
PERM_VERTICALS_WRITE = _permission(
    ResourceType.VERTICALS, Action.UPDATE, "Update Verticals", "Configure domain specialists"
)

# ============================================================================
# EMAIL PERMISSIONS
# ============================================================================

PERM_EMAIL_READ = _permission(ResourceType.EMAIL, Action.READ, "View Email", "View email messages")
PERM_EMAIL_CREATE = _permission(
    ResourceType.EMAIL, Action.CREATE, "Create Email", "Create and send emails"
)
PERM_EMAIL_UPDATE = _permission(
    ResourceType.EMAIL, Action.UPDATE, "Update Email", "Modify email drafts and settings"
)
PERM_EMAIL_DELETE = _permission(
    ResourceType.EMAIL, Action.DELETE, "Delete Email", "Delete email messages"
)

# ============================================================================
# TRANSCRIPTION PERMISSIONS
# ============================================================================

PERM_TRANSCRIPTION_READ = _permission(
    ResourceType.TRANSCRIPTION, Action.READ, "View Transcription", "View transcription results"
)
PERM_TRANSCRIPTION_CREATE = _permission(
    ResourceType.TRANSCRIPTION,
    Action.CREATE,
    "Create Transcription",
    "Create new transcription jobs",
)

# ============================================================================
# EVALUATION PERMISSIONS
# ============================================================================

PERM_EVALUATION_READ = _permission(
    ResourceType.EVALUATION, Action.READ, "View Evaluations", "View evaluation results"
)
PERM_EVALUATION_CREATE = _permission(
    ResourceType.EVALUATION, Action.CREATE, "Create Evaluations", "Run new evaluations"
)

# ============================================================================
# REVIEWS PERMISSIONS
# ============================================================================

PERM_REVIEWS_READ = _permission(
    ResourceType.REVIEWS, Action.READ, "View Reviews", "View user reviews and ratings"
)

# ============================================================================
# A2A (AGENT-TO-AGENT) PERMISSIONS
# ============================================================================

PERM_A2A_READ = _permission(
    ResourceType.A2A, Action.READ, "View A2A", "View agent-to-agent communication"
)
PERM_A2A_CREATE = _permission(
    ResourceType.A2A, Action.CREATE, "Create A2A", "Create agent-to-agent interactions"
)

# All resource-related permission exports
__all__ = [
    # Documents
    "PERM_DOCUMENTS_READ",
    "PERM_DOCUMENTS_CREATE",
    "PERM_DOCUMENTS_DELETE",
    "PERM_UPLOAD_CREATE",
    "PERM_SPEECH_CREATE",
    # Feedback
    "PERM_FEEDBACK_READ",
    "PERM_FEEDBACK_WRITE",
    "PERM_FEEDBACK_ALL",
    # Device
    "PERM_DEVICE_READ",
    "PERM_DEVICE_WRITE",
    "PERM_DEVICE_NOTIFY",
    # Bindings
    "PERM_BINDINGS_READ",
    "PERM_BINDINGS_CREATE",
    "PERM_BINDINGS_UPDATE",
    "PERM_BINDINGS_DELETE",
    # Marketplace
    "PERM_MARKETPLACE_READ",
    "PERM_MARKETPLACE_PUBLISH",
    "PERM_MARKETPLACE_IMPORT",
    "PERM_MARKETPLACE_RATE",
    "PERM_MARKETPLACE_REVIEW",
    "PERM_MARKETPLACE_DELETE",
    # Skills
    "PERM_SKILLS_READ",
    "PERM_SKILLS_INSTALL",
    "PERM_SKILLS_PUBLISH",
    "PERM_SKILLS_RATE",
    "PERM_SKILLS_INVOKE",
    # Template
    "PERM_TEMPLATE_CREATE",
    "PERM_TEMPLATE_READ",
    "PERM_TEMPLATE_UPDATE",
    "PERM_TEMPLATE_DELETE",
    # Inbox
    "PERM_INBOX_READ",
    "PERM_INBOX_UPDATE",
    "PERM_INBOX_CREATE",
    "PERM_INBOX_WRITE",
    "PERM_INBOX_DELETE",
    # Computer-Use
    "PERM_COMPUTER_USE_READ",
    "PERM_COMPUTER_USE_EXECUTE",
    "PERM_COMPUTER_USE_BROWSER",
    "PERM_COMPUTER_USE_SHELL",
    "PERM_COMPUTER_USE_FILE_READ",
    "PERM_COMPUTER_USE_FILE_WRITE",
    "PERM_COMPUTER_USE_SCREENSHOT",
    "PERM_COMPUTER_USE_NETWORK",
    "PERM_COMPUTER_USE_ADMIN",
    # Verticals
    "PERM_VERTICALS_READ",
    "PERM_VERTICALS_WRITE",
    # Email
    "PERM_EMAIL_READ",
    "PERM_EMAIL_CREATE",
    "PERM_EMAIL_UPDATE",
    "PERM_EMAIL_DELETE",
    # Transcription
    "PERM_TRANSCRIPTION_READ",
    "PERM_TRANSCRIPTION_CREATE",
    # Evaluation
    "PERM_EVALUATION_READ",
    "PERM_EVALUATION_CREATE",
    # Reviews
    "PERM_REVIEWS_READ",
    # A2A
    "PERM_A2A_READ",
    "PERM_A2A_CREATE",
]
