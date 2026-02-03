"""
Workflow HTTP API handlers for the Visual Workflow Builder.

This package provides modular HTTP handlers for workflow management:
- CRUD operations (create, read, update, delete workflows)
- Version management (version history, restore)
- Execution management (execute, terminate, status)
- Template management (list, create from template)
- Human approvals (list pending, resolve)

Provides CRUD operations and execution control for workflows:
- /api/workflows - List, create workflows
- /api/workflows/:id - Get, update, delete workflow
- /api/workflows/:id/execute - Execute workflow
- /api/workflows/:id/versions - Version history
- /api/workflows/:id/versions/:version/restore - Restore workflow to version
- /api/workflow-templates - Workflow template gallery
- /api/workflow-approvals - Human approval management
- /api/workflow-executions - List all workflow executions (for runtime dashboard)
- /api/workflow-executions/:id - Get/delete (terminate) execution

Backward Compatibility:
    All exports from the original workflows.py module are re-exported here,
    so existing code using `from aragora.server.handlers.workflows import ...`
    will continue to work.
"""

from __future__ import annotations

# Core utilities (for internal use and advanced users)
from .core import (
    logger,
    _step_result_to_dict,
    _get_store,
    _call_store_method,
    _get_engine,
    _engine,
    _store,
    _TemplateStore,
    WorkflowDefinition,
    WorkflowCategory,
    StepDefinition,
    StepResult,
    TransitionRule,
    PersistentWorkflowStore,
    RBAC_AVAILABLE,
    METRICS_AVAILABLE,
    record_rbac_check,
    track_handler,
    audit_data,
    _UnauthenticatedSentinel,
    _run_async,
)

# CRUD operations
from .crud import (
    list_workflows,
    get_workflow,
    create_workflow,
    update_workflow,
    delete_workflow,
)

# Version management
from .versions import (
    get_workflow_versions,
    restore_workflow_version,
)

# Execution operations
from .execution import (
    execute_workflow,
    get_execution,
    list_executions,
    terminate_execution,
)

# Template management
from .templates import (
    list_templates,
    get_template,
    create_workflow_from_template,
    register_template,
    load_yaml_templates_async,
    register_builtin_templates_async,
    initialize_templates,
    _create_contract_review_template,
    _create_code_review_template,
)

# Human approvals
from .approvals import (
    list_pending_approvals,
    resolve_approval,
    get_approval,
)

# HTTP handlers
from .handler import (
    WorkflowHandler,
    WorkflowHandlers,
)

# Optional RBAC helpers for tests and patching
try:
    from aragora.billing.auth import extract_user_from_request
except ImportError:  # pragma: no cover - optional dependency
    extract_user_from_request = None

try:
    from aragora.rbac import check_permission
except ImportError:  # pragma: no cover - optional dependency
    check_permission = None

# Initialize templates on module import (matches original behavior)
initialize_templates()

__all__ = [
    # Core utilities
    "logger",
    "_step_result_to_dict",
    "_get_store",
    "_call_store_method",
    "_get_engine",
    "_engine",
    "_store",
    "_TemplateStore",
    "WorkflowDefinition",
    "WorkflowCategory",
    "StepDefinition",
    "StepResult",
    "TransitionRule",
    "PersistentWorkflowStore",
    "RBAC_AVAILABLE",
    "METRICS_AVAILABLE",
    "record_rbac_check",
    "track_handler",
    "audit_data",
    "_UnauthenticatedSentinel",
    "_run_async",
    # CRUD operations
    "list_workflows",
    "get_workflow",
    "create_workflow",
    "update_workflow",
    "delete_workflow",
    # Version management
    "get_workflow_versions",
    "restore_workflow_version",
    # Execution operations
    "execute_workflow",
    "get_execution",
    "list_executions",
    "terminate_execution",
    # Template management
    "list_templates",
    "get_template",
    "create_workflow_from_template",
    "register_template",
    "load_yaml_templates_async",
    "register_builtin_templates_async",
    "initialize_templates",
    "_create_contract_review_template",
    "_create_code_review_template",
    # Human approvals
    "list_pending_approvals",
    "resolve_approval",
    "get_approval",
    # HTTP handlers
    "WorkflowHandler",
    "WorkflowHandlers",
    "extract_user_from_request",
    "check_permission",
]
