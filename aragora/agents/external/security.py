"""Security policy enforcement for external agent framework integration.

Provides hooks for:
- Pre-execution policy checks (tool permissions, resource limits)
- Tool permission gating via RBAC
- Output sanitization (credential redaction)
- Audit logging
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

from .config import ExternalAgentConfig, ToolConfig
from .models import TaskRequest, TaskResult

if TYPE_CHECKING:
    from aragora.rbac.models import AuthorizationContext

logger = logging.getLogger(__name__)


@dataclass
class PolicyCheckResult:
    """Result of a pre-execution policy check."""

    allowed: bool
    reason: str | None = None
    blocked_tools: set[str] | None = None
    warnings: list[str] = field(default_factory=list)


class ToolPermissionGate:
    """Gates external agent tool access based on Aragora RBAC permissions.

    Checks authorization context against tool permission mappings
    before allowing tool invocation.
    """

    # Default permission mappings for common external agent tools
    DEFAULT_TOOL_PERMISSIONS: dict[str, str] = {
        # OpenHands tools
        "TerminalTool": "computer_use.shell",
        "FileEditorTool": "computer_use.file_write",
        "TaskTrackerTool": "computer_use.read",
        "BrowserTool": "computer_use.browser",
        "ScreenshotTool": "computer_use.screenshot",
        "NetworkTool": "computer_use.network",
        # Generic tool names
        "shell": "computer_use.shell",
        "bash": "computer_use.shell",
        "terminal": "computer_use.shell",
        "file_read": "computer_use.file_read",
        "file_write": "computer_use.file_write",
        "file_edit": "computer_use.file_write",
        "browser": "computer_use.browser",
        "web": "computer_use.browser",
        "network": "computer_use.network",
        "http": "computer_use.network",
        "api": "computer_use.network",
    }

    def __init__(
        self,
        tool_configs: dict[str, ToolConfig] | None = None,
        custom_mappings: dict[str, str] | None = None,
    ):
        """Initialize the gate.

        Args:
            tool_configs: Per-tool configurations from adapter config.
            custom_mappings: Override default permission mappings.
        """
        self._tool_configs = tool_configs or {}
        self._mappings = {
            **self.DEFAULT_TOOL_PERMISSIONS,
            **(custom_mappings or {}),
        }

        # Compile blocked patterns for each tool
        self._compiled_patterns: dict[str, list[re.Pattern[str]]] = {}
        for name, config in self._tool_configs.items():
            if config.blocked_patterns:
                self._compiled_patterns[name] = [
                    re.compile(p, re.IGNORECASE) for p in config.blocked_patterns
                ]

    def check_permission(
        self,
        tool_name: str,
        context: AuthorizationContext,
    ) -> tuple[bool, str]:
        """Check if context has permission to use a tool.

        Args:
            tool_name: Name of the external agent tool.
            context: Aragora authorization context.

        Returns:
            Tuple of (allowed, reason).
        """
        # Check if tool is explicitly disabled
        config = self._tool_configs.get(tool_name)
        if config and not config.enabled:
            return False, f"Tool {tool_name} is disabled"

        # Get permission key for this tool
        permission_key = (
            config.permission_key
            if config and config.permission_key
            else self._mappings.get(tool_name)
        )

        if not permission_key:
            # Unknown tool - check for generic gateway.execute permission
            if hasattr(context, "has_permission"):
                if context.has_permission("gateway.execute"):
                    return True, "Allowed via gateway.execute"
            return False, f"No permission mapping for tool: {tool_name}"

        # Check permission via context
        if hasattr(context, "has_permission"):
            if context.has_permission(permission_key):
                return True, "Permission granted"
            return False, f"Missing permission: {permission_key}"

        # If no has_permission method, check permissions set directly
        if hasattr(context, "permissions"):
            if permission_key in context.permissions:
                return True, "Permission granted"
            return False, f"Missing permission: {permission_key}"

        return False, "Unable to verify permissions"

    def check_arguments(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[bool, str]:
        """Check if tool arguments pass safety filters.

        Args:
            tool_name: Name of the tool.
            arguments: Tool arguments to validate.

        Returns:
            Tuple of (allowed, reason).
        """
        patterns = self._compiled_patterns.get(tool_name, [])
        if not patterns:
            return True, "No argument filters"

        # Serialize arguments for pattern matching
        arg_str = str(arguments)

        for pattern in patterns:
            if pattern.search(arg_str):
                return False, f"Blocked by safety pattern: {pattern.pattern}"

        return True, "Arguments validated"

    def requires_approval(self, tool_name: str) -> bool:
        """Check if a tool requires human approval."""
        from .config import ApprovalMode

        config = self._tool_configs.get(tool_name)
        if not config:
            return False  # Unknown tools don't require approval by default
        return config.approval_mode == ApprovalMode.MANUAL

    def get_timeout(self, tool_name: str) -> float:
        """Get timeout for a tool."""
        config = self._tool_configs.get(tool_name)
        if not config:
            return 60.0  # Default 1 minute
        return config.timeout_seconds


class ExternalAgentSecurityPolicy:
    """Security policy enforcement for external agent tasks.

    Provides hooks for:
    - Pre-execution policy checks (tool permissions, resource limits)
    - Tool permission gating
    - Output sanitization
    - Audit logging
    """

    # Patterns for sensitive content detection
    SECRET_PATTERNS = [
        r'(?i)(api[_-]?key|secret|password|token|credential)["\']?\s*[:=]\s*["\']?[\w-]+',
        r"(?i)bearer\s+[\w-]+",
        r"(?i)aws[_-]?(access|secret)[_-]?key",
        r"sk-[a-zA-Z0-9]{20,}",  # OpenAI API keys
        r"sk-ant-[a-zA-Z0-9-]+",  # Anthropic API keys
        r"ghp_[a-zA-Z0-9]{36}",  # GitHub personal access tokens
        r"gho_[a-zA-Z0-9]{36}",  # GitHub OAuth tokens
        r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----",  # Private keys
        r"eyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+",  # JWT
    ]

    # Dangerous command patterns for shell tools
    DANGEROUS_COMMANDS = [
        r"\brm\s+-rf\s+/",  # rm -rf /
        r"\bdd\s+if=",  # dd commands
        r"\b(curl|wget)\s+.*\|\s*(bash|sh)",  # pipe to shell
        r"\bchmod\s+777",  # overly permissive chmod
        r"\b(sudo|su)\s+-?\s*\w*\s*$",  # privilege escalation
        r"\b:(){ :|:& };:",  # fork bomb
        r"\bmkfs\b",  # filesystem commands
        r"\bshred\b",  # secure delete
    ]

    def __init__(
        self,
        audit_logger: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        """Initialize the security policy.

        Args:
            audit_logger: Optional callback for audit logging.
        """
        self._audit_logger = audit_logger
        self._compiled_secret_patterns = [re.compile(p) for p in self.SECRET_PATTERNS]
        self._compiled_danger_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_COMMANDS
        ]

    def check_pre_execution(
        self,
        request: TaskRequest,
        context: AuthorizationContext,
        adapter_config: ExternalAgentConfig,
    ) -> PolicyCheckResult:
        """Pre-execution policy check.

        Validates:
        - User has permission to use external agents
        - Requested tools are allowed
        - Resource limits are within bounds

        Args:
            request: Task request to validate.
            context: Authorization context with user/roles.
            adapter_config: Adapter configuration for limits.

        Returns:
            PolicyCheckResult with allowed status and details.
        """
        warnings: list[str] = []
        blocked_tools: set[str] = set()

        # Check basic gateway permission
        if hasattr(context, "has_permission"):
            if not context.has_permission("gateway.execute"):
                return PolicyCheckResult(
                    allowed=False,
                    reason="User lacks gateway.execute permission",
                )

        # Check each requested tool permission
        for tool_perm in request.tool_permissions:
            perm_key = tool_perm.to_permission_key()
            if hasattr(context, "has_permission"):
                if not context.has_permission(perm_key):
                    blocked_tools.add(tool_perm.value)

        if blocked_tools:
            return PolicyCheckResult(
                allowed=False,
                reason=f"User lacks permission for tools: {blocked_tools}",
                blocked_tools=blocked_tools,
            )

        # Check against adapter blocked tools
        if adapter_config.blocked_tools:
            for tool in request.tool_permissions:
                if tool.value in adapter_config.blocked_tools:
                    blocked_tools.add(tool.value)

        if blocked_tools:
            return PolicyCheckResult(
                allowed=False,
                reason=f"Tools blocked by policy: {blocked_tools}",
                blocked_tools=blocked_tools,
            )

        # Resource limit warnings
        if hasattr(adapter_config, "max_cost_per_task_usd"):
            estimated_cost = request.metadata.get("estimated_cost", 0)
            if estimated_cost > adapter_config.max_cost_per_task_usd:
                warnings.append(
                    f"Estimated cost ${estimated_cost:.2f} exceeds "
                    f"limit ${adapter_config.max_cost_per_task_usd:.2f}"
                )

        # Audit log the check
        self._audit(
            "policy_check",
            {
                "task_id": request.id,
                "user_id": getattr(context, "user_id", None),
                "allowed": True,
                "warnings": warnings,
            },
        )

        return PolicyCheckResult(
            allowed=True,
            warnings=warnings if warnings else [],
        )

    def gate_tool_access(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        context: AuthorizationContext,
    ) -> tuple[bool, str | None]:
        """Gate access to a specific tool invocation.

        Called at runtime when external agent attempts to use a tool.

        Args:
            tool_name: Name of tool being invoked.
            tool_input: Tool parameters.
            context: Authorization context.

        Returns:
            Tuple of (allowed, denial_reason).
        """
        gate = ToolPermissionGate()

        # Check permission
        allowed, reason = gate.check_permission(tool_name, context)
        if not allowed:
            self._audit(
                "tool_blocked",
                {
                    "tool_name": tool_name,
                    "user_id": getattr(context, "user_id", None),
                    "reason": "permission_denied",
                },
            )
            return False, reason

        # Check for dangerous commands in shell tools
        if tool_name.lower() in ("terminal", "shell", "bash", "terminaltool"):
            command = tool_input.get("command", "")
            if self._is_dangerous_command(command):
                self._audit(
                    "tool_blocked",
                    {
                        "tool_name": tool_name,
                        "user_id": getattr(context, "user_id", None),
                        "reason": "dangerous_command",
                    },
                )
                return False, "Command blocked by security policy"

        return True, None

    def sanitize_output(
        self,
        result: TaskResult,
        redact_secrets: bool = True,
    ) -> TaskResult:
        """Sanitize task output before returning to user.

        Removes sensitive content like API keys, tokens, etc.

        Args:
            result: Task result to sanitize.
            redact_secrets: Whether to redact detected secrets.

        Returns:
            Sanitized TaskResult.
        """
        if not redact_secrets or not result.output:
            return result

        sanitized_output = self._redact_secrets(result.output)

        # Sanitize artifacts
        sanitized_artifacts = []
        for artifact in result.artifacts:
            if "content" in artifact and isinstance(artifact["content"], str):
                sanitized_content = self._redact_secrets(artifact["content"])
                artifact = {**artifact, "content": sanitized_content}
            sanitized_artifacts.append(artifact)

        # Sanitize logs
        sanitized_logs = []
        for log in result.logs:
            if "message" in log and isinstance(log["message"], str):
                sanitized_msg = self._redact_secrets(log["message"])
                log = {**log, "message": sanitized_msg}
            sanitized_logs.append(log)

        return TaskResult(
            task_id=result.task_id,
            status=result.status,
            output=sanitized_output,
            artifacts=sanitized_artifacts,
            steps_executed=result.steps_executed,
            tokens_used=result.tokens_used,
            cost_usd=result.cost_usd,
            started_at=result.started_at,
            completed_at=result.completed_at,
            error=result.error,
            logs=sanitized_logs,
        )

    def audit_task(
        self,
        event: str,
        task_id: str,
        user_id: str | None,
        details: dict[str, Any],
    ) -> None:
        """Audit log a task event.

        Args:
            event: Event type (submitted, completed, failed, cancelled).
            task_id: Task ID.
            user_id: User who initiated the task.
            details: Additional event details.
        """
        self._audit(
            f"task_{event}",
            {
                "task_id": task_id,
                "user_id": user_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **details,
            },
        )

    def _audit(self, event_type: str, data: dict[str, Any]) -> None:
        """Write to audit log."""
        if self._audit_logger:
            try:
                self._audit_logger(event_type, data)
            except (RuntimeError, TypeError, ValueError, OSError) as e:
                logger.warning("Audit log failed: %s", e)
        else:
            logger.info("AUDIT [%s]: %s", event_type, data)

    def _redact_secrets(self, content: str) -> str:
        """Redact secrets from content."""
        result = content
        for pattern in self._compiled_secret_patterns:
            result = pattern.sub("[REDACTED]", result)
        return result

    def _is_dangerous_command(self, command: str) -> bool:
        """Check if a shell command is dangerous."""
        for pattern in self._compiled_danger_patterns:
            if pattern.search(command):
                return True
        return False
