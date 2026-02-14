"""
Skills Registry Module.

Inspired by ClawdBot's skills architecture, this module provides a central
registry for discovering, registering, and invoking skills.

Key features:
- Dynamic skill registration
- RBAC-integrated permission checking
- Skill invocation with validation
- Function schema generation for LLM tool calling
- Execution tracking and metrics

Usage:
    from aragora.skills import SkillRegistry, SkillContext

    registry = SkillRegistry()
    registry.register(WebSearchSkill())

    # Invoke a skill
    context = SkillContext(user_id="user123", permissions=["web:search"])
    result = await registry.invoke("web_search", {"query": "climate change"}, context)

    # Get function schemas for LLM
    schemas = registry.get_function_schemas()
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from collections.abc import Callable

from .base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
    SkillStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy audit imports -- kept optional so skill execution never breaks when
# the audit subsystem is unavailable.
# ---------------------------------------------------------------------------
_audit_log_cls = None
_audit_event_cls = None
_audit_category_cls = None
_audit_outcome_cls = None


def _ensure_audit_imports() -> bool:
    """Lazily import audit types.  Returns True if available."""
    global _audit_log_cls, _audit_event_cls, _audit_category_cls, _audit_outcome_cls
    if _audit_event_cls is not None:
        return True
    try:
        from aragora.audit.log import AuditLog as _AuditLog
        from aragora.audit.log import AuditEvent as _AuditEvent
        from aragora.audit.log import AuditCategory as _AuditCategory
        from aragora.audit.log import AuditOutcome as _AuditOutcome

        _audit_log_cls = _AuditLog
        _audit_event_cls = _AuditEvent
        _audit_category_cls = _AuditCategory
        _audit_outcome_cls = _AuditOutcome
        return True
    except (ImportError, AttributeError):
        return False


def _skill_status_to_audit_outcome(status: SkillStatus) -> str:
    """Map SkillStatus to AuditOutcome value string."""
    if status == SkillStatus.SUCCESS:
        return "success"
    elif status == SkillStatus.PERMISSION_DENIED:
        return "denied"
    else:
        return "failure"


@dataclass
class SkillExecutionMetrics:
    """Metrics for skill executions."""

    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    total_execution_time_ms: float = 0
    average_execution_time_ms: float = 0
    last_invocation: datetime | None = None
    last_error: str | None = None

    def record_execution(self, success: bool, duration_ms: float, error: str | None = None):
        """Record an execution."""
        self.total_invocations += 1
        self.total_execution_time_ms += duration_ms
        self.average_execution_time_ms = self.total_execution_time_ms / self.total_invocations
        self.last_invocation = datetime.now(timezone.utc)

        if success:
            self.successful_invocations += 1
        else:
            self.failed_invocations += 1
            self.last_error = error

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_invocations == 0:
            return 0.0
        return (self.successful_invocations / self.total_invocations) * 100


@dataclass
class RateLimitState:
    """Rate limiting state for a skill."""

    window_start: float = 0.0
    request_count: int = 0


# Modules that indicate capabilities requiring declaration
_SHELL_MODULES = frozenset({"subprocess", "os", "shutil", "pathlib", "tempfile", "glob"})
_NETWORK_MODULES = frozenset({
    "socket", "ssl", "http", "urllib", "urllib3", "requests",
    "httpx", "aiohttp", "ftplib", "smtplib", "poplib",
    "imaplib", "telnetlib", "xmlrpc",
})


def validate_skill_imports(
    skill_class: type,
    manifest: SkillManifest,
) -> list[str]:
    """
    Validate that a skill's module imports are consistent with its declared capabilities.

    Uses inspect.getsource() to get the skill's module source, parses it with ast,
    and checks whether imports like subprocess, os, socket are present when the
    manifest does not declare SHELL_EXECUTION or NETWORK capabilities.

    This is advisory only -- returns a list of warning strings (not errors).

    Args:
        skill_class: The skill class to inspect.
        manifest: The skill's declared manifest.

    Returns:
        List of warning strings. Empty list means no issues found.
    """
    warnings: list[str] = []

    try:
        module = inspect.getmodule(skill_class)
        if module is None:
            return warnings
        source = inspect.getsource(module)
    except (OSError, TypeError):
        # Cannot get source -- dynamically generated or built-in
        return warnings

    try:
        tree = ast.parse(source)
    except SyntaxError:
        warnings.append(
            f"Skill '{manifest.name}': could not parse module source for import validation"
        )
        return warnings

    declared_capabilities = set(manifest.capabilities)
    has_shell = SkillCapability.SHELL_EXECUTION in declared_capabilities
    has_network = SkillCapability.NETWORK in declared_capabilities
    has_external_api = SkillCapability.EXTERNAL_API in declared_capabilities

    found_imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found_imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            found_imports.add(node.module.split(".")[0])

    # Check shell-related imports without SHELL_EXECUTION capability
    if not has_shell:
        shell_found = found_imports & _SHELL_MODULES
        if shell_found:
            warnings.append(
                f"Skill '{manifest.name}' imports {sorted(shell_found)} "
                f"but does not declare SHELL_EXECUTION capability"
            )

    # Check network-related imports without NETWORK or EXTERNAL_API capability
    if not has_network and not has_external_api:
        network_found = found_imports & _NETWORK_MODULES
        if network_found:
            warnings.append(
                f"Skill '{manifest.name}' imports {sorted(network_found)} "
                f"but does not declare NETWORK or EXTERNAL_API capability"
            )

    return warnings


def _validate_skill_code(skill: Skill) -> list[str]:
    """
    Validate a skill's execute method source code using AST-based validation.

    For skills that declare CODE_EXECUTION or SHELL_EXECUTION capabilities,
    attempts to get the source of the execute() method and runs
    validate_python_code() on it in non-strict mode.

    This is best-effort: some skills have dynamically generated code that
    cannot be inspected. Returns warnings, never raises.

    Args:
        skill: The skill instance to validate.

    Returns:
        List of warning strings. Empty list means no issues or could not validate.
    """
    import textwrap

    warnings: list[str] = []

    try:
        source = inspect.getsource(skill.execute)
    except (OSError, TypeError):
        # Cannot get source -- dynamically generated or C extension
        return warnings

    # Dedent the source since it's typically indented as a class method
    source = textwrap.dedent(source)

    try:
        from .builtin.code_execution import CodeValidationError, validate_python_code

        validate_python_code(source, strict_mode=False)
    except CodeValidationError as e:
        warnings.append(
            f"Skill '{skill.manifest.name}' execute() code validation warning: {e}"
        )
    except Exception as e:
        # Don't break registration for unexpected validation errors
        warnings.append(
            f"Skill '{skill.manifest.name}' code validation could not complete: {e}"
        )

    return warnings


class SkillRegistry:
    """
    Central registry for skills.

    Provides:
    - Skill registration and discovery
    - Permission-checked invocation
    - Rate limiting
    - Execution metrics
    - Function schema generation for LLM tool calling
    """

    def __init__(
        self,
        rbac_checker: Any | None = None,
        enable_metrics: bool = True,
        enable_rate_limiting: bool = True,
    ):
        """
        Initialize the skill registry.

        Args:
            rbac_checker: Optional RBAC permission checker
            enable_metrics: Whether to collect execution metrics
            enable_rate_limiting: Whether to enforce rate limits
        """
        self._skills: dict[str, Skill] = {}
        self._metrics: dict[str, SkillExecutionMetrics] = defaultdict(SkillExecutionMetrics)
        self._rate_limits: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._validation_warnings: dict[str, list[str]] = {}
        self._rbac_checker = rbac_checker
        self._enable_metrics = enable_metrics
        self._enable_rate_limiting = enable_rate_limiting
        self._hooks: dict[str, list[Callable]] = {
            "pre_invoke": [],
            "post_invoke": [],
            "on_error": [],
        }
        self._lock = asyncio.Lock()

    def register(self, skill: Skill, replace: bool = False) -> None:
        """
        Register a skill.

        Performs best-effort validation at registration time:
        - For skills with CODE_EXECUTION or SHELL_EXECUTION capabilities,
          validates the execute() method source using AST analysis.
        - Checks that module imports are consistent with declared capabilities.

        Validation failures produce warnings but do not prevent registration.

        Args:
            skill: The skill to register
            replace: If True, replace existing skill with same name

        Raises:
            ValueError: If skill already registered and replace=False
        """
        name = skill.manifest.name

        if name in self._skills and not replace:
            raise ValueError(f"Skill '{name}' already registered. Use replace=True to override.")

        # Clear old warnings on replace
        if replace:
            self._validation_warnings.pop(name, None)

        # Best-effort validation at registration time
        all_warnings: list[str] = []

        # Validate execute() source for skills with code/shell execution capabilities
        manifest = skill.manifest
        code_caps = {SkillCapability.CODE_EXECUTION, SkillCapability.SHELL_EXECUTION}
        if code_caps & set(manifest.capabilities):
            code_warnings = _validate_skill_code(skill)
            all_warnings.extend(code_warnings)

        # Validate imports vs declared capabilities
        import_warnings = validate_skill_imports(type(skill), manifest)
        all_warnings.extend(import_warnings)

        # Store warnings and log them
        if all_warnings:
            self._validation_warnings[name] = all_warnings
            for warning in all_warnings:
                logger.warning(f"Skill validation: {warning}")

        self._skills[name] = skill
        logger.info(
            f"Registered skill: {name} v{skill.manifest.version} "
            f"with capabilities: {[c.value for c in skill.manifest.capabilities]}"
        )

    def unregister(self, name: str) -> bool:
        """
        Unregister a skill.

        Args:
            name: Name of the skill to unregister

        Returns:
            True if skill was unregistered, False if not found
        """
        if name in self._skills:
            del self._skills[name]
            logger.info(f"Unregistered skill: {name}")
            return True
        return False

    def get(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_skills(
        self,
        capability: SkillCapability | None = None,
        tag: str | None = None,
        debate_compatible_only: bool = False,
    ) -> list[SkillManifest]:
        """
        List registered skills with optional filtering.

        Args:
            capability: Filter by capability
            tag: Filter by tag
            debate_compatible_only: Only return debate-compatible skills

        Returns:
            List of skill manifests
        """
        manifests = []
        for skill in self._skills.values():
            manifest = skill.manifest

            # Filter by capability
            if capability and capability not in manifest.capabilities:
                continue

            # Filter by tag
            if tag and tag not in manifest.tags:
                continue

            # Filter by debate compatibility
            if debate_compatible_only and not manifest.debate_compatible:
                continue

            manifests.append(manifest)

        return manifests

    def has_skill(self, name: str) -> bool:
        """Check if a skill is registered."""
        return name in self._skills

    async def invoke(
        self,
        skill_name: str,
        input_data: dict[str, Any],
        context: SkillContext,
    ) -> SkillResult:
        """
        Invoke a skill by name.

        Args:
            skill_name: Name of the skill to invoke
            input_data: Input data for the skill
            context: Execution context

        Returns:
            SkillResult from the execution
        """
        start_time = time.time()

        # Get skill
        skill = self._skills.get(skill_name)
        if not skill:
            logger.warning(f"Skill not found: {skill_name}")
            return SkillResult.create_failure(
                f"Skill '{skill_name}' not found",
                error_code="skill_not_found",
            )

        manifest = skill.manifest

        # --- Audit: log invocation start ---
        logger.info(
            "Skill invocation: skill=%s user=%s tenant=%s",
            skill_name,
            context.user_id,
            context.tenant_id,
        )

        # Run pre-invoke hooks
        for hook in self._hooks["pre_invoke"]:
            try:
                hook(skill_name, input_data, context)
            except Exception as e:
                logger.warning(f"Pre-invoke hook error: {e}")

        try:
            # Check rate limit
            if self._enable_rate_limiting and manifest.rate_limit_per_minute:
                is_allowed, wait_time = self._check_rate_limit(skill_name, manifest)
                if not is_allowed:
                    result: SkillResult = SkillResult.create_failure(
                        f"Rate limit exceeded. Try again in {wait_time:.1f}s",
                        status=SkillStatus.RATE_LIMITED,
                    )
                    self._record_metrics(skill_name, result, start_time)
                    duration = time.time() - start_time
                    logger.info(
                        "Skill completed: skill=%s user=%s status=%s duration=%.3fs",
                        skill_name,
                        context.user_id,
                        result.status.value,
                        duration,
                    )
                    self._emit_audit_event(
                        skill_name, manifest, context, result, duration,
                    )
                    return result

            # Validate input
            is_valid, error_msg = await skill.validate_input(input_data)
            if not is_valid:
                result = SkillResult.create_failure(
                    error_msg or "Invalid input",
                    status=SkillStatus.INVALID_INPUT,
                )
                self._record_metrics(skill_name, result, start_time)
                duration = time.time() - start_time
                logger.info(
                    "Skill completed: skill=%s user=%s status=%s duration=%.3fs",
                    skill_name,
                    context.user_id,
                    result.status.value,
                    duration,
                )
                self._emit_audit_event(
                    skill_name, manifest, context, result, duration,
                )
                return result

            # Check permissions
            has_perm, missing_perm = await self._check_permissions(skill, context)
            if not has_perm:
                result = SkillResult.create_permission_denied(missing_perm or "unknown")
                self._record_metrics(skill_name, result, start_time)
                duration = time.time() - start_time
                logger.info(
                    "Skill completed: skill=%s user=%s status=%s duration=%.3fs",
                    skill_name,
                    context.user_id,
                    result.status.value,
                    duration,
                )
                self._emit_audit_event(
                    skill_name, manifest, context, result, duration,
                    error_message=f"Missing permission: {missing_perm}",
                )
                return result

            # Check debate context if required
            if manifest.requires_debate_context and not context.debate_id:
                result = SkillResult.create_failure(
                    "This skill requires an active debate context",
                    error_code="debate_context_required",
                )
                self._record_metrics(skill_name, result, start_time)
                duration = time.time() - start_time
                logger.info(
                    "Skill completed: skill=%s user=%s status=%s duration=%.3fs",
                    skill_name,
                    context.user_id,
                    result.status.value,
                    duration,
                )
                self._emit_audit_event(
                    skill_name, manifest, context, result, duration,
                )
                return result

            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    skill.execute(input_data, context),
                    timeout=manifest.max_execution_time_seconds,
                )
            except asyncio.TimeoutError:
                result = SkillResult.create_timeout(manifest.max_execution_time_seconds)

            # Record execution start time in result
            result.started_at = datetime.fromtimestamp(start_time, tz=timezone.utc)
            if not result.completed_at:
                result.completed_at = datetime.now(timezone.utc)

            # Record metrics
            self._record_metrics(skill_name, result, start_time)

            # --- Audit: log completion ---
            duration = time.time() - start_time
            logger.info(
                "Skill completed: skill=%s user=%s status=%s duration=%.3fs",
                skill_name,
                context.user_id,
                result.status.value,
                duration,
            )
            self._emit_audit_event(
                skill_name, manifest, context, result, duration,
            )

            # Run post-invoke hooks
            for hook in self._hooks["post_invoke"]:
                try:
                    hook(skill_name, result, context)
                except Exception as e:
                    logger.warning(f"Post-invoke hook error: {e}")

            return result

        except Exception as e:
            logger.exception(f"Skill execution failed: {skill_name}")
            result = SkillResult.create_failure(str(e))
            self._record_metrics(skill_name, result, start_time)

            # --- Audit: log failure ---
            duration = time.time() - start_time
            logger.info(
                "Skill completed: skill=%s user=%s status=%s duration=%.3fs",
                skill_name,
                context.user_id,
                result.status.value,
                duration,
            )
            self._emit_audit_event(
                skill_name, manifest, context, result, duration,
                error_message=str(e),
            )

            # Run error hooks
            for hook in self._hooks["on_error"]:
                try:
                    hook(skill_name, e, context)
                except Exception as hook_error:
                    logger.warning(f"Error hook failed: {hook_error}")

            return result

    def _emit_audit_event(
        self,
        skill_name: str,
        manifest: SkillManifest,
        context: SkillContext,
        result: SkillResult,
        duration: float,
        error_message: str | None = None,
    ) -> None:
        """
        Emit a structured audit event for a skill invocation.

        This is best-effort: if the audit subsystem is unavailable or raises,
        the error is logged but never propagated to the caller.
        """
        try:
            if not _ensure_audit_imports():
                return

            outcome_str = _skill_status_to_audit_outcome(result.status)
            outcome = _audit_outcome_cls(outcome_str)

            details: dict[str, Any] = {
                "skill_name": skill_name,
                "skill_version": manifest.version,
                "required_permissions": manifest.required_permissions,
                "status": result.status.value,
                "duration_seconds": round(duration, 4),
            }
            if error_message or result.error_message:
                details["error"] = error_message or result.error_message

            event = _audit_event_cls(
                category=_audit_category_cls.ACCESS,
                action="skill_invoke",
                actor_id=context.user_id or "anonymous",
                resource_type="skill",
                resource_id=skill_name,
                outcome=outcome,
                org_id=context.tenant_id or "",
                correlation_id=context.correlation_id or "",
                details=details,
                reason=error_message or "",
            )

            audit_log = _audit_log_cls()
            audit_log.log(event)
        except (ImportError, AttributeError, RuntimeError, TypeError):
            logger.debug(
                "Audit event emission failed for skill=%s (non-fatal)",
                skill_name,
                exc_info=True,
            )

    async def _check_permissions(
        self,
        skill: Skill,
        context: SkillContext,
    ) -> tuple[bool, str | None]:
        """Check permissions for skill execution."""
        manifest = skill.manifest

        # Use skill's built-in permission check
        has_perm, missing = await skill.check_permissions(context)
        if not has_perm:
            return False, missing

        # Use RBAC checker if available
        if self._rbac_checker and manifest.required_permissions:
            try:
                for permission in manifest.required_permissions:
                    if not await self._check_rbac_permission(context, permission):
                        return False, permission
            except Exception as e:
                logger.warning(f"RBAC check failed, falling back to context: {e}")
                # Fall back to context permissions
                for permission in manifest.required_permissions:
                    if not context.has_permission(permission):
                        return False, permission

        return True, None

    async def _check_rbac_permission(
        self,
        context: SkillContext,
        permission: str,
    ) -> bool:
        """Check permission using RBAC checker."""
        if not self._rbac_checker:
            return True

        try:
            # Support both sync and async checkers
            check_result = self._rbac_checker.check_permission(
                user_id=context.user_id,
                tenant_id=context.tenant_id,
                permission=permission,
            )
            if asyncio.iscoroutine(check_result):
                return await check_result
            return check_result
        except Exception as e:
            logger.warning(f"RBAC check error: {e}")
            return False

    def _check_rate_limit(
        self,
        skill_name: str,
        manifest: SkillManifest,
    ) -> tuple[bool, float]:
        """
        Check if skill invocation is within rate limit.

        Returns:
            Tuple of (is_allowed, wait_time_seconds)
        """
        if not manifest.rate_limit_per_minute:
            return True, 0.0

        state = self._rate_limits[skill_name]
        current_time = time.time()
        window_duration = 60.0  # 1 minute window

        # Reset window if expired
        if current_time - state.window_start >= window_duration:
            state.window_start = current_time
            state.request_count = 0

        # Check limit
        if state.request_count >= manifest.rate_limit_per_minute:
            wait_time = window_duration - (current_time - state.window_start)
            return False, wait_time

        # Increment counter
        state.request_count += 1
        return True, 0.0

    def _record_metrics(
        self,
        skill_name: str,
        result: SkillResult,
        start_time: float,
    ) -> None:
        """Record execution metrics."""
        if not self._enable_metrics:
            return

        duration_ms = (time.time() - start_time) * 1000
        self._metrics[skill_name].record_execution(
            success=result.success,
            duration_ms=duration_ms,
            error=result.error_message,
        )

    def get_function_schemas(
        self,
        skills: list[str] | None = None,
        debate_compatible_only: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get function schemas for LLM tool calling.

        Args:
            skills: Optional list of skill names to include (all if None)
            debate_compatible_only: Only include debate-compatible skills

        Returns:
            List of function schemas in OpenAI/Anthropic format
        """
        schemas = []
        for name, skill in self._skills.items():
            if skills and name not in skills:
                continue

            manifest = skill.manifest
            if debate_compatible_only and not manifest.debate_compatible:
                continue

            schemas.append(manifest.to_function_schema())

        return schemas

    def get_metrics(self, skill_name: str | None = None) -> dict[str, Any]:
        """
        Get execution metrics.

        Args:
            skill_name: Optional specific skill (all if None)

        Returns:
            Dict of metrics
        """
        if skill_name:
            if skill_name in self._metrics:
                m = self._metrics[skill_name]
                return {
                    "skill": skill_name,
                    "total_invocations": m.total_invocations,
                    "successful": m.successful_invocations,
                    "failed": m.failed_invocations,
                    "success_rate": m.success_rate,
                    "average_execution_time_ms": m.average_execution_time_ms,
                    "last_invocation": m.last_invocation.isoformat() if m.last_invocation else None,
                    "last_error": m.last_error,
                }
            return {}

        return {
            name: {
                "total_invocations": m.total_invocations,
                "successful": m.successful_invocations,
                "failed": m.failed_invocations,
                "success_rate": m.success_rate,
                "average_execution_time_ms": m.average_execution_time_ms,
            }
            for name, m in self._metrics.items()
        }

    def add_hook(
        self,
        hook_type: str,
        callback: Callable,
    ) -> Callable[[], None]:
        """
        Add a hook callback.

        Args:
            hook_type: One of "pre_invoke", "post_invoke", "on_error"
            callback: Function to call

        Returns:
            Unregister function
        """
        if hook_type not in self._hooks:
            raise ValueError(f"Unknown hook type: {hook_type}")

        self._hooks[hook_type].append(callback)

        def unregister():
            if callback in self._hooks[hook_type]:
                self._hooks[hook_type].remove(callback)

        return unregister

    def get_capabilities(self) -> set[SkillCapability]:
        """Get all capabilities across registered skills."""
        capabilities = set()
        for skill in self._skills.values():
            capabilities.update(skill.manifest.capabilities)
        return capabilities

    @property
    def skill_count(self) -> int:
        """Get number of registered skills."""
        return len(self._skills)


# Global registry singleton
_default_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    """Get the default skill registry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = SkillRegistry()
    return _default_registry


def reset_skill_registry() -> None:
    """Reset the default registry (for testing)."""
    global _default_registry
    _default_registry = None
