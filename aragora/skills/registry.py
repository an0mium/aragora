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

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from .base import (
    Skill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
    SkillStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class SkillExecutionMetrics:
    """Metrics for skill executions."""

    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    total_execution_time_ms: float = 0
    average_execution_time_ms: float = 0
    last_invocation: Optional[datetime] = None
    last_error: Optional[str] = None

    def record_execution(self, success: bool, duration_ms: float, error: Optional[str] = None):
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
        rbac_checker: Optional[Any] = None,
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
        self._skills: Dict[str, Skill] = {}
        self._metrics: Dict[str, SkillExecutionMetrics] = defaultdict(SkillExecutionMetrics)
        self._rate_limits: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._rbac_checker = rbac_checker
        self._enable_metrics = enable_metrics
        self._enable_rate_limiting = enable_rate_limiting
        self._hooks: Dict[str, List[Callable]] = {
            "pre_invoke": [],
            "post_invoke": [],
            "on_error": [],
        }
        self._lock = asyncio.Lock()

    def register(self, skill: Skill, replace: bool = False) -> None:
        """
        Register a skill.

        Args:
            skill: The skill to register
            replace: If True, replace existing skill with same name

        Raises:
            ValueError: If skill already registered and replace=False
        """
        name = skill.manifest.name

        if name in self._skills and not replace:
            raise ValueError(f"Skill '{name}' already registered. Use replace=True to override.")

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

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def list_skills(
        self,
        capability: Optional[SkillCapability] = None,
        tag: Optional[str] = None,
        debate_compatible_only: bool = False,
    ) -> List[SkillManifest]:
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
        input_data: Dict[str, Any],
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
                    result = SkillResult.create_failure(
                        f"Rate limit exceeded. Try again in {wait_time:.1f}s",
                        status=SkillStatus.RATE_LIMITED,
                    )
                    self._record_metrics(skill_name, result, start_time)
                    return result

            # Validate input
            is_valid, error_msg = await skill.validate_input(input_data)
            if not is_valid:
                result = SkillResult.create_failure(
                    error_msg or "Invalid input",
                    status=SkillStatus.INVALID_INPUT,
                )
                self._record_metrics(skill_name, result, start_time)
                return result

            # Check permissions
            has_perm, missing_perm = await self._check_permissions(skill, context)
            if not has_perm:
                result = SkillResult.create_permission_denied(missing_perm or "unknown")
                self._record_metrics(skill_name, result, start_time)
                return result

            # Check debate context if required
            if manifest.requires_debate_context and not context.debate_id:
                result = SkillResult.create_failure(
                    "This skill requires an active debate context",
                    error_code="debate_context_required",
                )
                self._record_metrics(skill_name, result, start_time)
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

            # Run error hooks
            for hook in self._hooks["on_error"]:
                try:
                    hook(skill_name, e, context)
                except Exception as hook_error:
                    logger.warning(f"Error hook failed: {hook_error}")

            return result

    async def _check_permissions(
        self,
        skill: Skill,
        context: SkillContext,
    ) -> tuple[bool, Optional[str]]:
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
        skills: Optional[List[str]] = None,
        debate_compatible_only: bool = False,
    ) -> List[Dict[str, Any]]:
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

    def get_metrics(self, skill_name: Optional[str] = None) -> Dict[str, Any]:
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

    def get_capabilities(self) -> Set[SkillCapability]:
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
_default_registry: Optional[SkillRegistry] = None


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
