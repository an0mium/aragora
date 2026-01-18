"""
Base Vertical Specialist Agent.

Provides the base class for domain-specific AI agents with
specialized prompts, tools, and compliance checking.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from jinja2 import Template

from aragora.core import Critique, Message
from aragora.agents.api_agents.base import APIAgent
from aragora.verticals.config import (
    ComplianceConfig,
    ComplianceLevel,
    ToolConfig,
    VerticalConfig,
)

logger = logging.getLogger(__name__)


class VerticalSpecialistAgent(APIAgent):
    """
    Base class for vertical specialist agents.

    Extends APIAgent with domain-specific capabilities:
    - Domain-specific system prompts
    - Tool invocation
    - Compliance checking
    - Knowledge mound integration

    Subclasses should implement domain-specific methods.
    """

    def __init__(
        self,
        name: str,
        model: str,
        config: VerticalConfig,
        role: str = "specialist",
        api_key: Optional[str] = None,
        timeout: int = 120,
        **kwargs: Any,
    ):
        super().__init__(
            name=name,
            model=model,
            role=role,
            api_key=api_key,
            timeout=timeout,
            temperature=config.model_config.temperature,
            top_p=config.model_config.top_p,
            **kwargs,
        )

        self._config = config
        self._system_prompt_template = Template(config.system_prompt_template)
        self._tools = {t.name: t for t in config.tools}
        self._compliance_frameworks = config.compliance_frameworks

        # Track tool calls for audit
        self._tool_call_history: List[Dict[str, Any]] = []

    @property
    def vertical_id(self) -> str:
        """Get the vertical identifier."""
        return self._config.vertical_id

    @property
    def config(self) -> VerticalConfig:
        """Get the vertical configuration."""
        return self._config

    @property
    def expertise_areas(self) -> List[str]:
        """Get areas of expertise."""
        return self._config.expertise_areas

    def build_system_prompt(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Build the system prompt from the template.

        Args:
            context: Additional context for template rendering

        Returns:
            Rendered system prompt
        """
        template_context = {
            "vertical": self._config.vertical_id,
            "display_name": self._config.display_name,
            "expertise_areas": self._config.expertise_areas,
            "tools": [t.name for t in self._config.get_enabled_tools()],
            "compliance_frameworks": [
                c.framework for c in self._compliance_frameworks
            ],
            **(context or {}),
        }

        return self._system_prompt_template.render(**template_context)

    def get_tool(self, tool_name: str) -> Optional[ToolConfig]:
        """Get a tool configuration by name."""
        return self._tools.get(tool_name)

    def get_enabled_tools(self) -> List[ToolConfig]:
        """Get list of enabled tools."""
        return [t for t in self._tools.values() if t.enabled]

    async def invoke_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Invoke a domain tool.

        Args:
            tool_name: Name of the tool to invoke
            parameters: Tool parameters

        Returns:
            Tool result

        Raises:
            ValueError: If tool is not found or not enabled
        """
        tool = self._tools.get(tool_name)

        if not tool:
            raise ValueError(f"Tool not found: {tool_name}")

        if not tool.enabled:
            raise ValueError(f"Tool not enabled: {tool_name}")

        # Record for audit
        self._tool_call_history.append({
            "tool": tool_name,
            "parameters": parameters,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })

        # Invoke the actual tool implementation
        result = await self._execute_tool(tool, parameters)

        return result

    @abstractmethod
    async def _execute_tool(
        self,
        tool: ToolConfig,
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute a tool (implemented by subclasses).

        Args:
            tool: Tool configuration
            parameters: Tool parameters

        Returns:
            Tool execution result
        """
        pass

    async def check_compliance(
        self,
        content: str,
        framework: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Check content against compliance frameworks.

        Args:
            content: Content to check
            framework: Specific framework to check (or all if None)

        Returns:
            List of compliance violations
        """
        violations = []

        frameworks = self._compliance_frameworks
        if framework:
            frameworks = [f for f in frameworks if f.framework == framework]

        for fw in frameworks:
            fw_violations = await self._check_framework_compliance(content, fw)
            violations.extend(fw_violations)

        return violations

    @abstractmethod
    async def _check_framework_compliance(
        self,
        content: str,
        framework: ComplianceConfig,
    ) -> List[Dict[str, Any]]:
        """
        Check content against a specific framework (implemented by subclasses).

        Args:
            content: Content to check
            framework: Framework configuration

        Returns:
            List of violations
        """
        pass

    def should_block_on_compliance(
        self,
        violations: List[Dict[str, Any]],
    ) -> bool:
        """
        Determine if output should be blocked based on violations.

        Args:
            violations: List of compliance violations

        Returns:
            True if output should be blocked
        """
        for violation in violations:
            framework_name = violation.get("framework")
            framework = next(
                (f for f in self._compliance_frameworks if f.framework == framework_name),
                None,
            )
            if framework and framework.level == ComplianceLevel.ENFORCED:
                return True

        return False

    async def respond(
        self,
        task: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> Message:
        """
        Generate a response with domain expertise.

        Args:
            task: The task or question
            context: Previous messages in the conversation
            **kwargs: Additional arguments

        Returns:
            Response message
        """
        # Build system prompt with context
        system_prompt = self.build_system_prompt(kwargs.get("prompt_context"))

        # Generate response using parent class
        # The actual implementation depends on the model provider
        # Subclasses should override this to use their specific API

        return await self._generate_response(
            task=task,
            system_prompt=system_prompt,
            context=context,
            **kwargs,
        )

    @abstractmethod
    async def _generate_response(
        self,
        task: str,
        system_prompt: str,
        context: Optional[List[Message]] = None,
        **kwargs: Any,
    ) -> Message:
        """
        Generate a response (implemented by subclasses).

        Args:
            task: The task or question
            system_prompt: System prompt to use
            context: Previous messages
            **kwargs: Additional arguments

        Returns:
            Response message
        """
        pass

    def get_tool_call_history(self) -> List[Dict[str, Any]]:
        """Get history of tool calls for audit."""
        return self._tool_call_history.copy()

    def clear_tool_call_history(self) -> None:
        """Clear tool call history."""
        self._tool_call_history.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary for serialization."""
        return {
            "name": self.name,
            "model": self.model,
            "role": self.role,
            "vertical_id": self.vertical_id,
            "expertise_areas": self.expertise_areas,
            "tools": [t.name for t in self.get_enabled_tools()],
            "compliance_frameworks": [
                c.framework for c in self._compliance_frameworks
            ],
        }

    # Implement required Agent abstract methods

    async def generate(
        self,
        prompt: str,
        context: Optional[List[Message]] = None,
    ) -> str:
        """
        Generate a response to a prompt.

        This is a required method from the Agent ABC.
        For vertical specialists, this wraps the domain-specific response
        generation with compliance checking.

        Args:
            prompt: The prompt to respond to
            context: Previous messages in the conversation

        Returns:
            Generated response text
        """
        # Build system prompt
        system_prompt = self.build_system_prompt()

        # Generate response
        response = await self._generate_response(
            task=prompt,
            system_prompt=system_prompt,
            context=context,
        )

        return response.content

    async def critique(
        self,
        proposal: str,
        task: str,
        context: Optional[List[Message]] = None,
    ) -> Critique:
        """
        Critique a proposal from the perspective of this vertical.

        This is a required method from the Agent ABC.
        Vertical specialists provide domain-specific critiques.

        Args:
            proposal: The proposal to critique
            task: The original task
            context: Previous messages

        Returns:
            Critique with domain-specific issues and suggestions
        """
        # Check for compliance violations
        violations = await self.check_compliance(proposal)

        issues = []
        suggestions = []

        # Convert violations to critique issues
        for v in violations:
            issues.append(f"[{v.get('framework')}] {v.get('message')}")
            if v.get("severity") in ("critical", "high"):
                suggestions.append(f"Address {v.get('rule')} compliance requirement")

        # Calculate severity based on violations
        severity = 0.0
        if violations:
            severity_map = {"critical": 10.0, "high": 7.0, "medium": 4.0, "low": 2.0}
            max_severity = max(
                severity_map.get(v.get("severity", "low"), 2.0)
                for v in violations
            )
            severity = max_severity

        # If no violations, provide a neutral critique
        if not issues:
            issues.append(f"No {self.vertical_id} domain issues identified")
            suggestions.append(f"Consider reviewing against {self.vertical_id} best practices")

        return Critique(
            agent=self.name,
            target_agent="proposal",
            target_content=proposal[:200],
            issues=issues,
            suggestions=suggestions,
            severity=severity,
            reasoning=f"Vertical specialist ({self.vertical_id}) domain review",
        )
