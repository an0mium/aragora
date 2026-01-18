"""
Specialist Agent Factory.

Creates agents using vertical-specific fine-tuned models.
Provides seamless fallback to base models when specialists aren't available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from aragora.training.specialist_models import (
    Vertical,
    SpecialistModel,
    SpecialistModelRegistry,
    TrainingStatus,
    get_specialist_registry,
)

if TYPE_CHECKING:
    from aragora.agents.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for creating a specialist agent."""

    vertical: Vertical
    org_id: Optional[str] = None
    fallback_to_base: bool = True
    prefer_speed: bool = False  # Prefer faster models over accuracy
    require_specialist: bool = False  # Fail if specialist not available
    extra_context: str = ""  # Additional context for agent prompts
    temperature: float = 0.7
    max_tokens: int = 4096
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpecialistAgentInfo:
    """Information about a created specialist agent."""

    agent: "BaseAgent"
    specialist_model: Optional[SpecialistModel]
    is_specialist: bool
    vertical: Vertical
    model_id: str
    base_model: str
    adapter_name: Optional[str]


class SpecialistAgentFactory:
    """
    Factory for creating agents with vertical-specific fine-tuned models.

    The factory:
    1. Looks up the best specialist model for the requested vertical
    2. Falls back to base models if specialists aren't available
    3. Injects vertical-specific context into agent prompts
    4. Provides consistent agent creation across all verticals
    """

    def __init__(
        self,
        registry: Optional[SpecialistModelRegistry] = None,
    ):
        """
        Initialize the specialist agent factory.

        Args:
            registry: Specialist model registry (uses global if not provided)
        """
        self._registry = registry or get_specialist_registry()

        # Base model mappings for fallback
        self._base_models: Dict[str, str] = {
            "llama-3.3-70b": "openrouter:meta-llama/llama-3.3-70b-instruct",
            "llama-3.1-8b": "openrouter:meta-llama/llama-3.1-8b-instruct",
            "qwen-2.5-72b": "openrouter:qwen/qwen-2.5-72b-instruct",
            "qwen-3-32b": "openrouter:qwen/qwen3-32b",
            "deepseek-v3": "openrouter:deepseek/deepseek-v3",
            "deepseek-r1": "openrouter:deepseek/deepseek-r1",
        }

        # Vertical-specific system prompts
        self._vertical_prompts = self._build_vertical_prompts()

    def _build_vertical_prompts(self) -> Dict[Vertical, str]:
        """Build system prompts for each vertical."""
        return {
            Vertical.LEGAL: """You are a legal analysis specialist with expertise in:
- Contract review and clause analysis
- Regulatory compliance (GDPR, SOX, HIPAA)
- Legal risk assessment
- Obligation and liability identification
Always cite relevant legal frameworks and precedents when applicable.""",

            Vertical.HEALTHCARE: """You are a healthcare analysis specialist with expertise in:
- Clinical documentation review
- HIPAA compliance and PHI handling
- Medical terminology and coding
- Patient safety protocols
Never provide medical advice. Focus on documentation accuracy and compliance.""",

            Vertical.SECURITY: """You are a security analysis specialist with expertise in:
- Vulnerability assessment and threat modeling
- Authentication and authorization patterns
- Secure coding practices (OWASP Top 10)
- Encryption and key management
Always prioritize security over convenience in recommendations.""",

            Vertical.ACCOUNTING: """You are an accounting and audit specialist with expertise in:
- Financial statement analysis
- GAAP/IFRS compliance
- Internal controls assessment
- Tax regulation interpretation
Provide clear audit trails and documentation for all conclusions.""",

            Vertical.REGULATORY: """You are a regulatory compliance specialist with expertise in:
- Multi-jurisdictional compliance frameworks
- Policy development and interpretation
- Risk assessment and mitigation
- Audit preparation and response
Consider both letter and spirit of regulations in analysis.""",

            Vertical.ACADEMIC: """You are an academic research specialist with expertise in:
- Literature review and synthesis
- Citation verification and formatting
- Research methodology evaluation
- Plagiarism and originality assessment
Maintain rigorous academic standards in all analysis.""",

            Vertical.SOFTWARE: """You are a software engineering specialist with expertise in:
- Code architecture and design patterns
- Performance optimization
- API design and integration
- Testing and quality assurance
Balance technical excellence with practical constraints.""",

            Vertical.GENERAL: """You are a knowledgeable AI assistant capable of:
- Multi-domain analysis and synthesis
- Clear and structured communication
- Balanced perspective on complex issues
- Evidence-based reasoning
Adapt your approach based on the specific task requirements.""",
        }

    async def create(
        self,
        config: AgentConfig,
    ) -> SpecialistAgentInfo:
        """
        Create a specialist agent for the given configuration.

        Args:
            config: Agent configuration

        Returns:
            SpecialistAgentInfo with agent and model information
        """
        # Look up specialist model
        specialist = self._registry.get_for_vertical(
            vertical=config.vertical,
            org_id=config.org_id,
            include_global=config.fallback_to_base,
        )

        if specialist and specialist.status == TrainingStatus.READY:
            return await self._create_specialist_agent(specialist, config)
        elif config.require_specialist:
            raise ValueError(
                f"No specialist model available for vertical {config.vertical.value}"
            )
        else:
            return await self._create_base_agent(config)

    async def _create_specialist_agent(
        self,
        specialist: SpecialistModel,
        config: AgentConfig,
    ) -> SpecialistAgentInfo:
        """Create an agent using a specialist model."""
        # Import here to avoid circular dependency
        from aragora.agents.api_agents.tinker import TinkerAgent

        # Get the vertical-specific prompt
        system_prompt = self._get_system_prompt(config)

        # Create agent with specialist adapter
        agent = TinkerAgent(
            base_model=specialist.base_model,
            adapter_name=specialist.adapter_name,
            system_prompt=system_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        logger.info(
            f"Created specialist agent: {specialist.id} "
            f"(vertical={config.vertical.value})"
        )

        return SpecialistAgentInfo(
            agent=agent,
            specialist_model=specialist,
            is_specialist=True,
            vertical=config.vertical,
            model_id=specialist.id,
            base_model=specialist.base_model,
            adapter_name=specialist.adapter_name,
        )

    async def _create_base_agent(
        self,
        config: AgentConfig,
    ) -> SpecialistAgentInfo:
        """Create an agent using a base model (no specialist available)."""
        # Import here to avoid circular dependency
        from aragora.agents.api_agents.openrouter import OpenRouterAgent

        # Select base model based on vertical defaults
        from aragora.training.specialist_models import VERTICAL_DEFAULTS

        defaults = VERTICAL_DEFAULTS.get(config.vertical, {})
        base_model = defaults.get("base_model", "llama-3.3-70b")

        if config.prefer_speed:
            # Use smaller model for speed
            if "70b" in base_model or "72b" in base_model:
                base_model = base_model.replace("70b", "8b").replace("72b", "7b")

        # Get OpenRouter model ID
        openrouter_model = self._base_models.get(
            base_model, "meta-llama/llama-3.3-70b-instruct"
        )

        # Get the vertical-specific prompt
        system_prompt = self._get_system_prompt(config)

        # Create agent
        agent = OpenRouterAgent(
            model=openrouter_model,
            system_prompt=system_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        logger.info(
            f"Created base agent: {openrouter_model} "
            f"(vertical={config.vertical.value}, no specialist available)"
        )

        return SpecialistAgentInfo(
            agent=agent,
            specialist_model=None,
            is_specialist=False,
            vertical=config.vertical,
            model_id=openrouter_model,
            base_model=base_model,
            adapter_name=None,
        )

    def _get_system_prompt(self, config: AgentConfig) -> str:
        """Build the system prompt for a vertical."""
        base_prompt = self._vertical_prompts.get(
            config.vertical,
            self._vertical_prompts[Vertical.GENERAL],
        )

        if config.extra_context:
            return f"{base_prompt}\n\n{config.extra_context}"

        return base_prompt

    # Convenience methods for common verticals

    async def create_legal_agent(
        self,
        org_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SpecialistAgentInfo:
        """Create a legal specialist agent."""
        config = AgentConfig(vertical=Vertical.LEGAL, org_id=org_id, **kwargs)
        return await self.create(config)

    async def create_healthcare_agent(
        self,
        org_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SpecialistAgentInfo:
        """Create a healthcare specialist agent."""
        config = AgentConfig(vertical=Vertical.HEALTHCARE, org_id=org_id, **kwargs)
        return await self.create(config)

    async def create_security_agent(
        self,
        org_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SpecialistAgentInfo:
        """Create a security specialist agent."""
        config = AgentConfig(vertical=Vertical.SECURITY, org_id=org_id, **kwargs)
        return await self.create(config)

    async def create_accounting_agent(
        self,
        org_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SpecialistAgentInfo:
        """Create an accounting specialist agent."""
        config = AgentConfig(vertical=Vertical.ACCOUNTING, org_id=org_id, **kwargs)
        return await self.create(config)

    async def create_regulatory_agent(
        self,
        org_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SpecialistAgentInfo:
        """Create a regulatory specialist agent."""
        config = AgentConfig(vertical=Vertical.REGULATORY, org_id=org_id, **kwargs)
        return await self.create(config)

    async def create_academic_agent(
        self,
        org_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SpecialistAgentInfo:
        """Create an academic specialist agent."""
        config = AgentConfig(vertical=Vertical.ACADEMIC, org_id=org_id, **kwargs)
        return await self.create(config)

    async def create_software_agent(
        self,
        org_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SpecialistAgentInfo:
        """Create a software specialist agent."""
        config = AgentConfig(vertical=Vertical.SOFTWARE, org_id=org_id, **kwargs)
        return await self.create(config)

    async def select_for_task(
        self,
        task: str,
        org_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SpecialistAgentInfo:
        """
        Automatically select the best specialist for a task.

        Analyzes the task description to determine the most appropriate
        vertical, then creates the corresponding specialist agent.

        Args:
            task: Task description
            org_id: Organization ID
            **kwargs: Additional agent configuration

        Returns:
            SpecialistAgentInfo for the best matching vertical
        """
        vertical = self._infer_vertical(task)
        config = AgentConfig(vertical=vertical, org_id=org_id, **kwargs)
        return await self.create(config)

    def _infer_vertical(self, task: str) -> Vertical:
        """Infer the best vertical from a task description."""
        task_lower = task.lower()

        # Keyword matching for verticals
        from aragora.training.specialist_models import VERTICAL_DEFAULTS

        for vertical, defaults in VERTICAL_DEFAULTS.items():
            keywords = defaults.get("keywords", [])
            if any(kw in task_lower for kw in keywords):
                return vertical

        return Vertical.GENERAL

    def list_available_specialists(
        self,
        org_id: Optional[str] = None,
    ) -> Dict[Vertical, List[SpecialistModel]]:
        """
        List available specialists by vertical.

        Args:
            org_id: Organization ID (includes org-specific and global)

        Returns:
            Dictionary mapping verticals to available specialists
        """
        result: Dict[Vertical, List[SpecialistModel]] = {}

        for vertical in Vertical:
            models = self._registry.list_for_vertical(
                vertical=vertical,
                status=TrainingStatus.READY,
            )

            # Filter by org
            if org_id:
                models = [m for m in models if m.org_id is None or m.org_id == org_id]

            if models:
                result[vertical] = models

        return result


# Global factory instance
_specialist_factory: Optional[SpecialistAgentFactory] = None


def get_specialist_factory(
    registry: Optional[SpecialistModelRegistry] = None,
) -> SpecialistAgentFactory:
    """Get or create the global specialist agent factory."""
    global _specialist_factory
    if _specialist_factory is None:
        _specialist_factory = SpecialistAgentFactory(registry)
    return _specialist_factory


__all__ = [
    "AgentConfig",
    "SpecialistAgentInfo",
    "SpecialistAgentFactory",
    "get_specialist_factory",
]
