"""
Aragora Skills System.

A modular, pluggable skills architecture inspired by ClawdBot for extending
agent capabilities with standardized interfaces.

Key components:
- Skill: Abstract base class for skill implementations
- SkillManifest: Declarative skill metadata and requirements
- SkillResult: Typed result container with status and metadata
- SkillContext: Execution context with permissions and state
- SkillRegistry: Central registry for skill discovery and invocation
- SkillLoader: Dynamic skill loading from various sources

Basic usage:
    from aragora.skills import (
        Skill,
        SkillManifest,
        SkillResult,
        SkillContext,
        SkillCapability,
        get_skill_registry,
    )

    # Define a skill
    class MySkill(Skill):
        @property
        def manifest(self) -> SkillManifest:
            return SkillManifest(
                name="my_skill",
                version="1.0.0",
                capabilities=[SkillCapability.EXTERNAL_API],
                input_schema={"query": {"type": "string", "required": True}},
            )

        async def execute(self, input_data: dict, context: SkillContext) -> SkillResult:
            result = await self._do_work(input_data["query"])
            return SkillResult.create_success(result)

    # Register and invoke
    registry = get_skill_registry()
    registry.register(MySkill())

    result = await registry.invoke(
        "my_skill",
        {"query": "test"},
        SkillContext(user_id="user123"),
    )

For LLM function calling:
    # Get schemas for tool use
    schemas = registry.get_function_schemas()

    # Use in LLM prompt
    response = await llm.complete(
        prompt="...",
        tools=schemas,
    )
"""

from .base import (
    Skill,
    SyncSkill,
    SkillCapability,
    SkillContext,
    SkillManifest,
    SkillResult,
    SkillStatus,
)
from .registry import (
    SkillRegistry,
    SkillExecutionMetrics,
    get_skill_registry,
    reset_skill_registry,
)
from .loader import (
    SkillLoader,
    SkillLoadError,
    DeclarativeSkill,
    load_skills,
)

__all__ = [
    # Base types
    "Skill",
    "SyncSkill",
    "SkillCapability",
    "SkillContext",
    "SkillManifest",
    "SkillResult",
    "SkillStatus",
    # Registry
    "SkillRegistry",
    "SkillExecutionMetrics",
    "get_skill_registry",
    "reset_skill_registry",
    # Loader
    "SkillLoader",
    "SkillLoadError",
    "DeclarativeSkill",
    "load_skills",
]
