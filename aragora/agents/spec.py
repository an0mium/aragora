"""
Unified Agent Specification for Aragora.

This module provides a single AgentSpec class that separates four distinct concepts:

- provider: API/service type (e.g., 'anthropic-api', 'qwen', 'deepseek')
- model: Specific model identifier (e.g., 'claude-opus-4-5-20251101')
- persona: Behavioral archetype (e.g., 'philosopher', 'security_engineer')
- role: Debate function (proposer, critic, synthesizer, judge)

RECOMMENDED: Use explicit field creation for clarity and type safety:

    # Single agent
    spec = AgentSpec(provider="anthropic-api", persona="philosopher", role="proposer")

    # Team of agents
    team = AgentSpec.create_team([
        {"provider": "anthropic-api", "persona": "philosopher", "role": "proposer"},
        {"provider": "openai-api", "persona": "skeptic", "role": "critic"},
        {"provider": "gemini", "role": "synthesizer"},
    ])

DEPRECATED: String parsing is maintained for backward compatibility but discouraged:
    AgentSpec.parse("anthropic-api|claude-opus|philosopher|proposer")  # Deprecated
    AgentSpec.parse_list("anthropic-api,openai-api")  # Deprecated
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Sequence

if TYPE_CHECKING:
    from typing import TypedDict

    class AgentSpecDict(TypedDict, total=False):
        provider: str
        model: str | None
        persona: str | None
        role: str
        name: str | None

# Import ALLOWED_AGENT_TYPES for validation
from aragora.config.legacy import ALLOWED_AGENT_TYPES

# Valid debate roles (from core.py)
AgentRole = Literal["proposer", "critic", "synthesizer", "judge"]
VALID_ROLES: frozenset[str] = frozenset({"proposer", "critic", "synthesizer", "judge"})


@dataclass
class AgentSpec:
    """
    Unified specification for creating a debate agent.

    Separates four distinct concepts:
    - provider: Agent type from ALLOWED_AGENT_TYPES (e.g., 'anthropic-api', 'qwen')
    - model: Specific model identifier (optional, uses registry default)
    - persona: Behavioral persona for prompting (optional)
    - role: Debate role - one of 'proposer', 'critic', 'synthesizer', 'judge'

    RECOMMENDED - Explicit field creation:
        # Single agent with explicit fields
        spec = AgentSpec(provider="anthropic-api", persona="philosopher", role="proposer")

        # Team creation with dicts
        team = AgentSpec.create_team([
            {"provider": "anthropic-api", "persona": "philosopher", "role": "proposer"},
            {"provider": "openai-api", "role": "critic"},
        ])

    DEPRECATED - String parsing (for backward compatibility only):
        AgentSpec.parse("anthropic-api|claude-opus|philosopher|proposer")
        AgentSpec.parse_list("anthropic-api,openai-api")
    """

    provider: str
    model: Optional[str] = None
    persona: Optional[str] = None
    role: str = "proposer"  # Default to proposer
    name: Optional[str] = None  # Optional display name

    def __post_init__(self) -> None:
        """Validate the spec after initialization."""
        # Normalize provider to lowercase
        self.provider = self.provider.lower()

        # Validate provider
        if self.provider not in ALLOWED_AGENT_TYPES:
            raise ValueError(
                f"Invalid agent provider: '{self.provider}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_AGENT_TYPES))}"
            )

        # Validate role
        if self.role not in VALID_ROLES:
            raise ValueError(
                f"Invalid agent role: '{self.role}'. " f"Allowed: {', '.join(sorted(VALID_ROLES))}"
            )

        # Generate default name if not provided
        if self.name is None:
            parts = [self.provider]
            if self.persona:
                parts.append(self.persona)
            parts.append(self.role)
            self.name = "_".join(parts)

    @property
    def agent_type(self) -> str:
        """Alias for provider for backward compatibility with legacy code."""
        return self.provider

    @classmethod
    def create_team(
        cls,
        specs: Sequence["AgentSpecDict | AgentSpec"],
        default_role_rotation: bool = True,
    ) -> list["AgentSpec"]:
        """
        Create a team of agents from explicit field specifications.

        This is the RECOMMENDED way to create multiple agents for a debate.

        Args:
            specs: List of AgentSpec instances or dicts with spec fields.
                   Required field: 'provider'
                   Optional fields: 'model', 'persona', 'role', 'name'
            default_role_rotation: If True, assigns roles in rotation
                                   (proposer, critic, synthesizer, judge)
                                   for specs without explicit role.

        Returns:
            List of AgentSpec instances

        Example:
            team = AgentSpec.create_team([
                {"provider": "anthropic-api", "persona": "philosopher"},
                {"provider": "openai-api", "persona": "skeptic"},
                {"provider": "gemini"},
            ])
            # Results in: proposer (anthropic), critic (openai), synthesizer (gemini)
        """
        role_rotation = ["proposer", "critic", "synthesizer", "judge"]
        result = []

        for i, spec in enumerate(specs):
            if isinstance(spec, AgentSpec):
                result.append(spec)
            else:
                # Dict-based creation
                provider = spec.get("provider")
                if not provider:
                    raise ValueError(f"Agent spec at index {i} missing required 'provider' field")

                role = spec.get("role")
                if role is None and default_role_rotation:
                    role = role_rotation[i % len(role_rotation)]
                elif role is None:
                    role = "proposer"

                result.append(cls(
                    provider=provider,
                    model=spec.get("model"),
                    persona=spec.get("persona"),
                    role=role,
                    name=spec.get("name"),
                ))

        return result

    @classmethod
    def parse(cls, spec_str: str, _warn: bool = True) -> "AgentSpec":
        """
        Parse a spec string in either pipe format or legacy colon format.

        DEPRECATED: Prefer explicit field creation instead:
            AgentSpec(provider="anthropic-api", persona="philosopher", role="critic")

        Args:
            spec_str: Agent specification string
            _warn: Internal flag to suppress warning (for parse_list)

        Returns:
            Parsed AgentSpec instance

        Raises:
            ValueError: If the spec is invalid
        """
        if _warn:
            warnings.warn(
                "AgentSpec.parse() is deprecated. Use explicit field creation: "
                "AgentSpec(provider='...', persona='...', role='...')",
                DeprecationWarning,
                stacklevel=2,
            )
        spec_str = spec_str.strip()

        if not spec_str:
            raise ValueError("Empty agent spec string")

        if "|" in spec_str:
            # New pipe-delimited format: provider|model|persona|role
            return cls._parse_pipe_format(spec_str)
        else:
            # Legacy colon format: provider:persona or just provider
            return cls._parse_legacy_format(spec_str)

    @classmethod
    def _parse_pipe_format(cls, spec_str: str) -> "AgentSpec":
        """Parse new pipe-delimited format: provider|model|persona|role."""
        parts = spec_str.split("|")

        provider = parts[0] if parts else ""
        model = parts[1] if len(parts) > 1 and parts[1] else None
        persona = parts[2] if len(parts) > 2 and parts[2] else None
        role = parts[3] if len(parts) > 3 and parts[3] else "proposer"

        return cls(provider=provider, model=model, persona=persona, role=role)

    @classmethod
    def _parse_legacy_format(cls, spec_str: str) -> "AgentSpec":
        """
        Parse legacy colon format: provider:persona or just provider.

        Note: The second part is interpreted as PERSONA, not role.
        This was the actual behavior of question_classifier.py, even though
        parsers incorrectly treated it as role.
        """
        if ":" in spec_str:
            # Split only on first colon to handle edge cases
            parts = spec_str.split(":", 1)
            provider = parts[0]
            persona = parts[1] if len(parts) > 1 else None
        else:
            # Just provider name
            provider = spec_str
            persona = None

        # Legacy format defaults to proposer role
        return cls(provider=provider, persona=persona, role="proposer")

    @classmethod
    def parse_list(cls, specs_str: str, _warn: bool = True) -> list["AgentSpec"]:
        """
        Parse a comma-separated string of agent specs.

        DEPRECATED: Prefer AgentSpec.create_team() with explicit fields:
            AgentSpec.create_team([
                {"provider": "anthropic-api", "persona": "philosopher"},
                {"provider": "qwen", "role": "critic"},
            ])

        Args:
            specs_str: Comma-separated agent specs
            _warn: Internal flag to suppress warning

        Returns:
            List of AgentSpec instances
        """
        if _warn:
            warnings.warn(
                "AgentSpec.parse_list() is deprecated. Use AgentSpec.create_team() "
                "with explicit field dicts instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if not specs_str:
            return []

        specs = []
        for spec in specs_str.split(","):
            spec = spec.strip()
            if spec:
                specs.append(cls.parse(spec, _warn=False))

        return specs

    def to_string(self) -> str:
        """
        Serialize to new pipe-delimited format.

        Returns:
            Pipe-delimited spec string
        """
        return f"{self.provider}|{self.model or ''}|{self.persona or ''}|{self.role}"

    def to_legacy_string(self) -> str:
        """
        Serialize to legacy colon format (for backward compatibility).

        Returns:
            Colon-delimited spec string (provider:persona or just provider)
        """
        if self.persona:
            return f"{self.provider}:{self.persona}"
        return self.provider

    def with_role(self, role: str) -> "AgentSpec":
        """
        Create a new AgentSpec with a different role.

        Args:
            role: New role to assign

        Returns:
            New AgentSpec with the specified role
        """
        return AgentSpec(
            provider=self.provider,
            model=self.model,
            persona=self.persona,
            role=role,
            name=None,  # Regenerate name
        )

    def with_persona(self, persona: str) -> "AgentSpec":
        """
        Create a new AgentSpec with a different persona.

        Args:
            persona: New persona to assign

        Returns:
            New AgentSpec with the specified persona
        """
        return AgentSpec(
            provider=self.provider,
            model=self.model,
            persona=persona,
            role=self.role,
            name=None,  # Regenerate name
        )

    def __repr__(self) -> str:
        """Readable representation."""
        parts = [f"provider={self.provider!r}"]
        if self.model:
            parts.append(f"model={self.model!r}")
        if self.persona:
            parts.append(f"persona={self.persona!r}")
        parts.append(f"role={self.role!r}")
        return f"AgentSpec({', '.join(parts)})"


def parse_agents(agents_str: str) -> list[AgentSpec]:
    """
    Convenience function to parse agent specs.

    This function is provided for backward compatibility with existing code
    that imports a parse_agents function.

    Args:
        agents_str: Comma-separated agent specs

    Returns:
        List of AgentSpec instances
    """
    return AgentSpec.parse_list(agents_str)


__all__ = [
    "AgentSpec",
    "AgentRole",
    "VALID_ROLES",
    "parse_agents",
]
