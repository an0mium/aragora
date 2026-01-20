"""
Unified Agent Specification for Aragora.

This module provides a single AgentSpec class that separates four distinct concepts
that were previously conflated through colon-separated format:

- provider: API/service type (e.g., 'anthropic-api', 'qwen', 'deepseek')
- model: Specific model identifier (e.g., 'claude-opus-4-5-20251101')
- persona: Behavioral archetype (e.g., 'philosopher', 'security_engineer')
- role: Debate function (proposer, critic, synthesizer, judge)

Supports two formats:
- New format: provider|model|persona|role (unambiguous, recommended)
- Legacy format: provider:persona (backward compatible)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    pass

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

    Example specs:
        # New format (unambiguous)
        "anthropic-api|claude-opus|philosopher|proposer"  # Full spec
        "anthropic-api|||proposer"                        # Provider + role only
        "qwen||qwen|critic"                               # Provider + persona + role

        # Legacy format (backward compatible)
        "anthropic-api:claude"    # provider=anthropic-api, persona=claude
        "qwen"                    # provider=qwen
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
    def parse(cls, spec_str: str) -> "AgentSpec":
        """
        Parse a spec string in either new pipe format or legacy colon format.

        Args:
            spec_str: Agent specification string

        Returns:
            Parsed AgentSpec instance

        Raises:
            ValueError: If the spec is invalid

        Examples:
            # New format
            AgentSpec.parse("anthropic-api|claude-opus|philosopher|critic")
            AgentSpec.parse("qwen|||proposer")

            # Legacy format
            AgentSpec.parse("anthropic-api:claude")
            AgentSpec.parse("qwen")
        """
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
    def parse_list(cls, specs_str: str) -> list["AgentSpec"]:
        """
        Parse a comma-separated string of agent specs.

        Args:
            specs_str: Comma-separated agent specs

        Returns:
            List of AgentSpec instances

        Example:
            specs = AgentSpec.parse_list("anthropic-api|||,qwen||qwen|critic")
        """
        if not specs_str:
            return []

        specs = []
        for spec in specs_str.split(","):
            spec = spec.strip()
            if spec:
                specs.append(cls.parse(spec))

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
