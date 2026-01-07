"""
Base class for API-based agents.
"""

from aragora.agents.base import CritiqueMixin
from aragora.core import Agent, Message


class APIAgent(CritiqueMixin, Agent):
    """Base class for API-based agents."""

    def __init__(
        self,
        name: str,
        model: str,
        role: str = "proposer",
        timeout: int = 120,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        super().__init__(name, model, role)
        self.timeout = timeout
        self.api_key = api_key
        self.base_url = base_url
        self.agent_type = "api"  # Default for API agents

    def _build_context_prompt(self, context: list[Message] | None = None) -> str:
        """Build context from previous messages.

        Delegates to CritiqueMixin (no truncation for API agents).
        """
        return CritiqueMixin._build_context_prompt(self, context, truncate=False)

    # _parse_critique is inherited from CritiqueMixin


__all__ = ["APIAgent"]
