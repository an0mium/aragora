"""
Pydantic v2 request models for Aragora server endpoints.

These models provide strict input validation with clear error messages and are
used alongside (not instead of) the existing JSON-schema-based validation.
The Pydantic layer gives typed field access and richer constraint checking.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, field_validator

from aragora.agents.spec import AgentSpec


AgentSpecInput = str | dict[str, Any]


class DebateRequest(BaseModel):
    """Validated input model for debate creation endpoints.

    Accepts the union of fields accepted by POST /api/v1/debates and
    POST /api/v1/debate-this.

    Example (minimal)::

        req = DebateRequest(question="Should we adopt microservices?")

    Example (full)::

        req = DebateRequest(
            question="Should we adopt microservices?",
            rounds=5,
            agents=["claude", "gpt"],
        )
    """

    question: str = Field(..., min_length=10, max_length=2000)
    rounds: int = Field(default=3, ge=1, le=10)
    agents: list[AgentSpecInput] = Field(default_factory=list, max_length=10)

    # Optional passthrough fields accepted by the existing handler
    auto_select: bool = False
    use_trending: bool = False
    consensus: str = Field(default="majority", max_length=64)
    context: str | None = Field(default=None, max_length=10000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        """Reject blank / whitespace-only questions."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("question cannot be blank")
        return stripped

    @field_validator("agents", mode="before")
    @classmethod
    def parse_agents(cls, v: Any) -> list[AgentSpecInput]:
        """Accept flexible agent spec inputs used by the runtime."""
        if v is None:
            return []
        if isinstance(v, dict):
            return [v]
        if isinstance(v, str):
            stripped = v.strip()
            if not stripped:
                return []
            if stripped.startswith("[") or stripped.startswith("{"):
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    payload = None
                if payload is not None:
                    return cls.parse_agents(payload)
            return [a.strip() for a in stripped.split(",") if a.strip()]
        return v

    @field_validator("agents")
    @classmethod
    def validate_agents(cls, v: list[AgentSpecInput]) -> list[AgentSpecInput]:
        """Validate structured agent specs without narrowing legacy string inputs."""
        if not v:
            return []

        for index, item in enumerate(v):
            if isinstance(item, str):
                continue
            if not isinstance(item, dict):
                raise ValueError(
                    f"agents[{index}] must be a string or object, got {type(item).__name__}"
                )
            try:
                AgentSpec.coerce_list(item, warn=False)
            except ValueError as exc:
                raise ValueError(f"agents[{index}]: {exc}") from exc

        return v

    def to_handler_dict(self) -> dict[str, Any]:
        """Return a dict compatible with the existing debate handler body format."""
        d: dict[str, Any] = {
            "question": self.question,
            "rounds": self.rounds,
            "auto_select": self.auto_select,
            "use_trending": self.use_trending,
            "consensus": self.consensus,
            "metadata": self.metadata,
        }
        if self.agents:
            d["agents"] = self.agents
        if self.context is not None:
            d["context"] = self.context
        return d


def validate_debate_request(body: dict[str, Any]) -> tuple[DebateRequest | None, str | None]:
    """Validate a raw request body dict against :class:`DebateRequest`.

    Args:
        body: Parsed JSON body from the HTTP request.

    Returns:
        A ``(DebateRequest, None)`` tuple on success, or
        ``(None, error_message)`` on validation failure.
    """
    try:
        request = DebateRequest.model_validate(body)
        return request, None
    except Exception as exc:  # noqa: BLE001 – pydantic.ValidationError is the expected type
        # Collect all field errors into a single human-readable string
        try:
            # pydantic v2 ValidationError
            errors = exc.errors()  # type: ignore[union-attr]
            messages = []
            for err in errors:
                loc = " -> ".join(str(p) for p in err.get("loc", []))
                msg = err.get("msg", str(err))
                messages.append(f"{loc}: {msg}" if loc else msg)
            return None, "; ".join(messages)
        except AttributeError:
            return None, str(exc)


__all__ = ["DebateRequest", "validate_debate_request"]
