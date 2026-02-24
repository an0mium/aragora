"""Lightweight agent proxy for pipeline TeamSelector integration.

The core ``Agent`` class in ``aragora.core_types`` is abstract (requires
``generate`` and ``critique`` implementations).  The pipeline's Stage 4
orchestration only needs agent *scoring* via ``TeamSelector.select()``
which accesses ``name``, ``model``, and ``agent_type`` attributes.

``_AgentProxy`` satisfies that interface without pulling in heavy
dependencies or requiring real LLM connections.
"""

from __future__ import annotations


class _AgentProxy:
    """Minimal Agent-like object for TeamSelector scoring.

    TeamSelector accesses:
      - ``agent.name``  (ELO lookup, calibration, domain matching)
      - ``agent.model`` (domain capability matching)
      - ``agent.agent_type`` (domain capability matching)

    This proxy provides those attributes without inheriting from the
    abstract ``Agent`` base class.
    """

    __slots__ = ("name", "model", "agent_type", "role", "stance")

    def __init__(
        self,
        name: str,
        model: str = "unknown",
        agent_type: str = "unknown",
    ) -> None:
        self.name = name
        self.model = model
        self.agent_type = agent_type
        # Defaults expected by some TeamSelector code paths
        self.role = "proposer"
        self.stance = "neutral"

    def __repr__(self) -> str:
        return f"_AgentProxy(name={self.name!r}, model={self.model!r})"
