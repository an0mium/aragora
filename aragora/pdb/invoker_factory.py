"""Factory for building the real :class:`aragora.pdb.real_invoker.RealProviderInvoker`.

Responsibilities:

- Read the environment (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``) and
  optional per-family model overrides (``ARAGORA_PDB_CLAUDE_MODEL``,
  ``ARAGORA_PDB_OPENAI_MODEL``) and instantiate one
  :class:`AnthropicAPIAgent` + one :class:`OpenAIAPIAgent` per process.
- Mark heterodox slot families as **unavailable** so Phase A's two-slot
  scope surfaces cleanly through the existing degrade path.
- **Fail closed** when both core keys are missing — without either core
  slot there is no meaningful heterogeneous brief.

The factory is deliberately thin; rate-limit retries, caching, and
sophisticated model selection live either inside the base agent
classes (resilience.py / rate_limiter.py) or in Phase B layers. See
``aragora.pdb.real_invoker`` for the per-call logic.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from aragora.pdb.panel_config import PDBPanelConfig, load_panel_config
from aragora.pdb.protocol import ProviderInvoker
from aragora.pdb.real_invoker import (
    FAMILY_CLAUDE,
    FAMILY_GPT,
    HETERODOX_FAMILIES,
    RealProviderInvoker,
)

logger = logging.getLogger(__name__)

__all__ = [
    "InvokerFactoryError",
    "build_default_invoker",
    "default_invoker_factory",
    "unavailable_slots_for",
]


# ---------------------------------------------------------------------------
# Env-var names
# ---------------------------------------------------------------------------


ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
OPENAI_KEY_ENV = "OPENAI_API_KEY"
CLAUDE_MODEL_ENV = "ARAGORA_PDB_CLAUDE_MODEL"
OPENAI_MODEL_ENV = "ARAGORA_PDB_OPENAI_MODEL"
CLAUDE_MODEL_DEFAULT = "claude-sonnet-4-6"
OPENAI_MODEL_DEFAULT = "gpt-5.4"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InvokerFactoryError(RuntimeError):
    """Raised when the factory cannot build a viable invoker.

    The HTTP layer catches this and surfaces a 503 with the message so
    the UI can explain which API key is missing rather than fail with
    a generic ``NotImplementedError``.
    """


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def unavailable_slots_for(
    *,
    config: PDBPanelConfig | None = None,
    have_claude: bool,
    have_gpt: bool,
) -> frozenset[str]:
    """Return the set of slot ids that Phase A must mark unavailable.

    Includes:

    - every slot whose family is in :data:`HETERODOX_FAMILIES`
    - if ``have_claude`` is False: every ``claude`` slot
    - if ``have_gpt`` is False: every ``gpt`` slot
    """
    cfg = config if config is not None else load_panel_config()
    unavailable: set[str] = set()
    for slot_id, slot in cfg.slots.items():
        if slot.family in HETERODOX_FAMILIES:
            unavailable.add(slot_id)
        elif slot.family == FAMILY_CLAUDE and not have_claude:
            unavailable.add(slot_id)
        elif slot.family == FAMILY_GPT and not have_gpt:
            unavailable.add(slot_id)
    return frozenset(unavailable)


def build_default_invoker(
    *,
    config: PDBPanelConfig | None = None,
    env: dict[str, str] | None = None,
    anthropic_agent_factory: Callable[[str, str | None], Any] | None = None,
    openai_agent_factory: Callable[[str, str | None], Any] | None = None,
) -> RealProviderInvoker:
    """Construct the process-wide :class:`RealProviderInvoker`.

    Parameters
    ----------
    config:
        Optional panel config. If ``None`` we load the committed
        default via :func:`aragora.pdb.panel_config.load_panel_config`.
    env:
        Optional env mapping — tests pass this so the factory is
        deterministic without touching real ``os.environ``. Defaults to
        ``os.environ``.
    anthropic_agent_factory / openai_agent_factory:
        Optional factories that build the underlying agents. Injected
        by tests so real network-capable agents don't get instantiated.
        Each takes ``(model, api_key)`` and returns any object that
        satisfies the :class:`_AgentLike` contract used by
        :class:`RealProviderInvoker`.

    Raises
    ------
    InvokerFactoryError:
        When neither ``ANTHROPIC_API_KEY`` nor ``OPENAI_API_KEY`` is
        set. Without both core slots a brief would have a single
        model family — the executor's heterogeneity floor would then
        fail closed anyway, so the factory fails closed first with a
        clearer error.
    """
    environment = env if env is not None else dict(os.environ)
    anthropic_key = (environment.get(ANTHROPIC_KEY_ENV) or "").strip() or None
    openai_key = (environment.get(OPENAI_KEY_ENV) or "").strip() or None

    if anthropic_key is None and openai_key is None:
        raise InvokerFactoryError(
            "Cannot build PDB invoker: neither ANTHROPIC_API_KEY nor "
            "OPENAI_API_KEY is set. Mode 3 requires at least one core "
            "model family; set both for heterogeneous cross-verification."
        )

    claude_agent = None
    gpt_agent = None

    if anthropic_key is not None:
        model = (
            environment.get(CLAUDE_MODEL_ENV, CLAUDE_MODEL_DEFAULT).strip() or CLAUDE_MODEL_DEFAULT
        )
        claude_agent = _build_claude_agent(
            model=model,
            api_key=anthropic_key,
            factory=anthropic_agent_factory,
        )
    else:
        logger.info("pdb.invoker_factory: ANTHROPIC_API_KEY unset; claude_core slot unavailable")

    if openai_key is not None:
        model = (
            environment.get(OPENAI_MODEL_ENV, OPENAI_MODEL_DEFAULT).strip() or OPENAI_MODEL_DEFAULT
        )
        gpt_agent = _build_openai_agent(
            model=model,
            api_key=openai_key,
            factory=openai_agent_factory,
        )
    else:
        logger.info("pdb.invoker_factory: OPENAI_API_KEY unset; gpt_core slot unavailable")

    unavailable = unavailable_slots_for(
        config=config,
        have_claude=claude_agent is not None,
        have_gpt=gpt_agent is not None,
    )

    return RealProviderInvoker(
        claude=claude_agent,
        gpt=gpt_agent,
        unavailable_slots=unavailable,
    )


def default_invoker_factory() -> ProviderInvoker:
    """Zero-arg factory suitable for :class:`aragora.pdb.worker.JobRequest`.

    Each call returns a freshly constructed invoker. The underlying
    agent classes cache their own credentials at construction time so
    the overhead per call is small.
    """
    return build_default_invoker()


# ---------------------------------------------------------------------------
# Agent construction — isolated to ease test injection
# ---------------------------------------------------------------------------


def _build_claude_agent(
    *,
    model: str,
    api_key: str,
    factory: Callable[[str, str | None], Any] | None,
) -> Any:
    if factory is not None:
        return factory(model, api_key)
    # Imports live inside the function so test-time monkeypatching and
    # env-less import of ``aragora.pdb.invoker_factory`` stay cheap.
    from aragora.agents.api_agents.anthropic import AnthropicAPIAgent

    return AnthropicAPIAgent(
        name="pdb-claude",
        model=model,
        role="proposer",
        api_key=api_key,
    )


def _build_openai_agent(
    *,
    model: str,
    api_key: str,
    factory: Callable[[str, str | None], Any] | None,
) -> Any:
    if factory is not None:
        return factory(model, api_key)
    from aragora.agents.api_agents.openai import OpenAIAPIAgent

    return OpenAIAPIAgent(
        name="pdb-openai",
        model=model,
        role="proposer",
        api_key=api_key,
    )
