"""Context building lifecycle helpers for Arena debates.

Extracted from orchestrator.py to reduce its size. These functions handle
PromptContextBuilder and ContextDelegator initialization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aragora.debate.context_delegation import ContextDelegator
from aragora.debate.prompt_context import PromptContextBuilder

if TYPE_CHECKING:
    from aragora.debate.orchestrator import Arena


def init_prompt_context_builder(arena: Arena) -> None:
    """Initialize PromptContextBuilder for agent prompt context.

    Creates the PromptContextBuilder with connections to persona manager,
    flip detector, protocol, prompt builder, audience manager, and spectator.

    Args:
        arena: Arena instance to initialize.
    """
    arena._prompt_context = PromptContextBuilder(
        persona_manager=arena.persona_manager,
        flip_detector=arena.flip_detector,
        protocol=arena.protocol,
        prompt_builder=arena.prompt_builder,
        audience_manager=arena.audience_manager,
        spectator=arena.spectator,
        notify_callback=arena._notify_spectator,
        vertical=getattr(arena, "vertical", None),
        vertical_persona_manager=getattr(arena, "vertical_persona_manager", None),
    )


def init_context_delegator(arena: Arena) -> None:
    """Initialize ContextDelegator for context gathering operations.

    Creates the ContextDelegator with connections to context gatherer,
    memory manager, cache, evidence grounder, continuum memory, and environment.

    Args:
        arena: Arena instance to initialize.
    """
    arena._context_delegator = ContextDelegator(
        context_gatherer=arena.context_gatherer,
        memory_manager=arena.memory_manager,
        cache=arena._cache,
        evidence_grounder=getattr(arena, "evidence_grounder", None),
        continuum_memory=arena.continuum_memory,
        env=arena.env,
        auth_context=getattr(arena, "auth_context", None),
        extract_domain_fn=arena._extract_debate_domain,
    )


__all__ = [
    "init_prompt_context_builder",
    "init_context_delegator",
]
