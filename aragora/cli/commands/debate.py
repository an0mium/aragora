"""
Debate-related CLI commands and helpers.

Contains the core debate execution logic: agent parsing, debate running,
and the 'ask' command handler.
"""

import argparse
import asyncio
import logging
import os
from typing import Any, Literal, cast

from aragora.agents.base import AgentType, create_agent
from aragora.agents.spec import AgentSpec
from aragora.config import DEFAULT_CONSENSUS, DEFAULT_ROUNDS
from aragora.core import Environment
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.memory.store import CritiqueStore
from aragora.modes import ModeRegistry

logger = logging.getLogger(__name__)

# Default API URL from environment or localhost fallback
DEFAULT_API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")


def get_event_emitter_if_available(server_url: str = DEFAULT_API_URL) -> Any | None:
    """
    Try to connect to the streaming server for audience participation.
    Returns event emitter if server is available, None otherwise.
    """
    try:
        import httpx

        # Quick health check
        resp = httpx.get(f"{server_url}/api/health", timeout=2)
        if resp.status_code == 200:
            # Server is up, try to get emitter
            try:
                from aragora.server.stream import SyncEventEmitter

                return SyncEventEmitter()
            except ImportError:
                pass
    except (httpx.HTTPError, OSError, TimeoutError):
        # Server not available - network error, timeout, or connection refused
        pass
    return None


def parse_agents(agents_str: str) -> list[AgentSpec]:
    """Parse agent string using unified AgentSpec.

    Supports both formats:
    - New pipe format: provider|model|persona|role (explicit fields)
    - Legacy colon format: provider:role or provider:persona

    Args:
        agents_str: Comma-separated agent specs

    Returns:
        List of AgentSpec objects with all parsed fields
    """
    from aragora.agents.spec import AgentSpec

    return AgentSpec.parse_list(agents_str)


async def run_debate(
    task: str,
    agents_str: str,
    rounds: int = DEFAULT_ROUNDS,  # 9-round format (0-8) default
    consensus: str = DEFAULT_CONSENSUS,  # Judge-based consensus default
    context: str = "",
    learn: bool = True,
    db_path: str = "agora_memory.db",
    enable_audience: bool = True,
    server_url: str = DEFAULT_API_URL,
    protocol_overrides: dict[str, Any] | None = None,
    mode: str | None = None,
):
    """Run a decision stress-test (debate engine)."""

    # Get mode system prompt if specified
    mode_system_prompt = ""
    if mode:
        mode_obj = ModeRegistry.get(mode)
        if mode_obj:
            mode_system_prompt = mode_obj.get_system_prompt()
            print(f"[mode] Using '{mode}' mode - {mode_obj.description}")
        else:
            available = ", ".join(ModeRegistry.list_all())
            print(f"[mode] Warning: Mode '{mode}' not found. Available: {available}")

    # Parse and create agents
    agent_specs = parse_agents(agents_str)

    # Assign default roles based on position if not explicitly specified
    agents = []
    for i, spec in enumerate(agent_specs):
        role = spec.role
        # If role is None (not explicitly specified), assign based on position
        # This ensures diverse debate roles: proposer, critic(s), synthesizer
        if role is None:
            if i == 0:
                role = "proposer"
            elif i == len(agent_specs) - 1 and len(agent_specs) > 1:
                role = "synthesizer"
            else:
                role = "critic"

        agent = create_agent(
            model_type=cast(AgentType, spec.provider),
            name=spec.name or f"{spec.provider}_{role}",
            role=role,
            model=spec.model,  # Pass model from spec
        )

        # Apply persona as system prompt if specified
        if spec.persona:
            try:
                from aragora.agents.personas import DEFAULT_PERSONAS

                if spec.persona in DEFAULT_PERSONAS:
                    p = DEFAULT_PERSONAS[spec.persona]
                    traits_str = ", ".join(p.traits) if p.traits else "analytical"
                    persona_prompt = f"You are a {traits_str} agent. {p.description}"
                    if p.top_expertise:
                        top_domains = [d for d, _ in p.top_expertise]
                        persona_prompt += f" Your key areas of expertise: {', '.join(top_domains)}."
                    existing = getattr(agent, "system_prompt", "") or ""
                    agent.system_prompt = f"{persona_prompt}\n\n{existing}".strip()

                    # Apply generation parameters from persona
                    if hasattr(agent, "set_generation_params"):
                        agent.set_generation_params(
                            temperature=p.temperature,
                            top_p=p.top_p,
                            frequency_penalty=p.frequency_penalty,
                        )
                else:
                    # Use persona name as a behavioral hint
                    existing = getattr(agent, "system_prompt", "") or ""
                    agent.system_prompt = f"You are a {spec.persona} in this debate. Approach arguments from that perspective.\n\n{existing}".strip()
            except ImportError:
                pass  # Personas module not available

        # Apply mode system prompt if specified (takes precedence)
        if mode_system_prompt:
            agent.system_prompt = mode_system_prompt
        agents.append(agent)

    # Create environment
    env = Environment(
        task=task,
        context=context,
        max_rounds=rounds,
    )

    # Create protocol
    consensus_type = cast(
        Literal[
            "majority",
            "unanimous",
            "judge",
            "none",
            "weighted",
            "supermajority",
            "any",
            "byzantine",
        ],
        consensus,
    )
    protocol = DebateProtocol(
        rounds=rounds,
        consensus=consensus_type,
        **(protocol_overrides or {}),
    )

    # Create memory store
    memory = CritiqueStore(db_path) if learn else None

    # Try to get event emitter for audience participation
    event_emitter = None
    if enable_audience:
        event_emitter = get_event_emitter_if_available(server_url)
        if event_emitter:
            print("[audience] Connected to streaming server - audience participation enabled")

    # Run debate
    arena = Arena(env, agents, protocol, memory=memory, event_emitter=event_emitter)
    result = await arena.run()

    # Store result
    if memory:
        memory.store_debate(result)

    return result


def cmd_ask(args: argparse.Namespace) -> None:
    """Handle 'ask' command."""
    agents = args.agents
    rounds = args.rounds
    learn = args.learn
    enable_audience = True
    protocol_overrides: dict[str, Any] = {}

    # Apply cross-pollination feature flags
    if not getattr(args, "calibration", True):
        protocol_overrides["enable_calibration"] = False
    if not getattr(args, "evidence_weighting", True):
        protocol_overrides["enable_evidence_weighting"] = False
    if not getattr(args, "trending", True):
        protocol_overrides["enable_trending_injection"] = False
    # Note: ELO weighting is controlled via WeightCalculatorConfig, passed via protocol

    if getattr(args, "demo", False):
        print("Demo mode enabled - using built-in demo agents.")
        agents = "demo,demo,demo"
        rounds = min(args.rounds, 2)
        learn = False
        enable_audience = False
        protocol_overrides.update(
            {
                "convergence_detection": False,
                "vote_grouping": False,
                "enable_trickster": False,
                "enable_research": False,
                "enable_rhetorical_observer": False,
                "role_rotation": False,
                "role_matching": False,
            }
        )

    result = asyncio.run(
        run_debate(
            task=args.task,
            agents_str=agents,
            rounds=rounds,
            consensus=args.consensus,
            context=args.context or "",
            learn=learn,
            db_path=args.db,
            enable_audience=enable_audience,
            protocol_overrides=protocol_overrides,
            mode=getattr(args, "mode", None),
        )
    )

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result.final_answer)

    if result.dissenting_views and args.verbose:
        print("\n" + "-" * 60)
        print("DISSENTING VIEWS:")
        for view in result.dissenting_views:
            print(f"\n{view}")
