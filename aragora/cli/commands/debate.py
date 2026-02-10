"""
Debate-related CLI commands and helpers.

Contains the core debate execution logic: agent parsing, debate running,
and the 'ask' command handler.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Literal, cast

from aragora.agents.base import AgentType, create_agent
from aragora.agents.spec import AgentSpec
from aragora.config import (
    DEFAULT_AGENTS,
    DEFAULT_CONSENSUS,
    DEFAULT_ROUNDS,
    DEBATE_TIMEOUT_SECONDS,
    MAX_AGENTS_PER_DEBATE,
)
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
        import urllib.request

        # Quick health check
        with urllib.request.urlopen(f"{server_url}/api/health", timeout=2) as resp:
            status_code = getattr(resp, "status", None) or resp.getcode()
            if status_code == 200:
                # Server is up, try to get emitter
                try:
                    from aragora.server.stream import SyncEventEmitter

                    return SyncEventEmitter()
                except ImportError:
                    pass
    except (OSError, TimeoutError):
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

    return AgentSpec.coerce_list(agents_str, warn=False)


def _split_agents_list(agents_str: str) -> list[str]:
    """Split comma-separated agents string into a clean list."""
    if not agents_str:
        return []
    return [agent.strip() for agent in agents_str.split(",") if agent.strip()]


def _agent_names_for_graph_matrix(agents_str: str) -> list[str]:
    """Resolve agent names for graph/matrix debates (provider-only)."""
    try:
        specs = parse_agents(agents_str)
        return [spec.provider for spec in specs if spec.provider]
    except Exception:
        return _split_agents_list(agents_str)


def _agents_payload_for_api(agents_str: str) -> list[Any]:
    """Build API payload for agents (strings or dicts) from CLI input."""
    try:
        specs = parse_agents(agents_str)
    except Exception:
        return _split_agents_list(agents_str)

    if not specs:
        return []

    advanced = any(
        spec.model or spec.persona or spec.role or spec.name or spec.hierarchy_role
        for spec in specs
    )
    if not advanced:
        return [spec.provider for spec in specs if spec.provider]

    payload: list[dict[str, Any]] = []
    for spec in specs:
        item: dict[str, Any] = {"provider": spec.provider}
        if spec.model:
            item["model"] = spec.model
        if spec.persona:
            item["persona"] = spec.persona
        if spec.role:
            item["role"] = spec.role
        if spec.name:
            item["name"] = spec.name
        if spec.hierarchy_role:
            item["hierarchy_role"] = spec.hierarchy_role
        payload.append(item)
    return payload


def _is_server_available(server_url: str) -> bool:
    """Check if the API server is reachable."""
    try:
        import urllib.request

        with urllib.request.urlopen(f"{server_url}/api/health", timeout=2) as resp:
            status_code = getattr(resp, "status", None) or resp.getcode()
            return status_code == 200
    except (OSError, TimeoutError):
        return False


def _build_api_client(server_url: str, api_key: str | None):
    """Build an AragoraClient for API-backed runs."""
    from aragora.client import AragoraClient

    return AragoraClient(base_url=server_url, api_key=api_key)


def _parse_matrix_scenarios(raw: list[str] | None) -> list[dict[str, Any]]:
    """Parse matrix scenario CLI inputs into structured dicts."""
    scenarios: list[dict[str, Any]] = []
    for item in raw or []:
        value = str(item).strip()
        if not value:
            continue
        if value.startswith("{") or value.startswith("["):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid scenario JSON: {e}")
            if isinstance(parsed, list):
                scenarios.extend([s for s in parsed if isinstance(s, dict)])
            elif isinstance(parsed, dict):
                scenarios.append(parsed)
            else:
                raise ValueError("Scenario JSON must be an object or list of objects")
        else:
            scenarios.append({"name": value})
    return scenarios


def _parse_auto_select_config(raw: str | None) -> dict[str, Any] | None:
    """Parse auto-select config JSON string into a dict."""
    if not raw:
        return None
    value = str(raw).strip()
    if not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid auto-select config JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError("Auto-select config must be a JSON object")
    return parsed


def _auto_select_agents_local(task: str, config: dict[str, Any] | None) -> str | None:
    """Run local auto-selection using server selection logic (best-effort)."""
    try:
        from aragora.server.agent_selection import auto_select_agents

        return auto_select_agents(task, config or {})
    except Exception as e:
        logger.warning("Auto-select failed: %s", e)
        return None


def _maybe_add_vertical_specialist_local(
    task: str,
    agents: list[Any],
    enable_verticals: bool,
    vertical_id: str | None,
) -> list[Any]:
    """Optionally inject a vertical specialist into a local debate run."""
    if not enable_verticals:
        return agents

    try:
        import aragora.verticals.specialists  # noqa: F401
        from aragora.verticals.registry import VerticalRegistry
    except ImportError:
        logger.debug("Verticals registry not available; skipping specialist injection")
        return agents

    resolved_vertical = vertical_id or VerticalRegistry.get_for_task(task)
    if not resolved_vertical:
        logger.debug("No matching vertical found for task; skipping specialist injection")
        return agents

    for agent in agents:
        if getattr(agent, "vertical_id", None) == resolved_vertical:
            return agents

    if len(agents) >= MAX_AGENTS_PER_DEBATE:
        logger.info(
            "Skipping vertical specialist (%s): max agents limit reached (%s)",
            resolved_vertical,
            MAX_AGENTS_PER_DEBATE,
        )
        return agents

    try:
        specialist = VerticalRegistry.create_specialist(
            vertical_id=resolved_vertical,
            name=f"{resolved_vertical}_specialist",
            role="critic",
        )
        try:
            specialist.system_prompt = specialist.build_system_prompt()
        except Exception:
            logger.debug("Failed to build system prompt for specialist %s", resolved_vertical, exc_info=True)
        agents.append(specialist)
        print(f"[verticals] Injected specialist: {resolved_vertical}")
    except Exception as e:
        logger.warning("Failed to create vertical specialist %s: %s", resolved_vertical, e)

    return agents


def _print_debate_result(debate: Any, verbose: bool = False) -> None:
    """Print a standard debate result summary."""
    final_answer = None
    dissenting_agents: list[str] = []
    if getattr(debate, "consensus", None):
        final_answer = debate.consensus.final_answer
        dissenting_agents = debate.consensus.dissenting_agents

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    if final_answer:
        print(final_answer)
    else:
        print(f"Debate completed with status: {getattr(debate, 'status', 'unknown')}")

    if verbose and dissenting_agents:
        print("\n" + "-" * 60)
        print("DISSENTING AGENTS:")
        for agent in dissenting_agents:
            print(f"- {agent}")


def _print_graph_result(debate: Any, verbose: bool = False) -> None:
    """Print a graph debate result summary."""
    branches = getattr(debate, "branches", []) or []
    consensus = getattr(debate, "consensus", None)
    status = getattr(debate, "status", "unknown")
    if hasattr(status, "value"):
        status = status.value

    print("\n" + "=" * 60)
    print("GRAPH DEBATE RESULT:")
    print("=" * 60)
    print(f"Status: {status}")
    if getattr(debate, "branch_count", None) is not None:
        print(f"Branches: {getattr(debate, 'branch_count')}")
    else:
        print(f"Branches: {len(branches)}")
    if getattr(debate, "node_count", None) is not None:
        print(f"Nodes: {getattr(debate, 'node_count')}")

    if consensus and consensus.final_answer:
        print("\n" + "-" * 60)
        print("CONSENSUS:")
        print(consensus.final_answer)

    if verbose and branches:
        print("\n" + "-" * 60)
        print("BRANCHES:")
        for branch in branches:
            if isinstance(branch, dict):
                name = branch.get("name", "")
                nodes = branch.get("nodes", []) or []
                branch_id = branch.get("branch_id") or branch.get("id") or ""
            else:
                name = getattr(branch, "name", "")
                nodes = getattr(branch, "nodes", []) or []
                branch_id = getattr(branch, "branch_id", "")
            node_count = len(nodes)
            print(f"- {name or branch_id} ({node_count} nodes)")


def _print_matrix_result(debate: Any, verbose: bool = False) -> None:
    """Print a matrix debate result summary."""
    scenarios = getattr(debate, "scenarios", None) or getattr(debate, "results", None) or []
    conclusions = getattr(debate, "conclusions", None)
    status = getattr(debate, "status", "unknown")
    if hasattr(status, "value"):
        status = status.value

    print("\n" + "=" * 60)
    print("MATRIX DEBATE RESULT:")
    print("=" * 60)
    print(f"Status: {status}")
    print(f"Scenarios: {len(scenarios)}")

    if conclusions:
        if conclusions.universal:
            print("\n" + "-" * 60)
            print("UNIVERSAL CONCLUSIONS:")
            for item in conclusions.universal:
                print(f"- {item}")
        if conclusions.conditional:
            print("\n" + "-" * 60)
            print("CONDITIONAL CONCLUSIONS:")
            for scenario, items in conclusions.conditional.items():
                print(f"{scenario}:")
                for item in items:
                    print(f"- {item}")
        if conclusions.contradictions:
            print("\n" + "-" * 60)
            print("CONTRADICTIONS:")
            for item in conclusions.contradictions:
                print(f"- {item}")
    else:
        universal = getattr(debate, "universal_conclusions", []) or []
        conditional = getattr(debate, "conditional_conclusions", {}) or {}
        if universal:
            print("\n" + "-" * 60)
            print("UNIVERSAL CONCLUSIONS:")
            for item in universal:
                print(f"- {item}")
        if conditional:
            print("\n" + "-" * 60)
            print("CONDITIONAL CONCLUSIONS:")
            if isinstance(conditional, dict):
                for scenario, items in conditional.items():
                    print(f"{scenario}:")
                    for item in items:
                        print(f"- {item}")
            else:
                for item in conditional:
                    if isinstance(item, dict):
                        condition = item.get("condition") or item.get("scenario")
                        conclusion = item.get("conclusion")
                        confidence = item.get("confidence")
                        if condition:
                            print(f"{condition}:")
                        if conclusion:
                            suffix = (
                                f" (confidence {confidence:.2f})" if confidence is not None else ""
                            )
                            print(f"- {conclusion}{suffix}")
                    else:
                        print(f"- {item}")

    if verbose and scenarios:
        print("\n" + "-" * 60)
        print("SCENARIO RESULTS:")
        for scenario in scenarios:
            if isinstance(scenario, dict):
                name = scenario.get("scenario_name") or scenario.get("name", "")
                key_findings = scenario.get("key_findings") or scenario.get("key_claims") or []
                conclusion = scenario.get("conclusion")
            else:
                name = getattr(scenario, "scenario_name", "")
                key_findings = getattr(scenario, "key_findings", []) or []
                conclusion = getattr(scenario, "consensus", None)
                if hasattr(conclusion, "final_answer"):
                    conclusion = conclusion.final_answer

            print(f"- {name}")
            if conclusion:
                print(f"  Conclusion: {conclusion}")
            if key_findings:
                for finding in key_findings:
                    print(f"  {finding}")


def _print_decision_integrity_summary(package: dict[str, Any]) -> None:
    """Print a decision integrity package summary."""
    receipt = package.get("receipt") or {}
    plan = package.get("plan") or package.get("decision_plan") or {}
    receipt_id = package.get("receipt_id") or receipt.get("receipt_id") or receipt.get("id")
    plan_id = package.get("plan_id") or plan.get("id")
    approval = package.get("approval") or {}
    execution = package.get("execution") or package.get("workflow_execution") or {}

    print("\n" + "=" * 60)
    print("DECISION INTEGRITY PACKAGE")
    print("=" * 60)
    if receipt_id:
        print(f"Receipt ID: {receipt_id}")
    if plan_id:
        print(f"Plan ID: {plan_id}")
    if approval:
        status = approval.get("status") or approval.get("approval_status")
        if status:
            print(f"Approval: {status}")
    if execution:
        status = execution.get("status")
        if status:
            print(f"Execution: {status}")


def _run_debate_api(
    server_url: str,
    api_key: str | None,
    task: str,
    agents: list[Any],
    rounds: int,
    consensus: str,
    context: str | None,
    metadata: dict[str, Any],
    auto_select: bool | None,
    auto_select_config: dict[str, Any] | None,
    enable_verticals: bool,
    vertical_id: str | None,
) -> Any:
    """Run a standard debate via API and wait for completion."""
    client = _build_api_client(server_url, api_key)
    return client.debates.run(
        task=task,
        agents=agents,
        rounds=rounds,
        consensus=consensus,
        timeout=DEBATE_TIMEOUT_SECONDS,
        context=context,
        auto_select=auto_select,
        auto_select_config=auto_select_config,
        enable_verticals=enable_verticals,
        vertical_id=vertical_id,
        metadata=metadata,
    )


def _run_graph_debate_api(
    server_url: str,
    api_key: str | None,
    task: str,
    agents: list[str],
    max_rounds: int,
    branch_threshold: float,
    max_branches: int,
    verbose: bool = False,
) -> Any:
    """Run a graph debate via API and wait for completion."""
    from aragora.client.models import DebateStatus

    client = _build_api_client(server_url, api_key)
    response = client.graph_debates.create(
        task=task,
        agents=agents,
        max_rounds=max_rounds,
        branch_threshold=branch_threshold,
        max_branches=max_branches,
    )
    if getattr(response, "graph", None) or getattr(response, "branches", None):
        return response
    debate_id = response.debate_id

    start = time.time()
    while time.time() - start < DEBATE_TIMEOUT_SECONDS:
        debate = client.graph_debates.get(debate_id)
        if debate.status in (
            DebateStatus.COMPLETED,
            DebateStatus.FAILED,
            DebateStatus.CANCELLED,
        ):
            return debate
        if verbose:
            print(f"[graph] {debate_id} status={debate.status}")
        time.sleep(2)

    raise TimeoutError(f"Graph debate {debate_id} did not complete within timeout")


def _run_matrix_debate_api(
    server_url: str,
    api_key: str | None,
    task: str,
    agents: list[str],
    scenarios: list[dict[str, Any]],
    max_rounds: int,
    verbose: bool = False,
) -> Any:
    """Run a matrix debate via API and wait for completion."""
    from aragora.client.models import DebateStatus

    client = _build_api_client(server_url, api_key)
    response = client.matrix_debates.create(
        task=task,
        agents=agents,
        scenarios=scenarios,
        max_rounds=max_rounds,
    )
    if getattr(response, "results", None) or getattr(response, "universal_conclusions", None):
        return response
    matrix_id = response.matrix_id

    start = time.time()
    while time.time() - start < DEBATE_TIMEOUT_SECONDS:
        debate = client.matrix_debates.get(matrix_id)
        if debate.status in (
            DebateStatus.COMPLETED,
            DebateStatus.FAILED,
            DebateStatus.CANCELLED,
        ):
            return debate
        if verbose:
            print(f"[matrix] {matrix_id} status={debate.status}")
        time.sleep(2)

    raise TimeoutError(f"Matrix debate {matrix_id} did not complete within timeout")


def _build_decision_integrity_api(
    server_url: str,
    api_key: str | None,
    debate_id: str,
    *,
    include_context: bool = False,
    plan_strategy: str = "single_task",
    execution_mode: str | None = None,
) -> dict[str, Any]:
    """Build a decision integrity package via API."""
    client = _build_api_client(server_url, api_key)
    return client.debates.decision_integrity(
        debate_id=debate_id,
        include_context=include_context,
        plan_strategy=plan_strategy,
        execution_mode=execution_mode,
    )


def _build_decision_integrity_local(
    result: Any,
    *,
    include_context: bool = False,
    plan_strategy: str = "single_task",
) -> dict[str, Any]:
    """Build a decision integrity package locally from a DebateResult."""
    from aragora.pipeline.decision_integrity import build_decision_integrity_package

    if hasattr(result, "to_dict"):
        debate_payload = result.to_dict()
    else:
        debate_payload = {
            "debate_id": getattr(result, "debate_id", ""),
            "task": getattr(result, "task", ""),
            "final_answer": getattr(result, "final_answer", ""),
            "confidence": getattr(result, "confidence", 0.0),
            "consensus_reached": getattr(result, "consensus_reached", False),
            "rounds_used": getattr(result, "rounds_used", 0),
            "participants": getattr(result, "participants", []),
        }

    package = asyncio.run(
        build_decision_integrity_package(
            debate_payload,
            include_context=include_context,
            plan_strategy=plan_strategy,
        )
    )
    return package.to_dict()


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
    enable_verticals: bool = False,
    vertical_id: str | None = None,
    auto_select: bool = False,
    auto_select_config: dict[str, Any] | None = None,
    offline: bool = False,
):
    """Run a decision stress-test (debate engine)."""
    from aragora.utils.env import is_offline_mode

    offline = offline or is_offline_mode()
    if offline:
        # Offline mode should be network-free and quiet.
        enable_audience = False
        learn = False

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

    # Auto-select agents if requested and no explicit list provided
    if auto_select:
        if agents_str and agents_str != DEFAULT_AGENTS:
            print("Warning: --auto-select ignores explicit --agents", file=sys.stderr)
        if not agents_str or agents_str == DEFAULT_AGENTS:
            selected = _auto_select_agents_local(task, auto_select_config)
            if selected:
                agents_str = selected
                print(f"[auto-select] Selected agents: {agents_str}")
            else:
                agents_str = DEFAULT_AGENTS

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
                    agent.system_prompt = (
                        f"You are a {spec.persona} in this debate. "
                        f"Approach arguments from that perspective.\n\n{existing}"
                    ).strip()
            except ImportError:
                pass  # Personas module not available

        # Apply mode system prompt if specified (takes precedence)
        if mode_system_prompt:
            agent.system_prompt = mode_system_prompt
        agents.append(agent)

    agents = _maybe_add_vertical_specialist_local(
        task=task,
        agents=agents,
        enable_verticals=enable_verticals,
        vertical_id=vertical_id,
    )

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
    arena_kwargs: dict[str, Any] = {}
    if offline:
        arena_kwargs.update(
            {
                # Disable subsystems that can initialize adapters / embeddings or
                # attempt network calls in local demo/offline runs.
                "knowledge_mound": None,
                "auto_create_knowledge_mound": False,
                "enable_knowledge_retrieval": False,
                "enable_knowledge_ingestion": False,
                "enable_cross_debate_memory": False,
                # Avoid RLM-based compression and related model calls.
                "use_rlm_limiter": False,
                # Disable ML / quality-gate components that may rely on API agents.
                "enable_ml_delegation": False,
                "enable_quality_gates": False,
                "enable_consensus_estimation": False,
            }
        )

    arena = Arena(
        env,
        agents,
        protocol,
        memory=memory,
        event_emitter=event_emitter,
        **arena_kwargs,
    )
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

    # Demo mode forces local execution
    force_local = False
    if getattr(args, "demo", False):
        print("Demo mode enabled - using built-in demo agents.")
        # Demo mode is meant to be network-free; align with the global offline flag
        # so subsystems can short-circuit consistently.
        os.environ.setdefault("ARAGORA_OFFLINE", "1")
        agents = "demo,demo,demo"
        rounds = min(args.rounds, 2)
        learn = False
        enable_audience = False
        force_local = True
        protocol_overrides.update(
            {
                "convergence_detection": False,
                "vote_grouping": False,
                "enable_trickster": False,
                "enable_research": False,
                "enable_rhetorical_observer": False,
                "role_rotation": False,
                "role_matching": False,
                # Keep demo/local mode network-free and deterministic.
                "enable_trending_injection": False,
                "enable_llm_question_classification": False,
                "enable_llm_synthesis": False,
            }
        )

    from aragora.utils.env import is_offline_mode

    offline = is_offline_mode()
    if offline:
        enable_audience = False
        learn = False
        protocol_overrides.update(
            {
                "enable_trending_injection": False,
                "enable_llm_question_classification": False,
                "enable_llm_synthesis": False,
                "enable_research": False,
            }
        )

    server_url = getattr(args, "api_url", DEFAULT_API_URL)
    api_key = (
        getattr(args, "api_key", None)
        or os.environ.get("ARAGORA_API_TOKEN")
        or os.environ.get("ARAGORA_API_KEY")
    )

    requested_api = getattr(args, "api", False)
    requested_local = getattr(args, "local", False)
    graph_mode = getattr(args, "graph", False)
    matrix_mode = getattr(args, "matrix", False)
    decision_integrity = bool(getattr(args, "decision_integrity", False))
    auto_select = bool(getattr(args, "auto_select", False))
    try:
        auto_select_config = _parse_auto_select_config(getattr(args, "auto_select_config", None))
    except ValueError as e:
        print(f"Invalid --auto-select-config: {e}", file=sys.stderr)
        raise SystemExit(2)
    if auto_select_config and not auto_select:
        auto_select = True

    enable_verticals = bool(
        getattr(args, "enable_verticals", False) or getattr(args, "vertical", None)
    )
    vertical_id = getattr(args, "vertical", None)

    if force_local or offline:
        requested_local = True
        requested_api = False

    if graph_mode or matrix_mode:
        if requested_local:
            print("Graph/matrix debates require API mode. Remove --local.", file=sys.stderr)
            raise SystemExit(2)
        requested_api = True
        if auto_select:
            # Use local auto-select to choose a team, then pass to graph/matrix APIs
            selected = _auto_select_agents_local(args.task, auto_select_config)
            if selected:
                agents = selected
            else:
                print(
                    "Auto-select failed; provide --agents for graph/matrix debates.",
                    file=sys.stderr,
                )
                raise SystemExit(2)
    if decision_integrity and (graph_mode or matrix_mode):
        print("Decision integrity is only supported for standard debates.", file=sys.stderr)
        raise SystemExit(2)

    di_include_context = bool(getattr(args, "di_include_context", False))
    di_plan_strategy = getattr(args, "di_plan_strategy", "single_task")
    di_execution_mode = getattr(args, "di_execution_mode", None)

    use_api = requested_api
    if not requested_api and not requested_local:
        use_api = _is_server_available(server_url)

    if use_api:
        try:
            if graph_mode:
                graph_agents = _agent_names_for_graph_matrix(agents)
                result = _run_graph_debate_api(
                    server_url=server_url,
                    api_key=api_key,
                    task=args.task,
                    agents=graph_agents,
                    max_rounds=args.graph_rounds,
                    branch_threshold=args.branch_threshold,
                    max_branches=args.max_branches,
                    verbose=args.verbose,
                )
                _print_graph_result(result, verbose=args.verbose)
                return

            if matrix_mode:
                matrix_agents = _agent_names_for_graph_matrix(agents)
                scenarios = _parse_matrix_scenarios(args.scenario)
                result = _run_matrix_debate_api(
                    server_url=server_url,
                    api_key=api_key,
                    task=args.task,
                    agents=matrix_agents,
                    scenarios=scenarios,
                    max_rounds=args.matrix_rounds,
                    verbose=args.verbose,
                )
                _print_matrix_result(result, verbose=args.verbose)
                return

            if auto_select:
                if agents and agents != DEFAULT_AGENTS:
                    print("Warning: --auto-select ignores explicit --agents", file=sys.stderr)
                agents_payload = []
            else:
                agents_payload = _agents_payload_for_api(agents)

            result = _run_debate_api(
                server_url=server_url,
                api_key=api_key,
                task=args.task,
                agents=agents_payload,
                rounds=rounds,
                consensus=args.consensus,
                context=args.context or None,
                metadata={},
                auto_select=auto_select,
                auto_select_config=auto_select_config,
                enable_verticals=enable_verticals,
                vertical_id=vertical_id,
            )
            _print_debate_result(result, verbose=args.verbose)
            if decision_integrity:
                package = _build_decision_integrity_api(
                    server_url=server_url,
                    api_key=api_key,
                    debate_id=result.debate_id,
                    include_context=di_include_context,
                    plan_strategy=di_plan_strategy,
                    execution_mode=di_execution_mode,
                )
                _print_decision_integrity_summary(package)
            return
        except Exception as e:
            if requested_api or graph_mode or matrix_mode:
                print(f"API run failed: {e}", file=sys.stderr)
                raise SystemExit(1)
            if _is_server_available(server_url):
                print(f"API run failed: {e}", file=sys.stderr)
                raise SystemExit(1)
            print(
                "Warning: API server unavailable, falling back to local execution.",
                file=sys.stderr,
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
            server_url=server_url,
            protocol_overrides=protocol_overrides,
            mode=getattr(args, "mode", None),
            enable_verticals=enable_verticals,
            vertical_id=vertical_id,
            auto_select=auto_select,
            auto_select_config=auto_select_config,
            offline=offline or force_local,
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

    if decision_integrity:
        if di_execution_mode and di_execution_mode != "plan_only":
            print(
                "Decision integrity execution is only supported in API mode. "
                "Generating plan-only package.",
                file=sys.stderr,
            )
        package = _build_decision_integrity_local(
            result,
            include_context=di_include_context,
            plan_strategy=di_plan_strategy,
        )
        _print_decision_integrity_summary(package)
