"""
Quickstart CLI command: truthful first-run onboarding in one command.

Guides new users through a short debate:
1. Checks for supported API keys (loads .env if present)
2. Accepts a question via --question or interactive prompt
3. Runs a live debate when keys are available, otherwise falls back to demo
4. Displays verdict, confidence, mode, and elapsed time
5. Saves one deterministic result artifact
6. Optionally opens an HTML view in the browser
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import ssl
import sys
import tempfile
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

_DEFAULT_QUESTION = "Should we adopt microservices or keep our monolith?"
_DEMO_DEFAULT_ROUNDS = 2
_LIVE_DEFAULT_ROUNDS = 1
_LIVE_PRECHECK_TIMEOUT_SECONDS = 3.0
_LIVE_DEBATE_TIMEOUT_SECONDS = 120.0
_LIVE_ROLES: tuple[tuple[str, str], ...] = (
    ("proposer", "proposer"),
    ("critic", "critic"),
    ("synthesizer", "synthesizer"),
)
_LIVE_PROVIDER_PRIORITY: tuple[str, ...] = (
    "openai-api",
    "gemini",
    "anthropic-api",
    "mistral",
    "grok",
    "deepseek",
)
_LIVE_ARENA_KWARGS: dict[str, Any] = {
    "knowledge_mound": None,
    "auto_create_knowledge_mound": False,
    "enable_knowledge_retrieval": False,
    "enable_knowledge_ingestion": False,
    "enable_cross_debate_memory": False,
    "use_rlm_limiter": False,
    "enable_ml_delegation": False,
    "enable_quality_gates": False,
    "enable_consensus_estimation": False,
    "disable_post_debate_pipeline": True,
}
_PROVIDER_TLS_HOSTS: dict[str, str] = {
    "anthropic-api": "api.anthropic.com",
    "openai-api": "api.openai.com",
    "gemini": "generativelanguage.googleapis.com",
    "mistral": "api.mistral.ai",
    "grok": "api.x.ai",
    "deepseek": "openrouter.ai",
}
_PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "agent_type": "anthropic-api",
        "model": "claude-haiku-4-5-20251001",
        "env_vars": ("ANTHROPIC_API_KEY",),
    },
    "openai": {
        "agent_type": "openai-api",
        "model": "gpt-4o-mini",
        "env_vars": ("OPENAI_API_KEY",),
    },
    "gemini": {
        "agent_type": "gemini",
        "model": "gemini-2.0-flash",
        "env_vars": ("GEMINI_API_KEY",),
    },
    "mistral": {
        "agent_type": "mistral",
        "model": None,
        "env_vars": ("MISTRAL_API_KEY",),
    },
    "grok": {
        "agent_type": "grok",
        "model": None,
        "env_vars": ("XAI_API_KEY", "GROK_API_KEY"),
    },
    "openrouter": {
        "agent_type": "deepseek",
        "model": None,
        "env_vars": ("OPENROUTER_API_KEY",),
    },
}
_PROVIDER_ALIASES = {
    "anthropic-api": "anthropic",
    "openai-api": "openai",
    "xai": "grok",
    "deepseek": "openrouter",
}


def add_quickstart_parser(subparsers: Any) -> None:
    """Register the 'quickstart' subcommand."""
    qs_parser = subparsers.add_parser(
        "quickstart",
        help="Guided zero-to-receipt first debate (new user onboarding)",
        description="""
Run your first adversarial debate with a bounded live onboarding path.

Automatically detects available API keys, picks agents, runs a fast
live debate, and opens the decision receipt in your browser.
No configuration needed.

Examples:
  aragora quickstart --demo                              # Zero-config demo
  aragora quickstart --question "Should we use Kubernetes?"
  aragora quickstart --provider openai --api-key sk-... --save-key
  aragora quickstart --question "Migrate to TypeScript?" --output receipt.json
  aragora quickstart --demo --no-browser                 # CI/headless mode
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    qs_parser.add_argument(
        "--question",
        "-q",
        help="The question to debate (uses a default if omitted with --demo)",
    )
    qs_parser.add_argument(
        "--output",
        "-o",
        help="Save receipt to file (supports .json, .md, .html)",
    )
    qs_parser.add_argument(
        "--demo",
        action="store_true",
        help="Use mock agents (no API keys required)",
    )
    qs_parser.add_argument(
        "--provider",
        help=(
            "Live provider to use for quickstart (anthropic, openai, gemini, "
            "mistral, grok, openrouter). Required with --api-key."
        ),
    )
    qs_parser.add_argument(
        "--api-key",
        help="Provider API key to use for this run without pre-configuring env vars",
    )
    qs_parser.add_argument(
        "--save-key",
        action="store_true",
        help="Persist --api-key into the Aragora secure key store",
    )
    qs_parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=None,
        help="Number of debate rounds (default: 2 for demo, 1 for live quickstart)",
    )
    qs_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "md", "html"],
        default="json",
        help="Receipt output format (default: json)",
    )
    qs_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open receipt in browser (for CI/headless environments)",
    )
    qs_parser.set_defaults(func=cmd_quickstart)


def _normalize_provider(provider: str | None) -> str | None:
    """Normalize provider names from CLI input into quickstart keys."""
    if not provider:
        return None
    normalized = provider.strip().lower()
    if not normalized:
        return None
    normalized = _PROVIDER_ALIASES.get(normalized, normalized)
    return normalized if normalized in _PROVIDER_SPECS else None


def _load_dotenv() -> bool:
    """Try to load .env file from cwd or parent. Returns True if loaded."""
    for candidate in [Path.cwd() / ".env", Path.cwd().parent / ".env"]:
        if candidate.is_file():
            try:
                with open(candidate) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip("\"'")
                        if key and key not in os.environ:
                            os.environ[key] = value
                return True
            except OSError:
                pass
    return False


def _detect_agents(preferred_provider: str | None = None) -> list[tuple[str, str | None]]:
    """Detect available agents based on API keys.

    Returns list of (provider, model) tuples.
    """
    agents: list[tuple[str, str | None]] = []

    requested = _normalize_provider(preferred_provider)
    if preferred_provider and requested is None:
        raise ValueError(
            "Unsupported provider. Choose from: anthropic, openai, gemini, "
            "mistral, grok, openrouter."
        )

    for provider_name, spec in _PROVIDER_SPECS.items():
        if requested and provider_name != requested:
            continue
        if any(os.environ.get(env_var) for env_var in spec["env_vars"]):
            agents.append((str(spec["agent_type"]), cast(str | None, spec["model"])))

    return agents


def _configure_inline_api_key(
    provider: str | None,
    api_key: str | None,
    *,
    save_key: bool = False,
) -> tuple[str | None, dict[str, str] | None]:
    """Inject an inline API key into the current process and optionally persist it."""
    normalized_provider = _normalize_provider(provider)
    if not api_key:
        return normalized_provider, None
    if normalized_provider is None:
        raise ValueError(
            "--api-key requires --provider (anthropic, openai, gemini, mistral, grok, openrouter)"
        )

    spec = _PROVIDER_SPECS[normalized_provider]
    primary_env_var = str(spec["env_vars"][0])
    os.environ[primary_env_var] = api_key

    if not save_key:
        return normalized_provider, None

    from aragora.cli.api_keys import set_provider_key

    stored = set_provider_key(normalized_provider, api_key)
    return normalized_provider, {
        "provider": stored.provider,
        "env_var": stored.env_var,
        "backend": stored.backend,
        "masked_value": stored.masked_value,
    }


def _get_question(args: argparse.Namespace) -> str | None:
    """Get the debate question from args, default, or interactive prompt."""
    if args.question:
        return args.question

    # In demo mode, use the default question instead of prompting
    if getattr(args, "demo", False):
        return _DEFAULT_QUESTION

    # Interactive prompt
    try:
        print("\nWhat question should the agents debate?")
        print("(Example: 'Should we migrate from REST to GraphQL?')\n")
        question = input("> ").strip()
        return question if question else None
    except (EOFError, KeyboardInterrupt):
        return None


def _resolve_rounds(requested_rounds: int | None, *, use_demo: bool) -> int:
    """Resolve quickstart rounds with mode-specific defaults."""
    if requested_rounds is not None:
        return requested_rounds
    return _DEMO_DEFAULT_ROUNDS if use_demo else _LIVE_DEFAULT_ROUNDS


def _default_receipt_path(mode: str, fmt: str) -> Path:
    """Return the default saved artifact path for quickstart results."""
    receipts_dir = Path.cwd() / ".aragora" / "receipts"
    suffix = {
        "json": ".json",
        "md": ".md",
        "html": ".html",
    }.get(fmt, ".json")
    normalized_mode = (mode or "demo").strip().lower()
    return receipts_dir / f"quickstart-{normalized_mode}-receipt{suffix}"


def _save_receipt(receipt_data: dict[str, Any], path: str | Path, fmt: str) -> Path:
    """Save receipt to file in the specified format."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    fallback_json = json.dumps(receipt_data, indent=2, default=str)

    if fmt == "json" or suffix == ".json":
        output_path.write_text(fallback_json)
    elif fmt == "md" or suffix == ".md":
        try:
            from aragora.cli.receipt_formatter import receipt_to_markdown

            output_path.write_text(receipt_to_markdown(receipt_data))
        except ImportError as e:
            logger.debug("Receipt markdown formatter unavailable, writing JSON fallback: %s", e)
            output_path.write_text(fallback_json)
    elif fmt == "html" or suffix == ".html":
        try:
            from aragora.cli.receipt_formatter import receipt_to_html

            output_path.write_text(receipt_to_html(receipt_data))
        except ImportError as e:
            logger.debug("Receipt HTML formatter unavailable, writing JSON fallback: %s", e)
            output_path.write_text(fallback_json)
    else:
        output_path.write_text(fallback_json)

    return output_path.resolve()


def _open_receipt_in_browser(
    receipt_data: dict[str, Any], html_path: str | Path | None = None
) -> str | None:
    """Generate HTML receipt and open in browser.

    Returns the path to the saved HTML file, or None on failure.
    """
    try:
        if html_path is not None:
            resolved_path = str(Path(html_path).resolve())
            webbrowser.open(f"file://{resolved_path}")
            return resolved_path

        from aragora.cli.receipt_formatter import receipt_to_html

        html = receipt_to_html(receipt_data)
        # Create a persistent temp file (not auto-deleted)
        fd, path = tempfile.mkstemp(suffix=".html", prefix="aragora-receipt-")
        with os.fdopen(fd, "w") as f:
            f.write(html)
        webbrowser.open(f"file://{path}")
        return path
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.debug("Failed to open receipt in browser: %s", e)
        return None


async def _run_demo_debate(question: str, rounds: int) -> dict[str, Any]:
    """Run a debate with mock agents (no API keys needed)."""
    from aragora_debate.arena import Arena
    from aragora_debate.styled_mock import StyledMockAgent
    from aragora_debate.types import Agent as DebateAgent, DebateConfig

    agents: list[DebateAgent] = [
        StyledMockAgent("analyst", style="supportive"),
        StyledMockAgent("critic", style="critical"),
        StyledMockAgent("synthesizer", style="balanced"),
    ]
    arena = Arena(question=question, agents=agents, config=DebateConfig(rounds=rounds))
    result = await arena.run()

    return {
        "question": question,
        "verdict": result.verdict.value
        if hasattr(result, "verdict") and hasattr(result.verdict, "value")
        else str(result.verdict)
        if hasattr(result, "verdict")
        else "consensus",
        "confidence": result.confidence if hasattr(result, "confidence") else 0.85,
        "rounds": rounds,
        "agents": [a.name for a in agents],
        "summary": result.receipt.to_markdown() if hasattr(result, "receipt") else str(result),
        "dissent": [],
        "mode": "demo",
    }


async def _run_live_debate(
    question: str,
    agents_list: list[tuple[str, str | None]],
    rounds: int,
    *,
    provider: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run a debate with live API agents."""
    from aragora.agents.base import AgentType, create_agent
    from aragora.core import Environment
    from aragora.debate.orchestrator import Arena, DebateProtocol
    from aragora.memory.store import CritiqueStore

    agents_list = await _filter_reachable_live_agents(agents_list)

    team = _build_live_team(agents_list, provider=provider, api_key=api_key)
    if not team:
        raise RuntimeError("No live debate team could be assembled for quickstart")

    env = Environment(task=question)
    protocol = DebateProtocol(
        rounds=rounds,
        consensus="majority",
        convergence_detection=False,
        vote_grouping=False,
        enable_trickster=False,
        enable_research=False,
        enable_trending_injection=False,
        enable_llm_question_classification=False,
        enable_llm_synthesis=False,
    )
    store = CritiqueStore()

    agents = []
    agent_names = []
    for member in team:
        provider_name = str(member["provider"])
        agent = create_agent(
            cast(AgentType, provider_name),
            name=str(member["name"]),
            role=str(member["role"]),
            model=cast(str | None, member.get("model")),
            api_key=cast(str | None, member.get("api_key")),
        )
        agents.append(agent)
        agent_names.append(agent.name)

    # Quickstart is a bounded onboarding lane, not the full debate control plane.
    arena = Arena(env, agents, protocol, insight_store=store, **_LIVE_ARENA_KWARGS)
    arena.enable_introspection = False
    try:
        result = await asyncio.wait_for(arena.run(), timeout=_LIVE_DEBATE_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        raise RuntimeError(
            f"Live debate timed out after {_LIVE_DEBATE_TIMEOUT_SECONDS:.0f}s"
        ) from exc
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        raise RuntimeError(f"Live debate failed before producing a result: {detail}") from exc

    if result is None:
        raise RuntimeError("Live debate returned no result")

    live_receipt = _build_live_receipt(result, question, rounds, team)
    live_receipt["agents"] = agent_names
    return live_receipt


def _build_live_team(
    agents_list: list[tuple[str, str | None]],
    *,
    provider: str | None = None,
    api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Build a quickstart debate team, guaranteeing a real multi-role debate."""
    if not agents_list:
        return []

    normalized_provider = _normalize_provider(provider)
    provider_configs: list[dict[str, Any]] = []
    if normalized_provider:
        provider_configs.append(
            {
                "provider": agents_list[0][0],
                "model": agents_list[0][1],
                "api_key": api_key,
            }
        )
    else:
        selected_provider = next(
            (
                (agent_type, model)
                for preferred in _LIVE_PROVIDER_PRIORITY
                for agent_type, model in agents_list[:4]
                if agent_type == preferred
            ),
            agents_list[0],
        )
        provider_configs.append(
            {
                "provider": selected_provider[0],
                "model": selected_provider[1],
                "api_key": None,
            }
        )

    team: list[dict[str, Any]] = []
    for index, (role, role_label) in enumerate(_LIVE_ROLES):
        provider_cfg = provider_configs[index % len(provider_configs)]
        provider_name = str(provider_cfg["provider"])
        team.append(
            {
                "provider": provider_name,
                "model": provider_cfg.get("model"),
                "api_key": provider_cfg.get("api_key"),
                "role": role,
                "name": f"{provider_name}-{role_label}",
            }
        )

    return team


def _summarize_dissenting_views(
    dissenting_views: list[str], participants: list[str]
) -> list[dict[str, str]]:
    """Convert dissenting views into CLI-friendly agent/reason records."""
    dissent: list[dict[str, str]] = []
    fallback_agents = participants or ["agent"]
    for index, view in enumerate(dissenting_views):
        dissent.append(
            {
                "agent": fallback_agents[index % len(fallback_agents)],
                "reason": str(view),
            }
        )
    return dissent


def _build_live_receipt(
    result: Any,
    question: str,
    rounds: int,
    team: list[dict[str, Any]],
) -> dict[str, Any]:
    """Shape a live debate result into one deterministic receipt payload."""
    from aragora.gauntlet.receipt_models import ConsensusProof, DecisionReceipt, ProvenanceRecord

    participants = list(getattr(result, "participants", []) or [])
    if not participants:
        participants = [str(agent["name"]) for agent in team]

    final_answer = str(getattr(result, "final_answer", "") or "")
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    consensus_reached = bool(getattr(result, "consensus_reached", False))
    verdict = (
        "PASS"
        if consensus_reached and confidence >= 0.75
        else "CONDITIONAL"
        if confidence >= 0.45
        else "FAIL"
    )
    dissenting_views = [str(view) for view in list(getattr(result, "dissenting_views", []) or [])]
    dissent = _summarize_dissenting_views(dissenting_views, participants)
    receipt_id = str(
        getattr(result, "debate_id", "")
        or getattr(result, "id", "")
        or f"quickstart-{uuid.uuid4().hex[:12]}"
    )
    proposals = dict(getattr(result, "proposals", {}) or {})
    timestamp = datetime.now(timezone.utc).isoformat()
    input_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()

    supporting_agents: list[str] = []
    dissenting_agents: list[str] = []
    vote_records: list[dict[str, Any]] = []
    for vote in list(getattr(result, "votes", []) or []):
        vote_agent = str(getattr(vote, "agent", "") or "")
        vote_choice = str(getattr(vote, "choice", "") or "")
        vote_reasoning = str(getattr(vote, "reasoning", "") or "")
        if vote_agent or vote_choice or vote_reasoning:
            vote_records.append(
                {
                    "agent": vote_agent,
                    "choice": vote_choice,
                    "reasoning": vote_reasoning,
                }
            )
        if vote_choice == final_answer and vote_agent:
            supporting_agents.append(vote_agent)
        elif vote_agent:
            dissenting_agents.append(vote_agent)

    if not vote_records and consensus_reached:
        supporting_agents = participants[:]

    rounds_used = int(getattr(result, "rounds_used", 0) or rounds)
    receipt = DecisionReceipt(
        receipt_id=receipt_id,
        gauntlet_id=receipt_id,
        timestamp=timestamp,
        input_summary=question,
        input_hash=input_hash,
        risk_summary={
            "critical": 0 if consensus_reached else int(bool(dissenting_views)),
            "high": len(dissenting_agents),
            "medium": len(dissenting_views),
            "low": max(0, len(participants) - len(dissenting_views)),
        },
        attacks_attempted=rounds_used * max(1, len(participants)),
        attacks_successful=0 if consensus_reached else max(1, len(dissenting_views)),
        probes_run=len(vote_records),
        vulnerabilities_found=len(dissenting_views),
        verdict=verdict,
        confidence=confidence,
        robustness_score=confidence,
        verdict_reasoning=final_answer,
        dissenting_views=dissenting_views,
        consensus_proof=ConsensusProof(
            reached=consensus_reached,
            confidence=confidence,
            supporting_agents=supporting_agents,
            dissenting_agents=dissenting_agents,
            method="majority",
            evidence_hash=input_hash,
        ),
        provenance_chain=[
            ProvenanceRecord(
                timestamp=timestamp,
                event_type="task",
                description=question,
            ),
            ProvenanceRecord(
                timestamp=timestamp,
                event_type="verdict",
                description=final_answer or verdict,
            ),
        ],
        cost_summary={
            "cost_usd": float(getattr(result, "total_cost_usd", 0.0) or 0.0),
            "tokens_used": int(getattr(result, "total_tokens", 0) or 0),
        },
        config_used={
            "mode": "quickstart-live",
            "rounds": rounds_used,
            "participants": participants,
        },
    )

    payload = receipt.to_dict()
    payload.update(
        {
            "question": question,
            "rounds": rounds_used,
            "agents": participants,
            "summary": final_answer,
            "dissent": dissent,
            "mode": "live",
            "receipt": {
                "id": receipt.receipt_id,
                "artifact_hash": receipt.artifact_hash,
                "consensus_reached": consensus_reached,
                "confidence": confidence,
                "participants": participants,
            },
            "proposals": proposals,
            "votes": vote_records,
            "consensus_reached": consensus_reached,
        }
    )
    return payload


async def _can_reach_provider_tls(provider: str) -> tuple[bool, str | None]:
    """Return whether the provider host passes a basic TLS handshake."""
    host = _PROVIDER_TLS_HOSTS.get(provider)
    if not host:
        return True, None

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(
                host,
                443,
                ssl=ssl.create_default_context(),
                server_hostname=host,
            ),
            timeout=_LIVE_PRECHECK_TIMEOUT_SECONDS,
        )
    except ssl.SSLCertVerificationError:
        return False, "CERTIFICATE_VERIFY_FAILED"
    except Exception as exc:
        detail = str(exc).strip() or type(exc).__name__
        return False, detail

    writer.close()
    await writer.wait_closed()
    return True, None


async def _filter_reachable_live_agents(
    agents_list: list[tuple[str, str | None]],
) -> list[tuple[str, str | None]]:
    """Keep only providers that pass a fast TLS preflight.

    Fail closed before entering the debate engine when no detected live providers
    can establish a verified TLS connection.
    """
    limited_agents = agents_list[:4]
    probe_results = await asyncio.gather(
        *(_can_reach_provider_tls(provider) for provider, _ in limited_agents)
    )

    reachable: list[tuple[str, str | None]] = []
    failures: list[str] = []
    certificate_failure = False
    for agent_spec, (ok, detail) in zip(limited_agents, probe_results, strict=False):
        if ok:
            reachable.append(agent_spec)
            continue
        provider = agent_spec[0]
        if detail == "CERTIFICATE_VERIFY_FAILED":
            certificate_failure = True
        failures.append(f"{provider}: {detail}")

    if reachable:
        return reachable

    if certificate_failure:
        providers = ", ".join(provider for provider, _ in limited_agents)
        raise RuntimeError(
            f"Provider TLS verification failed for {providers}. Check the local CA trust store."
        )

    failure_summary = "; ".join(failures) if failures else "no providers available"
    raise RuntimeError(f"No live providers passed connectivity preflight: {failure_summary}")


def cmd_quickstart(args: argparse.Namespace) -> None:
    """Handle the 'quickstart' command."""
    print("\n" + "=" * 60)
    print("  ARAGORA QUICKSTART")
    print("  Zero-to-receipt adversarial debate")
    print("=" * 60)

    # Step 1: Load .env
    loaded = _load_dotenv()
    if loaded:
        print("\n[+] Loaded .env configuration")

    # Step 2: Get question
    question = _get_question(args)
    if not question:
        print("\nNo question provided. Exiting.")
        sys.exit(1)

    print(f"\nQuestion: {question}")

    # Step 3: Detect agents
    use_demo = getattr(args, "demo", False)
    rounds = _resolve_rounds(getattr(args, "rounds", None), use_demo=use_demo)
    provider = getattr(args, "provider", None)
    inline_api_key = getattr(args, "api_key", None)
    try:
        normalized_provider, saved_key = _configure_inline_api_key(
            provider,
            inline_api_key,
            save_key=getattr(args, "save_key", False),
        )
    except (RuntimeError, ValueError) as exc:
        print(f"\n[!] Quickstart configuration failed: {exc}")
        sys.exit(1)

    if saved_key:
        print(
            "\n[+] Saved "
            f"{saved_key['env_var']} to secure store ({saved_key['backend']}) as {saved_key['masked_value']}"
        )

    if use_demo:
        print("\n[*] Run mode: demo (requested with --demo)")
        print("    Agents: analyst (supportive), critic (critical), synthesizer (balanced)")
    else:
        try:
            detected = _detect_agents(normalized_provider)
        except ValueError as exc:
            print(f"\n[!] Quickstart configuration failed: {exc}")
            sys.exit(1)
        if not detected:
            print("\n[!] No supported API keys detected. Falling back to demo mode.")
            print("    This run will use local mock agents, not live model calls.")
            print("    Set ANTHROPIC_API_KEY or OPENAI_API_KEY for live debates.")
            print("    Agents: analyst (supportive), critic (critical), synthesizer (balanced)")
            use_demo = True
        else:
            preview_team = _build_live_team(
                detected[:4],
                provider=normalized_provider,
                api_key=inline_api_key,
            )
            providers = list(dict.fromkeys(str(member["provider"]) for member in preview_team))
            print("\n[+] Run mode: live")
            print(f"    Agents: {', '.join(providers)}")

    print(f"[*] Running {rounds}-round debate...\n")

    # Step 4: Run debate
    start_time = time.monotonic()
    try:
        if use_demo:
            result = asyncio.run(_run_demo_debate(question, rounds))
        else:
            result = asyncio.run(
                _run_live_debate(
                    question,
                    detected[:4],
                    rounds,
                    provider=normalized_provider,
                    api_key=inline_api_key,
                )
            )
    except (OSError, ConnectionError, RuntimeError, ValueError, TypeError) as e:
        logger.debug("Debate failed: %s", e)
        print(f"\n[!] Debate failed: {e}")
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            print("    Provider TLS verification failed. Check the local CA trust store.")
        print("    Try: aragora quickstart --demo")
        sys.exit(1)

    elapsed = time.monotonic() - start_time
    result["elapsed_seconds"] = elapsed

    # Step 5: Display results
    print("=" * 60)
    print("  RESULT")
    print("=" * 60)
    verdict_display = str(result["verdict"]).replace("_", " ").title()
    print(f"\n  Verdict:    {verdict_display}")
    print(f"  Confidence: {result['confidence']:.0%}")
    print(f"  Mode:       {str(result.get('mode', 'demo')).title()}")
    print(f"  Agents:     {', '.join(result['agents'])}")
    print(f"  Rounds:     {result['rounds']}")
    print(f"  Elapsed:    {elapsed:.1f}s")
    if "consensus_proof" in result:
        consensus_text = "Reached" if result["consensus_proof"].get("reached") else "Not reached"
        print(f"  Consensus:  {consensus_text}")
    if result.get("receipt_id"):
        print(f"  Receipt:    {result['receipt_id']}")
    if result.get("artifact_hash"):
        print(f"  Artifact:   {str(result['artifact_hash'])[:16]}...")

    if result.get("summary"):
        print(f"\n  Summary:\n  {result['summary'][:500]}")

    if result.get("dissent"):
        print("\n  Dissent:")
        for d in result["dissent"]:
            print(f"    - {d.get('agent', '?')}: {d.get('reason', 'N/A')}")

    print("\n" + "=" * 60)

    # Step 6: Save receipt
    output_path = getattr(args, "output", None)
    fmt = getattr(args, "format", "json")
    saved_artifact = _save_receipt(
        result,
        output_path or _default_receipt_path(str(result.get("mode", "demo")), fmt),
        fmt,
    )
    artifact_format = saved_artifact.suffix.lstrip(".") or fmt
    print(f"\nResult artifact ({result.get('mode', 'demo')}/{artifact_format}): {saved_artifact}")

    # Step 7: Open receipt in browser
    no_browser = getattr(args, "no_browser", False)
    if not no_browser:
        browser_path = _open_receipt_in_browser(
            result,
            saved_artifact if saved_artifact.suffix.lower() == ".html" else None,
        )
        if browser_path:
            if Path(browser_path) == saved_artifact:
                print("\nOpened saved artifact in browser.")
            else:
                print(f"\nOpened HTML preview in browser: {browser_path}")
        else:
            print("\nCould not open browser. View the saved artifact directly.")

    print("\nNext steps:")
    print("  aragora ask 'Your question' --agents anthropic-api,openai-api  # Full debate")
    print("  aragora decide 'Your question'                                  # Full pipeline")
    print("  aragora doctor                                                  # System health")
