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
import json
import logging
import os
import sys
import tempfile
import time
import webbrowser
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

_DEFAULT_QUESTION = "Should we adopt microservices or keep our monolith?"
_LIVE_ROLES: tuple[tuple[str, str], ...] = (
    ("proposer", "proposer"),
    ("critic", "critic"),
    ("synthesizer", "synthesizer"),
)
_PROVIDER_SPECS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "agent_type": "anthropic-api",
        "model": "claude-sonnet-4-5-20250929",
        "env_vars": ("ANTHROPIC_API_KEY",),
    },
    "openai": {
        "agent_type": "openai-api",
        "model": "gpt-4o",
        "env_vars": ("OPENAI_API_KEY",),
    },
    "gemini": {
        "agent_type": "gemini",
        "model": None,
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
Run your first adversarial debate in under 60 seconds.

Automatically detects available API keys, picks agents, runs a fast
2-round debate, and opens the decision receipt in your browser.
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
        help="Persist --api-key into .env in the current directory",
    )
    qs_parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=2,
        help="Number of debate rounds (default: 2)",
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
) -> tuple[str | None, Path | None]:
    """Inject an inline API key into the current process and optionally persist it."""
    if not api_key:
        return _normalize_provider(provider), None

    normalized_provider = _normalize_provider(provider)
    if normalized_provider is None:
        raise ValueError(
            "--api-key requires --provider (anthropic, openai, gemini, mistral, grok, openrouter)"
        )

    spec = _PROVIDER_SPECS[normalized_provider]
    primary_env_var = spec["env_vars"][0]
    os.environ[str(primary_env_var)] = api_key

    saved_path: Path | None = None
    if save_key:
        saved_path = _persist_api_key(normalized_provider, api_key)

    return normalized_provider, saved_path


def _persist_api_key(provider: str, api_key: str) -> Path:
    """Upsert one provider API key into the local .env file."""
    normalized_provider = _normalize_provider(provider)
    if normalized_provider is None:
        raise ValueError("Unsupported provider for .env persistence")

    spec = _PROVIDER_SPECS[normalized_provider]
    target_path = Path.cwd() / ".env"
    existing_lines = target_path.read_text().splitlines() if target_path.exists() else []
    primary_env_var = str(spec["env_vars"][0])
    relevant_env_vars = tuple(str(env_var) for env_var in spec["env_vars"])

    replaced = False
    updated_lines: list[str] = []
    for line in existing_lines:
        stripped = line.strip()
        if any(stripped.startswith(f"{env_var}=") for env_var in relevant_env_vars):
            if not replaced:
                updated_lines.append(f"{primary_env_var}={api_key}")
                replaced = True
            continue
        updated_lines.append(line)

    if not replaced:
        if updated_lines and updated_lines[-1].strip():
            updated_lines.append("")
        updated_lines.append(f"{primary_env_var}={api_key}")

    target_path.write_text("\n".join(updated_lines) + "\n")
    return target_path.resolve()


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
        for agent_type, model in agents_list[:4]:
            provider_configs.append({"provider": agent_type, "model": model, "api_key": None})

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
    participants = list(getattr(result, "participants", []) or [])
    if not participants:
        participants = [str(agent["name"]) for agent in team]

    final_answer = str(getattr(result, "final_answer", "") or "")
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    consensus_reached = bool(getattr(result, "consensus_reached", False))
    verdict = "consensus" if consensus_reached else "no_consensus"
    dissenting_views = [str(view) for view in list(getattr(result, "dissenting_views", []) or [])]
    dissent = _summarize_dissenting_views(dissenting_views, participants)
    receipt_id = str(getattr(result, "debate_id", "") or getattr(result, "id", "") or "")
    proposals = dict(getattr(result, "proposals", {}) or {})

    supporting_agents: list[str] = []
    dissenting_agents: list[str] = []
    vote_records: list[dict[str, str]] = []
    for vote in list(getattr(result, "votes", []) or []):
        voter = str(getattr(vote, "agent", "") or getattr(vote, "voter", "") or "agent")
        choice = str(getattr(vote, "choice", "") or "")
        reasoning = str(getattr(vote, "reasoning", "") or "")
        vote_records.append({"agent": voter, "choice": choice, "reasoning": reasoning})
        if final_answer and choice == final_answer:
            supporting_agents.append(voter)
        elif voter:
            dissenting_agents.append(voter)

    if not vote_records and consensus_reached:
        supporting_agents = participants[:]

    return {
        "question": question,
        "verdict": verdict,
        "confidence": confidence,
        "rounds": int(getattr(result, "rounds_used", 0) or rounds),
        "agents": participants,
        "summary": final_answer,
        "dissent": dissent,
        "dissenting_views": dissenting_views,
        "mode": "live",
        "receipt_id": receipt_id,
        "receipt": {
            "id": receipt_id,
            "consensus_reached": consensus_reached,
            "confidence": confidence,
            "participants": participants,
        },
        "consensus_proof": {
            "reached": consensus_reached,
            "method": "majority",
            "confidence": confidence,
            "supporting_agents": supporting_agents,
            "dissenting_agents": dissenting_agents,
        },
        "proposals": proposals,
        "votes": vote_records,
    }


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
    team: list[dict[str, Any]],
    rounds: int,
) -> dict[str, Any]:
    """Run a debate with live API agents."""
    from aragora.agents.base import AgentType, create_agent
    from aragora.core import Environment
    from aragora.debate.orchestrator import Arena, DebateProtocol
    from aragora.memory.store import CritiqueStore

    env = Environment(task=question)
    protocol = DebateProtocol(rounds=rounds, consensus="majority")
    store = CritiqueStore()

    agents = []
    for agent_cfg in team[:4]:
        agent = create_agent(
            cast(AgentType, str(agent_cfg["provider"])),
            name=str(agent_cfg["name"]),
            role=str(agent_cfg["role"]),
            model=cast(str | None, agent_cfg.get("model")),
            api_key=cast(str | None, agent_cfg.get("api_key")),
        )
        agents.append(agent)

    arena = Arena(env, agents, protocol, insight_store=store)
    result = await arena.run()
    return _build_live_receipt(result, question, rounds, team)


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

    requested_provider_raw = getattr(args, "provider", None)
    requested_provider = _normalize_provider(requested_provider_raw)
    inline_api_key = getattr(args, "api_key", None)
    save_key = bool(getattr(args, "save_key", False))

    try:
        requested_provider, saved_key_path = _configure_inline_api_key(
            requested_provider_raw,
            inline_api_key,
            save_key=save_key,
        )
    except ValueError as e:
        print(f"\n[!] {e}")
        sys.exit(2)

    if save_key and not inline_api_key:
        print("\n[!] --save-key requires --api-key.")
        sys.exit(2)

    if saved_key_path is not None:
        provider_spec = _PROVIDER_SPECS[cast(str, requested_provider)]
        print(f"\n[+] Saved {provider_spec['env_vars'][0]} to {saved_key_path}")

    # Step 2: Get question
    question = _get_question(args)
    if not question:
        print("\nNo question provided. Exiting.")
        sys.exit(1)

    print(f"\nQuestion: {question}")

    # Step 3: Detect agents
    use_demo = getattr(args, "demo", False)
    rounds = getattr(args, "rounds", 2)

    if use_demo:
        print("\n[*] Run mode: demo (requested with --demo)")
        print("    Agents: analyst (supportive), critic (critical), synthesizer (balanced)")
    else:
        try:
            detected = _detect_agents(requested_provider)
        except ValueError as e:
            print(f"\n[!] {e}")
            sys.exit(2)

        if not detected:
            print("\n[!] No supported API keys detected. Falling back to demo mode.")
            print("    This run will use local mock agents, not live model calls.")
            print(
                "    Set ANTHROPIC_API_KEY or OPENAI_API_KEY, or pass "
                "--provider <name> --api-key <key>, for live debates."
            )
            print("    Agents: analyst (supportive), critic (critical), synthesizer (balanced)")
            use_demo = True
        else:
            live_team = _build_live_team(
                detected,
                provider=requested_provider,
                api_key=inline_api_key,
            )
            print("\n[+] Run mode: live")
            print(
                "    Agents: "
                + ", ".join(f"{agent['provider']} ({agent['role']})" for agent in live_team)
            )

    print(f"[*] Running {rounds}-round debate...\n")

    # Step 4: Run debate
    start_time = time.monotonic()
    try:
        if use_demo:
            result = asyncio.run(_run_demo_debate(question, rounds))
        else:
            result = asyncio.run(_run_live_debate(question, live_team, rounds))
    except (OSError, ConnectionError, RuntimeError, ValueError) as e:
        logger.debug("Debate failed: %s", e)
        print(f"\n[!] Debate failed: {e}")
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
    if result.get("receipt_id"):
        print(f"  Receipt:    {result['receipt_id']}")

    if result.get("summary"):
        print(f"\n  Summary:\n  {result['summary'][:500]}")

    if result.get("dissent"):
        print("\n  Dissent:")
        for d in result["dissent"]:
            if isinstance(d, dict):
                print(f"    - {d.get('agent', '?')}: {d.get('reason', 'N/A')}")
            else:
                print(f"    - {d}")

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
