#!/usr/bin/env python3
"""
Aragora CLI - AI Red Team for Decision Stress-Testing

Usage:
    aragora ask "Design a rate limiter" --agents anthropic-api,openai-api --rounds 3
    aragora ask "Implement auth system" --agents anthropic-api,openai-api,gemini
    aragora stats

Environment Variables:
    ARAGORA_API_URL: API server URL (default: http://localhost:8080)
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aragora.agents.base import create_agent
from aragora.core import Environment
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.memory.store import CritiqueStore
from aragora.modes import ModeRegistry, register_all_builtins

# Ensure built-in modes are registered
register_all_builtins()

# Default API URL from environment or localhost fallback
DEFAULT_API_URL = os.environ.get("ARAGORA_API_URL", "http://localhost:8080")


def get_event_emitter_if_available(server_url: str = DEFAULT_API_URL) -> Optional[Any]:
    """
    Try to connect to the streaming server for audience participation.
    Returns event emitter if server is available, None otherwise.
    """
    try:
        import urllib.request

        # Quick health check
        req = urllib.request.Request(f"{server_url}/api/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                # Server is up, try to get emitter
                try:
                    from aragora.server.stream import SyncEventEmitter

                    return SyncEventEmitter()
                except ImportError:
                    pass
    except (urllib.error.URLError, OSError, TimeoutError):
        # Server not available - network error, timeout, or connection refused
        pass
    return None


def parse_agents(agents_str: str) -> list[tuple[str, str | None]]:
    """Parse agent string using unified AgentSpec.

    Supports both new pipe format (provider|model|persona|role) and
    legacy colon format (provider:persona). Returns tuples of (provider, role)
    for backward compatibility with existing code.
    """
    from aragora.agents.spec import AgentSpec

    specs = AgentSpec.parse_list(agents_str)
    return [(spec.provider, spec.role) for spec in specs]


async def run_debate(
    task: str,
    agents_str: str,
    rounds: int = 8,  # 9-round format (0-8) default
    consensus: str = "judge",  # Judge-based consensus default
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

    # Assign default roles
    agents = []
    for i, (agent_type, role) in enumerate(agent_specs):
        if role is None:
            if i == 0:
                role = "proposer"
            elif i == len(agent_specs) - 1:
                role = "synthesizer"
            else:
                role = "critic"

        agent = create_agent(
            model_type=agent_type,  # type: ignore[arg-type]
            name=f"{agent_type}_{role}",
            role=role,
        )
        # Apply mode system prompt if specified
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
    protocol = DebateProtocol(
        rounds=rounds,
        consensus=consensus,  # type: ignore[arg-type]
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
    arena = Arena(env, agents, protocol, memory, event_emitter=event_emitter)  # type: ignore[arg-type]
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
    protocol_overrides: dict[str, Any] | None = None
    if getattr(args, "demo", False):
        print("Demo mode enabled - using built-in demo agents.")
        agents = "demo,demo,demo"
        rounds = min(args.rounds, 2)
        learn = False
        enable_audience = False
        protocol_overrides = {
            "convergence_detection": False,
            "vote_grouping": False,
            "enable_trickster": False,
            "enable_research": False,
            "enable_rhetorical_observer": False,
            "role_rotation": False,
            "role_matching": False,
        }

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


def cmd_stats(args: argparse.Namespace) -> None:
    """Handle 'stats' command."""
    store = CritiqueStore(args.db)
    stats = store.get_stats()

    print("\nAgora Memory Statistics")
    print("=" * 40)
    print(f"Total debates: {stats['total_debates']}")
    print(f"Consensus reached: {stats['consensus_debates']}")
    print(f"Total critiques: {stats['total_critiques']}")
    print(f"Total patterns: {stats['total_patterns']}")
    print(f"Avg consensus confidence: {stats['avg_consensus_confidence']:.1%}")

    if stats["patterns_by_type"]:
        print("\nPatterns by type:")
        for ptype, count in sorted(stats["patterns_by_type"].items(), key=lambda x: -x[1]):
            print(f"  {ptype}: {count}")


def cmd_status(args: argparse.Namespace) -> None:
    """Handle 'status' command - show environment health and agent availability."""
    import os
    import shutil

    print("\nAragora Environment Status")
    print("=" * 60)

    # Check API keys
    print("\n📡 API Keys:")
    api_keys = [
        ("ANTHROPIC_API_KEY", "Anthropic (Claude)"),
        ("OPENAI_API_KEY", "OpenAI (GPT/Codex)"),
        ("OPENROUTER_API_KEY", "OpenRouter (Fallback)"),
        ("GEMINI_API_KEY", "Google (Gemini)"),
        ("XAI_API_KEY", "xAI (Grok)"),
        ("DEEPSEEK_API_KEY", "DeepSeek"),
    ]
    for env_var, name in api_keys:
        value = os.environ.get(env_var, "")
        if value:
            # Show masked key
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"  ✓ {name}: {masked}")
        else:
            print(f"  ✗ {name}: not set")

    # Check CLI tools
    print("\n🔧 CLI Tools:")
    cli_tools = [
        ("claude", "Claude Code CLI"),
        ("codex", "OpenAI Codex CLI"),
        ("gemini", "Gemini CLI"),
        ("grok", "Grok CLI"),
    ]
    for cmd, name in cli_tools:
        path = shutil.which(cmd)
        if path:
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: not installed")

    # Check server health
    print("\n🌐 Server Status:")
    server_url = args.server if hasattr(args, "server") else DEFAULT_API_URL
    try:
        import urllib.request

        req = urllib.request.Request(f"{server_url}/api/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                print(f"  ✓ Server running at {server_url}")
            else:
                print(f"  ⚠ Server returned status {resp.status}")
    except (OSError, TimeoutError):
        print(f"  ✗ Server not reachable at {server_url}")

    # Check database
    print("\n💾 Databases:")
    from aragora.config import DB_ELO_PATH, DB_INSIGHTS_PATH, DB_MEMORY_PATH

    db_paths = [
        (DB_MEMORY_PATH, "Memory store"),
        (DB_INSIGHTS_PATH, "Insights store"),
        (DB_ELO_PATH, "ELO rankings"),
    ]
    for db_path, name in db_paths:
        if Path(db_path).exists():
            size_mb = Path(db_path).stat().st_size / (1024 * 1024)
            print(f"  ✓ {name}: {size_mb:.1f} MB")
        else:
            print(f"  ✗ {name}: not found")

    # Show nomic loop state if available
    nomic_state = Path(".nomic/nomic_state.json")
    if nomic_state.exists():
        print("\n🔄 Nomic Loop:")
        try:
            import json

            with open(nomic_state) as f:
                state = json.load(f)
            total_cycles = state.get("total_cycles", 0)
            last_cycle = state.get("last_cycle_timestamp", "unknown")
            print(f"  Total cycles: {total_cycles}")
            print(f"  Last run: {last_cycle}")
        except Exception as e:
            print(f"  ⚠ Could not read state: {e}")

    print("\n" + "=" * 60)
    print("Run 'aragora ask' to start a debate or 'aragora serve' to start the server")


def cmd_agents(args: argparse.Namespace) -> None:
    """Handle 'agents' command - list available agents and their configuration."""
    from aragora.cli.agents import main as agents_main

    agents_main(args)


def cmd_modes(args: argparse.Namespace) -> None:
    """Handle 'modes' command - list available operational modes."""
    modes = ModeRegistry.get_all()

    print("\n" + "=" * 60)
    print("AVAILABLE OPERATIONAL MODES")
    print("=" * 60 + "\n")

    if not modes:
        print("No modes registered. This shouldn't happen!")
        return

    verbose = getattr(args, "verbose", False)

    for mode in modes:
        # Mode header
        print(f"[{mode.name}]")
        print(f"  {mode.description}")

        # Show tool access
        tools = []
        from aragora.modes.tool_groups import ToolGroup

        if ToolGroup.READ in mode.tool_groups:
            tools.append("read")
        if ToolGroup.EDIT in mode.tool_groups:
            tools.append("edit")
        if ToolGroup.COMMAND in mode.tool_groups:
            tools.append("command")
        if ToolGroup.BROWSER in mode.tool_groups:
            tools.append("browser")
        if ToolGroup.DEBATE in mode.tool_groups:
            tools.append("debate")

        print(f"  Tools: {', '.join(tools) if tools else 'none'}")

        if verbose:
            # Show full system prompt in verbose mode
            prompt = mode.get_system_prompt()
            # Truncate for display
            lines = prompt.strip().split("\n")
            preview = "\n    ".join(lines[:10])
            if len(lines) > 10:
                preview += "\n    ..."
            print(f"\n  System Prompt:\n    {preview}\n")
        else:
            print()

    print("-" * 60)
    print("Usage: aragora ask 'task' --mode <mode-name>")
    print("       aragora modes --verbose  (show full system prompts)")


def cmd_patterns(args: argparse.Namespace) -> None:
    """Handle 'patterns' command."""
    store = CritiqueStore(args.db)
    patterns = store.retrieve_patterns(
        issue_type=args.type,
        min_success=args.min_success,
        limit=args.limit,
    )

    print(f"\nTop {len(patterns)} Patterns")
    print("=" * 60)

    for p in patterns:
        print(f"\n[{p.issue_type}] (success: {p.success_count}, severity: {p.avg_severity:.1f})")
        print(f"  Issue: {p.issue_text[:80]}...")
        if p.suggestion_text:
            print(f"  Suggestion: {p.suggestion_text[:80]}...")


def cmd_demo(args: argparse.Namespace) -> None:
    """Handle 'demo' command - run a quick compelling demo."""
    from aragora.cli.demo import main as demo_main

    demo_main(args)


def cmd_templates(args: argparse.Namespace) -> None:
    """Handle 'templates' command - list available debate templates."""
    from aragora.templates import list_templates

    templates = list_templates()

    print("\n" + "=" * 60)
    print("📋 AVAILABLE DEBATE TEMPLATES")
    print("=" * 60 + "\n")

    for t in templates:
        print(f"[{t['type']}] {t['name']}")
        print(f"  {t['description'][:60]}...")
        print(f"  Agents: {t['agents']}, Domain: {t['domain']}")
        print()


def cmd_export(args: argparse.Namespace) -> None:
    """Handle 'export' command - export debate artifacts."""
    from aragora.cli.export import main as export_main

    export_main(args)


def cmd_doctor(args: argparse.Namespace) -> None:
    """Handle 'doctor' command - run system health checks."""
    from aragora.cli.doctor import main as doctor_main

    sys.exit(doctor_main())


def cmd_validate(_: argparse.Namespace) -> None:
    """Handle 'validate' command - validate API keys."""
    # run_validate doesn't exist; reuse doctor main for now
    from aragora.cli.doctor import main as doctor_main

    sys.exit(doctor_main())


def cmd_improve(args: argparse.Namespace) -> None:
    """Handle 'improve' command - self-improvement mode."""
    print("\n" + "=" * 60)
    print("🔧 SELF-IMPROVEMENT MODE")
    print("=" * 60)
    print(f"\nTarget: {args.path or 'current directory'}")
    print(f"Focus: {args.focus or 'general improvements'}")
    print()

    # This is a placeholder - full implementation would use SelfImprover
    print("⚠️  Self-improvement mode is experimental.")
    print("   Use 'aragora ask' to debate specific improvements.")
    print()

    if args.analyze:
        from aragora.tools.code import CodeReader

        reader = CodeReader(args.path or ".")
        tree = reader.get_file_tree(max_depth=2)

        print("📂 Codebase structure:")

        def print_tree(t, indent=0):
            for k, v in sorted(t.items()):
                if isinstance(v, dict):
                    print("  " * indent + f"📁 {k}")
                    print_tree(v, indent + 1)
                else:
                    print("  " * indent + f"📄 {k} ({v} bytes)")

        print_tree(tree)


def cmd_serve(args: argparse.Namespace) -> None:
    """Handle 'serve' command - run live debate server."""
    import asyncio
    import multiprocessing
    import signal
    from pathlib import Path

    try:
        from aragora.server.unified_server import run_unified_server
    except ImportError as e:
        print(f"Error importing server modules: {e}")
        print("Make sure websockets and aiohttp are installed: pip install websockets aiohttp")
        return

    # Determine static directory (Live Dashboard)
    static_dir = None
    live_dir = Path(__file__).parent.parent / "live" / "dist"
    if live_dir.exists():
        static_dir = live_dir
    else:
        # Fall back to docs directory for viewer.html
        docs_dir = Path(__file__).parent.parent.parent / "docs"
        if docs_dir.exists():
            static_dir = docs_dir

    workers = getattr(args, "workers", 1)
    workers = max(1, workers)

    print("\n" + "=" * 60)
    print("ARAGORA LIVE DEBATE SERVER")
    print("=" * 60)

    if workers == 1:
        print(f"\nWebSocket: ws://{args.host}:{args.ws_port}")
        print(f"HTTP API:  http://{args.host}:{args.api_port}")
        if static_dir:
            print(f"Dashboard: http://{args.host}:{args.api_port}/")
        print("\nPress Ctrl+C to stop\n")
        print("=" * 60 + "\n")

        try:
            asyncio.run(
                run_unified_server(
                    http_port=args.api_port,
                    ws_port=args.ws_port,
                    http_host=args.host,
                    ws_host=args.host,
                    static_dir=static_dir,
                )
            )
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
    else:
        # Multi-worker mode
        print(f"\nWorkers: {workers}")
        print(f"HTTP ports: {args.api_port}-{args.api_port + workers - 1}")
        print(f"WS ports: {args.ws_port}-{args.ws_port + workers - 1}")
        print("\nTip: Use a load balancer (nginx/haproxy) to distribute traffic.")
        print("\nPress Ctrl+C to stop\n")
        print("=" * 60 + "\n")

        def run_worker(http_port, ws_port, host, static):
            asyncio.run(
                run_unified_server(
                    http_port=http_port,
                    ws_port=ws_port,
                    http_host=host,
                    ws_host=host,
                    static_dir=static,
                )
            )

        processes = []

        def shutdown_workers(signum, frame):
            print("\nShutting down workers...")
            for p in processes:
                if p.is_alive():
                    p.terminate()
            sys.exit(0)

        signal.signal(signal.SIGINT, shutdown_workers)
        signal.signal(signal.SIGTERM, shutdown_workers)

        for i in range(workers):
            http_port = args.api_port + i
            ws_port = args.ws_port + i
            p = multiprocessing.Process(
                target=run_worker,
                args=(http_port, ws_port, args.host, static_dir),
                name=f"aragora-worker-{i}",
            )
            p.start()
            processes.append(p)
            print(f"  Worker {i}: HTTP={http_port}, WS={ws_port} (PID {p.pid})")

        try:
            for p in processes:
                p.join()
        except KeyboardInterrupt:
            shutdown_workers(None, None)


def cmd_init(args: argparse.Namespace) -> None:
    """Handle 'init' command - project scaffolding."""
    from aragora.cli.init import cmd_init as init_handler

    init_handler(args)


def cmd_repl(args: argparse.Namespace) -> None:
    """Handle 'repl' command - interactive debate mode."""
    from aragora.cli.repl import cmd_repl as repl_handler

    repl_handler(args)


def cmd_config(args: argparse.Namespace) -> None:
    """Handle 'config' command - manage configuration."""
    from aragora.cli.config import cmd_config as config_handler

    config_handler(args)


def cmd_replay(args: argparse.Namespace) -> None:
    """Handle 'replay' command - replay stored debates."""
    from aragora.cli.replay import cmd_replay as replay_handler

    replay_handler(args)


def cmd_bench(args: argparse.Namespace) -> None:
    """Handle 'bench' command - benchmark agents."""
    from aragora.cli.bench import cmd_bench as bench_handler

    bench_handler(args)


def cmd_review(args: argparse.Namespace) -> int:
    """Handle 'review' command - AI red team code review."""
    from aragora.cli.review import cmd_review as review_handler

    return review_handler(args)


def cmd_gauntlet(args: argparse.Namespace) -> None:
    """Handle 'gauntlet' command - adversarial stress-testing."""
    from aragora.cli.gauntlet import cmd_gauntlet as gauntlet_handler

    return gauntlet_handler(args)


def cmd_badge(args) -> None:
    """Generate badge markdown for README."""
    from aragora.cli.badge import main as badge_main

    badge_main(args)


def cmd_billing(args: argparse.Namespace) -> int:
    """Handle 'billing' command - manage billing and usage."""
    from aragora.cli.billing import main as billing_main

    return billing_main(args)


def cmd_mcp_server(args: argparse.Namespace) -> None:
    """Handle 'mcp-server' command - run MCP server."""
    try:
        from aragora.mcp.server import main as mcp_main

        mcp_main()
    except ImportError as e:
        print("\nError: MCP dependencies not installed")
        print(f"\nMissing: {e}")
        print("\nTo install MCP support:")
        print("  pip install mcp")
        print("  # or")
        print("  pip install aragora[mcp]")
        print(
            "\nMCP (Model Context Protocol) enables integration with Claude Desktop and other MCP clients."
        )


def cmd_memory(args: argparse.Namespace) -> None:
    """Handle 'memory' command - inspect ContinuumMemory tiers."""
    from aragora.config import DB_MEMORY_PATH
    from aragora.memory.continuum import ContinuumMemory, MemoryTier

    db_path = getattr(args, "db", None) or DB_MEMORY_PATH
    memory = ContinuumMemory(db_path=db_path)

    action = getattr(args, "action", "stats")

    if action == "stats":
        stats = memory.get_stats()
        print("\nContinuum Memory Statistics")
        print("=" * 50)
        print(f"Total memories: {stats.get('total_memories', 0)}")

        by_tier = stats.get("by_tier", {})
        if by_tier:
            print("\nMemories by tier:")
            tier_order = ["fast", "medium", "slow", "glacial"]
            for tier in tier_order:
                count = by_tier.get(tier, 0)
                bar = "█" * min(count // 10, 30) if count > 0 else ""
                print(f"  {tier:8}: {count:5} {bar}")

        tier_metrics = memory.get_tier_metrics()
        if tier_metrics:
            print("\nTier Metrics:")
            for tier, metrics in tier_metrics.items():
                if isinstance(metrics, dict):
                    promotions = metrics.get("promotions", 0)
                    demotions = metrics.get("demotions", 0)
                    if promotions or demotions:
                        print(f"  {tier}: ↑{promotions} promotions, ↓{demotions} demotions")

    elif action == "list":
        tier_name = getattr(args, "tier", "fast")
        limit = getattr(args, "limit", 10)

        try:
            memory_tier: MemoryTier = MemoryTier[tier_name.upper()]
        except KeyError:
            print(f"Invalid tier: {tier_name}. Use: fast, medium, slow, glacial")
            return

        entries = memory.retrieve(tiers=[memory_tier], limit=limit)
        print(f"\n{tier_name.upper()} Tier Memories ({len(entries)} entries)")
        print("=" * 60)

        for entry in entries:
            importance = f"[{entry.importance:.2f}]" if hasattr(entry, "importance") else ""
            content = entry.content[:80] + "..." if len(entry.content) > 80 else entry.content
            print(f"  {importance} {entry.id}: {content}")

    elif action == "consolidate":
        print("Running memory consolidation...")
        stats = memory.consolidate()
        print("Consolidation complete:")
        print(f"  Promotions: {stats.get('promotions', 0)}")
        print(f"  Demotions: {stats.get('demotions', 0)}")

    elif action == "cleanup":
        print("Cleaning up expired memories...")
        stats = memory.cleanup_expired_memories()
        print(f"Cleanup complete: {stats}")


def cmd_elo(args: argparse.Namespace) -> None:
    """Handle 'elo' command - view ELO ratings and history."""
    from aragora.config import DB_ELO_PATH
    from aragora.ranking.elo import EloSystem

    db_path = getattr(args, "db", None) or DB_ELO_PATH
    elo = EloSystem(db_path=db_path)

    action = getattr(args, "action", "leaderboard")

    if action == "leaderboard":
        limit = getattr(args, "limit", 10)
        domain = getattr(args, "domain", None)

        if domain:
            ratings = elo.get_top_agents_for_domain(domain, limit=limit)
            print(f"\nTop Agents in {domain}")
        else:
            ratings = elo.get_all_ratings()[:limit]
            print("\nGlobal Leaderboard")

        print("=" * 60)
        print(f"{'Rank':>4}  {'Agent':<20}  {'ELO':>7}  {'W/L':>8}  {'Win%':>6}")
        print("-" * 60)

        for i, rating in enumerate(ratings, 1):
            wins = rating.wins
            losses = rating.losses
            win_rate = f"{rating.win_rate:.1%}" if rating.games_played > 0 else "N/A"
            print(
                f"{i:>4}  {rating.agent_name:<20}  {rating.elo:>7.0f}  {wins:>3}/{losses:<3}  {win_rate:>6}"
            )

    elif action == "history":
        agent = getattr(args, "agent", None)
        if not agent:
            print("Error: --agent is required for history")
            return

        limit = getattr(args, "limit", 20)
        history = elo.get_elo_history(agent, limit=limit)

        print(f"\nELO History for {agent}")
        print("=" * 40)

        if not history:
            print("  No history found")
            return

        for timestamp, elo_value in history:
            print(f"  {timestamp[:19]}  {elo_value:>7.0f}")

    elif action == "matches":
        limit = getattr(args, "limit", 10)
        matches = elo.get_recent_matches(limit=limit)

        print("\nRecent Matches")
        print("=" * 70)

        if not matches:
            print("  No matches found")
            return

        for match in matches:
            winner = match.get("winner_name", "?")
            loser = match.get("loser_name", "?")
            is_draw = match.get("is_draw", False)
            domain = match.get("domain", "general")[:15]

            if is_draw:
                print(f"  DRAW: {winner} vs {loser} [{domain}]")
            else:
                print(f"  {winner} beat {loser} [{domain}]")

    elif action == "agent":
        agent = getattr(args, "agent", None)
        if not agent:
            print("Error: --agent is required")
            return

        try:
            rating = elo.get_rating(agent)
            print(f"\nAgent: {rating.agent_name}")
            print("=" * 40)
            print(f"  ELO Rating:    {rating.elo:>7.0f}")
            print(f"  Wins/Losses:   {rating.wins}/{rating.losses}")
            print(f"  Win Rate:      {rating.win_rate:.1%}")
            print(f"  Total Games:   {rating.games_played}")

            if rating.calibration_accuracy > 0:
                print(f"  Calibration:   {rating.calibration_accuracy:.1%}")

            # Show best domains
            best_domains = elo.get_best_domains(agent, limit=3)
            if best_domains:
                print("\n  Best Domains:")
                for domain, elo_rating in best_domains:
                    print(f"    {domain}: {elo_rating:.0f}")

            # Show rivals
            rivals = elo.get_rivals(agent, limit=3)
            if rivals:
                print("\n  Top Rivals:")
                for rival in rivals:
                    name = rival.get("partner", "?")
                    losses = rival.get("total_losses", 0)
                    print(f"    {name}: {losses} losses")

        except Exception:  # noqa: BLE001 - Agent lookup may fail in various ways
            print(f"Agent not found: {agent}")


def get_version() -> str:
    """Get package version from pyproject.toml or fallback."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("aragora")
    except ImportError:
        # importlib.metadata not available (Python < 3.8)
        return "0.8.0-dev"
    except PackageNotFoundError:
        # Package not installed in editable mode - use dev version
        return "0.8.0-dev"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aragora - AI Red Team for Decision Stress-Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  aragora ask "Design a rate limiter" --agents anthropic-api,openai-api
  aragora ask "Implement auth" --agents anthropic-api,openai-api,gemini --rounds 4
  aragora stats
  aragora patterns --type security
        """,
    )

    parser.add_argument("--version", "-V", action="version", version=f"aragora {get_version()}")
    parser.add_argument("--db", default="agora_memory.db", help="SQLite database path")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Run a decision stress-test (debate engine)")
    ask_parser.add_argument("task", help="The task/question to debate")
    ask_parser.add_argument(
        "--agents",
        "-a",
        default="codex,claude",
        help=(
            "Comma-separated agents (demo, anthropic-api,openai-api,gemini,grok or codex,claude). "
            "Use agent:role for specific roles."
        ),
    )
    ask_parser.add_argument(
        "--rounds",
        "-r",
        type=int,
        default=8,
        help="Number of debate rounds (default: 8 for 9-round format)",
    )
    ask_parser.add_argument(
        "--consensus",
        "-c",
        choices=["majority", "unanimous", "judge", "none"],
        default="judge",
        help="Consensus mechanism (default: judge)",
    )
    ask_parser.add_argument("--context", help="Additional context for the task")
    ask_parser.add_argument(
        "--no-learn", dest="learn", action="store_false", help="Don't store patterns"
    )
    ask_parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with built-in demo agents (no API keys required)",
    )
    ask_parser.add_argument(
        "--mode",
        "-m",
        choices=["architect", "coder", "reviewer", "debugger", "orchestrator"],
        help="Operational mode for agents (architect, coder, reviewer, debugger, orchestrator)",
    )
    ask_parser.set_defaults(func=cmd_ask)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # Status command - environment health check
    status_parser = subparsers.add_parser(
        "status", help="Show environment health and agent availability"
    )
    status_parser.add_argument(
        "--server",
        "-s",
        default=DEFAULT_API_URL,
        help=f"Server URL to check (default: {DEFAULT_API_URL})",
    )
    status_parser.set_defaults(func=cmd_status)

    # Agents command - list available agents
    agents_parser = subparsers.add_parser(
        "agents",
        help="List available agents and their configuration",
        description="Show all available agent types, their API key requirements, and configuration status.",
    )
    agents_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed descriptions"
    )
    agents_parser.set_defaults(func=cmd_agents)

    # Modes command - list available operational modes
    modes_parser = subparsers.add_parser(
        "modes",
        help="List available operational modes",
        description="Show all available operational modes (architect, coder, reviewer, etc.) for debates.",
    )
    modes_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show full system prompts"
    )
    modes_parser.set_defaults(func=cmd_modes)

    # Patterns command
    patterns_parser = subparsers.add_parser("patterns", help="Show learned patterns")
    patterns_parser.add_argument("--type", "-t", help="Filter by issue type")
    patterns_parser.add_argument("--min-success", type=int, default=1, help="Minimum success count")
    patterns_parser.add_argument("--limit", "-l", type=int, default=10, help="Max patterns to show")
    patterns_parser.set_defaults(func=cmd_patterns)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a quick demo debate")
    demo_parser.add_argument("name", nargs="?", help="Demo name (rate-limiter, auth, cache)")
    demo_parser.set_defaults(func=cmd_demo)

    # Templates command
    templates_parser = subparsers.add_parser("templates", help="List available debate templates")
    templates_parser.set_defaults(func=cmd_templates)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export debate artifacts")
    export_parser.add_argument("--debate-id", "-d", help="Debate ID to export")
    export_parser.add_argument(
        "--format",
        "-f",
        choices=["html", "json", "md"],
        default="html",
        help="Output format (default: html)",
    )
    export_parser.add_argument(
        "--output",
        "-o",
        default=".",
        help="Output directory (default: current)",
    )
    export_parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a demo export",
    )
    export_parser.set_defaults(func=cmd_export)

    # Doctor command
    doctor_parser = subparsers.add_parser("doctor", help="Run system health checks")
    doctor_parser.add_argument(
        "--validate", "-v", action="store_true", help="Validate API keys by making test calls"
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    # Validate command (API key validation)
    validate_parser = subparsers.add_parser(
        "validate", help="Validate API keys by making test calls"
    )
    validate_parser.set_defaults(func=cmd_validate)

    # Improve command (self-improvement mode)
    improve_parser = subparsers.add_parser("improve", help="Self-improvement mode")
    improve_parser.add_argument("--path", "-p", help="Path to codebase (default: current dir)")
    improve_parser.add_argument("--focus", "-f", help="Focus area for improvements")
    improve_parser.add_argument(
        "--analyze", "-a", action="store_true", help="Analyze codebase structure"
    )
    improve_parser.set_defaults(func=cmd_improve)

    # Serve command (live debate server)
    serve_parser = subparsers.add_parser("serve", help="Run live debate server")
    serve_parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket port")
    serve_parser.add_argument("--api-port", type=int, default=8080, help="HTTP API port")
    serve_parser.add_argument("--host", default="localhost", help="Host to bind to")
    serve_parser.set_defaults(func=cmd_serve)

    # Init command (project scaffolding)
    init_parser = subparsers.add_parser("init", help="Initialize Aragora project")
    init_parser.add_argument("directory", nargs="?", help="Target directory (default: current)")
    init_parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing files")
    init_parser.add_argument("--no-git", action="store_true", help="Don't modify .gitignore")
    init_parser.set_defaults(func=cmd_init)

    # REPL command (interactive mode)
    repl_parser = subparsers.add_parser("repl", help="Interactive debate mode")
    repl_parser.add_argument(
        "--agents",
        "-a",
        default="anthropic-api,openai-api",
        help="Comma-separated agents for debates",
    )
    repl_parser.add_argument(
        "--rounds", "-r", type=int, default=8, help="Debate rounds (default: 8)"
    )
    repl_parser.set_defaults(func=cmd_repl)

    # Config command (manage settings)
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument(
        "action",
        nargs="?",
        default="show",
        choices=["show", "get", "set", "env", "path"],
        help="Config action",
    )
    config_parser.add_argument("key", nargs="?", help="Config key (for get/set)")
    config_parser.add_argument("value", nargs="?", help="Config value (for set)")
    config_parser.set_defaults(func=cmd_config)

    # Replay command (replay stored debates)
    replay_parser = subparsers.add_parser("replay", help="Replay stored debates")
    replay_parser.add_argument(
        "action", nargs="?", default="list", choices=["list", "show", "play"], help="Replay action"
    )
    replay_parser.add_argument("id", nargs="?", help="Replay ID (for show/play)")
    replay_parser.add_argument("--directory", "-d", help="Replays directory")
    replay_parser.add_argument("--limit", "-n", type=int, default=10, help="Max replays to list")
    replay_parser.add_argument("--speed", "-s", type=float, default=1.0, help="Playback speed")
    replay_parser.set_defaults(func=cmd_replay)

    # Bench command (benchmark agents)
    bench_parser = subparsers.add_parser("bench", help="Benchmark agents")
    bench_parser.add_argument(
        "--agents",
        "-a",
        default="anthropic-api,openai-api",
        help="Comma-separated agents to benchmark",
    )
    bench_parser.add_argument("--iterations", "-n", type=int, default=3, help="Iterations per task")
    bench_parser.add_argument("--task", "-t", help="Custom benchmark task")
    bench_parser.add_argument("--quick", "-q", action="store_true", help="Quick mode (1 iteration)")
    bench_parser.set_defaults(func=cmd_bench)

    # Review command (AI red team code review)
    from aragora.cli.review import create_review_parser

    create_review_parser(subparsers)

    # Gauntlet command (adversarial stress-testing)
    from aragora.cli.gauntlet import create_gauntlet_parser

    create_gauntlet_parser(subparsers)

    # Badge command (generate README badges)
    badge_parser = subparsers.add_parser(
        "badge",
        help="Generate Aragora badge for your README",
        description="Generate shareable badges to show your project uses Aragora.",
    )
    badge_parser.add_argument(
        "--type",
        "-t",
        choices=["reviewed", "consensus", "gauntlet"],
        default="reviewed",
        help="Badge type: reviewed (blue), consensus (green), gauntlet (orange)",
    )
    badge_parser.add_argument(
        "--style",
        "-s",
        choices=["flat", "flat-square", "for-the-badge", "plastic"],
        default="flat",
        help="Badge style (default: flat)",
    )
    badge_parser.add_argument(
        "--repo",
        "-r",
        help="Link to specific repo (default: aragora repo)",
    )
    badge_parser.set_defaults(func=cmd_badge)

    # Batch command (process multiple debates)
    from aragora.cli.batch import create_batch_parser

    create_batch_parser(subparsers)

    # Billing command
    from aragora.cli.billing import create_billing_parser

    create_billing_parser(subparsers)

    # Audit command (compliance audit logs)
    from aragora.cli.audit import create_audit_parser

    create_audit_parser(subparsers)

    # Document audit command (document analysis)
    from aragora.cli.document_audit import create_document_audit_parser

    create_document_audit_parser(subparsers)

    # Documents command (upload, list, show with folder support)
    from aragora.cli.documents import create_documents_parser

    create_documents_parser(subparsers)

    # Knowledge command (knowledge base operations)
    from aragora.cli.knowledge import create_knowledge_parser

    create_knowledge_parser(subparsers)

    # RLM command (recursive language model operations)
    from aragora.cli.rlm import create_rlm_parser

    create_rlm_parser(subparsers)

    # Memory command (inspect ContinuumMemory tiers)
    memory_parser = subparsers.add_parser(
        "memory",
        help="Inspect multi-tier memory system",
        description="View and manage ContinuumMemory - the multi-tier learning system.",
    )
    memory_parser.add_argument(
        "action",
        nargs="?",
        default="stats",
        choices=["stats", "list", "consolidate", "cleanup"],
        help="Action: stats (default), list, consolidate, cleanup",
    )
    memory_parser.add_argument(
        "--tier",
        "-t",
        choices=["fast", "medium", "slow", "glacial"],
        default="fast",
        help="Memory tier to list (for 'list' action)",
    )
    memory_parser.add_argument("--limit", "-n", type=int, default=10, help="Max entries to show")
    memory_parser.add_argument("--db", help="Database path (default: from config)")
    memory_parser.set_defaults(func=cmd_memory)

    # ELO command (view agent rankings and history)
    elo_parser = subparsers.add_parser(
        "elo",
        help="View ELO ratings, leaderboards, and match history",
        description="Inspect agent skill ratings, match history, and leaderboards.",
    )
    elo_parser.add_argument(
        "action",
        nargs="?",
        default="leaderboard",
        choices=["leaderboard", "history", "matches", "agent"],
        help="Action: leaderboard (default), history, matches, agent",
    )
    elo_parser.add_argument("--agent", "-a", help="Agent name (for history/agent actions)")
    elo_parser.add_argument("--domain", "-d", help="Filter by domain (for leaderboard)")
    elo_parser.add_argument("--limit", "-n", type=int, default=10, help="Max entries to show")
    elo_parser.add_argument("--db", help="Database path (default: from config)")
    elo_parser.set_defaults(func=cmd_elo)

    # MCP Server command
    mcp_parser = subparsers.add_parser(
        "mcp-server",
        help="Run the MCP (Model Context Protocol) server",
        description="""
Run the Aragora MCP server for integration with Claude and other MCP clients.

The MCP server exposes Aragora's capabilities as tools:
- run_debate: Run decision stress-tests (debate engine)
- run_gauntlet: Stress-test documents
- list_agents: List available agents
- get_debate: Retrieve debate results

Configure in claude_desktop_config.json:
{
    "mcpServers": {
        "aragora": {
            "command": "aragora",
            "args": ["mcp-server"]
        }
    }
}
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mcp_parser.set_defaults(func=cmd_mcp_server)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
