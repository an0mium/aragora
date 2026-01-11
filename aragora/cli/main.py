#!/usr/bin/env python3
"""
Aragora CLI - Multi-Agent Debate Framework

Usage:
    aragora ask "Design a rate limiter" --agents anthropic-api,openai-api --rounds 3
    aragora ask "Implement auth system" --agents anthropic-api,openai-api,gemini
    aragora stats
"""

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path
from typing import Any, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aragora.agents.base import create_agent
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.memory.store import CritiqueStore
from aragora.core import Environment


def get_event_emitter_if_available(server_url: str = "http://localhost:8080") -> Optional[Any]:
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


def parse_agents(agents_str: str) -> list[tuple[str, str]]:
    """Parse agent string like 'codex,claude:critic,openai'."""
    agents = []
    for spec in agents_str.split(","):
        spec = spec.strip()
        if ":" in spec:
            agent_type, role = spec.split(":", 1)
        else:
            agent_type = spec
            role = None
        agents.append((agent_type, role))
    return agents


async def run_debate(
    task: str,
    agents_str: str,
    rounds: int = 3,
    consensus: str = "majority",
    context: str = "",
    learn: bool = True,
    db_path: str = "agora_memory.db",
    enable_audience: bool = True,
    server_url: str = "http://localhost:8080",
):
    """Run a multi-agent debate."""

    # Parse and create agents
    agent_specs = parse_agents(agents_str)

    # Assign default roles
    roles = ["proposer", "critic", "synthesizer"]
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
    result = asyncio.run(
        run_debate(
            task=args.task,
            agents_str=args.agents,
            rounds=args.rounds,
            consensus=args.consensus,
            context=args.context or "",
            learn=args.learn,
            db_path=args.db,
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
    server_url = args.server if hasattr(args, 'server') else "http://localhost:8080"
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
    from aragora.config import DB_MEMORY_PATH, DB_INSIGHTS_PATH, DB_ELO_PATH
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
    import time

    demo_tasks: dict[str, dict[str, str | int]] = {
        "rate-limiter": {
            "task": "Design a distributed rate limiter that handles 1M requests/second across multiple regions",
            "agents": "codex,claude",
            "rounds": 2,
        },
        "auth": {
            "task": "Design a secure authentication system with passwordless login and MFA support",
            "agents": "claude,codex",
            "rounds": 2,
        },
        "cache": {
            "task": "Design a cache invalidation strategy for a social media feed with 100M users",
            "agents": "codex,claude",
            "rounds": 2,
        },
    }

    demo_name = args.name or "rate-limiter"
    if demo_name not in demo_tasks:
        print(f"Unknown demo: {demo_name}")
        print(f"Available demos: {', '.join(demo_tasks.keys())}")
        return

    demo = demo_tasks[demo_name]
    task = str(demo["task"])
    agents = str(demo["agents"])
    rounds = int(demo["rounds"])

    print("\n" + "=" * 60)
    print("🎭 AAGORA DEMO - Multi-Agent Debate")
    print("=" * 60)
    print(f"\n📋 Task: {task[:80]}...")
    print(f"🤖 Agents: {agents}")
    print(f"🔄 Rounds: {rounds}")
    print("\n" + "-" * 60)
    print("Starting debate...")
    print("-" * 60 + "\n")

    start = time.time()

    result = asyncio.run(
        run_debate(
            task=task,
            agents_str=agents,
            rounds=rounds,
            consensus="majority",
            learn=False,
        )
    )

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("✅ DEBATE COMPLETE")
    print("=" * 60)
    print(f"⏱️  Duration: {elapsed:.1f}s")
    print(f"🎯 Consensus: {'Reached' if result.consensus_reached else 'Not reached'}")
    print(f"📊 Confidence: {result.confidence:.0%}")
    print("\n" + "-" * 60)
    print("FINAL ANSWER:")
    print("-" * 60)
    print(result.final_answer[:1000])
    if len(result.final_answer) > 1000:
        print("...")
    print("\n" + "=" * 60)


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
    from pathlib import Path
    from aragora.export.artifact import DebateArtifact, ArtifactBuilder, ConsensusProof
    from aragora.export.static_html import StaticHTMLExporter

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If demo mode, create a sample artifact
    if args.demo:
        from datetime import datetime
        from aragora.core import DebateResult, Message, Critique

        # Create demo result
        demo_result = DebateResult(
            task="Design a distributed rate limiter for a high-traffic API",
            final_answer="""## Recommended Architecture

1. **Token Bucket Algorithm** - Use a distributed token bucket with Redis as the backing store
2. **Sliding Window Counters** - Combine with sliding window for burst handling
3. **Consistent Hashing** - Distribute rate limit state across multiple nodes
4. **Circuit Breaker** - Implement fallback when rate limit service is unavailable

### Key Implementation Details:
- Use Redis MULTI/EXEC for atomic operations
- Implement local caching with 100ms TTL for hot keys
- Add monitoring for rate limit violations
- Include graceful degradation mode""",
            confidence=0.85,
            consensus_reached=True,
            rounds_used=2,
            duration_seconds=45.3,
            messages=[
                Message(role="proposer", agent="codex", content="Token bucket with Redis...", round=0),
                Message(role="proposer", agent="claude", content="Consider sliding window...", round=0),
                Message(role="critic", agent="claude", content="Redis single point of failure...", round=1),
                Message(role="synthesizer", agent="codex", content="Combined approach with fallback...", round=2),
            ],
            critiques=[
                Critique(
                    agent="claude",
                    target_agent="codex",
                    target_content="Redis proposal",
                    issues=["Single point of failure", "Network latency concerns"],
                    suggestions=["Add local caching", "Implement circuit breaker"],
                    severity=0.4,
                    reasoning="Good base but needs resilience",
                ),
            ],
        )

        # Build artifact
        artifact = (ArtifactBuilder()
            .from_result(demo_result)
            .with_verification("claim-1", "Token bucket is O(1)", "verified", "simulation")
            .build())

        artifact_id = artifact.artifact_id
    else:
        # Load from database or file
        if args.debate_id:
            # Load from trace database
            from aragora.debate.traces import list_traces, DebateReplayer

            try:
                replayer = DebateReplayer.from_database(
                    f"trace-{args.debate_id}",
                    args.db or "aragora_traces.db"
                )
                trace = replayer.trace

                # Build artifact from trace
                artifact = DebateArtifact(
                    debate_id=trace.debate_id,
                    task=trace.task,
                    trace_data={"events": [e.to_dict() for e in trace.events]},
                    agents=trace.agents,
                    duration_seconds=trace.duration_ms / 1000 if trace.duration_ms else 0,
                )

                if trace.final_result:
                    artifact.consensus_proof = ConsensusProof(
                        reached=trace.final_result.get("consensus_reached", False),
                        confidence=trace.final_result.get("confidence", 0),
                        vote_breakdown={},
                        final_answer=trace.final_result.get("final_answer", ""),
                        rounds_used=trace.final_result.get("rounds_used", 0),
                    )

                artifact_id = artifact.artifact_id
            except Exception as e:
                print(f"Error loading debate: {e}")
                print("Use --demo for a sample export, or ensure the debate ID exists.")
                return
        else:
            print("Please provide a debate ID (--debate-id) or use --demo for a sample export.")
            return

    # Generate output
    format_type = args.format.lower()

    if format_type == "html":
        exporter = StaticHTMLExporter(artifact)
        filepath = output_dir / f"debate_{artifact_id}.html"
        exporter.save(filepath)
        print(f"HTML export saved: {filepath}")

    elif format_type == "json":
        filepath = output_dir / f"debate_{artifact_id}.json"
        artifact.save(filepath)
        print(f"JSON export saved: {filepath}")

    elif format_type == "md":
        # Use existing publish.py markdown generator or simple markdown
        from aragora.cli.publish import generate_markdown_report
        from aragora.core import DebateResult, Message

        # Reconstruct minimal result for markdown generator
        result = DebateResult(
            id=artifact.artifact_id,
            task=artifact.task,
            final_answer=artifact.consensus_proof.final_answer if artifact.consensus_proof else "",
            confidence=artifact.consensus_proof.confidence if artifact.consensus_proof else 0,
            consensus_reached=artifact.consensus_proof.reached if artifact.consensus_proof else False,
            rounds_used=artifact.rounds,
            duration_seconds=artifact.duration_seconds,
            messages=[],
            critiques=[],
        )

        md_content = generate_markdown_report(result)
        filepath = output_dir / f"debate_{artifact_id}.md"
        filepath.write_text(md_content)
        print(f"Markdown export saved: {filepath}")

    else:
        print(f"Unknown format: {format_type}. Use html, json, or md.")
        return

    print(f"\nArtifact ID: {artifact_id}")
    print(f"Content Hash: {artifact.content_hash}")


def cmd_doctor(args: argparse.Namespace) -> None:
    """Handle 'doctor' command - run system health checks."""
    from aragora.cli.doctor import main as doctor_main
    validate = getattr(args, "validate", False)
    sys.exit(doctor_main(validate_keys=validate))


def cmd_validate(_: argparse.Namespace) -> None:
    """Handle 'validate' command - validate API keys."""
    from aragora.cli.doctor import run_validate
    sys.exit(run_validate())


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

    print("\n" + "=" * 60)
    print("ARAGORA LIVE DEBATE SERVER")
    print("=" * 60)
    print(f"\nWebSocket: ws://{args.host}:{args.ws_port}")
    print(f"HTTP API:  http://{args.host}:{args.api_port}")
    if static_dir:
        print(f"Dashboard: http://{args.host}:{args.api_port}/")
    print(f"\nPress Ctrl+C to stop\n")
    print("=" * 60 + "\n")

    try:
        asyncio.run(run_unified_server(
            http_port=args.api_port,
            ws_port=args.ws_port,
            static_dir=static_dir,
        ))
    except KeyboardInterrupt:
        print("\n\nServer stopped.")


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
    import time
    from pathlib import Path

    from aragora.agents.base import create_agent
    from aragora.gauntlet import (
        GauntletOrchestrator,
        GauntletConfig,
        GauntletProgress,
        InputType,
        QUICK_GAUNTLET,
        THOROUGH_GAUNTLET,
        CODE_REVIEW_GAUNTLET,
        POLICY_GAUNTLET,
        GDPR_GAUNTLET,
        HIPAA_GAUNTLET,
        AI_ACT_GAUNTLET,
        SECURITY_GAUNTLET,
        SOX_GAUNTLET,
        get_compliance_gauntlet,
        DecisionReceipt,
    )
    from aragora.gauntlet.personas import list_personas, get_persona

    print("\n" + "=" * 60)
    print("GAUNTLET - Adversarial Stress-Testing")
    print("=" * 60)

    # Load input content
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    input_content = input_path.read_text()
    print(f"\nInput: {input_path} ({len(input_content)} chars)")

    # Determine input type
    input_type_map = {
        "spec": InputType.SPEC,
        "architecture": InputType.ARCHITECTURE,
        "policy": InputType.POLICY,
        "code": InputType.CODE,
        "strategy": InputType.STRATEGY,
        "contract": InputType.CONTRACT,
    }
    input_type = input_type_map.get(args.input_type, InputType.SPEC)
    print(f"Type: {input_type.value}")

    # Create agents
    agent_specs = parse_agents(args.agents)
    agents = []
    for i, (agent_type, role) in enumerate(agent_specs):
        role = role or f"agent_{i}"
        try:
            agent = create_agent(
                model_type=agent_type,  # type: ignore[arg-type]
                name=f"{agent_type}_{role}",
                role=role,
            )
            agents.append(agent)
        except Exception as e:
            print(f"Warning: Could not create agent {agent_type}: {e}")

    if not agents:
        print("Error: No agents could be created. Check your API keys.")
        return

    print(f"Agents: {', '.join(a.name for a in agents)}")

    # Select config profile
    persona = None
    if hasattr(args, 'persona') and args.persona:
        # Use persona-based compliance profile
        persona = args.persona
        print(f"Persona: {persona}")
        if args.profile == "quick":
            base_config = QUICK_GAUNTLET
        elif args.profile == "thorough":
            base_config = THOROUGH_GAUNTLET
        else:
            base_config = get_compliance_gauntlet(persona)
    elif args.profile == "quick":
        base_config = QUICK_GAUNTLET
    elif args.profile == "thorough":
        base_config = THOROUGH_GAUNTLET
    elif args.profile == "code":
        base_config = CODE_REVIEW_GAUNTLET
    elif args.profile == "policy":
        base_config = POLICY_GAUNTLET
    elif args.profile == "gdpr":
        base_config = GDPR_GAUNTLET
        persona = "gdpr"
    elif args.profile == "hipaa":
        base_config = HIPAA_GAUNTLET
        persona = "hipaa"
    elif args.profile == "ai_act":
        base_config = AI_ACT_GAUNTLET
        persona = "ai_act"
    elif args.profile == "security":
        base_config = SECURITY_GAUNTLET
        persona = "security"
    elif args.profile == "sox":
        base_config = SOX_GAUNTLET
        persona = "sox"
    else:
        base_config = GauntletConfig()

    # Build config
    config = GauntletConfig(
        input_type=input_type,
        input_content=input_content,
        input_path=input_path,
        severity_threshold=base_config.severity_threshold,
        risk_threshold=base_config.risk_threshold,
        max_duration_seconds=args.timeout or base_config.max_duration_seconds,
        deep_audit_rounds=args.rounds or base_config.deep_audit_rounds,
        enable_redteam=not args.no_redteam,
        enable_probing=not args.no_probing,
        enable_deep_audit=not args.no_audit,
        enable_verification=args.verify,
        persona=persona,
    )

    print(f"Profile: {args.profile}")
    print(f"Max duration: {config.max_duration_seconds}s")
    print("\n" + "-" * 60)
    print("Running stress-test...")
    print("-" * 60 + "\n")

    # Progress callback for CLI display
    last_phase = [None]  # Use list for mutable closure

    def on_progress(progress: GauntletProgress) -> None:
        """Display progress updates in the CLI."""
        # Progress bar
        bar_width = 40
        filled = int(bar_width * progress.percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Clear line and print progress
        line = f"\r[{bar}] {progress.percent:5.1f}% | {progress.phase}"
        if progress.findings_so_far > 0:
            line += f" | {progress.findings_so_far} findings"

        # Print to stderr for live updates (stdout may be buffered)
        import sys
        sys.stderr.write(line + " " * 10)  # Extra spaces to clear old text
        sys.stderr.flush()

        # Print phase change message on new line
        if progress.phase != last_phase[0] and last_phase[0] is not None:
            sys.stderr.write("\n")
            sys.stderr.flush()
        last_phase[0] = progress.phase

        # Print completion message
        if progress.percent >= 100:
            sys.stderr.write("\n")
            sys.stderr.flush()

    # Run gauntlet with progress callback
    start = time.time()
    orchestrator = GauntletOrchestrator(agents, on_progress=on_progress)
    result = asyncio.run(orchestrator.run(config))
    elapsed = time.time() - start

    # Print summary
    print("\n" + result.summary())

    # Generate and save receipt
    if args.output:
        output_path = Path(args.output)

        input_hash = hashlib.sha256(config.input_content.encode()).hexdigest()
        receipt = DecisionReceipt.from_mode_result(result, input_hash=input_hash)

        # Determine format from extension or --format
        format_ext = args.format or output_path.suffix.lstrip(".")
        if format_ext not in ("json", "md", "html", "pdf"):
            format_ext = "html"

        # Save in appropriate format
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format_ext == "json":
            output_file = output_path.with_suffix(".json")
            output_file.write_text(receipt.to_json())
        elif format_ext == "md":
            output_file = output_path.with_suffix(".md")
            output_file.write_text(receipt.to_markdown())
        else:
            output_file = output_path.with_suffix(".html")
            output_file.write_text(receipt.to_html())

        print(f"\nDecision Receipt saved: {output_file}")
        print(f"Artifact Hash: {receipt.artifact_hash[:16]}...")

    # Exit with non-zero if rejected
    if result.verdict.value == "rejected":
        print("\n[REJECTED] This input failed the stress-test.")
        sys.exit(1)
    elif result.verdict.value == "needs_review":
        print("\n[NEEDS REVIEW] This input requires human review.")
        sys.exit(2)


def cmd_badge(args) -> None:
    """Generate badge markdown for README."""
    style = getattr(args, "style", "flat")
    repo = getattr(args, "repo", None)

    # Badge URLs
    badges = {
        "reviewed": {
            "flat": "https://img.shields.io/badge/Reviewed%20by-Aragora%20AI%20Red%20Team-blue?style=flat",
            "flat-square": "https://img.shields.io/badge/Reviewed%20by-Aragora%20AI%20Red%20Team-blue?style=flat-square",
            "for-the-badge": "https://img.shields.io/badge/Reviewed%20by-Aragora%20AI%20Red%20Team-blue?style=for-the-badge",
            "plastic": "https://img.shields.io/badge/Reviewed%20by-Aragora%20AI%20Red%20Team-blue?style=plastic",
        },
        "consensus": {
            "flat": "https://img.shields.io/badge/AI%20Consensus-Unanimous-brightgreen?style=flat",
            "flat-square": "https://img.shields.io/badge/AI%20Consensus-Unanimous-brightgreen?style=flat-square",
            "for-the-badge": "https://img.shields.io/badge/AI%20Consensus-Unanimous-brightgreen?style=for-the-badge",
            "plastic": "https://img.shields.io/badge/AI%20Consensus-Unanimous-brightgreen?style=plastic",
        },
        "gauntlet": {
            "flat": "https://img.shields.io/badge/Stress%20Tested-Aragora%20Gauntlet-orange?style=flat",
            "flat-square": "https://img.shields.io/badge/Stress%20Tested-Aragora%20Gauntlet-orange?style=flat-square",
            "for-the-badge": "https://img.shields.io/badge/Stress%20Tested-Aragora%20Gauntlet-orange?style=for-the-badge",
            "plastic": "https://img.shields.io/badge/Stress%20Tested-Aragora%20Gauntlet-orange?style=plastic",
        },
    }

    badge_type = getattr(args, "type", "reviewed")
    badge_url = badges.get(badge_type, badges["reviewed"]).get(style, badges[badge_type]["flat"])

    link_url = "https://github.com/an0mium/aragora"
    if repo:
        link_url = f"https://github.com/{repo}"

    # Generate markdown
    markdown = f"[![Aragora]({badge_url})]({link_url})"

    print("\n📛 Aragora Badge\n")
    print("Add this to your README.md:\n")
    print("```markdown")
    print(markdown)
    print("```\n")

    # Also show HTML version
    print("Or use HTML:\n")
    print("```html")
    print(f'<a href="{link_url}"><img src="{badge_url}" alt="Aragora"></a>')
    print("```\n")

    # Copy hint
    print("Preview:")
    print(f"  {markdown}\n")


def cmd_mcp_server(args: argparse.Namespace) -> None:
    """Handle 'mcp-server' command - run MCP server."""
    try:
        from aragora.mcp.server import main as mcp_main
        mcp_main()
    except ImportError as e:
        print(f"Error: MCP dependencies not installed: {e}")
        print("Install with: pip install mcp")
        sys.exit(1)
    except Exception as e:
        print(f"MCP server error: {e}")
        sys.exit(1)


def cmd_batch(args: argparse.Namespace) -> None:
    """Handle 'batch' command - run multiple debates from file."""
    import json
    import time
    from pathlib import Path

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("BATCH DEBATE PROCESSING")
    print("=" * 60)

    # Read input file (JSONL or JSON array)
    items = []
    try:
        content = input_path.read_text().strip()
        if content.startswith("["):
            # JSON array
            items = json.loads(content)
        else:
            # JSONL format
            for line_num, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)

    if not items:
        print("Error: No valid debate items found in input file")
        sys.exit(1)

    print(f"\nInput: {input_path}")
    print(f"Items: {len(items)}")
    print(f"Mode: {'server' if args.server else 'local'}")

    if args.server:
        # Submit to server batch API
        _batch_via_server(items, args)
    else:
        # Process locally
        _batch_local(items, args)


def _batch_via_server(items: list, args: argparse.Namespace) -> None:
    """Submit batch to server API."""
    import json
    import urllib.request
    import urllib.error
    import time

    server_url = args.url.rstrip("/")

    print(f"\nSubmitting to {server_url}/api/debates/batch...")

    # Prepare batch request
    batch_data = {
        "items": items,
    }

    if args.webhook:
        batch_data["webhook_url"] = args.webhook

    # Submit batch
    try:
        req = urllib.request.Request(
            f"{server_url}/api/debates/batch",
            data=json.dumps(batch_data).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        if args.token:
            req.add_header("Authorization", f"Bearer {args.token}")

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())

        if not result.get("success"):
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)

        batch_id = result.get("batch_id")
        print(f"\nBatch submitted successfully!")
        print(f"Batch ID: {batch_id}")
        print(f"Items queued: {result.get('items_queued', len(items))}")
        print(f"Status URL: {result.get('status_url', '')}")

        if args.wait:
            print("\nWaiting for completion...")
            _poll_batch_status(server_url, batch_id, args.token)

    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"Server error ({e.code}): {error_body}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection error: {e.reason}")
        sys.exit(1)


def _poll_batch_status(server_url: str, batch_id: str, token: str = None) -> None:
    """Poll batch status until completion."""
    import json
    import urllib.request
    import time

    poll_interval = 5  # seconds
    max_polls = 360  # 30 minutes max

    for i in range(max_polls):
        try:
            req = urllib.request.Request(
                f"{server_url}/api/debates/batch/{batch_id}/status",
                method="GET",
            )
            if token:
                req.add_header("Authorization", f"Bearer {token}")

            with urllib.request.urlopen(req, timeout=10) as resp:
                status = json.loads(resp.read().decode())

            progress = status.get("progress_percent", 0)
            completed = status.get("completed", 0)
            failed = status.get("failed", 0)
            total = status.get("total_items", 0)
            batch_status = status.get("status", "unknown")

            print(f"\r[{progress:5.1f}%] {completed}/{total} completed, {failed} failed - {batch_status}", end="", flush=True)

            if batch_status in ("completed", "partial", "failed", "cancelled"):
                print("\n")
                if batch_status == "completed":
                    print("Batch completed successfully!")
                elif batch_status == "partial":
                    print(f"Batch partially completed: {completed} succeeded, {failed} failed")
                elif batch_status == "failed":
                    print("Batch failed!")
                else:
                    print("Batch cancelled")
                return

            time.sleep(poll_interval)

        except Exception as e:
            print(f"\nWarning: Poll error: {e}")
            time.sleep(poll_interval)

    print("\nTimeout: Batch did not complete within 30 minutes")


def _batch_local(items: list, args: argparse.Namespace) -> None:
    """Process batch locally (sequential)."""
    import time

    results = []
    total = len(items)
    start_time = time.time()

    print("\nProcessing debates locally...\n")

    for i, item in enumerate(items):
        question = item.get("question", "")
        agents = item.get("agents", args.agents)
        rounds = item.get("rounds", args.rounds)

        print(f"[{i+1}/{total}] {question[:50]}...")

        try:
            result = asyncio.run(
                run_debate(
                    task=question,
                    agents_str=agents,
                    rounds=rounds,
                    consensus="majority",
                    learn=False,
                    enable_audience=False,
                )
            )

            results.append({
                "question": question,
                "success": True,
                "consensus_reached": result.consensus_reached,
                "confidence": result.confidence,
                "final_answer": result.final_answer[:200],
            })
            print(f"    => {'Consensus' if result.consensus_reached else 'No consensus'} ({result.confidence:.0%})")

        except Exception as e:
            results.append({
                "question": question,
                "success": False,
                "error": str(e),
            })
            print(f"    => ERROR: {e}")

    elapsed = time.time() - start_time
    succeeded = sum(1 for r in results if r.get("success"))

    print("\n" + "=" * 60)
    print("BATCH COMPLETE")
    print("=" * 60)
    print(f"Total: {total}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {total - succeeded}")
    print(f"Duration: {elapsed:.1f}s")

    # Save results if output specified
    if args.output:
        import json
        from pathlib import Path

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved: {output_path}")


def get_version() -> str:
    """Get package version from pyproject.toml or fallback."""
    try:
        from importlib.metadata import version
        return version("aragora")
    except Exception:
        return "0.0.7-dev"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aragora - Multi-Agent Debate Framework",
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
    ask_parser = subparsers.add_parser("ask", help="Run a multi-agent debate")
    ask_parser.add_argument("task", help="The task/question to debate")
    ask_parser.add_argument(
        "--agents",
        "-a",
        default="codex,claude",
        help=(
            "Comma-separated agents (anthropic-api,openai-api,gemini,grok or codex,claude). "
            "Use agent:role for specific roles."
        ),
    )
    ask_parser.add_argument("--rounds", "-r", type=int, default=3, help="Number of debate rounds")
    ask_parser.add_argument(
        "--consensus",
        "-c",
        choices=["majority", "unanimous", "judge", "none"],
        default="majority",
        help="Consensus mechanism",
    )
    ask_parser.add_argument("--context", help="Additional context for the task")
    ask_parser.add_argument(
        "--no-learn", dest="learn", action="store_false", help="Don't store patterns"
    )
    ask_parser.set_defaults(func=cmd_ask)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # Status command - environment health check
    status_parser = subparsers.add_parser("status", help="Show environment health and agent availability")
    status_parser.add_argument("--server", "-s", default="http://localhost:8080", help="Server URL to check")
    status_parser.set_defaults(func=cmd_status)

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
        "--format", "-f",
        choices=["html", "json", "md"],
        default="html",
        help="Output format (default: html)",
    )
    export_parser.add_argument(
        "--output", "-o",
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
        "--validate", "-v",
        action="store_true",
        help="Validate API keys by making test calls"
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    # Validate command (API key validation)
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate API keys by making test calls"
    )
    validate_parser.set_defaults(func=cmd_validate)

    # Improve command (self-improvement mode)
    improve_parser = subparsers.add_parser("improve", help="Self-improvement mode")
    improve_parser.add_argument("--path", "-p", help="Path to codebase (default: current dir)")
    improve_parser.add_argument("--focus", "-f", help="Focus area for improvements")
    improve_parser.add_argument("--analyze", "-a", action="store_true", help="Analyze codebase structure")
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
        "--agents", "-a", default="anthropic-api,openai-api",
        help="Comma-separated agents for debates"
    )
    repl_parser.add_argument("--rounds", "-r", type=int, default=3, help="Debate rounds")
    repl_parser.set_defaults(func=cmd_repl)

    # Config command (manage settings)
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument(
        "action", nargs="?", default="show",
        choices=["show", "get", "set", "env", "path"],
        help="Config action"
    )
    config_parser.add_argument("key", nargs="?", help="Config key (for get/set)")
    config_parser.add_argument("value", nargs="?", help="Config value (for set)")
    config_parser.set_defaults(func=cmd_config)

    # Replay command (replay stored debates)
    replay_parser = subparsers.add_parser("replay", help="Replay stored debates")
    replay_parser.add_argument(
        "action", nargs="?", default="list",
        choices=["list", "show", "play"],
        help="Replay action"
    )
    replay_parser.add_argument("id", nargs="?", help="Replay ID (for show/play)")
    replay_parser.add_argument("--directory", "-d", help="Replays directory")
    replay_parser.add_argument("--limit", "-n", type=int, default=10, help="Max replays to list")
    replay_parser.add_argument("--speed", "-s", type=float, default=1.0, help="Playback speed")
    replay_parser.set_defaults(func=cmd_replay)

    # Bench command (benchmark agents)
    bench_parser = subparsers.add_parser("bench", help="Benchmark agents")
    bench_parser.add_argument(
        "--agents", "-a", default="anthropic-api,openai-api",
        help="Comma-separated agents to benchmark"
    )
    bench_parser.add_argument("--iterations", "-n", type=int, default=3, help="Iterations per task")
    bench_parser.add_argument("--task", "-t", help="Custom benchmark task")
    bench_parser.add_argument("--quick", "-q", action="store_true", help="Quick mode (1 iteration)")
    bench_parser.set_defaults(func=cmd_bench)

    # Review command (AI red team code review)
    from aragora.cli.review import create_review_parser
    create_review_parser(subparsers)

    # Gauntlet command (adversarial stress-testing)
    gauntlet_parser = subparsers.add_parser(
        "gauntlet",
        help="Adversarial stress-test a specification, architecture, or policy",
        description="""
Run comprehensive adversarial stress-testing on documents.

Gauntlet combines multiple validation techniques:
- Red-team attacks (logical fallacies, edge cases, security)
- Capability probing (hallucination, sycophancy, consistency)
- Deep audit (multi-round intensive analysis)
- Formal verification (Z3/Lean proofs where applicable)
- Risk assessment (domain-specific hazards)

Produces Decision Receipts - audit-ready artifacts for compliance.

Examples:
    aragora gauntlet spec.md --input-type spec
    aragora gauntlet architecture.md --input-type architecture --profile thorough
    aragora gauntlet policy.yaml --input-type policy --output receipt.html
    aragora gauntlet code.py --input-type code --profile code --verify
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    gauntlet_parser.add_argument(
        "input",
        help="Path to input file (spec, architecture, policy, code)",
    )
    gauntlet_parser.add_argument(
        "--input-type", "-t",
        choices=["spec", "architecture", "policy", "code", "strategy", "contract"],
        default="spec",
        help="Type of input document (default: spec)",
    )
    gauntlet_parser.add_argument(
        "--agents", "-a",
        default="anthropic-api,openai-api",
        help="Comma-separated agents for stress-testing",
    )
    gauntlet_parser.add_argument(
        "--profile", "-p",
        choices=["default", "quick", "thorough", "code", "policy", "gdpr", "hipaa", "ai_act", "security", "sox"],
        default="default",
        help="Pre-configured test profile (default: default)",
    )
    try:
        from aragora.gauntlet.personas import list_personas
        persona_choices = sorted(list_personas())
    except Exception:
        persona_choices = ["gdpr", "hipaa", "ai_act", "security", "sox"]
    gauntlet_parser.add_argument(
        "--persona",
        choices=persona_choices,
        help="Regulatory persona for compliance-focused stress testing",
    )
    gauntlet_parser.add_argument(
        "--rounds", "-r",
        type=int,
        help="Number of deep audit rounds (overrides profile)",
    )
    gauntlet_parser.add_argument(
        "--timeout",
        type=int,
        help="Maximum duration in seconds (overrides profile)",
    )
    gauntlet_parser.add_argument(
        "--output", "-o",
        help="Output path for Decision Receipt",
    )
    gauntlet_parser.add_argument(
        "--format", "-f",
        choices=["json", "md", "html"],
        help="Output format (default: inferred from extension or html)",
    )
    gauntlet_parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable formal verification (Z3/Lean)",
    )
    gauntlet_parser.add_argument(
        "--no-redteam",
        action="store_true",
        help="Disable red-team attacks",
    )
    gauntlet_parser.add_argument(
        "--no-probing",
        action="store_true",
        help="Disable capability probing",
    )
    gauntlet_parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Disable deep audit",
    )
    gauntlet_parser.set_defaults(func=cmd_gauntlet)

    # Badge command (generate README badges)
    badge_parser = subparsers.add_parser(
        "badge",
        help="Generate Aragora badge for your README",
        description="Generate shareable badges to show your project uses Aragora.",
    )
    badge_parser.add_argument(
        "--type", "-t",
        choices=["reviewed", "consensus", "gauntlet"],
        default="reviewed",
        help="Badge type: reviewed (blue), consensus (green), gauntlet (orange)",
    )
    badge_parser.add_argument(
        "--style", "-s",
        choices=["flat", "flat-square", "for-the-badge", "plastic"],
        default="flat",
        help="Badge style (default: flat)",
    )
    badge_parser.add_argument(
        "--repo", "-r",
        help="Link to specific repo (default: aragora repo)",
    )
    badge_parser.set_defaults(func=cmd_badge)

    # Batch command (process multiple debates)
    batch_parser = subparsers.add_parser(
        "batch",
        help="Process multiple debates from a file",
        description="""
Run multiple debates from a JSONL or JSON file.

Input file format (JSONL - one JSON object per line):
    {"question": "Design a rate limiter", "agents": "anthropic-api,openai-api"}
    {"question": "Implement caching", "rounds": 4}
    {"question": "Security review", "priority": 10}

Or JSON array:
    [{"question": "Topic 1"}, {"question": "Topic 2"}]

Examples:
    aragora batch debates.jsonl
    aragora batch debates.json --server --wait
    aragora batch debates.jsonl --output results.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    batch_parser.add_argument(
        "input",
        help="Path to JSONL or JSON file with debate items",
    )
    batch_parser.add_argument(
        "--server", "-s",
        action="store_true",
        help="Submit to server batch API instead of processing locally",
    )
    batch_parser.add_argument(
        "--url", "-u",
        default="http://localhost:8080",
        help="Server URL (default: http://localhost:8080)",
    )
    batch_parser.add_argument(
        "--token", "-t",
        help="API authentication token",
    )
    batch_parser.add_argument(
        "--webhook", "-w",
        help="Webhook URL for completion notification",
    )
    batch_parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for batch completion (server mode only)",
    )
    batch_parser.add_argument(
        "--agents", "-a",
        default="anthropic-api,openai-api",
        help="Default agents for items without agents specified",
    )
    batch_parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=3,
        help="Default rounds for items without rounds specified",
    )
    batch_parser.add_argument(
        "--output", "-o",
        help="Output path for results JSON (local mode only)",
    )
    batch_parser.set_defaults(func=cmd_batch)

    # MCP Server command
    mcp_parser = subparsers.add_parser(
        "mcp-server",
        help="Run the MCP (Model Context Protocol) server",
        description="""
Run the Aragora MCP server for integration with Claude and other MCP clients.

The MCP server exposes Aragora's capabilities as tools:
- run_debate: Run multi-agent debates
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
