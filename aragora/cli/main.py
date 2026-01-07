#!/usr/bin/env python3
"""
Agora CLI - Multi-Agent Debate Framework

Usage:
    agora ask "Design a rate limiter" --agents codex,claude --rounds 3
    agora debate --task "Implement auth system" --agents codex,claude,openai
    agora stats
"""

import argparse
import asyncio
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
                    from aragora.server.stream import StreamEmitter
                    return StreamEmitter(server_url=server_url)
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
            model_type=agent_type,
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
        consensus=consensus,
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
    arena = Arena(env, agents, protocol, memory, event_emitter=event_emitter)
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

    demo_tasks = {
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

    print("\n" + "=" * 60)
    print("🎭 AAGORA DEMO - Multi-Agent Debate")
    print("=" * 60)
    print(f"\n📋 Task: {demo['task'][:80]}...")
    print(f"🤖 Agents: {demo['agents']}")
    print(f"🔄 Rounds: {demo['rounds']}")
    print("\n" + "-" * 60)
    print("Starting debate...")
    print("-" * 60 + "\n")

    start = time.time()

    result = asyncio.run(
        run_debate(
            task=demo["task"],
            agents_str=demo["agents"],
            rounds=demo["rounds"],
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agora - Multi-Agent Debate Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  agora ask "Design a rate limiter" --agents codex,claude
  agora ask "Implement auth" --agents codex,claude,openai --rounds 4
  agora stats
  agora patterns --type security
        """,
    )

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
        help="Comma-separated agents (codex,claude,openai). Use agent:role for specific roles.",
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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
