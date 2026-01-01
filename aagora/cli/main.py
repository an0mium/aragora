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

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aagora.agents.base import create_agent
from aagora.debate.orchestrator import Arena, DebateProtocol
from aagora.memory.store import CritiqueStore
from aagora.core import Environment


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

    # Run debate
    arena = Arena(env, agents, protocol, memory)
    result = await arena.run()

    # Store result
    if memory:
        memory.store_debate(result)

    return result


def cmd_ask(args):
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


def cmd_stats(args):
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


def cmd_patterns(args):
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


def cmd_demo(args):
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


def cmd_templates(args):
    """Handle 'templates' command - list available debate templates."""
    from aagora.templates import list_templates

    templates = list_templates()

    print("\n" + "=" * 60)
    print("📋 AVAILABLE DEBATE TEMPLATES")
    print("=" * 60 + "\n")

    for t in templates:
        print(f"[{t['type']}] {t['name']}")
        print(f"  {t['description'][:60]}...")
        print(f"  Agents: {t['agents']}, Domain: {t['domain']}")
        print()


def cmd_improve(args):
    """Handle 'improve' command - self-improvement mode."""
    print("\n" + "=" * 60)
    print("🔧 SELF-IMPROVEMENT MODE")
    print("=" * 60)
    print(f"\nTarget: {args.path or 'current directory'}")
    print(f"Focus: {args.focus or 'general improvements'}")
    print()

    # This is a placeholder - full implementation would use SelfImprover
    print("⚠️  Self-improvement mode is experimental.")
    print("   Use 'aagora ask' to debate specific improvements.")
    print()

    if args.analyze:
        from aagora.tools.code import CodeReader

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


def main():
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

    # Improve command (self-improvement mode)
    improve_parser = subparsers.add_parser("improve", help="Self-improvement mode")
    improve_parser.add_argument("--path", "-p", help="Path to codebase (default: current dir)")
    improve_parser.add_argument("--focus", "-f", help="Focus area for improvements")
    improve_parser.add_argument("--analyze", "-a", action="store_true", help="Analyze codebase structure")
    improve_parser.set_defaults(func=cmd_improve)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
