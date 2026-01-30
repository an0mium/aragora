"""
Tool and utility CLI commands.

Contains commands for operational modes, templates, self-improvement,
and codebase context building.
"""

import argparse
import asyncio
from pathlib import Path

from aragora.modes import ModeRegistry


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


def cmd_templates(args: argparse.Namespace) -> None:
    """Handle 'templates' command - list available debate templates."""
    from aragora.templates import list_templates

    templates = list_templates()

    print("\n" + "=" * 60)
    print("\U0001f4cb AVAILABLE DEBATE TEMPLATES")
    print("=" * 60 + "\n")

    for t in templates:
        print(f"[{t['type']}] {t['name']}")
        print(f"  {t['description'][:60]}...")
        print(f"  Agents: {t['agents']}, Domain: {t['domain']}")
        print()


def cmd_improve(args: argparse.Namespace) -> None:
    """Handle 'improve' command - self-improvement mode."""
    print("\n" + "=" * 60)
    print("\U0001f527 SELF-IMPROVEMENT MODE")
    print("=" * 60)
    print(f"\nTarget: {args.path or 'current directory'}")
    print(f"Focus: {args.focus or 'general improvements'}")
    print()

    # This is a placeholder - full implementation would use SelfImprover
    print("\u26a0\ufe0f  Self-improvement mode is experimental.")
    print("   Use 'aragora ask' to debate specific improvements.")
    print()

    if args.analyze:
        from aragora.tools.code import CodeReader

        reader = CodeReader(args.path or ".")
        tree = reader.get_file_tree(max_depth=2)

        print("\U0001f4c2 Codebase structure:")

        def print_tree(t, indent=0):
            for k, v in sorted(t.items()):
                if isinstance(v, dict):
                    print("  " * indent + f"\U0001f4c1 {k}")
                    print_tree(v, indent + 1)
                else:
                    print("  " * indent + f"\U0001f4c4 {k} ({v} bytes)")

        print_tree(tree)


def cmd_context(args: argparse.Namespace) -> None:
    """Handle 'context' command - build codebase context for RLM."""
    from aragora.rlm.codebase_context import CodebaseContextBuilder

    root = Path(args.path or ".").resolve()
    include_tests: bool | None = None
    if args.include_tests:
        include_tests = True
    elif args.exclude_tests:
        include_tests = False

    builder = CodebaseContextBuilder(
        root_path=root,
        max_context_bytes=args.max_bytes or 0,
        include_tests=include_tests,
        full_corpus=args.full_corpus,
    )

    async def _run() -> None:
        index = await builder.build_index()
        print("\n" + "=" * 60)
        print("\U0001f4da CODEBASE CONTEXT")
        print("=" * 60)
        print(f"Root: {index.root_path}")
        print(
            f"Files: {index.total_files} | Lines: {index.total_lines} | "
            f"Bytes: {index.total_bytes} | ~Tokens: {index.total_tokens_estimate}"
        )
        print(f"Index build time: {index.build_time_seconds:.2f}s")

        if args.rlm:
            print("\n\U0001f50d Building RLM context (TRUE RLM preferred)...")
            context = await builder.build_rlm_context()
            if context is None:
                print("\u26a0\ufe0f  RLM context unavailable (missing RLM package or disabled).")
            else:
                print("\u2705 RLM context ready.")

        if args.summary_out or args.preview:
            context = await builder.build_debate_context()
            if args.summary_out:
                output_path = Path(args.summary_out).resolve()
                output_path.write_text(context, encoding="utf-8")
                print(f"\n\U0001f4dd Context summary written to: {output_path}")
            if args.preview:
                print("\n\U0001f4c4 Context preview (first 40 lines):")
                for line in context.splitlines()[:40]:
                    print(line)
                if len(context.splitlines()) > 40:
                    print("... (truncated)")

        print()

    asyncio.run(_run())
