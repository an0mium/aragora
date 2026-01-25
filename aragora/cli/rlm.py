"""
CLI commands for RLM (Recursive Language Models) operations.

Usage:
    aragora rlm compress <file> [--output <path>]
    aragora rlm query <query> --context <path>
    aragora rlm stats
    aragora rlm clear-cache

Commands:
    compress    Compress content into hierarchical context
    query       Query compressed context with RLM
    stats       Show RLM cache statistics
    clear-cache Clear the compression cache
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any


def cmd_compress(args: argparse.Namespace) -> int:
    """Handle 'rlm compress' command."""
    from aragora.rlm.bridge import AragoraRLM
    from aragora.rlm.types import RLMConfig, AbstractionLevel

    # Read input content
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    content = input_path.read_text()
    print(f"Read {len(content)} characters from {input_path}")

    # Determine source type
    source_type = args.type
    if not source_type:
        ext = input_path.suffix.lower()
        if ext in {".py", ".js", ".ts", ".go", ".rs", ".java"}:
            source_type = "code"
        elif ext in {".md", ".txt", ".rst"}:
            source_type = "text"
        else:
            source_type = "text"

    print(f"Source type: {source_type}")

    # Create RLM and compress
    config = RLMConfig(
        cache_compressions=not args.no_cache,
        max_depth=args.levels,
    )
    rlm = AragoraRLM(aragora_config=config)

    print("Compressing...")

    async def do_compress():
        compression = await rlm._compressor.compress(
            content,
            source_type=source_type,
            max_levels=args.levels,
        )
        return compression

    compression = asyncio.run(do_compress())

    # Show results
    print("\n" + "=" * 60)
    print("COMPRESSION RESULTS")
    print("=" * 60)
    print(f"Original tokens: {compression.original_tokens:,}")

    compressed_by_level = compression.compressed_tokens
    for level in [AbstractionLevel.DETAILED, AbstractionLevel.SUMMARY, AbstractionLevel.ABSTRACT]:
        if level in compressed_by_level:
            tokens = compressed_by_level[level]
            ratio = compression.compression_ratio.get(level, 0)
            print(f"  {level.name:10}: {tokens:,} tokens ({ratio:.1%} of original)")

    print(f"Estimated fidelity: {compression.estimated_fidelity:.1%}")
    print(f"Sub-calls made: {compression.sub_calls_made}")
    print(f"Cache hits: {compression.cache_hits}")
    print(f"Time: {compression.time_seconds:.2f}s")

    if compression.key_topics_extracted:
        print(f"\nKey topics: {', '.join(compression.key_topics_extracted[:10])}")

    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "original_tokens": compression.original_tokens,
            "compressed_tokens": {k.name: v for k, v in compressed_by_level.items()},
            "compression_ratio": {k.name: v for k, v in compression.compression_ratio.items()},
            "estimated_fidelity": compression.estimated_fidelity,
            "key_topics": compression.key_topics_extracted,
            "levels": {},
        }

        # Save level contents
        for level, nodes in compression.context.levels.items():
            output_data["levels"][level.name] = [
                {
                    "id": n.id,
                    "content": n.content,
                    "token_count": n.token_count,
                    "key_topics": n.key_topics,
                }
                for n in nodes
            ]

        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nSaved to {output_path}")

    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Handle 'rlm query' command."""
    from aragora.rlm.bridge import AragoraRLM
    from aragora.rlm.types import AbstractionLevel, AbstractionNode, RLMContext

    # Load context
    context_path = Path(args.context)
    if not context_path.exists():
        print(f"Error: Context file not found: {context_path}")
        return 1

    try:
        with open(context_path) as f:
            context_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in context file: {e}")
        return 1

    # Reconstruct RLMContext
    context = RLMContext(
        original_content="[Loaded from file]",
        original_tokens=context_data.get("original_tokens", 0),
        source_type="loaded",
    )

    # Rebuild levels
    for level_name, nodes in context_data.get("levels", {}).items():
        try:
            level = AbstractionLevel[level_name]
        except KeyError:
            continue

        context.levels[level] = []
        for node_data in nodes:
            node = AbstractionNode(
                id=node_data["id"],
                level=level,
                content=node_data["content"],
                token_count=node_data.get("token_count", len(node_data["content"]) // 4),
                key_topics=node_data.get("key_topics", []),
            )
            context.levels[level].append(node)
            context.nodes_by_id[node.id] = node

    print(f"Loaded context with {len(context.nodes_by_id)} nodes")

    # Create RLM and query
    rlm = AragoraRLM()

    print(f"\nQuery: {args.query}")
    print(f"Strategy: {args.strategy}")
    print("\n" + "-" * 60)

    async def do_query():
        if args.stream:
            # Stream query
            from aragora.rlm.types import RLMStreamEventType

            print("Streaming...")
            async for event in rlm.query_stream(args.query, context, args.strategy):
                if event.event_type == RLMStreamEventType.LEVEL_ENTERED:
                    print(f"  Entering level: {event.level.name if event.level else '?'}")
                elif event.event_type == RLMStreamEventType.NODE_EXAMINED:
                    print(f"  Examining node: {event.node_id}")
                elif event.event_type == RLMStreamEventType.QUERY_COMPLETE:
                    return event.result
            return None
        else:
            # Regular query
            if args.refine:
                return await rlm.query_with_refinement(
                    args.query,
                    context,
                    args.strategy,
                    max_iterations=args.max_iterations,
                )
            else:
                return await rlm.query(args.query, context, args.strategy)

    result = asyncio.run(do_query())

    if not result:
        print("Query failed")
        return 1

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result.answer)

    print("\n" + "-" * 60)
    print(f"Ready: {result.ready}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Iteration: {result.iteration}")
    print(f"Tokens processed: {result.tokens_processed:,}")
    print(f"Sub-calls made: {result.sub_calls_made}")

    if result.nodes_examined:
        print(f"Nodes examined: {', '.join(result.nodes_examined[:5])}")

    if result.refinement_history:
        print(f"\nRefinement history ({len(result.refinement_history)} iterations):")
        for i, history in enumerate(result.refinement_history):
            print(f"  [{i + 1}] {history[:100]}...")

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Handle 'rlm stats' command."""
    from aragora.rlm.compressor import get_compression_cache_stats

    stats = get_compression_cache_stats()

    print("\n" + "=" * 60)
    print("RLM CACHE STATISTICS")
    print("=" * 60)

    print(f"\nCache size: {stats.get('size', 0)} entries")
    print(f"Max size: {stats.get('max_size', 1000)} entries")

    hits = stats.get("hits", 0)
    misses = stats.get("misses", 0)
    hit_rate = stats.get("hit_rate", 0)

    print(f"Cache hits: {hits}")
    print(f"Cache misses: {misses}")
    print(f"Hit rate: {hit_rate:.1%}")

    if stats.get("memory_bytes"):
        mb = stats["memory_bytes"] / (1024 * 1024)
        print(f"Memory usage: {mb:.2f} MB")

    if stats.get("entries"):
        print("\nCached entries:")
        for entry in stats["entries"][:10]:  # Show first 10
            key = entry.get("key", "?")[:30]
            tokens = entry.get("original_tokens", 0)
            age = entry.get("age_seconds", 0)
            print(f"  {key}... ({tokens:,} tokens, {age:.0f}s old)")

    return 0


def cmd_clear_cache(args: argparse.Namespace) -> int:
    """Handle 'rlm clear-cache' command."""
    from aragora.rlm.compressor import clear_compression_cache, get_compression_cache_stats

    stats_before = get_compression_cache_stats()
    count_before = stats_before["size"]

    clear_compression_cache()

    print(f"Cleared {count_before} cached compressions")
    return 0


def cmd_rlm(args: argparse.Namespace) -> int:
    """Handle 'rlm' command."""
    action = getattr(args, "action", None)

    if action == "compress":
        return cmd_compress(args)
    elif action == "query":
        return cmd_query(args)
    elif action == "stats":
        return cmd_stats(args)
    elif action == "clear-cache":
        return cmd_clear_cache(args)
    else:
        print("Usage: aragora rlm <command> [options]")
        print("\nCommands:")
        print("  compress    Compress content into hierarchical context")
        print("  query       Query compressed context with RLM")
        print("  stats       Show RLM cache statistics")
        print("  clear-cache Clear the compression cache")
        return 0


def create_rlm_parser(subparsers: Any) -> None:
    """Add RLM subcommand to main parser."""
    rlm_parser = subparsers.add_parser(
        "rlm",
        help="RLM (Recursive Language Models) operations",
        description="""
Recursive Language Models (RLM) for efficient processing of long contexts.

RLM compresses long documents into hierarchical abstractions and enables
efficient querying through recursive sub-calls. Based on Prime Intellect's
iterative refinement protocol (arXiv:2512.24601).

Examples:
    aragora rlm compress document.txt --output context.json
    aragora rlm query "What is the main topic?" --context context.json
    aragora rlm stats
    aragora rlm clear-cache
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    rlm_subparsers = rlm_parser.add_subparsers(dest="action", help="RLM commands")

    # Compress command
    compress_parser = rlm_subparsers.add_parser(
        "compress",
        help="Compress content into hierarchical context",
        description="Compress a document into a hierarchical abstraction hierarchy.",
    )
    compress_parser.add_argument(
        "input",
        help="Input file to compress",
    )
    compress_parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file for compressed context",
    )
    compress_parser.add_argument(
        "--type",
        "-t",
        choices=["text", "code", "debate", "document"],
        help="Content type (auto-detected if not specified)",
    )
    compress_parser.add_argument(
        "--levels",
        "-l",
        type=int,
        default=4,
        help="Number of abstraction levels (default: 4)",
    )
    compress_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable compression caching",
    )

    # Query command
    query_parser = rlm_subparsers.add_parser(
        "query",
        help="Query compressed context with RLM",
        description="Query a compressed context using RLM's hierarchical navigation.",
    )
    query_parser.add_argument(
        "query",
        help="Query to answer",
    )
    query_parser.add_argument(
        "--context",
        "-c",
        required=True,
        help="Path to compressed context JSON file",
    )
    query_parser.add_argument(
        "--strategy",
        "-s",
        choices=["auto", "peek", "grep", "partition_map", "hierarchical"],
        default="auto",
        help="Decomposition strategy (default: auto)",
    )
    query_parser.add_argument(
        "--refine",
        "-r",
        action="store_true",
        help="Enable iterative refinement",
    )
    query_parser.add_argument(
        "--max-iterations",
        "-m",
        type=int,
        default=3,
        help="Maximum refinement iterations (default: 3)",
    )
    query_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream query progress",
    )

    # Stats command
    rlm_subparsers.add_parser(
        "stats",
        help="Show RLM cache statistics",
        description="Display statistics about the RLM compression cache.",
    )

    # Clear cache command
    rlm_subparsers.add_parser(
        "clear-cache",
        help="Clear the compression cache",
        description="Clear all cached compressions from memory.",
    )

    rlm_parser.set_defaults(func=cmd_rlm)
