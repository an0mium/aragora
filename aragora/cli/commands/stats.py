"""
Statistics and data inspection CLI commands.

Contains commands for viewing debate stats, patterns, memory tiers,
ELO ratings, and cross-pollination event diagnostics.
"""

import argparse
import logging

from aragora.memory.store import CritiqueStore

logger = logging.getLogger(__name__)


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

    # Cross-pollination statistics (v2.0.3)
    _print_cross_pollination_stats(args)


def _print_cross_pollination_stats(args: argparse.Namespace) -> None:
    """Print cross-pollination statistics."""
    print("\nCross-Pollination Statistics (v2.0.3)")
    print("=" * 40)

    # ELO and learning efficiency
    try:
        from aragora.ranking.elo import get_elo_store

        elo = get_elo_store()
        leaderboard = elo.get_leaderboard(limit=5)
        if leaderboard:
            print("\nTop 5 Agents by ELO:")
            for i, entry in enumerate(leaderboard, 1):
                name = entry.agent_name
                rating = entry.elo
                # Get learning efficiency
                efficiency = elo.get_learning_efficiency(name)
                category = efficiency.get("learning_category", "unknown")
                print(f"  {i}. {name}: {rating:.0f} ELO ({category} learner)")
    except ImportError as e:
        logger.warning("ELO module not available: %s", e)
        print(f"  ELO system: unavailable ({e})")
    except (KeyError, TypeError, OSError) as e:
        logger.warning("ELO system error: %s", e)
        print(f"  ELO system: unavailable ({e})")

    # RLM cache stats
    try:
        from aragora.rlm.bridge import RLMHierarchyCache

        cache = RLMHierarchyCache()
        cache_stats = cache.get_stats()
        hits = cache_stats.get("hits", 0)
        misses = cache_stats.get("misses", 0)
        hit_rate = cache_stats.get("hit_rate", 0.0)
        print(f"\nRLM Cache: {hits} hits, {misses} misses ({hit_rate:.1%} hit rate)")
    except ImportError:
        logger.warning("RLM module not available")
        print("\nRLM Cache: not initialized")
    except (KeyError, TypeError, AttributeError) as e:
        logger.warning("RLM cache unavailable: %s", e)
        print("\nRLM Cache: not initialized")

    # Calibration stats
    try:
        from aragora.ranking.calibration import CalibrationTracker

        CalibrationTracker()
        # Get summary for any available agents
        print("\nCalibration: enabled")
    except ImportError:
        logger.warning("CalibrationTracker module not available")
        print("\nCalibration: unavailable")
    except (OSError, TypeError) as e:
        logger.warning("Calibration unavailable: %s", e)
        print("\nCalibration: unavailable")


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


def cmd_memory(args: argparse.Namespace) -> None:
    """Handle 'memory' command - inspect ContinuumMemory tiers."""
    from aragora.memory.continuum import ContinuumMemory, MemoryTier
    from aragora.persistence.db_config import DatabaseType, get_db_path

    db_path = getattr(args, "db", None) or get_db_path(DatabaseType.CONTINUUM_MEMORY)
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
                bar = "\u2588" * min(count // 10, 30) if count > 0 else ""
                print(f"  {tier:8}: {count:5} {bar}")

        tier_metrics = memory.get_tier_metrics()
        if tier_metrics:
            print("\nTier Metrics:")
            for tier, metrics in tier_metrics.items():
                if isinstance(metrics, dict):
                    promotions = metrics.get("promotions", 0)
                    demotions = metrics.get("demotions", 0)
                    if promotions or demotions:
                        print(f"  {tier}: \u2191{promotions} promotions, \u2193{demotions} demotions")

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
    from aragora.persistence.db_config import DatabaseType, get_db_path
    from aragora.ranking.elo import EloSystem

    db_path = getattr(args, "db", None) or get_db_path(DatabaseType.ELO)
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

        except (KeyError, ValueError) as e:
            logger.warning("Agent lookup failed for '%s': %s", agent, e)
            print(f"Agent not found: {agent}")
        except (OSError, TypeError) as e:
            logger.warning("ELO database error for agent '%s': %s", agent, e)
            print(f"Agent not found: {agent}")


def cmd_cross_pollination(args: argparse.Namespace) -> None:
    """Handle 'cross-pollination' command - view event system diagnostics."""
    import json as json_module

    from aragora.events.cross_subscribers import get_cross_subscriber_manager

    manager = get_cross_subscriber_manager()
    action = getattr(args, "action", "stats")
    output_json = getattr(args, "json", False)

    if action == "stats":
        stats = manager.get_stats()

        if output_json:
            print(json_module.dumps(stats, indent=2, default=str))
            return

        print("\nCross-Pollination Event Statistics")
        print("=" * 70)
        print(f"{'Handler':<25} {'Events':>8} {'Failed':>8} {'Avg (ms)':>10} {'Enabled':>8}")
        print("-" * 70)

        total_events = 0
        total_failed = 0

        for name, data in stats.items():
            total_events += data["events_processed"]
            total_failed += data["events_failed"]
            latency = data.get("latency_ms", {})
            avg_ms = latency.get("avg", 0)
            enabled = "Yes" if data["enabled"] else "No"
            print(
                f"{name:<25} {data['events_processed']:>8} {data['events_failed']:>8} "
                f"{avg_ms:>10.3f} {enabled:>8}"
            )

        print("-" * 70)
        print(f"{'TOTAL':<25} {total_events:>8} {total_failed:>8}")
        print()

    elif action == "subscribers":
        stats = manager.get_stats()

        if output_json:
            print(json_module.dumps(list(stats.keys()), indent=2))
            return

        print("\nRegistered Cross-Subscribers")
        print("=" * 50)

        for i, (name, data) in enumerate(stats.items(), 1):
            status = "[+]" if data["enabled"] else "[-]"
            last = data.get("last_event", "never")
            if last and last != "never":
                last = last[:19]  # Truncate to datetime
            print(f"  {status} {i}. {name}")
            print(f"      Last event: {last}")
            print()

    elif action == "reset":
        manager.reset_stats()
        print("Cross-pollination statistics reset successfully.")
