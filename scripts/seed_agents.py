#!/usr/bin/env python3
"""
Seed the ELO database with default agents.

This script populates the ratings table with common AI agents
so the leaderboard displays useful data from the start.

Run: python scripts/seed_agents.py
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Try to import aragora modules
try:
    from aragora.ranking.elo import EloSystem, AgentRating, DEFAULT_ELO

    RANKING_AVAILABLE = True
except ImportError:
    RANKING_AVAILABLE = False
    DEFAULT_ELO = 1500.0
    logger.warning("EloSystem not available, will create minimal database")


# Default agents to seed - major AI providers
DEFAULT_AGENTS = [
    # Anthropic
    ("claude-opus", "anthropic"),
    ("claude-sonnet", "anthropic"),
    ("claude-haiku", "anthropic"),
    # OpenAI
    ("gpt-4o", "openai"),
    ("gpt-4-turbo", "openai"),
    ("o1", "openai"),
    ("o1-mini", "openai"),
    # Google
    ("gemini-pro", "google"),
    ("gemini-ultra", "google"),
    # xAI
    ("grok-2", "xai"),
    ("grok-beta", "xai"),
    # Meta (via OpenRouter)
    ("llama-3.1-405b", "meta"),
    ("llama-3.1-70b", "meta"),
    # Mistral
    ("mistral-large", "mistral"),
    ("codestral", "mistral"),
    # DeepSeek
    ("deepseek-v3", "deepseek"),
    ("deepseek-coder", "deepseek"),
    # Alibaba
    ("qwen-2.5-72b", "alibaba"),
    # Cohere
    ("command-r-plus", "cohere"),
]


def seed_with_elo_system(db_path: Path, agents: list[tuple[str, str]], force: bool = False) -> int:
    """Seed agents using the EloSystem class."""
    elo_system = EloSystem(str(db_path))
    seeded = 0

    for agent_name, provider in agents:
        try:
            # Check if agent already exists
            existing = elo_system.get_rating(agent_name, use_cache=False)

            # Skip if agent has actual data (debates_count > 0)
            if existing.debates_count > 0 and not force:
                logger.info(f"Skipping {agent_name}: has {existing.debates_count} debates")
                continue

            # Create and save default rating
            rating = AgentRating(
                agent_name=agent_name,
                elo=DEFAULT_ELO,
                domain_elos={},
                wins=0,
                losses=0,
                draws=0,
                debates_count=0,
                critiques_accepted=0,
                critiques_total=0,
                updated_at=datetime.now().isoformat(),
            )

            # Use internal save method (acceptable for seed scripts)
            elo_system._save_rating(rating)
            logger.info(f"Seeded: {agent_name} ({provider}) at ELO {DEFAULT_ELO}")
            seeded += 1

        except Exception as e:
            logger.warning(f"Failed to seed {agent_name}: {e}")

    return seeded


def seed_minimal_database(db_path: Path, agents: list[tuple[str, str]]) -> int:
    """Seed agents directly with SQLite (fallback if EloSystem unavailable)."""
    import sqlite3

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)

    # Create minimal schema
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ratings (
            agent_name TEXT PRIMARY KEY,
            elo REAL DEFAULT 1500.0,
            domain_elos TEXT DEFAULT '{}',
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            draws INTEGER DEFAULT 0,
            debates_count INTEGER DEFAULT 0,
            critiques_accepted INTEGER DEFAULT 0,
            critiques_total INTEGER DEFAULT 0,
            calibration_correct INTEGER DEFAULT 0,
            calibration_total INTEGER DEFAULT 0,
            calibration_brier_sum REAL DEFAULT 0.0,
            updated_at TEXT
        )
    """
    )

    seeded = 0
    now = datetime.now().isoformat()

    for agent_name, provider in agents:
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO ratings (agent_name, elo, updated_at)
                VALUES (?, ?, ?)
                """,
                (agent_name, DEFAULT_ELO, now),
            )
            seeded += 1
            logger.info(f"Seeded: {agent_name} ({provider}) at ELO {DEFAULT_ELO}")
        except Exception as e:
            logger.warning(f"Failed to seed {agent_name}: {e}")

    conn.commit()
    conn.close()
    return seeded


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Seed default agents into ELO database")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Path to ELO database (default: .nomic/elo.db)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite agents that already have match history",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be seeded without making changes",
    )
    args = parser.parse_args()

    # Determine database path
    base_dir = Path(__file__).parent.parent
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = base_dir / ".nomic" / "elo.db"

    logger.info(f"Target database: {db_path}")
    logger.info(f"Agents to seed: {len(DEFAULT_AGENTS)}")

    if args.dry_run:
        logger.info("DRY RUN - no changes will be made")
        for agent_name, provider in DEFAULT_AGENTS:
            logger.info(f"  Would seed: {agent_name} ({provider})")
        return

    # Ensure .nomic directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Seed agents
    if RANKING_AVAILABLE:
        seeded = seed_with_elo_system(db_path, DEFAULT_AGENTS, force=args.force)
    else:
        seeded = seed_minimal_database(db_path, DEFAULT_AGENTS)

    logger.info(f"Seeded {seeded} agents successfully")

    # Print summary
    if seeded > 0:
        logger.info("Leaderboard should now show seeded agents at ELO 1500")
        logger.info("Agents will diverge in ELO as they participate in debates")


if __name__ == "__main__":
    main()
