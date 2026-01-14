"""
Demo fixtures for seeding databases.

Provides sample consensus data for search functionality.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from aragora.memory.consensus import ConsensusMemory

logger = logging.getLogger(__name__)

# Available demo domains
DEMO_DOMAINS = [
    "architecture",
    "security",
    "performance",
    "testing",
    "design",
    "debugging",
    "api",
    "database",
]


def get_demo_domains() -> list[str]:
    """Return list of available demo domains."""
    return DEMO_DOMAINS.copy()


def get_demo_records() -> list[dict]:
    """Load and return all demo consensus records."""
    fixtures_dir = Path(__file__).parent
    demo_file = fixtures_dir / "demo_consensus.json"

    if not demo_file.exists():
        logger.warning(f"Demo consensus file not found: {demo_file}")
        return []

    try:
        with open(demo_file) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load demo consensus: {e}")
        return []


def get_demo_records_by_domain(domain: str) -> list[dict]:
    """Get demo records filtered by domain."""
    records = get_demo_records()
    return [r for r in records if r.get("domain") == domain]


def get_demo_statistics() -> dict:
    """Get statistics about demo data."""
    records = get_demo_records()
    domains: dict[str, int] = {}
    strengths = {"strong": 0, "medium": 0, "weak": 0}

    for record in records:
        domain = record.get("domain", "unknown")
        domains[domain] = domains.get(domain, 0) + 1
        strength = record.get("strength", "medium")
        if strength in strengths:
            strengths[strength] += 1

    return {
        "total_records": len(records),
        "domains": domains,
        "by_strength": strengths,
        "avg_confidence": (
            sum(r.get("confidence", 0) for r in records) / len(records) if records else 0
        ),
    }


def load_demo_consensus(consensus_memory: Optional["ConsensusMemory"] = None) -> int:
    """
    Load demo consensus data into the database.

    Args:
        consensus_memory: Optional ConsensusMemory instance.
                         If None, creates one with default path.

    Returns:
        Number of records seeded.
    """
    fixtures_dir = Path(__file__).parent
    demo_file = fixtures_dir / "demo_consensus.json"

    if not demo_file.exists():
        logger.warning(f"Demo consensus file not found: {demo_file}")
        return 0

    try:
        with open(demo_file) as f:
            demos = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load demo consensus: {e}")
        return 0

    # Import ConsensusMemory if not provided
    if consensus_memory is None:
        try:
            from aragora.memory.consensus import ConsensusMemory

            consensus_memory = ConsensusMemory()
        except ImportError:
            logger.warning("ConsensusMemory not available")
            return 0

    # Check if already seeded
    try:
        stats = consensus_memory.get_statistics()
        logger.info(f"Current consensus stats: {stats}")
        total = stats.get("total_consensus", 0)
        if total > 0:
            logger.info(f"Consensus memory already has {total} topics, skipping seed")
            return 0
        logger.info("Database is empty, proceeding with seeding")
    except Exception as e:
        logger.warning(f"Could not check existing data: {e}, proceeding with seeding")

    # Seed the demos
    seeded = 0
    logger.info(f"Attempting to seed {len(demos)} demo records")

    try:
        from aragora.memory.consensus import ConsensusStrength

        for i, demo in enumerate(demos):
            try:
                # Map strength string to enum
                strength_map = {
                    "strong": ConsensusStrength.STRONG,
                    "medium": ConsensusStrength.MODERATE,  # JSON uses "medium", enum uses MODERATE
                    "moderate": ConsensusStrength.MODERATE,
                    "weak": ConsensusStrength.WEAK,
                }
                strength = strength_map.get(
                    demo.get("strength", "medium"), ConsensusStrength.MODERATE
                )

                logger.debug(f"Seeding demo {i + 1}: {demo['topic'][:50]}...")
                consensus_memory.store_consensus(
                    topic=demo["topic"],
                    conclusion=demo["conclusion"],
                    strength=strength,
                    confidence=demo.get("confidence", 0.7),
                    participating_agents=demo.get("participants", []),
                    agreeing_agents=demo.get("participants", []),
                    domain=demo.get("domain", "general"),
                )
                seeded += 1
                logger.debug(f"Successfully seeded demo {i + 1}")
            except Exception as e:
                logger.warning(
                    f"Failed to seed demo {i + 1} '{demo.get('topic', 'unknown')[:30]}': {e}"
                )

        logger.info(f"Seeded {seeded}/{len(demos)} demo consensus records")

    except ImportError as e:
        logger.warning(f"ConsensusStrength not available: {e}")

    return seeded


def ensure_demo_data():
    """
    Ensure demo data is loaded. Safe to call multiple times.

    Called automatically on server startup if database is empty.
    """
    try:
        seeded = load_demo_consensus()
        if seeded > 0:
            logger.info(f"Demo data initialized: {seeded} consensus records")
    except Exception as e:
        logger.warning(f"Failed to ensure demo data: {e}")
