"""
Demo fixtures for seeding databases.

Provides sample consensus data for search functionality.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_demo_consensus(consensus_memory: Optional[object] = None) -> int:
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
        stats = consensus_memory.get_stats()
        if stats.get("total_topics", 0) > 0:
            logger.info(f"Consensus memory already has {stats['total_topics']} topics, skipping seed")
            return 0
    except Exception:
        pass

    # Seed the demos
    seeded = 0
    try:
        from aragora.memory.consensus import ConsensusStrength

        for demo in demos:
            try:
                # Map strength string to enum
                strength_map = {
                    "strong": ConsensusStrength.STRONG,
                    "medium": ConsensusStrength.MEDIUM,
                    "weak": ConsensusStrength.WEAK,
                }
                strength = strength_map.get(demo.get("strength", "medium"), ConsensusStrength.MEDIUM)

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
            except Exception as e:
                logger.warning(f"Failed to seed demo: {e}")

        logger.info(f"Seeded {seeded} demo consensus records")

    except ImportError:
        logger.warning("ConsensusStrength not available")

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
