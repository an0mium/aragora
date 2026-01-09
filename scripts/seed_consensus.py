#!/usr/bin/env python3
"""
Seed the consensus memory database from existing debate data.

This script populates the search database from:
1. .nomic/debate.json and related files
2. .nomic/replays/*/events.json
3. Any stored debate artifacts

Run: python scripts/seed_consensus.py
"""

import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Try to import aragora modules
try:
    from aragora.memory.consensus import ConsensusMemory
    CONSENSUS_AVAILABLE = True
except ImportError:
    CONSENSUS_AVAILABLE = False
    logger.warning("ConsensusMemory not available, will create minimal database")


def load_debate_files(nomic_dir: Path) -> list[dict]:
    """Load all debate JSON files from .nomic directory."""
    debates = []

    # Main debate files
    debate_files = [
        nomic_dir / "debate.json",
        nomic_dir / "design.json",
        nomic_dir / "phase_implement_debate.json",
        nomic_dir / "3_provider_debate_transcript.json",
    ]

    for f in debate_files:
        if f.exists():
            try:
                with open(f) as fp:
                    data = json.load(fp)
                    if data.get("final_answer") or data.get("consensus"):
                        debates.append({
                            "source": str(f),
                            "data": data,
                        })
                        logger.info(f"Loaded: {f.name}")
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")

    # Replay cycle data
    replays_dir = nomic_dir / "replays"
    if replays_dir.exists():
        for cycle_dir in sorted(replays_dir.iterdir()):
            if cycle_dir.is_dir():
                events_file = cycle_dir / "events.json"
                meta_file = cycle_dir / "meta.json"

                if events_file.exists():
                    try:
                        with open(events_file) as fp:
                            events = json.load(fp)
                            # Extract debate events
                            for event in events:
                                if event.get("type") == "debate_end":
                                    debates.append({
                                        "source": str(events_file),
                                        "data": event.get("data", {}),
                                        "cycle": cycle_dir.name,
                                    })
                    except Exception as e:
                        logger.warning(f"Failed to load {events_file}: {e}")

    return debates


def extract_topic(debate_data: dict) -> str:
    """Extract topic from debate data."""
    # Try various fields
    if "topic" in debate_data:
        return debate_data["topic"][:200]
    if "task" in debate_data:
        return debate_data["task"][:200]
    if "final_answer" in debate_data:
        # Extract first line as topic
        answer = debate_data["final_answer"]
        lines = answer.split("\n")
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                return line[:200]
            if line.startswith("##"):
                return line.replace("#", "").strip()[:200]
    return "Unknown topic"


def seed_consensus_memory(debates: list[dict], db_path: Path):
    """Seed the consensus memory database."""
    if CONSENSUS_AVAILABLE:
        memory = ConsensusMemory(str(db_path))
    else:
        # Create minimal SQLite database
        import sqlite3
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS consensus (
                id TEXT PRIMARY KEY,
                topic TEXT NOT NULL,
                conclusion TEXT,
                confidence REAL DEFAULT 0.5,
                strength TEXT DEFAULT 'weak',
                domain TEXT DEFAULT 'general',
                participants TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dissent (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                consensus_id TEXT,
                agent TEXT,
                dissent_text TEXT,
                confidence REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_consensus_topic ON consensus(topic)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_consensus_domain ON consensus(domain)")
        conn.commit()

    seeded = 0
    for debate in debates:
        data = debate["data"]

        # Generate unique ID
        topic = extract_topic(data)
        debate_id = hashlib.sha256(f"{topic}{data.get('timestamp', '')}".encode()).hexdigest()[:16]

        # Extract consensus data
        conclusion = data.get("final_answer", data.get("consensus", ""))
        confidence = data.get("confidence", 0.5)

        # Determine strength
        if confidence >= 0.8:
            strength = "strong"
        elif confidence >= 0.6:
            strength = "medium"
        else:
            strength = "weak"

        # Extract participants
        participants = []
        for msg in data.get("messages", []):
            agent = msg.get("agent", "")
            if agent and agent not in participants:
                participants.append(agent)

        if CONSENSUS_AVAILABLE:
            try:
                from aragora.memory.consensus import ConsensusStrength
                # Map confidence to strength enum
                if confidence >= 0.8:
                    strength_enum = ConsensusStrength.STRONG
                elif confidence >= 0.6:
                    strength_enum = ConsensusStrength.MEDIUM
                else:
                    strength_enum = ConsensusStrength.WEAK

                memory.store_consensus(
                    topic=topic,
                    conclusion=conclusion[:5000],
                    strength=strength_enum,
                    confidence=confidence,
                    participating_agents=participants,
                    agreeing_agents=participants,  # Assume all agreed for consensus
                    domain="architecture",  # Most nomic debates are architecture
                    rounds=len(data.get("messages", [])) // max(len(participants), 1),
                    debate_duration=data.get("duration", 0.0),
                )
                seeded += 1
            except Exception as e:
                logger.warning(f"Failed to store via ConsensusMemory: {e}")
        else:
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO consensus
                       (id, topic, conclusion, confidence, strength, domain, participants)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (debate_id, topic, conclusion[:5000], confidence, strength,
                     "architecture", json.dumps(participants))
                )
                seeded += 1
            except Exception as e:
                logger.warning(f"Failed to store: {e}")

    if not CONSENSUS_AVAILABLE:
        conn.commit()
        conn.close()

    return seeded


def main():
    """Main entry point."""
    base_dir = Path(__file__).parent.parent
    nomic_dir = base_dir / ".nomic"
    db_path = base_dir / "consensus_memory.db"

    logger.info(f"Loading debates from {nomic_dir}")
    debates = load_debate_files(nomic_dir)
    logger.info(f"Found {len(debates)} debates with consensus data")

    if debates:
        logger.info(f"Seeding database at {db_path}")
        seeded = seed_consensus_memory(debates, db_path)
        logger.info(f"Seeded {seeded} consensus records")
    else:
        logger.warning("No debate data found to seed")


if __name__ == "__main__":
    main()
