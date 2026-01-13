"""
Genesis Ledger - Unified provenance tracking for fractal debates and agent evolution.

Provides:
- Immutable recording of debate spawns, merges, and outcomes
- Agent birth, death, and mutation tracking
- Lineage and debate tree queries
- Merkle tree verification for batch integrity
- Export to JSON, Markdown, and HTML
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
from aragora.genesis.database import GenesisDatabase

logger = logging.getLogger(__name__)

from aragora.reasoning.provenance import (
    ProvenanceChain,
    ProvenanceRecord,
    SourceType,
    TransformationType,
    MerkleTree,
)
from aragora.debate.consensus import ConsensusProof
from aragora.genesis.genome import AgentGenome


class GenesisEventType(Enum):
    """Types of events in the genesis ledger."""

    # Debate events
    DEBATE_START = "debate_start"
    DEBATE_END = "debate_end"
    DEBATE_SPAWN = "debate_spawn"  # Sub-debate created
    DEBATE_MERGE = "debate_merge"  # Sub-debate resolved

    # Consensus events
    CONSENSUS_REACHED = "consensus_reached"
    TENSION_DETECTED = "tension_detected"
    TENSION_RESOLVED = "tension_resolved"
    TENSION_UNRESOLVED = "tension_unresolved"

    # Agent evolution events
    AGENT_BIRTH = "agent_birth"  # New genome created
    AGENT_DEATH = "agent_death"  # Genome culled from population
    AGENT_MUTATION = "agent_mutation"  # Genome mutated
    AGENT_CROSSOVER = "agent_crossover"  # Two genomes bred
    FITNESS_UPDATE = "fitness_update"  # Fitness score changed

    # Population events
    POPULATION_CREATED = "population_created"
    POPULATION_EVOLVED = "population_evolved"
    GENERATION_ADVANCE = "generation_advance"


@dataclass
class GenesisEvent:
    """A single event in the genesis ledger."""

    event_id: str
    event_type: GenesisEventType
    timestamp: datetime
    parent_event_id: Optional[str] = None
    content_hash: str = ""
    data: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute content hash from event data."""
        content = json.dumps(
            {
                "event_id": self.event_id,
                "event_type": self.event_type.value,
                "timestamp": self.timestamp.isoformat(),
                "parent_event_id": self.parent_event_id,
                "data": self.data,
            },
            sort_keys=True,
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "parent_event_id": self.parent_event_id,
            "content_hash": self.content_hash,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenesisEvent":
        return cls(
            event_id=data["event_id"],
            event_type=GenesisEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            parent_event_id=data.get("parent_event_id"),
            content_hash=data.get("content_hash", ""),
            data=data.get("data", {}),
        )


@dataclass
class FractalTree:
    """Tree structure of debates and sub-debates."""

    root_id: str
    nodes: dict[str, dict] = field(default_factory=dict)  # debate_id -> node data

    def add_node(
        self,
        debate_id: str,
        parent_id: Optional[str],
        tension: Optional[str] = None,
        success: bool = False,
        depth: int = 0,
    ) -> None:
        self.nodes[debate_id] = {
            "debate_id": debate_id,
            "parent_id": parent_id,
            "tension": tension,
            "success": success,
            "depth": depth,
            "children": [],
        }
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id]["children"].append(debate_id)

    def get_children(self, debate_id: str) -> list[str]:
        if debate_id in self.nodes:
            return self.nodes[debate_id]["children"]
        return []

    def to_dict(self) -> dict:
        def build_subtree(node_id: str) -> dict:
            node = self.nodes.get(node_id, {})
            return {
                "debate_id": node_id,
                "tension": node.get("tension"),
                "success": node.get("success"),
                "depth": node.get("depth", 0),
                "children": [build_subtree(child_id) for child_id in node.get("children", [])],
            }

        return build_subtree(self.root_id)


class GenesisLedger:
    """
    Unified ledger for tracking fractal debates and agent evolution.

    Extends ProvenanceChain with genesis-specific events and queries.
    """

    def __init__(self, db_path: str = ".nomic/genesis.db"):
        self.db_path = Path(db_path)
        self.db = GenesisDatabase(db_path)
        self.provenance = ProvenanceChain(chain_id="genesis-ledger")
        self._events: list[GenesisEvent] = []

    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        import uuid

        return str(uuid.uuid4())[:12]

    def _record_event(self, event: GenesisEvent) -> None:
        """Record an event to database and memory."""
        self._events.append(event)

        with self.db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO genesis_events (event_id, event_type, timestamp, parent_event_id, content_hash, data)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    event.event_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    event.parent_event_id,
                    event.content_hash,
                    json.dumps(event.data),
                ),
            )

            conn.commit()

        # Also add to provenance chain
        self.provenance.add_record(
            content=json.dumps(event.to_dict()),
            source_type=SourceType.SYNTHESIS,
            source_id=f"genesis:{event.event_type.value}",
            transformation=TransformationType.ORIGINAL,
            metadata={"event_id": event.event_id},
        )

    # === Debate Events ===

    def record_debate_start(
        self,
        debate_id: str,
        task: str,
        agents: list[str],
        parent_debate_id: Optional[str] = None,
    ) -> GenesisEvent:
        """Record the start of a debate."""
        event = GenesisEvent(
            event_id=self._generate_event_id(),
            event_type=GenesisEventType.DEBATE_START,
            timestamp=datetime.now(),
            parent_event_id=self._get_last_event_id(parent_debate_id),
            data={
                "debate_id": debate_id,
                "task": task,
                "agents": agents,
                "parent_debate_id": parent_debate_id,
            },
        )
        self._record_event(event)
        return event

    def record_debate_spawn(
        self,
        parent_id: str,
        child_id: str,
        trigger: str,
        tension_description: str,
    ) -> GenesisEvent:
        """Record spawning of a sub-debate."""
        event = GenesisEvent(
            event_id=self._generate_event_id(),
            event_type=GenesisEventType.DEBATE_SPAWN,
            timestamp=datetime.now(),
            parent_event_id=self._get_last_event_id(parent_id),
            data={
                "parent_debate_id": parent_id,
                "child_debate_id": child_id,
                "trigger": trigger,
                "tension": tension_description,
            },
        )
        self._record_event(event)
        return event

    def record_debate_merge(
        self,
        parent_id: str,
        child_id: str,
        success: bool,
        resolution: str,
    ) -> GenesisEvent:
        """Record merging of a sub-debate back to parent."""
        event = GenesisEvent(
            event_id=self._generate_event_id(),
            event_type=GenesisEventType.DEBATE_MERGE,
            timestamp=datetime.now(),
            data={
                "parent_debate_id": parent_id,
                "child_debate_id": child_id,
                "success": success,
                "resolution": resolution,
            },
        )
        self._record_event(event)
        return event

    def record_consensus(
        self,
        debate_id: str,
        proof: ConsensusProof,
    ) -> GenesisEvent:
        """Record consensus reached in a debate."""
        event = GenesisEvent(
            event_id=self._generate_event_id(),
            event_type=GenesisEventType.CONSENSUS_REACHED,
            timestamp=datetime.now(),
            data={
                "debate_id": debate_id,
                "proof_id": proof.proof_id,
                "confidence": proof.confidence,
                "supporting_agents": proof.supporting_agents,
                "dissenting_agents": proof.dissenting_agents,
                "final_claim": proof.final_claim[:500] if proof.final_claim else "",
            },
        )
        self._record_event(event)
        return event

    # === Agent Evolution Events ===

    def record_agent_birth(
        self,
        genome: AgentGenome,
        parents: list[str],
        birth_type: str = "crossover",  # crossover, mutation, specialist
    ) -> GenesisEvent:
        """Record the creation of a new agent genome."""
        event = GenesisEvent(
            event_id=self._generate_event_id(),
            event_type=GenesisEventType.AGENT_BIRTH,
            timestamp=datetime.now(),
            data={
                "genome_id": genome.genome_id,
                "name": genome.name,
                "parent_genomes": parents,
                "generation": genome.generation,
                "birth_type": birth_type,
                "traits": genome.traits,
                "expertise": genome.expertise,
                "model_preference": genome.model_preference,
            },
        )
        self._record_event(event)
        return event

    def record_agent_death(
        self,
        genome_id: str,
        reason: str,
        final_fitness: float,
    ) -> GenesisEvent:
        """Record the culling of an agent genome."""
        event = GenesisEvent(
            event_id=self._generate_event_id(),
            event_type=GenesisEventType.AGENT_DEATH,
            timestamp=datetime.now(),
            data={
                "genome_id": genome_id,
                "reason": reason,
                "final_fitness": final_fitness,
            },
        )
        self._record_event(event)
        return event

    def record_mutation(
        self,
        before: AgentGenome,
        after: AgentGenome,
    ) -> GenesisEvent:
        """Record a genome mutation."""
        event = GenesisEvent(
            event_id=self._generate_event_id(),
            event_type=GenesisEventType.AGENT_MUTATION,
            timestamp=datetime.now(),
            data={
                "before_genome_id": before.genome_id,
                "after_genome_id": after.genome_id,
                "before_traits": before.traits,
                "after_traits": after.traits,
                "before_expertise": before.expertise,
                "after_expertise": after.expertise,
            },
        )
        self._record_event(event)
        return event

    def record_fitness_update(
        self,
        genome_id: str,
        old_fitness: float,
        new_fitness: float,
        reason: str,
    ) -> GenesisEvent:
        """Record a fitness score change."""
        event = GenesisEvent(
            event_id=self._generate_event_id(),
            event_type=GenesisEventType.FITNESS_UPDATE,
            timestamp=datetime.now(),
            data={
                "genome_id": genome_id,
                "old_fitness": old_fitness,
                "new_fitness": new_fitness,
                "change": new_fitness - old_fitness,
                "reason": reason,
            },
        )
        self._record_event(event)
        return event

    # === Query Methods ===

    def get_lineage(self, genome_id: str) -> list[dict]:
        """Get the full lineage (ancestry) of a genome."""
        from aragora.genesis.genome import GenomeStore

        store = GenomeStore(str(self.db_path))

        lineage = []
        current = store.get(genome_id)

        visited = set()
        while current and current.genome_id not in visited:
            lineage.append(
                {
                    "genome_id": current.genome_id,
                    "name": current.name,
                    "generation": current.generation,
                    "fitness_score": current.fitness_score,
                    "parent_genomes": current.parent_genomes,
                }
            )
            visited.add(current.genome_id)

            if current.parent_genomes:
                current = store.get(current.parent_genomes[0])
            else:
                break

        return lineage

    def get_debate_tree(self, root_debate_id: str) -> FractalTree:
        """Get the fractal tree structure for a debate."""
        tree = FractalTree(root_id=root_debate_id)

        with self.db.connection() as conn:
            cursor = conn.cursor()

            # Get all spawn events
            cursor.execute(
                """
                SELECT data FROM genesis_events
                WHERE event_type IN ('debate_spawn', 'debate_merge', 'debate_start')
                ORDER BY timestamp
            """
            )

            for (data_json,) in cursor.fetchall():
                try:
                    data = json.loads(data_json)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping corrupted genesis event: {e}")
                    continue

                if "child_debate_id" in data:
                    # This is a spawn or merge event
                    tree.add_node(
                        debate_id=data["child_debate_id"],
                        parent_id=data.get("parent_debate_id"),
                        tension=data.get("tension"),
                        success=data.get("success", False),
                        depth=len(tree.get_children(data.get("parent_debate_id", ""))) + 1,
                    )
                elif data.get("debate_id") == root_debate_id:
                    # Root debate
                    tree.add_node(
                        debate_id=root_debate_id,
                        parent_id=None,
                        depth=0,
                    )

        return tree

    def get_events_by_type(self, event_type: GenesisEventType) -> list[GenesisEvent]:
        """Get all events of a specific type."""
        with self.db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT event_id, event_type, timestamp, parent_event_id, content_hash, data
                FROM genesis_events
                WHERE event_type = ?
                ORDER BY timestamp
            """,
                (event_type.value,),
            )

            events = []
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[5]) if row[5] else {}
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping event with corrupted data: {e}")
                    continue
                events.append(
                    GenesisEvent(
                        event_id=row[0],
                        event_type=GenesisEventType(row[1]),
                        timestamp=datetime.fromisoformat(row[2]),
                        parent_event_id=row[3],
                        content_hash=row[4],
                        data=data,
                    )
                )

        return events

    def _get_last_event_id(self, debate_id: Optional[str]) -> Optional[str]:
        """Get the last event ID for a debate."""
        if not debate_id:
            return None

        with self.db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT event_id FROM genesis_events
                WHERE data LIKE ?
                ORDER BY timestamp DESC
                LIMIT 1
            """,
                (f'%"debate_id": "{debate_id}"%',),
            )

            row = cursor.fetchone()

        return row[0] if row else None

    # === Verification ===

    def verify_integrity(self) -> bool:
        """Verify the integrity of the ledger using the provenance chain."""
        valid, _errors = self.provenance.verify_chain()
        return valid

    def get_merkle_root(self) -> str:
        """Get Merkle root of all events for verification."""
        # Convert events to provenance records for Merkle tree
        records = self.provenance.records
        if not records:
            return hashlib.sha256(b"empty").hexdigest()

        tree = MerkleTree(records)
        return tree.root or ""

    # === Export ===

    def export(
        self,
        format: str = "json",
        include_lineage: bool = False,
    ) -> str:
        """Export the ledger in various formats."""
        if format == "json":
            return self._export_json(include_lineage)
        elif format == "markdown":
            return self._export_markdown(include_lineage)
        elif format == "html":
            return self._export_html(include_lineage)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _export_json(self, include_lineage: bool) -> str:
        """Export as JSON."""
        with self.db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM genesis_events ORDER BY timestamp")
            events = [
                {
                    "event_id": row[0],
                    "event_type": row[1],
                    "timestamp": row[2],
                    "parent_event_id": row[3],
                    "content_hash": row[4],
                    "data": json.loads(row[5]) if row[5] else {},
                }
                for row in cursor.fetchall()
            ]

        output = {
            "ledger_id": self.provenance.chain_id,
            "merkle_root": self.get_merkle_root(),
            "event_count": len(events),
            "events": events,
        }

        return json.dumps(output, indent=2)

    def _export_markdown(self, include_lineage: bool) -> str:
        """Export as Markdown."""
        lines = [
            "# Genesis Ledger",
            "",
            f"**Ledger ID:** `{self.provenance.chain_id}`",
            f"**Merkle Root:** `{self.get_merkle_root()[:16]}...`",
            "",
            "## Events",
            "",
        ]

        with self.db.connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM genesis_events ORDER BY timestamp")

            for row in cursor.fetchall():
                event_type = row[1]
                timestamp = row[2]
                data = json.loads(row[5]) if row[5] else {}

                lines.append(f"### {event_type}")
                lines.append(f"*{timestamp}*")
                lines.append("")

                for key, value in data.items():
                    if isinstance(value, dict):
                        value = json.dumps(value)[:100]
                    elif isinstance(value, list):
                        value = ", ".join(str(v) for v in value[:5])
                    lines.append(f"- **{key}:** {value}")

                lines.append("")

        return "\n".join(lines)

    def _export_html(self, include_lineage: bool) -> str:
        """Export as interactive HTML."""
        json_data = self._export_json(include_lineage)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Genesis Ledger</title>
    <style>
        body {{ font-family: system-ui, sans-serif; margin: 2em; background: #1a1a2e; color: #eee; }}
        .event {{ background: #16213e; padding: 1em; margin: 0.5em 0; border-radius: 8px; }}
        .event-type {{ color: #00d4ff; font-weight: bold; }}
        .timestamp {{ color: #888; font-size: 0.9em; }}
        .data {{ margin-top: 0.5em; font-size: 0.9em; }}
        .data-key {{ color: #ffd700; }}
        h1 {{ color: #00d4ff; }}
        pre {{ background: #0a0a1a; padding: 1em; border-radius: 4px; overflow-x: auto; }}
    </style>
</head>
<body>
    <h1>Genesis Ledger</h1>
    <p><strong>Merkle Root:</strong> <code>{self.get_merkle_root()[:32]}...</code></p>

    <h2>Events</h2>
    <div id="events"></div>

    <h2>Raw Data</h2>
    <pre>{json_data}</pre>

    <script>
        const data = {json_data};
        const container = document.getElementById('events');

        data.events.forEach(event => {{
            const div = document.createElement('div');
            div.className = 'event';
            div.innerHTML = `
                <span class="event-type">${{event.event_type}}</span>
                <span class="timestamp">${{event.timestamp}}</span>
                <div class="data">
                    ${{Object.entries(event.data).map(([k, v]) =>
                        `<div><span class="data-key">${{k}}:</span> ${{JSON.stringify(v).slice(0, 100)}}</div>`
                    ).join('')}}
                </div>
            `;
            container.appendChild(div);
        }});
    </script>
</body>
</html>"""

        return html
