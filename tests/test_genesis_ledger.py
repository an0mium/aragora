"""
Tests for genesis ledger system.

Tests:
- Genesis event creation and serialization
- Event type enumeration
- Fractal tree construction
- Ledger persistence
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from aragora.genesis.ledger import (
    GenesisEvent,
    GenesisEventType,
    FractalTree,
    GenesisLedger,
)


class TestGenesisEventType:
    """Test GenesisEventType enumeration."""

    def test_debate_events_exist(self):
        """Debate-related event types should exist."""
        assert GenesisEventType.DEBATE_START.value == "debate_start"
        assert GenesisEventType.DEBATE_END.value == "debate_end"
        assert GenesisEventType.DEBATE_SPAWN.value == "debate_spawn"
        assert GenesisEventType.DEBATE_MERGE.value == "debate_merge"

    def test_consensus_events_exist(self):
        """Consensus-related event types should exist."""
        assert GenesisEventType.CONSENSUS_REACHED.value == "consensus_reached"
        assert GenesisEventType.TENSION_DETECTED.value == "tension_detected"
        assert GenesisEventType.TENSION_RESOLVED.value == "tension_resolved"

    def test_agent_evolution_events_exist(self):
        """Agent evolution event types should exist."""
        assert GenesisEventType.AGENT_BIRTH.value == "agent_birth"
        assert GenesisEventType.AGENT_DEATH.value == "agent_death"
        assert GenesisEventType.AGENT_MUTATION.value == "agent_mutation"
        assert GenesisEventType.AGENT_CROSSOVER.value == "agent_crossover"
        assert GenesisEventType.FITNESS_UPDATE.value == "fitness_update"

    def test_population_events_exist(self):
        """Population event types should exist."""
        assert GenesisEventType.POPULATION_CREATED.value == "population_created"
        assert GenesisEventType.POPULATION_EVOLVED.value == "population_evolved"
        assert GenesisEventType.GENERATION_ADVANCE.value == "generation_advance"


class TestGenesisEvent:
    """Test GenesisEvent dataclass."""

    def test_create_event(self):
        """Should create event with all fields."""
        now = datetime.now()
        event = GenesisEvent(
            event_id="evt-001",
            event_type=GenesisEventType.DEBATE_START,
            timestamp=now,
            parent_event_id=None,
            data={"debate_id": "d-123", "topic": "Test debate"},
        )

        assert event.event_id == "evt-001"
        assert event.event_type == GenesisEventType.DEBATE_START
        assert event.timestamp == now
        assert event.data["debate_id"] == "d-123"

    def test_computes_content_hash(self):
        """Event should auto-compute content hash."""
        event = GenesisEvent(
            event_id="evt-001",
            event_type=GenesisEventType.DEBATE_START,
            timestamp=datetime.now(),
        )

        assert event.content_hash
        assert len(event.content_hash) == 64  # SHA256 hex

    def test_same_content_same_hash(self):
        """Same content should produce same hash."""
        now = datetime.now()
        event1 = GenesisEvent(
            event_id="evt-001",
            event_type=GenesisEventType.DEBATE_START,
            timestamp=now,
            data={"key": "value"},
        )

        event2 = GenesisEvent(
            event_id="evt-001",
            event_type=GenesisEventType.DEBATE_START,
            timestamp=now,
            data={"key": "value"},
        )

        assert event1.content_hash == event2.content_hash

    def test_different_content_different_hash(self):
        """Different content should produce different hash."""
        now = datetime.now()
        event1 = GenesisEvent(
            event_id="evt-001",
            event_type=GenesisEventType.DEBATE_START,
            timestamp=now,
        )

        event2 = GenesisEvent(
            event_id="evt-002",  # Different ID
            event_type=GenesisEventType.DEBATE_START,
            timestamp=now,
        )

        assert event1.content_hash != event2.content_hash

    def test_to_dict(self):
        """Should serialize to dictionary."""
        now = datetime.now()
        event = GenesisEvent(
            event_id="evt-001",
            event_type=GenesisEventType.AGENT_BIRTH,
            timestamp=now,
            parent_event_id="evt-000",
            data={"genome_id": "g-123"},
        )

        data = event.to_dict()

        assert data["event_id"] == "evt-001"
        assert data["event_type"] == "agent_birth"
        assert data["timestamp"] == now.isoformat()
        assert data["parent_event_id"] == "evt-000"
        assert data["data"]["genome_id"] == "g-123"
        assert "content_hash" in data

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        now = datetime.now()
        data = {
            "event_id": "evt-001",
            "event_type": "agent_birth",
            "timestamp": now.isoformat(),
            "parent_event_id": "evt-000",
            "content_hash": "abc123",
            "data": {"genome_id": "g-123"},
        }

        event = GenesisEvent.from_dict(data)

        assert event.event_id == "evt-001"
        assert event.event_type == GenesisEventType.AGENT_BIRTH
        assert event.parent_event_id == "evt-000"
        assert event.data["genome_id"] == "g-123"

    def test_roundtrip(self):
        """Event should survive to_dict/from_dict roundtrip."""
        original = GenesisEvent(
            event_id="evt-001",
            event_type=GenesisEventType.CONSENSUS_REACHED,
            timestamp=datetime.now(),
            parent_event_id="evt-000",
            data={"confidence": 0.95, "agents": ["a", "b"]},
        )

        data = original.to_dict()
        restored = GenesisEvent.from_dict(data)

        assert restored.event_id == original.event_id
        assert restored.event_type == original.event_type
        assert restored.parent_event_id == original.parent_event_id
        assert restored.data == original.data


class TestFractalTree:
    """Test FractalTree structure."""

    def test_create_tree(self):
        """Should create tree with root."""
        tree = FractalTree(root_id="debate-root")

        assert tree.root_id == "debate-root"
        assert tree.nodes == {}

    def test_add_root_node(self):
        """Should add root node."""
        tree = FractalTree(root_id="root")
        tree.add_node(
            debate_id="root",
            parent_id=None,
            tension=None,
            success=False,
            depth=0,
        )

        assert "root" in tree.nodes
        assert tree.nodes["root"]["depth"] == 0
        assert tree.nodes["root"]["children"] == []

    def test_add_child_nodes(self):
        """Should add child nodes and link to parent."""
        tree = FractalTree(root_id="root")
        tree.add_node("root", None, depth=0)
        tree.add_node("child1", "root", tension="scope", depth=1)
        tree.add_node("child2", "root", tension="definition", depth=1)

        assert len(tree.nodes["root"]["children"]) == 2
        assert "child1" in tree.nodes["root"]["children"]
        assert "child2" in tree.nodes["root"]["children"]

    def test_add_grandchild_nodes(self):
        """Should support multi-level nesting."""
        tree = FractalTree(root_id="root")
        tree.add_node("root", None, depth=0)
        tree.add_node("child", "root", depth=1)
        tree.add_node("grandchild", "child", depth=2)

        assert "grandchild" in tree.nodes["child"]["children"]
        assert tree.nodes["grandchild"]["depth"] == 2

    def test_get_children(self):
        """Should return child IDs."""
        tree = FractalTree(root_id="root")
        tree.add_node("root", None)
        tree.add_node("c1", "root")
        tree.add_node("c2", "root")

        children = tree.get_children("root")
        assert set(children) == {"c1", "c2"}

    def test_get_children_empty(self):
        """Should return empty list for leaf nodes."""
        tree = FractalTree(root_id="root")
        tree.add_node("root", None)
        tree.add_node("leaf", "root")

        assert tree.get_children("leaf") == []

    def test_get_children_nonexistent(self):
        """Should return empty list for non-existent nodes."""
        tree = FractalTree(root_id="root")
        assert tree.get_children("nonexistent") == []

    def test_to_dict_simple(self):
        """Should serialize simple tree to nested dict."""
        tree = FractalTree(root_id="root")
        tree.add_node("root", None, tension=None, success=True, depth=0)

        data = tree.to_dict()

        assert data["debate_id"] == "root"
        assert data["success"] is True
        assert data["depth"] == 0
        assert data["children"] == []

    def test_to_dict_nested(self):
        """Should serialize nested tree correctly."""
        tree = FractalTree(root_id="root")
        tree.add_node("root", None, success=True, depth=0)
        tree.add_node("child1", "root", tension="scope", success=True, depth=1)
        tree.add_node("child2", "root", tension="def", success=False, depth=1)
        tree.add_node("grandchild", "child1", tension="detail", success=True, depth=2)

        data = tree.to_dict()

        assert data["debate_id"] == "root"
        assert len(data["children"]) == 2

        child1_data = next(c for c in data["children"] if c["debate_id"] == "child1")
        assert child1_data["tension"] == "scope"
        assert len(child1_data["children"]) == 1
        assert child1_data["children"][0]["debate_id"] == "grandchild"


class TestGenesisLedger:
    """Test GenesisLedger persistence."""

    def test_init_creates_db(self):
        """Ledger should create database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "ledger" / "genesis.db"
            ledger = GenesisLedger(db_path=str(db_path))

            assert db_path.exists()

    def test_generates_event_ids(self):
        """Should generate unique event IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "genesis.db"
            ledger = GenesisLedger(db_path=str(db_path))

            id1 = ledger._generate_event_id()
            id2 = ledger._generate_event_id()

            assert id1 != id2

    def test_ledger_has_provenance_chain(self):
        """Ledger should have provenance chain."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "genesis.db"
            ledger = GenesisLedger(db_path=str(db_path))

            assert ledger.provenance is not None
            assert ledger.provenance.chain_id == "genesis-ledger"


class TestLedgerEventTypes:
    """Test that all event types are properly handled."""

    def test_debate_event_types(self):
        """Debate events should have correct values."""
        types = [
            GenesisEventType.DEBATE_START,
            GenesisEventType.DEBATE_END,
            GenesisEventType.DEBATE_SPAWN,
            GenesisEventType.DEBATE_MERGE,
        ]

        for t in types:
            event = GenesisEvent(
                event_id=f"test-{t.value}",
                event_type=t,
                timestamp=datetime.now(),
            )
            assert event.event_type == t

    def test_consensus_event_types(self):
        """Consensus events should have correct values."""
        types = [
            GenesisEventType.CONSENSUS_REACHED,
            GenesisEventType.TENSION_DETECTED,
            GenesisEventType.TENSION_RESOLVED,
            GenesisEventType.TENSION_UNRESOLVED,
        ]

        for t in types:
            event = GenesisEvent(
                event_id=f"test-{t.value}",
                event_type=t,
                timestamp=datetime.now(),
            )
            assert event.event_type == t

    def test_evolution_event_types(self):
        """Evolution events should have correct values."""
        types = [
            GenesisEventType.AGENT_BIRTH,
            GenesisEventType.AGENT_DEATH,
            GenesisEventType.AGENT_MUTATION,
            GenesisEventType.AGENT_CROSSOVER,
            GenesisEventType.FITNESS_UPDATE,
        ]

        for t in types:
            event = GenesisEvent(
                event_id=f"test-{t.value}",
                event_type=t,
                timestamp=datetime.now(),
            )
            assert event.event_type == t
