"""Tests for aragora.export.artifact module.

Comprehensive tests for DebateArtifact, ConsensusProof, VerificationResult,
ArtifactBuilder, and related functionality.
"""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from aragora.export.artifact import (
    ConsensusProof,
    VerificationResult,
    DebateArtifact,
    ArtifactBuilder,
    create_artifact_from_debate,
)


# =============================================================================
# ConsensusProof Tests
# =============================================================================


class TestConsensusProof:
    """Tests for ConsensusProof dataclass."""

    def test_create_with_required_fields(self):
        """Test creation with all required fields."""
        proof = ConsensusProof(
            reached=True,
            confidence=0.85,
            vote_breakdown={"agent1": True, "agent2": False},
            final_answer="The answer is 42",
            rounds_used=3,
        )
        assert proof.reached is True
        assert proof.confidence == 0.85
        assert proof.vote_breakdown == {"agent1": True, "agent2": False}
        assert proof.final_answer == "The answer is 42"
        assert proof.rounds_used == 3

    def test_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated."""
        proof = ConsensusProof(
            reached=True,
            confidence=0.9,
            vote_breakdown={},
            final_answer="Answer",
            rounds_used=1,
        )
        assert proof.timestamp is not None
        assert len(proof.timestamp) > 0
        # Should be ISO format
        datetime.fromisoformat(proof.timestamp)

    def test_timestamp_custom_value(self):
        """Test custom timestamp value."""
        custom_ts = "2024-01-15T10:30:00"
        proof = ConsensusProof(
            reached=False,
            confidence=0.5,
            vote_breakdown={},
            final_answer="",
            rounds_used=5,
            timestamp=custom_ts,
        )
        assert proof.timestamp == custom_ts

    def test_to_dict_all_fields(self):
        """Test to_dict includes all fields."""
        proof = ConsensusProof(
            reached=True,
            confidence=0.95,
            vote_breakdown={"a": True, "b": True, "c": False},
            final_answer="Final consensus",
            rounds_used=4,
            timestamp="2024-01-01T00:00:00",
        )
        result = proof.to_dict()

        assert result["reached"] is True
        assert result["confidence"] == 0.95
        assert result["vote_breakdown"] == {"a": True, "b": True, "c": False}
        assert result["final_answer"] == "Final consensus"
        assert result["rounds_used"] == 4
        assert result["timestamp"] == "2024-01-01T00:00:00"

    def test_to_dict_empty_vote_breakdown(self):
        """Test to_dict with empty vote breakdown."""
        proof = ConsensusProof(
            reached=False,
            confidence=0.0,
            vote_breakdown={},
            final_answer="",
            rounds_used=0,
        )
        result = proof.to_dict()
        assert result["vote_breakdown"] == {}

    def test_consensus_not_reached(self):
        """Test consensus proof when consensus not reached."""
        proof = ConsensusProof(
            reached=False,
            confidence=0.3,
            vote_breakdown={"a": True, "b": False, "c": False},
            final_answer="No consensus",
            rounds_used=10,
        )
        assert proof.reached is False
        assert proof.confidence == 0.3

    def test_confidence_boundary_values(self):
        """Test confidence at boundary values."""
        # Zero confidence
        proof_zero = ConsensusProof(
            reached=False, confidence=0.0, vote_breakdown={},
            final_answer="", rounds_used=1,
        )
        assert proof_zero.confidence == 0.0

        # Full confidence
        proof_full = ConsensusProof(
            reached=True, confidence=1.0, vote_breakdown={},
            final_answer="", rounds_used=1,
        )
        assert proof_full.confidence == 1.0

    def test_long_final_answer(self):
        """Test with very long final answer."""
        long_answer = "A" * 10000
        proof = ConsensusProof(
            reached=True,
            confidence=0.9,
            vote_breakdown={},
            final_answer=long_answer,
            rounds_used=1,
        )
        assert proof.final_answer == long_answer
        assert proof.to_dict()["final_answer"] == long_answer


# =============================================================================
# VerificationResult Tests
# =============================================================================


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_create_minimal(self):
        """Test creation with only required fields."""
        result = VerificationResult(
            claim_id="claim-001",
            claim_text="X > 0 implies X is positive",
            status="verified",
            method="z3",
        )
        assert result.claim_id == "claim-001"
        assert result.claim_text == "X > 0 implies X is positive"
        assert result.status == "verified"
        assert result.method == "z3"
        assert result.proof_trace is None
        assert result.counterexample is None
        assert result.duration_ms == 0
        assert result.metadata == {}

    def test_create_with_all_fields(self):
        """Test creation with all fields."""
        result = VerificationResult(
            claim_id="claim-002",
            claim_text="All inputs are valid",
            status="refuted",
            method="lean",
            proof_trace="Step 1: ...\nStep 2: ...",
            counterexample="x = -1 violates constraint",
            duration_ms=1500,
            metadata={"solver": "lean4", "timeout": 30},
        )
        assert result.proof_trace == "Step 1: ...\nStep 2: ..."
        assert result.counterexample == "x = -1 violates constraint"
        assert result.duration_ms == 1500
        assert result.metadata == {"solver": "lean4", "timeout": 30}

    def test_status_verified(self):
        """Test verified status."""
        result = VerificationResult(
            claim_id="v1", claim_text="claim", status="verified", method="z3",
            proof_trace="QED",
        )
        assert result.status == "verified"
        assert result.proof_trace is not None

    def test_status_refuted(self):
        """Test refuted status."""
        result = VerificationResult(
            claim_id="v2", claim_text="claim", status="refuted", method="z3",
            counterexample="x=0",
        )
        assert result.status == "refuted"
        assert result.counterexample is not None

    def test_status_timeout(self):
        """Test timeout status."""
        result = VerificationResult(
            claim_id="v3", claim_text="complex claim", status="timeout", method="z3",
            duration_ms=60000,
        )
        assert result.status == "timeout"

    def test_status_undecidable(self):
        """Test undecidable status."""
        result = VerificationResult(
            claim_id="v4", claim_text="Godel sentence", status="undecidable", method="lean",
        )
        assert result.status == "undecidable"

    def test_to_dict_complete(self):
        """Test to_dict with all fields."""
        result = VerificationResult(
            claim_id="test-id",
            claim_text="Test claim",
            status="verified",
            method="simulation",
            proof_trace="Proof here",
            counterexample=None,
            duration_ms=500,
            metadata={"key": "value"},
        )
        d = result.to_dict()

        assert d["claim_id"] == "test-id"
        assert d["claim_text"] == "Test claim"
        assert d["status"] == "verified"
        assert d["method"] == "simulation"
        assert d["proof_trace"] == "Proof here"
        assert d["counterexample"] is None
        assert d["duration_ms"] == 500
        assert d["metadata"] == {"key": "value"}

    def test_to_dict_minimal(self):
        """Test to_dict with minimal fields."""
        result = VerificationResult(
            claim_id="min", claim_text="text", status="timeout", method="z3",
        )
        d = result.to_dict()
        assert d["proof_trace"] is None
        assert d["counterexample"] is None
        assert d["duration_ms"] == 0
        assert d["metadata"] == {}

    def test_long_claim_text(self):
        """Test with very long claim text."""
        long_text = "Claim: " + "x" * 5000
        result = VerificationResult(
            claim_id="long", claim_text=long_text, status="verified", method="z3",
        )
        assert result.claim_text == long_text

    def test_special_characters_in_claim(self):
        """Test special characters in claim text."""
        special = "∀x ∈ ℝ: x² ≥ 0 ∧ (x = 0 → x² = 0)"
        result = VerificationResult(
            claim_id="special", claim_text=special, status="verified", method="lean",
        )
        assert result.claim_text == special
        assert result.to_dict()["claim_text"] == special


# =============================================================================
# DebateArtifact Tests
# =============================================================================


class TestDebateArtifact:
    """Tests for DebateArtifact dataclass."""

    @pytest.fixture
    def minimal_artifact(self):
        """Create minimal artifact."""
        return DebateArtifact(
            debate_id="debate-123",
            task="Test task",
        )

    @pytest.fixture
    def full_artifact(self):
        """Create artifact with all fields."""
        return DebateArtifact(
            artifact_id="artifact-001",
            debate_id="debate-456",
            task="Full test task",
            created_at="2024-01-15T12:00:00",
            graph_data={"nodes": {"n1": {"type": "proposal"}}},
            trace_data={"events": [{"type": "message", "content": "Hello"}]},
            provenance_data={"chain": {"records": []}},
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                vote_breakdown={"a1": True, "a2": True},
                final_answer="Consensus answer",
                rounds_used=3,
                timestamp="2024-01-15T12:30:00",
            ),
            verification_results=[
                VerificationResult(
                    claim_id="c1",
                    claim_text="Claim 1",
                    status="verified",
                    method="z3",
                ),
            ],
            agents=["agent1", "agent2"],
            rounds=3,
            duration_seconds=120.5,
            message_count=15,
            critique_count=5,
            version="1.0",
            generator="test-generator",
        )

    def test_create_minimal(self, minimal_artifact):
        """Test minimal artifact creation."""
        assert minimal_artifact.debate_id == "debate-123"
        assert minimal_artifact.task == "Test task"
        assert minimal_artifact.graph_data is None
        assert minimal_artifact.trace_data is None
        assert minimal_artifact.consensus_proof is None
        assert minimal_artifact.verification_results == []
        assert minimal_artifact.agents == []

    def test_artifact_id_auto_generated(self):
        """Test artifact_id is auto-generated."""
        a1 = DebateArtifact()
        a2 = DebateArtifact()
        assert a1.artifact_id != a2.artifact_id
        assert len(a1.artifact_id) == 12  # UUID truncated to 12 chars

    def test_created_at_auto_generated(self):
        """Test created_at is auto-generated."""
        artifact = DebateArtifact()
        assert artifact.created_at is not None
        datetime.fromisoformat(artifact.created_at)

    def test_content_hash_consistency(self, minimal_artifact):
        """Test content hash is consistent."""
        hash1 = minimal_artifact.content_hash
        hash2 = minimal_artifact.content_hash
        assert hash1 == hash2
        assert len(hash1) == 16  # SHA256 truncated to 16 chars

    def test_content_hash_changes_with_task(self):
        """Test hash changes when task changes."""
        a1 = DebateArtifact(task="Task A")
        a2 = DebateArtifact(task="Task B")
        assert a1.content_hash != a2.content_hash

    def test_content_hash_changes_with_graph(self):
        """Test hash changes when graph changes."""
        a1 = DebateArtifact(task="Same task", graph_data={"nodes": {}})
        a2 = DebateArtifact(task="Same task", graph_data={"nodes": {"n1": {}}})
        assert a1.content_hash != a2.content_hash

    def test_content_hash_ignores_metadata(self):
        """Test hash ignores metadata fields."""
        a1 = DebateArtifact(task="Task", agents=["a1"])
        a2 = DebateArtifact(task="Task", agents=["a1", "a2"])
        # Hash only includes task, graph, trace, provenance, consensus, verifications
        assert a1.content_hash == a2.content_hash

    def test_to_dict_minimal(self, minimal_artifact):
        """Test to_dict with minimal artifact."""
        d = minimal_artifact.to_dict()
        assert d["debate_id"] == "debate-123"
        assert d["task"] == "Test task"
        assert d["graph"] is None
        assert d["trace"] is None
        assert d["consensus_proof"] is None
        assert d["verification_results"] == []

    def test_to_dict_full(self, full_artifact):
        """Test to_dict with full artifact."""
        d = full_artifact.to_dict()
        assert d["artifact_id"] == "artifact-001"
        assert d["debate_id"] == "debate-456"
        assert d["graph"] == {"nodes": {"n1": {"type": "proposal"}}}
        assert d["trace"] == {"events": [{"type": "message", "content": "Hello"}]}
        assert d["consensus_proof"]["reached"] is True
        assert d["consensus_proof"]["confidence"] == 0.9
        assert len(d["verification_results"]) == 1
        assert d["agents"] == ["agent1", "agent2"]
        assert d["rounds"] == 3
        assert d["duration_seconds"] == 120.5
        assert "content_hash" in d

    def test_to_json_valid(self, full_artifact):
        """Test to_json produces valid JSON."""
        json_str = full_artifact.to_json()
        parsed = json.loads(json_str)
        assert parsed["debate_id"] == "debate-456"

    def test_to_json_indent(self, minimal_artifact):
        """Test to_json with custom indent."""
        json_no_indent = minimal_artifact.to_json(indent=None)
        json_indented = minimal_artifact.to_json(indent=4)
        assert len(json_indented) > len(json_no_indent)

    def test_from_dict_minimal(self):
        """Test from_dict with minimal data."""
        data = {
            "debate_id": "restored-123",
            "task": "Restored task",
        }
        artifact = DebateArtifact.from_dict(data)
        assert artifact.debate_id == "restored-123"
        assert artifact.task == "Restored task"
        assert artifact.agents == []

    def test_from_dict_full(self):
        """Test from_dict with full data."""
        data = {
            "artifact_id": "art-999",
            "debate_id": "deb-999",
            "task": "Full restore",
            "created_at": "2024-06-01T00:00:00",
            "graph": {"nodes": {}},
            "trace": {"events": []},
            "provenance": {"chain": {}},
            "consensus_proof": {
                "reached": True,
                "confidence": 0.75,
                "vote_breakdown": {"x": True},
                "final_answer": "Answer",
                "rounds_used": 2,
                "timestamp": "2024-06-01T01:00:00",
            },
            "verification_results": [
                {
                    "claim_id": "c1",
                    "claim_text": "Claim",
                    "status": "verified",
                    "method": "z3",
                },
            ],
            "agents": ["ag1", "ag2"],
            "rounds": 5,
            "duration_seconds": 60.0,
            "message_count": 20,
            "critique_count": 8,
            "version": "2.0",
            "generator": "custom",
        }
        artifact = DebateArtifact.from_dict(data)

        assert artifact.artifact_id == "art-999"
        assert artifact.graph_data == {"nodes": {}}
        assert artifact.consensus_proof is not None
        assert artifact.consensus_proof.reached is True
        assert artifact.consensus_proof.confidence == 0.75
        assert len(artifact.verification_results) == 1
        assert artifact.verification_results[0].status == "verified"
        assert artifact.version == "2.0"

    def test_from_dict_missing_optional_fields(self):
        """Test from_dict handles missing optional fields."""
        data = {"task": "Only task"}
        artifact = DebateArtifact.from_dict(data)
        assert artifact.task == "Only task"
        assert artifact.debate_id == ""
        assert artifact.graph_data is None
        assert artifact.consensus_proof is None

    def test_from_json_valid(self):
        """Test from_json with valid JSON."""
        json_str = '{"debate_id": "json-123", "task": "JSON task"}'
        artifact = DebateArtifact.from_json(json_str)
        assert artifact.debate_id == "json-123"

    def test_roundtrip_to_dict_from_dict(self, full_artifact):
        """Test roundtrip: to_dict -> from_dict."""
        d = full_artifact.to_dict()
        restored = DebateArtifact.from_dict(d)

        assert restored.artifact_id == full_artifact.artifact_id
        assert restored.debate_id == full_artifact.debate_id
        assert restored.task == full_artifact.task
        assert restored.graph_data == full_artifact.graph_data
        assert restored.agents == full_artifact.agents
        assert restored.consensus_proof.reached == full_artifact.consensus_proof.reached

    def test_roundtrip_to_json_from_json(self, full_artifact):
        """Test roundtrip: to_json -> from_json."""
        json_str = full_artifact.to_json()
        restored = DebateArtifact.from_json(json_str)

        assert restored.debate_id == full_artifact.debate_id
        assert restored.content_hash == full_artifact.content_hash

    def test_save_and_load(self, full_artifact, tmp_path):
        """Test save to file and load from file."""
        file_path = tmp_path / "artifact.json"
        full_artifact.save(file_path)

        assert file_path.exists()

        loaded = DebateArtifact.load(file_path)
        assert loaded.debate_id == full_artifact.debate_id
        assert loaded.task == full_artifact.task

    def test_save_creates_valid_json(self, minimal_artifact, tmp_path):
        """Test save creates valid JSON file."""
        file_path = tmp_path / "test.json"
        minimal_artifact.save(file_path)

        content = file_path.read_text()
        parsed = json.loads(content)
        assert "debate_id" in parsed

    def test_verify_integrity_no_provenance(self, minimal_artifact):
        """Test verify_integrity without provenance data."""
        valid, errors = minimal_artifact.verify_integrity()
        assert valid is True
        assert errors == []

    def test_verify_integrity_with_provenance_mock(self):
        """Test verify_integrity with mocked provenance verification."""
        artifact = DebateArtifact(
            task="Test",
            provenance_data={"chain": {"records": []}},
        )

        with patch("aragora.reasoning.provenance.ProvenanceChain") as MockChain:
            mock_chain = MagicMock()
            mock_chain.verify_chain.return_value = (True, [])
            MockChain.from_dict.return_value = mock_chain

            valid, errors = artifact.verify_integrity()
            assert valid is True
            assert errors == []

    def test_verify_integrity_provenance_fails(self):
        """Test verify_integrity when provenance verification fails."""
        artifact = DebateArtifact(
            task="Test",
            provenance_data={"chain": {}},
        )

        with patch("aragora.reasoning.provenance.ProvenanceChain") as MockChain:
            mock_chain = MagicMock()
            mock_chain.verify_chain.return_value = (False, ["Hash mismatch at record 2"])
            MockChain.from_dict.return_value = mock_chain

            valid, errors = artifact.verify_integrity()
            assert valid is False
            assert "Hash mismatch at record 2" in errors

    def test_verify_integrity_provenance_exception(self):
        """Test verify_integrity handles provenance exceptions."""
        artifact = DebateArtifact(
            task="Test",
            provenance_data={"invalid": "data"},
        )

        with patch("aragora.reasoning.provenance.ProvenanceChain") as MockChain:
            MockChain.from_dict.side_effect = ValueError("Invalid format")

            valid, errors = artifact.verify_integrity()
            assert valid is False
            assert any("Failed to verify provenance" in e for e in errors)


class TestDebateArtifactEdgeCases:
    """Edge case tests for DebateArtifact."""

    def test_empty_task(self):
        """Test with empty task."""
        artifact = DebateArtifact(task="")
        assert artifact.task == ""
        d = artifact.to_dict()
        assert d["task"] == ""

    def test_unicode_task(self):
        """Test with unicode characters in task."""
        unicode_task = "Débate: 日本語 テスト 🎯"
        artifact = DebateArtifact(task=unicode_task)
        assert artifact.task == unicode_task

        # Roundtrip
        json_str = artifact.to_json()
        restored = DebateArtifact.from_json(json_str)
        assert restored.task == unicode_task

    def test_deeply_nested_graph(self):
        """Test with deeply nested graph data."""
        nested = {"level1": {"level2": {"level3": {"level4": {"data": [1, 2, 3]}}}}}
        artifact = DebateArtifact(task="Nested", graph_data=nested)

        d = artifact.to_dict()
        assert d["graph"]["level1"]["level2"]["level3"]["level4"]["data"] == [1, 2, 3]

    def test_large_trace_data(self):
        """Test with large trace data."""
        events = [{"event": i, "data": "x" * 100} for i in range(1000)]
        artifact = DebateArtifact(task="Large", trace_data={"events": events})

        d = artifact.to_dict()
        assert len(d["trace"]["events"]) == 1000

    def test_many_verification_results(self):
        """Test with many verification results."""
        verifications = [
            VerificationResult(
                claim_id=f"claim-{i}",
                claim_text=f"Claim number {i}",
                status="verified" if i % 2 == 0 else "refuted",
                method="z3",
            )
            for i in range(50)
        ]
        artifact = DebateArtifact(task="Many verifications", verification_results=verifications)

        assert len(artifact.verification_results) == 50
        d = artifact.to_dict()
        assert len(d["verification_results"]) == 50

    def test_special_characters_in_strings(self):
        """Test special characters don't break serialization."""
        artifact = DebateArtifact(
            task='Task with "quotes" and \\backslash and \nnewline',
            graph_data={"key": 'Value with "quotes"'},
        )

        json_str = artifact.to_json()
        restored = DebateArtifact.from_json(json_str)
        assert restored.task == artifact.task


# =============================================================================
# ArtifactBuilder Tests
# =============================================================================


class TestArtifactBuilder:
    """Tests for ArtifactBuilder class."""

    def test_empty_builder(self):
        """Test empty builder creates default artifact."""
        artifact = ArtifactBuilder().build()
        assert artifact is not None
        assert artifact.task == ""
        assert artifact.agents == []

    def test_with_graph_dict(self):
        """Test with_graph accepts dict."""
        graph_data = {"nodes": {"n1": {"type": "proposal"}}}
        artifact = ArtifactBuilder().with_graph(graph_data).build()
        assert artifact.graph_data == graph_data

    def test_with_graph_object(self):
        """Test with_graph accepts object with to_dict."""
        mock_graph = MagicMock()
        mock_graph.to_dict.return_value = {"nodes": {"n1": {}}}

        artifact = ArtifactBuilder().with_graph(mock_graph).build()
        assert artifact.graph_data == {"nodes": {"n1": {}}}
        mock_graph.to_dict.assert_called_once()

    def test_with_trace_dict(self):
        """Test with_trace accepts dict."""
        trace_data = {"events": [{"type": "message"}]}
        artifact = ArtifactBuilder().with_trace(trace_data).build()
        assert artifact.trace_data == trace_data

    def test_with_trace_object(self):
        """Test with_trace accepts object with to_json."""
        mock_trace = MagicMock()
        mock_trace.to_json.return_value = '{"events": [{"type": "vote"}]}'

        artifact = ArtifactBuilder().with_trace(mock_trace).build()
        assert artifact.trace_data == {"events": [{"type": "vote"}]}

    def test_with_provenance_dict(self):
        """Test with_provenance accepts dict."""
        provenance_data = {"chain": {"records": []}}
        artifact = ArtifactBuilder().with_provenance(provenance_data).build()
        assert artifact.provenance_data == provenance_data

    def test_with_provenance_object(self):
        """Test with_provenance accepts object with export."""
        mock_prov = MagicMock()
        mock_prov.export.return_value = {"chain": {"records": [{"id": "r1"}]}}

        artifact = ArtifactBuilder().with_provenance(mock_prov).build()
        assert artifact.provenance_data["chain"]["records"][0]["id"] == "r1"

    def test_with_verification_minimal(self):
        """Test with_verification with minimal args."""
        artifact = (ArtifactBuilder()
            .with_verification("c1", "Claim 1", "verified")
            .build())

        assert len(artifact.verification_results) == 1
        v = artifact.verification_results[0]
        assert v.claim_id == "c1"
        assert v.claim_text == "Claim 1"
        assert v.status == "verified"
        assert v.method == "z3"  # default

    def test_with_verification_custom_method(self):
        """Test with_verification with custom method."""
        artifact = (ArtifactBuilder()
            .with_verification("c1", "Claim", "refuted", method="lean")
            .build())

        assert artifact.verification_results[0].method == "lean"

    def test_with_verification_kwargs(self):
        """Test with_verification passes kwargs."""
        artifact = (ArtifactBuilder()
            .with_verification(
                "c1", "Claim", "verified",
                proof_trace="QED",
                duration_ms=100,
                metadata={"key": "val"},
            )
            .build())

        v = artifact.verification_results[0]
        assert v.proof_trace == "QED"
        assert v.duration_ms == 100
        assert v.metadata == {"key": "val"}

    def test_multiple_verifications(self):
        """Test adding multiple verifications."""
        artifact = (ArtifactBuilder()
            .with_verification("c1", "Claim 1", "verified")
            .with_verification("c2", "Claim 2", "refuted")
            .with_verification("c3", "Claim 3", "timeout")
            .build())

        assert len(artifact.verification_results) == 3

    def test_chaining_order_independent(self):
        """Test that chaining order doesn't matter."""
        graph = {"nodes": {}}
        trace = {"events": []}

        a1 = (ArtifactBuilder()
            .with_graph(graph)
            .with_trace(trace)
            .build())

        a2 = (ArtifactBuilder()
            .with_trace(trace)
            .with_graph(graph)
            .build())

        assert a1.graph_data == a2.graph_data
        assert a1.trace_data == a2.trace_data

    def test_from_result_basic(self):
        """Test from_result with mock DebateResult."""
        mock_result = MagicMock()
        mock_result.id = "debate-from-result"
        mock_result.task = "Debate task"
        mock_result.rounds_used = 4
        mock_result.duration_seconds = 180.0
        mock_result.messages = [
            MagicMock(agent="agent1"),
            MagicMock(agent="agent2"),
            MagicMock(agent="agent1"),
        ]
        mock_result.critiques = [MagicMock(), MagicMock()]
        mock_result.votes = []
        mock_result.consensus_reached = True
        mock_result.confidence = 0.88
        mock_result.final_answer = "The final answer"

        # No need to patch - MagicMock works without type checking
        artifact = ArtifactBuilder().from_result(mock_result).build()

        assert artifact.debate_id == "debate-from-result"
        assert artifact.task == "Debate task"
        assert artifact.rounds == 4
        assert artifact.duration_seconds == 180.0
        assert artifact.message_count == 3
        assert artifact.critique_count == 2
        assert set(artifact.agents) == {"agent1", "agent2"}
        assert artifact.consensus_proof is not None
        assert artifact.consensus_proof.reached is True
        assert artifact.consensus_proof.confidence == 0.88

    def test_from_result_vote_breakdown(self):
        """Test from_result builds vote breakdown correctly."""
        mock_result = MagicMock()
        mock_result.id = "debate-votes"
        mock_result.task = "Vote test"
        mock_result.rounds_used = 2
        mock_result.duration_seconds = 60.0
        mock_result.messages = []
        mock_result.critiques = []
        mock_result.final_answer = "Answer that is longer than twenty characters"
        mock_result.consensus_reached = True
        mock_result.confidence = 0.9

        # Create votes - choice matches first 20 chars of final_answer
        vote1 = MagicMock()
        vote1.agent = "agent1"
        vote1.choice = "Answer that is longe"  # Matches first 20

        vote2 = MagicMock()
        vote2.agent = "agent2"
        vote2.choice = "Different answer"  # Doesn't match

        mock_result.votes = [vote1, vote2]

        artifact = ArtifactBuilder().from_result(mock_result).build()

        assert artifact.consensus_proof.vote_breakdown["agent1"] is True
        assert artifact.consensus_proof.vote_breakdown["agent2"] is False

    def test_builder_returns_self(self):
        """Test all builder methods return self for chaining."""
        builder = ArtifactBuilder()

        assert builder.with_graph({}) is builder
        assert builder.with_trace({}) is builder
        assert builder.with_provenance({}) is builder
        assert builder.with_verification("c", "t", "s") is builder


# =============================================================================
# create_artifact_from_debate Tests
# =============================================================================


class TestCreateArtifactFromDebate:
    """Tests for create_artifact_from_debate convenience function."""

    def test_with_result_only(self):
        """Test with only result."""
        mock_result = MagicMock()
        mock_result.id = "conv-123"
        mock_result.task = "Convenience test"
        mock_result.rounds_used = 1
        mock_result.duration_seconds = 30.0
        mock_result.messages = []
        mock_result.critiques = []
        mock_result.votes = []
        mock_result.consensus_reached = False
        mock_result.confidence = 0.5
        mock_result.final_answer = ""

        artifact = create_artifact_from_debate(mock_result)

        assert artifact.debate_id == "conv-123"
        assert artifact.graph_data is None
        assert artifact.trace_data is None
        assert artifact.provenance_data is None

    def test_with_all_components(self):
        """Test with all optional components."""
        mock_result = MagicMock()
        mock_result.id = "full-conv"
        mock_result.task = "Full test"
        mock_result.rounds_used = 3
        mock_result.duration_seconds = 90.0
        mock_result.messages = [MagicMock(agent="a1")]
        mock_result.critiques = []
        mock_result.votes = []
        mock_result.consensus_reached = True
        mock_result.confidence = 0.95
        mock_result.final_answer = "Final"

        mock_graph = MagicMock()
        mock_graph.to_dict.return_value = {"nodes": {"n1": {}}}

        mock_trace = MagicMock()
        mock_trace.to_json.return_value = '{"events": []}'

        mock_prov = MagicMock()
        mock_prov.export.return_value = {"chain": {}}

        artifact = create_artifact_from_debate(
            mock_result,
            graph=mock_graph,
            trace=mock_trace,
            provenance=mock_prov,
        )

        assert artifact.graph_data == {"nodes": {"n1": {}}}
        assert artifact.trace_data == {"events": []}
        assert artifact.provenance_data == {"chain": {}}

    def test_with_partial_components(self):
        """Test with some optional components."""
        mock_result = MagicMock()
        mock_result.id = "partial"
        mock_result.task = "Partial"
        mock_result.rounds_used = 1
        mock_result.duration_seconds = 10.0
        mock_result.messages = []
        mock_result.critiques = []
        mock_result.votes = []
        mock_result.consensus_reached = False
        mock_result.confidence = 0.0
        mock_result.final_answer = ""

        artifact = create_artifact_from_debate(
            mock_result,
            graph={"nodes": {}},  # dict directly
            trace=None,
            provenance=None,
        )

        assert artifact.graph_data == {"nodes": {}}
        assert artifact.trace_data is None
        assert artifact.provenance_data is None
