"""
Tests for aragora.export.artifact module.

Tests cover:
- ConsensusProof dataclass
- VerificationResult dataclass
- DebateArtifact dataclass
- ArtifactBuilder class
- Serialization/deserialization (to_dict, from_dict, to_json, from_json)
- File operations (save, load)
- Content hash computation
- Integrity verification
- Convenience function (create_artifact_from_debate)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aragora.export.artifact import (
    ArtifactBuilder,
    ConsensusProof,
    DebateArtifact,
    VerificationResult,
    create_artifact_from_debate,
)


# =============================================================================
# TestConsensusProof
# =============================================================================


class TestConsensusProofInit:
    """Tests for ConsensusProof initialization."""

    def test_create_with_required_fields(self):
        """Should create with required fields."""
        proof = ConsensusProof(
            reached=True,
            confidence=0.85,
            vote_breakdown={"claude": True, "gpt-4": True},
            final_answer="Agreed on REST API",
            rounds_used=3,
        )

        assert proof.reached is True
        assert proof.confidence == 0.85
        assert len(proof.vote_breakdown) == 2
        assert proof.final_answer == "Agreed on REST API"
        assert proof.rounds_used == 3

    def test_auto_generates_timestamp(self):
        """Should auto-generate timestamp if not provided."""
        proof = ConsensusProof(
            reached=True,
            confidence=0.9,
            vote_breakdown={},
            final_answer="Answer",
            rounds_used=1,
        )

        assert proof.timestamp is not None
        assert len(proof.timestamp) > 0

    def test_accepts_custom_timestamp(self):
        """Should accept custom timestamp."""
        custom_timestamp = "2024-01-15T10:00:00Z"
        proof = ConsensusProof(
            reached=True,
            confidence=0.9,
            vote_breakdown={},
            final_answer="Answer",
            rounds_used=1,
            timestamp=custom_timestamp,
        )

        assert proof.timestamp == custom_timestamp


class TestConsensusProofToDict:
    """Tests for ConsensusProof.to_dict()."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        proof = ConsensusProof(
            reached=True,
            confidence=0.85,
            vote_breakdown={"claude": True},
            final_answer="Answer",
            rounds_used=3,
        )

        result = proof.to_dict()
        assert isinstance(result, dict)

    def test_includes_all_fields(self):
        """Should include all fields in dict."""
        proof = ConsensusProof(
            reached=True,
            confidence=0.85,
            vote_breakdown={"claude": True, "gpt-4": False},
            final_answer="Final answer",
            rounds_used=3,
            timestamp="2024-01-15T10:00:00Z",
        )

        result = proof.to_dict()

        assert result["reached"] is True
        assert result["confidence"] == 0.85
        assert result["vote_breakdown"] == {"claude": True, "gpt-4": False}
        assert result["final_answer"] == "Final answer"
        assert result["rounds_used"] == 3
        assert result["timestamp"] == "2024-01-15T10:00:00Z"


# =============================================================================
# TestVerificationResult
# =============================================================================


class TestVerificationResultInit:
    """Tests for VerificationResult initialization."""

    def test_create_with_required_fields(self):
        """Should create with required fields."""
        result = VerificationResult(
            claim_id="claim-001",
            claim_text="All inputs are sanitized",
            status="verified",
            method="z3",
        )

        assert result.claim_id == "claim-001"
        assert result.claim_text == "All inputs are sanitized"
        assert result.status == "verified"
        assert result.method == "z3"

    def test_optional_fields_default_to_none(self):
        """Should have None/empty defaults for optional fields."""
        result = VerificationResult(
            claim_id="claim",
            claim_text="Claim",
            status="verified",
            method="z3",
        )

        assert result.proof_trace is None
        assert result.counterexample is None
        assert result.duration_ms == 0
        assert result.metadata == {}

    def test_accepts_all_optional_fields(self):
        """Should accept all optional fields."""
        result = VerificationResult(
            claim_id="claim",
            claim_text="Claim",
            status="refuted",
            method="z3",
            proof_trace="(assert ...)",
            counterexample="x = 5",
            duration_ms=150,
            metadata={"solver": "z3", "version": "4.8"},
        )

        assert result.proof_trace == "(assert ...)"
        assert result.counterexample == "x = 5"
        assert result.duration_ms == 150
        assert result.metadata["solver"] == "z3"


class TestVerificationResultToDict:
    """Tests for VerificationResult.to_dict()."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        result = VerificationResult(
            claim_id="claim",
            claim_text="Claim",
            status="verified",
            method="z3",
        )

        assert isinstance(result.to_dict(), dict)

    def test_includes_all_fields(self):
        """Should include all fields in dict."""
        result = VerificationResult(
            claim_id="claim-001",
            claim_text="Test claim",
            status="verified",
            method="z3",
            proof_trace="proof",
            counterexample=None,
            duration_ms=100,
            metadata={"key": "value"},
        )

        d = result.to_dict()

        assert d["claim_id"] == "claim-001"
        assert d["claim_text"] == "Test claim"
        assert d["status"] == "verified"
        assert d["method"] == "z3"
        assert d["proof_trace"] == "proof"
        assert d["counterexample"] is None
        assert d["duration_ms"] == 100
        assert d["metadata"] == {"key": "value"}


# =============================================================================
# TestDebateArtifactInit
# =============================================================================


class TestDebateArtifactInit:
    """Tests for DebateArtifact initialization."""

    def test_creates_with_defaults(self):
        """Should create with sensible defaults."""
        artifact = DebateArtifact()

        assert artifact.artifact_id is not None
        assert len(artifact.artifact_id) == 12
        assert artifact.debate_id == ""
        assert artifact.task == ""
        assert artifact.agents == []
        assert artifact.rounds == 0

    def test_accepts_all_fields(self):
        """Should accept all fields."""
        artifact = DebateArtifact(
            artifact_id="custom-id",
            debate_id="debate-001",
            task="Test task",
            agents=["claude", "gpt-4"],
            rounds=3,
            duration_seconds=60.5,
            message_count=10,
            critique_count=5,
        )

        assert artifact.artifact_id == "custom-id"
        assert artifact.debate_id == "debate-001"
        assert artifact.task == "Test task"
        assert artifact.agents == ["claude", "gpt-4"]
        assert artifact.rounds == 3
        assert artifact.duration_seconds == 60.5
        assert artifact.message_count == 10
        assert artifact.critique_count == 5

    def test_auto_generates_created_at(self):
        """Should auto-generate created_at timestamp."""
        artifact = DebateArtifact()

        assert artifact.created_at is not None
        assert len(artifact.created_at) > 0


# =============================================================================
# TestDebateArtifactContentHash
# =============================================================================


class TestDebateArtifactContentHash:
    """Tests for DebateArtifact.content_hash property."""

    def test_returns_hash_string(self):
        """Should return a hash string."""
        artifact = DebateArtifact(task="Test task")
        hash_value = artifact.content_hash

        assert isinstance(hash_value, str)
        assert len(hash_value) == 16

    def test_same_content_same_hash(self):
        """Should produce same hash for same content."""
        artifact1 = DebateArtifact(
            task="Same task",
            graph_data={"nodes": {}},
        )
        artifact2 = DebateArtifact(
            task="Same task",
            graph_data={"nodes": {}},
        )

        assert artifact1.content_hash == artifact2.content_hash

    def test_different_content_different_hash(self):
        """Should produce different hash for different content."""
        artifact1 = DebateArtifact(task="Task 1")
        artifact2 = DebateArtifact(task="Task 2")

        assert artifact1.content_hash != artifact2.content_hash


# =============================================================================
# TestDebateArtifactSerialization
# =============================================================================


class TestDebateArtifactToDict:
    """Tests for DebateArtifact.to_dict()."""

    def test_returns_dict(self):
        """Should return a dictionary."""
        artifact = DebateArtifact()
        result = artifact.to_dict()

        assert isinstance(result, dict)

    def test_includes_all_fields(self):
        """Should include all fields."""
        artifact = DebateArtifact(
            artifact_id="test-id",
            debate_id="debate-001",
            task="Test task",
            agents=["claude"],
            rounds=3,
        )

        result = artifact.to_dict()

        assert result["artifact_id"] == "test-id"
        assert result["debate_id"] == "debate-001"
        assert result["task"] == "Test task"
        assert result["agents"] == ["claude"]
        assert result["rounds"] == 3

    def test_includes_content_hash(self):
        """Should include computed content hash."""
        artifact = DebateArtifact(task="Test")
        result = artifact.to_dict()

        assert "content_hash" in result
        assert len(result["content_hash"]) == 16

    def test_serializes_consensus_proof(self):
        """Should serialize consensus proof."""
        artifact = DebateArtifact(
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                vote_breakdown={"claude": True},
                final_answer="Answer",
                rounds_used=2,
            )
        )

        result = artifact.to_dict()

        assert result["consensus_proof"] is not None
        assert result["consensus_proof"]["reached"] is True
        assert result["consensus_proof"]["confidence"] == 0.9

    def test_serializes_verification_results(self):
        """Should serialize verification results."""
        artifact = DebateArtifact(
            verification_results=[
                VerificationResult(
                    claim_id="c1",
                    claim_text="Claim 1",
                    status="verified",
                    method="z3",
                ),
                VerificationResult(
                    claim_id="c2",
                    claim_text="Claim 2",
                    status="refuted",
                    method="z3",
                ),
            ]
        )

        result = artifact.to_dict()

        assert len(result["verification_results"]) == 2
        assert result["verification_results"][0]["claim_id"] == "c1"
        assert result["verification_results"][1]["status"] == "refuted"


class TestDebateArtifactToJson:
    """Tests for DebateArtifact.to_json()."""

    def test_returns_json_string(self):
        """Should return a valid JSON string."""
        artifact = DebateArtifact(task="Test")
        result = artifact.to_json()

        assert isinstance(result, str)
        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["task"] == "Test"

    def test_respects_indent_parameter(self):
        """Should respect indent parameter."""
        artifact = DebateArtifact()

        result_compact = artifact.to_json(indent=None)
        result_indented = artifact.to_json(indent=4)

        # Indented version should be longer due to whitespace
        assert len(result_indented) > len(result_compact)
        assert "    " in result_indented


class TestDebateArtifactFromDict:
    """Tests for DebateArtifact.from_dict()."""

    def test_creates_artifact_from_dict(self):
        """Should create artifact from dictionary."""
        data = {
            "artifact_id": "test-id",
            "debate_id": "debate-001",
            "task": "Test task",
            "agents": ["claude", "gpt-4"],
            "rounds": 3,
        }

        artifact = DebateArtifact.from_dict(data)

        assert artifact.artifact_id == "test-id"
        assert artifact.debate_id == "debate-001"
        assert artifact.task == "Test task"
        assert artifact.agents == ["claude", "gpt-4"]
        assert artifact.rounds == 3

    def test_reconstructs_consensus_proof(self):
        """Should reconstruct consensus proof from dict."""
        data = {
            "consensus_proof": {
                "reached": True,
                "confidence": 0.85,
                "vote_breakdown": {"claude": True},
                "final_answer": "Answer",
                "rounds_used": 2,
                "timestamp": "2024-01-15T10:00:00Z",
            }
        }

        artifact = DebateArtifact.from_dict(data)

        assert artifact.consensus_proof is not None
        assert artifact.consensus_proof.reached is True
        assert artifact.consensus_proof.confidence == 0.85

    def test_reconstructs_verification_results(self):
        """Should reconstruct verification results from dict."""
        data = {
            "verification_results": [
                {
                    "claim_id": "c1",
                    "claim_text": "Claim",
                    "status": "verified",
                    "method": "z3",
                }
            ]
        }

        artifact = DebateArtifact.from_dict(data)

        assert len(artifact.verification_results) == 1
        assert artifact.verification_results[0].claim_id == "c1"

    def test_handles_missing_optional_fields(self):
        """Should handle missing optional fields with defaults."""
        data = {"artifact_id": "test"}

        artifact = DebateArtifact.from_dict(data)

        assert artifact.artifact_id == "test"
        assert artifact.debate_id == ""
        assert artifact.consensus_proof is None


class TestDebateArtifactFromJson:
    """Tests for DebateArtifact.from_json()."""

    def test_creates_artifact_from_json(self):
        """Should create artifact from JSON string."""
        json_str = '{"artifact_id": "test-id", "task": "Test task"}'

        artifact = DebateArtifact.from_json(json_str)

        assert artifact.artifact_id == "test-id"
        assert artifact.task == "Test task"


class TestDebateArtifactRoundtrip:
    """Tests for serialization roundtrip."""

    def test_dict_roundtrip(self):
        """Should preserve data through dict roundtrip."""
        original = DebateArtifact(
            artifact_id="test",
            debate_id="debate-001",
            task="Test task",
            agents=["claude", "gpt-4"],
            rounds=3,
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.85,
                vote_breakdown={"claude": True},
                final_answer="Answer",
                rounds_used=2,
            ),
        )

        restored = DebateArtifact.from_dict(original.to_dict())

        assert restored.artifact_id == original.artifact_id
        assert restored.task == original.task
        assert restored.consensus_proof.reached == original.consensus_proof.reached

    def test_json_roundtrip(self):
        """Should preserve data through JSON roundtrip."""
        original = DebateArtifact(
            artifact_id="test",
            task="Test task",
            verification_results=[
                VerificationResult(
                    claim_id="c1",
                    claim_text="Claim",
                    status="verified",
                    method="z3",
                )
            ],
        )

        json_str = original.to_json()
        restored = DebateArtifact.from_json(json_str)

        assert restored.artifact_id == original.artifact_id
        assert len(restored.verification_results) == 1


# =============================================================================
# TestDebateArtifactFileOperations
# =============================================================================


class TestDebateArtifactSave:
    """Tests for DebateArtifact.save()."""

    def test_saves_to_file(self):
        """Should save artifact to file."""
        artifact = DebateArtifact(
            artifact_id="test",
            task="Test task",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.json"
            artifact.save(path)

            assert path.exists()
            content = path.read_text()
            assert "test" in content

    def test_saved_file_is_valid_json(self):
        """Should save valid JSON content."""
        artifact = DebateArtifact(task="Test")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.json"
            artifact.save(path)

            content = path.read_text()
            parsed = json.loads(content)
            assert parsed["task"] == "Test"


class TestDebateArtifactLoad:
    """Tests for DebateArtifact.load()."""

    def test_loads_from_file(self):
        """Should load artifact from file."""
        artifact = DebateArtifact(
            artifact_id="test-load",
            task="Loaded task",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.json"
            artifact.save(path)

            loaded = DebateArtifact.load(path)

            assert loaded.artifact_id == "test-load"
            assert loaded.task == "Loaded task"

    def test_raises_for_missing_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            DebateArtifact.load(Path("/nonexistent/path.json"))


# =============================================================================
# TestDebateArtifactIntegrity
# =============================================================================


class TestDebateArtifactVerifyIntegrity:
    """Tests for DebateArtifact.verify_integrity()."""

    def test_returns_valid_for_artifact_without_provenance(self):
        """Should return valid for artifact without provenance data."""
        artifact = DebateArtifact(task="Test")

        is_valid, errors = artifact.verify_integrity()

        assert is_valid is True
        assert errors == []


# =============================================================================
# TestArtifactBuilder
# =============================================================================


class TestArtifactBuilderInit:
    """Tests for ArtifactBuilder initialization."""

    def test_creates_empty_artifact(self):
        """Should create an empty artifact on init."""
        builder = ArtifactBuilder()
        artifact = builder.build()

        assert artifact is not None
        assert isinstance(artifact, DebateArtifact)


class TestArtifactBuilderFromResult:
    """Tests for ArtifactBuilder.from_result()."""

    def test_extracts_basic_fields(self):
        """Should extract basic fields from result."""
        mock_result = MagicMock()
        mock_result.id = "debate-001"
        mock_result.task = "Test task"
        mock_result.rounds_used = 3
        mock_result.duration_seconds = 45.5
        mock_result.messages = [MagicMock(agent="claude"), MagicMock(agent="gpt-4")]
        mock_result.critiques = [MagicMock()]
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.final_answer = "Final answer"
        mock_result.votes = []

        artifact = ArtifactBuilder().from_result(mock_result).build()

        assert artifact.debate_id == "debate-001"
        assert artifact.task == "Test task"
        assert artifact.rounds == 3
        assert artifact.duration_seconds == 45.5
        assert artifact.message_count == 2
        assert artifact.critique_count == 1

    def test_extracts_agents(self):
        """Should extract unique agent names."""
        mock_result = MagicMock()
        mock_result.id = "debate"
        mock_result.task = "Task"
        mock_result.rounds_used = 1
        mock_result.duration_seconds = 10
        mock_result.messages = [
            MagicMock(agent="claude"),
            MagicMock(agent="gpt-4"),
            MagicMock(agent="claude"),  # Duplicate
        ]
        mock_result.critiques = []
        mock_result.consensus_reached = True
        mock_result.confidence = 0.9
        mock_result.final_answer = "Answer"
        mock_result.votes = []

        artifact = ArtifactBuilder().from_result(mock_result).build()

        assert len(artifact.agents) == 2
        assert "claude" in artifact.agents
        assert "gpt-4" in artifact.agents

    def test_creates_consensus_proof(self):
        """Should create consensus proof from result."""
        mock_result = MagicMock()
        mock_result.id = "debate"
        mock_result.task = "Task"
        mock_result.rounds_used = 2
        mock_result.duration_seconds = 30
        mock_result.messages = []
        mock_result.critiques = []
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.final_answer = "Final answer here"
        mock_result.votes = []

        artifact = ArtifactBuilder().from_result(mock_result).build()

        assert artifact.consensus_proof is not None
        assert artifact.consensus_proof.reached is True
        assert artifact.consensus_proof.confidence == 0.85
        assert artifact.consensus_proof.final_answer == "Final answer here"


class TestArtifactBuilderWithGraph:
    """Tests for ArtifactBuilder.with_graph()."""

    def test_accepts_dict_graph(self):
        """Should accept graph as dictionary."""
        graph_dict = {"nodes": {"n1": {}}, "edges": []}

        artifact = ArtifactBuilder().with_graph(graph_dict).build()

        assert artifact.graph_data == graph_dict

    def test_accepts_graph_object_with_to_dict(self):
        """Should accept graph object with to_dict method."""
        mock_graph = MagicMock()
        mock_graph.to_dict.return_value = {"nodes": {}, "edges": []}

        artifact = ArtifactBuilder().with_graph(mock_graph).build()

        assert artifact.graph_data == {"nodes": {}, "edges": []}


class TestArtifactBuilderWithTrace:
    """Tests for ArtifactBuilder.with_trace()."""

    def test_accepts_dict_trace(self):
        """Should accept trace as dictionary."""
        trace_dict = {"events": [{"type": "message"}]}

        artifact = ArtifactBuilder().with_trace(trace_dict).build()

        assert artifact.trace_data == trace_dict

    def test_accepts_trace_object_with_to_json(self):
        """Should accept trace object with to_json method."""
        mock_trace = MagicMock()
        mock_trace.to_json.return_value = '{"events": []}'

        artifact = ArtifactBuilder().with_trace(mock_trace).build()

        assert artifact.trace_data == {"events": []}


class TestArtifactBuilderWithProvenance:
    """Tests for ArtifactBuilder.with_provenance()."""

    def test_accepts_dict_provenance(self):
        """Should accept provenance as dictionary."""
        prov_dict = {"chain": {"records": []}}

        artifact = ArtifactBuilder().with_provenance(prov_dict).build()

        assert artifact.provenance_data == prov_dict

    def test_accepts_provenance_object_with_export(self):
        """Should accept provenance object with export method."""
        mock_prov = MagicMock()
        mock_prov.export.return_value = {"chain": {"records": []}}

        artifact = ArtifactBuilder().with_provenance(mock_prov).build()

        assert artifact.provenance_data == {"chain": {"records": []}}


class TestArtifactBuilderWithVerification:
    """Tests for ArtifactBuilder.with_verification()."""

    def test_adds_verification_result(self):
        """Should add verification result to artifact."""
        artifact = (
            ArtifactBuilder()
            .with_verification(
                claim_id="c1",
                claim_text="Test claim",
                status="verified",
                method="z3",
            )
            .build()
        )

        assert len(artifact.verification_results) == 1
        assert artifact.verification_results[0].claim_id == "c1"

    def test_adds_multiple_verifications(self):
        """Should accumulate multiple verifications."""
        artifact = (
            ArtifactBuilder()
            .with_verification(
                claim_id="c1",
                claim_text="Claim 1",
                status="verified",
                method="z3",
            )
            .with_verification(
                claim_id="c2",
                claim_text="Claim 2",
                status="refuted",
                method="z3",
            )
            .build()
        )

        assert len(artifact.verification_results) == 2


class TestArtifactBuilderChaining:
    """Tests for ArtifactBuilder method chaining."""

    def test_supports_method_chaining(self):
        """Should support fluent method chaining."""
        mock_result = MagicMock()
        mock_result.id = "debate"
        mock_result.task = "Task"
        mock_result.rounds_used = 1
        mock_result.duration_seconds = 10
        mock_result.messages = []
        mock_result.critiques = []
        mock_result.consensus_reached = True
        mock_result.confidence = 0.9
        mock_result.final_answer = "Answer"
        mock_result.votes = []

        artifact = (
            ArtifactBuilder()
            .from_result(mock_result)
            .with_graph({"nodes": {}})
            .with_trace({"events": []})
            .with_provenance({"chain": {}})
            .with_verification(
                claim_id="c1",
                claim_text="Claim",
                status="verified",
                method="z3",
            )
            .build()
        )

        assert artifact.debate_id == "debate"
        assert artifact.graph_data is not None
        assert artifact.trace_data is not None
        assert artifact.provenance_data is not None
        assert len(artifact.verification_results) == 1


# =============================================================================
# TestCreateArtifactFromDebate
# =============================================================================


class TestCreateArtifactFromDebate:
    """Tests for create_artifact_from_debate convenience function."""

    def test_creates_artifact_from_result(self):
        """Should create artifact from debate result."""
        mock_result = MagicMock()
        mock_result.id = "debate-001"
        mock_result.task = "Test task"
        mock_result.rounds_used = 2
        mock_result.duration_seconds = 30
        mock_result.messages = [MagicMock(agent="claude")]
        mock_result.critiques = []
        mock_result.consensus_reached = True
        mock_result.confidence = 0.9
        mock_result.final_answer = "Answer"
        mock_result.votes = []

        artifact = create_artifact_from_debate(mock_result)

        assert artifact.debate_id == "debate-001"
        assert artifact.task == "Test task"

    def test_includes_optional_components(self):
        """Should include optional components when provided."""
        mock_result = MagicMock()
        mock_result.id = "debate"
        mock_result.task = "Task"
        mock_result.rounds_used = 1
        mock_result.duration_seconds = 10
        mock_result.messages = []
        mock_result.critiques = []
        mock_result.consensus_reached = True
        mock_result.confidence = 0.9
        mock_result.final_answer = "Answer"
        mock_result.votes = []

        artifact = create_artifact_from_debate(
            mock_result,
            graph={"nodes": {}},
            trace={"events": []},
            provenance={"chain": {}},
        )

        assert artifact.graph_data is not None
        assert artifact.trace_data is not None
        assert artifact.provenance_data is not None


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestArtifactEdgeCases:
    """Edge case tests for artifact module."""

    def test_handles_empty_vote_breakdown(self):
        """Should handle empty vote breakdown."""
        proof = ConsensusProof(
            reached=False,
            confidence=0.0,
            vote_breakdown={},
            final_answer="",
            rounds_used=0,
        )

        assert proof.vote_breakdown == {}

    def test_handles_special_characters_in_task(self):
        """Should handle special characters in task."""
        artifact = DebateArtifact(
            task='Task with "quotes" and <brackets> & ampersands',
        )

        json_str = artifact.to_json()
        restored = DebateArtifact.from_json(json_str)

        assert '"quotes"' in restored.task
        assert "<brackets>" in restored.task

    def test_handles_unicode_in_content(self):
        """Should handle unicode characters."""
        artifact = DebateArtifact(
            task="International task",
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                vote_breakdown={},
                final_answer="International answer",
                rounds_used=1,
            ),
        )

        json_str = artifact.to_json()
        restored = DebateArtifact.from_json(json_str)

        assert "International" in restored.task

    def test_handles_very_long_content(self):
        """Should handle very long content."""
        long_task = "A" * 10000
        artifact = DebateArtifact(task=long_task)

        json_str = artifact.to_json()
        restored = DebateArtifact.from_json(json_str)

        assert len(restored.task) == 10000
