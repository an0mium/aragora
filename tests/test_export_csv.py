"""Tests for aragora.export.csv_exporter module.

Comprehensive tests for CSVExporter class and related functions.
"""

import csv
import pytest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

from aragora.export.artifact import (
    DebateArtifact,
    ConsensusProof,
    VerificationResult,
)
from aragora.export.csv_exporter import (
    CSVExporter,
    export_debate_to_csv,
    export_multiple_debates,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def minimal_artifact():
    """Create minimal artifact for testing."""
    return DebateArtifact(
        artifact_id="art-001",
        debate_id="debate-123",
        task="Test task",
        agents=["agent1", "agent2"],
        rounds=3,
        duration_seconds=60.0,
        message_count=10,
        critique_count=5,
    )


@pytest.fixture
def artifact_with_trace():
    """Create artifact with trace data."""
    return DebateArtifact(
        artifact_id="art-trace",
        debate_id="debate-trace",
        task="Trace test task",
        trace_data={
            "events": [
                {
                    "event_type": "message",
                    "round": 1,
                    "agent": "agent1",
                    "role": "proposer",
                    "content": "This is my proposal.",
                    "timestamp": "2024-01-15T10:00:00",
                },
                {
                    "event_type": "message",
                    "round": 1,
                    "agent": "agent2",
                    "role": "critic",
                    "content": "I have concerns about this.",
                    "timestamp": "2024-01-15T10:01:00",
                },
                {
                    "event_type": "critique",
                    "round": 1,
                    "agent": "agent2",
                    "target": "agent1",
                    "severity": 0.7,
                    "issues": ["Missing evidence", "Unclear reasoning"],
                    "accepted": True,
                },
                {
                    "event_type": "vote",
                    "agent": "agent1",
                    "choice": "Option A",
                },
            ]
        },
        agents=["agent1", "agent2"],
        rounds=2,
    )


@pytest.fixture
def artifact_with_consensus():
    """Create artifact with consensus proof."""
    return DebateArtifact(
        artifact_id="art-consensus",
        debate_id="debate-consensus",
        task="Consensus test",
        consensus_proof=ConsensusProof(
            reached=True,
            confidence=0.85,
            vote_breakdown={"agent1": True, "agent2": True, "agent3": False},
            final_answer="The consensus answer is that we should proceed with option A.",
            rounds_used=3,
            timestamp="2024-01-15T12:00:00",
        ),
        agents=["agent1", "agent2", "agent3"],
        rounds=3,
    )


@pytest.fixture
def artifact_with_verifications():
    """Create artifact with verification results."""
    return DebateArtifact(
        artifact_id="art-verify",
        debate_id="debate-verify",
        task="Verification test",
        verification_results=[
            VerificationResult(
                claim_id="claim-1",
                claim_text="All inputs are validated",
                status="verified",
                method="z3",
                proof_trace="QED",
                duration_ms=150,
            ),
            VerificationResult(
                claim_id="claim-2",
                claim_text="Output is always positive",
                status="refuted",
                method="z3",
                counterexample="x = -1",
                duration_ms=200,
            ),
            VerificationResult(
                claim_id="claim-3",
                claim_text="Algorithm terminates",
                status="timeout",
                method="lean",
                duration_ms=60000,
            ),
        ],
    )


@pytest.fixture
def full_artifact(artifact_with_trace, artifact_with_consensus, artifact_with_verifications):
    """Create artifact with all components."""
    return DebateArtifact(
        artifact_id="art-full",
        debate_id="debate-full",
        task="Full test task with all components",
        created_at="2024-01-15T12:00:00",
        trace_data=artifact_with_trace.trace_data,
        consensus_proof=artifact_with_consensus.consensus_proof,
        verification_results=artifact_with_verifications.verification_results,
        agents=["agent1", "agent2", "agent3"],
        rounds=3,
        duration_seconds=180.0,
        message_count=20,
        critique_count=8,
    )


# =============================================================================
# CSVExporter Tests - Messages
# =============================================================================


class TestCSVExporterMessages:
    """Tests for CSVExporter.export_messages()."""

    def test_export_messages_header(self, minimal_artifact):
        """Test messages export includes correct header."""
        exporter = CSVExporter(minimal_artifact)
        content = exporter.export_messages()

        reader = csv.reader(StringIO(content))
        header = next(reader)
        assert header == ['debate_id', 'round', 'agent', 'role', 'content', 'timestamp']

    def test_export_messages_no_trace_data(self, minimal_artifact):
        """Test export with no trace data returns only header."""
        exporter = CSVExporter(minimal_artifact)
        content = exporter.export_messages()

        lines = content.strip().split('\n')
        assert len(lines) == 1  # Only header

    def test_export_messages_with_events(self, artifact_with_trace):
        """Test export with message events."""
        exporter = CSVExporter(artifact_with_trace)
        content = exporter.export_messages()

        reader = csv.reader(StringIO(content))
        rows = list(reader)

        assert len(rows) == 3  # Header + 2 messages
        # First message
        assert rows[1][0] == "debate-trace"  # debate_id
        assert rows[1][1] == "1"  # round
        assert rows[1][2] == "agent1"  # agent
        assert rows[1][3] == "proposer"  # role
        assert rows[1][4] == "This is my proposal."  # content

    def test_export_messages_filters_event_types(self, artifact_with_trace):
        """Test export only includes message events, not critiques or votes."""
        exporter = CSVExporter(artifact_with_trace)
        content = exporter.export_messages()

        # Should not contain critique or vote data
        assert "Missing evidence" not in content
        assert "Option A" not in content

    def test_export_messages_to_file(self, artifact_with_trace, tmp_path):
        """Test export writes to file."""
        exporter = CSVExporter(artifact_with_trace)
        output_path = tmp_path / "messages.csv"

        content = exporter.export_messages(output_path)

        assert output_path.exists()
        file_content = output_path.read_text()
        # Normalize line endings for comparison
        assert file_content.replace('\r\n', '\n') == content.replace('\r\n', '\n')

    def test_export_messages_special_characters(self):
        """Test export handles special characters in content."""
        artifact = DebateArtifact(
            debate_id="special",
            trace_data={
                "events": [
                    {
                        "event_type": "message",
                        "round": 1,
                        "agent": "agent1",
                        "content": 'Content with "quotes", commas, and\nnewlines',
                    }
                ]
            },
        )
        exporter = CSVExporter(artifact)
        content = exporter.export_messages()

        # CSV module should properly escape special characters
        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert 'Content with "quotes", commas, and\nnewlines' == rows[1][4]

    def test_export_messages_missing_fields(self):
        """Test export handles events with missing fields."""
        artifact = DebateArtifact(
            debate_id="missing",
            trace_data={
                "events": [
                    {"event_type": "message"},  # All other fields missing
                ]
            },
        )
        exporter = CSVExporter(artifact)
        content = exporter.export_messages()

        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert len(rows) == 2
        assert rows[1][1] == "0"  # Default round
        assert rows[1][2] == ""  # Empty agent


# =============================================================================
# CSVExporter Tests - Critiques
# =============================================================================


class TestCSVExporterCritiques:
    """Tests for CSVExporter.export_critiques()."""

    def test_export_critiques_header(self, minimal_artifact):
        """Test critiques export includes correct header."""
        exporter = CSVExporter(minimal_artifact)
        content = exporter.export_critiques()

        reader = csv.reader(StringIO(content))
        header = next(reader)
        assert header == ['debate_id', 'round', 'critic', 'target', 'severity', 'issue_count', 'issues', 'accepted']

    def test_export_critiques_no_trace_data(self, minimal_artifact):
        """Test export with no trace data returns only header."""
        exporter = CSVExporter(minimal_artifact)
        content = exporter.export_critiques()

        lines = content.strip().split('\n')
        assert len(lines) == 1

    def test_export_critiques_with_events(self, artifact_with_trace):
        """Test export with critique events."""
        exporter = CSVExporter(artifact_with_trace)
        content = exporter.export_critiques()

        reader = csv.reader(StringIO(content))
        rows = list(reader)

        assert len(rows) == 2  # Header + 1 critique
        assert rows[1][2] == "agent2"  # critic
        assert rows[1][3] == "agent1"  # target
        assert rows[1][4] == "0.7"  # severity
        assert rows[1][5] == "2"  # issue_count
        assert "Missing evidence" in rows[1][6]  # issues

    def test_export_critiques_issues_as_list(self):
        """Test export joins issues list with semicolon."""
        artifact = DebateArtifact(
            debate_id="issues-list",
            trace_data={
                "events": [
                    {
                        "event_type": "critique",
                        "issues": ["Issue 1", "Issue 2", "Issue 3"],
                    }
                ]
            },
        )
        exporter = CSVExporter(artifact)
        content = exporter.export_critiques()

        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert rows[1][6] == "Issue 1; Issue 2; Issue 3"

    def test_export_critiques_issues_as_string(self):
        """Test export handles issues as string."""
        artifact = DebateArtifact(
            debate_id="issues-string",
            trace_data={
                "events": [
                    {
                        "event_type": "critique",
                        "issues": "Single issue string",
                    }
                ]
            },
        )
        exporter = CSVExporter(artifact)
        content = exporter.export_critiques()

        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert rows[1][6] == "Single issue string"

    def test_export_critiques_empty_issues(self):
        """Test export handles empty issues list."""
        artifact = DebateArtifact(
            debate_id="empty-issues",
            trace_data={
                "events": [
                    {
                        "event_type": "critique",
                        "issues": [],
                    }
                ]
            },
        )
        exporter = CSVExporter(artifact)
        content = exporter.export_critiques()

        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert rows[1][5] == "0"  # issue_count
        assert rows[1][6] == ""  # empty issues

    def test_export_critiques_to_file(self, artifact_with_trace, tmp_path):
        """Test export writes to file."""
        exporter = CSVExporter(artifact_with_trace)
        output_path = tmp_path / "critiques.csv"

        exporter.export_critiques(output_path)
        assert output_path.exists()


# =============================================================================
# CSVExporter Tests - Votes
# =============================================================================


class TestCSVExporterVotes:
    """Tests for CSVExporter.export_votes()."""

    def test_export_votes_header(self, minimal_artifact):
        """Test votes export includes correct header."""
        exporter = CSVExporter(minimal_artifact)
        content = exporter.export_votes()

        reader = csv.reader(StringIO(content))
        header = next(reader)
        assert header == ['debate_id', 'agent', 'agreed_with_consensus', 'final_answer', 'confidence']

    def test_export_votes_no_consensus(self, minimal_artifact):
        """Test export with no consensus returns only header."""
        exporter = CSVExporter(minimal_artifact)
        content = exporter.export_votes()

        lines = content.strip().split('\n')
        assert len(lines) == 1

    def test_export_votes_with_consensus(self, artifact_with_consensus):
        """Test export with consensus proof."""
        exporter = CSVExporter(artifact_with_consensus)
        content = exporter.export_votes()

        reader = csv.reader(StringIO(content))
        rows = list(reader)

        assert len(rows) == 4  # Header + 3 agents
        # Check one vote
        agents_found = {row[1]: row[2] for row in rows[1:]}
        assert agents_found["agent1"] == "True"
        assert agents_found["agent2"] == "True"
        assert agents_found["agent3"] == "False"

    def test_export_votes_truncates_final_answer(self):
        """Test export truncates final answer to 100 chars."""
        long_answer = "A" * 200
        artifact = DebateArtifact(
            debate_id="long-answer",
            consensus_proof=ConsensusProof(
                reached=True,
                confidence=0.9,
                vote_breakdown={"agent1": True},
                final_answer=long_answer,
                rounds_used=1,
            ),
        )
        exporter = CSVExporter(artifact)
        content = exporter.export_votes()

        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert len(rows[1][3]) == 100

    def test_export_votes_includes_confidence(self, artifact_with_consensus):
        """Test export includes confidence value."""
        exporter = CSVExporter(artifact_with_consensus)
        content = exporter.export_votes()

        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert rows[1][4] == "0.85"

    def test_export_votes_to_file(self, artifact_with_consensus, tmp_path):
        """Test export writes to file."""
        exporter = CSVExporter(artifact_with_consensus)
        output_path = tmp_path / "votes.csv"

        exporter.export_votes(output_path)
        assert output_path.exists()


# =============================================================================
# CSVExporter Tests - Summary
# =============================================================================


class TestCSVExporterSummary:
    """Tests for CSVExporter.export_summary()."""

    def test_export_summary_header(self, minimal_artifact):
        """Test summary export includes correct header."""
        exporter = CSVExporter(minimal_artifact)
        content = exporter.export_summary()

        reader = csv.reader(StringIO(content))
        header = next(reader)
        expected = [
            'debate_id', 'artifact_id', 'task', 'agents', 'rounds',
            'messages', 'critiques', 'consensus_reached', 'confidence',
            'duration_seconds', 'created_at',
        ]
        assert header == expected

    def test_export_summary_basic(self, minimal_artifact):
        """Test summary export with basic artifact."""
        exporter = CSVExporter(minimal_artifact)
        content = exporter.export_summary()

        reader = csv.reader(StringIO(content))
        rows = list(reader)

        assert len(rows) == 2
        assert rows[1][0] == "debate-123"  # debate_id
        assert rows[1][1] == "art-001"  # artifact_id
        assert rows[1][2] == "Test task"  # task
        assert "agent1" in rows[1][3]  # agents

    def test_export_summary_with_consensus(self, artifact_with_consensus):
        """Test summary includes consensus info."""
        exporter = CSVExporter(artifact_with_consensus)
        content = exporter.export_summary()

        reader = csv.reader(StringIO(content))
        rows = list(reader)

        assert rows[1][7] == "True"  # consensus_reached
        assert rows[1][8] == "0.85"  # confidence

    def test_export_summary_no_consensus(self, minimal_artifact):
        """Test summary handles missing consensus."""
        exporter = CSVExporter(minimal_artifact)
        content = exporter.export_summary()

        reader = csv.reader(StringIO(content))
        rows = list(reader)

        assert rows[1][7] == ""  # consensus_reached
        assert rows[1][8] == ""  # confidence

    def test_export_summary_truncates_task(self):
        """Test summary truncates task to 200 chars."""
        long_task = "T" * 500
        artifact = DebateArtifact(
            artifact_id="long",
            debate_id="long",
            task=long_task,
        )
        exporter = CSVExporter(artifact)
        content = exporter.export_summary()

        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert len(rows[1][2]) == 200

    def test_export_summary_agents_semicolon_separated(self):
        """Test agents are semicolon-separated."""
        artifact = DebateArtifact(
            debate_id="agents",
            agents=["agent1", "agent2", "agent3"],
        )
        exporter = CSVExporter(artifact)
        content = exporter.export_summary()

        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert rows[1][3] == "agent1;agent2;agent3"

    def test_export_summary_to_file(self, minimal_artifact, tmp_path):
        """Test export writes to file."""
        exporter = CSVExporter(minimal_artifact)
        output_path = tmp_path / "summary.csv"

        exporter.export_summary(output_path)
        assert output_path.exists()


# =============================================================================
# CSVExporter Tests - Verifications
# =============================================================================


class TestCSVExporterVerifications:
    """Tests for CSVExporter.export_verifications()."""

    def test_export_verifications_header(self, minimal_artifact):
        """Test verifications export includes correct header."""
        exporter = CSVExporter(minimal_artifact)
        content = exporter.export_verifications()

        reader = csv.reader(StringIO(content))
        header = next(reader)
        expected = [
            'debate_id', 'claim_id', 'claim_text', 'status', 'method',
            'duration_ms', 'has_proof', 'has_counterexample',
        ]
        assert header == expected

    def test_export_verifications_no_results(self, minimal_artifact):
        """Test export with no verification results."""
        exporter = CSVExporter(minimal_artifact)
        content = exporter.export_verifications()

        lines = content.strip().split('\n')
        assert len(lines) == 1  # Only header

    def test_export_verifications_with_results(self, artifact_with_verifications):
        """Test export with verification results."""
        exporter = CSVExporter(artifact_with_verifications)
        content = exporter.export_verifications()

        reader = csv.reader(StringIO(content))
        rows = list(reader)

        assert len(rows) == 4  # Header + 3 verifications

        # Check verified result
        assert rows[1][1] == "claim-1"
        assert rows[1][3] == "verified"
        assert rows[1][6] == "True"  # has_proof
        assert rows[1][7] == "False"  # has_counterexample

        # Check refuted result
        assert rows[2][3] == "refuted"
        assert rows[2][6] == "False"  # has_proof
        assert rows[2][7] == "True"  # has_counterexample

    def test_export_verifications_truncates_claim_text(self):
        """Test export truncates claim text to 200 chars."""
        long_claim = "C" * 500
        artifact = DebateArtifact(
            debate_id="long-claim",
            verification_results=[
                VerificationResult(
                    claim_id="c1",
                    claim_text=long_claim,
                    status="timeout",
                    method="z3",
                ),
            ],
        )
        exporter = CSVExporter(artifact)
        content = exporter.export_verifications()

        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert len(rows[1][2]) == 200

    def test_export_verifications_to_file(self, artifact_with_verifications, tmp_path):
        """Test export writes to file."""
        exporter = CSVExporter(artifact_with_verifications)
        output_path = tmp_path / "verifications.csv"

        exporter.export_verifications(output_path)
        assert output_path.exists()


# =============================================================================
# CSVExporter Tests - Export All
# =============================================================================


class TestCSVExporterAll:
    """Tests for CSVExporter.export_all()."""

    def test_export_all_creates_directory(self, minimal_artifact, tmp_path):
        """Test export_all creates output directory."""
        exporter = CSVExporter(minimal_artifact)
        output_dir = tmp_path / "output" / "nested"

        exporter.export_all(output_dir)

        assert output_dir.exists()

    def test_export_all_returns_paths(self, minimal_artifact, tmp_path):
        """Test export_all returns dict of paths."""
        exporter = CSVExporter(minimal_artifact)
        output_dir = tmp_path / "output"

        outputs = exporter.export_all(output_dir)

        assert "messages" in outputs
        assert "critiques" in outputs
        assert "votes" in outputs
        assert "summary" in outputs
        # No verifications since artifact has none
        assert "verifications" not in outputs

    def test_export_all_creates_files(self, minimal_artifact, tmp_path):
        """Test export_all creates all expected files."""
        exporter = CSVExporter(minimal_artifact)
        output_dir = tmp_path / "output"

        outputs = exporter.export_all(output_dir)

        for path in outputs.values():
            assert path.exists()

    def test_export_all_includes_verifications(self, artifact_with_verifications, tmp_path):
        """Test export_all includes verifications when present."""
        exporter = CSVExporter(artifact_with_verifications)
        output_dir = tmp_path / "output"

        outputs = exporter.export_all(output_dir)

        assert "verifications" in outputs
        assert outputs["verifications"].exists()

    def test_export_all_file_naming(self, minimal_artifact, tmp_path):
        """Test export_all uses artifact_id in filenames."""
        exporter = CSVExporter(minimal_artifact)
        output_dir = tmp_path / "output"

        outputs = exporter.export_all(output_dir)

        assert "art-001" in str(outputs["messages"])
        assert "_messages.csv" in str(outputs["messages"])

    def test_export_all_directory_exists(self, minimal_artifact, tmp_path):
        """Test export_all works with existing directory."""
        exporter = CSVExporter(minimal_artifact)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        outputs = exporter.export_all(output_dir)

        assert len(outputs) >= 4


# =============================================================================
# export_debate_to_csv Tests
# =============================================================================


class TestExportDebateToCSV:
    """Tests for export_debate_to_csv convenience function."""

    def test_export_summary(self, minimal_artifact, tmp_path):
        """Test export with table='summary'."""
        output_path = tmp_path / "summary.csv"

        content = export_debate_to_csv(minimal_artifact, output_path, table="summary")

        assert output_path.exists()
        assert "debate_id" in content

    def test_export_messages(self, artifact_with_trace, tmp_path):
        """Test export with table='messages'."""
        output_path = tmp_path / "messages.csv"

        content = export_debate_to_csv(artifact_with_trace, output_path, table="messages")

        assert "agent1" in content
        assert "This is my proposal" in content

    def test_export_critiques(self, artifact_with_trace, tmp_path):
        """Test export with table='critiques'."""
        output_path = tmp_path / "critiques.csv"

        content = export_debate_to_csv(artifact_with_trace, output_path, table="critiques")

        assert "Missing evidence" in content

    def test_export_votes(self, artifact_with_consensus, tmp_path):
        """Test export with table='votes'."""
        output_path = tmp_path / "votes.csv"

        content = export_debate_to_csv(artifact_with_consensus, output_path, table="votes")

        assert "agent1" in content
        assert "True" in content

    def test_export_verifications(self, artifact_with_verifications, tmp_path):
        """Test export with table='verifications'."""
        output_path = tmp_path / "verifications.csv"

        content = export_debate_to_csv(artifact_with_verifications, output_path, table="verifications")

        assert "claim-1" in content
        assert "verified" in content

    def test_export_invalid_table(self, minimal_artifact, tmp_path):
        """Test export raises error for invalid table."""
        output_path = tmp_path / "invalid.csv"

        with pytest.raises(ValueError) as exc_info:
            export_debate_to_csv(minimal_artifact, output_path, table="invalid")

        assert "Unknown table" in str(exc_info.value)


# =============================================================================
# export_multiple_debates Tests
# =============================================================================


class TestExportMultipleDebates:
    """Tests for export_multiple_debates function."""

    def test_export_multiple_summary(self, tmp_path):
        """Test exporting multiple debates as summary."""
        artifacts = [
            DebateArtifact(debate_id=f"debate-{i}", task=f"Task {i}")
            for i in range(3)
        ]
        output_path = tmp_path / "combined.csv"

        content = export_multiple_debates(artifacts, output_path, table="summary")

        reader = csv.reader(StringIO(content))
        rows = list(reader)

        assert len(rows) == 4  # Header + 3 debates
        assert rows[1][0] == "debate-0"
        assert rows[2][0] == "debate-1"
        assert rows[3][0] == "debate-2"

    def test_export_multiple_messages(self, tmp_path):
        """Test exporting multiple debates as messages."""
        artifacts = [
            DebateArtifact(
                debate_id=f"debate-{i}",
                trace_data={
                    "events": [
                        {"event_type": "message", "agent": f"agent-{i}", "content": f"Content {i}"}
                    ]
                },
            )
            for i in range(2)
        ]
        output_path = tmp_path / "combined.csv"

        content = export_multiple_debates(artifacts, output_path, table="messages")

        assert "agent-0" in content
        assert "agent-1" in content
        assert "Content 0" in content
        assert "Content 1" in content

    def test_export_multiple_critiques(self, tmp_path):
        """Test exporting multiple debates as critiques."""
        artifacts = [
            DebateArtifact(
                debate_id=f"debate-{i}",
                trace_data={
                    "events": [
                        {"event_type": "critique", "agent": f"critic-{i}"}
                    ]
                },
            )
            for i in range(2)
        ]
        output_path = tmp_path / "combined.csv"

        content = export_multiple_debates(artifacts, output_path, table="critiques")

        assert "critic-0" in content
        assert "critic-1" in content

    def test_export_multiple_votes(self, tmp_path):
        """Test exporting multiple debates as votes."""
        artifacts = [
            DebateArtifact(
                debate_id=f"debate-{i}",
                consensus_proof=ConsensusProof(
                    reached=True,
                    confidence=0.9,
                    vote_breakdown={f"agent-{i}": True},
                    final_answer=f"Answer {i}",
                    rounds_used=1,
                ),
            )
            for i in range(2)
        ]
        output_path = tmp_path / "combined.csv"

        content = export_multiple_debates(artifacts, output_path, table="votes")

        assert "agent-0" in content
        assert "agent-1" in content

    def test_export_multiple_invalid_table(self, tmp_path):
        """Test export raises error for invalid table."""
        artifacts = [DebateArtifact(debate_id="d1")]
        output_path = tmp_path / "combined.csv"

        with pytest.raises(ValueError):
            export_multiple_debates(artifacts, output_path, table="invalid")

    def test_export_multiple_single_artifact(self, tmp_path):
        """Test export works with single artifact."""
        artifacts = [DebateArtifact(debate_id="single", task="Single task")]
        output_path = tmp_path / "single.csv"

        content = export_multiple_debates(artifacts, output_path, table="summary")

        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert len(rows) == 2  # Header + 1 debate

    def test_export_multiple_writes_file(self, tmp_path):
        """Test export writes to file."""
        artifacts = [DebateArtifact(debate_id="d1"), DebateArtifact(debate_id="d2")]
        output_path = tmp_path / "output.csv"

        export_multiple_debates(artifacts, output_path)

        assert output_path.exists()


# =============================================================================
# Edge Cases
# =============================================================================


class TestCSVExporterEdgeCases:
    """Edge case tests for CSV export."""

    def test_empty_debate_id(self):
        """Test export handles empty debate_id."""
        artifact = DebateArtifact(debate_id="")
        exporter = CSVExporter(artifact)

        content = exporter.export_summary()
        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert rows[1][0] == ""

    def test_unicode_content(self):
        """Test export handles unicode content."""
        artifact = DebateArtifact(
            debate_id="unicode",
            task="日本語のタスク 🎯",
            trace_data={
                "events": [
                    {
                        "event_type": "message",
                        "content": "Émoji: 👍 and 日本語",
                    }
                ]
            },
        )
        exporter = CSVExporter(artifact)
        content = exporter.export_messages()

        assert "日本語" in content
        assert "👍" in content

    def test_very_long_content(self):
        """Test export handles very long content."""
        long_content = "X" * 10000
        artifact = DebateArtifact(
            debate_id="long",
            trace_data={
                "events": [
                    {"event_type": "message", "content": long_content}
                ]
            },
        )
        exporter = CSVExporter(artifact)
        content = exporter.export_messages()

        # Content should be present (CSV doesn't truncate messages)
        assert long_content in content

    def test_newlines_in_content(self):
        """Test export handles newlines in content."""
        artifact = DebateArtifact(
            debate_id="newlines",
            trace_data={
                "events": [
                    {
                        "event_type": "message",
                        "content": "Line 1\nLine 2\nLine 3",
                    }
                ]
            },
        )
        exporter = CSVExporter(artifact)
        content = exporter.export_messages()

        # CSV should properly quote content with newlines
        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert "Line 1\nLine 2\nLine 3" == rows[1][4]

    def test_empty_agents_list(self):
        """Test export handles empty agents list."""
        artifact = DebateArtifact(debate_id="no-agents", agents=[])
        exporter = CSVExporter(artifact)
        content = exporter.export_summary()

        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert rows[1][3] == ""  # Empty agents

    def test_trace_data_without_events_key(self):
        """Test export handles trace_data without 'events' key."""
        artifact = DebateArtifact(
            debate_id="no-events",
            trace_data={"other_key": []},
        )
        exporter = CSVExporter(artifact)
        content = exporter.export_messages()

        lines = content.strip().split('\n')
        assert len(lines) == 1  # Only header

    def test_multiple_quotes_in_content(self):
        """Test export properly escapes multiple quotes."""
        artifact = DebateArtifact(
            debate_id="quotes",
            trace_data={
                "events": [
                    {
                        "event_type": "message",
                        "content": 'He said "Hello" and she said "World"',
                    }
                ]
            },
        )
        exporter = CSVExporter(artifact)
        content = exporter.export_messages()

        reader = csv.reader(StringIO(content))
        rows = list(reader)
        assert 'He said "Hello" and she said "World"' == rows[1][4]
