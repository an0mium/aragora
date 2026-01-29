"""
Tests for aragora.export.csv_exporter module.

Tests cover:
- CSVExporter initialization and basic operations
- Message export functionality
- Critique export functionality
- Vote export functionality
- Summary export functionality
- Verification export functionality
- Batch export (export_all)
- Convenience functions (export_debate_to_csv, export_multiple_debates)
"""

from __future__ import annotations

import csv
import tempfile
from io import StringIO
from pathlib import Path

import pytest

from aragora.export.artifact import (
    ConsensusProof,
    DebateArtifact,
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
def basic_artifact() -> DebateArtifact:
    """Create a basic debate artifact for testing."""
    return DebateArtifact(
        artifact_id="test-artifact-001",
        debate_id="debate-001",
        task="Analyze the security of the API",
        agents=["claude", "gpt-4", "gemini"],
        rounds=3,
        message_count=12,
        critique_count=6,
        duration_seconds=45.5,
    )


@pytest.fixture
def artifact_with_trace() -> DebateArtifact:
    """Create an artifact with trace data containing messages and critiques."""
    return DebateArtifact(
        artifact_id="test-artifact-002",
        debate_id="debate-002",
        task="Review code quality",
        agents=["claude", "gpt-4"],
        rounds=2,
        trace_data={
            "events": [
                {
                    "event_type": "message",
                    "round": 1,
                    "agent": "claude",
                    "role": "proposer",
                    "content": "I propose we focus on input validation.",
                    "timestamp": "2024-01-15T10:00:00Z",
                },
                {
                    "event_type": "message",
                    "round": 1,
                    "agent": "gpt-4",
                    "role": "critic",
                    "content": "Good point, but we should also consider rate limiting.",
                    "timestamp": "2024-01-15T10:01:00Z",
                },
                {
                    "event_type": "critique",
                    "round": 1,
                    "agent": "gpt-4",
                    "target": "claude",
                    "severity": 0.6,
                    "issues": ["Missing edge cases", "No error handling"],
                    "accepted": True,
                },
                {
                    "event_type": "message",
                    "round": 2,
                    "agent": "claude",
                    "role": "reviser",
                    "content": "Incorporating feedback on edge cases.",
                    "timestamp": "2024-01-15T10:02:00Z",
                },
                {
                    "event_type": "critique",
                    "round": 2,
                    "agent": "claude",
                    "target": "gpt-4",
                    "severity": 0.3,
                    "issues": ["Minor style issue"],
                    "accepted": False,
                },
            ]
        },
    )


@pytest.fixture
def artifact_with_consensus() -> DebateArtifact:
    """Create an artifact with consensus proof."""
    return DebateArtifact(
        artifact_id="test-artifact-003",
        debate_id="debate-003",
        task="Decide on API design",
        agents=["claude", "gpt-4", "gemini"],
        rounds=3,
        consensus_proof=ConsensusProof(
            reached=True,
            confidence=0.85,
            vote_breakdown={
                "claude": True,
                "gpt-4": True,
                "gemini": False,
            },
            final_answer="Use REST API with GraphQL for complex queries",
            rounds_used=3,
        ),
    )


@pytest.fixture
def artifact_with_verifications() -> DebateArtifact:
    """Create an artifact with verification results."""
    return DebateArtifact(
        artifact_id="test-artifact-004",
        debate_id="debate-004",
        task="Verify security claims",
        verification_results=[
            VerificationResult(
                claim_id="claim-001",
                claim_text="All inputs are sanitized",
                status="verified",
                method="z3",
                proof_trace="(assert (forall (x) (sanitized x)))",
                duration_ms=150,
            ),
            VerificationResult(
                claim_id="claim-002",
                claim_text="No SQL injection possible",
                status="refuted",
                method="z3",
                counterexample="Input: '; DROP TABLE users; --'",
                duration_ms=200,
            ),
            VerificationResult(
                claim_id="claim-003",
                claim_text="Rate limiting prevents abuse",
                status="timeout",
                method="simulation",
                duration_ms=10000,
            ),
        ],
    )


# =============================================================================
# TestCSVExporterInit
# =============================================================================


class TestCSVExporterInit:
    """Tests for CSVExporter initialization."""

    def test_init_with_artifact(self, basic_artifact: DebateArtifact):
        """Should initialize with a DebateArtifact."""
        exporter = CSVExporter(basic_artifact)
        assert exporter.artifact is basic_artifact

    def test_stores_artifact_reference(self, basic_artifact: DebateArtifact):
        """Should store artifact reference for later use."""
        exporter = CSVExporter(basic_artifact)
        assert exporter.artifact.artifact_id == "test-artifact-001"


# =============================================================================
# TestCSVExporterMessages
# =============================================================================


class TestCSVExporterMessages:
    """Tests for CSVExporter.export_messages()."""

    def test_returns_csv_string(self, artifact_with_trace: DebateArtifact):
        """Should return a valid CSV string."""
        exporter = CSVExporter(artifact_with_trace)
        result = exporter.export_messages()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_header_row(self, artifact_with_trace: DebateArtifact):
        """Should include correct header row."""
        exporter = CSVExporter(artifact_with_trace)
        result = exporter.export_messages()

        reader = csv.reader(StringIO(result))
        header = next(reader)

        assert header == ["debate_id", "round", "agent", "role", "content", "timestamp"]

    def test_includes_message_data(self, artifact_with_trace: DebateArtifact):
        """Should include message data from trace events."""
        exporter = CSVExporter(artifact_with_trace)
        result = exporter.export_messages()

        reader = csv.reader(StringIO(result))
        rows = list(reader)

        # Header + 3 message events
        assert len(rows) == 4

        # Check first message
        assert rows[1][0] == "debate-002"  # debate_id
        assert rows[1][1] == "1"  # round
        assert rows[1][2] == "claude"  # agent
        assert rows[1][3] == "proposer"  # role

    def test_handles_empty_trace(self, basic_artifact: DebateArtifact):
        """Should handle artifact without trace data."""
        exporter = CSVExporter(basic_artifact)
        result = exporter.export_messages()

        reader = csv.reader(StringIO(result))
        rows = list(reader)

        # Only header row
        assert len(rows) == 1

    def test_writes_to_file(self, artifact_with_trace: DebateArtifact):
        """Should write CSV to file when path provided."""
        exporter = CSVExporter(artifact_with_trace)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "messages.csv"
            result = exporter.export_messages(output_path)

            assert output_path.exists()
            assert output_path.read_text() == result


# =============================================================================
# TestCSVExporterCritiques
# =============================================================================


class TestCSVExporterCritiques:
    """Tests for CSVExporter.export_critiques()."""

    def test_returns_csv_string(self, artifact_with_trace: DebateArtifact):
        """Should return a valid CSV string."""
        exporter = CSVExporter(artifact_with_trace)
        result = exporter.export_critiques()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_header_row(self, artifact_with_trace: DebateArtifact):
        """Should include correct header row."""
        exporter = CSVExporter(artifact_with_trace)
        result = exporter.export_critiques()

        reader = csv.reader(StringIO(result))
        header = next(reader)

        expected = [
            "debate_id",
            "round",
            "critic",
            "target",
            "severity",
            "issue_count",
            "issues",
            "accepted",
        ]
        assert header == expected

    def test_includes_critique_data(self, artifact_with_trace: DebateArtifact):
        """Should include critique data from trace events."""
        exporter = CSVExporter(artifact_with_trace)
        result = exporter.export_critiques()

        reader = csv.reader(StringIO(result))
        rows = list(reader)

        # Header + 2 critique events
        assert len(rows) == 3

        # Check first critique
        assert rows[1][2] == "gpt-4"  # critic
        assert rows[1][3] == "claude"  # target
        assert rows[1][4] == "0.6"  # severity
        assert rows[1][5] == "2"  # issue_count

    def test_formats_issues_list(self, artifact_with_trace: DebateArtifact):
        """Should format issues list as semicolon-separated string."""
        exporter = CSVExporter(artifact_with_trace)
        result = exporter.export_critiques()

        reader = csv.reader(StringIO(result))
        rows = list(reader)

        # Issues should be semicolon-separated
        assert "Missing edge cases; No error handling" in rows[1][6]


# =============================================================================
# TestCSVExporterVotes
# =============================================================================


class TestCSVExporterVotes:
    """Tests for CSVExporter.export_votes()."""

    def test_returns_csv_string(self, artifact_with_consensus: DebateArtifact):
        """Should return a valid CSV string."""
        exporter = CSVExporter(artifact_with_consensus)
        result = exporter.export_votes()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_header_row(self, artifact_with_consensus: DebateArtifact):
        """Should include correct header row."""
        exporter = CSVExporter(artifact_with_consensus)
        result = exporter.export_votes()

        reader = csv.reader(StringIO(result))
        header = next(reader)

        expected = [
            "debate_id",
            "agent",
            "agreed_with_consensus",
            "final_answer",
            "confidence",
        ]
        assert header == expected

    def test_includes_vote_breakdown(self, artifact_with_consensus: DebateArtifact):
        """Should include vote breakdown from consensus proof."""
        exporter = CSVExporter(artifact_with_consensus)
        result = exporter.export_votes()

        reader = csv.reader(StringIO(result))
        rows = list(reader)

        # Header + 3 agent votes
        assert len(rows) == 4

        # Find the gemini row (disagreed)
        gemini_row = [r for r in rows[1:] if r[1] == "gemini"][0]
        assert gemini_row[2] == "False"  # agreed_with_consensus

    def test_handles_no_consensus(self, basic_artifact: DebateArtifact):
        """Should handle artifact without consensus proof."""
        exporter = CSVExporter(basic_artifact)
        result = exporter.export_votes()

        reader = csv.reader(StringIO(result))
        rows = list(reader)

        # Only header row
        assert len(rows) == 1


# =============================================================================
# TestCSVExporterSummary
# =============================================================================


class TestCSVExporterSummary:
    """Tests for CSVExporter.export_summary()."""

    def test_returns_csv_string(self, basic_artifact: DebateArtifact):
        """Should return a valid CSV string."""
        exporter = CSVExporter(basic_artifact)
        result = exporter.export_summary()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_header_row(self, basic_artifact: DebateArtifact):
        """Should include correct header row."""
        exporter = CSVExporter(basic_artifact)
        result = exporter.export_summary()

        reader = csv.reader(StringIO(result))
        header = next(reader)

        expected = [
            "debate_id",
            "artifact_id",
            "task",
            "agents",
            "rounds",
            "messages",
            "critiques",
            "consensus_reached",
            "confidence",
            "duration_seconds",
            "created_at",
        ]
        assert header == expected

    def test_includes_debate_metadata(self, basic_artifact: DebateArtifact):
        """Should include debate metadata in summary."""
        exporter = CSVExporter(basic_artifact)
        result = exporter.export_summary()

        reader = csv.reader(StringIO(result))
        rows = list(reader)

        # Header + 1 summary row
        assert len(rows) == 2

        summary = rows[1]
        assert summary[0] == "debate-001"  # debate_id
        assert summary[1] == "test-artifact-001"  # artifact_id
        assert summary[4] == "3"  # rounds
        assert summary[5] == "12"  # messages
        assert summary[6] == "6"  # critiques

    def test_formats_agents_as_semicolon_list(self, basic_artifact: DebateArtifact):
        """Should format agents as semicolon-separated string."""
        exporter = CSVExporter(basic_artifact)
        result = exporter.export_summary()

        reader = csv.reader(StringIO(result))
        rows = list(reader)

        agents = rows[1][3]
        assert agents == "claude;gpt-4;gemini"


# =============================================================================
# TestCSVExporterVerifications
# =============================================================================


class TestCSVExporterVerifications:
    """Tests for CSVExporter.export_verifications()."""

    def test_returns_csv_string(self, artifact_with_verifications: DebateArtifact):
        """Should return a valid CSV string."""
        exporter = CSVExporter(artifact_with_verifications)
        result = exporter.export_verifications()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_includes_header_row(self, artifact_with_verifications: DebateArtifact):
        """Should include correct header row."""
        exporter = CSVExporter(artifact_with_verifications)
        result = exporter.export_verifications()

        reader = csv.reader(StringIO(result))
        header = next(reader)

        expected = [
            "debate_id",
            "claim_id",
            "claim_text",
            "status",
            "method",
            "duration_ms",
            "has_proof",
            "has_counterexample",
        ]
        assert header == expected

    def test_includes_verification_data(self, artifact_with_verifications: DebateArtifact):
        """Should include verification results."""
        exporter = CSVExporter(artifact_with_verifications)
        result = exporter.export_verifications()

        reader = csv.reader(StringIO(result))
        rows = list(reader)

        # Header + 3 verifications
        assert len(rows) == 4

        # Check verified claim
        verified = rows[1]
        assert verified[3] == "verified"  # status
        assert verified[6] == "True"  # has_proof

        # Check refuted claim
        refuted = rows[2]
        assert refuted[3] == "refuted"  # status
        assert refuted[7] == "True"  # has_counterexample

    def test_truncates_long_claim_text(self, artifact_with_verifications: DebateArtifact):
        """Should truncate claim text to 200 characters."""
        # Add a long claim
        artifact_with_verifications.verification_results.append(
            VerificationResult(
                claim_id="claim-long",
                claim_text="A" * 500,
                status="timeout",
                method="z3",
            )
        )

        exporter = CSVExporter(artifact_with_verifications)
        result = exporter.export_verifications()

        reader = csv.reader(StringIO(result))
        rows = list(reader)

        # Last row should have truncated claim
        assert len(rows[-1][2]) == 200


# =============================================================================
# TestCSVExporterExportAll
# =============================================================================


class TestCSVExporterExportAll:
    """Tests for CSVExporter.export_all()."""

    def test_creates_all_csv_files(self, artifact_with_trace: DebateArtifact):
        """Should create all CSV files in output directory."""
        # Add consensus proof for votes export
        artifact_with_trace.consensus_proof = ConsensusProof(
            reached=True,
            confidence=0.9,
            vote_breakdown={"claude": True, "gpt-4": True},
            final_answer="Agreed",
            rounds_used=2,
        )

        exporter = CSVExporter(artifact_with_trace)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            outputs = exporter.export_all(output_dir)

            assert "messages" in outputs
            assert "critiques" in outputs
            assert "votes" in outputs
            assert "summary" in outputs

            for path in outputs.values():
                assert path.exists()

    def test_includes_verifications_when_present(self, artifact_with_verifications: DebateArtifact):
        """Should include verifications file when verification results exist."""
        exporter = CSVExporter(artifact_with_verifications)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            outputs = exporter.export_all(output_dir)

            assert "verifications" in outputs
            assert outputs["verifications"].exists()

    def test_creates_output_directory(self, basic_artifact: DebateArtifact):
        """Should create output directory if it doesn't exist."""
        exporter = CSVExporter(basic_artifact)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "output"
            exporter.export_all(output_dir)

            assert output_dir.exists()


# =============================================================================
# TestExportDebateToCSV
# =============================================================================


class TestExportDebateToCSV:
    """Tests for export_debate_to_csv convenience function."""

    def test_exports_summary_by_default(self, basic_artifact: DebateArtifact):
        """Should export summary table by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.csv"
            result = export_debate_to_csv(basic_artifact, output_path)

            assert "debate_id" in result
            assert "artifact_id" in result
            assert output_path.exists()

    def test_exports_messages_table(self, artifact_with_trace: DebateArtifact):
        """Should export messages when table='messages'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "messages.csv"
            result = export_debate_to_csv(artifact_with_trace, output_path, table="messages")

            assert "role" in result
            assert "content" in result

    def test_exports_critiques_table(self, artifact_with_trace: DebateArtifact):
        """Should export critiques when table='critiques'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "critiques.csv"
            result = export_debate_to_csv(artifact_with_trace, output_path, table="critiques")

            assert "critic" in result
            assert "severity" in result

    def test_exports_votes_table(self, artifact_with_consensus: DebateArtifact):
        """Should export votes when table='votes'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "votes.csv"
            result = export_debate_to_csv(artifact_with_consensus, output_path, table="votes")

            assert "agreed_with_consensus" in result

    def test_exports_verifications_table(self, artifact_with_verifications: DebateArtifact):
        """Should export verifications when table='verifications'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "verifications.csv"
            result = export_debate_to_csv(
                artifact_with_verifications, output_path, table="verifications"
            )

            assert "claim_id" in result
            assert "has_proof" in result

    def test_raises_for_unknown_table(self, basic_artifact: DebateArtifact):
        """Should raise ValueError for unknown table type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.csv"

            with pytest.raises(ValueError, match="Unknown table"):
                export_debate_to_csv(basic_artifact, output_path, table="unknown")


# =============================================================================
# TestExportMultipleDebates
# =============================================================================


class TestExportMultipleDebates:
    """Tests for export_multiple_debates convenience function."""

    def test_exports_multiple_summaries(self):
        """Should combine multiple debate summaries into one CSV."""
        artifacts = [
            DebateArtifact(
                artifact_id=f"artifact-{i}",
                debate_id=f"debate-{i}",
                task=f"Task {i}",
                rounds=i + 1,
            )
            for i in range(3)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "combined.csv"
            result = export_multiple_debates(artifacts, output_path)

            reader = csv.reader(StringIO(result))
            rows = list(reader)

            # Header + 3 summary rows
            assert len(rows) == 4

    def test_exports_multiple_messages(self):
        """Should combine multiple debates' messages into one CSV."""
        artifacts = [
            DebateArtifact(
                artifact_id=f"artifact-{i}",
                debate_id=f"debate-{i}",
                trace_data={
                    "events": [
                        {
                            "event_type": "message",
                            "round": 1,
                            "agent": "claude",
                            "role": "proposer",
                            "content": f"Message from debate {i}",
                            "timestamp": "2024-01-15T10:00:00Z",
                        }
                    ]
                },
            )
            for i in range(2)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "messages.csv"
            result = export_multiple_debates(artifacts, output_path, table="messages")

            reader = csv.reader(StringIO(result))
            rows = list(reader)

            # Header + 2 message rows
            assert len(rows) == 3

    def test_raises_for_unknown_table(self):
        """Should raise ValueError for unknown table type."""
        artifacts = [DebateArtifact()]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "export.csv"

            with pytest.raises(ValueError, match="Unknown table"):
                export_multiple_debates(artifacts, output_path, table="unknown")


# =============================================================================
# TestCSVExporterEdgeCases
# =============================================================================


class TestCSVExporterEdgeCases:
    """Edge case tests for CSV exporter."""

    def test_handles_special_characters_in_content(self):
        """Should properly escape special characters in CSV."""
        artifact = DebateArtifact(
            artifact_id="test",
            debate_id="debate",
            trace_data={
                "events": [
                    {
                        "event_type": "message",
                        "round": 1,
                        "agent": "claude",
                        "role": "proposer",
                        "content": 'Content with "quotes", commas, and\nnewlines',
                        "timestamp": "2024-01-15T10:00:00Z",
                    }
                ]
            },
        )

        exporter = CSVExporter(artifact)
        result = exporter.export_messages()

        # Should be valid CSV that can be parsed
        reader = csv.reader(StringIO(result))
        rows = list(reader)
        assert len(rows) == 2
        assert "quotes" in rows[1][4]

    def test_handles_unicode_content(self):
        """Should handle unicode content properly."""
        artifact = DebateArtifact(
            artifact_id="test",
            debate_id="debate",
            trace_data={
                "events": [
                    {
                        "event_type": "message",
                        "round": 1,
                        "agent": "claude",
                        "role": "proposer",
                        "content": "Unicode test: International text",
                        "timestamp": "2024-01-15T10:00:00Z",
                    }
                ]
            },
        )

        exporter = CSVExporter(artifact)
        result = exporter.export_messages()

        assert "International" in result

    def test_handles_empty_trace_events(self):
        """Should handle empty events list in trace data."""
        artifact = DebateArtifact(
            artifact_id="test",
            debate_id="debate",
            trace_data={"events": []},
        )

        exporter = CSVExporter(artifact)
        result = exporter.export_messages()

        reader = csv.reader(StringIO(result))
        rows = list(reader)
        assert len(rows) == 1  # Only header

    def test_handles_missing_event_fields(self):
        """Should handle events with missing optional fields."""
        artifact = DebateArtifact(
            artifact_id="test",
            debate_id="debate",
            trace_data={
                "events": [
                    {
                        "event_type": "message",
                        # Missing round, agent, role, content, timestamp
                    }
                ]
            },
        )

        exporter = CSVExporter(artifact)
        result = exporter.export_messages()

        reader = csv.reader(StringIO(result))
        rows = list(reader)
        assert len(rows) == 2  # Header + one row with defaults
