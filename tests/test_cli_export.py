"""Tests for CLI export command - debate artifact exports."""

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.cli.export import (
    create_demo_artifact,
    export_to_html,
    export_to_json,
    export_to_markdown,
    load_artifact_from_debate,
    main,
)


@pytest.fixture
def mock_args():
    """Create mock args object."""
    args = MagicMock()
    args.output = "./exports"
    args.format = "html"
    args.demo = False
    args.debate_id = None
    return args


@pytest.fixture
def mock_artifact():
    """Create mock debate artifact."""
    artifact = MagicMock()
    artifact.artifact_id = "artifact-123"
    artifact.task = "Design a rate limiter"
    artifact.content_hash = "abc123"
    artifact.rounds = 3
    artifact.duration_seconds = 45.3
    artifact.consensus_proof = MagicMock()
    artifact.consensus_proof.reached = True
    artifact.consensus_proof.confidence = 0.85
    artifact.consensus_proof.final_answer = "Use token bucket algorithm"
    return artifact


class TestCreateDemoArtifact:
    """Tests for create_demo_artifact function."""

    @patch("aragora.export.artifact.ArtifactBuilder")
    @patch("aragora.core.DebateResult")
    @patch("aragora.core.Message")
    @patch("aragora.core.Critique")
    def test_creates_artifact(self, mock_critique, mock_message, mock_result, mock_builder):
        """Create a demo artifact."""
        mock_artifact = MagicMock()
        mock_builder_instance = MagicMock()
        mock_builder_instance.from_result.return_value = mock_builder_instance
        mock_builder_instance.with_verification.return_value = mock_builder_instance
        mock_builder_instance.build.return_value = mock_artifact
        mock_builder.return_value = mock_builder_instance

        result = create_demo_artifact()

        assert result == mock_artifact
        mock_builder_instance.from_result.assert_called_once()
        mock_builder_instance.build.assert_called_once()

    @patch("aragora.export.artifact.ArtifactBuilder")
    @patch("aragora.core.DebateResult")
    @patch("aragora.core.Message")
    @patch("aragora.core.Critique")
    def test_includes_verification(self, mock_critique, mock_message, mock_result, mock_builder):
        """Include verification in demo artifact."""
        mock_builder_instance = MagicMock()
        mock_builder_instance.from_result.return_value = mock_builder_instance
        mock_builder_instance.with_verification.return_value = mock_builder_instance
        mock_builder_instance.build.return_value = MagicMock()
        mock_builder.return_value = mock_builder_instance

        create_demo_artifact()

        mock_builder_instance.with_verification.assert_called_once()
        call_args = mock_builder_instance.with_verification.call_args
        assert call_args[0][0] == "claim-1"
        assert call_args[0][1] == "Token bucket is O(1)"


class TestLoadArtifactFromDebate:
    """Tests for load_artifact_from_debate function."""

    @patch("aragora.debate.traces.DebateReplayer")
    @patch("aragora.export.artifact.DebateArtifact")
    @patch("aragora.export.artifact.ConsensusProof")
    def test_loads_artifact(self, mock_proof, mock_artifact_cls, mock_replayer_cls):
        """Load artifact from debate trace."""
        mock_trace = MagicMock()
        mock_trace.debate_id = "debate-123"
        mock_trace.task = "Test task"
        mock_trace.agents = ["claude", "gpt4"]
        mock_trace.events = []
        mock_trace.duration_ms = 45000
        mock_trace.final_result = {
            "consensus_reached": True,
            "confidence": 0.85,
            "final_answer": "Answer",
            "rounds_used": 3,
        }

        mock_replayer = MagicMock()
        mock_replayer.trace = mock_trace
        mock_replayer_cls.from_database.return_value = mock_replayer

        mock_artifact = MagicMock()
        mock_artifact_cls.return_value = mock_artifact

        result = load_artifact_from_debate("debate-123")

        assert result == mock_artifact
        mock_replayer_cls.from_database.assert_called_once()

    @patch("aragora.debate.traces.DebateReplayer")
    @patch("aragora.export.artifact.DebateArtifact")
    def test_handles_no_final_result(self, mock_artifact_cls, mock_replayer_cls):
        """Handle missing final result."""
        mock_trace = MagicMock()
        mock_trace.debate_id = "debate-123"
        mock_trace.task = "Test task"
        mock_trace.agents = ["claude"]
        mock_trace.events = []
        mock_trace.duration_ms = None
        mock_trace.final_result = None

        mock_replayer = MagicMock()
        mock_replayer.trace = mock_trace
        mock_replayer_cls.from_database.return_value = mock_replayer

        mock_artifact = MagicMock()
        mock_artifact.consensus_proof = None
        mock_artifact_cls.return_value = mock_artifact

        result = load_artifact_from_debate("debate-123")

        # Should not raise, just have no consensus proof
        assert result == mock_artifact


class TestExportToHtml:
    """Tests for export_to_html function."""

    @patch("aragora.export.static_html.StaticHTMLExporter")
    def test_exports_html(self, mock_exporter_cls, mock_artifact, tmp_path):
        """Export artifact to HTML."""
        mock_exporter = MagicMock()
        mock_exporter_cls.return_value = mock_exporter

        result = export_to_html(mock_artifact, tmp_path)

        assert result == tmp_path / "debate_artifact-123.html"
        mock_exporter.save.assert_called_once()

    @patch("aragora.export.static_html.StaticHTMLExporter")
    def test_uses_artifact_id_in_filename(self, mock_exporter_cls, mock_artifact, tmp_path):
        """Use artifact ID in filename."""
        mock_artifact.artifact_id = "custom-id-456"
        mock_exporter_cls.return_value = MagicMock()

        result = export_to_html(mock_artifact, tmp_path)

        assert "custom-id-456" in str(result)


class TestExportToJson:
    """Tests for export_to_json function."""

    def test_exports_json(self, mock_artifact, tmp_path):
        """Export artifact to JSON."""
        result = export_to_json(mock_artifact, tmp_path)

        assert result == tmp_path / "debate_artifact-123.json"
        mock_artifact.save.assert_called_once_with(result)


class TestExportToMarkdown:
    """Tests for export_to_markdown function."""

    @patch("aragora.cli.publish.generate_markdown_report")
    @patch("aragora.core.DebateResult")
    def test_exports_markdown(self, mock_result_cls, mock_generate_md, mock_artifact, tmp_path):
        """Export artifact to Markdown."""
        mock_generate_md.return_value = "# Debate Report\n\nContent here."

        result = export_to_markdown(mock_artifact, tmp_path)

        assert result == tmp_path / "debate_artifact-123.md"
        assert result.exists()
        content = result.read_text()
        assert "# Debate Report" in content


class TestMain:
    """Tests for main function."""

    @patch("aragora.cli.export.create_demo_artifact")
    @patch("aragora.cli.export.export_to_html")
    def test_demo_mode(
        self, mock_export_html, mock_create_demo, mock_args, mock_artifact, tmp_path, capsys
    ):
        """Use demo mode."""
        mock_args.demo = True
        mock_args.output = str(tmp_path)
        mock_args.format = "html"
        mock_create_demo.return_value = mock_artifact
        mock_export_html.return_value = tmp_path / "debate.html"

        main(mock_args)

        mock_create_demo.assert_called_once()
        captured = capsys.readouterr()
        assert "HTML export saved" in captured.out

    @patch("aragora.cli.export.load_artifact_from_debate")
    @patch("aragora.cli.export.export_to_json")
    def test_loads_debate(
        self, mock_export_json, mock_load, mock_args, mock_artifact, tmp_path, capsys
    ):
        """Load artifact from debate ID."""
        mock_args.debate_id = "debate-123"
        mock_args.output = str(tmp_path)
        mock_args.format = "json"
        mock_args.db = None  # Explicitly set db to None
        mock_load.return_value = mock_artifact
        mock_export_json.return_value = tmp_path / "debate.json"

        main(mock_args)

        mock_load.assert_called_once_with("debate-123", None)
        captured = capsys.readouterr()
        assert "JSON export saved" in captured.out

    @patch("aragora.cli.export.load_artifact_from_debate")
    @patch("aragora.cli.export.export_to_markdown")
    def test_markdown_format(
        self, mock_export_md, mock_load, mock_args, mock_artifact, tmp_path, capsys
    ):
        """Export to Markdown format."""
        mock_args.debate_id = "debate-123"
        mock_args.output = str(tmp_path)
        mock_args.format = "md"
        mock_load.return_value = mock_artifact
        mock_export_md.return_value = tmp_path / "debate.md"

        main(mock_args)

        mock_export_md.assert_called_once()
        captured = capsys.readouterr()
        assert "Markdown export saved" in captured.out

    def test_no_source_specified(self, mock_args, tmp_path, capsys):
        """Show error when no source specified."""
        mock_args.demo = False
        mock_args.debate_id = None
        mock_args.output = str(tmp_path)

        main(mock_args)

        captured = capsys.readouterr()
        assert "Please provide a debate ID" in captured.out

    @patch("aragora.cli.export.load_artifact_from_debate")
    def test_load_error(self, mock_load, mock_args, tmp_path, capsys):
        """Handle load error gracefully."""
        mock_args.debate_id = "nonexistent"
        mock_args.output = str(tmp_path)
        mock_load.side_effect = Exception("Debate not found")

        main(mock_args)

        captured = capsys.readouterr()
        assert "Error loading debate" in captured.out
        assert "--demo" in captured.out

    def test_unknown_format(self, mock_args, tmp_path, capsys):
        """Handle unknown format."""
        mock_args.demo = True
        mock_args.output = str(tmp_path)
        mock_args.format = "unknown"

        with patch("aragora.cli.export.create_demo_artifact") as mock_create:
            mock_create.return_value = MagicMock()
            main(mock_args)

        captured = capsys.readouterr()
        assert "Unknown format" in captured.out

    @patch("aragora.cli.export.create_demo_artifact")
    @patch("aragora.cli.export.export_to_html")
    def test_prints_artifact_info(
        self, mock_export, mock_create, mock_args, mock_artifact, tmp_path, capsys
    ):
        """Print artifact info after export."""
        mock_args.demo = True
        mock_args.output = str(tmp_path)
        mock_args.format = "html"
        mock_create.return_value = mock_artifact
        mock_export.return_value = tmp_path / "debate.html"

        main(mock_args)

        captured = capsys.readouterr()
        assert "Artifact ID: artifact-123" in captured.out
        assert "Content Hash: abc123" in captured.out

    @patch("aragora.cli.export.create_demo_artifact")
    @patch("aragora.cli.export.export_to_html")
    def test_creates_output_directory(
        self, mock_export, mock_create, mock_args, mock_artifact, tmp_path, capsys
    ):
        """Create output directory if it doesn't exist."""
        output_dir = tmp_path / "nested" / "output"
        mock_args.demo = True
        mock_args.output = str(output_dir)
        mock_args.format = "html"
        mock_create.return_value = mock_artifact
        mock_export.return_value = output_dir / "debate.html"

        main(mock_args)

        assert output_dir.exists()


class TestMainAsScript:
    """Tests for running as script."""

    def test_has_main_block(self):
        """Module can be run as script."""
        # Just verify the module structure is correct
        from aragora.cli import export

        assert hasattr(export, "main")
        assert callable(export.main)
