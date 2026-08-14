"""Tests for CLI init command - project scaffolding."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aragora.cli.init import (
    DEFAULT_CONFIG,
    GITIGNORE_CONTENT,
    cmd_init,
    init_project,
)


@pytest.fixture
def clean_dir(tmp_path):
    """Create a clean directory for testing."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    return project_dir


@pytest.fixture
def existing_gitignore(clean_dir):
    """Create directory with existing .gitignore."""
    gitignore = clean_dir / ".gitignore"
    gitignore.write_text("# Existing content\n*.pyc\n")
    return gitignore


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG constant."""

    def test_is_valid_yaml(self):
        """DEFAULT_CONFIG is valid YAML."""
        import yaml

        config = yaml.safe_load(DEFAULT_CONFIG)
        assert isinstance(config, dict)

    def test_contains_agents_section(self):
        """Config contains agents section."""
        import yaml

        config = yaml.safe_load(DEFAULT_CONFIG)
        assert "agents" in config
        assert isinstance(config["agents"], list)

    def test_contains_debate_section(self):
        """Config contains debate section."""
        import yaml

        config = yaml.safe_load(DEFAULT_CONFIG)
        assert "debate" in config
        assert "rounds" in config["debate"]
        assert "consensus" in config["debate"]

    def test_contains_server_section(self):
        """Config contains server section."""
        import yaml

        config = yaml.safe_load(DEFAULT_CONFIG)
        assert "server" in config
        assert "http_port" in config["server"]


class TestGitignoreContent:
    """Tests for GITIGNORE_CONTENT constant."""

    def test_includes_aragora_dir(self):
        """Gitignore includes .aragora directory."""
        assert ".aragora/" in GITIGNORE_CONTENT
        assert ".nomic/" in GITIGNORE_CONTENT

    def test_includes_db_files(self):
        """Gitignore includes database files."""
        assert "*.db" in GITIGNORE_CONTENT
        assert "*.db-journal" in GITIGNORE_CONTENT

    def test_includes_env_files(self):
        """Gitignore includes env files."""
        assert ".env" in GITIGNORE_CONTENT

    def test_includes_pycache(self):
        """Gitignore includes pycache."""
        assert "__pycache__/" in GITIGNORE_CONTENT


class TestInitProject:
    """Tests for init_project function."""

    def test_creates_data_directory(self, clean_dir):
        """Create .aragora data directory."""
        result = init_project(str(clean_dir))

        data_dir = clean_dir / ".aragora"
        assert data_dir.exists()
        assert str(data_dir) in result["directories"]

    def test_creates_config_file(self, clean_dir):
        """Create .aragora.yaml config file."""
        result = init_project(str(clean_dir))

        config_file = clean_dir / ".aragora.yaml"
        assert config_file.exists()
        assert str(config_file) in result["files"]

        # Verify content
        content = config_file.read_text()
        assert "agents:" in content
        assert "debate:" in content

    def test_creates_traces_directory(self, clean_dir):
        """Create traces subdirectory."""
        result = init_project(str(clean_dir))

        traces_dir = clean_dir / ".aragora" / "traces"
        assert traces_dir.exists()
        assert str(traces_dir) in result["directories"]

    def test_creates_gitignore(self, clean_dir):
        """Create .gitignore file."""
        result = init_project(str(clean_dir))

        gitignore = clean_dir / ".gitignore"
        assert gitignore.exists()
        assert str(gitignore) in result["files"]

    def test_updates_existing_gitignore(self, clean_dir, existing_gitignore):
        """Update existing .gitignore."""
        result = init_project(str(clean_dir))

        content = existing_gitignore.read_text()
        assert "# Existing content" in content
        assert ".aragora/" in content
        # Check that gitignore was updated - it may or may not be first in list
        gitignore_updated = any("gitignore" in f for f in result["files"])
        assert gitignore_updated

    def test_skips_gitignore_if_already_has_aragora(self, clean_dir):
        """Skip gitignore update if already has .aragora entry."""
        gitignore = clean_dir / ".gitignore"
        gitignore.write_text("# Existing\n.aragora/\n")

        result = init_project(str(clean_dir))

        # Should not add duplicate
        content = gitignore.read_text()
        assert content.count(".aragora/") == 1

    def test_adds_secret_entries_when_gitignore_already_has_aragora(self, clean_dir):
        """Adds .env entries even when .gitignore already contains '.aragora/'.

        Regression: a single substring test on '.aragora/' caused the entire
        augmentation block (including .env / .env.local secret ignores) to be
        skipped, risking accidental credential commits.
        """
        gitignore = clean_dir / ".gitignore"
        gitignore.write_text(".aragora/\nnode_modules/\n")

        result = init_project(str(clean_dir))

        content = gitignore.read_text()
        # Secret-ignore entries must be present.
        env_lines = [line.strip() for line in content.splitlines()]
        assert ".env" in env_lines
        assert ".env.local" in env_lines
        # Pre-existing entry must not be duplicated.
        assert content.count(".aragora/") == 1
        assert any("gitignore" in f for f in result["files"])

    def test_adds_secret_entries_when_aragora_only_in_comment(self, clean_dir):
        """A comment mentioning '.aragora/' must not suppress real entries."""
        gitignore = clean_dir / ".gitignore"
        gitignore.write_text("# note: ignores .aragora/ data dir\n")

        init_project(str(clean_dir))

        env_lines = [line.strip() for line in gitignore.read_text().splitlines()]
        assert ".env" in env_lines
        assert ".aragora/" in env_lines

    def test_gitignore_update_is_idempotent(self, clean_dir):
        """Running init twice does not duplicate gitignore entries."""
        gitignore = clean_dir / ".gitignore"
        gitignore.write_text(".aragora/\n")

        init_project(str(clean_dir))
        init_project(str(clean_dir))

        content = gitignore.read_text()
        assert content.count("\n.env\n") <= 1
        assert content.count(".env.local") == 1
        assert content.count(".aragora/") == 1

    def test_no_gitignore_when_disabled(self, clean_dir):
        """Skip gitignore when with_git=False."""
        result = init_project(str(clean_dir), with_git=False)

        gitignore = clean_dir / ".gitignore"
        assert not gitignore.exists()

    def test_force_overwrites_config(self, clean_dir):
        """Force overwrites existing config."""
        config_file = clean_dir / ".aragora.yaml"
        config_file.write_text("# Old config\n")

        init_project(str(clean_dir), force=True)

        content = config_file.read_text()
        assert "# Aragora Configuration" in content

    def test_skips_existing_config_without_force(self, clean_dir):
        """Skip existing config without force."""
        config_file = clean_dir / ".aragora.yaml"
        config_file.write_text("# Old config\n")

        init_project(str(clean_dir), force=False)

        content = config_file.read_text()
        assert "# Old config" in content

    def test_review_preset_skipped_warns_when_config_exists(self, clean_dir):
        """Re-running with --preset review on an existing config warns the user.

        Regression: the preset was silently dropped (config writing is gated by
        not-exists-or-force) while init still reported success.
        """
        # First run writes the default config.
        init_project(str(clean_dir))

        # Second run requests the review preset without --force.
        result = init_project(str(clean_dir), preset="review")

        config = (clean_dir / ".aragora.yaml").read_text()
        # Preset NOT applied (no --force) ...
        assert "Code Review Preset" not in config
        # ... but the user is warned about it.
        assert result.get("warnings")
        assert any("review" in w and "--force" in w for w in result["warnings"])

    def test_review_preset_applied_with_force(self, clean_dir):
        """--force lets --preset review overwrite an existing config."""
        init_project(str(clean_dir))

        result = init_project(str(clean_dir), preset="review", force=True)

        config = (clean_dir / ".aragora.yaml").read_text()
        assert "Code Review Preset" in config
        assert "fail_on_critical: true" in config
        assert not result.get("warnings")

    def test_review_preset_fresh_no_warning(self, clean_dir):
        """Fresh init with --preset review applies it with no warning."""
        result = init_project(str(clean_dir), preset="review")

        config = (clean_dir / ".aragora.yaml").read_text()
        assert "Code Review Preset" in config
        assert not result.get("warnings")

    def test_uses_current_dir_by_default(self, clean_dir, monkeypatch):
        """Use current directory when none specified."""
        monkeypatch.chdir(clean_dir)

        result = init_project()

        assert (clean_dir / ".aragora").exists()
        assert (clean_dir / ".aragora.yaml").exists()

    def test_skips_existing_directories(self, clean_dir):
        """Skip creating existing directories."""
        (clean_dir / ".aragora").mkdir()
        (clean_dir / ".aragora" / "traces").mkdir()

        result = init_project(str(clean_dir))

        # Should not be in created directories
        data_dir_str = str(clean_dir / ".aragora")
        assert data_dir_str not in result["directories"]


class TestCmdInit:
    """Tests for cmd_init function."""

    def test_prints_created_directories(self, clean_dir, capsys):
        """Print created directories."""
        args = MagicMock()
        args.directory = str(clean_dir)
        args.force = False
        args.no_git = False

        cmd_init(args)

        captured = capsys.readouterr()
        assert "Created directories" in captured.out
        assert ".aragora" in captured.out

    def test_prints_created_files(self, clean_dir, capsys):
        """Print created files."""
        args = MagicMock()
        args.directory = str(clean_dir)
        args.force = False
        args.no_git = False

        cmd_init(args)

        captured = capsys.readouterr()
        assert "Created files" in captured.out
        assert ".aragora.yaml" in captured.out

    def test_prints_success_message(self, clean_dir, capsys):
        """Print success message."""
        args = MagicMock()
        args.directory = str(clean_dir)
        args.force = False
        args.no_git = False

        cmd_init(args)

        captured = capsys.readouterr()
        assert "Aragora project initialized" in captured.out

    def test_prints_next_steps(self, clean_dir, capsys, monkeypatch):
        """Print next steps including API key prompt when no keys set."""
        # Clear API keys so the 'no keys detected' path runs
        for key in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "XAI_API_KEY",
            "MISTRAL_API_KEY",
            "OPENROUTER_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

        args = MagicMock()
        args.directory = str(clean_dir)
        args.force = False
        args.no_git = False

        cmd_init(args)

        captured = capsys.readouterr()
        assert "GOLDEN PATH" in captured.out
        assert "No API keys detected" in captured.out
        assert "aragora" in captured.out

    def test_handles_missing_directory_attr(self, clean_dir, capsys, monkeypatch):
        """Handle missing directory attribute."""
        monkeypatch.chdir(clean_dir)

        args = MagicMock(spec=[])  # No attributes

        cmd_init(args)

        # Should succeed using cwd
        assert (clean_dir / ".aragora").exists()

    def test_handles_missing_force_attr(self, clean_dir, capsys):
        """Handle missing force attribute."""
        args = MagicMock()
        args.directory = str(clean_dir)
        del args.force
        args.no_git = False

        cmd_init(args)

        # Should succeed with default force=False
        captured = capsys.readouterr()
        assert "initialized" in captured.out

    def test_prints_warning_when_preset_skipped(self, clean_dir, capsys):
        """cmd_init surfaces the skipped-preset warning in its output."""
        # First initialize with defaults.
        first = MagicMock()
        first.directory = str(clean_dir)
        first.force = False
        first.no_git = False
        first.preset = None
        cmd_init(first)
        capsys.readouterr()

        # Re-run requesting the review preset without --force.
        second = MagicMock()
        second.directory = str(clean_dir)
        second.force = False
        second.no_git = False
        second.preset = "review"
        cmd_init(second)

        captured = capsys.readouterr()
        assert "Warnings:" in captured.out
        assert "--force" in captured.out

    def test_handles_no_git_flag(self, clean_dir, capsys):
        """Handle no_git flag."""
        args = MagicMock()
        args.directory = str(clean_dir)
        args.force = False
        args.no_git = True

        cmd_init(args)

        # Should not create .gitignore
        assert not (clean_dir / ".gitignore").exists()


class TestEndToEnd:
    """End-to-end tests for init command."""

    def test_full_initialization(self, clean_dir):
        """Test full project initialization."""
        result = init_project(str(clean_dir))

        # Verify all expected files/dirs exist
        assert (clean_dir / ".aragora").exists()
        assert (clean_dir / ".aragora" / "traces").exists()
        assert (clean_dir / ".aragora.yaml").exists()
        assert (clean_dir / ".gitignore").exists()

        # Verify config is valid YAML
        import yaml

        config = yaml.safe_load((clean_dir / ".aragora.yaml").read_text())
        assert config["debate"]["rounds"] == 3

        # Verify gitignore has aragora entries
        gitignore = (clean_dir / ".gitignore").read_text()
        assert ".aragora/" in gitignore

    def test_idempotent_initialization(self, clean_dir):
        """Running init twice is safe."""
        # First run
        init_project(str(clean_dir))

        # Second run should not fail
        result = init_project(str(clean_dir))

        # Should report fewer created items
        assert len(result["directories"]) == 0 or all(
            "traces" not in d for d in result["directories"]
        )
