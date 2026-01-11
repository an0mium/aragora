"""
Aragora init command - Project scaffolding.

Creates a new Aragora project with configuration files and directory structure.
"""

import os
from pathlib import Path
from typing import Optional

DEFAULT_CONFIG = """\
# Aragora Configuration
# See https://github.com/aragora/aragora for documentation

# Default agents for debates
agents:
  - anthropic-api
  - openai-api

# Debate settings
debate:
  rounds: 3
  consensus: majority
  enable_memory: true

# Server settings
server:
  http_port: 8080
  ws_port: 8765

# Data directories
data:
  db_dir: .aragora
  memory_db: agora_memory.db

# Optional API keys (or set as environment variables)
# api_keys:
#   ANTHROPIC_API_KEY: ""
#   OPENAI_API_KEY: ""
"""

GITIGNORE_CONTENT = """\
# Aragora data
.aragora/
*.db
*.db-journal
*.db-wal
*.db-shm

# Environment
.env
.env.local

# Python
__pycache__/
*.pyc
.venv/
"""


def init_project(
    directory: Optional[str] = None,
    force: bool = False,
    with_git: bool = True,
) -> dict:
    """Initialize a new Aragora project.

    Args:
        directory: Target directory (default: current directory)
        force: Overwrite existing files
        with_git: Add .gitignore entries

    Returns:
        Dict with created files and directories
    """
    target = Path(directory) if directory else Path.cwd()
    created: dict[str, list[str]] = {"files": [], "directories": []}

    # Create data directory
    data_dir = target / ".aragora"
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        created["directories"].append(str(data_dir))

    # Create config file
    config_file = target / ".aragora.yaml"
    if not config_file.exists() or force:
        config_file.write_text(DEFAULT_CONFIG)
        created["files"].append(str(config_file))

    # Create/update .gitignore
    if with_git:
        gitignore = target / ".gitignore"
        if gitignore.exists():
            existing = gitignore.read_text()
            if ".aragora/" not in existing:
                with gitignore.open("a") as f:
                    f.write("\n# Aragora\n")
                    f.write(GITIGNORE_CONTENT)
                created["files"].append(str(gitignore) + " (updated)")
        else:
            gitignore.write_text(GITIGNORE_CONTENT)
            created["files"].append(str(gitignore))

    # Create traces directory
    traces_dir = data_dir / "traces"
    if not traces_dir.exists():
        traces_dir.mkdir()
        created["directories"].append(str(traces_dir))

    return created


def cmd_init(args) -> None:
    """Handle 'init' command."""
    print("\nInitializing Aragora project...")

    result = init_project(
        directory=getattr(args, "directory", None),
        force=getattr(args, "force", False),
        with_git=not getattr(args, "no_git", False),
    )

    if result["directories"]:
        print("\nCreated directories:")
        for d in result["directories"]:
            print(f"  - {d}")

    if result["files"]:
        print("\nCreated files:")
        for f in result["files"]:
            print(f"  - {f}")

    print("\nAragora project initialized!")
    print("\nNext steps:")
    print("  1. Set API keys in environment or .aragora.yaml")
    print("  2. Run: aragora ask 'Your question here'")
    print("  3. Or start server: aragora serve")
